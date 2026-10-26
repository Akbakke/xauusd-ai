"""Synthetic C mechanics only: no saved states, physical inputs, fit or CUDA."""
from copy import deepcopy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import numpy as np
import torch

from gx1.contracts import entry_native_direction_checkpoint_diagnostic_v1 as contract
from gx1.scripts import audit_entry_native_direction_checkpoints_v1 as worker


def _ref(name):
    return {'path': contract.PROGRAM + '/' + name, 'sha256': '0' * 64}


def _source_inputs():
    previous = {name: {'path': contract.REPO + '/' + name, 'sha256': digest}
        for name, digest in contract.TRANSPORT_ORIGINAL_SHA256.items()}
    shared = ('gx1/contracts/entry_native_direction_context_report_v1.py',
        'gx1/contracts/entry_native_direction_feature_view_v1.py',
        'gx1/contracts/entry_native_direction_research_state_v1.py',
        'gx1/contracts/gx1_capped_execution_v1.py',
        'gx1/models/entry_v10/entry_v10_ctx_train_v3.py',
        'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py')
    previous.update({name: {'path': contract.REPO + '/' + name, 'sha256': '0' * 64} for name in shared})
    previous.update({f'gx1/synthetic_owner_{i}.py': {'path': contract.REPO + f'/gx1/synthetic_owner_{i}.py',
        'sha256': '0' * 64} for i in range(40)})
    reused = {name: deepcopy(previous[name]) for name in shared}
    extra = 'gx1/contracts/entry_direction_probability_reference_v1.py'
    reused[extra] = {'path': contract.REPO + '/' + extra, 'sha256': '0' * 64}
    return {'source_owners': previous}, {'existing_sources_reused_without_change': reused}


def _plan():
    previous, proposal = _source_inputs()
    owners = {**previous['source_owners'], **proposal['existing_sources_reused_without_change'],
        **{name: {'path': contract.REPO + '/' + name, 'sha256': '2' * 64} for name in contract.NEW_FILES}}
    transitions = {}
    for name, digest in contract.TRANSPORT_ORIGINAL_SHA256.items():
        current = {'path': contract.REPO + '/' + name, 'sha256': '1' * 64}
        transitions[name] = {'old276': deepcopy(owners[name]), 'current': current}
        owners[name] = deepcopy(current)
    root = contract.PROGRAM + '/C_CHECKPOINT_DIAGNOSTIC_SYNTHETIC'
    return {'schema_version': contract.SCHEMA, 'operation': contract.OPERATION,
        'admission_stage': contract.STAGE, 'proposal': deepcopy(contract.PROPOSAL),
        'source_owners': owners, 'transport_source_transitions': transitions,
        'run_root': root, 'runtime_root': root + '_RUNTIME',
        'execution_binding_path': root + '_RUNTIME/EXECUTION_BINDING.json',
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


# Guarded-route fixtures below are synthetic metadata only. They never prepare
# physical inputs or construct/restore/forward the genuine native model.
def _route_window(recipe, recipe_ref):
    return contract._seal({'schema_version': contract.WINDOW_SCHEMA, 'recipe': recipe_ref,
        'invocation_number': 1, 'max_invocation_seconds': 12000,
        'progress_path': recipe['run_root'] + '/PROGRESS.json',
        'campaign_cursor_path': recipe['run_root'] + '/DIAGNOSTIC_POINTER.json',
        'output_directory': recipe['run_root'], 'test_data_used': False}, 'policy_sha256')


def _route_invocation(recipe, window):
    c = contract.campaign
    argv = [contract.REPO + '/scripts/gx1_capped_run.sh', '--class', 'trainer', '--mem', '20G', '--swap', '512M',
        '--', contract.REPO + '/.venv/bin/python', '-m', contract.MODULE,
        '--window-policy', _ref('SYNTHETIC_WINDOW.json')['path'],
        '--window-policy-file-sha256', '0' * 64, '--progress-path', window['progress_path']]
    value = {'schema_version': c.INVOCATION_SCHEMA, 'decision': 'PASS', 'invocation_number': 1,
        'invocation_id': 'invocation-0001', 'kind': contract.KIND, 'batch_size': 16,
        'epoch_index': None, 'window_index': 0, 'window_count': 1, 'optimizer_step_budget': 0,
        'expected_success_outcome': 'COMPLETE', 'source_commit': 'a' * 40,
        'execution_manifest': _ref('SYNTHETIC_EXECUTION.json'), 'launcher_argv': argv,
        'launcher_argv_sha256': c.canonical_sha256(argv), 'progress_path': window['progress_path'],
        'guard_log_path': recipe['runtime_root'] + '/logs/invocation-0001.signed-guard.log',
        'checkpoint': {'pointer_path': window['campaign_cursor_path'], 'before_mode': 'GENESIS',
            'predecessor_invocation_number': None, 'write_mode': 'ATOMIC_UPDATE'},
        'maximum_wall_seconds': 13800, 'requires_fresh_windows_boot': True,
        'signed_guard_only': True, 'test_data_used': False}
    return contract._seal(value, 'invocation_sha256')


def _route_reseal(value, seal):
    value[seal] = contract.campaign.canonical_sha256({k: v for k, v in value.items() if k != seal})
    return value


@pytest.fixture
def route_invocation_case(monkeypatch):
    recipe, recipe_ref = _plan(), _ref('SYNTHETIC_RECIPE.json')
    window = _route_window(recipe, recipe_ref)
    invocation = _route_invocation(recipe, window)
    execution = {'prelaunch_manifest': recipe_ref, 'train_session_manifest': _ref('SYNTHETIC_WINDOW.json'),
        'prelaunch_manifest_sha256': contract.campaign.canonical_sha256(recipe),
        'train_session_manifest_sha256': window['policy_sha256']}
    monkeypatch.setattr(contract.campaign, 'require_binding', lambda ref, **_kw: contract.plain_ref(ref))
    monkeypatch.setattr(contract.campaign, 'read_bound_json', lambda *_args: deepcopy(window))
    monkeypatch.setattr(contract, 'require_checkpoint_diagnostic_recipe', lambda *_args, **_kw: deepcopy(recipe))
    monkeypatch.setattr(contract, '_path', lambda value, _label: Path(value))
    return recipe, window, invocation, execution


def test_route_static_recipe_has_no_head_or_policy_hash_cycle():
    value = contract.require_plan(_plan())
    assert 'source_head' not in value and 'policy' not in value
    assert value['runtime_root'] != value['run_root']
    assert value['execution_binding_path'] == value['runtime_root'] + '/EXECUTION_BINDING.json'
    for key in ('source_head', 'policy'):
        changed = deepcopy(value); changed[key] = 'not-admitted'
        with pytest.raises(RuntimeError, match='PLAN_SCHEMA'):
            contract.require_plan(changed)


@pytest.mark.parametrize('fault', ['changed_other', 'changed_reused', 'missing_owner', 'extra_owner',
    'missing_transition', 'extra_transition', 'wrong_old276', 'unbound_current', 'conflicting_alias'])
def test_route_exact_four_transport_changes_only(fault):
    recipe = _plan(); original, proposal = _source_inputs()
    contract.require_source_closure(contract.require_plan(recipe), original, proposal)
    transition = next(iter(contract.TRANSPORT_ORIGINAL_SHA256))
    if fault == 'changed_other': recipe['source_owners']['gx1/synthetic_owner_0.py']['sha256'] = '9' * 64
    elif fault == 'changed_reused': recipe['source_owners']['gx1/contracts/entry_direction_probability_reference_v1.py']['sha256'] = '9' * 64
    elif fault == 'missing_owner': del recipe['source_owners']['gx1/synthetic_owner_0.py']
    elif fault == 'extra_owner': recipe['source_owners']['gx1/extra.py'] = {'path': contract.REPO + '/gx1/extra.py', 'sha256': '9' * 64}
    elif fault == 'missing_transition': del recipe['transport_source_transitions'][transition]
    elif fault == 'extra_transition': recipe['transport_source_transitions']['extra'] = {}
    elif fault == 'wrong_old276': recipe['transport_source_transitions'][transition]['old276']['sha256'] = '9' * 64
    elif fault == 'unbound_current': recipe['transport_source_transitions'][transition]['current']['sha256'] = '9' * 64
    elif fault == 'conflicting_alias':
        proposal['existing_sources_reused_without_change']['gx1/contracts/entry_native_direction_context_report_v1.py']['sha256'] = '9' * 64
    with pytest.raises(RuntimeError):
        contract.require_source_closure(contract.require_plan(recipe), original, proposal)


def test_route_canonical_invocation_metadata_is_accepted(route_invocation_case):
    _, window, invocation, execution = route_invocation_case
    c = contract.campaign
    c._require_launcher(invocation['launcher_argv'], source_repo=Path(contract.REPO), kind=contract.KIND,
        claimed_sha256=invocation['launcher_argv_sha256'])
    contract.require_checkpoint_diagnostic_invocation_execution(invocation, execution,
        source_repo=Path(contract.REPO), source_commit='a' * 40)
    assert invocation['checkpoint_diagnostic_window'] == window
    assert invocation['checkpoint_diagnostic_recipe'] == window['recipe']


@pytest.mark.parametrize('fault', ['direct_module', 'direct_py', 'audit', 'producer', 'env_disguise',
    'step', 'boolean_step', 'batch', 'boolean_invocation', 'resume', 'wall', 'window_count', 'wrong_progress'])
def test_route_invocation_bypasses_and_mutations_reject(route_invocation_case, fault):
    _, _, invocation, execution = route_invocation_case
    c = contract.campaign; argv = invocation['launcher_argv']; separator = argv.index('--')
    if fault == 'direct_module': argv[separator + 3] = 'gx1.scripts.audit_entry_native_direction_checkpoints_v1'
    elif fault == 'direct_py': argv[separator + 2:] = [contract.REPO + '/gx1/scripts/audit_entry_native_direction_checkpoints_v1.py']
    elif fault in ('audit', 'producer'): argv[2] = fault
    elif fault == 'env_disguise': argv.insert(0, '/usr/bin/env')
    elif fault == 'step': invocation['optimizer_step_budget'] = 1
    elif fault == 'boolean_step': invocation['optimizer_step_budget'] = False
    elif fault == 'batch': invocation['batch_size'] = 2
    elif fault == 'boolean_invocation': invocation['invocation_number'] = True
    elif fault == 'resume': invocation['checkpoint']['before_mode'] = 'PREVIOUS_RECEIPT_AFTER'
    elif fault == 'wall': invocation['maximum_wall_seconds'] = 13801
    elif fault == 'window_count': invocation['window_count'] = 2
    elif fault == 'wrong_progress': invocation['progress_path'] += '.other'
    invocation['launcher_argv_sha256'] = c.canonical_sha256(argv)
    with pytest.raises(RuntimeError):
        c._require_launcher(argv, source_repo=Path(contract.REPO), kind=contract.KIND,
            claimed_sha256=invocation['launcher_argv_sha256'])
        contract.require_checkpoint_diagnostic_invocation_execution(invocation, execution,
            source_repo=Path(contract.REPO), source_commit='a' * 40)


@pytest.mark.parametrize('kind,budget,accepted', [(contract.KIND, 0, True),
    ('native_direction_research_window', 0, False), ('native_direction_prefix_fit_window', 0, False),
    ('native_candidate_window', 0, False), ('epoch1_window', 0, False),
    ('native_direction_research_window', 4, True), ('native_direction_prefix_fit_window', 2976, True),
    ('native_candidate_window', None, True)])
def test_route_zero_step_exception_is_only_exact_C_kind(kind, budget, accepted):
    recipe = _plan(); window = _route_window(recipe, _ref('SYNTHETIC_RECIPE.json'))
    invocation = _route_invocation(recipe, window)
    invocation.update(kind=kind, optimizer_step_budget=budget, batch_size=2 if kind == 'native_direction_research_window' else 16,
        epoch_index=0 if kind == 'epoch1_window' else None,
        expected_success_outcome='RESUMABLE_OR_COMPLETE' if kind == 'native_candidate_window' else 'COMPLETE')
    _route_reseal(invocation, 'invocation_sha256')
    def validate():
        return contract.campaign.require_invocation(invocation, source_repo=Path(contract.REPO),
            source_commit='a' * 40, verify_files=False)
    if accepted:
        assert validate()['optimizer_step_budget'] == budget
    else:
        with pytest.raises(RuntimeError): validate()


def _route_boot(number):
    return contract._seal({'schema_version': contract.campaign.BOOT_SCHEMA, 'computer_name': 'SYNTHETIC',
        'last_boot_utc': f'2026-10-26T0{number}:00:00+00:00', 'boot_id': number}, 'identity_sha256')


@pytest.fixture
def route_context_case(monkeypatch, tmp_path):
    import subprocess
    recipe = _plan(); unique = tmp_path.name
    recipe['run_root'] += '_' + unique; recipe['runtime_root'] += '_' + unique
    recipe['execution_binding_path'] = recipe['runtime_root'] + '/EXECUTION_BINDING.json'
    recipe_ref = _ref('SYNTHETIC_RECIPE_' + unique + '.json')
    window_ref = _ref('SYNTHETIC_WINDOW_' + unique + '.json')
    window = _route_window(recipe, recipe_ref); invocation = _route_invocation(recipe, window)
    invocation.update(checkpoint_diagnostic_recipe=recipe_ref, checkpoint_diagnostic_window=window)
    plan_ref = _ref('SYNTHETIC_CAMPAIGN_' + unique + '.json')
    plan = {'phase': contract.PHASE, 'plan_sha256': '3' * 64, 'source_commit': 'a' * 40,
        'runtime_root': recipe['runtime_root'], 'recipe': recipe_ref, 'checked_invocations': [invocation],
        'prepared_windows_boot': _route_boot(1)}
    marker = contract._seal({'schema_version': contract.campaign.ACTIVE_SCHEMA, 'plan_sha256': plan['plan_sha256'],
        'invocation_sha256': invocation['invocation_sha256'], 'invocation_number': 1, 'invocation_id': 'invocation-0001',
        'kind': contract.KIND, 'launcher_argv_sha256': invocation['launcher_argv_sha256'],
        'pointer_before_sha256': 'GENESIS', 'boot': _route_boot(2), 'started_utc': '2026-10-26T02:01:00+00:00'}, 'marker_sha256')
    admission_ref = _ref('SYNTHETIC_ADMISSION_' + unique + '.json')
    gate = {'recipe': recipe_ref, 'admission': admission_ref, 'run_root': recipe['run_root'],
        'runtime_root': recipe['runtime_root'], 'max_invocations': 1, 'claimed_invocations': 0,
        'completed_invocations': 0, 'relaunch_allowed': False}
    for key in ('execution_authorized', 'data_reads_authorized', 'checkpoint_reads_authorized',
            'genuine_constructor_authorized', 'genuine_forwards_authorized', 'gradient_reverse_authorized'):
        gate[key] = True
    for key in ('optimizer_steps_authorized', 'training_enabled', 'native_training_enabled', 'full_epoch_training_allowed',
            'full_val_allowed', 'genuine_predictor_fit_authorized', 'normalization_fit_authorized',
            'synthetic_test_execution_authorized', 'native_launch_authorized', 'test_data_used', 'exit_evaluation', 'broker_access_allowed'):
        gate[key] = False
    policy = {'schema_version': 'gx1_next_native_run_policy_v1', 'canonical_source_repo': contract.REPO,
        'canonical_branch': 'work/gx1-current', 'training_module': contract.MODULE,
        'training_enabled': False, 'native_training_enabled': False, 'full_epoch_training_allowed': False,
        'full_val_allowed': False, contract.STAGE: gate, 'gpu_clock_launcher': _ref('SYNTHETIC_CLOCK.ps1')}
    policy_ref = {'path': contract.REPO + '/NEXT_RUN_POLICY.json', 'sha256': '0' * 64}
    binding = {'plan': plan_ref, 'source_head': 'a' * 40, 'policy': policy_ref}
    admission = {'operation': contract.OPERATION, 'recipe': recipe_ref, 'execution_authorized': True,
        'max_invocations': 1, 'genuine_payload_reads_authorized': True, 'genuine_forwards_authorized': True,
        'gradient_reverse_authorized': True, 'optimizer_steps_authorized': False, 'test_data_used': False}
    registry = {window_ref['path']: window, plan_ref['path']: plan, recipe['runtime_root'] + '/ACTIVE_INVOCATION.json': marker,
        recipe['execution_binding_path']: binding, policy_ref['path']: policy, admission_ref['path']: admission}
    def read(path, digest):
        assert digest == '0' * 64
        return deepcopy(registry[str(path)])
    def git(argv, **_kw):
        if argv[-2:] == ['rev-parse', 'HEAD']: return 'a' * 40 + '\n'
        if argv[-2:] == ['branch', '--show-current']: return 'work/gx1-current\n'
        assert 'status' in argv
        return ''
    monkeypatch.setattr(contract.campaign, 'read_bound_json', read)
    monkeypatch.setattr(contract.campaign, 'file_sha256', lambda _path: '0' * 64)
    monkeypatch.setattr(contract.campaign, 'require_binding', lambda ref, **_kw: contract.plain_ref(ref))
    monkeypatch.setattr(contract.campaign, 'require_plan', lambda _value: deepcopy(plan))
    monkeypatch.setattr(contract.campaign, 'require_clean_source', lambda _value: None)
    monkeypatch.setattr(contract, 'require_checkpoint_diagnostic_recipe', lambda *_args, **_kw: deepcopy(recipe))
    monkeypatch.setattr(contract, '_path', lambda value, _label: Path(value))
    monkeypatch.setattr(subprocess, 'check_output', git)
    for name, value in {'GX1_CAMPAIGN_PLAN_PATH': plan_ref['path'], 'GX1_CAMPAIGN_PLAN_FILE_SHA256': plan_ref['sha256'],
        'GX1_CAMPAIGN_PLAN_SHA256': plan['plan_sha256'], 'GX1_CAMPAIGN_INVOCATION_SHA256': invocation['invocation_sha256'],
        'GX1_CAMPAIGN_GUARD_LOG_PATH': invocation['guard_log_path']}.items(): monkeypatch.setenv(name, value)
    return {'recipe': recipe, 'window': window, 'window_ref': window_ref, 'plan': plan, 'marker': marker,
        'binding': binding, 'policy': policy, 'gate': gate, 'admission': admission, 'registry': registry}


def test_route_context_accepts_only_fresh_actual_bound_metadata(route_context_case):
    case = route_context_case
    result = contract.require_native_direction_checkpoint_diagnostic_context(Path(case['window_ref']['path']),
        case['window_ref']['sha256'], Path(case['window']['progress_path']))
    assert result['recipe'] == case['recipe'] and result['binding'] == case['binding']
    assert not Path(case['recipe']['run_root']).exists()
    assert len(result['metadata_refs']) == 6


@pytest.mark.parametrize('fault', ['phase', 'plan_env', 'marker_kind', 'marker_boolean_number', 'old_boot',
    'binding_head', 'binding_policy', 'guard_log_env', 'claimed', 'completed', 'steps', 'training',
    'wrong_recipe', 'admission', 'progress', 'output_exists'])
def test_route_context_mutations_reject_before_any_payload(route_context_case, monkeypatch, fault):
    c = route_context_case; progress = c['window']['progress_path']
    if fault == 'phase': c['plan']['phase'] = 'native_direction_prefix_fit'
    elif fault == 'plan_env': monkeypatch.setenv('GX1_CAMPAIGN_PLAN_SHA256', '9' * 64)
    elif fault == 'marker_kind': c['marker']['kind'] = 'native_direction_prefix_fit_window'; _route_reseal(c['marker'], 'marker_sha256')
    elif fault == 'marker_boolean_number': c['marker']['invocation_number'] = True; _route_reseal(c['marker'], 'marker_sha256')
    elif fault == 'old_boot': c['marker']['boot'] = _route_boot(1); _route_reseal(c['marker'], 'marker_sha256')
    elif fault == 'binding_head': c['binding']['source_head'] = 'b' * 40
    elif fault == 'binding_policy': c['binding']['policy'] = {'path': contract.REPO + '/NEXT_RUN_POLICY.json', 'sha256': '9' * 64}
    elif fault == 'guard_log_env': monkeypatch.setenv('GX1_CAMPAIGN_GUARD_LOG_PATH', '/tmp/wrong-guard.log')
    elif fault == 'claimed': c['gate']['claimed_invocations'] = 1
    elif fault == 'completed': c['gate']['completed_invocations'] = 1
    elif fault == 'steps': c['gate']['optimizer_steps_authorized'] = True
    elif fault == 'training': c['policy']['training_enabled'] = True
    elif fault == 'wrong_recipe': c['gate']['recipe'] = _ref('SYNTHETIC_WRONG_RECIPE.json')
    elif fault == 'admission': c['admission']['optimizer_steps_authorized'] = True
    elif fault == 'progress': progress += '.other'
    elif fault == 'output_exists':
        original = Path.exists
        monkeypatch.setattr(Path, 'exists', lambda self: True if str(self) == c['recipe']['run_root'] else original(self))
    with pytest.raises(RuntimeError):
        contract.require_native_direction_checkpoint_diagnostic_context(Path(c['window_ref']['path']),
            c['window_ref']['sha256'], Path(progress))


@pytest.mark.parametrize('boundary', ['guard', 'context'])
def test_route_public_entry_rejects_before_payload_imports(monkeypatch, boundary):
    import builtins
    from gx1.contracts import gx1_capped_execution_v1 as guard
    events = []
    def guarded():
        events.append('guard')
        if boundary == 'guard': raise RuntimeError('SYNTHETIC_GUARD_REJECT')
        return {}
    def context(*_args):
        events.append('context')
        raise RuntimeError('SYNTHETIC_CONTEXT_REJECT')
    original_import = builtins.__import__
    def imports(name, globals=None, locals=None, fromlist=(), level=0):
        assert name not in ('numpy', 'torch')
        assert not (name == 'gx1.scripts' and any(n.startswith('run_entry_native_direction_') for n in fromlist))
        return original_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(guard, 'require_guarded_cuda_trainer_execution', guarded)
    monkeypatch.setattr(contract, 'require_native_direction_checkpoint_diagnostic_context', context)
    monkeypatch.setattr(builtins, '__import__', imports)
    with pytest.raises(RuntimeError, match='SYNTHETIC_' + boundary.upper() + '_REJECT'):
        worker.run_native_direction_checkpoint_diagnostic_window(policy_path=Path('/tmp/SYNTHETIC_WINDOW.json'),
            policy_file_sha256='0' * 64, progress_path=Path('/tmp/SYNTHETIC_PROGRESS.json'))
    assert events == (['guard'] if boundary == 'guard' else ['guard', 'context'])


def test_route_direct_C_main_has_no_cli_or_env_alias():
    with pytest.raises(RuntimeError, match='DIRECT_C_MODULE_FORBIDDEN'):
        worker.main()
    assert not hasattr(worker, 'run_checkpoint_diagnostic')


def _terminal_case(tmp_path, complete=True):
    recipe_ref = {'path': str(tmp_path / 'SYNTHETIC_RECIPE.json'), 'sha256': '0' * 64}
    invocation = {'kind': contract.KIND, 'invocation_sha256': '2' * 64,
        'checkpoint_diagnostic_recipe': recipe_ref, 'checkpoint': {'pointer_path': str(tmp_path / 'DIAGNOSTIC_POINTER.json')}}
    route = {'recipe_ref': recipe_ref, 'recipe': {'budget': deepcopy(contract.RESOURCE_LIMITS)},
        'plan': {'plan_sha256': '1' * 64}, 'invocation': invocation,
        'window': {'recipe': recipe_ref, 'campaign_cursor_path': invocation['checkpoint']['pointer_path'],
            'progress_path': str(tmp_path / 'PROGRESS.json')}}
    budget = contract.CallBudget()
    if complete: budget.calls = dict(contract.CALL_LIMITS); budget.returns = dict(contract.CALL_LIMITS)
    result = {'recipe': recipe_ref, 'admission': {'path': str(tmp_path / 'ADMISSION.json'), 'sha256': '0' * 64},
        'campaign_plan': {'path': str(tmp_path / 'CAMPAIGN.json'), 'sha256': '0' * 64}, 'operation': contract.OPERATION,
        'decision': 'COMPLETE_READONLY_TRAIN_DIAGNOSIS' if complete else 'FAILED', 'error': None,
        'observed_calls': dict(budget.calls), 'observed_returns': dict(budget.returns),
        'optimizer_steps': 0, 'test_data_used': False}
    return route, budget, result


@pytest.mark.parametrize('complete', [True, False])
def test_route_terminal_pointer_progress_use_actual_counts_and_zero_steps(tmp_path, monkeypatch, complete):
    route, budget, result = _terminal_case(tmp_path, complete)
    result_ref = {'path': str(tmp_path / 'RESULT.json'), 'sha256': '0' * 64}
    pointer = contract.build_checkpoint_diagnostic_pointer(recipe_ref=route['recipe_ref'], result_ref=result_ref,
        outcome='COMPLETE' if complete else 'FAILED', calls=budget.calls, returns=budget.returns)
    pointer_ref = {'path': route['window']['campaign_cursor_path'], 'sha256': '0' * 64}
    progress = contract.build_checkpoint_diagnostic_progress(window=route['window'], plan=route['plan'],
        invocation=route['invocation'], checkpoint_pointer=pointer_ref, pointer=pointer)
    registry = {pointer_ref['path']: pointer, result_ref['path']: result}
    monkeypatch.setattr(contract.campaign, 'require_binding', lambda ref, **_kw: contract.plain_ref(ref))
    monkeypatch.setattr(contract.campaign, 'read_bound_json', lambda path, _sha: deepcopy(registry[str(path)]))
    checked = contract.campaign.require_progress(progress, plan_sha256=route['plan']['plan_sha256'],
        invocation=route['invocation'], expected_selection_receipt_sha256=None, verify_file=True)
    assert checked['completed_units'] == (194 if complete else 0)
    assert checked['reverse_returns'] == (6 if complete else 0)
    assert checked['checkpoint_load_returns'] == (4 if complete else 0)
    assert checked['optimizer_steps'] == 0 and checked['terminal'] is True


@pytest.mark.parametrize('field,value', [('optimizer_steps', 1), ('completed_units', 193), ('forward_calls', 195),
    ('reverse_returns', 7), ('checkpoint_loads', 5), ('forward_returns', True), ('outcome', 'RESUMABLE')])
def test_route_progress_mutations_reject_without_loading_payload(tmp_path, field, value):
    route, budget, _ = _terminal_case(tmp_path)
    pointer = contract.build_checkpoint_diagnostic_pointer(recipe_ref=route['recipe_ref'],
        result_ref={'path': str(tmp_path / 'RESULT.json'), 'sha256': '0' * 64}, outcome='COMPLETE', calls=budget.calls, returns=budget.returns)
    progress = contract.build_checkpoint_diagnostic_progress(window=route['window'], plan=route['plan'],
        invocation=route['invocation'], checkpoint_pointer={'path': route['window']['campaign_cursor_path'], 'sha256': '0' * 64}, pointer=pointer)
    progress[field] = value; _route_reseal(progress, 'progress_sha256')
    with pytest.raises(RuntimeError):
        contract.require_checkpoint_diagnostic_progress(progress, plan_sha256=route['plan']['plan_sha256'],
            invocation=route['invocation'], expected_selection_receipt_sha256=None, verify_file=False)


@pytest.mark.parametrize('oversize', [False, True])
def test_route_three_terminal_writes_share_cap_and_cannot_overwrite(tmp_path, oversize):
    route, budget, result = _terminal_case(tmp_path)
    if oversize: result['optional'] = 'x' * (2 * contract.TERMINAL_JSON_RESERVE_BYTES)
    limit = route['recipe']['budget']['max_json_output_bytes']
    counts = {'json_bytes': limit - contract.TERMINAL_JSON_RESERVE_BYTES}
    def publish(path, value, counters, maximum):
        encoded = worker._json_bytes(value)
        assert counters['json_bytes'] + len(encoded) <= maximum == limit
        with Path(path).open('xb') as handle: handle.write(encoded)
        counters['json_bytes'] += len(encoded)
        return {'path': str(path), 'sha256': hashlib.sha256(encoded).hexdigest()}
    common = SimpleNamespace(_json=publish, _hash_file=lambda path: {'path': str(path),
        'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    actual = worker._publish_terminal(result, route=route, output=tmp_path, budget=budget, counts=counts, common=common)
    assert actual['decision'] == ('FAILED' if oversize else 'COMPLETE_READONLY_TRAIN_DIAGNOSIS')
    progress = json.loads((tmp_path / 'PROGRESS.json').read_text())
    assert progress['outcome'] == ('FAILED' if oversize else 'COMPLETE') and progress['completed_units'] == 194
    assert counts['json_bytes'] <= limit
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        worker._publish_terminal(actual, route=route, output=tmp_path, budget=budget,
            counts={'json_bytes': 0}, common=common)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_route_running_progress_matches_existing_observer_and_is_not_terminal_receipt(tmp_path):
    route, budget, _ = _terminal_case(tmp_path, complete=False)
    before = contract.build_checkpoint_diagnostic_running_progress(window=route['window'], plan=route['plan'],
        invocation=route['invocation'], calls=budget.calls, returns=budget.returns)
    budget.calls['panel_forward'] = budget.returns['panel_forward'] = 16
    after = contract.build_checkpoint_diagnostic_running_progress(window=route['window'], plan=route['plan'],
        invocation=route['invocation'], calls=budget.calls, returns=budget.returns)
    # The unchanged observer checks these identities/ranges while the process
    # lives, then requires terminal=true only after the trainer has exited.
    for value, completed in ((before, 0), (after, 16)):
        assert value['schema_version'] == 'gx1_local_random_access_progress_v2'
        assert value['plan_sha256'] == route['plan']['plan_sha256']
        assert value['invocation_sha256'] == route['invocation']['invocation_sha256']
        assert value['completed_units'] == completed <= value['total_units'] == 194
        assert value['outcome'] == 'RUNNING' and value['terminal'] is False
        assert value['checkpoint_pointer'] is None and value['optimizer_steps'] == 0
        with pytest.raises(RuntimeError, match='PROGRESS_IDENTITY'):
            contract.require_checkpoint_diagnostic_progress(value, plan_sha256=route['plan']['plan_sha256'],
                invocation=route['invocation'], expected_selection_receipt_sha256=None, verify_file=False)


def test_route_readonly_output_directory_fsync_is_allowed_by_both_real_open_filters(monkeypatch):
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    hooks = []
    monkeypatch.setattr(sys, 'addaudithook', hooks.append)
    output = Path(contract.PROGRAM + '/C_CHECKPOINT_DIAGNOSTIC_SYNTHETIC_AUDIT_EVENTS_ONLY')
    initial_ref = {'path': str(output.parent / 'SYNTHETIC_INITIAL.npz'), 'sha256': '0' * 64}
    monkeypatch.setattr(contract, 'INITIAL_PREDICTIONS', initial_ref)
    context = {'ancestor_plan': {'payloads': {}, 'native_cache_dir': '/synthetic/cache',
        'gc_surfaces': {'path': '/synthetic/gc/layout.json'}},
        'info': {'metadata': {}, 'audit': {'manifest_path': '/synthetic/forbidden.json'}}}
    delegated = worker._read_guard_context(context, output, [], {initial_ref['path']: {}})
    assert context['info']['metadata'] == {}
    assert delegated['info']['metadata'][str(output)] == {'read_guard_directory_only': True}
    worker._install_guard({'name': 'METADATA'}, output)
    common._install_read_guard(delegated, output, output)
    assert len(hooks) == 2
    # Invoke actual callbacks with synthetic events; no hook is installed and
    # no file, payload or directory is opened by this fixture.
    for hook in hooks:
        hook('open', (str(output), None, os.O_RDONLY | os.O_DIRECTORY))
        hook('open', (str(output / 'SYNTHETIC.json'), None, os.O_WRONLY))
    with pytest.raises(RuntimeError): hooks[0]('open', (str(output), None, os.O_RDONLY))
    with pytest.raises(RuntimeError): hooks[0]('open', (str(output), None, os.O_WRONLY))
    with pytest.raises(RuntimeError): hooks[1]('open', (str(output.parent / 'UNBOUND.npy'), None, os.O_RDONLY))


def test_route_missing_original_in_finally_still_publishes_consumed_failed_progress(tmp_path, monkeypatch):
    from gx1.contracts import gx1_capped_execution_v1 as guard
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix
    output = tmp_path / 'output'
    route, _, _ = _terminal_case(output, complete=False)
    route['recipe'].update(run_root=str(output), transport_source_transitions={})
    route['admission_ref'] = {'path': str(tmp_path / 'SYNTHETIC_ADMISSION.json'), 'sha256': '0' * 64}
    route['binding'] = {'plan': {'path': str(tmp_path / 'SYNTHETIC_CAMPAIGN.json'), 'sha256': '0' * 64}}
    route['marker'] = {'marker_sha256': '0' * 64}
    refs = [{'path': str(tmp_path / f'SYNTHETIC_STATE_{i}.pt'), 'sha256': '0' * 64} for i in range(4)]
    initial = {'path': str(tmp_path / 'SYNTHETIC_INITIAL.npz'), 'sha256': '0' * 64}
    monkeypatch.setattr(contract, 'INITIAL_PREDICTIONS', initial)
    original_names = {r['path'] for r in refs + [initial]}; seen = []
    original_stat = common._stat
    def stat(path):
        if str(path) not in original_names: return original_stat(path)
        seen.append(str(path))
        if len(seen) > 5: raise FileNotFoundError('SYNTHETIC_ORIGINAL_DISAPPEARED')
        return {'synthetic': 1}
    def metadata(*_args): raise RuntimeError('SYNTHETIC_STOP_BEFORE_NUMERICAL_IMPORTS')
    monkeypatch.setattr(guard, 'require_guarded_cuda_trainer_execution', lambda: {})
    monkeypatch.setattr(contract, 'require_native_direction_checkpoint_diagnostic_context', lambda *_args: route)
    monkeypatch.setattr(worker, '_install_guard', lambda *_args: [])
    monkeypatch.setattr(worker, '_checkpoint_metadata', lambda *_args: ({}, {}, {}, refs))
    monkeypatch.setattr(prefix, '_metadata', metadata)
    monkeypatch.setattr(common, '_stat', stat)
    monkeypatch.setattr(common, '_io', lambda: {'rchar': 0})
    monkeypatch.setattr(worker.signal, 'signal', lambda *_args: None)
    monkeypatch.setattr(worker.signal, 'alarm', lambda *_args: None)
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    actual = worker.run_native_direction_checkpoint_diagnostic_window(policy_path=tmp_path / 'SYNTHETIC_WINDOW.json',
        policy_file_sha256='0' * 64, progress_path=output / 'PROGRESS.json')
    assert actual['decision'] == 'FAILED' and actual['originals_stat_preserved'] is False
    assert actual['error']['type'] == 'FileNotFoundError' and len(seen) == 6
    assert (output / 'CLAIM.json').is_file()
    progress = json.loads((output / 'PROGRESS.json').read_text())
    assert progress['terminal'] is True and progress['outcome'] == 'FAILED'
    assert progress['completed_units'] == 0 and progress['optimizer_steps'] == 0
    with pytest.raises(FileExistsError):
        worker.run_native_direction_checkpoint_diagnostic_window(policy_path=tmp_path / 'SYNTHETIC_WINDOW.json',
            policy_file_sha256='0' * 64, progress_path=output / 'PROGRESS.json')
    assert len(seen) == 6


@pytest.mark.parametrize('C_schema', [True, False])
def test_route_actual_capped_log_parent_fragment_keeps_C_output_absent(tmp_path, C_schema):
    # Execute just the exact source-owned log-parent selection fragment. This
    # fixture never starts capped_run, a guard, a controller or a model job.
    source = (Path(contract.REPO) / 'scripts/gx1_capped_run.sh').read_text()
    begin = source.index('    TRAINER_OUT_BUNDLE_DIR="${target_args[8]}"',
        source.index('if [[ "$module" == "$NATIVE_CANDIDATE_WINDOW_MODULE" ]]'))
    end = source.index('\n    return', begin)
    fragment = source[begin:end]
    output = tmp_path / 'fresh_output'; logs = tmp_path / 'runtime' / 'logs'; logs.mkdir(parents=True)
    progress = output / 'PROGRESS.json'; guard_log = logs / 'invocation-0001.signed-guard.log'
    if not C_schema: output.mkdir()
    window = tmp_path / 'SYNTHETIC_WINDOW.json'
    window.write_text(json.dumps({'schema_version': contract.WINDOW_SCHEMA if C_schema else
        'gx1_native_direction_prefix_fit_window_policy_v1'}))
    script = 'set -euo pipefail\nCANONICAL_TRAINER_PYTHON="$1"\n' + \
        'target_args=(unused unused unused unused "$2" unused unused unused "$3")\n' + \
        'GX1_CAMPAIGN_GUARD_LOG_PATH="$4"\n' + fragment + \
        '\n[[ "$TRAINER_OUT_BUNDLE_DIR" == "$5" ]]\n[[ -d "${TRAINER_OUT_BUNDLE_DIR%/*}" ]]\n'
    expected = guard_log if C_schema else progress
    result = subprocess.run(['bash', '--noprofile', '--norc', '-c', script, '--', sys.executable,
        str(window), str(progress), str(guard_log), str(expected)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert output.exists() is (not C_schema)
    assert not progress.exists() and not guard_log.exists()


# C001 repair: exercise actual CPython cache fallback without installing a
# persistent audit hook, starting a child, executing code or using real payloads.
def _captured_metadata_cache_guard(monkeypatch):
    hooks = []
    monkeypatch.setattr(sys, 'addaudithook', hooks.append)
    probes = {'total_count': 0, 'sample_paths': [],
        'protected_cache_bytes_read': 0, 'protected_cache_reads_authorized': False}
    denied = worker._install_guard({'name': 'METADATA'},
        Path(contract.PROGRAM + '/C_CHECKPOINT_DIAGNOSTIC_SYNTHETIC_CACHE_OUTPUT'), probes)
    assert len(hooks) == 1
    return hooks[0], denied, probes


def test_route_metadata_cache_probe_uses_real_source_loader_fallback(tmp_path, monkeypatch):
    from importlib import _bootstrap_external, machinery, util
    from types import CodeType
    source = tmp_path / 'synthetic_cache_fallback.py'
    source_bytes = b'marker = "SOURCE_ONLY_SENTINEL"\n'
    source.write_bytes(source_bytes)
    cache = Path(util.cache_from_source(str(source)))
    cache.parent.mkdir()
    cached_code = compile('marker = "CACHE_ONLY_SENTINEL"\n', str(source), 'exec')
    source_stat = source.stat()
    # This cache is valid for the current source timestamp and source size.
    cache.write_bytes(_bootstrap_external._code_to_timestamp_pyc(
        cached_code, int(source_stat.st_mtime), len(source_bytes)))
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    control = machinery.SourceFileLoader('synthetic_cache_control', str(source)).get_code('synthetic_cache_control')
    assert 'CACHE_ONLY_SENTINEL' in control.co_consts
    assert 'SOURCE_ONLY_SENTINEL' not in control.co_consts
    before_source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    hook, denied, probes = _captured_metadata_cache_guard(monkeypatch)
    logical_root = Path(contract.BASE) / 'SYNTHETIC_CACHE_FALLBACK'
    attempted, returned = [], []
    class CapturedGuardSourceLoader(machinery.SourceFileLoader):
        def get_data(self, path):
            path = Path(path)
            logical = logical_root / path.relative_to(tmp_path)
            attempted.append(str(logical))
            # Run the actual captured C audit callback before the actual stdlib
            # get_data/open_code body. It must deny the cache before super().
            hook('open', (str(logical), 'r', os.O_RDONLY))
            data = super().get_data(str(path))
            returned.append((str(logical), len(data)))
            return data
    code = CapturedGuardSourceLoader('synthetic_guarded_source', str(source)).get_code('synthetic_guarded_source')
    assert isinstance(code, CodeType) and 'SOURCE_ONLY_SENTINEL' in code.co_consts
    assert 'CACHE_ONLY_SENTINEL' not in code.co_consts
    logical_cache = str(logical_root / cache.relative_to(tmp_path))
    logical_source = str(logical_root / source.name)
    assert attempted == [logical_cache, logical_source]
    assert returned == [(logical_source, len(source_bytes))]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before_source_sha
    assert probes == {'total_count': 1, 'sample_paths': [logical_cache],
        'protected_cache_bytes_read': 0, 'protected_cache_reads_authorized': False}
    assert denied == []
    # get_code returns a CodeType only; neither the cache nor source is exec'd.


@pytest.mark.parametrize('relative,flags,error', [
    ('GX1_RUNS/SYNTHETIC/raw.npz', os.O_RDONLY, RuntimeError),
    ('GX1_RUNS/SYNTHETIC/state.pt', os.O_RDONLY, RuntimeError),
    ('GX1_RUNS/SYNTHETIC/raw.parquet', os.O_RDONLY, RuntimeError),
    ('GX1_RUNS/SYNTHETIC/stray.pyc', os.O_RDONLY, RuntimeError),
    ('GX1_RUNS/SYNTHETIC/__pycache__/write.pyc', os.O_WRONLY | os.O_CREAT, RuntimeError),
    ('GX1_DATA/SYNTHETIC/__pycache__/read.pyc', os.O_RDONLY, PermissionError)],
    ids=['npz', 'pt', 'parquet', 'stray_pyc', 'cache_write', 'data_cache'])
def test_route_metadata_cache_scope_keeps_payload_and_write_guards(monkeypatch, relative, flags, error):
    hook, denied, probes = _captured_metadata_cache_guard(monkeypatch)
    path = '/home/andre2/' + relative
    with pytest.raises(error):
        hook('open', (path, None, flags))
    assert probes['protected_cache_bytes_read'] == 0
    assert probes['protected_cache_reads_authorized'] is False
    if error is PermissionError:
        assert probes['total_count'] == 1 and probes['sample_paths'] == [path] and denied == []
    else:
        assert probes['total_count'] == 0 and probes['sample_paths'] == []
        if not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            assert denied == [path]


def test_route_metadata_cache_probe_evidence_has_total_and_bounded_samples(monkeypatch):
    hook, denied, probes = _captured_metadata_cache_guard(monkeypatch)
    paths = [contract.BASE + f'/SYNTHETIC/__pycache__/probe_{i}.pyc' for i in range(18)]
    for path in paths:
        with pytest.raises(PermissionError):
            hook('open', (path, 'r', os.O_RDONLY))
    assert probes['total_count'] == 18 and probes['sample_paths'] == paths[:16]
    assert probes['protected_cache_bytes_read'] == 0 and probes['protected_cache_reads_authorized'] is False
    assert denied == []


@pytest.mark.parametrize('decision', ['FAILED', 'COMPLETE_READONLY_TRAIN_DIAGNOSIS'])
def test_route_native_main_requires_C_complete_and_preserves_terminal(tmp_path, monkeypatch, capsys, decision):
    from gx1.scripts import run_unified_exit_native_candidate_window_v1 as dispatcher
    calls = []
    progress = tmp_path / 'SYNTHETIC_PROGRESS.json'
    policy = tmp_path / 'SYNTHETIC_WINDOW.json'
    def guarded(*, execution_tier):
        assert execution_tier == 'canonical'
        calls.append('guard')
    def context(policy_path, policy_file_sha256):
        assert policy_path == policy and policy_file_sha256 == '0' * 64
        calls.append('context')
        return {'schema_version': contract.WINDOW_SCHEMA}, {}, {}, {}
    terminal = {'terminal': True, 'outcome': 'COMPLETE' if decision.startswith('COMPLETE') else 'FAILED',
        'optimizer_steps': 0, 'test_data_used': False}
    expected = {'decision': decision, 'progress': {'path': str(progress)}}
    def diagnostic(*, policy_path, policy_file_sha256, progress_path):
        assert policy_path == policy and policy_file_sha256 == '0' * 64 and progress_path == progress
        calls.append('C_worker')
        progress.write_text(json.dumps(terminal, sort_keys=True) + '\n')
        return expected
    monkeypatch.setattr(dispatcher, 'native', SimpleNamespace(trainer=SimpleNamespace(
        _require_cuda_trainer_guard_execution=guarded)))
    monkeypatch.setattr(dispatcher, '_context', context)
    monkeypatch.setattr(worker, 'run_native_direction_checkpoint_diagnostic_window', diagnostic)
    argv = ['--window-policy', str(policy), '--window-policy-file-sha256', '0' * 64,
        '--progress-path', str(progress)]
    if decision == 'FAILED':
        with pytest.raises(RuntimeError, match='NATIVE_DIRECTION_CHECKPOINT_DIAGNOSTIC_FAILED'):
            dispatcher.main(argv)
        assert capsys.readouterr().out == ''
    else:
        assert dispatcher.main(argv) == 0
        assert json.loads(capsys.readouterr().out) == expected
    assert calls == ['guard', 'context', 'C_worker']
    assert json.loads(progress.read_text()) == terminal
