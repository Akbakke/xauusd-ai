"""Capture frozen native direction-head inputs through the canonical GPU route.

No fit, reverse pass, optimizer step or raw feature rewrite. Original prefix
evaluation/gather/state ownership is reused, with original B16 boundaries.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

from gx1.contracts import entry_native_direction_frozen_extraction_v1 as authority
from gx1.scripts import audit_entry_native_direction_checkpoints_v1 as saved

require = authority.require


def phase_codes(functions):
    """Map Python bodies; cached functions emit body events only on cache misses."""
    from inspect import unwrap
    return {unwrap(function).__code__: name for function, name in functions.items()}


class PhaseTimings:
    """Inclusive observer; the original operation guard receives every event first."""
    def __init__(self, codes, *, wall=time.perf_counter_ns, cpu=time.process_time_ns):
        self.codes = dict(codes)
        self.wall, self.cpu = wall, cpu
        self.pending, self.stats = {}, {}

    def observe(self, frame, event):
        name = self.codes.get(frame.f_code)
        if name is None or event not in ('call', 'return'):
            return
        if event == 'call':
            require(id(frame) not in self.pending, 'TIMING_DUPLICATE_FRAME')
            self.pending[id(frame)] = (frame, name, self.wall(), self.cpu())
            stats = self.stats.setdefault(name, {'calls': 0, 'returns': 0,
                'wall_ns': 0, 'cpu_ns': 0, 'min_wall_ns': None, 'max_wall_ns': 0})
            stats['calls'] += 1
        else:
            require(id(frame) in self.pending, 'TIMING_UNMATCHED_RETURN')
            original, entered_name, wall, cpu = self.pending.pop(id(frame))
            require(original is frame and entered_name == name, 'TIMING_FRAME_IDENTITY')
            wall, cpu = self.wall() - wall, self.cpu() - cpu
            require(wall >= 0 and cpu >= 0, 'TIMING_MONOTONIC_CLOCK')
            stats = self.stats[name]
            stats['returns'] += 1
            stats['wall_ns'] += wall
            stats['cpu_ns'] += cpu
            stats['min_wall_ns'] = wall if stats['min_wall_ns'] is None else min(stats['min_wall_ns'], wall)
            stats['max_wall_ns'] = max(stats['max_wall_ns'], wall)

    def profile(self, guard):
        def profile(frame, event, arg):
            guard(frame, event, arg)
            self.observe(frame, event)
        return profile

    def require_complete(self, forwards):
        expected = {'prepare': 1, 'evaluate': 1, 'batch': forwards, 'context_tags': forwards,
            'device_batch': forwards, 'forward': forwards, 'hidden_capture': forwards}
        require(not self.pending and all(self.stats.get(name, {}).get('calls') == count
            and self.stats[name]['returns'] == count for name, count in expected.items()), 'TIMING_COMPLETE')

    def report(self):
        return {'scope': 'inclusive_wall_and_process_cpu', 'pending_frames': len(self.pending),
            'phases': {name: {'calls': value['calls'], 'returns': value['returns'],
                **{key.removesuffix('_ns') + '_seconds': (number / 1e9 if number is not None else None)
                    for key, number in value.items() if key.endswith('_ns')}}
                for name, value in self.stats.items()},
            'limits': ['Nested phase times overlap and must not be summed.',
                'Process CPU includes all process threads; this is not CUDA kernel timing.',
                'No extra CUDA synchronization; existing hidden/logit host copies are preserved.',
                'Observer overhead is included and has not been subtracted.']}


class HiddenCapture:
    """Observe exactly the FP32 128D argument of the existing linear head."""
    def __init__(self, model, *, torch, np, budget, progress):
        self.parts = []
        self.pending = False
        self.rows = 0
        self.last_progress = time.monotonic()
        def before(_module, args):
            require(not self.pending and len(args) == 1, 'HEAD_CAPTURE_ORDER')
            value = args[0]
            require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                and value.ndim == 2 and value.shape[1] == 128 and 1 <= value.shape[0] <= 16
                and not value.requires_grad, 'HEAD_CAPTURE_DOMAIN')
            budget.start('panel_forward')
            self.pending = True
            part = value.detach().cpu().numpy().copy()
            require(np.isfinite(part).all(), 'HEAD_CAPTURE_FINITE')
            self.parts.append(part)
            self.rows += len(part)
        def after(_module, _args, result):
            require(self.pending and result.dtype == torch.float32
                and tuple(result.shape) == (len(self.parts[-1]), 3), 'HEAD_OUTPUT_DOMAIN')
            self.pending = False
            budget.returned('panel_forward')
            # Same human-status cadence as the native campaign owner.
            if time.monotonic() - self.last_progress >= 900:
                progress()
                self.last_progress = time.monotonic()
        self.timing_codes = {before.__code__: 'hidden_capture'}
        self.handles = [model.direction_head.register_forward_pre_hook(before),
            model.direction_head.register_forward_hook(after)]

    def close(self):
        for handle in self.handles:
            handle.remove()


def select_partition_parents(full_selected, spec, np):
    require(spec['unit'] in authority.PARTITION_UNITS and spec == authority.unit_spec(spec['unit'])
        and all(type(spec['partition'][k]) is int for k in ('presentation_start', 'presentation_end', 'full_rows')),
        'FIXED_PARTITION_SPEC')
    part = spec['partition']
    require(isinstance(full_selected, np.ndarray) and full_selected.dtype == np.int64
        and full_selected.shape == (part['full_rows'],), 'FULL_PRESENTATION_DOMAIN')
    expected = np.arange(0, authority.FOLD['prefix_rows'], dtype=np.int64) if spec['cohort'] == 'prefix' else np.arange(
        authority.FOLD['oof_start_row'], authority.FOLD['oof_end_row'], dtype=np.int64)
    require(np.array_equal(np.sort(full_selected) if spec['cohort'] == 'prefix' else full_selected, expected),
        'FULL_ORIGINAL_PARENT_COHORT')
    return full_selected[part['presentation_start']:part['presentation_end']]


def canonical_arrays(measured, hidden_parts, *, selected, spec, weight, bias, np, reference_parents=None):
    arrays = measured['arrays']
    require(np.array_equal(arrays['parent_rows'], selected), 'EXTRACTION_PRESENTATION_JOIN')
    hidden = np.ascontiguousarray(np.concatenate(hidden_parts))
    require(hidden.dtype == np.float32 and hidden.shape == (spec['rows'], 128)
        and len(selected) == spec['rows'] and len(np.unique(selected)) == spec['rows']
        and np.isfinite(hidden).all(), 'EXACT_HIDDEN_COHORT')
    require(weight.dtype == bias.dtype == np.float32 and weight.shape == (3, 128) and bias.shape == (3,)
        and np.isfinite(weight).all() and np.isfinite(bias).all(), 'HEAD_PARAMETER_DOMAIN')
    # The forward order remains historical. Only the stored row layout changes.
    permutation = np.argsort(selected, kind='stable')
    if spec['unit'] in authority.PARTITION_UNITS:
        original = select_partition_parents(reference_parents, spec, np)
        require(np.array_equal(selected, original), 'PARTITION_PRESENTATION_JOIN')
        expected = np.sort(original)
    elif spec.get('diagnostic_only', False):
        require(spec == authority.unit_spec(authority.TIMING_UNIT)
            and isinstance(reference_parents, np.ndarray) and reference_parents.dtype == np.int64
            and reference_parents.shape == (256,) and np.array_equal(selected, reference_parents)
            and np.all((selected >= 0) & (selected < authority.FOLD['prefix_rows'])), 'TIMING_ORIGINAL_PARENTS')
        expected = np.sort(reference_parents)
    else:
        require(reference_parents is None, 'FULL_COHORT_NOT_SUBSET')
        expected = np.arange(0, authority.FOLD['prefix_rows'], dtype=np.int64) if spec['cohort'] == 'prefix' else np.arange(
            authority.FOLD['oof_start_row'], authority.FOLD['oof_end_row'], dtype=np.int64)
    require(np.array_equal(selected[permutation], expected), 'CANONICAL_PARENT_COHORT')
    result = {name: np.ascontiguousarray(value[permutation]) for name, value in arrays.items()}
    result.update(hidden=np.ascontiguousarray(hidden[permutation]),
        presentation_index=np.ascontiguousarray(
            permutation + spec.get('partition', {}).get('presentation_start', 0), dtype=np.int64),
        head_weight=np.ascontiguousarray(weight), head_bias=np.ascontiguousarray(bias))
    return result


def require_parity(actual, reference, *, first_rows, np):
    require(set(actual) == set(reference), 'PARITY_MEMBER_IDENTITY')
    require(all(actual[name].dtype == reference[name].dtype
        and np.array_equal(actual[name][:first_rows], reference[name])
        for name in reference), 'ORIGINAL_PREDICTION_BIT_PARITY')


def _baseline_arrays(ref, *, rows, common, np, contexts):
    require(common._hash_file(ref['path']) == ref, 'SAVED_PREDICTION_BYTES')
    before = common._stat(ref['path'])
    names = ('parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign', 'logit_up')
    context_names = tuple('context__' + name for name in contexts)
    with np.load(ref['path'], allow_pickle=False) as archive:
        require(archive.files == list(names) + list(context_names), 'SAVED_PREDICTION_MEMBERS')
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    require(common._stat(ref['path']) == before, 'SAVED_PREDICTION_BOOKEND')
    expected = {'parent_rows': (np.int64, (rows,)), 'bar_start_ns': (np.int64, (rows,)),
        'decision_ns': (np.int64, (rows,)), 'terminal_time_ns': (np.int64, (rows, 3)),
        'observed': (np.bool_, (rows, 3)), 'sign': (np.int8, (rows, 3)), 'logit_up': (np.float32, (rows, 3))}
    expected.update({name: (np.int8, (rows,)) for name in context_names})
    require(all(a.dtype == expected[name][0] and a.shape == expected[name][1] and np.isfinite(a).all()
        for name, a in arrays.items()), 'SAVED_PREDICTION_HEADERS')
    for array in arrays.values():
        array.setflags(write=False)
    return arrays


def _metadata(plan, route, common, metadata):
    for ref in [route['recipe_ref'], *route['metadata_refs'], authority.DESIGN, plan['units_design']]:
        common._read(ref, metadata)
    authority.require_admission(common._read(route['admission_ref'], metadata), route['recipe_ref'], plan['unit'])
    if 'gc_layout_metadata_parity' in plan:
        proof = common._read(plan['gc_layout_metadata_parity'], metadata)
        require(proof['status'] == 'COMPLETE_CONSUMED' and proof['metadata_exact_parity'] is True
            and proof['test_cases_passed'] == 6
            and proof['layout_contract_sha256'] == 'b67b9b94b78b4cf933c23e6738666911cd2c262ae3ab696448d42571e13bec6f'
            and all(value == 0 for value in proof['counters'].values()), 'QUALIFIED_GC_METADATA_PARITY')
    if plan['unit'] in authority.PARTITION_UNITS:
        proof = common._read(authority.PARTITION_PARITY, metadata)
        require(proof['status'] == 'COMPLETE_CONSUMED_PARITY_SPEED_PASS_FULL_BUDGET_FAIL'
            and proof['original_bit_parity']['bit_identical'] is True
            and proof['original_bit_parity']['rows'] == 256
            and all(proof['parity_fields'].values()), 'QUALIFIED_PARTITION_OUTPUT_PARITY')
    authority.require_independent_completion(plan, lambda ref: common._read(ref, metadata))
    proposal = authority.require_proposal(common._read(authority.PROPOSAL, metadata))
    original = common._read(proposal['existing_input_refs']['recipe'], metadata)
    qualified = common._read(proposal['existing_input_refs']['first_prefix_qualification'], metadata)
    require(qualified['technical_fit_resume_scoring_qualified'] is True
        and original['input_manifest'] == proposal['existing_input_refs']['input_manifest']
        and original['fold'] == authority.FOLD and original['numerical_recipe']['batch_size'] == 16
        and original['numerical_recipe']['max_updates'] == 2976, 'ORIGINAL_276_RECIPE')
    initial = common._read(authority.INITIAL_REPORT, metadata)
    require(initial['role'] == 'INITIAL_TRAIN' and initial['predictions'] == authority.INITIAL_PREDICTIONS
        and initial['fold'] == authority.FOLD and initial['initial_grad_nograd_parity'] is True,
        'ORIGINAL_INITIAL_REPORT')
    terminal = common._read(authority.TERMINAL_REPORT, metadata)
    require(terminal['role'] == 'OOF' and terminal['predictions'] == authority.TERMINAL_PREDICTIONS
        and terminal['fold'] == authority.FOLD, 'ORIGINAL_TERMINAL_REPORT')
    spec = authority.unit_spec(plan['unit'])
    item = next(s for s in proposal['planned_saved_states'] if s['role'] == spec['state'][0])
    declared = common._read(item['declaring_JSON'], metadata)
    ref = item['planned_PT_ref']
    key = 'own_initial_state' if item['updates'] == 0 else 'full_state'
    require(declared[key] == ref and declared['recipe'] == proposal['existing_input_refs']['recipe']
        and declared['fold'] == authority.FOLD and declared['feature_view_identity'] == initial['feature_view_identity'],
        'DECLARED_CHECKPOINT')
    if item['updates'] == 0:
        require(declared['initial_measurement'] == authority.INITIAL_REPORT
            and initial['own_initial_state'] == ref == qualified['own_initial_state'], 'DECLARED_OWN_0U')
    else:
        require(declared['parent_cursor'] == item['consumed_parent_cursor'] and declared['optimizer_updates'] == item['updates']
            and declared['outcome'] == declared['phase_state'] == 'COMPLETE'
            and declared['oof_measurement'] == authority.TERMINAL_REPORT, 'DECLARED_TERMINAL')
    observed = common._stat(ref['path'])
    require(observed['st_size'] == item['stat_size_bytes'] and not observed['st_mode'] & 0o222, 'CHECKPOINT_READONLY_SIZE')
    authority.require_source_closure(plan, original, proposal)
    for owner in plan['source_owners'].values():
        require(common._hash_file(owner['path']) == owner, 'CURRENT_SOURCE_BYTES')
    return original, initial, ref


def _publish_running(route, output, budget, counts, common):
    progress = authority.build_frozen_extraction_running_progress(window=route['window'], plan=route['plan'],
        invocation=route['invocation'], calls=budget.calls, returns=budget.returns)
    return saved._publish_progress(output / 'PROGRESS.json', progress, counts,
        route['recipe']['budget']['max_json_output_bytes'] - authority.TERMINAL_JSON_RESERVE_BYTES, common)


def _publish_terminal(result, *, route, output, budget, counts, common):
    """Three immutable terminal JSONs share the original total write budget."""
    limit = route['recipe']['budget']['max_json_output_bytes']
    reserve = authority.TERMINAL_JSON_RESERVE_BYTES
    result = dict(result)
    for compact in (False, True):
        if compact:
            result = {key: result[key] for key in ('recipe', 'admission', 'campaign_plan', 'operation',
                'unit', 'optimizer_steps', 'test_data_used', 'observed_calls', 'observed_returns')}
            result.update(decision='FAILED', error={'type': 'RuntimeError', 'message': 'TERMINAL_JSON_RESERVE_EXCEEDED'})
        result_ref = {'path': str(output / 'RESULT.json'), 'sha256': hashlib.sha256(saved._json_bytes(result)).hexdigest()}
        outcome = 'COMPLETE' if result['decision'] == 'COMPLETE_FROZEN_READOUT_EXTRACTION' else 'FAILED'
        pointer = authority.build_frozen_extraction_pointer(recipe_ref=route['recipe_ref'], result_ref=result_ref,
            outcome=outcome, calls=budget.calls, returns=budget.returns, unit=route['recipe']['unit'])
        authority.require_frozen_extraction_pointer(pointer, expected_recipe=route['recipe_ref'], expected_unit=route['recipe']['unit'], verify_files=False)
        pointer_ref = {'path': route['window']['campaign_cursor_path'], 'sha256': hashlib.sha256(saved._json_bytes(pointer)).hexdigest()}
        progress = authority.build_frozen_extraction_progress(window=route['window'], plan=route['plan'],
            invocation=route['invocation'], checkpoint_pointer=pointer_ref, pointer=pointer)
        authority.require_frozen_extraction_progress(progress, plan_sha256=route['plan']['plan_sha256'],
            invocation=route['invocation'], expected_selection_receipt_sha256=None, verify_file=False)
        total = sum(len(saved._json_bytes(value)) for value in (result, pointer, progress))
        if total <= reserve:
            break
    authority.require(total <= reserve and counts['json_bytes'] + total <= limit, 'TERMINAL_JSON_BUDGET')
    authority.require(common._json(output / 'RESULT.json', result, counts, limit) == result_ref, 'RESULT_PUBLISH_HASH')
    authority.require(common._json(Path(pointer_ref['path']), pointer, counts, limit) == pointer_ref, 'POINTER_PUBLISH_HASH')
    saved._publish_progress(Path(route['window']['progress_path']), progress, counts, limit, common)
    return result


def run_native_direction_frozen_extraction_window(*, policy_path, policy_file_sha256, progress_path):
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    limits = require_guarded_cuda_trainer_execution()
    route = authority.require_native_direction_frozen_extraction_context(policy_path, policy_file_sha256, progress_path)
    sys.dont_write_bytecode = True
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix
    plan, plan_ref, binding = route['recipe'], route['recipe_ref'], route['binding']
    spec = authority.unit_spec(plan['unit'])
    output = Path(plan['run_root'])
    output.mkdir()  # Exclusive once-only claim, before payload.
    counts = {'json_bytes': 0, 'resume_state_load_calls': 0, 'resume_state_load_returns': 0}
    budget = authority.CallBudget(plan['unit'])
    phase = {'name': 'METADATA'}
    denied, delegated_denied, originals, metadata = [], [], {}, {}
    cache_probes = {'total_count': 0, 'sample_paths': [], 'protected_cache_bytes_read': 0,
        'protected_cache_reads_authorized': False}
    started = time.monotonic()
    before_io = common._io()
    old_alarm, old_profile = signal.getsignal(signal.SIGALRM), sys.getprofile()
    previous_tempdir = tempfile.tempdir
    context = None
    timings = None
    body_limit = plan['budget']['max_json_output_bytes'] - authority.TERMINAL_JSON_RESERVE_BYTES
    result = {'decision': 'FAILED', 'error': None, 'operation': authority.OPERATION, 'unit': plan['unit'],
        'diagnostic_only': spec['diagnostic_only'],
        'recipe': plan_ref, 'admission': route['admission_ref'], 'campaign_plan': binding['plan'],
        'test_data_used': False, 'optimizer_steps': 0, 'native_optimizer_steps': 0,
        'limits': list(authority.PARTITION_LIMITS if plan['unit'] in authority.PARTITION_UNITS else authority.LIMITS),
        'source_transport_transitions': plan['transport_source_transitions'],
        'metadata_blocked_bytecode_cache_probes': cache_probes}
    def deadline(_signum, _frame):
        raise TimeoutError('NATIVE_DIRECTION_FROZEN_EXTRACTION_WALL_LIMIT')
    def progress():
        return _publish_running(route, output, budget, counts, common)
    try:
        tempfile.tempdir = str(output)
        denied = saved._install_guard(phase, output, cache_probes)
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(plan['budget']['max_wall_seconds'])
        common._json(output / 'CLAIM.json', {'recipe': plan_ref, 'admission': route['admission_ref'],
            'campaign_plan': binding['plan'], 'invocation_sha256': route['invocation']['invocation_sha256'],
            'active_marker_sha256': route['marker']['marker_sha256'], 'limits': limits,
            'operation_limits': budget.limits}, counts, body_limit)
        progress()
        recipe, initial, state_ref = _metadata(plan, route, common, metadata)
        payload_refs = [state_ref] + ([spec['parity']] if spec['parity'] is not None else [])
        originals = {ref['path']: common._stat(ref['path']) for ref in payload_refs}
        metadata_options = {'gc_layout_metadata_parity': plan.get('gc_layout_metadata_parity')}
        if 'independent_completion' in plan:
            metadata_options['independent_completion'] = plan['independent_completion']
        context = prefix._metadata(recipe, common, **metadata_options)
        context['info']['metadata'].update(metadata)
        common._freeze(plan, binding, context)
        guard_context = {**context, 'info': {**context['info'], 'metadata': {
            **context['info']['metadata'], str(output): {'read_guard_directory_only': True},
            **{ref['path']: {'ref': ref, 'stat': originals[ref['path']]} for ref in payload_refs}}}}
        delegated_denied = common._install_read_guard(guard_context, output, output)
        import numpy as np
        import torch
        from gx1.contracts import entry_native_direction_feature_view_v1 as view_owner
        from gx1.contracts import entry_native_direction_research_state_v1 as state_owner
        from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
        from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as tree_sha
        from gx1.contracts.entry_native_direction_context_report_v1 import CONTEXT_SPEC
        codes = saved._monitored_codes(context, np, torch, prefix, view_owner, trainer, state_owner)
        profile_guard = saved._profile_budget(budget, codes, phase, old_profile)
        if spec['diagnostic_only']:
            from gx1.contracts import entry_native_direction_context_report_v1 as reporter
            timings = PhaseTimings(phase_codes({prefix._prepare: 'prepare', common._load_state: 'checkpoint_load',
                view_owner.build_native_direction_research_model: 'model_build',
                state_owner.restore_native_direction_research_state: 'restore',
                prefix._evaluate: 'evaluate', prefix._batch: 'batch',
                reporter.tag_native_direction_contexts: 'context_tags',
                reporter.score_native_direction_logits: 'scoring',
                prefix._device_batch: 'device_batch', prefix._forward: 'forward',
                view_owner._canonical: 'canonical_layout',
                view_owner.history.require_gc_mtf_history_surfaces: 'gc_surface_validation',
                view_owner.get_model_native_multi_tf_route_windows: 'mtf_windows',
                view_owner.history.gather_gc_mtf_history_rows: 'gc_windows'}))
        sys.setprofile(timings.profile(profile_guard) if timings is not None else profile_guard)
        phase['name'] = 'PREPARE'
        prepared, targets, view, context_map, order = prefix._prepare(context, recipe, np)
        prior = prefix._prefix_prior(prepared['arrays'], targets, authority.FOLD, np)
        require(prior == initial['own_prefix_prior'], 'OWN_PREFIX_PRIOR')
        phase['name'] = 'BASELINE'
        baseline_rows = 256 if spec['cohort'] == 'prefix' else spec['rows']
        baseline = None if spec['parity'] is None else _baseline_arrays(spec['parity'],
            rows=baseline_rows, common=common, np=np, contexts=[name for name, _ in CONTEXT_SPEC])
        phase['name'] = 'LOAD'
        budget.start('checkpoint_load')
        state = common._load_state(state_ref, torch=torch, maximum=plan['budget']['max_checkpoint_bytes'], counters=counts)
        budget.returned('checkpoint_load')
        require(state['schema_version'] == state_owner.SCHEMA_VERSION
            and state['contract_sha256'] == tree_sha({k: v for k, v in state.items() if k != 'contract_sha256'})
            and state['cursor'] == spec['state'][1] and state['optimizer_updates'] == spec['state'][2], 'STATE_SEAL_CURSOR')
        device = torch.device('cuda', 0)
        trainer._set_deterministic(recipe['numerical_recipe']['seed'], device, recipe['numerical_recipe']['precision_policy'])
        phase['name'] = 'CONSTRUCT'
        budget.start('model_build')
        model = view_owner.build_native_direction_research_model(view,
            prefix_normalization=context['fitted'], constructor_metadata=context['constructor'], device=device)
        budget.returned('model_build')
        model.train()
        budget.start('optimizer_build')
        optimizer, _binding = state_owner.build_native_direction_research_optimizer(model, recipe=recipe['numerical_recipe'])
        budget.returned('optimizer_build')
        parent_order = torch.from_numpy(order.copy())
        fresh = state_owner.capture_native_direction_research_state(model, optimizer, recipe=recipe['numerical_recipe'],
            feature_view_identity=model.feature_view_identity, parent_order=parent_order, cursor=0, optimizer_updates=0, device=device)
        identity_keys = ('binding', 'model_state', 'buffer_state', 'optimizer_state', 'rng_state', 'module_training')
        fresh_identity = tree_sha({key: fresh[key] for key in identity_keys})
        require(fresh_identity == initial['fresh_identity_sha256']
            and model.feature_view_identity == initial['feature_view_identity'], 'OWN_FRESH_INITIAL_IDENTITY')
        if spec['state'][0] == 'own0U':
            require(fresh_identity == tree_sha({key: state[key] for key in identity_keys}), 'SAVED_0U_IDENTITY')
        del fresh
        phase['name'] = 'RESTORE'
        state_owner.restore_native_direction_research_state(state, model, optimizer, recipe=recipe['numerical_recipe'],
            feature_view_identity=model.feature_view_identity, parent_order=parent_order, device=device)
        selected = order if spec['cohort'] == 'prefix' else prepared['arrays']['parent_rows'][
            authority.FOLD['oof_start_row']:authority.FOLD['oof_end_row']]
        full_selected = selected
        if plan['unit'] in authority.PARTITION_UNITS:
            selected = select_partition_parents(full_selected, spec, np)
        elif spec['diagnostic_only']:
            selected = selected[:256]
            require(np.array_equal(selected, baseline['parent_rows']), 'TIMING_INITIAL_PRESENTATION_ORDER')
        require(selected.dtype == np.int64 and selected.shape == (spec['rows'],), 'EXACT_PRESENTATION_ROWS')
        forward_counts = {'forward_calls': 0, 'forward_returns': 0, 'forward_kind_calls': {}}
        phase['name'] = 'EXTRACT'
        recorder = HiddenCapture(model, torch=torch, np=np, budget=budget, progress=progress)
        if timings is not None:
            timings.codes.update(recorder.timing_codes)
        try:
            with saved._measurement(model, optimizer, device=device, trainer=trainer,
                    prefix=prefix, tree_sha=tree_sha, evaluation=True):
                measured, completed = prefix._evaluate(model, prepared, targets, view, context_map, selected,
                    cutoff=spec['cutoff_ns'], kind='F_FROZEN_EXTRACTION', counters=forward_counts,
                    ceiling=spec['forwards'], device=device, torch=torch, np=np, tree_sha=tree_sha)
            require(completed == spec['rows'] and recorder.rows == spec['rows'] and not recorder.pending, 'COMPLETE_COHORT')
        finally:
            recorder.close()
        parity = None
        if baseline is not None:
            require_parity(measured['arrays'], baseline, first_rows=baseline_rows, np=np)
            parity = {'reference': spec['parity'], 'rows': baseline_rows, 'bit_identical': True}
        arrays = canonical_arrays(measured, recorder.parts, selected=selected, spec=spec,
            weight=model.direction_head.weight.detach().cpu().numpy().copy(),
            bias=model.direction_head.bias.detach().cpu().numpy().copy(), np=np,
            reference_parents=full_selected if plan['unit'] in authority.PARTITION_UNITS else (
                baseline['parent_rows'] if spec['diagnostic_only'] else None))
        representation = prefix._save_predictions(output / 'REPRESENTATIONS.npz', arrays, np=np, common=common, tree_sha=tree_sha)
        report_ref = common._json(output / 'NATIVE_REPORT.json', {'report': measured['report'],
            'unit': plan['unit'], 'cutoff_ns': spec['cutoff_ns'], 'own_prefix_prior': prior,
            'original_prediction_parity': parity, 'state_preservation': measured['state_preservation']}, counts, body_limit)
        require(counts['resume_state_load_calls'] == counts['resume_state_load_returns'] == 1
            and forward_counts['forward_calls'] == forward_counts['forward_returns'] == spec['forwards'], 'DELEGATED_COUNTS')
        require(not denied and not delegated_denied, 'FORBIDDEN_INPUT_OPEN')
        if timings is not None:
            timings.require_complete(spec['forwards'])
        result.update(decision='COMPLETE_FROZEN_READOUT_EXTRACTION', calls=budget.complete(),
            state=state_ref, state_role=spec['state'][0], cohort=spec['cohort'], representations=representation,
            representation_rows=spec['rows'], hidden_width=128, native_report=report_ref,
            presentation_parent_rows_sha256=tree_sha(selected), canonical_parent_rows_sha256=tree_sha(arrays['parent_rows']),
            head_parameters_sha256=tree_sha({'weight': arrays['head_weight'], 'bias': arrays['head_bias']}),
            feature_view_identity=model.feature_view_identity, own_prefix_prior=prior,
            original_prediction_parity=parity, fresh_identity_sha256=fresh_identity,
            predictive_value_qualified=False, economic_edge_qualified=False)
        if plan['unit'] in authority.PARTITION_UNITS:
            result['partition'] = {**spec['partition'],
                'full_presentation_parent_rows_sha256': tree_sha(full_selected)}
    except Exception as exc:
        result['error'] = saved._failure_metadata(exc)
    finally:
        try:
            sys.setprofile(old_profile)
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_alarm)
            phase['name'] = 'FINAL'
            originals_preserved = source_unchanged = False
            try:
                originals_preserved = len(originals) == 1 + int(spec['parity'] is not None) and all(
                    common._stat(path) == before for path, before in originals.items())
                require(not originals or originals_preserved, 'ORIGINAL_STAT_BOOKENDS')
                if context is not None:
                    common._freeze(plan, binding, context)
                    source_unchanged = True
                require(common._io()['rchar'] - before_io['rchar'] <= plan['budget']['max_rchar_bytes'], 'RCHAR_CEILING')
            except Exception as exc:
                result.update(decision='FAILED', error=saved._failure_metadata(exc))
            result.update(elapsed_seconds=time.monotonic() - started, limits_proof=limits,
                observed_calls=dict(budget.calls), observed_returns=dict(budget.returns),
                denied_phase_opens=denied, delegated_denied_opens=delegated_denied,
                originals_stat_inventory_count=len(originals), originals_stat_preserved=originals_preserved,
                source_unchanged=source_unchanged)
            if timings is not None:
                result['phase_timings'] = timings.report()
            result = _publish_terminal(result, route=route, output=output, budget=budget, counts=counts, common=common)
        finally:
            tempfile.tempdir = previous_tempdir
    return result


def main():
    require(False, 'DIRECT_MODULE_FORBIDDEN_USE_CANONICAL_NATIVE_WINDOW')


if __name__ == '__main__':
    main()
