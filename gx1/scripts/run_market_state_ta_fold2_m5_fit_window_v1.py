"""Fold 2 M5 direction fit core. No CLI, campaign route, or admission yet."""
from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import torch

from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import (
    ADAMW_DEFAULTS, BATCH_VISITS, INITIAL_MODEL_SHA256, LOSS_SHA256,
    MODEL_CONTRACT_SHA256, TRAIN_ROWS, VIEW_SHA256,
    batch_bounds, INITIAL_PLAN_REF, require_reused_input_refs,
)
from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_fit_batch
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.market_state_time_query_native_F_objective_v1 import (
    require_m5_equal_observation_axes,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('FOLD2_M5_FIT_' + reason)


def fit_one_pass(*, model, optimizer, parent_rows: np.ndarray,
                 issue_ns: np.ndarray, state_bound, future_bound,
                 gather_features: Callable, deadline_monotonic: float,
                 progress_state: dict | None = None) -> dict:
    """Consume every fixed TRAIN batch once after the caller proves admission.

    The caller owns guarded launch, exact source/start binding, feature inputs,
    receipts, and failure consumption. This function cannot grant execution.
    """
    _need(canonical_model_state_sha256(model.state_dict())
          == INITIAL_MODEL_SHA256, 'FRESH_INITIAL_STATE')
    parameters = list(model.parameters())
    _need(type(optimizer) is torch.optim.AdamW and not optimizer.state
          and optimizer.defaults == ADAMW_DEFAULTS
          and len(optimizer.param_groups) == 1
          and {id(p) for p in optimizer.param_groups[0]['params']}
              == {id(p) for p in parameters}
          and model.training and all(parameter.grad is None for parameter in parameters),
          'EMPTY_OPTIMIZER_AND_TRAIN_MODE')
    _need(isinstance(parent_rows, np.ndarray) and isinstance(issue_ns, np.ndarray)
          and parent_rows.dtype == issue_ns.dtype == np.dtype('int64')
          and parent_rows.shape == issue_ns.shape == (TRAIN_ROWS,), 'TRAIN_AXES')
    require_m5_equal_observation_axes(parent_rows, issue_ns, fold_id=2)
    for bound in (state_bound, future_bound):
        _need(np.array_equal(bound.parent_rows, parent_rows)
              and np.array_equal(bound.decision_ns, issue_ns), 'TARGET_AXES')
    _need(type(deadline_monotonic) is float
          and time.monotonic() < deadline_monotonic, 'DEADLINE')
    optimizer.zero_grad(set_to_none=True)
    records, updates, skipped = [], 0, 0
    if progress_state is not None:
        progress_state.update(batch_visits=0, optimizer_updates=0,
                              skipped_batches=0, parent_cursor=0,
                              batch_visit_in_flight=False,
                              optimizer_step_in_flight=False)
    for index in range(BATCH_VISITS):
        _need(time.monotonic() < deadline_monotonic, 'TIME_BUDGET')
        left, right = batch_bounds(index)
        if progress_state is not None:
            progress_state['batch_visit_in_flight'] = True
        rows, issues = parent_rows[left:right], issue_ns[left:right]
        features = gather_features(left, right, rows, issues)
        report = evaluate_fit_batch(
            model, features, state_bound, future_bound, rows, issues,
            fold_id=2, pass_index=0, seed=20260911, future_weight=1.0,
            m5_equal_observation=True, fit_batch_start=left,
            m5_direction_only=True)
        _need(report['loss_contract_sha256'] == LOSS_SHA256
              and report['model_contract_sha256'] == MODEL_CONTRACT_SHA256
              and bool(torch.isfinite(report['loss']).all())
              and report['skip_update'] is (report['count'] == 0), 'LOSS_AND_SUPPORT')
        record = {'batch_index': index, 'start': left, 'stop': right,
                  'known_side_count': int(report['count']),
                  'skip_update': report['skip_update'],
                  'loss': float(report['loss'].detach())}
        if report['skip_update']:
            skipped += 1
        else:
            report['loss'].backward()
            if progress_state is not None:
                progress_state['optimizer_step_in_flight'] = True
            norm = float(trainer._optimizer_step_with_finite_gradients(
                model=model, optimizer=optimizer))
            _need(math.isfinite(norm), 'GRADIENT_NORM')
            record['gradient_norm'] = norm
            updates += 1
        optimizer.zero_grad(set_to_none=True)
        records.append(record)
        if progress_state is not None:
            progress_state.update(batch_visits=index + 1,
                                  optimizer_updates=updates,
                                  skipped_batches=skipped, parent_cursor=right,
                                  batch_visit_in_flight=False,
                                  optimizer_step_in_flight=False)
        del features, report
    _need(len(records) == BATCH_VISITS and records[-1]['stop'] == TRAIN_ROWS
          and updates + skipped == BATCH_VISITS
          and all(parameter.grad is None for parameter in model.parameters()),
          'COMPLETE_ONE_PASS')
    return {'batch_visits': BATCH_VISITS, 'optimizer_updates': updates,
            'skipped_batches': skipped, 'training_records': records,
            'parent_cursor': TRAIN_ROWS, 'checkpoint_writes': 0,
            'test_data_used': False}


def prepare_qualified_inputs(*, initial_plan_ref: dict, design: dict,
                             device: torch.device) -> dict:
    """Reconstruct the completed fold 2 INITIAL input path for one admitted fit."""
    import hashlib
    from pathlib import Path

    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
    from gx1.audit.observe_market_state_ta_fold2_fresh_initial_v1 import (
        TF, panel_positions, read, ref,
    )
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts.entry_anchor_span_feature_inputs_v1 import build_anchor_span_inputs
    from gx1.contracts.entry_exit_feature_surface_v1 import load_m1_feature_surface_times
    from gx1.contracts.entry_model_native_input_normalization_v1 import (
        project_ta_input_normalization_contract,
    )
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import (
        load_bound_five_endpoint_targets,
    )
    from gx1.contracts.market_state_future_displacement_candidate_v1 import (
        load_bound_future_displacement,
    )
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import (
        require_endpoint_priors,
    )

    _need(device.type == 'cuda' and device.index == 0, 'CUDA_DEVICE')
    inputs = require_reused_input_refs(initial_plan_ref)
    train_plan, later_plan = read(inputs['train_plan']), read(inputs['later_plan'])
    train_result, later_result = read(inputs['physical_result']), read(inputs['later_result'])
    later_terminal, coverage = read(inputs['later_terminal']), read(inputs['later_coverage'])
    train_gather, later_gather = read(inputs['train_gather_result']), read(inputs['later_gather_result'])
    _need(train_gather['status'] == 'COMPLETE_FULL_FOLD2_TRAIN_INPUT_GATHER'
          and train_gather['source_unchanged'] is True
          and later_result['status'] == later_terminal['status'] == 'COMPLETE'
          and later_terminal['exit_code'] == 0
          and later_terminal['result'] == inputs['later_result']
          and later_terminal['gather_result'] == inputs['later_gather_result']
          and later_gather['status'] == coverage['status'] == 'COMPLETE'
          and later_result['source_unchanged'] is later_gather['source_unchanged'] is True
          and later_gather['counts']['parent_rows'] == 4096
          and not any(x['test_data_used'] for x in (train_gather, later_gather, later_result)),
          'QUALIFIED_INPUTS')
    view, packet = read(inputs['view']), read(inputs['packet'])
    normalization, constructor, priors = [read(inputs[key])
                                          for key in ('normalization', 'constructor', 'priors')]
    _need(view['contract_sha256'] == VIEW_SHA256
          and view['prefix']['rows'] == TRAIN_ROWS
          and view['dedicated_m5_side'] is view['dedicated_tf_activity'] is True
          and view['full_ta_feature_profile'] is True
          and view['prefix_normalization']['ta_base_normalization_sha256']
              == normalization['contract_sha256']
          and normalization == project_ta_input_normalization_contract(
              packet['base_normalization'], unobserved_units_profile=None),
          'FOLD2_VIEW')
    require_endpoint_priors(priors, fold_id=2, ta_only=True, dedicated_m5_side=True)
    train_pos, later_pos = panel_positions(np, design, later_plan)
    provider = mature._provider_module()
    provider_plan = read(inputs['provider_plan'])
    info = provider.validate_metadata(provider_plan)
    axes, _ = provider.read_axes(provider_plan, info, np)
    _need(len(axes['parent_rows']) == view['full_axes']['rows'], 'FULL_AXES')
    for array in axes.values():
        array.setflags(write=False)
    for key in ('parent_rows', 'bar_start_ns', 'decision_ns'):
        _need(hashlib.sha256(axes[key].astype('<i8').tobytes()).hexdigest()
              == view['full_axes'][key + '_sha256'], 'FULL_AXIS_' + key)
    state_bound, future_bound = load_bound_five_endpoint_targets(), load_bound_future_displacement()
    oof = state_bound.mature_oof_axes(fold_id=2)
    _need(np.array_equal(axes['parent_rows'][:TRAIN_ROWS], oof['fit_parent_rows'])
          and np.array_equal(axes['parent_rows'][47602:64890], oof['score_parent_rows'])
          and np.array_equal(axes['decision_ns'][:TRAIN_ROWS], oof['fit_decision_ns'])
          and np.array_equal(axes['decision_ns'][47602:64890], oof['score_decision_ns']),
          'FOLD2_TARGET_AXES')
    runtime = later_plan['replay_plan']
    groups = provider.map_reuse(info['reuse'], np)
    source, current = groups['source050'], groups['current051']
    times = load_m1_feature_surface_times(
        Path(runtime['local_clock_path']), context='FOLD2_M5_FIT_FULL_PARENT_CLOCK',
        expected_bar_seconds=300).asi8
    positions = np.searchsorted(times, axes['bar_start_ns']).astype(np.int64)
    _need(np.all(positions < len(times))
          and np.array_equal(times[positions], axes['bar_start_ns']), 'LOCAL_CLOCK')
    times.setflags(write=False)
    positions.setflags(write=False)
    seq = dict(signal=source['signal'], times_ns=times,
               sample_positions=positions, audit=info['audit'])
    values, clocks, mapped = {}, {}, []
    for tf in TF:
        values[tf] = np.load(runtime['caches'][tf]['feats']['path'], mmap_mode='r',
                             allow_pickle=False)
        clocks[tf] = np.load(runtime['caches'][tf]['ts']['path'], mmap_mode='r',
                             allow_pickle=False)
        mapped.extend((values[tf], clocks[tf]))
    names = packet['parent_prefix_normalization']['base_normalization']['surfaces']

    def overlay(result, *, later):
        payloads = {}
        for lane in ('local', *TF):
            binding = result['surfaces'][lane]['artifact']
            _need(ref(binding['path']) == binding, 'SIDECAR_' + lane)
            with np.load(binding['path'], allow_pickle=False) as archive:
                payloads[lane] = {key: archive[key] for key in archive.files}
        return build_anchor_span_inputs(
            view, entry_sequence_source=seq, snap_x=current['snap'],
            ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
            parent_signal_names=names['signal']['field_names'],
            parent_mtf_names=names['mtf_m5']['field_names'],
            parent_mtf_values=values, parent_mtf_times=clocks,
            span_payloads=payloads,
            physical_populations=(coverage['physical_populations'] if later else
                                  {lane: x['population'] for lane, x in result['surfaces'].items()}),
            unobserved_units_profile=None,
            **({'later_coverage': coverage, 'later_axes': axes} if later else {}))

    overlays = {'TRAIN': overlay(train_result, later=False),
                'LATER': overlay(later_result, later=True)}

    def features(label, rows, issues):
        _need(label in overlays and len(rows) == len(issues), 'GATHER_REQUEST')
        gathered = views.gather_native_direction_feature_batch(
            view, parent_rows=axes['parent_rows'], bar_start_ns=axes['bar_start_ns'],
            decision_ns=axes['decision_ns'], entry_sequence_source=seq,
            snap_x=current['snap'], ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
            multi_tf_sources=overlays[label].mtf_sources, gc_surfaces=None, gc_layout=None,
            batch_parent_rows=rows, unobserved_units_profile=None,
            anchor_span_inputs=overlays[label])
        _need(gathered['witness']['parent_rows'] == rows.tolist()
              and gathered['witness']['decision_ns'] == issues.tolist(), 'GATHER_AXES')
        return {key: torch.from_numpy(value).to(device)
                for key, value in gathered['features'].items()}

    def release():
        for group in groups.values():
            for array in group.values():
                provider.flush_pages(array)
        for array in mapped:
            provider.flush_pages(array)

    return {'inputs': inputs, 'view': view, 'packet': packet,
            'normalization': normalization, 'constructor': constructor,
            'priors': priors, 'axes': axes, 'train_positions': train_pos,
            'later_positions': later_pos, 'state_bound': state_bound,
            'future_bound': future_bound, 'features': features, 'release': release}


def evaluate_fixed_panels(*, model, context: dict, deadline_monotonic: float) -> dict:
    """Score the preselected TRAIN and Jan-Mar later panels after the fixed pass."""
    from gx1.audit.score_market_state_ta_fold2_m5_fit_v1 import score_capture
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import (
        evaluate_endpoint_batch,
    )
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective

    _need(type(deadline_monotonic) is float
          and time.monotonic() < deadline_monotonic, 'READOUT_DEADLINE')
    model.eval()
    state_before = canonical_model_state_sha256(model.state_dict())
    axes = context['axes']
    captures, readouts, measurements = {}, {}, {}
    forwards = 0
    with torch.inference_mode():
        for label, selected, partition in (
            ('TRAIN', context['train_positions'], 'fit'),
            ('LATER', context['later_positions'], 'score'),
        ):
            rows = axes['parent_rows'][selected]
            issues = axes['decision_ns'][selected]
            _need(len(rows) == len(issues) == 4096, 'PANEL_ROWS')
            truth = {
                task: bound.gather(rows, issues, fold_id=2, partition=partition)
                for task, bound in (('state', context['state_bound']),
                                    ('future', context['future_bound']))
            }
            if partition == 'fit':
                _need(all(bool(value['mature'].all()) for value in truth.values()),
                      'TRAIN_TARGET_MATURITY')
            raw = {task: np.empty((4096, 5, 2), dtype=np.float32)
                   for task in ('state', 'future')}
            total = None
            for left in range(0, 4096, 16):
                _need(time.monotonic() < deadline_monotonic, 'READOUT_TIME_BUDGET')
                right = left + 16
                batch_rows, batch_issues = rows[left:right], issues[left:right]
                features = context['features'](label, batch_rows, batch_issues)
                report = evaluate_endpoint_batch(
                    model, features, context['state_bound'], context['future_bound'],
                    batch_rows, batch_issues, fold_id=2, partition=partition,
                    future_weight=1.0, frozen_priors=context['priors'])
                total = objective.merge_endpoint_measurements(
                    total, report['measurement'])
                for task in raw:
                    raw[task][left:right] = report[task + '_activity_side_logits']
                forwards += 1
                del features, report
            _need(total['rows'] == 4096 and total['batches'] == 256,
                  'MEASUREMENT_COUNTS')
            measurements[label] = objective.finalize_endpoint_measurement(total)
            capture = {'parents': rows, 'issues': issues,
                       'state_raw': raw['state'], 'future_raw': raw['future'],
                       **{task + '_' + key: truth[task][key]
                          for task in ('state', 'future')
                          for key in ('allowed', 'eligible', 'mature')}}
            captures[label] = capture
            readouts[label] = score_capture(capture)
    _need(forwards == 512
          and canonical_model_state_sha256(model.state_dict()) == state_before,
          'PASSIVE_FINAL_READOUT')
    return {'captures': captures, 'readouts': readouts,
            'measurements': measurements, 'evaluation_forwards': forwards,
            'final_model_state_sha256': state_before,
            'test_data_used': False}


def _execute_fixed_fit_and_readout(*, launch: dict, review_ref: dict,
                                   initial_result_ref: dict,
                                   deadline_monotonic: float,
                                   progress_state: dict | None = None) -> dict:
    """Inner one-pass work; only a verified guarded campaign may call it."""
    import json
    from pathlib import Path

    from gx1.audit.preflight_market_state_ta_fold2_m5_fit_v1 import read_ref
    from gx1.audit.score_market_state_ta_fold2_m5_fit_v1 import (
        paired_later_f_m5, read_capture,
    )
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_guarded_cuda_trainer_execution,
    )
    from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import (
        REPO, SCOPE, STAGE, require_fixed_design,
    )
    from gx1.contracts.market_state_time_query_native_F_model_v1 import (
        NativeFutureDualModel,
    )
    from gx1.contracts.entry_native_direction_feature_view_v1 import (
        build_native_direction_research_model,
    )
    from gx1.audit.observe_market_state_ta_fold2_fresh_initial_v1 import read

    policy = json.loads((REPO / 'NEXT_RUN_POLICY.json').read_text())
    gate = policy.get(STAGE, {})
    _need(policy['training_enabled'] is True
          and policy['native_training_enabled'] is True
          and policy['full_epoch_training_allowed'] is True
          and gate.get('status') == 'ADMITTED'
          and gate.get('execution_authorized') is True
          and gate.get('data_reads_authorized') is True
          and gate.get('genuine_constructor_authorized') is True
          and gate.get('genuine_forward_authorized') is True
          and gate.get('genuine_backward_authorized') is True
          and gate.get('genuine_updates_authorized') is True
          and gate.get('genuine_predictor_fit_authorized') is True
          and gate.get('training_enabled') is True
          and gate.get('native_training_enabled') is True
          and gate.get('full_epoch_training_allowed') is True
          and gate.get('test_data_used') is False
          and launch.get('run_scope') == SCOPE
          and launch.get('admission_stage') == STAGE
          and launch.get('numerical_recipe', {}).get('fold_id') == 2
          and launch.get('training_enabled') is True
          and launch.get('test_data_used') is False
          and launch.get('max_model_seconds') == 10500,
          'POLICY_CLOSED')
    require_guarded_cuda_trainer_execution()
    _need(type(deadline_monotonic) is float
          and time.monotonic() < deadline_monotonic, 'MODEL_DEADLINE')
    _need(review_ref == launch['input_refs']['frozen_design']
          and initial_result_ref == launch['input_refs']['initial_result']
          and launch['input_refs']['initial_plan'] == INITIAL_PLAN_REF,
          'FROZEN_RECIPE_INPUTS')
    review = read_ref(review_ref, base=REPO)
    hypothesis = review['fold2_m5_fit_hypothesis']
    require_fixed_design(hypothesis)
    _need(initial_result_ref == hypothesis['observed_basis']['fresh_fold2_initial_result'],
          'INITIAL_RESULT_BINDING')
    initial_result = read(initial_result_ref)
    _need(initial_result['status'] == 'COMPLETE_FRESH_INITIAL'
          and initial_result['optimizer_updates'] == 0
          and initial_result['model_state_sha256'] == INITIAL_MODEL_SHA256
          and initial_result['test_data_used'] is False,
          'FRESH_INITIAL_RECEIPT')
    device = torch.device('cuda', 0)
    context = None
    try:
        context = prepare_qualified_inputs(
            initial_plan_ref=INITIAL_PLAN_REF,
            design=review['fresh_initial_design'], device=device)
        trainer._set_deterministic(20260911, device, trainer.DETERMINISTIC_FP32)
        backbone = build_native_direction_research_model(
            context['view'], prefix_normalization=context['packet'],
            constructor_metadata=context['constructor'],
            device=torch.device('cpu'), unobserved_units_profile=None)
        model = NativeFutureDualModel(backbone).to(device).train()
        _need(model.research_objective['contract_sha256'] == MODEL_CONTRACT_SHA256
              and canonical_model_state_sha256(model.state_dict())
                  == INITIAL_MODEL_SHA256, 'EXACT_FRESH_CONSTRUCTOR')
        optimizer = torch.optim.AdamW(model.parameters(), **ADAMW_DEFAULTS)
        axes = context['axes']
        fit = fit_one_pass(
            model=model, optimizer=optimizer,
            parent_rows=axes['parent_rows'][:TRAIN_ROWS],
            issue_ns=axes['decision_ns'][:TRAIN_ROWS],
            state_bound=context['state_bound'],
            future_bound=context['future_bound'],
            gather_features=lambda left, right, rows, issues:
                context['features']('TRAIN', rows, issues),
            deadline_monotonic=deadline_monotonic,
            progress_state=progress_state)
        final = evaluate_fixed_panels(
            model=model, context=context, deadline_monotonic=deadline_monotonic)
        initial_later = read_capture(initial_result['captures']['LATER'])
        _need(time.monotonic() < deadline_monotonic, 'MODEL_DEADLINE')
        paired = paired_later_f_m5(initial_later, final['captures']['LATER'])
        _need(time.monotonic() < deadline_monotonic, 'MODEL_DEADLINE')
        _need(fit['batch_visits'] == 2959
              and fit['parent_cursor'] == TRAIN_ROWS
              and fit['optimizer_updates'] + fit['skipped_batches'] == 2959
              and final['evaluation_forwards'] == 512
              and final['test_data_used'] is fit['test_data_used'] is paired['test_data_used'] is False,
              'FIXED_WORK_COMPLETE')
        return {'fit': fit, 'final': final, 'paired_later_f_m5': paired,
                'initial_result': initial_result_ref,
                'initial_model_state_sha256': INITIAL_MODEL_SHA256,
                'test_data_used': False}
    finally:
        if context is not None:
            context['release']()


def run_fold2_m5_fit_window(*, policy_path, policy_file_sha256: str,
                            progress_path) -> dict:
    """One admitted native campaign window; failure consumes this plan."""
    started = time.monotonic()
    import hashlib
    import io
    import json
    from pathlib import Path

    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    from gx1.contracts import market_state_ta_fold2_m5_fit_v1 as scope
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_guarded_cuda_trainer_execution,
    )
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    from gx1.audit.score_market_state_ta_fold2_m5_fit_v1 import read_capture
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common

    limits = require_guarded_cuda_trainer_execution()
    window, plan, invocation, marker, recipe = (
        authority.require_native_direction_research_context(
            policy_path, policy_file_sha256, progress_path))
    _need(recipe['run_scope'] == scope.SCOPE
          and recipe['admission_stage'] == scope.STAGE
          and window['max_invocation_seconds'] == 12000
          and recipe['one_window'] == scope.ONE_WINDOW
          and invocation['optimizer_step_budget'] == scope.BATCH_VISITS,
          'EXACT_ADMITTED_WINDOW')
    scope.require_recipe(recipe, source_repo=scope.REPO,
                         source_commit=plan['source_commit'])
    root, output = scope.ROOT, Path(window['output_directory'])
    _need(not any((root / name).exists() for name in (
        'RESULT.json', 'TERMINAL.json', 'PAIRED_READOUT.json',
        'FINAL_TRAIN_READOUT.npz', 'FINAL_LATER_READOUT.npz'))
        and not output.exists() and not Path(window['campaign_cursor_path']).exists()
        and not Path(progress_path).exists(), 'ONE_USE_GENESIS')
    output.mkdir(parents=True)
    counters = {'json_bytes': 0}
    claim = common._json(output / 'CLAIM.json', {
        'schema_version': 'gx1_ta_fold2_m5_fit_claim_v1',
        'recipe': window['recipe'], 'plan_sha256': plan['plan_sha256'],
        'source_commit': plan['source_commit'], 'limits': limits,
        'test_data_used': False}, counters, recipe['max_published_json_bytes'])
    state = {'batch_visits': 0, 'optimizer_updates': 0,
             'skipped_batches': 0, 'parent_cursor': 0,
             'batch_visit_in_flight': False,
             'optimizer_step_in_flight': False}
    captures, paired_ref, inner = {}, None, None
    error, source_unchanged = None, False

    def clock_guard() -> None:
        _need(time.monotonic() - started < window['max_invocation_seconds'],
              'INVOCATION_DEADLINE')

    try:
        inner = _execute_fixed_fit_and_readout(
            launch=recipe, review_ref=recipe['input_refs']['frozen_design'],
            initial_result_ref=recipe['input_refs']['initial_result'],
            deadline_monotonic=float(min(started + window['max_invocation_seconds'],
                                         started + recipe['max_model_seconds'])),
            progress_state=state)
        fit, final = inner['fit'], inner['final']
        _need(all(fit[key] == state[key] for key in
                  ('batch_visits', 'optimizer_updates', 'skipped_batches', 'parent_cursor')),
              'PROGRESS_STATE_PARITY')
        for label in ('TRAIN', 'LATER'):
            clock_guard()
            stream = io.BytesIO()
            np.savez(stream, **final['captures'][label])
            blob = stream.getvalue()
            _need(len(blob) <= 2 * 1024 * 1024, 'FINAL_CAPTURE_BYTES')
            path = root / ('FINAL_' + label + '_READOUT.npz')
            _publish_bytes(path, blob)
            binding = {'path': str(path),
                       'sha256': hashlib.sha256(blob).hexdigest()}
            read_capture(binding)
            captures[label] = binding
        clock_guard()
        paired_ref = common._json(root / 'PAIRED_READOUT.json',
            inner['paired_later_f_m5'], counters, recipe['max_published_json_bytes'])
        campaign.require_clean_source(plan)
        source_unchanged = True
        clock_guard()
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        try:
            campaign.require_clean_source(plan)
            source_unchanged = True
        except BaseException as source_exc:
            error += '; source: ' + type(source_exc).__name__ + ': ' + str(source_exc)

    outcome = 'COMPLETE' if error is None and source_unchanged else 'FAILED'
    visits, updates = state['batch_visits'], state['optimizer_updates']
    parent_cursor = scope.cursor_from_batch_visits(visits)
    _need(state['parent_cursor'] == parent_cursor
          and 0 <= updates <= visits <= scope.BATCH_VISITS,
          'TERMINAL_COUNT_BOUND')
    payload = {
        'schema_version': scope.RESULT_SCHEMA,
        'status': ('PASS_FOLD2_M5_FIT_AND_READOUT' if outcome == 'COMPLETE'
                   else 'FAILED_CONSUMED_NO_RESUME'),
        'outcome': outcome, 'recipe': window['recipe'],
        'recipe_inputs': recipe['input_refs'],
        'initial_result': recipe['input_refs']['initial_result'],
        'claim': claim, 'source_commit': plan['source_commit'],
        'source_unchanged': source_unchanged,
        'batch_visits': visits, 'optimizer_updates': updates,
        'skipped_batches': state['skipped_batches'],
        'parent_cursor': parent_cursor,
        'parent_order_sha256': scope.TRAIN_PARENT_ORDER_SHA256,
        'checkpoint_writes': 0, 'canonical_resume_authority': False,
        'batch_visit_in_flight_at_terminal': state['batch_visit_in_flight'],
        'optimizer_updates_is_lower_bound': state['optimizer_step_in_flight'],
        'final_captures': captures, 'paired_readout': paired_ref,
        'fit': inner['fit'] if outcome == 'COMPLETE' else None,
        'readouts': inner['final']['readouts'] if outcome == 'COMPLETE' else None,
        'measurements': inner['final']['measurements'] if outcome == 'COMPLETE' else None,
        'final_model_state_sha256': (inner['final']['final_model_state_sha256']
                                     if outcome == 'COMPLETE' else None),
        'elapsed_seconds': time.monotonic() - started,
        'error': error, 'test_data_used': False,
        'predictive_edge_proven': False, 'economic_value_measured': False,
    }
    result = common._json(root / 'RESULT.json', payload, counters,
                          recipe['max_published_json_bytes'])
    terminal = common._json(root / 'TERMINAL.json', {
        'status': outcome, 'exit_code': 0 if outcome == 'COMPLETE' else 1,
        'result': result, 'recipe': window['recipe'],
        'source_unchanged': source_unchanged,
        'no_relaunch_under_this_plan': True, 'test_data_used': False,
    }, counters, recipe['max_published_json_bytes'])
    cursor = authority.build_research_cursor(
        recipe_ref=window['recipe'], outcome=outcome,
        optimizer_updates=updates, parent_cursor=parent_cursor,
        parent_order_sha256=scope.TRAIN_PARENT_ORDER_SHA256,
        full_state=None, result=result, batch_size=scope.BATCH_SIZE)
    cursor_ref = common._json(Path(window['campaign_cursor_path']), cursor,
                              counters, recipe['max_published_json_bytes'])
    progress = authority.build_research_progress(
        window=window, plan=plan, invocation=invocation,
        checkpoint_pointer=cursor_ref, optimizer_updates=updates,
        parent_cursor=parent_cursor,
        parent_order_sha256=scope.TRAIN_PARENT_ORDER_SHA256, outcome=outcome)
    common._json(Path(progress_path), progress, counters,
                 recipe['max_published_json_bytes'])
    authority.require_research_progress(
        progress, plan_sha256=plan['plan_sha256'], invocation=invocation,
        expected_selection_receipt_sha256=None, verify_file=True)
    if error is not None:
        raise RuntimeError('FOLD2_M5_FIT_CONSUMED_FAILURE: ' + error)
    return {'result': result, 'terminal': terminal,
            'checkpoint_pointer': cursor_ref, 'progress_path': str(progress_path),
            'predictive_edge_proven': False, 'test_data_used': False}
