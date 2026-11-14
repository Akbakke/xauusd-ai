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
    batch_bounds,
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
                 gather_features: Callable, deadline_monotonic: float) -> dict:
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
    for index in range(BATCH_VISITS):
        _need(time.monotonic() < deadline_monotonic, 'TIME_BUDGET')
        left, right = batch_bounds(index)
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
            norm = float(trainer._optimizer_step_with_finite_gradients(
                model=model, optimizer=optimizer))
            _need(math.isfinite(norm), 'GRADIENT_NORM')
            record['gradient_norm'] = norm
            updates += 1
        optimizer.zero_grad(set_to_none=True)
        records.append(record)
        del features, report
    _need(len(records) == BATCH_VISITS and records[-1]['stop'] == TRAIN_ROWS
          and updates + skipped == BATCH_VISITS
          and all(parameter.grad is None for parameter in model.parameters()),
          'COMPLETE_ONE_PASS')
    return {'batch_visits': BATCH_VISITS, 'optimizer_updates': updates,
            'skipped_batches': skipped, 'training_records': records,
            'parent_cursor': TRAIN_ROWS, 'checkpoint_writes': 0,
            'test_data_used': False}


INITIAL_PLAN_REF = {
    'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_FRESH_INITIAL_20261114_004/PLAN.json',
    'sha256': '6fe63d5e06812122cf5a7c9a7e816fa547c718fd65039fac2588b2988542390a',
}


def require_reused_input_refs(initial_plan_ref: dict) -> dict:
    """Bind completed input receipts; the evolving design is separately frozen."""
    from gx1.audit.observe_market_state_ta_fold2_fresh_initial_v1 import read
    _need(initial_plan_ref == INITIAL_PLAN_REF, 'INITIAL_PLAN_BINDING')
    plan = read(initial_plan_ref)
    inputs = plan['inputs']
    _need(plan['stage'] == 'market_state_ta_fold2_fresh_initial_20261114_004'
          and plan['source_repo'] == '/home/andre2/src/GX1_CURRENT'
          and plan['train_rows'] == plan['later_rows'] == 4096
          and plan['forwards'] == 512 and plan['optimizer_updates'] == 0
          and plan['checkpoint_loads'] == 0 and plan['test_data_used'] is False,
          'INITIAL_PLAN_STAGE')
    expected = {'design', 'train_plan', 'later_plan', 'view', 'normalization',
                'constructor', 'priors', 'packet', 'provider_plan',
                'physical_result', 'train_gather_result', 'later_result',
                'later_terminal', 'later_gather_result', 'later_coverage',
                'train_validation', 'later_validation', 'target_validation'}
    _need(set(inputs) == expected, 'INITIAL_INPUT_SET')
    for name, binding in inputs.items():
        if name != 'design':
            read(binding)
    for name, status in (('train_validation', 'PASS_FULL_FOLD2_TRAIN_INPUT_GATHER'),
                         ('later_validation', 'PASS_JSON_HASH_JOIN_AND_EXACT_COUNTS'),
                         ('target_validation', 'PASS_JSON_PREFIX_MASK_BATCH_AND_PRIOR_JOINS')):
        receipt = read(inputs[name])
        _need(receipt['status'] == status and receipt['test_data_used'] is False,
              'VALIDATION_' + name.upper())
    return inputs


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
