"""One guarded direct-cost M5 fit using the already qualified five-TF TA core."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import time

from gx1.audit.market_state_ta_apr_jun_campaign_inputs_v1 import (
    REPO, SCORE_ROWS, TRAIN_ROWS, COLD_FIT_ROWS, COLD_SCORE_START, COLD_SCORE_ROWS,
    COLD_VIEW_SHA256, campaign_input_refs, cold_campaign_input_refs, need, prepare, read, ref,
)
from gx1.audit.run_market_state_ta_apr_jun_paired_fit_v1 import (
    BATCH, BATCHES, SCORE_BATCHES, SEED, SOURCES as PAIRED_SOURCES,
    physical_windows_boot_id, score,
)

STAGE = 'ta_m5_decision_fit_20261117_004'
PLAN = REPO / 'configs/research/TA_M5_DECISION_FIT_20261117_004.json'
ROOT = Path('/home/andre2/GX1_RUNS/TA_M5_DECISION_FIT_20261117_004')
COLD_STAGE = 'ta_m5_cold_joint_fit_20261117_001'
COLD_PLAN = REPO / 'configs/research/TA_M5_COLD_JOINT_FIT_20261117_001.json'
COLD_ROOT = Path('/home/andre2/GX1_RUNS/TA_M5_COLD_JOINT_FIT_20261117_001')
COLD_TARGETS = Path('/home/andre2/GX1_RUNS/TA_M5_COST_PATH_TARGETS_20261117_001/FIT_TARGETS.npz')
COLD_SCORE_BATCHES = (COLD_SCORE_ROWS + BATCH - 1) // BATCH
COLD_FIT_BATCHES = (COLD_FIT_ROWS + BATCH - 1) // BATCH
SOURCES = tuple(dict.fromkeys((*PAIRED_SOURCES,
    'gx1/audit/run_market_state_ta_m5_decision_fit_v1.py',
    'gx1/contracts/market_state_ta_m5_decision_v1.py')))


def publish(name, value, *, root=ROOT):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    raw = value if isinstance(value, bytes) else (
        json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    _publish_bytes(root / name, raw)
    return ref(root / name)


def source_clean(plan):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return (subprocess.check_output(['git', 'branch', '--show-current'],
            cwd=REPO, text=True).strip() == 'work/gx1-current'
        and subprocess.run(['git', 'merge-base', '--is-ancestor',
                            plan['source_base_commit'], head],
                           cwd=REPO, capture_output=True).returncode == 0
        and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
        and set(plan['source_owners']) == set(SOURCES)
        and all(ref(REPO / name)['sha256'] == plan['source_owners'][name]
                for name in SOURCES))


def inspect(plan_sha):
    from gx1.contracts.market_state_ta_m5_decision_v1 import contract, fit_statistics
    import numpy as np
    need(ref(PLAN)['sha256'] == plan_sha and not ROOT.exists(), 'PLAN_OR_OUTPUT')
    plan = read(ref(PLAN))
    need(plan['schema_version'] == 'gx1_ta_m5_decision_fit_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['output_root'] == str(ROOT) and plan['fit_rows'] == TRAIN_ROWS
         and plan['score_rows'] == SCORE_ROWS and plan['batch_size'] == BATCH
         and plan['batch_visits'] == BATCHES and plan['final_score_forwards'] == SCORE_BATCHES
         and plan['max_optimizer_updates'] == BATCHES and plan['max_seconds'] == 10500
         and plan['seed'] == SEED and plan['test_data_used'] is False
         and plan['prior_physical_windows_boot_id'] == 662
         and plan['required_invocation_windows_boot_id_greater_than'] == 662
         and plan['warm_core_result'] == plan['inputs']['prior_fit_result']
         and plan['target_result'] == plan['inputs']['target_result']
         and source_clean(plan), 'FIXED_SCOPE_SOURCE')
    for binding in plan['inputs'].values():
        need(ref(binding['path']) == binding, 'INPUT_REF')
    need(all(plan['inputs'][key] == value for key, value in campaign_input_refs().items()),
         'CAMPAIGN_INPUTS')
    old = read(plan['inputs']['prior_fit_result'])
    old_plan = read(old['plan'])
    old_terminal = read(plan['inputs']['prior_fit_terminal'])
    targets = read(plan['inputs']['target_result'])
    target_terminal = read(plan['inputs']['target_terminal'])
    need(old['status'] == 'COMPLETE_ONE_PASS_PAIRED_S_F_STRUCTURE_APR_JUN_SCORE'
         and old['checkpoint'] == plan['inputs']['warm_core_checkpoint']
         and old['test_data_used'] is False
         and old_terminal['status'] == 'COMPLETE' and old_terminal['exit_code'] == 0
         and old_terminal['result'] == plan['inputs']['prior_fit_result']
         and targets['status'] == 'COMPLETE_FIXED_FOLD2_APR_JUN_COST_PATH_TARGETS'
         and targets['targets']['FIT'] == plan['inputs']['fit_targets']
         and targets['targets']['SCORE'] == plan['inputs']['score_targets']
         and target_terminal['status'] == 'COMPLETE' and target_terminal['exit_code'] == 0
         and target_terminal['result'] == plan['inputs']['target_result']
         and targets['test_data_used'] is False and targets['economic_edge_proven'] is False,
         'REUSED_RESULTS')
    with np.load(plan['inputs']['fit_targets']['path'], allow_pickle=False) as archive:
        stats = fit_statistics(archive)
    need(stats['sha256'] == plan['fit_statistics_sha256']
         and contract(old_plan['model_contract_sha256'], stats['sha256'])['contract_sha256']
             == plan['decision_contract_sha256'], 'FIT_STATISTICS_AND_CONTRACT')
    policy = json.loads((REPO / 'NEXT_RUN_POLICY.json').read_text())
    gate = policy.get(STAGE, {})
    prior_gate = policy['ta_integrated_apr_jun_paired_fit_20261117_002']
    target_gate = policy['ta_m5_cost_path_targets_20261117_001']
    need(prior_gate['remaining_invocations'] == target_gate['remaining_invocations'] == 0
         and prior_gate['execution_authorized'] is target_gate['execution_authorized'] is False
         and target_gate['result'] == plan['inputs']['target_result']
         and policy['full_val_allowed'] is False
         and policy['training_enabled'] is policy['native_training_enabled'] is True
         and policy['full_epoch_training_allowed'] is policy['native_launch_authorized'] is True
         and gate.get('status') == 'ADMITTED_ONE_USE'
         and gate.get('plan') == ref(PLAN)
         and gate.get('remaining_invocations') == 1
         and gate.get('execution_authorized') is gate.get('training_authorized') is True
         and gate.get('relaunch_allowed') is False
         and gate.get('max_optimizer_updates') == BATCHES
         and gate.get('model_forwards') == BATCHES + SCORE_BATCHES
         and gate.get('test_data_used') is False, 'ONE_USE_ADMISSION')
    need(physical_windows_boot_id() > plan['required_invocation_windows_boot_id_greater_than'],
         'FRESH_PHYSICAL_WINDOWS_BOOT')
    return plan, old, stats


def load_targets(binding, expected_rows):
    import numpy as np
    with np.load(binding['path'], allow_pickle=False) as archive:
        value = {key: archive[key] for key in archive.files}
    need(len(value['parent_rows']) == len(value['issue_ns']) == expected_rows,
         'TARGET_ROWS')
    for item in value.values():
        item.setflags(write=False)
    return value


def target_slice(full, start, stop):
    need(type(start) is type(stop) is int and 0 <= start < stop <= len(full['parent_rows']),
         'TARGET_SLICE')
    return {key: value if key == 'horizon_minutes' else value[start:stop]
            for key, value in full.items()}


def decision_readout(predicted, targets):
    import numpy as np
    from gx1.contracts.market_state_ta_m5_decision_v1 import HORIZONS, choose_actions
    n = len(predicted)
    need(predicted.shape == (n, 6, 2, 3) and n > 0
         and np.isfinite(predicted).all(), 'DECISION_CAPTURE')
    action, horizon = choose_actions(predicted)
    chosen = np.flatnonzero(action < 2)
    h = horizon[chosen].astype(np.int64)
    side = action[chosen].astype(np.int64)
    observed = targets['terminal_observed'][chosen, h]
    complete = targets['path_observed'][chosen, h]
    actual = targets['net_bps'][chosen[observed], h[observed], side[observed]]
    def selected(rows):
        pick = rows[action[rows] < 2]
        hh = horizon[pick].astype(np.int64)
        ss = action[pick].astype(np.int64)
        known = targets['terminal_observed'][pick, hh]
        net = targets['net_bps'][pick[known], hh[known], ss[known]]
        return {'issues': int(len(rows)), 'LONG': int((action[rows] == 0).sum()),
            'SHORT': int((action[rows] == 1).sum()),
            'FLAT': int((action[rows] == 2).sum()),
            'selected_terminal_observed': int(known.sum()),
            'selected_terminal_unknown': int((~known).sum()),
            'selected_net_bps_mean': float(net.mean()) if len(net) else None,
            'net_bps_mean_including_known_FLAT':
                float(net.sum() / (len(rows) - int((~known).sum())))
                if len(rows) > int((~known).sum()) else None}
    periods = targets['issue_ns'].astype('datetime64[ns]').astype('datetime64[M]')
    months = {str(month): selected(np.flatnonzero(periods == month))
              for month in np.unique(periods)}
    panels = {}
    for j, minute in enumerate(HORIZONS):
        terminal = targets['terminal_observed'][:, j]
        path = targets['path_observed'][:, j]
        net = targets['net_bps'][:, j]
        mae = targets['pre_mfe_mae_upper_bps'][:, j]
        panel = {'terminal_observed': int(terminal.sum()),
                 'path_observed': int(path.sum()),
                 'selected': int((horizon == j).sum()), 'sides': {}}
        for s, name in enumerate(('LONG', 'SHORT')):
            panel['sides'][name] = {
                'always_side_net_bps_on_path': float(net[path, s].mean()) if path.any() else None,
                'predicted_net_mae_bps_on_terminal':
                    float(np.abs(predicted[terminal, j, s, 0] - net[terminal, s]).mean())
                    if terminal.any() else None,
                'predicted_pre_mfe_mae_mae_bps_on_path':
                    float(np.abs(predicted[path, j, s, 2] - mae[path, s]).mean())
                    if path.any() else None}
        panels[str(minute)] = panel
    return {'entry_timeframe': 'M5', 'query_horizon_minutes': list(HORIZONS),
        'exit_evaluation': False, 'role': 'overlapping_markout_not_trading_PnL',
        'issues': n, 'actions': selected(np.arange(n)),
        'selected_path_observed': int(complete.sum()),
        'selected_path_unknown': int((~complete).sum()),
        'selected_horizon_counts': {str(minute): int((horizon == j).sum())
                                   for j, minute in enumerate(HORIZONS)},
        'selected_positive_net_on_terminal': int((actual > 0).sum()),
        'months': months, 'panels': panels,
        'economic_edge_proven': False}


def _fit_one_pass(model, context, fit, *, device, fit_rows, fold_id,
                  contract_sha256, deadline):
    import numpy as np
    import torch
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets, sampled_query_k
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
    from gx1.contracts.market_state_ta_m5_decision_v1 import decision_loss
    from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import ADAMW_DEFAULTS
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    batches = (fit_rows + BATCH - 1) // BATCH
    axes = context['axes']
    rows, issues = axes['parent_rows'][:fit_rows], axes['decision_ns'][:fit_rows]
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), **ADAMW_DEFAULTS)
    need(not optimizer.state and len(optimizer.param_groups) == 1, 'EMPTY_OPTIMIZER')
    optimizer.zero_grad(set_to_none=True)
    state_bound = load_bound_five_endpoint_targets()
    future_bound = load_bound_future_displacement()
    sampled_bound = load_bound_time_query_targets()
    tape = context['fit_tape']
    blocks, updates, skipped = [], 0, 0
    current = {'first_batch': 0, 'loss': 0., 'decision': 0., 'net': 0.,
               'mfe': 0., 'mae': 0., 'quality': 0., 'side_pair': 0.,
               'mae_under': 0., 'aux': 0.,
               'grad_norm': 0., 'updates': 0, 'skipped': 0}
    for index in range(batches):
        need(time.monotonic() < deadline, 'FIT_DEADLINE')
        left, right = index * BATCH, min((index + 1) * BATCH, fit_rows)
        r, i = rows[left:right], issues[left:right]
        features = context['features']('TRAIN', r, i)
        sampled = sampled_query_k(i, pass_index=0, seed=SEED)
        native = np.broadcast_to(np.asarray(NATIVE_K, dtype=np.int64), sampled.shape)
        grid = torch.from_numpy(np.stack((sampled, native), axis=1).copy()).to(device)
        output = model(features, query_k=grid,
            m1_path_bps=torch.from_numpy(tape['m1_mid_ohlc_bps'][left:right].copy()).to(device),
            m1_observed=torch.from_numpy(tape['m1_observed'][left:right].copy()).to(device))
        target = {key: fit[key][left:right] for key in
            ('terminal_observed', 'path_observed', 'net_bps', 'mfe_bps',
             'pre_mfe_mae_upper_bps')}
        loss = decision_loss(model_output=output, targets=target, model=model,
            sampled_truth=sampled_bound.gather_fit_sampled(r, i, fold_id=fold_id,
                pass_index=0, seed=SEED),
            native_truth=state_bound.gather(r, i, fold_id=fold_id, partition='fit'),
            future_truth=future_bound.gather(r, i, fold_id=fold_id, partition='fit'),
            query_grid=grid, structure_hit=tape['structure_hit'][left:right],
            structure_observed=tape['structure_observed'][left:right])
        need(loss['contract_sha256'] == contract_sha256
             and bool(torch.isfinite(loss['loss'])), 'FIT_LOSS')
        for key, field in (('loss', 'loss'), ('decision', 'decision_loss'),
                           ('net', 'net_loss'), ('mfe', 'mfe_loss'),
                           ('mae', 'mae_loss'), ('quality', 'quality_loss'),
                           ('side_pair', 'side_pair_loss'),
                           ('mae_under', 'mae_under_loss'), ('aux', 'aux_loss')):
            current[key] += float(loss[field].detach())
        if loss['skip_update']:
            skipped += 1
            current['skipped'] += 1
        else:
            loss['loss'].backward()
            norm = float(trainer._optimizer_step_with_finite_gradients(
                model=model, optimizer=optimizer))
            need(math.isfinite(norm), 'GRADIENT_NORM')
            current['grad_norm'] += norm
            updates += 1
            current['updates'] += 1
        optimizer.zero_grad(set_to_none=True)
        del features, output, loss
        if (index + 1) % 256 == 0 or index + 1 == batches:
            visits = index + 1 - current['first_batch']
            blocks.append({'first_batch': current['first_batch'],
                'last_batch_exclusive': index + 1,
                **{'mean_' + key: current[key] / visits for key in
                   ('loss', 'decision', 'net', 'mfe', 'mae', 'quality',
                    'side_pair', 'mae_under', 'aux')},
                'mean_gradient_norm_per_update':
                    current['grad_norm'] / current['updates'] if current['updates'] else None,
                'updates': current['updates'], 'skipped': current['skipped']})
            current = {'first_batch': index + 1, 'loss': 0., 'decision': 0.,
                       'net': 0., 'mfe': 0., 'mae': 0., 'quality': 0.,
                       'side_pair': 0., 'mae_under': 0., 'aux': 0.,
                       'grad_norm': 0., 'updates': 0, 'skipped': 0}
    need(updates + skipped == batches, 'FIT_VISITS')
    return blocks, updates, skipped

def run(plan_sha):
    import numpy as np
    import torch
    from gx1.contracts.entry_native_direction_feature_view_v1 import build_native_direction_research_model
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_ta_integrated_model_v1 import IntegratedTAMarketStateModel
    from gx1.contracts.market_state_ta_m5_decision_v1 import IntegratedTAM5DecisionModel, decision_loss
    from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets, sampled_query_k
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import ADAMW_DEFAULTS

    guard = require_guarded_cuda_trainer_execution()
    plan, old, stats = inspect(plan_sha)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    ROOT.mkdir()
    claim = publish('CLAIM.json', {'plan': ref(PLAN), 'source_commit': head,
        'physical_windows_boot_id': physical_windows_boot_id(),
        'guard': guard, 'test_data_used': False})
    started = time.monotonic()
    deadline = started + plan['max_seconds'] - 60
    context, result = None, None
    try:
        device = torch.device('cuda', 0)
        context = prepare(device=device)
        fit = load_targets(plan['inputs']['fit_targets'], TRAIN_ROWS)
        later = load_targets(plan['inputs']['score_targets'], SCORE_ROWS)
        axes = context['axes']
        rows = axes['parent_rows'][:TRAIN_ROWS]
        issues = axes['decision_ns'][:TRAIN_ROWS]
        selected = context['selected']
        need(np.array_equal(rows, fit['parent_rows'])
             and np.array_equal(issues, fit['issue_ns'])
             and np.array_equal(axes['parent_rows'][selected], later['parent_rows'])
             and np.array_equal(axes['decision_ns'][selected], later['issue_ns'])
             and context['view']['contract_sha256'] == plan['view_sha256'],
             'TARGET_FEATURE_AXES')
        trainer._set_deterministic(SEED, device, trainer.DETERMINISTIC_FP32)
        state = build_native_direction_research_model(context['view'],
            prefix_normalization=context['packet'], constructor_metadata=context['constructor'],
            device=torch.device('cpu'), unobserved_units_profile=None)
        core = IntegratedTAMarketStateModel(NativeFutureDualModel(state))
        core.load_state_dict(torch.load(plan['inputs']['warm_core_checkpoint']['path'],
            map_location='cpu', weights_only=True), strict=True)
        need(canonical_model_state_sha256(core.state_dict()) == old['final_model_state_sha256'],
             'WARM_CORE_STATE')
        model = IntegratedTAM5DecisionModel(core, stats).to(device)
        initial_sha = canonical_model_state_sha256(model.state_dict())
        need(initial_sha == plan['initial_model_state_sha256']
             and model.research_objective['contract_sha256'] == plan['decision_contract_sha256'],
             'INITIAL_STATE_AND_CONTRACT')
        initial = decision_readout(np.broadcast_to(stats['mean_bps'],
            (SCORE_ROWS, 6, 2, 3)), later)
        blocks, updates, skipped = _fit_one_pass(model, context, fit, device=device,
            fit_rows=TRAIN_ROWS, fold_id=2,
            contract_sha256=plan['decision_contract_sha256'], deadline=deadline)
        final_core, capture = score(model, context, deadline=deadline,
                                    fold_id=3, partition='score', collect_decision=True)
        final = decision_readout(capture['m5_cost_path_bps'], later)
        final_sha = canonical_model_state_sha256(model.state_dict())
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint_ref = publish('FINAL_MODEL.pt', checkpoint.getvalue())
        stream = io.BytesIO()
        np.savez_compressed(stream, **capture)
        capture_ref = publish('FINAL_SCORE_READOUT.npz', stream.getvalue())
        need(source_clean(plan)
             and subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                         cwd=REPO, text=True).strip() == head
             and time.monotonic() < deadline, 'SOURCE_TIME_BOOKEND')
        context['release']()
        context = None
        result = publish('RESULT.json', {
            'status': 'COMPLETE_ONE_PASS_COSTED_M5_DECISION_APR_JUN_SCORE',
            'plan': ref(PLAN), 'claim': claim, 'source_commit': head,
            'source_unchanged': True, 'guard': guard,
            'warm_core_checkpoint': plan['inputs']['warm_core_checkpoint'],
            'fit_statistics_sha256': stats['sha256'],
            'initial_model_state_sha256': initial_sha,
            'final_model_state_sha256': final_sha,
            'fit_rows': TRAIN_ROWS, 'batch_visits': BATCHES,
            'optimizer_updates': updates, 'skipped_batches': skipped,
            'training_forwards': BATCHES, 'evaluation_forwards': SCORE_BATCHES,
            'fit_blocks': blocks, 'initial_decision': initial,
            'initial_core_readout': plan['inputs']['prior_fit_result'],
            'final_core': final_core, 'final_decision': final,
            'checkpoint': checkpoint_ref, 'final_capture': capture_ref,
            'test_data_used': False, 'economic_edge_proven': False,
            'elapsed_seconds': time.monotonic() - started})
        publish('TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'plan': ref(PLAN), 'result': result, 'source_unchanged': True,
            'relaunch_allowed': False, 'remaining_invocations': 0,
            'test_data_used': False})
    except BaseException as error:
        publish('TERMINAL.json', {'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'plan': ref(PLAN), 'result': result,
            'error': type(error).__name__ + ':' + str(error)[:2000],
            'relaunch_allowed': False, 'remaining_invocations': 0,
            'test_data_used': False})
        raise
    finally:
        if context is not None:
            context['release']()



def inspect_cold(plan_sha, *, admitted):
    """Bind a new cold plan to qualified full later coverage and Fold1-only loss."""
    import numpy as np
    from gx1.contracts.market_state_ta_m5_decision_v1 import contract, fit_statistics
    need(ref(COLD_PLAN)['sha256'] == plan_sha and not COLD_ROOT.exists(), 'COLD_PLAN_OUTPUT')
    plan = read(ref(COLD_PLAN))
    expected_inputs = {**cold_campaign_input_refs(),
        'fit_targets': ref(COLD_TARGETS),
        'target_result': ref(COLD_TARGETS.parent / 'RESULT.json'),
        'target_terminal': ref(COLD_TARGETS.parent / 'TERMINAL.json')}
    need(plan['schema_version'] == 'gx1_ta_m5_cold_joint_fit_plan_v1'
         and plan['stage'] == COLD_STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['output_root'] == str(COLD_ROOT)
         and plan['fit_rows'] == COLD_FIT_ROWS
         and plan['score_parent_start'] == COLD_SCORE_START
         and plan['score_rows'] == COLD_SCORE_ROWS
         and plan['excluded_gap_rows'] == COLD_SCORE_START - COLD_FIT_ROWS
         and plan['batch_size'] == BATCH and plan['batch_visits'] == COLD_FIT_BATCHES
         and plan['initial_score_forwards'] == COLD_SCORE_BATCHES
         and plan['final_score_forwards'] == COLD_SCORE_BATCHES
         and plan['max_optimizer_updates'] == COLD_FIT_BATCHES
         and plan['max_seconds'] == 10500 and plan['seed'] == SEED
         and plan['fresh_initial'] is True and plan['test_data_used'] is False
         and plan['inputs'] == expected_inputs
         and plan['view_sha256'] == COLD_VIEW_SHA256
         and plan['prior_physical_windows_boot_id'] >= 663
         and plan['required_invocation_windows_boot_id_greater_than']
             == plan['prior_physical_windows_boot_id']
         and source_clean(plan), 'COLD_FIXED_SCOPE_SOURCE')
    result = read(expected_inputs['coverage_result'])
    coverage = read(expected_inputs['coverage'])
    terminal = read(expected_inputs['coverage_terminal'])
    target_result = read(expected_inputs['target_result'])
    target_terminal = read(expected_inputs['target_terminal'])
    need(result['status'] == coverage['status'] == terminal['status'] == 'COMPLETE'
         and result['view'] == expected_inputs['model_view']
         and coverage['qualification'] == expected_inputs['coverage_result']
         and terminal['result'] == expected_inputs['coverage_result']
         and terminal['coverage'] == expected_inputs['coverage']
         and terminal['exit_code'] == 0
         and coverage['schema_version'] == 'entry_anchor_span_cold_full_later_coverage_v1'
         and coverage['view_sha256'] == COLD_VIEW_SHA256
         and coverage['normalization_prefix']['rows'] == COLD_FIT_ROWS
         and coverage['score_parent_start'] == COLD_SCORE_START
         and coverage['score_rows'] == COLD_SCORE_ROWS
         and coverage['tail_rows'] == 1
         and target_result['targets']['FIT'] == expected_inputs['fit_targets']
         and target_result['test_data_used'] is False
         and target_terminal['status'] == 'COMPLETE'
         and target_terminal['exit_code'] == 0
         and target_terminal['result'] == expected_inputs['target_result'],
         'COLD_REUSED_RESULTS')
    full = load_targets(expected_inputs['fit_targets'], TRAIN_ROWS)
    fit = target_slice(full, 0, COLD_FIT_ROWS)
    later = target_slice(full, COLD_SCORE_START, TRAIN_ROWS)
    stats = fit_statistics(fit)
    need(stats['sha256'] == plan['fit_statistics_sha256']
         and contract(plan['core_contract_sha256'], stats['sha256'],
                      aux_fold_id=1)['contract_sha256']
             == plan['decision_contract_sha256']
         and len(later['parent_rows']) == COLD_SCORE_ROWS, 'COLD_FIT_STATISTICS')
    policy = json.loads((REPO / 'NEXT_RUN_POLICY.json').read_text())
    gate = policy.get(COLD_STAGE, {})
    closed = (policy['full_val_allowed'] is False
              and policy['exit_training_and_evaluation_this_wave'] is False
              and gate.get('test_data_used') is False)
    if admitted:
        need(closed and policy['training_enabled'] is policy['native_training_enabled'] is True
             and policy['full_epoch_training_allowed'] is policy['native_launch_authorized'] is True
             and gate.get('status') == 'ADMITTED_ONE_USE'
             and gate.get('plan') == ref(COLD_PLAN)
             and gate.get('remaining_invocations') == 1
             and gate.get('execution_authorized') is gate.get('training_authorized') is True
             and gate.get('relaunch_allowed') is False
             and gate.get('max_optimizer_updates') == COLD_FIT_BATCHES
             and gate.get('model_forwards') == COLD_FIT_BATCHES + 2 * COLD_SCORE_BATCHES
             and physical_windows_boot_id() >
                 plan['required_invocation_windows_boot_id_greater_than'], 'COLD_ONE_USE_ADMISSION')
    else:
        need(closed and policy['training_enabled'] is policy['native_training_enabled'] is False
             and gate.get('execution_authorized') is not True, 'COLD_TRAINING_CLOSED')
    return plan, stats


def run_cold(plan_sha):
    """One cold FIT-only pass and two later readouts; no warm checkpoint."""
    import numpy as np
    import torch
    from gx1.contracts.entry_native_direction_feature_view_v1 import build_native_direction_research_model
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    from gx1.contracts.market_state_ta_integrated_model_v1 import IntegratedTAMarketStateModel
    from gx1.contracts.market_state_ta_m5_decision_v1 import IntegratedTAM5DecisionModel
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    guard = require_guarded_cuda_trainer_execution()
    plan, stats = inspect_cold(plan_sha, admitted=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    COLD_ROOT.mkdir()
    claim = publish('CLAIM.json', {'plan': ref(COLD_PLAN), 'source_commit': head,
        'physical_windows_boot_id': physical_windows_boot_id(), 'guard': guard,
        'test_data_used': False}, root=COLD_ROOT)
    started = time.monotonic()
    deadline = started + plan['max_seconds'] - 60
    context, result = None, None
    try:
        device = torch.device('cuda', 0)
        context = prepare(device=device, cold_inputs={key: plan['inputs'][key]
            for key in cold_campaign_input_refs()})
        full = load_targets(plan['inputs']['fit_targets'], TRAIN_ROWS)
        fit = target_slice(full, 0, COLD_FIT_ROWS)
        later = target_slice(full, COLD_SCORE_START, TRAIN_ROWS)
        axes, selected = context['axes'], context['selected']
        need(context['fit_rows'] == COLD_FIT_ROWS
             and context['score_rows'] == COLD_SCORE_ROWS
             and context['fit_fold_id'] == 1
             and context['score_fold_id'] == 1
             and context['score_partition'] == 'score'
             and np.array_equal(axes['parent_rows'][:COLD_FIT_ROWS], fit['parent_rows'])
             and np.array_equal(axes['decision_ns'][:COLD_FIT_ROWS], fit['issue_ns'])
             and np.array_equal(axes['parent_rows'][selected], later['parent_rows'])
             and np.array_equal(axes['decision_ns'][selected], later['issue_ns'])
             and context['view']['contract_sha256'] == plan['view_sha256'],
             'COLD_TARGET_FEATURE_AXES')
        trainer._set_deterministic(SEED, device, trainer.DETERMINISTIC_FP32)
        state = build_native_direction_research_model(context['view'],
            prefix_normalization=context['packet'], constructor_metadata=context['constructor'],
            device=torch.device('cpu'), unobserved_units_profile=context['units_profile'])
        core = IntegratedTAMarketStateModel(NativeFutureDualModel(state))
        need(core.research_objective['contract_sha256'] == plan['core_contract_sha256'],
             'COLD_CORE_CONTRACT')
        model = IntegratedTAM5DecisionModel(core, stats, aux_fold_id=1).to(device)
        initial_sha = canonical_model_state_sha256(model.state_dict())
        need(initial_sha == plan['initial_model_state_sha256']
             and model.research_objective['contract_sha256'] == plan['decision_contract_sha256'],
             'COLD_INITIAL_STATE_CONTRACT')
        initial_core, initial_capture = score(model, context, deadline=deadline,
            fold_id=1, partition='score', collect_decision=True)
        initial = decision_readout(initial_capture['m5_cost_path_bps'], later)
        del initial_capture
        blocks, updates, skipped = _fit_one_pass(model, context, fit, device=device,
            fit_rows=COLD_FIT_ROWS, fold_id=1,
            contract_sha256=plan['decision_contract_sha256'], deadline=deadline)
        final_core, capture = score(model, context, deadline=deadline,
            fold_id=1, partition='score', collect_decision=True)
        final = decision_readout(capture['m5_cost_path_bps'], later)
        final_sha = canonical_model_state_sha256(model.state_dict())
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint_ref = publish('FINAL_MODEL.pt', checkpoint.getvalue(), root=COLD_ROOT)
        stream = io.BytesIO()
        np.savez_compressed(stream, **capture)
        capture_ref = publish('FINAL_SCORE_READOUT.npz', stream.getvalue(), root=COLD_ROOT)
        need(source_clean(plan) and subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=REPO, text=True).strip() == head and time.monotonic() < deadline,
            'COLD_SOURCE_TIME_BOOKEND')
        context['release']()
        context = None
        result = publish('RESULT.json', {
            'status': 'COMPLETE_ONE_PASS_COLD_FOLD1_M5_DECISION_LATER_SCORE',
            'plan': ref(COLD_PLAN), 'claim': claim, 'source_commit': head,
            'source_unchanged': True, 'guard': guard, 'fresh_initial': True,
            'fold1_aux_contract_sha256': model.research_objective['cold_aux_contract_sha256'],
            'fit_statistics_sha256': stats['sha256'],
            'initial_model_state_sha256': initial_sha,
            'final_model_state_sha256': final_sha,
            'fit_rows': COLD_FIT_ROWS, 'excluded_gap_rows': COLD_SCORE_START - COLD_FIT_ROWS,
            'score_rows': COLD_SCORE_ROWS, 'batch_visits': COLD_FIT_BATCHES,
            'optimizer_updates': updates, 'skipped_batches': skipped,
            'training_forwards': COLD_FIT_BATCHES,
            'evaluation_forwards': 2 * COLD_SCORE_BATCHES,
            'fit_blocks': blocks, 'initial_core': initial_core, 'initial_decision': initial,
            'final_core': final_core, 'final_decision': final,
            'checkpoint': checkpoint_ref, 'final_capture': capture_ref,
            'test_data_used': False, 'economic_edge_proven': False,
            'elapsed_seconds': time.monotonic() - started}, root=COLD_ROOT)
        publish('TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'plan': ref(COLD_PLAN), 'result': result, 'source_unchanged': True,
            'relaunch_allowed': False, 'remaining_invocations': 0,
            'test_data_used': False}, root=COLD_ROOT)
    except BaseException as error:
        publish('TERMINAL.json', {'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'plan': ref(COLD_PLAN), 'result': result,
            'error': type(error).__name__ + ':' + str(error)[:2000],
            'relaunch_allowed': False, 'remaining_invocations': 0,
            'test_data_used': False}, root=COLD_ROOT)
        raise
    finally:
        if context is not None:
            context['release']()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--device', choices=('cuda',), required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    if args.plan == str(PLAN) and args.out_dir == str(ROOT):
        run(args.plan_sha256)
    elif args.plan == str(COLD_PLAN) and args.out_dir == str(COLD_ROOT):
        run_cold(args.plan_sha256)
    else:
        need(False, 'CLI_SCOPE')


if __name__ == '__main__':
    main()
