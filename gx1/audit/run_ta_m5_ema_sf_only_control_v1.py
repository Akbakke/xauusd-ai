"""One guarded S/F-only control on the exact consumed EMA Fold1 view.

This is a research contrast with the decision-primary EMA fit, not an Entry
model or trading result. It fits once and scores fixed FIT and later axes.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import subprocess
import time

from gx1.audit import run_market_state_ta_m5_decision_fit_v1 as decision
from gx1.audit import run_market_state_ta_apr_jun_paired_fit_v1 as paired
from gx1.audit import score_ta_ema_final_fit_v1 as frozen_fit
from gx1.audit.market_state_ta_ema_delta_campaign_inputs_v1 import (
    FIT_ROWS, LATER_ROWS, LATER_START, LATER_STOP, REPO, _receipts,
    prepare as prepare_ema, successor_input_refs,
)
from gx1.audit.market_state_ta_apr_jun_campaign_inputs_v1 import need, read, ref
from gx1.contracts.market_state_ta_integrated_paired_objective_v1 import (
    contract as sf_contract,
)

STAGE = 'ta_m5_ema_sf_only_control_20261118_001'
PLAN = REPO / 'configs/research/TA_M5_EMA_SF_ONLY_CONTROL_20261118_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/TA_M5_EMA_SF_ONLY_CONTROL_20261118_001')
BATCH, SEED = paired.BATCH, paired.SEED
FIT_BATCHES = (FIT_ROWS + BATCH - 1) // BATCH
LATER_BATCHES = (LATER_ROWS + BATCH - 1) // BATCH
MAX_SECONDS = 10500
PRIOR_BOOT = 665
ORIGINAL_CORE_INITIAL_SHA = 'b73dd0c23c0e62fb20bcb79bbb7a87dd84794580ebdcea814559fab508f63826'
SOURCES = tuple(dict.fromkeys((*decision.EMA_SOURCES,
    'gx1/audit/score_ta_ema_final_fit_v1.py',
    'gx1/audit/run_ta_m5_ema_sf_only_control_v1.py',
    'tests/test_ta_m5_ema_sf_only_control_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/model_state_digest_v1.py')))


def build_plan(*, source_base_commit=None):
    """Bind one controlled fit to existing complete EMA inputs and baselines."""
    inputs = successor_input_refs()
    _receipts(inputs)
    consumed, old = frozen_fit.consumed_refs()
    fit_score = ref(frozen_fit.ROOT / 'RESULT.json')
    fit_score_result = read(fit_score)
    need(old['initial_model_state_sha256'] == decision.EMA_INITIAL_SHA
         and old['fit_statistics_sha256'] == decision.EMA_FIT_STATS_SHA
         and fit_score_result['status'] == 'COMPLETE_FROZEN_EMA_FOLD1_FIT_SCORE_NO_UPDATE'
         and fit_score_result['consumed_ema'] == consumed
         and fit_score_result['fit_rows'] == FIT_ROWS
         and fit_score_result['model_forwards'] == FIT_BATCHES
         and fit_score_result['model_state_sha256'] == old['final_model_state_sha256']
         and fit_score_result['test_data_used'] is False,
         'BASELINE_CHAIN')
    head = source_base_commit or subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return {
        'schema_version': 'gx1_ta_m5_ema_sf_only_control_plan_v1',
        'stage': STAGE, 'status': 'PREPARED_NOT_ADMITTED',
        'source_base_commit': head,
        'source_owners': {name: ref(REPO / name)['sha256'] for name in SOURCES},
        'inputs': inputs, 'consumed_ema': consumed, 'decision_fit_score': fit_score,
        'output_root': str(ROOT), 'fit_rows': FIT_ROWS, 'later_rows': LATER_ROWS,
        'excluded_gap_rows': LATER_START - FIT_ROWS,
        'batch_size': BATCH, 'batch_visits': FIT_BATCHES,
        'fit_score_forwards': FIT_BATCHES,
        'later_score_forwards': LATER_BATCHES,
        'model_forwards': FIT_BATCHES * 2 + LATER_BATCHES,
        'max_optimizer_updates': FIT_BATCHES,
        'max_seconds': MAX_SECONDS, 'seed': SEED,
        'view_sha256': decision.EMA_VIEW_SHA,
        'core_contract_sha256': decision.EMA_CORE_SHA,
        'sf_objective_sha256': sf_contract(fold_id=1)['contract_sha256'],
        'original_initial_wrapper_sha256': decision.EMA_INITIAL_SHA,
        'original_initial_core_sha256': ORIGINAL_CORE_INITIAL_SHA,
        'fit_statistics_sha256': decision.EMA_FIT_STATS_SHA,
        'required_invocation_windows_boot_id_greater_than': PRIOR_BOOT,
        'capped_execution': {'class': 'trainer', 'memory': '20G', 'swap': '512M'},
        'primary_diagnostic': (
            'full_FIT_native_F_M5_known_side_Brier_and_log_vs_fixed_FIT_prior;'
            'same_later;FIT_failure_means_no_SF_learning_without_decision;'
            'FIT_gain_later_failure_means_transfer_or_regime_failure'),
        'secondary_diagnostics': (
            'all_native_S_F_x_five_TF_proper_scores_and_F_H4_D1_month_direction;'
            'fixed_F_H4_logit_sign_side_on_complete_h480_vs_always_LONG'),
        'fixed_proxy_uncertainty': {
            'comparison': 'F_H4_sign_minus_always_LONG_quality_bps',
            'population': 'paired_complete_h480_paths',
            'blocks': 'ISO_week_of_issue_UTC',
            'bootstrap_draws': 20000, 'seed': 20261118,
            'interval': 'percentile_95_percent'},
        'economic_edge_proven': False, 'test_data_used': False,
        'training_authorized': False, 'test_authorized': False,
        'online_authorized': False, 'broker_access_allowed': False,
    }


def source_clean(plan):
    return decision.source_clean(plan, SOURCES)


def inspect(plan_sha, *, admitted):
    need(ref(PLAN)['sha256'] == plan_sha and not ROOT.exists(), 'PLAN_OUTPUT')
    plan = read(ref(PLAN))
    need(plan == build_plan(source_base_commit=plan.get('source_base_commit'))
         and Path.cwd().resolve() == REPO and source_clean(plan),
         'FIXED_SCOPE_SOURCE')
    policy_ref = ref(REPO / 'NEXT_RUN_POLICY.json')
    policy = read(policy_ref)
    gate = policy.get(STAGE, {})
    prior = policy.get(decision.EMA_STAGE, {})
    prior_score = policy.get(frozen_fit.STAGE, {})
    need(prior.get('status') == 'REVIEWED_CONSUMED_EMA_SIDE_COLLAPSE_EDGE_UNPROVEN'
         and prior.get('result') == plan['consumed_ema']['result']
         and prior.get('remaining_invocations') == 0
         and prior_score.get('status') == 'REVIEWED_CONSUMED_FIT_SF_WEAK_EDGE_UNPROVEN'
         and prior_score.get('result') == plan['decision_fit_score']
         and prior_score.get('remaining_invocations') == 0
         and policy.get('full_val_allowed') is False
         and policy.get('exit_training_and_evaluation_this_wave') is False
         and gate.get('plan') == ref(PLAN)
         and gate.get('output_root') == str(ROOT)
         and gate.get('fit_rows') == FIT_ROWS
         and gate.get('later_rows') == LATER_ROWS
         and gate.get('max_optimizer_updates') == FIT_BATCHES
         and gate.get('model_forwards') == plan['model_forwards']
         and gate.get('test_authorized') is False
         and gate.get('online_authorized') is False
         and gate.get('broker_access_allowed') is False
         and gate.get('test_data_used') is False
         and gate.get('relaunch_allowed') is False,
         'CLOSED_POLICY')
    if admitted:
        need(policy.get('training_enabled') is policy.get('native_training_enabled') is True
             and policy.get('full_epoch_training_allowed') is policy.get('native_launch_authorized') is True
             and gate.get('status') == 'ADMITTED_ONE_USE'
             and gate.get('execution_authorized') is gate.get('training_authorized') is True
             and gate.get('remaining_invocations') == gate.get('max_invocations') == 1
             and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
             and paired.physical_windows_boot_id() > PRIOR_BOOT,
             'ONE_USE_ADMISSION')
    else:
        need(policy.get('training_enabled') is policy.get('native_training_enabled') is False
             and policy.get('full_epoch_training_allowed') is policy.get('native_launch_authorized') is False
             and gate.get('status') == 'PREPARED_NOT_ADMITTED'
             and gate.get('execution_authorized') is gate.get('training_authorized') is False
             and gate.get('remaining_invocations') == 0
             and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0,
             'NOT_ADMITTED')
    return plan, policy_ref


def initial_core(context, fit, *, device):
    """Prove the exact consumed decision start, then keep only its core."""
    import torch
    from gx1.contracts.entry_native_direction_feature_view_v1 import build_native_direction_research_model
    from gx1.contracts.market_state_ta_integrated_model_v1 import IntegratedTAMarketStateModel
    from gx1.contracts.market_state_ta_m5_decision_v1 import IntegratedTAM5DecisionModel, fit_statistics
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    stats = fit_statistics(fit)
    need(stats['sha256'] == decision.EMA_FIT_STATS_SHA
         and context['view']['contract_sha256'] == decision.EMA_VIEW_SHA,
         'START_STATISTICS_VIEW')
    trainer._set_deterministic(SEED, device, trainer.DETERMINISTIC_FP32)
    state = build_native_direction_research_model(context['view'],
        prefix_normalization=context['packet'],
        constructor_metadata=context['constructor'], device=torch.device('cpu'),
        unobserved_units_profile=context['units_profile'])
    core = IntegratedTAMarketStateModel(NativeFutureDualModel(state))
    wrapper = IntegratedTAM5DecisionModel(core, stats, aux_fold_id=1).to(device)
    need(wrapper.research_objective['contract_sha256'] == decision.EMA_DECISION_SHA
         and canonical_model_state_sha256(wrapper.state_dict()) == decision.EMA_INITIAL_SHA
         and core.research_objective['contract_sha256'] == decision.EMA_CORE_SHA,
         'ORIGINAL_FRESH_INITIAL')
    core_sha = canonical_model_state_sha256(core.state_dict())
    need(core_sha == ORIGINAL_CORE_INITIAL_SHA, 'ORIGINAL_CORE_INITIAL_STATE')
    del wrapper
    return core, core_sha


def score_fit(model, context, *, deadline):
    """Fixed full Fold1 FIT S/F score, matching the later proper-score meaning."""
    import numpy as np
    import torch
    from gx1.audit.observe_market_state_ta_fold2_fresh_initial_v1 import direction_readout
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets, sampled_query_k
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K

    axes, tape = context['axes'], context['fit_tape']
    rows, issues = axes['parent_rows'][:FIT_ROWS], axes['decision_ns'][:FIT_ROWS]
    sampled = sampled_query_k(issues, pass_index=0, seed=SEED)
    native = np.broadcast_to(np.asarray(NATIVE_K, dtype=np.int64), sampled.shape).copy()
    truth = {
        'sampled_S': load_bound_time_query_targets().gather_fit_sampled(
            rows, issues, fold_id=1, pass_index=0, seed=SEED),
        'native_S': load_bound_five_endpoint_targets().gather(
            rows, issues, fold_id=1, partition='fit'),
        'native_F': load_bound_future_displacement().gather(
            rows, issues, fold_id=1, partition='fit'),
    }
    raw = {name: np.empty((FIT_ROWS, 5, 2), dtype=np.float32) for name in truth}
    structure = np.empty((FIT_ROWS, 2), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for left in range(0, FIT_ROWS, BATCH):
            need(time.monotonic() < deadline, 'FIT_SCORE_DEADLINE')
            right = min(left + BATCH, FIT_ROWS)
            features = context['features']('TRAIN', rows[left:right], issues[left:right])
            grid = torch.from_numpy(np.stack((sampled[left:right], native[left:right]),
                                             axis=1).copy()).to(next(model.parameters()).device)
            output = model(features, query_k=grid)
            raw['sampled_S'][left:right] = output['state_activity_side_logits'].cpu().numpy()
            raw['native_S'][left:right] = output['state_native_activity_side_logits'].cpu().numpy()
            raw['native_F'][left:right] = output['future_activity_side_logits'].cpu().numpy()
            structure[left:right] = output['m5_structure_hit_logits'].cpu().numpy()
            del features, output
    prior = context['priors']['tasks']
    prior_raw = {name: np.asarray(prior[key]['raw_logits'], dtype=np.float32)
                 for name, key in (('native_S', 'state'), ('native_F', 'future'))}
    report = {'proper': {name: paired._proper(raw[name], truth[name],
        sampled if name == 'sampled_S' else native, prior=prior_raw.get(name))
        for name in raw},
        'direction': {name: direction_readout(raw[name], truth[name], issues, np)
                      for name in raw},
        'rows': FIT_ROWS, 'model_forwards': FIT_BATCHES,
        'test_data_used': False}
    capture = {'parent_rows': rows, 'issue_ns': issues,
               'sampled_query_k': sampled, 'native_query_k': native,
               'structure_raw': structure,
               'structure_hit': tape['structure_hit'],
               'structure_observed': tape['structure_observed']}
    for name in raw:
        capture[name + '_raw'] = raw[name]
        for key in ('allowed', 'eligible', 'mature'):
            capture[name + '_' + key] = truth[name][key]
    return report, capture


def h480_sign_panel(capture, targets):
    """F-H4 sign only: complete-path side markout against fixed LONG."""
    import numpy as np
    n = len(capture['parent_rows'])
    complete = targets['path_observed'][:, 5].astype(bool)
    quality = targets['net_bps'][:, 5, :].astype(np.float64) - \
              targets['pre_mfe_mae_upper_bps'][:, 5, :].astype(np.float64)
    need(complete.shape == (n,) and quality.shape == (n, 2)
         and np.isfinite(quality[complete]).all(), 'H480_TARGETS')
    up = capture['native_F_raw'][:, 3, 1] >= 0
    chosen = quality[complete, np.where(up[complete], 0, 1)]
    always = quality[complete, 0]
    capture['h480_complete'] = complete
    capture['h480_quality_long_short'] = quality.astype(np.float32)
    return {'complete_path_rows': int(complete.sum()),
            'F_H4_sign_short_count': int((~up[complete]).sum()),
            'F_H4_sign_quality_bps_mean': float(chosen.mean()),
            'always_LONG_quality_bps_mean': float(always.mean()),
            'F_H4_sign_minus_always_LONG_bps_mean': float((chosen-always).mean()),
            'role': 'overlapping_h480_markout_not_trading_PnL',
            'test_data_used': False}


def fit_one_pass(model, context, *, deadline):
    import numpy as np
    import torch
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_ta_integrated_paired_objective_v1 import paired_integrated_loss
    from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets, sampled_query_k
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
    from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import ADAMW_DEFAULTS
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    axes, tape = context['axes'], context['fit_tape']
    rows, issues = axes['parent_rows'][:FIT_ROWS], axes['decision_ns'][:FIT_ROWS]
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), **ADAMW_DEFAULTS)
    need(not optimizer.state and len(optimizer.param_groups) == 1, 'EMPTY_OPTIMIZER')
    optimizer.zero_grad(set_to_none=True)
    state_bound = load_bound_five_endpoint_targets()
    future_bound = load_bound_future_displacement()
    sampled_bound = load_bound_time_query_targets()
    blocks, updates, skipped = [], 0, 0
    current = {'first_batch': 0, 'loss': 0., 'sampled_S': 0.,
               'native_S': 0., 'native_F': 0., 'structure': 0.,
               'grad_norm': 0., 'updates': 0, 'skipped': 0}
    for index in range(FIT_BATCHES):
        need(time.monotonic() < deadline, 'FIT_DEADLINE')
        left, right = index * BATCH, min((index + 1) * BATCH, FIT_ROWS)
        r, i = rows[left:right], issues[left:right]
        features = context['features']('TRAIN', r, i)
        sampled = sampled_query_k(i, pass_index=0, seed=SEED)
        native = np.broadcast_to(np.asarray(NATIVE_K, dtype=np.int64), sampled.shape)
        grid = torch.from_numpy(np.stack((sampled, native), axis=1).copy()).to(next(model.parameters()).device)
        output = model(features, query_k=grid)
        loss = paired_integrated_loss(model_output=output,
            sampled_truth=sampled_bound.gather_fit_sampled(
                r, i, fold_id=1, pass_index=0, seed=SEED),
            native_truth=state_bound.gather(r, i, fold_id=1, partition='fit'),
            future_truth=future_bound.gather(r, i, fold_id=1, partition='fit'),
            query_grid=grid, structure_hit=tape['structure_hit'][left:right],
            structure_observed=tape['structure_observed'][left:right], fold_id=1)
        need(loss['contract_sha256'] == sf_contract(fold_id=1)['contract_sha256']
             and bool(torch.isfinite(loss['loss'])), 'FIT_LOSS')
        for key, source in (('loss', 'loss'), ('sampled_S', 'sampled_S_loss'),
                            ('native_S', 'native_S_loss'), ('native_F', 'native_F_loss'),
                            ('structure', 'structure_loss')):
            current[key] += float(loss[source].detach())
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
        if (index + 1) % 256 == 0 or index + 1 == FIT_BATCHES:
            visits = index + 1 - current['first_batch']
            blocks.append({'first_batch': current['first_batch'],
                'last_batch_exclusive': index + 1,
                **{'mean_' + key: current[key] / visits for key in
                   ('loss', 'sampled_S', 'native_S', 'native_F', 'structure')},
                'mean_gradient_norm_per_update':
                    current['grad_norm'] / current['updates'] if current['updates'] else None,
                'updates': current['updates'], 'skipped': current['skipped']})
            current = {'first_batch': index + 1, 'loss': 0., 'sampled_S': 0.,
                       'native_S': 0., 'native_F': 0., 'structure': 0.,
                       'grad_norm': 0., 'updates': 0, 'skipped': 0}
    need(updates == FIT_BATCHES and skipped == 0, 'ALL_FIT_UPDATES')
    return blocks, updates


def run(plan_sha):
    import numpy as np
    import torch
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256

    guard = require_guarded_cuda_trainer_execution()
    plan, policy_at_claim = inspect(plan_sha, admitted=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    ROOT.mkdir()
    claim = decision.publish('CLAIM.json', {'stage': STAGE, 'plan': ref(PLAN),
        'policy_at_claim': policy_at_claim, 'source_commit': head,
        'physical_windows_boot_id': paired.physical_windows_boot_id(),
        'guard': guard, 'test_data_used': False}, root=ROOT)
    started = time.monotonic()
    deadline = started + MAX_SECONDS - 60
    context, result_ref = None, None
    try:
        device = torch.device('cuda', 0)
        inputs = successor_input_refs()
        need(inputs == plan['inputs'], 'INPUT_REFS')
        context = prepare_ema(device=device, inputs=inputs)
        full = decision.load_targets(inputs['fit_targets'], decision.TRAIN_ROWS)
        fit = decision.target_slice(full, 0, FIT_ROWS)
        later = decision.target_slice(full, LATER_START, LATER_STOP)
        axes = context['axes']
        selected = context['selected']
        need(context['fit_rows'] == FIT_ROWS and context['score_rows'] == LATER_ROWS
             and context['fit_fold_id'] == context['score_fold_id'] == 1
             and context['score_partition'] == 'score'
             and np.array_equal(axes['parent_rows'][:FIT_ROWS], fit['parent_rows'])
             and np.array_equal(axes['decision_ns'][:FIT_ROWS], fit['issue_ns'])
             and np.array_equal(axes['parent_rows'][selected], later['parent_rows'])
             and np.array_equal(axes['decision_ns'][selected], later['issue_ns'])
             and context['view']['contract_sha256'] == plan['view_sha256'],
             'FIT_LATER_AXES')
        model, initial_core_sha = initial_core(context, fit, device=device)
        need(model.research_objective['contract_sha256'] == plan['core_contract_sha256'],
             'CORE_CONTRACT')
        blocks, updates = fit_one_pass(model, context, deadline=deadline)
        final_fit, fit_capture = score_fit(model, context, deadline=deadline)
        final_later, later_capture = paired.score(model, context,
            deadline=deadline, fold_id=1, partition='score')
        fit_panel = h480_sign_panel(fit_capture, fit)
        later_panel = h480_sign_panel(later_capture, later)
        final_sha = canonical_model_state_sha256(model.state_dict())
        need(source_clean(plan)
             and ref(REPO / 'NEXT_RUN_POLICY.json') == policy_at_claim
             and successor_input_refs() == inputs
             and frozen_fit.consumed_refs()[0] == plan['consumed_ema']
             and ref(frozen_fit.ROOT / 'RESULT.json') == plan['decision_fit_score']
             and subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                        text=True).strip() == head
             and time.monotonic() < deadline,
             'SOURCE_INPUT_POLICY_TIME_BOOKEND')
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint_ref = decision.publish('FINAL_CORE_MODEL.pt', checkpoint.getvalue(), root=ROOT)
        refs = {}
        for name, capture in (('FIT_READOUT.npz', fit_capture),
                              ('LATER_READOUT.npz', later_capture)):
            stream = io.BytesIO()
            np.savez_compressed(stream, **capture)
            refs[name] = decision.publish(name, stream.getvalue(), root=ROOT)
        context['release']()
        context = None
        baseline = read(plan['decision_fit_score'])
        fit_f = final_fit['proper']['native_F']['M5']['conditional_side']
        later_f = final_later['proper']['native_F']['M5']['conditional_side']
        primary = {'fit_native_F_M5': fit_f, 'later_native_F_M5': later_f,
            'fit_brier_below_FIT_prior': fit_f['brier'] < fit_f['prior_brier'],
            'fit_log_below_FIT_prior': fit_f['log'] < fit_f['prior_log'],
            'later_brier_below_FIT_prior': later_f['brier'] < later_f['prior_brier'],
            'later_log_below_FIT_prior': later_f['log'] < later_f['prior_log'],
            'decision_fit_native_F_M5': baseline['native_S_F']['native_F']['proper']['M5']['conditional_side'],
            'decision_later_native_F_M5': read(plan['consumed_ema']['result'])['final_core']['proper']['native_F']['M5']['conditional_side']}
        result_ref = decision.publish('RESULT.json', {
            'status': 'COMPLETE_ONE_PASS_EMA_SF_ONLY_CONTROL_FIT_LATER_SCORE',
            'plan': ref(PLAN), 'claim': claim, 'policy_at_claim': policy_at_claim,
            'source_commit': head, 'source_unchanged': True,
            'guard': guard, 'inputs': inputs,
            'consumed_ema': plan['consumed_ema'],
            'decision_fit_score': plan['decision_fit_score'],
            'initial_core_state_sha256': initial_core_sha,
            'final_core_state_sha256': final_sha,
            'checkpoint': checkpoint_ref,
            'fit_capture': refs['FIT_READOUT.npz'],
            'later_capture': refs['LATER_READOUT.npz'],
            'fit_rows': FIT_ROWS, 'later_rows': LATER_ROWS,
            'optimizer_updates': updates, 'training_forwards': FIT_BATCHES,
            'evaluation_forwards': FIT_BATCHES + LATER_BATCHES,
            'fit_blocks': blocks, 'final_fit_core': final_fit,
            'final_later_core': final_later,
            'primary_diagnostic': primary,
            'fixed_F_H4_sign_h480': {'FIT': fit_panel, 'LATER': later_panel},
            'normalization_fits': 0, 'test_data_used': False,
            'economic_edge_proven': False,
            'elapsed_seconds': time.monotonic() - started}, root=ROOT)
        decision.publish('TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'plan': ref(PLAN), 'claim': claim, 'result': result_ref,
            'policy_at_claim': policy_at_claim,
            'source_unchanged': True, 'relaunch_allowed': False,
            'remaining_invocations': 0, 'test_data_used': False}, root=ROOT)
    except BaseException as error:
        decision.publish('TERMINAL.json', {'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'plan': ref(PLAN), 'claim': claim, 'result': result_ref,
            'policy_at_claim': policy_at_claim,
            'error': type(error).__name__ + ':' + str(error)[:2000],
            'relaunch_allowed': False, 'remaining_invocations': 0,
            'test_data_used': False}, root=ROOT)
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
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    need(args.plan == str(PLAN) and args.out_dir == str(ROOT), 'CLI_SCOPE')
    if args.preflight:
        plan, policy_ref = inspect(args.plan_sha256, admitted=False)
        import torch
        inputs = successor_input_refs()
        context = prepare_ema(device=torch.device('cpu'), inputs=inputs)
        try:
            full = decision.load_targets(inputs['fit_targets'], decision.TRAIN_ROWS)
            fit = decision.target_slice(full, 0, FIT_ROWS)
            core, core_sha = initial_core(context, fit, device=torch.device('cpu'))
            del core
        finally:
            context['release']()
        print(json.dumps({'status': 'PREFLIGHT_PASS', 'plan': ref(PLAN),
            'policy': policy_ref, 'initial_core_state_sha256': core_sha,
            'scope': {key: plan[key] for key in ('fit_rows', 'later_rows',
                'batch_size', 'batch_visits', 'fit_score_forwards',
                'later_score_forwards', 'model_forwards', 'max_optimizer_updates',
                'max_seconds', 'capped_execution')},
            'physical_windows_boot_id': paired.physical_windows_boot_id(),
            'fresh_physical_boot': paired.physical_windows_boot_id() > PRIOR_BOOT,
            'model_constructions': 1, 'model_forwards': 0,
            'optimizer_updates': 0, 'output_root_absent': True,
            'claim_created': False, 'test_data_used': False}, sort_keys=True))
    else:
        run(args.plan_sha256)


if __name__ == '__main__':
    main()
