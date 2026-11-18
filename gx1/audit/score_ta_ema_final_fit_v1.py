"""One-use, zero-update replay of the consumed EMA checkpoint on Fold1 FIT."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import time

from gx1.audit import run_market_state_ta_m5_decision_fit_v1 as prior_fit
from gx1.audit.market_state_ta_ema_delta_campaign_inputs_v1 import (
    FIT_ROWS, REPO, _receipts, prepare, successor_input_refs,
)
from gx1.audit.run_market_state_ta_apr_jun_paired_fit_v1 import BATCH, SEED, _proper
from gx1.audit.run_market_state_ta_m5_decision_fit_v1 import (
    EMA_ROOT, EMA_SOURCES, decision_readout, load_targets, publish, target_slice,
)
from gx1.audit.market_state_ta_apr_jun_campaign_inputs_v1 import need, read, ref

STAGE = 'ta_m5_ema_final_fit_score_20261118_001'
PLAN = REPO / 'configs/research/TA_M5_EMA_FINAL_FIT_SCORE_20261118_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/TA_M5_EMA_FINAL_FIT_SCORE_20261118_001')
FORWARDS = (FIT_ROWS + BATCH - 1) // BATCH
MAX_SECONDS = 3600
SOURCES = tuple(dict.fromkeys((*EMA_SOURCES,
    'gx1/audit/score_ta_ema_final_fit_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/model_state_digest_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/market_state_time_query_native_F_batch_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/market_state_time_query_target_v1.py',
    'gx1/contracts/entry_model_native_input_normalization_v1.py',
    'gx1/contracts/entry_exit_feature_surface_v1.py',
    'gx1/audit/observe_market_state_ta_fold2_fresh_initial_v1.py',
    'gx1/audit/qualify_market_state_mature_prefix_normalization_v1.py',
    'tests/test_score_ta_ema_final_fit_v1.py',
    'scripts/gx1_guarded_trainer_exec.sh')))
CONSUMED_PATHS = {
    'plan': prior_fit.EMA_PLAN,
    'claim': EMA_ROOT / 'CLAIM.json',
    'result': EMA_ROOT / 'RESULT.json',
    'terminal': EMA_ROOT / 'TERMINAL.json',
    'checkpoint': EMA_ROOT / 'FINAL_MODEL.pt',
    'final_capture': EMA_ROOT / 'FINAL_SCORE_READOUT.npz',
}


def consumed_refs():
    refs = {name: ref(path) for name, path in CONSUMED_PATHS.items()}
    result, terminal = read(refs['result']), read(refs['terminal'])
    need(result['status'] == 'COMPLETE_ONE_PASS_EMA_DELTA_FOLD1_M5_DECISION_LATER_SCORE'
         and result['plan'] == refs['plan'] and result['claim'] == refs['claim']
         and result['checkpoint'] == refs['checkpoint']
         and result['final_capture'] == refs['final_capture']
         and result['final_model_state_sha256'] ==
             '7323cd0103f37250ec1ed6bc8ab7d680836feab5cd9be84992e23cf7762f9a5d'
         and result['fit_rows'] == FIT_ROWS and result['score_rows'] == 17489
         and result['optimizer_updates'] == 1849
         and result['source_unchanged'] is True
         and result['test_data_used'] is False
         and result['economic_edge_proven'] is False
         and terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
         and terminal['plan'] == refs['plan'] and terminal['result'] == refs['result']
         and terminal['source_unchanged'] is True
         and terminal['remaining_invocations'] == 0, 'CONSUMED_EMA_CHAIN')
    return refs, result


def build_plan(*, source_base_commit=None):
    """Exact source and input declaration; does not admit or execute a model."""
    from gx1.contracts.market_state_ta_m5_decision_v1 import fit_statistics

    inputs = successor_input_refs()
    _receipts(inputs)
    consumed, result = consumed_refs()
    ema_plan = read(consumed['plan'])
    need(ema_plan['inputs'] == prior_fit.ema_input_refs()
         and {key: ema_plan['inputs'][key] for key in inputs} == inputs
         and result['fit_statistics_sha256'] == ema_plan['fit_statistics_sha256']
         and result['final_model_state_sha256'] ==
             '7323cd0103f37250ec1ed6bc8ab7d680836feab5cd9be84992e23cf7762f9a5d',
         'EMA_PLAN_INPUTS')
    full = load_targets(inputs['fit_targets'], prior_fit.TRAIN_ROWS)
    fit = target_slice(full, 0, FIT_ROWS)
    stats = fit_statistics(fit)
    need(stats['sha256'] == result['fit_statistics_sha256'], 'FIT_STATISTICS')
    head = source_base_commit or subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return {
        'schema_version': 'gx1_ta_m5_ema_final_fit_score_plan_v1',
        'stage': STAGE, 'status': 'PREPARED_NOT_ADMITTED',
        'source_base_commit': head,
        'source_owners': {name: ref(REPO / name)['sha256'] for name in SOURCES},
        'inputs': inputs, 'consumed_ema': consumed,
        'output_root': str(ROOT), 'partition': 'fit', 'fold_id': 1,
        'fit_parent_start': 0, 'fit_rows': FIT_ROWS,
        'batch_size': BATCH, 'model_forwards': FORWARDS,
        'max_optimizer_updates': 0, 'max_seconds': MAX_SECONDS,
        'seed': SEED, 'fit_statistics_sha256': stats['sha256'],
        'final_model_state_sha256': result['final_model_state_sha256'],
        'h480_complete_panel_predeclared': True,
        'native_S_F_fit_auc_brier_predeclared': True,
        'capped_execution': {'class': 'producer', 'memory': '20G', 'swap': '512M'},
        'training_authorized': False, 'test_authorized': False,
        'online_authorized': False, 'broker_access_allowed': False,
        'test_data_used': False, 'economic_edge_proven': False,
    }


def source_clean(plan):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                   text=True).strip()
    return (subprocess.check_output(['git', 'branch', '--show-current'],
                                    cwd=REPO, text=True).strip() == 'work/gx1-current'
            and subprocess.run(['git', 'merge-base', '--is-ancestor',
                                plan['source_base_commit'], head],
                               cwd=REPO, capture_output=True).returncode == 0
            and not subprocess.check_output(['git', 'status', '--porcelain'],
                                            cwd=REPO, text=True)
            and set(plan['source_owners']) == set(SOURCES)
            and all(ref(REPO / name)['sha256'] == plan['source_owners'][name]
                    for name in SOURCES))


def inspect(plan_sha, *, admitted):
    need(ref(PLAN)['sha256'] == plan_sha and not ROOT.exists(), 'PLAN_OUTPUT')
    plan = read(ref(PLAN))
    need(plan == build_plan(source_base_commit=plan.get('source_base_commit'))
         and Path.cwd().resolve() == REPO and source_clean(plan),
         'FIXED_SCOPE_SOURCE')
    policy_binding = ref(REPO / 'NEXT_RUN_POLICY.json')
    policy = read(policy_binding)
    gate = policy.get(STAGE, {})
    consumed_gate = policy.get(prior_fit.EMA_STAGE, {})
    need(consumed_gate.get('status') == 'REVIEWED_CONSUMED_EMA_SIDE_COLLAPSE_EDGE_UNPROVEN'
         and consumed_gate.get('plan') == plan['consumed_ema']['plan']
         and consumed_gate.get('claim') == plan['consumed_ema']['claim']
         and consumed_gate.get('result') == plan['consumed_ema']['result']
         and consumed_gate.get('terminal') == plan['consumed_ema']['terminal']
         and consumed_gate.get('checkpoint') == plan['consumed_ema']['checkpoint']
         and consumed_gate.get('final_capture') == plan['consumed_ema']['final_capture']
         and consumed_gate.get('remaining_invocations') == 0
         and consumed_gate.get('completed_invocations') == 1
         and consumed_gate.get('execution_authorized') is False
         and policy.get('training_enabled') is False
         and policy.get('native_training_enabled') is False
         and policy.get('full_epoch_training_allowed') is False
         and policy.get('native_launch_authorized') is False
         and policy.get('full_val_allowed') is False
         and policy.get('exit_training_and_evaluation_this_wave') is False
         and gate.get('plan') == ref(PLAN)
         and gate.get('output_root') == str(ROOT)
         and gate.get('fit_rows') == FIT_ROWS
         and gate.get('model_forwards') == FORWARDS
         and gate.get('max_optimizer_updates') == 0
         and gate.get('training_authorized') is False
         and gate.get('test_authorized') is False
         and gate.get('online_authorized') is False
         and gate.get('broker_access_allowed') is False
         and gate.get('test_data_used') is False
         and gate.get('relaunch_allowed') is False, 'CLOSED_POLICY')
    if admitted:
        need(gate.get('status') == 'ADMITTED_ONE_USE'
             and gate.get('execution_authorized') is True
             and gate.get('remaining_invocations') == gate.get('max_invocations') == 1
             and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0,
             'ONE_USE_ADMISSION')
    else:
        need(gate.get('status') == 'PREPARED_NOT_ADMITTED'
             and gate.get('execution_authorized') is False
             and gate.get('remaining_invocations') == 0,
             'NOT_ADMITTED')
    return plan, policy_binding


def h480_side_readout(predicted, targets):
    """Fixed full complete-path comparison, independent of action selection."""
    import numpy as np
    from gx1.audit.observe_market_state_ta_fold2_fresh_initial_v1 import auc_with_ties
    need(predicted.shape == (FIT_ROWS, 6, 2, 3)
         and np.isfinite(predicted).all(), 'H480_PREDICTION')
    path = targets['path_observed'][:, 5]
    need(path.shape == (FIT_ROWS,) and int(path.sum()) > 1,
         'H480_COMPLETE_PATH')
    q_pred = predicted[path, 5, :, 0].astype(np.float64) - np.maximum(
        predicted[path, 5, :, 2].astype(np.float64), 0.0)
    q_actual = targets['net_bps'][path, 5, :].astype(np.float64) - \
        targets['pre_mfe_mae_upper_bps'][path, 5, :].astype(np.float64)
    delta_pred = q_pred[:, 0] - q_pred[:, 1]
    delta_actual = q_actual[:, 0] - q_actual[:, 1]
    need(np.isfinite(delta_actual).all(), 'H480_ACTUAL')
    predicted_long = delta_pred >= 0
    actual_long = delta_actual >= 0
    corr = float(np.corrcoef(delta_pred, delta_actual)[0, 1]) \
        if np.std(delta_pred) > 0 and np.std(delta_actual) > 0 else None
    auc = auc_with_ties(delta_pred, actual_long, np)
    chosen_side = np.where(predicted_long, 0, 1)
    return {
        'complete_path_rows': int(path.sum()),
        'predicted_side_ties': int((delta_pred == 0).sum()),
        'actual_side_ties': int((delta_actual == 0).sum()),
        'predicted_long_minus_short_quality_bps_mean': float(delta_pred.mean()),
        'actual_long_minus_short_quality_bps_mean': float(delta_actual.mean()),
        'side_difference_pearson': corr,
        'side_ranking_auc_for_actual_long_better': auc,
        'predicted_long_fraction': float(predicted_long.mean()),
        'actual_long_fraction': float(actual_long.mean()),
        'side_accuracy': float((predicted_long == actual_long).mean()),
        'predicted_long_quality_bps_mean': float(q_pred[:, 0].mean()),
        'predicted_short_quality_bps_mean': float(q_pred[:, 1].mean()),
        'actual_long_quality_bps_mean': float(q_actual[:, 0].mean()),
        'actual_short_quality_bps_mean': float(q_actual[:, 1].mean()),
        'side_only_quality_bps_mean': float(q_actual[np.arange(len(q_actual)), chosen_side].mean()),
        'always_long_quality_bps_mean': float(q_actual[:, 0].mean()),
        'side_only_net_bps_mean': float(targets['net_bps'][path, 5, :]
                                       [np.arange(len(q_actual)), chosen_side].mean()),
        'always_long_net_bps_mean': float(targets['net_bps'][path, 5, 0].mean()),
        'side_only_short_count': int((~predicted_long).sum()),
        'role': 'overlapping_markout_not_trading_PnL',
    }


def run(plan_sha):
    import numpy as np
    import torch
    from gx1.audit.observe_market_state_ta_fold2_fresh_initial_v1 import direction_readout
    from gx1.contracts.entry_native_direction_feature_view_v1 import build_native_direction_research_model
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_producer_execution
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_ta_integrated_model_v1 import IntegratedTAMarketStateModel
    from gx1.contracts.market_state_ta_m5_decision_v1 import IntegratedTAM5DecisionModel, fit_statistics
    from gx1.contracts.market_state_time_query_binding_v1 import sampled_query_k
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    guard = require_guarded_cuda_producer_execution()
    plan, policy_binding = inspect(plan_sha, admitted=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                   text=True).strip()
    ROOT.mkdir()
    claim = publish('CLAIM.json', {'plan': ref(PLAN), 'source_commit': head,
        'policy': policy_binding, 'inputs': plan['inputs'],
        'consumed_ema': plan['consumed_ema'], 'guard': guard,
        'planned_model_forwards': FORWARDS,
        'optimizer_updates': 0, 'test_data_used': False}, root=ROOT)
    started = time.monotonic()
    deadline = started + plan['max_seconds'] - 60
    context, result_ref = None, None
    try:
        device = torch.device('cuda', 0)
        inputs = successor_input_refs()
        need(inputs == plan['inputs'], 'CAMPAIGN_INPUT_REFS')
        context = prepare(device=device, inputs=inputs)
        full = load_targets(inputs['fit_targets'], prior_fit.TRAIN_ROWS)
        fit = target_slice(full, 0, FIT_ROWS)
        stats = fit_statistics(fit)
        axes, tape = context['axes'], context['fit_tape']
        rows, issues = axes['parent_rows'][:FIT_ROWS], axes['decision_ns'][:FIT_ROWS]
        need(context['fit_rows'] == FIT_ROWS and context['fit_fold_id'] == 1
             and np.array_equal(rows, fit['parent_rows'])
             and np.array_equal(issues, fit['issue_ns'])
             and stats['sha256'] == plan['fit_statistics_sha256'],
             'FIT_TARGET_FEATURE_AXES')
        trainer._set_deterministic(SEED, device, trainer.DETERMINISTIC_FP32)
        state = build_native_direction_research_model(context['view'],
            prefix_normalization=context['packet'], constructor_metadata=context['constructor'],
            device=torch.device('cpu'), unobserved_units_profile=context['units_profile'])
        core = IntegratedTAMarketStateModel(NativeFutureDualModel(state))
        model = IntegratedTAM5DecisionModel(core, stats, aux_fold_id=1).to(device)
        need(model.research_objective['contract_sha256'] == prior_fit.EMA_DECISION_SHA,
             'MODEL_CONTRACT')
        frozen = torch.load(plan['consumed_ema']['checkpoint']['path'],
                            map_location='cpu', weights_only=True)
        model.load_state_dict(frozen, strict=True)
        del frozen
        model_sha = canonical_model_state_sha256(model.state_dict())
        need(model_sha == plan['final_model_state_sha256'], 'FROZEN_MODEL_STATE')
        sampled = sampled_query_k(issues, pass_index=0, seed=SEED)
        native = np.broadcast_to(np.asarray(NATIVE_K, dtype=np.int64), sampled.shape).copy()
        truth = {
            'native_S': load_bound_five_endpoint_targets().gather(
                rows, issues, fold_id=1, partition='fit'),
            'native_F': load_bound_future_displacement().gather(
                rows, issues, fold_id=1, partition='fit'),
        }
        raw = {name: np.empty((FIT_ROWS, 5, 2), dtype=np.float32)
               for name in ('native_S', 'native_F')}
        decision = np.empty((FIT_ROWS, 6, 2, 3), dtype=np.float32)
        model.eval()
        with torch.inference_mode():
            for left in range(0, FIT_ROWS, BATCH):
                need(time.monotonic() < deadline, 'SCORE_DEADLINE')
                right = min(left + BATCH, FIT_ROWS)
                features = context['features']('TRAIN', rows[left:right], issues[left:right])
                query = torch.from_numpy(np.stack((sampled[left:right], native[left:right]),
                                                  axis=1).copy()).to(device)
                out = model(features, query_k=query)
                need(out['m5_cost_path_bps'].shape == (right - left, 6, 2, 3),
                     'DECISION_SHAPE')
                raw['native_S'][left:right] = out['state_native_activity_side_logits'].cpu().numpy()
                raw['native_F'][left:right] = out['future_activity_side_logits'].cpu().numpy()
                decision[left:right] = out['m5_cost_path_bps'].cpu().numpy()
                del features, out
        prior = context['priors']['tasks']
        prior_raw = {name: np.asarray(prior[key]['raw_logits'], dtype=np.float32)
                     for name, key in (('native_S', 'state'), ('native_F', 'future'))}
        native_report = {
            name: {'direction': direction_readout(raw[name], truth[name], issues, np),
                   'proper': _proper(raw[name], truth[name], native,
                                     prior=prior_raw[name])}
            for name in raw
        }
        h480 = h480_side_readout(decision, fit)
        action = decision_readout(decision, fit)
        need(source_clean(plan)
             and ref(REPO / 'NEXT_RUN_POLICY.json') == policy_binding
             and successor_input_refs() == inputs
             and consumed_refs()[0] == plan['consumed_ema']
             and canonical_model_state_sha256(model.state_dict()) == model_sha
             and time.monotonic() < deadline, 'SOURCE_INPUT_POLICY_TIME_BOOKEND')
        capture = {'parent_rows': rows, 'issue_ns': issues,
                   'native_query_k': native, 'm5_cost_path_bps': decision}
        for name in raw:
            capture[name + '_raw'] = raw[name]
            for key in ('allowed', 'eligible', 'mature'):
                capture[name + '_' + key] = truth[name][key]
        stream = io.BytesIO()
        np.savez_compressed(stream, **capture)
        capture_ref = publish('FIT_SCORE_READOUT.npz', stream.getvalue(), root=ROOT)
        context['release']()
        context = None
        result_ref = publish('RESULT.json', {
            'status': 'COMPLETE_FROZEN_EMA_FOLD1_FIT_SCORE_NO_UPDATE',
            'plan': ref(PLAN), 'claim': claim, 'policy_at_claim': policy_binding,
            'source_commit': head, 'source_unchanged': True,
            'consumed_ema': plan['consumed_ema'], 'guard': guard,
            'fit_rows': FIT_ROWS, 'model_forwards': FORWARDS,
            'optimizer_constructions': 0, 'optimizer_updates': 0,
            'backward_calls': 0, 'normalization_fits': 0,
            'model_state_sha256': model_sha, 'fit_capture': capture_ref,
            'h480_side': h480, 'native_S_F': native_report,
            'decision': action, 'test_data_used': False,
            'economic_edge_proven': False,
            'elapsed_seconds': time.monotonic() - started}, root=ROOT)
        publish('TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'plan': ref(PLAN), 'result': result_ref, 'claim': claim,
            'policy_at_claim': policy_binding, 'source_unchanged': True,
            'relaunch_allowed': False, 'remaining_invocations': 0,
            'test_data_used': False}, root=ROOT)
    except BaseException as error:
        publish('TERMINAL.json', {'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'plan': ref(PLAN), 'claim': claim, 'result': result_ref,
            'policy_at_claim': policy_binding,
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
        plan, policy_binding = inspect(args.plan_sha256, admitted=False)
        print(json.dumps({'status': 'PREFLIGHT_PASS', 'plan': ref(PLAN),
            'policy': policy_binding, 'scope': {
                'fit_rows': plan['fit_rows'],
                'planned_model_forwards': plan['model_forwards'],
                'max_optimizer_updates': 0, 'max_seconds': plan['max_seconds']},
            'model_constructions': 0, 'model_forwards': 0,
            'optimizer_updates': 0, 'output_root_absent': True,
            'claim_created': False,
            'test_data_used': False}, sort_keys=True))
    else:
        run(args.plan_sha256)


if __name__ == '__main__':
    main()
