"""One-use, zero-update EMA objective geometry at the shared fused-TF boundary."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time

from gx1.audit import score_ta_ema_final_fit_v1 as replay
from gx1.audit import run_market_state_ta_m5_decision_fit_v1 as fit_owner
from gx1.audit.market_state_ta_ema_delta_campaign_inputs_v1 import prepare, successor_input_refs
from gx1.audit.market_state_ta_apr_jun_campaign_inputs_v1 import need, read, ref
from gx1.audit.run_market_state_ta_m5_decision_fit_v1 import load_targets, publish, target_slice

REPO = replay.REPO
STAGE = 'ta_m5_ema_loss_gradient_probe_20261118_001'
PLAN = REPO / 'configs/research/TA_M5_EMA_LOSS_GRADIENT_PROBE_20261118_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/TA_M5_EMA_LOSS_GRADIENT_PROBE_20261118_001')
BATCH = replay.BATCH
FIT_ROWS = replay.FIT_ROWS
FULL_BATCHES = FIT_ROWS // BATCH
BATCH_INDICES = tuple(i * (FULL_BATCHES - 1) // 19 for i in range(20))
MAX_SECONDS = 1800
SOURCE = 'gx1/audit/probe_ta_ema_loss_gradient_v1.py'
SOURCES = tuple(dict.fromkeys((*replay.SOURCES, SOURCE)))
SCORE_PATHS = {'result': replay.ROOT / 'RESULT.json',
               'terminal': replay.ROOT / 'TERMINAL.json',
               'capture': replay.ROOT / 'FIT_SCORE_READOUT.npz'}


def frozen_score_refs():
    refs = {key: ref(path) for key, path in SCORE_PATHS.items()}
    result, terminal = read(refs['result']), read(refs['terminal'])
    need(result['status'] == 'COMPLETE_FROZEN_EMA_FOLD1_FIT_SCORE_NO_UPDATE'
         and result['fit_rows'] == FIT_ROWS and result['model_forwards'] == replay.FORWARDS
         and result['optimizer_updates'] == result['backward_calls'] == 0
         and result['fit_capture'] == refs['capture']
         and result['source_unchanged'] is True and result['test_data_used'] is False
         and terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
         and terminal['result'] == refs['result']
         and terminal['remaining_invocations'] == 0, 'FROZEN_SCORE_CHAIN')
    return refs


def build_plan(*, source_base_commit=None):
    """Bind existing frozen inputs/checkpoint and 20 full B16 FIT batches."""
    base = replay.build_plan(source_base_commit=source_base_commit)
    need(FULL_BATCHES == 1848 and len(BATCH_INDICES) == len(set(BATCH_INDICES)) == 20
         and BATCH_INDICES[0] == 0 and BATCH_INDICES[-1] == 1847,
         'PROBE_BATCH_SELECTION')
    score_refs = frozen_score_refs()
    need(read(score_refs['result'])['model_state_sha256'] ==
         base['final_model_state_sha256'], 'SCORE_MODEL_DIGEST')
    return {'schema_version': 'gx1_ta_m5_ema_loss_gradient_probe_plan_v1',
        'stage': STAGE, 'status': 'PREPARED_NOT_ADMITTED',
        'source_base_commit': base['source_base_commit'],
        'source_owners': {**base['source_owners'], SOURCE: ref(REPO / SOURCE)['sha256']},
        'inputs': base['inputs'], 'consumed_ema': base['consumed_ema'],
        'frozen_fit_score': score_refs,
        'fit_statistics_sha256': base['fit_statistics_sha256'],
        'final_model_state_sha256': base['final_model_state_sha256'],
        'output_root': str(ROOT), 'partition': 'fit', 'fold_id': 1,
        'fit_rows': FIT_ROWS, 'batch_size': BATCH,
        'full_batch_indices': list(BATCH_INDICES),
        'model_forwards': len(BATCH_INDICES) + 1,
        'autograd_calls': 4 * len(BATCH_INDICES),
        'max_optimizer_updates': 0, 'max_seconds': MAX_SECONDS,
        'falsifiable_question': 'Does decision primary oppose or overwhelm native F M5 side at the shared fused-TF boundary on fixed Fold1 FIT B16 batches?',
        'gradient_scope': 'fused_TF_hidden_only_not_upstream_parameter_gradient',
        'interpretation_rule': 'Opposition only when pooled decision-vs-F-side dot < 0 and a majority of F-supported batches have negative dot; otherwise not established. No automatic fit or edge claim.',
        'capped_execution': {'class': 'producer', 'memory': '20G', 'swap': '512M'},
        'training_authorized': False, 'test_authorized': False,
        'online_authorized': False, 'broker_access_allowed': False,
        'test_data_used': False, 'economic_edge_proven': False}


def source_clean(plan):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
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
         and Path.cwd().resolve() == REPO and source_clean(plan), 'FIXED_SCOPE_SOURCE')
    policy_binding = ref(REPO / 'NEXT_RUN_POLICY.json')
    policy = read(policy_binding)
    gate = policy.get(STAGE, {})
    prior = policy.get(replay.STAGE, {})
    need(prior.get('status') == 'REVIEWED_CONSUMED_FIT_SF_WEAK_EDGE_UNPROVEN'
         and prior.get('result') == plan['frozen_fit_score']['result']
         and prior.get('terminal') == plan['frozen_fit_score']['terminal']
         and prior.get('fit_capture') == plan['frozen_fit_score']['capture']
         and prior.get('remaining_invocations') == 0
         and prior.get('execution_authorized') is False
         and policy.get('training_enabled') is False
         and policy.get('native_training_enabled') is False
         and policy.get('full_epoch_training_allowed') is False
         and policy.get('native_launch_authorized') is False
         and policy.get('full_val_allowed') is False
         and policy.get('exit_training_and_evaluation_this_wave') is False
         and gate.get('plan') == ref(PLAN)
         and gate.get('output_root') == str(ROOT)
         and gate.get('fit_rows') == FIT_ROWS
         and gate.get('model_forwards') == len(BATCH_INDICES) + 1
         and gate.get('autograd_calls') == 4 * len(BATCH_INDICES)
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
             and gate.get('remaining_invocations') == 0, 'NOT_ADMITTED')
    return plan, policy_binding


def isolated_native_f_m5_side(raw, truth, *, torch):
    """Exact native F M5 conditional-side term inside 0.10 times paired loss."""
    from torch.nn import functional as F
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import m5_equal_observation_profile
    from gx1.contracts.market_state_ta_m5_decision_v1 import AUX_WEIGHT
    from gx1.contracts.market_state_ta_integrated_paired_objective_v1 import F_WEIGHT
    profile = m5_equal_observation_profile(fold_id=1)['future']
    need(profile == {'known_side_rows': 11909, 'nonempty_batches': 1844}
         and AUX_WEIGHT == .10 and F_WEIGHT == 1.0
         and raw.ndim == 3 and raw.shape[1:] == (5, 2)
         and raw.dtype == torch.float32, 'F_SIDE_CONTRACT')
    allowed = torch.from_numpy(truth['allowed'][:, 0].copy()).to(raw.device)
    eligible = torch.from_numpy(truth['eligible'][:, 0].copy()).to(raw.device)
    mature = torch.from_numpy(truth['mature'][:, 0].copy()).to(raw.device)
    known = eligible & mature & ((allowed == 1) | (allowed == 4))
    nll = F.binary_cross_entropy_with_logits(raw[:, 0, 1],
        (allowed == 4).to(raw.dtype), reduction='none')
    coefficient = (AUX_WEIGHT * F_WEIGHT * profile['nonempty_batches']
                   / profile['known_side_rows'] / 5)
    return torch.where(known, nll, torch.zeros_like(nll)).sum() * coefficient, int(known.sum())


def geometry(a, b):
    import torch
    aa = a.detach().to(dtype=torch.float64).reshape(-1)
    bb = b.detach().to(dtype=torch.float64).reshape(-1)
    dot = float(torch.dot(aa, bb))
    a2, b2 = float(torch.dot(aa, aa)), float(torch.dot(bb, bb))
    return {'dot': dot, 'a_norm': math.sqrt(a2), 'b_norm': math.sqrt(b2),
            'cosine': dot / math.sqrt(a2 * b2) if a2 > 0 and b2 > 0 else None}


def summarize_records(records):
    """Predeclared fused-boundary decision, without selecting a training arm."""
    need(len(records) == len(BATCH_INDICES), 'SUMMARY_BATCH_COUNT')
    keys = ('decision_vs_F_side', 'auxiliary_vs_F_side',
            'total_vs_F_side', 'decision_vs_auxiliary')
    pooled = {}
    for key in keys:
        values = [item[key] for item in records]
        dot = math.fsum(v['dot'] for v in values)
        a2 = math.fsum(v['a_norm'] ** 2 for v in values)
        b2 = math.fsum(v['b_norm'] ** 2 for v in values)
        supported = sum(item['native_F_M5_known_side'] > 0 for item in records)
        negative = sum(item['native_F_M5_known_side'] > 0 and item[key]['dot'] < 0
                       for item in records)
        pooled[key] = {'dot': dot, 'a_norm': math.sqrt(a2), 'b_norm': math.sqrt(b2),
            'cosine': dot / math.sqrt(a2 * b2) if a2 > 0 and b2 > 0 else None,
            'supported_batches': supported, 'negative_supported_batches': negative,
            'a_to_b_norm_ratio': math.sqrt(a2 / b2) if b2 > 0 else None}
    comparison = pooled['decision_vs_F_side']
    opposed = (comparison['dot'] < 0
               and comparison['negative_supported_batches']
                   > comparison['supported_batches'] / 2)
    return {'pooled': pooled,
            'interpretation': ('MEASURED_OPPOSITION_AT_FUSED_BOUNDARY'
                               if opposed else 'OPPOSITION_NOT_ESTABLISHED'),
            'interpretation_limit': 'Local fused-hidden derivative; not historical optimizer or upstream parameter direction, later generalization, or economic edge.'}


def run(plan_sha):
    import numpy as np
    import torch
    from gx1.contracts.entry_native_direction_feature_view_v1 import build_native_direction_research_model
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_producer_execution
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_ta_integrated_model_v1 import IntegratedTAMarketStateModel
    from gx1.contracts.market_state_ta_m5_decision_v1 import (AUX_WEIGHT, CORE_KEYS,
        IntegratedTAM5DecisionModel, decision_loss, fit_statistics)
    from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets, sampled_query_k
    from gx1.contracts.market_state_time_query_model_v2 import factorized_log_probabilities
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    guard = require_guarded_cuda_producer_execution()
    plan, policy_binding = inspect(plan_sha, admitted=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    ROOT.mkdir()
    claim = publish('CLAIM.json', {'plan': ref(PLAN), 'source_commit': head,
        'policy': policy_binding, 'guard': guard, 'model_forwards': plan['model_forwards'],
        'autograd_calls': plan['autograd_calls'], 'optimizer_updates': 0,
        'test_data_used': False}, root=ROOT)
    started = time.monotonic()
    deadline = started + MAX_SECONDS - 60
    context, result_ref = None, None
    try:
        device = torch.device('cuda', 0)
        inputs = successor_input_refs()
        need(inputs == plan['inputs'], 'INPUT_REFS')
        context = prepare(device=device, inputs=inputs)
        full = load_targets(inputs['fit_targets'], fit_owner.TRAIN_ROWS)
        fit = target_slice(full, 0, FIT_ROWS)
        stats = fit_statistics(fit)
        axes, tape = context['axes'], context['fit_tape']
        rows, issues = axes['parent_rows'][:FIT_ROWS], axes['decision_ns'][:FIT_ROWS]
        need(context['fit_rows'] == FIT_ROWS and context['fit_fold_id'] == 1
             and np.array_equal(rows, fit['parent_rows'])
             and np.array_equal(issues, fit['issue_ns'])
             and stats['sha256'] == plan['fit_statistics_sha256'],
             'FIT_TARGET_FEATURE_AXES')
        trainer._set_deterministic(replay.SEED, device, trainer.DETERMINISTIC_FP32)
        state = build_native_direction_research_model(context['view'],
            prefix_normalization=context['packet'], constructor_metadata=context['constructor'],
            device=torch.device('cpu'), unobserved_units_profile=context['units_profile'])
        model = IntegratedTAM5DecisionModel(
            IntegratedTAMarketStateModel(NativeFutureDualModel(state)),
            stats, aux_fold_id=1).to(device)
        need(model.research_objective['contract_sha256'] == fit_owner.EMA_DECISION_SHA,
             'MODEL_CONTRACT')
        frozen = torch.load(plan['consumed_ema']['checkpoint']['path'],
                            map_location='cpu', weights_only=True)
        model.load_state_dict(frozen, strict=True)
        del frozen
        model_sha = canonical_model_state_sha256(model.state_dict())
        need(model_sha == plan['final_model_state_sha256'], 'MODEL_DIGEST')
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        state_bound = load_bound_five_endpoint_targets()
        future_bound = load_bound_future_displacement()
        sampled_bound = load_bound_time_query_targets()
        records = []
        for position, batch_index in enumerate(BATCH_INDICES):
            need(time.monotonic() < deadline, 'DEADLINE')
            left, right = batch_index * BATCH, (batch_index + 1) * BATCH
            r, issue = rows[left:right], issues[left:right]
            features = context['features']('TRAIN', r, issue)
            sampled = sampled_query_k(issue, pass_index=0, seed=replay.SEED)
            native = np.broadcast_to(np.asarray(NATIVE_K, dtype=np.int64), sampled.shape)
            grid = torch.from_numpy(np.stack((sampled, native), axis=1).copy()).to(device)
            m1 = torch.from_numpy(tape['m1_mid_ohlc_bps'][left:right].copy()).to(device)
            m1_observed = torch.from_numpy(tape['m1_observed'][left:right].copy()).to(device)
            with torch.no_grad():
                original = model.core(features, query_k=grid,
                    m1_path_bps=m1, m1_observed=m1_observed,
                    return_fused_tf_hidden=True)
                if position == 0:
                    original_decision = model(features, query_k=grid,
                        m1_path_bps=m1, m1_observed=m1_observed)['m5_cost_path_bps']
            fused = original.pop('fused_tf_hidden').detach().requires_grad_(True)
            state_grid = model.core.dual.state_model.decode_raw_query_grid(fused, grid)
            state_raw, native_raw = state_grid[:, 0], state_grid[:, 1]
            future_raw = model.core.dual.decode_native_F_raw(fused)
            structure_raw = model.core.m5_structure_head(fused[:, 0])
            prediction = model.target_mean_bps + model.target_std_bps * model.decision_head(
                fused.reshape(BATCH, -1)).reshape(BATCH, 6, 2, 3)
            raw = {'state_activity_side_logits': state_raw,
                'state_native_activity_side_logits': native_raw,
                'future_activity_side_logits': future_raw,
                'm5_structure_hit_logits': structure_raw}
            need(all(torch.equal(raw[name].detach(), original[name]) for name in raw)
                 and (position != 0 or torch.equal(prediction.detach(), original_decision)),
                 'DETACHED_FUSED_FORWARD_PARITY')
            output = {**raw,
                'state_log_probs': factorized_log_probabilities(state_raw),
                'state_native_log_probs': factorized_log_probabilities(native_raw),
                'future_log_probs': factorized_log_probabilities(future_raw),
                'm5_cost_path_bps': prediction}
            need(set(output) == CORE_KEYS | {'m5_cost_path_bps'}, 'OUTPUT_KEYS')
            target = {key: fit[key][left:right] for key in
                ('terminal_observed', 'path_observed', 'net_bps', 'mfe_bps',
                 'pre_mfe_mae_upper_bps')}
            future_truth = future_bound.gather(r, issue, fold_id=1, partition='fit')
            report = decision_loss(model_output=output, targets=target, model=model,
                sampled_truth=sampled_bound.gather_fit_sampled(r, issue, fold_id=1,
                    pass_index=0, seed=replay.SEED),
                native_truth=state_bound.gather(r, issue, fold_id=1, partition='fit'),
                future_truth=future_truth, query_grid=grid,
                structure_hit=tape['structure_hit'][left:right],
                structure_observed=tape['structure_observed'][left:right])
            need(report['contract_sha256'] == fit_owner.EMA_DECISION_SHA,
                 'LOSS_CONTRACT')
            isolated, support = isolated_native_f_m5_side(future_raw, future_truth, torch=torch)
            primary = torch.autograd.grad(report['decision_loss'], fused, retain_graph=True)[0]
            auxiliary, aux_logit = torch.autograd.grad(AUX_WEIGHT * report['aux_loss'],
                (fused, future_raw), retain_graph=True)
            f_side, side_logit = torch.autograd.grad(isolated,
                (fused, future_raw), retain_graph=True)
            total = torch.autograd.grad(report['loss'], fused)[0]
            closure = (total - primary - auxiliary).detach()
            closure_max = float(closure.abs().max())
            scale = max(1.0, float(total.detach().abs().max()))
            side_logit_error = float((aux_logit[:, 0, 1] - side_logit[:, 0, 1]).abs().max())
            need(closure_max <= 1e-5 * scale and side_logit_error <= 1e-7
                 and bool(torch.isfinite(total).all())
                 and all(parameter.grad is None for parameter in model.parameters()),
                 'GRADIENT_CLOSURE')
            records.append({'batch_index': batch_index, 'first_parent_row': int(r[0]),
                'last_parent_row': int(r[-1]), 'first_issue_ns': int(issue[0]),
                'last_issue_ns': int(issue[-1]), 'native_F_M5_known_side': support,
                'decision_loss': float(report['decision_loss'].detach()),
                'auxiliary_loss_times_0p10': float((AUX_WEIGHT * report['aux_loss']).detach()),
                'native_F_M5_side_loss_times_0p10': float(isolated.detach()),
                'decision_vs_F_side': geometry(primary, f_side),
                'auxiliary_vs_F_side': geometry(auxiliary, f_side),
                'total_vs_F_side': geometry(total, f_side),
                'decision_vs_auxiliary': geometry(primary, auxiliary),
                'closure_max_abs': closure_max,
                'F_side_logit_max_abs_error': side_logit_error})
            del features, original, output, report, fused, primary, auxiliary, f_side, total
        need(len(records) == len(BATCH_INDICES) and source_clean(plan)
             and ref(REPO / 'NEXT_RUN_POLICY.json') == policy_binding
             and successor_input_refs() == inputs
             and replay.consumed_refs()[0] == plan['consumed_ema']
             and frozen_score_refs() == plan['frozen_fit_score']
             and canonical_model_state_sha256(model.state_dict()) == model_sha
             and all(parameter.grad is None for parameter in model.parameters())
             and time.monotonic() < deadline, 'BOOKEND')
        context['release']()
        context = None
        result_ref = publish('RESULT.json', {
            'status': 'COMPLETE_FROZEN_EMA_FUSED_GRADIENT_PROBE_NO_UPDATE',
            'plan': ref(PLAN), 'claim': claim, 'policy_at_claim': policy_binding,
            'source_commit': head, 'source_unchanged': True, 'guard': guard,
            'model_state_sha256': model_sha,
            'batch_count': len(records), 'model_forwards': len(records) + 1,
            'autograd_calls': 4 * len(records), 'optimizer_constructions': 0,
            'optimizer_updates': 0, 'parameter_gradients': 0,
            'gradient_scope': plan['gradient_scope'],
            **summarize_records(records),
            'batches': records, 'test_data_used': False,
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
            'policy': policy_binding, 'batch_count': len(BATCH_INDICES),
            'model_forwards': plan['model_forwards'],
            'autograd_calls': plan['autograd_calls'],
            'optimizer_updates': 0, 'output_root_absent': True,
            'claim_created': False, 'test_data_used': False}, sort_keys=True))
    else:
        run(args.plan_sha256)


if __name__ == '__main__':
    main()
