"""One guarded paired-S Fold2 fit with fixed full Apr-Jun development score."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time

from gx1.audit.market_state_ta_apr_jun_campaign_inputs_v1 import (
    REPO, SCORE_ROWS, TRAIN_ROWS, campaign_input_refs, need, prepare, read, ref,
)

STAGE = 'ta_integrated_apr_jun_paired_fit_20261117_002'
PLAN = REPO / 'configs/research/TA_INTEGRATED_APR_JUN_PAIRED_FIT_20261117_002.json'
ROOT = Path('/home/andre2/GX1_RUNS/TA_INTEGRATED_APR_JUN_PAIRED_FIT_20261117_002')
BATCH, SEED = 16, 20260911
BATCHES = (TRAIN_ROWS + BATCH - 1) // BATCH
SCORE_BATCHES = SCORE_ROWS // BATCH
SOURCES = ('gx1/audit/run_market_state_ta_apr_jun_paired_fit_v1.py',
           'gx1/audit/market_state_ta_apr_jun_campaign_inputs_v1.py',
           'gx1/contracts/market_state_ta_integrated_model_v1.py',
           'gx1/contracts/market_state_ta_integrated_paired_objective_v1.py',
           'gx1/contracts/market_state_ta_decoder_view_rebind_v1.py',
           'gx1/contracts/entry_native_direction_feature_view_v1.py',
           'gx1/contracts/entry_anchor_span_feature_inputs_v1.py',
           'gx1/contracts/market_state_time_query_binding_v1.py',
           'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
           'gx1/contracts/market_state_future_displacement_candidate_v1.py',
           'gx1/contracts/market_state_time_query_scorer_v2.py',
           'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
           'gx1/models/entry_v10/entry_v10_ctx_train_v3.py',
           'scripts/gx1_capped_run.sh', 'scripts/gx1_guarded_trainer_exec.sh')


def publish(name, value):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    raw = (value if isinstance(value, bytes) else
           (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
            + '\n').encode())
    _publish_bytes(ROOT / name, raw)
    return ref(ROOT / name)


def source_clean(plan):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    return (subprocess.check_output(['git', 'branch', '--show-current'],
                                    cwd=REPO, text=True).strip() == 'work/gx1-current'
            and subprocess.run(['git', 'merge-base', '--is-ancestor',
                                plan['source_base_commit'], head],
                               cwd=REPO, capture_output=True).returncode == 0
            and not subprocess.check_output(['git', 'status', '--porcelain'],
                                            cwd=REPO, text=True)
            and all(ref(REPO / name)['sha256'] == plan['source_owners'][name]
                    for name in SOURCES))


def physical_windows_boot_id():
    executable = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
    command = "(Get-ItemProperty -LiteralPath 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management\\PrefetchParameters' -Name BootId).BootId"
    value = subprocess.check_output([executable, '-NoProfile', '-Command', command],
                                    text=True, timeout=20).strip()
    need(value.isdecimal(), 'WINDOWS_BOOT_ID')
    return int(value)


def inspect(plan_sha, *, admitted):
    from gx1.contracts.market_state_ta_integrated_paired_objective_v1 import contract
    from gx1.audit.qualify_market_state_ta_apr_jun_paired_preflight_v1 import (
        STAGE as PREFLIGHT, bound_inputs,
    )
    need(ref(PLAN)['sha256'] == plan_sha, 'FIT_PLAN_SHA')
    plan = read(ref(PLAN))
    need(plan['schema_version'] == 'gx1_ta_integrated_apr_jun_paired_fit_plan_v1'
         and plan['stage'] == STAGE and plan['output_root'] == str(ROOT)
         and plan['seed'] == SEED and plan['fit_rows'] == TRAIN_ROWS
         and plan['score_rows'] == SCORE_ROWS and plan['batch_size'] == BATCH
         and plan['batch_visits'] == BATCHES
         and plan['initial_score_forwards'] == SCORE_BATCHES
         and plan['final_score_forwards'] == SCORE_BATCHES
         and plan['max_optimizer_updates'] == BATCHES
         and plan['max_seconds'] == 10500 and plan['test_data_used'] is False
         and plan['prior_physical_windows_boot_id'] == 658
         and plan['required_invocation_windows_boot_id_greater_than'] == 658
         and plan['objective_contract_sha256'] == contract()['contract_sha256']
         and set(plan['source_owners']) == set(SOURCES)
         and set(plan['inputs']) == set(bound_inputs()) | {
             'preflight_result', 'preflight_terminal'}
         and all(plan['inputs'][key] == value for key, value in bound_inputs().items())
         and all(plan['inputs'][key] == value for key, value in campaign_input_refs().items())
         and plan['inputs']['preflight_result'] == ref(
             '/home/andre2/GX1_RUNS/TA_INTEGRATED_APR_JUN_PAIRED_PREFLIGHT_20261117_001/RESULT.json')
         and plan['inputs']['preflight_terminal'] == ref(
             '/home/andre2/GX1_RUNS/TA_INTEGRATED_APR_JUN_PAIRED_PREFLIGHT_20261117_001/TERMINAL.json')
         and not ROOT.exists() and source_clean(plan), 'FIT_FIXED_SCOPE')
    preflight = read(plan['inputs']['preflight_result'])
    terminal = read(plan['inputs']['preflight_terminal'])
    need(preflight['status'] == 'PASS_ACTUAL_APR_JUN_PAIRED_BATCH_NO_FIT'
         and preflight['view_sha256'] == plan['view_sha256']
         and preflight['model_contract_sha256'] == plan['model_contract_sha256']
         and preflight['objective_contract_sha256'] == plan['objective_contract_sha256']
         and preflight['initial_model_state_sha256_CPU'] == plan['initial_model_state_sha256']
         and preflight['optimizer_steps'] == 0 and preflight['test_data_used'] is False
         and terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
         and terminal['result'] == plan['inputs']['preflight_result'],
         'FIT_PREFLIGHT_EVIDENCE')
    import numpy as np
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets, sampled_query_k
    issue_result = read(plan['inputs']['issue_result'])
    tape_ref = issue_result['input_archive']
    need(ref(tape_ref['path']) == tape_ref and issue_result['score_rows'] == SCORE_ROWS,
         'FIT_SCORE_TAPE_REF')
    with np.load(tape_ref['path'], allow_pickle=False) as tape:
        rows, issues = tape['parent_rows'], tape['issue_ns']
    need(len(rows) == len(issues) == SCORE_ROWS
         and bool(np.all((issues >= 1743465600000000000)
                         & (issues < 1751328000000000000))), 'FIT_SCORE_INTERVAL')
    sampled = sampled_query_k(issues, pass_index=0, seed=SEED)
    for bound, args in ((load_bound_time_query_targets(), (sampled,)),
                        (load_bound_five_endpoint_targets(), ()),
                        (load_bound_future_displacement(), ())):
        target = bound.gather(rows, issues, *args, fold_id=3, partition='score')
        need(target['allowed'].shape == (SCORE_ROWS, 5)
             and target['eligible'].shape == target['mature'].shape == (SCORE_ROWS, 5),
             'FIT_SCORE_TARGET_COVERAGE')
    policy = json.loads((REPO / 'NEXT_RUN_POLICY.json').read_text())
    old = policy[PREFLIGHT]
    need(old['status'] == 'COMPLETE_CONSUMED_ACTUAL_PAIRED_SCORE_BATCH_NO_FIT'
         and old['remaining_invocations'] == 0
         and old['result'] == plan['inputs']['preflight_result']
         and old['terminal'] == plan['inputs']['preflight_terminal'],
         'FIT_PREFLIGHT_CLOSED')
    gate = policy.get(STAGE, {})
    if admitted:
        need(physical_windows_boot_id() >
             plan['required_invocation_windows_boot_id_greater_than'],
             'FRESH_PHYSICAL_WINDOWS_BOOT')
        need(policy['training_enabled'] is policy['native_training_enabled'] is True
             and policy['full_epoch_training_allowed'] is True
             and policy['native_launch_authorized'] is True
             and policy['full_val_allowed'] is False
             and gate.get('status') == 'ADMITTED_ONE_USE'
             and gate.get('plan') == ref(PLAN)
             and gate.get('execution_authorized') is gate.get('training_authorized') is True
             and gate.get('remaining_invocations') == 1
             and gate.get('relaunch_allowed') is False
             and gate.get('max_optimizer_updates') == BATCHES
             and gate.get('model_forwards') == BATCHES + 2 * SCORE_BATCHES
             and gate.get('test_data_used') is False, 'FIT_POLICY_ADMISSION')
    else:
        need(policy['training_enabled'] is policy['native_training_enabled'] is False
             and policy['full_epoch_training_allowed'] is False
             and gate.get('execution_authorized') is not True,
             'FIT_TRAINING_CLOSED')
    return plan


def _proper(raw, truth, query, *, prior=None):
    import numpy as np
    from gx1.contracts.market_state_time_query_scorer_v2 import proper_score_sums
    report = proper_score_sums(raw, truth['allowed'], truth['eligible'],
                               truth['mature'], query)['per_tf']
    reference = (proper_score_sums(np.broadcast_to(prior[None], raw.shape).copy(),
                 truth['allowed'], truth['eligible'], truth['mature'], query)['per_tf']
                 if prior is not None else None)
    out = {}
    for tf, cells in report.items():
        out[tf] = {}
        for component, item in cells.items():
            n = item['count']
            prior_item = reference[tf][component] if reference is not None else None
            out[tf][component] = {'count': n,
                'brier': item['brier_sum'] / n if n else None,
                'prior_brier': prior_item['brier_sum'] / n if n and prior_item else None,
                **({'log': item['log_sum'] / n if n else None,
                    'prior_log': prior_item['log_sum'] / n if n and prior_item else None}
                   if 'log_sum' in item else {})}
    return out


def _turn_and_disagreement(state_raw, state_truth, issues):
    import numpy as np
    from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES
    known = state_truth['eligible'] & state_truth['mature'] & np.isin(
        state_truth['allowed'], (1, 4))
    p = 1.0 / (1.0 + np.exp(-np.clip(state_raw[:, :, 1].astype(np.float64), -80, 80)))
    paired = known[:-1, 0] & known[1:, 0] & (np.diff(issues) == 300_000_000_000)
    truth = state_truth['allowed'][:, 0]
    actual_turn = truth[:-1] != truth[1:]
    prediction = p[:, 0] >= 0.5
    predicted_turn = prediction[:-1] != prediction[1:]
    turn = {'known_contiguous_pairs': int(paired.sum()),
            'actual_turns': int((paired & actual_turn).sum()),
            'predicted_turns': int((paired & predicted_turn).sum()),
            'captured_actual_turns': int((paired & actual_turn & predicted_turn).sum()),
            'false_predicted_turns': int((paired & ~actual_turn & predicted_turn).sum())}
    tf = {}
    for j, name in enumerate(TIMEFRAMES[1:], 1):
        mask = known[:, 0] & known[:, j]
        different = mask & (state_truth['allowed'][:, 0] != state_truth['allowed'][:, j])
        opposing_predictions = (p[:, 0] >= 0.5) != (p[:, j] >= 0.5)
        tf[name] = {'joint_known': int(mask.sum()),
                    'opposite_truth': int(different.sum()),
                    'opposite_prediction_on_opposite_truth': int(
                        (different & opposing_predictions).sum()),
                    'mean_abs_up_probability_gap_on_opposite_truth': (
                        float(np.abs(p[different, 0] - p[different, j]).mean())
                        if different.any() else None)}
    return {'M5_turns': turn, 'TF_disagreement': tf}


def score(model, context, *, deadline, collect_decision=False):
    import numpy as np
    import torch
    from gx1.audit.observe_market_state_ta_fold2_fresh_initial_v1 import direction_readout
    from gx1.audit.run_market_state_ta_integrated_fold2_fit_v1 import structure_readout
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets, sampled_query_k
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
    axes, selected, tape = context['axes'], context['selected'], context['score_tape']
    rows, issues = axes['parent_rows'][selected], axes['decision_ns'][selected]
    sampled = sampled_query_k(issues, pass_index=0, seed=SEED)
    native = np.broadcast_to(np.asarray(NATIVE_K, dtype=np.int64), sampled.shape).copy()
    truth = {'sampled_S': load_bound_time_query_targets().gather(
                 rows, issues, sampled, fold_id=3, partition='score'),
             'native_S': load_bound_five_endpoint_targets().gather(
                 rows, issues, fold_id=3, partition='score'),
             'native_F': load_bound_future_displacement().gather(
                 rows, issues, fold_id=3, partition='score')}
    raw = {name: np.empty((SCORE_ROWS, 5, 2), dtype=np.float32)
           for name in ('sampled_S', 'native_S', 'native_F')}
    structure = np.empty((SCORE_ROWS, 2), dtype=np.float32)
    decision = np.empty((SCORE_ROWS, 6, 2, 3), dtype=np.float32) if collect_decision else None
    model.eval()
    with torch.inference_mode():
        for left in range(0, SCORE_ROWS, BATCH):
            need(time.monotonic() < deadline, 'SCORE_DEADLINE')
            right = left + BATCH
            features = context['features']('SCORE', rows[left:right], issues[left:right])
            grid = torch.from_numpy(np.stack((sampled[left:right], native[left:right]),
                                               axis=1).copy()).to(next(model.parameters()).device)
            out = model(features, query_k=grid,
                m1_path_bps=torch.from_numpy(tape['m1_mid_ohlc_bps'][left:right].copy()).to(grid.device),
                m1_observed=torch.from_numpy(tape['m1_observed'][left:right].copy()).to(grid.device))
            raw['sampled_S'][left:right] = out['state_activity_side_logits'].cpu().numpy()
            raw['native_S'][left:right] = out['state_native_activity_side_logits'].cpu().numpy()
            raw['native_F'][left:right] = out['future_activity_side_logits'].cpu().numpy()
            structure[left:right] = out['m5_structure_hit_logits'].cpu().numpy()
            if collect_decision:
                need(out['m5_cost_path_bps'].shape == (BATCH, 6, 2, 3), 'DECISION_SHAPE')
                decision[left:right] = out['m5_cost_path_bps'].cpu().numpy()
            del features, out
    prior = context['priors']['tasks']
    prior_raw = {name: np.asarray(prior[key]['raw_logits'], dtype=np.float32)
                 for name, key in (('native_S', 'state'), ('native_F', 'future'))}
    prior_hit = np.array([context['fit_tape']['structure_hit'][:, j].sum() /
                          context['fit_tape']['structure_observed'][:, j].sum()
                          for j in range(2)], dtype=np.float64)
    need(np.all((prior_hit > 0) & (prior_hit < 1)), 'STRUCTURE_PRIOR')
    readout = {'proper': {name: _proper(raw[name], truth[name],
                   sampled if name == 'sampled_S' else native,
                   prior=prior_raw.get(name)) for name in raw},
               'direction': {name: direction_readout(raw[name], truth[name], issues, np)
                             for name in raw},
               'structure': structure_readout(structure,
                    tape['structure_hit'], tape['structure_observed'], prior_hit),
               'turn_and_tf': _turn_and_disagreement(raw['native_S'], truth['native_S'], issues),
               'structure_prior': prior_hit.tolist(),
               'rows': SCORE_ROWS, 'model_forwards': SCORE_BATCHES,
               'score_start_ns': int(issues[0]), 'score_last_ns': int(issues[-1]),
               'test_data_used': False}
    capture = {'parent_rows': rows, 'issue_ns': issues,
               'sampled_query_k': sampled, 'native_query_k': native,
               'structure_raw': structure,
               'structure_hit': tape['structure_hit'],
               'structure_observed': tape['structure_observed']}
    if collect_decision:
        capture['m5_cost_path_bps'] = decision
    for name in raw:
        capture[name + '_raw'] = raw[name]
        for key in ('allowed', 'eligible', 'mature'):
            capture[name + '_' + key] = truth[name][key]
    return readout, capture


def run(plan_sha):
    import numpy as np
    import torch
    from gx1.contracts.entry_native_direction_feature_view_v1 import build_native_direction_research_model
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
    from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
    from gx1.contracts.market_state_ta_integrated_model_v1 import IntegratedTAMarketStateModel
    from gx1.contracts.market_state_ta_integrated_paired_objective_v1 import paired_integrated_loss
    from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets, sampled_query_k
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import ADAMW_DEFAULTS

    guard = require_guarded_cuda_trainer_execution()
    plan = inspect(plan_sha, admitted=True)
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
        need(context['view']['contract_sha256'] == plan['view_sha256']
             and all(context['input_refs'][key] == plan['inputs'][key]
                     for key in context['input_refs']), 'FIT_VIEW_AND_INPUTS')
        trainer._set_deterministic(SEED, device, trainer.DETERMINISTIC_FP32)
        state = build_native_direction_research_model(context['view'],
            prefix_normalization=context['packet'], constructor_metadata=context['constructor'],
            device=torch.device('cpu'), unobserved_units_profile=None)
        model = IntegratedTAMarketStateModel(NativeFutureDualModel(state)).to(device)
        initial_sha = canonical_model_state_sha256(model.state_dict())
        need(model.research_objective['contract_sha256'] == plan['model_contract_sha256']
             and initial_sha == plan['initial_model_state_sha256'], 'FRESH_INITIAL')
        initial, initial_capture = score(model, context, deadline=deadline)
        need(canonical_model_state_sha256(model.state_dict()) == initial_sha,
             'INITIAL_READOUT_MUTATION')
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), **ADAMW_DEFAULTS)
        need(not optimizer.state and len(optimizer.param_groups) == 1,
             'EMPTY_OPTIMIZER')
        optimizer.zero_grad(set_to_none=True)
        axes, tape = context['axes'], context['fit_tape']
        state_bound = load_bound_five_endpoint_targets()
        future_bound = load_bound_future_displacement()
        sampled_bound = load_bound_time_query_targets()
        blocks, updates, skipped = [], 0, 0
        current = {'first_batch': 0, 'loss': 0., 'sampled_S': 0.,
                   'native_S': 0., 'native_F': 0., 'structure': 0.,
                   'grad_norm': 0., 'updates': 0, 'skipped': 0}
        for index in range(BATCHES):
            need(time.monotonic() < deadline, 'FIT_DEADLINE')
            left, right = index * BATCH, min((index + 1) * BATCH, TRAIN_ROWS)
            rows, issues = axes['parent_rows'][left:right], axes['decision_ns'][left:right]
            features = context['features']('TRAIN', rows, issues)
            sampled = sampled_query_k(issues, pass_index=0, seed=SEED)
            native = np.broadcast_to(np.asarray(NATIVE_K, dtype=np.int64), sampled.shape)
            grid = torch.from_numpy(np.stack((sampled, native), axis=1).copy()).to(device)
            output = model(features, query_k=grid,
                m1_path_bps=torch.from_numpy(tape['m1_mid_ohlc_bps'][left:right].copy()).to(device),
                m1_observed=torch.from_numpy(tape['m1_observed'][left:right].copy()).to(device))
            sampled_truth = sampled_bound.gather_fit_sampled(
                rows, issues, fold_id=2, pass_index=0, seed=SEED)
            native_truth = state_bound.gather(rows, issues, fold_id=2, partition='fit')
            future_truth = future_bound.gather(rows, issues, fold_id=2, partition='fit')
            loss = paired_integrated_loss(model_output=output,
                sampled_truth=sampled_truth, native_truth=native_truth,
                future_truth=future_truth, query_grid=grid,
                structure_hit=tape['structure_hit'][left:right],
                structure_observed=tape['structure_observed'][left:right])
            need(loss['contract_sha256'] == plan['objective_contract_sha256']
                 and bool(torch.isfinite(loss['loss'])), 'FIT_LOSS')
            for source, key in (('loss', 'loss'), ('sampled_S', 'sampled_S_loss'),
                                ('native_S', 'native_S_loss'), ('native_F', 'native_F_loss'),
                                ('structure', 'structure_loss')):
                current[source] += float(loss[key].detach())
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
            if (index + 1) % 256 == 0 or index + 1 == BATCHES:
                visits = index + 1 - current['first_batch']
                blocks.append({'first_batch': current['first_batch'],
                               'last_batch_exclusive': index + 1,
                               'mean_loss': current['loss'] / visits,
                               'mean_sampled_S_loss': current['sampled_S'] / visits,
                               'mean_native_S_loss': current['native_S'] / visits,
                               'mean_native_F_loss': current['native_F'] / visits,
                               'mean_structure_loss': current['structure'] / visits,
                               'mean_gradient_norm_per_update': (
                                   current['grad_norm'] / current['updates']
                                   if current['updates'] else None),
                               'updates': current['updates'], 'skipped': current['skipped']})
                current = {'first_batch': index + 1, 'loss': 0., 'sampled_S': 0.,
                           'native_S': 0., 'native_F': 0., 'structure': 0.,
                           'grad_norm': 0., 'updates': 0, 'skipped': 0}
        need(updates + skipped == BATCHES, 'FIT_VISITS')
        final, final_capture = score(model, context, deadline=deadline)
        final_sha = canonical_model_state_sha256(model.state_dict())
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint_ref = publish('FINAL_MODEL.pt', checkpoint.getvalue())
        captures = {}
        for phase, capture in (('INITIAL', initial_capture), ('FINAL', final_capture)):
            stream = io.BytesIO()
            np.savez_compressed(stream, **capture)
            captures[phase] = publish(phase + '_SCORE_READOUT.npz', stream.getvalue())
        need(source_clean(plan)
             and subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                         cwd=REPO, text=True).strip() == head
             and time.monotonic() < deadline, 'SOURCE_TIME_BOOKEND')
        context['release']()
        context = None
        result = publish('RESULT.json', {'status': 'COMPLETE_ONE_PASS_PAIRED_S_F_STRUCTURE_APR_JUN_SCORE',
            'plan': ref(PLAN), 'claim': claim, 'source_commit': head,
            'source_unchanged': True, 'guard': guard,
            'initial_model_state_sha256': initial_sha,
            'final_model_state_sha256': final_sha,
            'fit_rows': TRAIN_ROWS, 'batch_visits': BATCHES,
            'optimizer_updates': updates, 'skipped_batches': skipped,
            'training_forwards': BATCHES,
            'evaluation_forwards': 2 * SCORE_BATCHES,
            'fit_blocks': blocks, 'initial': initial, 'final': final,
            'checkpoint': checkpoint_ref, 'captures': captures,
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


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    need(args.plan == str(PLAN) and args.device == 'cuda'
         and args.out_dir == str(ROOT)
         and len(args.plan_sha256) == 64, 'CLI')
    if args.preflight:
        plan = inspect(args.plan_sha256, admitted=False)
        print(json.dumps({'status': 'READY_SOURCE_AND_RECEIPTS_NO_CLAIM',
                          'stage': plan['stage'], 'fit_rows': TRAIN_ROWS,
                          'score_rows': SCORE_ROWS, 'model_forwards': 0,
                          'optimizer_steps': 0, 'test_data_used': False}))
    else:
        run(args.plan_sha256)


if __name__ == '__main__':
    main()
