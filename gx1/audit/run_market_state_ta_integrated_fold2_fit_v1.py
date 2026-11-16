"""One guarded Fold2 full TRAIN pass and fixed initial/final development score."""
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

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/TA_INTEGRATED_FOLD2_FIT_20261116_001')
STAGE = 'ta_integrated_fold2_fit_20261116_001'
TRAIN_ROWS = 47338
BATCH = 16
BATCHES = 2959
PANEL = 4096
SEED = 20260911
MODEL_SHA = '70a0497b27ba0ab45277ba622443ab2fca36a448357f1ccb3158689193f8bc82'
OBJECTIVE_SHA = 'f4cd885a7994f66367abc54fcbdedbe5339275ef8404dee29362b38bf2819bd1'
VIEW_SHA = '8056fe96b4340bccc149da7f22dde85e4ac1b62c5e58c874ec7fbb771c1dff1a'
INITIAL_CPU_SHA = '0a04141985c261b82414b5020937a65b0905aa3fb7d441aa9a4ea45bebb27525'
SOURCES = (
    'gx1/audit/run_market_state_ta_integrated_fold2_fit_v1.py',
    'gx1/scripts/run_market_state_ta_fold2_m5_fit_window_v1.py',
    'gx1/contracts/market_state_ta_decoder_view_rebind_v1.py',
    'gx1/contracts/market_state_ta_integrated_model_v1.py',
    'gx1/contracts/market_state_ta_integrated_objective_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/entry_anchor_span_feature_inputs_v1.py',
    'scripts/gx1_capped_run.sh',
    'scripts/gx1_guarded_trainer_exec.sh',
)


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_INTEGRATED_FIT_' + reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_ref(ref):
    need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'REF_FIELDS')
    path = Path(ref['path'])
    need(path.is_absolute() and path.is_file() and not path.is_symlink()
         and sha(path) == ref['sha256'], 'REF_BYTES')
    return json.loads(path.read_text())


def publish(name, value):
    blob = value if isinstance(value, bytes) else (
        json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n')
    path = ROOT / name
    with path.open('xb') as out:
        out.write(blob)
        out.flush()
        os.fsync(out.fileno())
    return {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest()}


def source_clean(plan):
    head = subprocess.check_output(('git', 'rev-parse', 'HEAD'), cwd=REPO, text=True).strip()
    return (subprocess.check_output(('git', 'branch', '--show-current'), cwd=REPO, text=True).strip()
            == 'work/gx1-current'
            and subprocess.run(('git', 'merge-base', '--is-ancestor', plan['source_base_commit'], head),
                               cwd=REPO, capture_output=True).returncode == 0
            and not subprocess.check_output(('git', 'status', '--porcelain'), cwd=REPO, text=True)
            and all(sha(REPO / path) == plan['sources'][path]['sha256'] for path in SOURCES))


def preflight(plan_path, expected_sha, *, admitted):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import contract as loss_contract
    from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import INITIAL_PLAN_REF, require_reused_input_refs

    need(plan_path == ROOT / 'PLAN.json' and plan_path.is_file() and not plan_path.is_symlink()
         and sha(plan_path) == expected_sha, 'PLAN_SHA')
    plan = json.loads(plan_path.read_text())
    need(set(plan) == {'schema_version', 'stage', 'source_base_commit', 'sources',
                       'inputs', 'output_root', 'max_seconds', 'fit_rows', 'batch_size',
                       'batch_visits', 'panel_rows', 'evaluation_forwards',
                       'max_optimizer_updates', 'model_contract_sha256',
                       'objective_contract_sha256', 'view_sha256',
                       'initial_model_state_sha256', 'seed', 'test_data_used'}
         and plan['schema_version'] == 'gx1_ta_integrated_fold2_fit_plan_v1'
         and plan['stage'] == STAGE and plan['output_root'] == str(ROOT)
         and plan['max_seconds'] == 10500 and plan['fit_rows'] == TRAIN_ROWS
         and plan['batch_size'] == BATCH and plan['batch_visits'] == BATCHES
         and plan['panel_rows'] == PANEL and plan['evaluation_forwards'] == 1024
         and plan['max_optimizer_updates'] == BATCHES
         and plan['model_contract_sha256'] == MODEL_SHA
         and plan['objective_contract_sha256'] == OBJECTIVE_SHA
         and plan['view_sha256'] == VIEW_SHA
         and plan['initial_model_state_sha256'] == INITIAL_CPU_SHA
         and plan['seed'] == SEED and plan['test_data_used'] is False
         and set(plan['sources']) == set(SOURCES)
         and set(plan['inputs']) == {'fit_review', 'later_review', 'objective_review',
                                     'initial_plan', 'design'}, 'PLAN_SCOPE')
    for path in SOURCES:
        need(plan['sources'][path] == {'path': str(REPO / path),
             'sha256': sha(REPO / path)}, 'SOURCE_' + path)
    need(source_clean(plan), 'SOURCE_DIRTY')
    inputs = {key: read_ref(value) for key, value in plan['inputs'].items()}
    need(plan['inputs']['initial_plan'] == INITIAL_PLAN_REF
         and inputs['fit_review']['fit_axis']['rows'] == TRAIN_ROWS
         and inputs['later_review']['later_development_panel']['score_rows'] == PANEL
         and inputs['objective_review']['objective_contract_sha256'] == OBJECTIVE_SHA
         and inputs['objective_review']['model_contract_sha256'] == MODEL_SHA
         and inputs['objective_review']['view_sha256'] == VIEW_SHA
         and inputs['design']['fresh_initial_design']['panels']['train']['selected_rows'] == PANEL
         and inputs['design']['fresh_initial_design']['panels']['later']['selected_rows'] == PANEL
         and inputs['design']['fresh_initial_design']['panels']['train']['positions_sha256_le_int64']
             == 'c3dda8026de191c65defd9935e508af178272690ad2d558610ccd02ed5c60f42'
         and inputs['design']['fresh_initial_design']['panels']['later']['positions_sha256_le_int64']
             == 'f9e7ab8dfdcc76a43cb9260ba6bae72118794e8922b735deffe5b6a9ceb067c7'
         and inputs['fit_review']['fit_axis']['issue_axis_sha256']
             == '380184a9c95a02f2716326765783746b1834371a7266545113fb1eae6722da47'
         and inputs['later_review']['later_development_panel']['issue_axis_sha256']
             == 'c03ec2ae8b418e21739f9f7966b946615878344364e6dc4e4d80087a0fee99d3',
         'INPUT_DESIGN')
    import numpy as np
    for key, ref, rows, issue_sha in (
        ('fit_review', inputs['fit_review']['input_archive'], TRAIN_ROWS,
         inputs['fit_review']['fit_axis']['issue_axis_sha256']),
        ('later_review', inputs['later_review']['later_development_panel']['private_archive'], PANEL,
         inputs['later_review']['later_development_panel']['issue_axis_sha256']),
    ):
        path = Path(ref['path'])
        need(path.is_file() and not path.is_symlink() and sha(path) == ref['sha256'],
             'INPUT_TAPE_' + key)
        with np.load(path, allow_pickle=False) as archive:
            parent, issues = archive['parent_rows'], archive['issue_ns']
        tape = load_tape(ref, rows, parent, issues)
        need(hashlib.sha256(issues.astype('<i8').tobytes()).hexdigest() == issue_sha
             and bool(np.all(np.diff(parent) > 0)) and bool(np.all(np.diff(issues) > 0))
             and bool(np.all(~tape['structure_hit'] | tape['structure_observed'])),
             'INPUT_TAPE_CONTENT_' + key)
    require_reused_input_refs(INITIAL_PLAN_REF)
    need(loss_contract()['contract_sha256'] == OBJECTIVE_SHA
         and not any((ROOT / name).exists() for name in ('CLAIM.json', 'RESULT.json', 'TERMINAL.json')),
         'ONE_USE_AND_LOSS')
    policy = json.loads((REPO / 'NEXT_RUN_POLICY.json').read_text())
    gate = policy.get(STAGE, {})
    if admitted:
        need(policy['training_enabled'] is policy['native_training_enabled'] is True
             and policy['full_epoch_training_allowed'] is True
             and gate.get('status') == 'ADMITTED_ONE_USE'
             and gate.get('plan') == {'path': str(plan_path), 'sha256': expected_sha}
             and gate.get('remaining_invocations') == 1
             and gate.get('execution_authorized') is gate.get('training_authorized') is True
             and gate.get('max_optimizer_updates') == BATCHES
             and gate.get('model_forwards') == BATCHES + 1024
             and gate.get('test_data_used') is False, 'POLICY_ADMISSION')
    else:
        need(policy['training_enabled'] is policy['native_training_enabled'] is False
             and policy['full_epoch_training_allowed'] is False
             and gate.get('execution_authorized') is not True, 'TRAINING_CLOSED')
    return plan, inputs


def load_tape(ref, expected_rows, parents, issues):
    import numpy as np
    need(sha(ref['path']) == ref['sha256'], 'TAPE_SHA')
    with np.load(ref['path'], allow_pickle=False) as archive:
        need(set(archive.files) == {'parent_rows', 'issue_ns', 'm1_mid_ohlc_bps',
                                    'm1_observed', 'structure_hit', 'structure_observed'},
             'TAPE_KEYS')
        tape = {key: archive[key] for key in archive.files}
    need(np.array_equal(tape['parent_rows'], parents)
         and np.array_equal(tape['issue_ns'], issues)
         and len(parents) == expected_rows
         and tape['parent_rows'].dtype == tape['issue_ns'].dtype == np.int64
         and tape['m1_mid_ohlc_bps'].dtype == np.float32
         and tape['m1_mid_ohlc_bps'].shape == (expected_rows, 5, 4)
         and tape['m1_observed'].dtype == np.bool_
         and tape['m1_observed'].shape == (expected_rows, 5)
         and tape['structure_hit'].dtype == tape['structure_observed'].dtype == np.bool_
         and tape['structure_hit'].shape == tape['structure_observed'].shape == (expected_rows, 2),
         'TAPE_AXIS_AND_SHAPES')
    return tape


def structure_readout(raw, hit, observed, prior):
    import numpy as np
    from gx1.audit.observe_market_state_ta_fold2_fresh_initial_v1 import auc_with_ties
    out = {}
    for j, side in enumerate(('high', 'low')):
        mask = observed[:, j]
        target = hit[mask, j]
        probability = 1.0 / (1.0 + np.exp(-np.clip(raw[mask, j].astype(np.float64), -80., 80.)))
        p0 = float(prior[j])
        out[side] = {'observed': int(mask.sum()), 'hit': int(target.sum()),
                     'brier': float(np.square(probability - target).mean()) if len(target) else None,
                     'prior_brier': float(np.square(p0 - target).mean()) if len(target) else None,
                     'auc': auc_with_ties(probability, target, np),
                     'mean_probability': float(probability.mean()) if len(target) else None}
    return out


def score_panels(model, context, fit_tape, later_tape, prior, deadline):
    import numpy as np
    import torch
    from gx1.audit.observe_market_state_ta_fold2_fresh_initial_v1 import direction_readout
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import (
        endpoint_measurement_sums, merge_endpoint_measurements, finalize_endpoint_measurement,
    )
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K

    model.eval()
    panels, captures = {}, {}
    axes = context['axes']
    for name, positions, tape, tape_positions, partition in (
        ('TRAIN', context['train_positions'], fit_tape, context['train_positions'], 'fit'),
        ('LATER', context['later_positions'], later_tape,
         np.arange(PANEL, dtype=np.int64), 'score')):
        rows, issues = axes['parent_rows'][positions], axes['decision_ns'][positions]
        state = context['state_bound'].gather(rows, issues, fold_id=2, partition=partition)
        future = context['future_bound'].gather(rows, issues, fold_id=2, partition=partition)
        raw = {key: np.empty((PANEL, 5, 2), dtype=np.float32)
               for key in ('state', 'future')}
        structure = np.empty((PANEL, 2), dtype=np.float32)
        for left in range(0, PANEL, BATCH):
            need(time.monotonic() < deadline, 'SCORE_DEADLINE')
            right = left + BATCH
            selected = tape_positions[left:right]
            features = context['features'](name, rows[left:right], issues[left:right])
            query = torch.tensor([NATIVE_K] * BATCH, dtype=torch.int64,
                                 device=model.dual.state_model.query_native_k.device)
            with torch.no_grad():
                out = model(features, query_k=query,
                    m1_path_bps=torch.from_numpy(tape['m1_mid_ohlc_bps'][selected]).to(query.device),
                    m1_observed=torch.from_numpy(tape['m1_observed'][selected]).to(query.device))
            for task in raw:
                raw[task][left:right] = out[task + '_activity_side_logits'].detach().cpu().numpy()
            structure[left:right] = out['m5_structure_hit_logits'].detach().cpu().numpy()
            del features, out
        measurement = endpoint_measurement_sums(
            state_raw=raw['state'], future_raw=raw['future'],
            state_truth=state, future_truth=future, parent_rows=rows,
            issue_ns=issues, fold_id=2, partition=partition,
            future_weight=1.0, model_contract_sha256=MODEL_SHA,
            feature_view_sha256=VIEW_SHA, frozen_priors=context['priors'],
            ta_only=True, dedicated_m5_side=True,
            dedicated_non_m5_side=True, dedicated_tf_activity=True,
            direct_tf_hidden=True, m5_equal_observation=True)
        measured = finalize_endpoint_measurement(merge_endpoint_measurements(None, measurement))
        from gx1.contracts.market_state_ta_integrated_objective_v1 import contract as loss_contract
        need(measured['loss_contract_sha256'] == loss_contract()['S_F_loss_contract_sha256'],
             'SCORE_OBJECTIVE_BINDING')
        hit = tape['structure_hit'][tape_positions]
        observed = tape['structure_observed'][tape_positions]
        panels[name] = {'S_F': measured,
                        'direction': {task: direction_readout(raw[task],
                                      state if task == 'state' else future, issues, np)
                                      for task in raw},
                        'structure': structure_readout(structure, hit, observed, prior)}
        captures[name] = {'parent_rows': rows, 'issue_ns': issues,
                          'state_raw': raw['state'], 'future_raw': raw['future'],
                          'structure_raw': structure, 'structure_hit': hit,
                          'structure_observed': observed,
                          **{task + '_' + key: truth[key]
                             for task, truth in (('state', state), ('future', future))
                             for key in ('allowed', 'eligible', 'mature')}}
    return panels, captures


def run(plan_path, plan_sha):
    import numpy as np
    import torch
    from gx1.contracts.entry_native_direction_feature_view_v1 import build_native_direction_research_model
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import ADAMW_DEFAULTS, INITIAL_PLAN_REF
    from gx1.contracts.market_state_ta_integrated_model_v1 import IntegratedTAMarketStateModel
    from gx1.contracts.market_state_ta_integrated_objective_v1 import integrated_loss
    from gx1.contracts.market_state_time_query_binding_v1 import sampled_query_k
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import require_m5_equal_observation_axes
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts.run_market_state_ta_fold2_m5_fit_window_v1 import prepare_qualified_inputs

    guard = require_guarded_cuda_trainer_execution()
    plan, inputs = preflight(plan_path, plan_sha, admitted=True)
    launch_head = subprocess.check_output(('git', 'rev-parse', 'HEAD'),
                                          cwd=REPO, text=True).strip()
    claim = publish('CLAIM.json', {'stage': STAGE, 'plan_sha256': plan_sha,
                                  'source_commit': launch_head,
                                  'test_data_used': False})
    started = time.monotonic()
    deadline = started + plan['max_seconds'] - 30.
    context = None
    try:
        device = torch.device('cuda', 0)
        context = prepare_qualified_inputs(initial_plan_ref=INITIAL_PLAN_REF,
            design=inputs['design']['fresh_initial_design'], device=device,
            integrated_direct_tf=True)
        need(context['view']['contract_sha256'] == VIEW_SHA, 'VIEW')
        axes = context['axes']
        require_m5_equal_observation_axes(axes['parent_rows'][:TRAIN_ROWS],
                                           axes['decision_ns'][:TRAIN_ROWS], fold_id=2)
        fit_ref = inputs['fit_review']['input_archive']
        later_ref = inputs['later_review']['later_development_panel']['private_archive']
        fit_tape = load_tape(fit_ref, TRAIN_ROWS,
            axes['parent_rows'][:TRAIN_ROWS], axes['decision_ns'][:TRAIN_ROWS])
        selected = context['later_positions']
        later_tape = load_tape(later_ref, PANEL,
            axes['parent_rows'][selected], axes['decision_ns'][selected])
        prior = np.array([fit_tape['structure_hit'][:, j].sum() /
                          fit_tape['structure_observed'][:, j].sum()
                          for j in range(2)], dtype=np.float64)
        need(bool(np.isfinite(prior).all()) and bool(np.all((prior > 0) & (prior < 1))),
             'STRUCTURE_PRIOR')
        trainer._set_deterministic(SEED, device, trainer.DETERMINISTIC_FP32)
        state = build_native_direction_research_model(context['view'],
            prefix_normalization=context['packet'],
            constructor_metadata=context['constructor'],
            device=torch.device('cpu'), unobserved_units_profile=None)
        model = IntegratedTAMarketStateModel(NativeFutureDualModel(state)).to(device)
        need(model.research_objective['contract_sha256'] == MODEL_SHA
             and canonical_model_state_sha256(model.state_dict()) == INITIAL_CPU_SHA,
             'FRESH_MODEL_INITIAL')
        initial, initial_captures = score_panels(model, context, fit_tape, later_tape,
                                                 prior, deadline)
        initial_model_sha = canonical_model_state_sha256(model.state_dict())
        need(initial_model_sha == INITIAL_CPU_SHA, 'INITIAL_SCORE_MUTATION')
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), **ADAMW_DEFAULTS)
        need(not optimizer.state and len(optimizer.param_groups) == 1, 'FRESH_OPTIMIZER')
        optimizer.zero_grad(set_to_none=True)
        blocks = []
        current = {'first_batch': 0, 'sum_loss': 0., 'sum_S_F': 0.,
                   'sum_structure': 0., 'sum_grad_norm': 0.,
                   'updates': 0, 'skipped': 0}
        updates = skipped = 0
        for index in range(BATCHES):
            need(time.monotonic() < deadline, 'FIT_DEADLINE')
            left, right = index * BATCH, min((index + 1) * BATCH, TRAIN_ROWS)
            rows, issues = axes['parent_rows'][left:right], axes['decision_ns'][left:right]
            features = context['features']('TRAIN', rows, issues)
            query = torch.from_numpy(sampled_query_k(issues, pass_index=0,
                seed=SEED)).to(device)
            output = model(features, query_k=query,
                m1_path_bps=torch.from_numpy(fit_tape['m1_mid_ohlc_bps'][left:right]).to(device),
                m1_observed=torch.from_numpy(fit_tape['m1_observed'][left:right]).to(device))
            state_truth = context['state_fit_bound'].gather_fit_sampled(
                rows, issues, fold_id=2, pass_index=0, seed=SEED)
            future_truth = context['future_bound'].gather(
                rows, issues, fold_id=2, partition='fit')
            loss = integrated_loss(model_output=output,
                state_truth=state_truth, future_truth=future_truth,
                state_query_k=query,
                structure_hit=fit_tape['structure_hit'][left:right],
                structure_observed=fit_tape['structure_observed'][left:right])
            need(loss['contract_sha256'] == OBJECTIVE_SHA
                 and bool(torch.isfinite(loss['loss'])), 'FIT_LOSS')
            current['sum_loss'] += float(loss['loss'].detach())
            current['sum_S_F'] += float(loss['S_F_loss'].detach())
            current['sum_structure'] += float(loss['structure_loss'].detach())
            if loss['skip_update']:
                skipped += 1
                current['skipped'] += 1
            else:
                loss['loss'].backward()
                norm = float(trainer._optimizer_step_with_finite_gradients(
                    model=model, optimizer=optimizer))
                need(math.isfinite(norm), 'GRADIENT_NORM')
                current['sum_grad_norm'] += norm
                updates += 1
                current['updates'] += 1
            optimizer.zero_grad(set_to_none=True)
            del features, output, loss
            if (index + 1) % 256 == 0 or index + 1 == BATCHES:
                visits = index + 1 - current['first_batch']
                blocks.append({'first_batch': current['first_batch'],
                               'last_batch_exclusive': index + 1,
                               'mean_loss': current['sum_loss'] / visits,
                               'mean_S_F_loss': current['sum_S_F'] / visits,
                               'mean_structure_loss': current['sum_structure'] / visits,
                               'mean_gradient_norm_per_update': (
                                   current['sum_grad_norm'] / current['updates']
                                   if current['updates'] else None),
                               'updates': current['updates'], 'skipped': current['skipped']})
                current = {'first_batch': index + 1, 'sum_loss': 0., 'sum_S_F': 0.,
                           'sum_structure': 0., 'sum_grad_norm': 0.,
                           'updates': 0, 'skipped': 0}
        need(updates + skipped == BATCHES, 'FIT_VISITS')
        model.eval()
        final, final_captures = score_panels(model, context, fit_tape, later_tape,
                                             prior, deadline)
        final_model_sha = canonical_model_state_sha256(model.state_dict())
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint_ref = publish('FINAL_MODEL.pt', checkpoint.getvalue())
        captures = {}
        for phase, values in (('INITIAL', initial_captures), ('FINAL', final_captures)):
            for panel, arrays in values.items():
                stream = io.BytesIO()
                np.savez_compressed(stream, **arrays)
                captures[phase + '_' + panel] = publish(phase + '_' + panel + '_READOUT.npz',
                                                       stream.getvalue())
        need(source_clean(plan)
             and subprocess.check_output(('git', 'rev-parse', 'HEAD'),
                                         cwd=REPO, text=True).strip() == launch_head
             and time.monotonic() < deadline,
             'SOURCE_AND_TIME_BOOKEND')
        context['release']()
        context = None
        result = publish('RESULT.json', {
            'status': 'COMPLETE_ONE_PASS_INTEGRATED_TA_FIT_AND_FIXED_SCORE',
            'plan': {'path': str(plan_path), 'sha256': plan_sha},
            'claim': claim, 'guard': guard,
            'source_commit': launch_head, 'source_unchanged': True,
            'initial_model_state_sha256': initial_model_sha,
            'final_model_state_sha256': final_model_sha,
            'fit_rows': TRAIN_ROWS, 'batch_visits': BATCHES,
            'optimizer_updates': updates, 'skipped_batches': skipped,
            'evaluation_forwards': 1024, 'training_forwards': BATCHES,
            'checkpoint': checkpoint_ref, 'captures': captures,
            'fit_blocks': blocks, 'initial': initial, 'final': final,
            'structure_prior': prior.tolist(), 'test_data_used': False,
            'elapsed_seconds': time.monotonic() - started,
            'economic_edge_proven': False})
        publish('TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'plan': {'path': str(plan_path), 'sha256': plan_sha},
            'result': result, 'source_unchanged': True, 'test_data_used': False})
    except BaseException as error:
        publish('TERMINAL.json', {'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'plan': {'path': str(plan_path), 'sha256': plan_sha},
            'error': type(error).__name__ + ':' + str(error), 'test_data_used': False})
        raise
    finally:
        if context is not None:
            context['release']()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    need(args.device == 'cuda' and args.out_dir == str(ROOT), 'CLI')
    path = Path(args.plan)
    if args.preflight:
        plan, _ = preflight(path, args.plan_sha256, admitted=False)
        print(json.dumps({'status': 'PASS_SOURCE_INPUT_AND_POLICY_PREFLIGHT',
                          'stage': STAGE, 'source_base_commit': plan['source_base_commit'],
                          'model_forwards': 0, 'optimizer_updates': 0,
                          'claim_created': False, 'test_data_used': False}))
    else:
        run(path, args.plan_sha256)


if __name__ == '__main__':
    main()
