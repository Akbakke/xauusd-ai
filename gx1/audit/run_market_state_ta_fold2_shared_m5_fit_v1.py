"""One source-bound fold-2 TRAIN pass for the shared S/F M5 side candidate."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_TA_FOLD2_SHARED_M5_FIT_20261114_001')
REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'market_state_ta_fold2_shared_m5_fit_20261114_001'
SOURCES = (
    'gx1/audit/run_market_state_ta_fold2_shared_m5_fit_v1.py',
    'gx1/scripts/run_market_state_ta_fold2_m5_fit_window_v1.py',
    'gx1/contracts/market_state_ta_fold2_m5_fit_v1.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/market_state_time_query_native_F_batch_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/audit/score_market_state_ta_fold2_m5_fit_v1.py',
    'scripts/gx1_capped_run.sh',
    'scripts/gx1_guarded_trainer_exec.sh',
)


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('FOLD2_SHARED_M5_FIT_' + reason)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ref(ref: dict) -> dict:
    need(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'REF_FIELDS')
    path = Path(ref['path'])
    need(path.is_absolute() and path.is_file() and not path.is_symlink()
         and sha(path) == ref['sha256'], 'REF_BYTES')
    return json.loads(path.read_text())


def publish(name: str, value: dict | bytes) -> dict:
    path = ROOT / name
    blob = value if isinstance(value, bytes) else (
        json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n')
    with path.open('xb') as out:
        out.write(blob)
        out.flush()
        os.fsync(out.fileno())
    return {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest()}


def source_clean(plan: dict) -> bool:
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_commit'], head],
                      cwd=REPO, capture_output=True).returncode != 0:
        return False
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO):
        return False
    return all(sha(REPO / path) == plan['source_owners'][path]['sha256']
               for path in SOURCES)


def preflight(plan_path: Path, expected_sha: str, *, require_admission: bool) -> dict:
    from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import (
        BATCH_VISITS, INITIAL_PLAN_REF, TRAIN_ROWS, require_reused_input_refs,
    )
    from gx1.contracts.market_state_time_query_native_F_model_v1 import contract as model_contract
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import contract as loss_contract

    need(plan_path == ROOT / 'PLAN.json' and plan_path.is_file()
         and not plan_path.is_symlink() and sha(plan_path) == expected_sha,
         'PLAN_BINDING')
    plan = json.loads(plan_path.read_text())
    expected = {'schema_version', 'stage', 'source_commit', 'source_owners',
                'source_review', 'frozen_design', 'old_initial_plan',
                'shared_initial_result', 'shared_initial_review', 'output_root',
                'max_seconds', 'train_rows', 'batch_visits', 'max_optimizer_updates',
                'final_panel_forwards', 'checkpoint_writes', 'test_data_used',
                'model_contract_sha256', 'loss_contract_sha256',
                'initial_model_state_sha256'}
    need(set(plan) == expected and plan['schema_version'] == 'gx1_ta_fold2_shared_m5_fit_plan_v1'
         and plan['stage'] == STAGE and plan['output_root'] == str(ROOT)
         and plan['max_seconds'] == 10500 and plan['train_rows'] == TRAIN_ROWS
         and plan['batch_visits'] == BATCH_VISITS
         and plan['max_optimizer_updates'] == BATCH_VISITS
         and plan['final_panel_forwards'] == 512 and plan['checkpoint_writes'] == 0
         and plan['test_data_used'] is False and plan['old_initial_plan'] == INITIAL_PLAN_REF
         and set(plan['source_owners']) == set(SOURCES), 'PLAN_SCOPE')
    for path in SOURCES:
        need(plan['source_owners'][path] == {'path': str(REPO / path),
             'sha256': sha(REPO / path)}, 'SOURCE_' + path)
    need(source_clean(plan), 'SOURCE_DIRTY')
    review = read_ref(plan['source_review'])
    design = read_ref(plan['frozen_design'])
    initial = read_ref(plan['shared_initial_result'])
    baseline_review = read_ref(plan['shared_initial_review'])
    need(review['status'] == 'SOURCE_QUALIFIED_PHYSICAL_BASELINE_PENDING'
         and design['schema_version'] == 'gx1_ta_fold2_m5_direction_fit_frozen_design_v1'
         and initial['status'] == 'COMPLETE_ZERO_UPDATE_SHARED_M5_INITIAL'
         and initial['model_state_sha256'] == plan['initial_model_state_sha256']
         and initial['model_constructions'] == 1
         and initial['evaluation_forwards'] == 512
         and initial['optimizer_updates'] == 0 and initial['test_data_used'] is False
         and baseline_review['status'] == 'PHYSICAL_ZERO_UPDATE_BASELINE_COMPLETE_EDGE_UNQUALIFIED'
         and baseline_review['result'] == plan['shared_initial_result']
         and baseline_review['model_state_sha256'] == plan['initial_model_state_sha256'],
         'REUSED_EVIDENCE')
    require_reused_input_refs(INITIAL_PLAN_REF)
    mode = dict(ta_only=True, dedicated_m5_side=True,
                dedicated_non_m5_side=False, dedicated_tf_activity=True,
                shared_m5_side=True)
    need(model_contract(**mode)['contract_sha256'] == plan['model_contract_sha256']
         and loss_contract(**mode, m5_equal_observation=True,
                           m5_direction_only=True, fold_id=2)['contract_sha256']
             == plan['loss_contract_sha256'], 'MODEL_LOSS_CONTRACT')
    need(not any((ROOT / name).exists() for name in ('CLAIM.json', 'RESULT.json', 'TERMINAL.json')),
         'ALREADY_CLAIMED')
    if require_admission:
        policy = json.loads((REPO / 'NEXT_RUN_POLICY.json').read_text())
        gate = policy.get(STAGE, {})
        need(policy['training_enabled'] is True and policy['native_training_enabled'] is True
             and policy['full_epoch_training_allowed'] is True
             and gate.get('status') == 'ADMITTED'
             and gate.get('execution_authorized') is True
             and gate.get('training_authorized') is True
             and gate.get('data_reads_authorized') is True
             and gate.get('model_constructions_authorized') == 1
             and gate.get('model_forwards_authorized') == 512 + BATCH_VISITS
             and gate.get('optimizer_updates_authorized') == BATCH_VISITS
             and gate.get('test_data_used') is False
             and gate.get('plan') == {'path': str(plan_path), 'sha256': expected_sha},
             'POLICY_ADMISSION')
    return {'plan': plan, 'design': design, 'initial': initial}


def preflight_actual_inputs(bound: dict) -> None:
    import torch
    from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import INITIAL_PLAN_REF, TRAIN_ROWS
    from gx1.scripts.run_market_state_ta_fold2_m5_fit_window_v1 import prepare_qualified_inputs

    context = prepare_qualified_inputs(initial_plan_ref=INITIAL_PLAN_REF,
        design=bound['design']['fresh_initial_design'], device=torch.device('cuda', 0))
    try:
        need(context['view']['prefix']['rows'] == TRAIN_ROWS
             and context['view']['contract_sha256']
                 == bound['initial']['measurements']['TRAIN']['feature_view_sha256'],
             'ACTUAL_VIEW_IDENTITY')
    finally:
        context['release']()


def run(plan_path: Path, plan_sha: str) -> None:
    import numpy as np
    import torch
    from gx1.audit.score_market_state_ta_fold2_m5_fit_v1 import read_capture, paired_later_f_m5
    from gx1.contracts.entry_native_direction_feature_view_v1 import build_native_direction_research_model
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import (
        ADAMW_DEFAULTS, BATCH_VISITS, INITIAL_PLAN_REF, TRAIN_ROWS,
    )
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts.run_market_state_ta_fold2_m5_fit_window_v1 import (
        evaluate_fixed_panels, fit_one_pass, prepare_qualified_inputs,
    )

    proof = require_guarded_cuda_trainer_execution()
    bound = preflight(plan_path, plan_sha, require_admission=True)
    plan = bound['plan']
    claim = publish('CLAIM.json', {'stage': STAGE, 'plan_sha256': plan_sha,
                                  'source_commit': plan['source_commit'],
                                  'test_data_used': False})
    started = time.monotonic()
    deadline = started + plan['max_seconds'] - 30.
    context = None
    try:
        device = torch.device('cuda', 0)
        context = prepare_qualified_inputs(initial_plan_ref=INITIAL_PLAN_REF,
            design=bound['design']['fresh_initial_design'], device=device)
        need(context['view']['contract_sha256']
             == bound['initial']['measurements']['TRAIN']['feature_view_sha256'],
             'ACTUAL_VIEW_IDENTITY')
        trainer._set_deterministic(20260911, device, trainer.DETERMINISTIC_FP32)
        backbone = build_native_direction_research_model(
            context['view'], prefix_normalization=context['packet'],
            constructor_metadata=context['constructor'], device=torch.device('cpu'),
            unobserved_units_profile=None)
        model = NativeFutureDualModel(backbone, shared_m5_side=True).to(device).train()
        model._require_binding()
        need(model.research_objective['contract_sha256'] == plan['model_contract_sha256']
             and canonical_model_state_sha256(model.state_dict())
                 == plan['initial_model_state_sha256']
             and model.future_m5_side_head.weight is model.state_model.m5_side_head.weight
             and model.future_m5_side_head.bias is model.state_model.m5_side_head.bias,
             'FRESH_SHARED_CONSTRUCTOR')
        optimizer = torch.optim.AdamW(model.parameters(), **ADAMW_DEFAULTS)
        axes = context['axes']
        fit = fit_one_pass(
            model=model, optimizer=optimizer,
            parent_rows=axes['parent_rows'][:TRAIN_ROWS],
            issue_ns=axes['decision_ns'][:TRAIN_ROWS],
            state_bound=context['state_fit_bound'], future_bound=context['future_bound'],
            gather_features=lambda left, right, rows, issues:
                context['features']('TRAIN', rows, issues),
            deadline_monotonic=deadline,
            expected_initial_model_sha256=plan['initial_model_state_sha256'],
            expected_loss_sha256=plan['loss_contract_sha256'],
            expected_model_contract_sha256=plan['model_contract_sha256'])
        final = evaluate_fixed_panels(model=model, context=context,
                                      deadline_monotonic=deadline)
        initial_later = read_capture(bound['initial']['captures']['LATER'])
        paired = paired_later_f_m5(initial_later, final['captures']['LATER'])
        captures = {}
        for label in ('TRAIN', 'LATER'):
            stream = io.BytesIO()
            np.savez(stream, **final['captures'][label])
            captures[label] = publish('FINAL_' + label + '_READOUT.npz', stream.getvalue())
        need(fit['batch_visits'] == BATCH_VISITS
             and fit['optimizer_updates'] + fit['skipped_batches'] == BATCH_VISITS
             and fit['parent_cursor'] == TRAIN_ROWS
             and final['evaluation_forwards'] == 512
             and fit['checkpoint_writes'] == 0
             and not fit['test_data_used'] and not final['test_data_used']
             and not paired['test_data_used'] and source_clean(plan)
             and time.monotonic() < deadline, 'COMPLETE_BOOKEND')
        context['release']()
        context = None
        result = publish('RESULT.json', {
            'status': 'COMPLETE_SHARED_M5_FIT_AND_READOUT',
            'plan': {'path': str(plan_path), 'sha256': plan_sha}, 'claim': claim,
            'source_commit': plan['source_commit'], 'source_unchanged': True,
            'guard': proof, 'initial_model_state_sha256': plan['initial_model_state_sha256'],
            'final_model_state_sha256': final['final_model_state_sha256'],
            'model_constructions': 1, 'batch_visits': fit['batch_visits'],
            'optimizer_updates': fit['optimizer_updates'],
            'skipped_batches': fit['skipped_batches'],
            'evaluation_forwards': final['evaluation_forwards'],
            'checkpoint_writes': fit['checkpoint_writes'],
            'fit': fit, 'captures': captures, 'readouts': final['readouts'],
            'measurements': final['measurements'], 'paired_later_f_m5': paired,
            'test_data_used': False, 'elapsed_seconds': time.monotonic() - started})
        publish('TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'plan': {'path': str(plan_path), 'sha256': plan_sha}, 'result': result,
            'source_unchanged': True, 'test_data_used': False})
    except BaseException as exc:
        publish('TERMINAL.json', {'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'plan': {'path': str(plan_path), 'sha256': plan_sha},
            'error': type(exc).__name__ + ':' + str(exc), 'test_data_used': False})
        raise
    finally:
        if context is not None:
            context['release']()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    need(args.device == 'cuda' and args.out_dir == str(ROOT), 'CLI_SCOPE')
    plan_path = Path(args.plan)
    if args.preflight:
        bound = preflight(plan_path, args.plan_sha256, require_admission=False)
        preflight_actual_inputs(bound)
        print(json.dumps({'status': 'PASS_READ_ONLY_PREFLIGHT', 'stage': STAGE,
                          'source_commit': bound['plan']['source_commit'],
                          'model_forwards': 0, 'optimizer_updates': 0,
                          'claim_created': False, 'test_data_used': False}))
    else:
        run(plan_path, args.plan_sha256)


if __name__ == '__main__':
    main()
