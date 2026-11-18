"""One capped fresh cold M5 FIT/SCORE B16 model preflight; no fit or trade."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch

from gx1.audit import qualify_market_state_ta_cold_model_input_gather_v1 as raw_audit
from gx1.audit.market_state_ta_apr_jun_campaign_inputs_v1 import (
    REPO, COLD_FIT_ROWS, COLD_SCORE_START, COLD_VIEW_SHA256,
    cold_campaign_input_refs, prepare, read, ref,
)
from gx1.audit.run_market_state_ta_apr_jun_paired_fit_v1 import score
from gx1.audit.run_market_state_ta_m5_decision_fit_v1 import (
    COLD_TARGETS, load_targets, target_slice,
)
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
from gx1.contracts.market_state_ta_integrated_model_v1 import IntegratedTAMarketStateModel
from gx1.contracts.market_state_ta_integrated_paired_objective_v1 import contract as paired_contract
from gx1.contracts.market_state_ta_m5_decision_v1 import (
    IntegratedTAM5DecisionModel, decision_loss, fit_statistics,
)
from gx1.contracts.market_state_time_query_binding_v1 import (
    load_bound_time_query_targets, sampled_query_k,
)
from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

STAGE = 'ta_m5_cold_initial_preflight_20261117_001'
PLAN = REPO / 'configs/research/TA_M5_COLD_INITIAL_PREFLIGHT_20261117_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/TA_M5_COLD_INITIAL_PREFLIGHT_20261117_001')
RAW = Path('/home/andre2/GX1_RUNS/TA_M5_COLD_MODEL_INPUT_GATHER_20261117_001')
SOURCE_BASE = 'd958155aef0e85388217a955c07ac19358f1074c'
CORE_SHA = '70a0497b27ba0ab45277ba622443ab2fca36a448357f1ccb3158689193f8bc82'
SEED = 20260911
BATCH = 16
WALL_SECONDS = 900
SOURCES = tuple(dict.fromkeys((*raw_audit.SOURCES,
    'gx1/audit/qualify_market_state_ta_cold_initial_preflight_v1.py',
    'gx1/audit/run_market_state_ta_apr_jun_paired_fit_v1.py',
    'gx1/audit/run_market_state_ta_m5_decision_fit_v1.py',
    'gx1/contracts/market_state_ta_integrated_model_v1.py',
    'gx1/contracts/market_state_ta_integrated_paired_objective_v1.py',
    'gx1/contracts/market_state_ta_m5_decision_v1.py',
    'gx1/contracts/market_state_time_query_binding_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_future_displacement_candidate_v1.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/market_state_time_query_target_v1.py',
    'gx1/contracts/model_state_digest_v1.py',
    'gx1/models/entry_v10/entry_v10_ctx_train_v3.py',
    'tests/test_qualify_market_state_ta_cold_initial_preflight_v1.py',
)))


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_COLD_INITIAL_PREFLIGHT_' + reason)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def raw_refs():
    return {'raw_plan': ref(raw_audit.PLAN), 'raw_result': ref(RAW / 'RESULT.json'),
            'raw_terminal': ref(RAW / 'TERMINAL.json')}


def inputs():
    return {**cold_campaign_input_refs(), 'fit_targets': ref(COLD_TARGETS), **raw_refs()}


def verify_raw(result, terminal, bindings):
    expected = {k:v for k,v in bindings.items() if not k.startswith('raw_')}
    batches = result.get('batches')
    need(result.get('status') == 'COMPLETE'
         and result.get('plan') == bindings['raw_plan']
         and result.get('inputs') == expected
         and result.get('view_sha256') == COLD_VIEW_SHA256
         and result.get('source_unchanged') is True
         and result.get('raw_values_not_normalized') is True
         and result.get('model_constructions') == result.get('model_forwards')
             == result.get('optimizer_steps') == result.get('normalization_fits') == 0
         and result.get('test_data_used') is False
         and isinstance(batches, list) and len(batches) == 4
         and [(b.get('name'), b.get('rows')) for b in batches]
             == [('FIT_B16',16),('FIT_TAIL6',6),('SCORE_B16',16),('SCORE_TAIL1',1)]
         and all(b.get('raw_values_not_normalized') is True
                 and set(b.get('features', {})) == raw_audit.FEATURE_KEYS for b in batches)
         and terminal.get('status') == 'COMPLETE' and terminal.get('exit_code') == 0
         and terminal.get('plan') == bindings['raw_plan']
         and terminal.get('result') == bindings['raw_result']
         and terminal.get('source_unchanged') is True
         and terminal.get('no_relaunch') is True, 'RAW_RECEIPTS')
    return batches


def build_plan():
    bindings = inputs()
    batches = verify_raw(read(bindings['raw_result']), read(bindings['raw_terminal']), bindings)
    need(paired_contract(fold_id=1)['contract_sha256'] != paired_contract()['contract_sha256'],
         'COLD_FOLD1_OBJECTIVE')
    scope = dict(view_sha256=COLD_VIEW_SHA256, fit_rows=COLD_FIT_ROWS,
        score_parent_start=COLD_SCORE_START, fit_batch_rows=BATCH, score_batch_rows=BATCH,
        raw_fit_parent_rows_sha256=batches[0]['parent_rows_sha256'],
        raw_fit_decision_ns_sha256=batches[0]['decision_ns_sha256'],
        raw_score_parent_rows_sha256=batches[2]['parent_rows_sha256'],
        raw_score_decision_ns_sha256=batches[2]['decision_ns_sha256'],
        core_contract_sha256=CORE_SHA,
        cold_paired_objective_sha256=paired_contract(fold_id=1)['contract_sha256'],
        fresh_model_constructions=1, model_forwards=2, backward_calls=0,
        optimizer_constructions=0, optimizer_steps=0, normalization_fits=0,
        test_data_used=False, economic_edge_proven=False)
    return dict(schema_version='gx1_ta_m5_cold_initial_preflight_plan_v1',
        stage=STAGE, source_base_commit=SOURCE_BASE,
        source_owners={name:ref(REPO / name) for name in SOURCES},
        inputs=bindings, scope=scope, output_root=str(ROOT),
        execution=dict(runner='scripts/gx1_capped_run.sh', job_class='audit',
            memory='4G', swap='512M', max_seconds=WALL_SECONDS,
            max_invocations=1, training_authorized=False, test_authorized=False))


def inspect(plan_sha, *, execute):
    plan_ref = ref(PLAN)
    plan = read(plan_ref)
    need(plan_ref['sha256'] == plan_sha and plan == build_plan(), 'PLAN_IDENTITY')
    need(not ROOT.exists(), 'ONE_USE_ROOT')
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    branch = subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    clean = subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True) == ''
    ancestor = subprocess.run(['git','merge-base','--is-ancestor',SOURCE_BASE,head],
        cwd=REPO,capture_output=True).returncode == 0
    need(Path.cwd().resolve() == REPO and branch == 'work/gx1-current' and ancestor,
         'SOURCE_BRANCH_CWD')
    if not execute:
        return plan, plan_ref, None, head, clean
    need(clean, 'CLEAN_SOURCE')
    policy_ref = ref(REPO / 'NEXT_RUN_POLICY.json')
    policy = read(policy_ref)
    need(all(policy.get(key) is False for key in ('training_enabled',
        'native_training_enabled','native_launch_authorized',
        'full_epoch_training_allowed','full_val_allowed'))
        and policy.get('exit_training_and_evaluation_this_wave') is False,
        'TRAINING_AND_TEST_CLOSED')
    gate = policy.get(STAGE, {})
    need(gate.get('status') == 'ADMITTED' and gate.get('plan') == plan_ref
         and gate.get('scope') == plan['scope']
         and gate.get('execution_authorized') is True
         and gate.get('data_reads_authorized') is True
         and gate.get('model_constructions_authorized') == 1
         and gate.get('model_forwards_authorized') == 2
         and gate.get('training_authorized') is False
         and gate.get('test_authorized') is False
         and gate.get('broker_access_allowed') is False
         and gate.get('optimizer_steps_authorized') == 0
         and gate.get('max_invocations') == gate.get('remaining_invocations') == 1
         and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
         and gate.get('relaunch_allowed') is False, 'ONE_USE_ADMISSION')
    return plan, plan_ref, policy_ref, head, clean


def normalized_surfaces(model, features):
    backbone = model.core.dual.state_model.backbone
    details = {}
    with torch.no_grad():
        for key, surface in [('seq_x','signal'),('snap_x','signal'),('ctx_cont','ctx_cont'),
                             *((f'seq_{tf}',f'mtf_{tf}') for tf in
                               ('m5','m15','h1','h4','d1'))]:
            value = backbone._normalize_input_surface(features[key], surface=surface)
            need(value.shape == features[key].shape and bool(torch.isfinite(value).all()),
                 'NORMALIZED_' + key)
            details[key] = dict(shape=list(value.shape),
                nonzero_cells=int(torch.count_nonzero(value)),
                sha256=hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest())
    need(features['ctx_cat'].dtype == torch.int64
         and features['ctx_cat'].shape == (BATCH,1), 'CATEGORY_SURFACE')
    return details


def verify_actual_batch(name, features, rows, issues, expected):
    actual = raw_audit.summarize_batch(name, features, rows, issues)
    need(actual == expected, 'RAW_BATCH_' + name)


def fit_probe(model, context, fit_targets, expected):
    axes = context['axes']
    rows, issues = axes['parent_rows'][:BATCH], axes['decision_ns'][:BATCH]
    features = context['features']('TRAIN', rows, issues)
    verify_actual_batch('FIT_B16', features, rows, issues, expected)
    normalized = normalized_surfaces(model, features)
    sampled = sampled_query_k(issues, pass_index=0, seed=SEED)
    native = np.broadcast_to(np.asarray(NATIVE_K,dtype=np.int64), sampled.shape).copy()
    grid = torch.from_numpy(np.stack((sampled,native),axis=1).copy())
    tape = context['fit_tape']
    sampled_bound = load_bound_time_query_targets()
    sampled_truth = sampled_bound.gather(rows, issues, sampled, fold_id=1,
        partition='fit', pass_index=0, seed=SEED)
    native_truth = load_bound_five_endpoint_targets().gather(rows,issues,
        fold_id=1,partition='fit')
    future_truth = load_bound_future_displacement().gather(rows,issues,
        fold_id=1,partition='fit')
    target = {key:fit_targets[key][:BATCH] for key in
        ('terminal_observed','path_observed','net_bps','mfe_bps','pre_mfe_mae_upper_bps')}
    model.eval()
    with torch.inference_mode():
        output = model(features, query_k=grid)
        loss = decision_loss(model_output=output, targets=target, model=model,
            sampled_truth=sampled_truth, native_truth=native_truth,
            future_truth=future_truth, query_grid=grid,
            structure_hit=tape['structure_hit'][:BATCH],
            structure_observed=tape['structure_observed'][:BATCH])
    need(bool(torch.isfinite(loss['loss']))
         and loss['contract_sha256'] == model.research_objective['contract_sha256']
         and output['m5_cost_path_bps'].shape == (BATCH,6,2,3)
         and bool(torch.isfinite(output['m5_cost_path_bps']).all()), 'FIT_LOSS')
    return normalized, float(loss['loss']), loss['contract_sha256']


def score_probe(model, context, expected, *, deadline):
    axes = context['axes']
    selected = context['selected'][:BATCH]
    rows, issues = axes['parent_rows'][selected], axes['decision_ns'][selected]
    features = context['features']('SCORE', rows, issues)
    verify_actual_batch('SCORE_B16', features, rows, issues, expected)
    tape = {key:value[:BATCH] for key,value in context['score_tape'].items()}
    def cached(label, got_rows, got_issues):
        need(label == 'SCORE' and np.array_equal(got_rows, rows)
             and np.array_equal(got_issues, issues), 'SCORE_CACHED_AXIS')
        return features
    shallow = {**context, 'selected':selected, 'score_tape':tape, 'features':cached}
    readout, capture = score(model, shallow, deadline=deadline,
        fold_id=1, partition='score', collect_decision=True)
    predicted = capture['m5_cost_path_bps']
    need(readout['rows'] == BATCH and readout['model_forwards'] == 1
         and predicted.shape == (BATCH,6,2,3) and np.isfinite(predicted).all(),
         'SCORE_FORWARD')
    return hashlib.sha256(np.ascontiguousarray(predicted).tobytes()).hexdigest()


def publish(name, value):
    data = encoded(value)
    need(len(data) <= 2 * 1024**2, 'JSON_BUDGET')
    _publish_bytes(ROOT / name, data)
    return ref(ROOT / name)


def bookend(plan_ref, policy_ref, head, plan):
    need(ref(PLAN) == plan_ref and ref(REPO / 'NEXT_RUN_POLICY.json') == policy_ref
         and subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip() == head
         and subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True) == ''
         and all(ref(REPO / path) == binding for path,binding in plan['source_owners'].items())
         and inputs() == plan['inputs'], 'SOURCE_INPUT_BOOKEND')


def run(plan_sha):
    started = time.monotonic()
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    plan, plan_ref, policy_ref, head, _ = inspect(plan_sha, execute=True)
    need(signal.getitimer(signal.ITIMER_REAL) == (0.,0.), 'AMBIENT_TIMER')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ROOT.mkdir()
    claim = publish('CLAIM.json', dict(plan=plan_ref, policy=policy_ref,
        source_head=head, cap=cap, scope=plan['scope'],
        claimed_utc=datetime.now(timezone.utc).isoformat()))
    status, result, failure, context = 'FAILED_CONSUMED', None, None, None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def expired(*_):raise TimeoutError('TA_COLD_INITIAL_PREFLIGHT_TIMEOUT')
    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(1,WALL_SECONDS-30-(time.monotonic()-started)))
        context = prepare(device=torch.device('cpu'), cold_inputs=cold_campaign_input_refs())
        need(context['input_refs'] == {key:plan['inputs'][key]
             for key in cold_campaign_input_refs()}
             and context['fit_rows'] == COLD_FIT_ROWS
             and context['fit_fold_id'] == context['score_fold_id'] == 1
             and context['score_partition'] == 'score'
             and context['view']['contract_sha256'] == COLD_VIEW_SHA256
             and context['units_profile'] == context['view']['unobserved_units_profile'],
             'COLD_PREPARED_CONTEXT')
        full = load_targets(plan['inputs']['fit_targets'], 47338)
        fit = target_slice(full, 0, COLD_FIT_ROWS)
        axes = context['axes']
        need(np.array_equal(axes['parent_rows'][:COLD_FIT_ROWS], fit['parent_rows'])
             and np.array_equal(axes['decision_ns'][:COLD_FIT_ROWS], fit['issue_ns']),
             'FIT_COST_AXES')
        stats = fit_statistics(fit)
        trainer._set_deterministic(SEED, torch.device('cpu'), trainer.DETERMINISTIC_FP32)
        state = views.build_native_direction_research_model(context['view'],
            prefix_normalization=context['packet'],
            constructor_metadata=context['constructor'], device=torch.device('cpu'),
            unobserved_units_profile=context['units_profile'])
        core = IntegratedTAMarketStateModel(NativeFutureDualModel(state))
        need(core.research_objective['contract_sha256'] == CORE_SHA, 'CORE_CONTRACT')
        model = IntegratedTAM5DecisionModel(core, stats, aux_fold_id=1).eval()
        digest = canonical_model_state_sha256(model.state_dict())
        need(model.research_objective['cold_aux_contract_sha256'] ==
             plan['scope']['cold_paired_objective_sha256'], 'COLD_DECISION_CONTRACT')
        raw_result = read(plan['inputs']['raw_result'])
        batches = verify_raw(raw_result, read(plan['inputs']['raw_terminal']),plan['inputs'])
        normalized, loss, loss_sha = fit_probe(model, context, fit, batches[0])
        score_hash = score_probe(model, context, batches[2],
            deadline=started+WALL_SECONDS-30)
        need(canonical_model_state_sha256(model.state_dict()) == digest,
             'INITIAL_STATE_UNCHANGED')
        context['release']()
        context = None
        bookend(plan_ref, policy_ref, head, plan)
        result = publish('RESULT.json', dict(status='COMPLETE_COLD_INITIAL_FIT_SCORE_NO_FIT',
            plan=plan_ref, claim=claim, policy=policy_ref, source_head=head,
            source_unchanged=True, inputs=plan['inputs'], scope=plan['scope'],
            initial_model_state_sha256=digest,
            core_contract_sha256=core.research_objective['contract_sha256'],
            decision_contract_sha256=model.research_objective['contract_sha256'],
            cold_paired_objective_sha256=plan['scope']['cold_paired_objective_sha256'],
            fit_statistics_sha256=stats['sha256'], fit_loss_contract_sha256=loss_sha,
            fit_loss=loss, normalized_numeric_surfaces=normalized,
            score_b16_decision_output_sha256=score_hash,
            fresh_model_constructions=1, model_forwards=2,
            backward_calls=0, optimizer_constructions=0, optimizer_steps=0,
            normalization_fits=0, test_data_used=False, economic_edge_proven=False,
            elapsed_seconds=time.monotonic()-started))
        status = 'COMPLETE'
    except BaseException as error:
        failure = publish('FAILED.json', dict(status='FAILED_CONSUMED',
            plan=plan_ref, claim=claim, error_type=type(error).__name__,
            error=str(error)[:1800], result=result))
    finally:
        if context is not None:
            try:context['release']()
            except BaseException as error:
                if failure is None:
                    failure = publish('FAILED.json', dict(status='FAILED_CONSUMED',
                        plan=plan_ref, claim=claim, error_type=type(error).__name__,
                        error=str(error)[:1800], result=result))
                status = 'FAILED_CONSUMED'
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        terminal = publish('TERMINAL.json', dict(status=status,
            exit_code=0 if status == 'COMPLETE' else 1,
            plan=plan_ref, claim=claim, result=result, failure=failure,
            no_relaunch=True, remaining_invocations=0,
            source_unchanged=subprocess.check_output(['git','rev-parse','HEAD'],
                cwd=REPO,text=True).strip()==head))
    return status,terminal


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--execute',action='store_true')
    parser.add_argument('--plan-sha256',required=True)
    args = parser.parse_args()
    if args.preflight:
        plan,plan_ref,_,head,clean=inspect(args.plan_sha256,execute=False)
        print(json.dumps(dict(status='PREFLIGHT_PASS' if clean else 'PREFLIGHT_SOURCE_DIRTY',
            plan=plan_ref,source_head=head,source_clean=clean,
            output_root_absent=True,claim_created=False,scope=plan['scope']),sort_keys=True))
        return
    status,terminal=run(args.plan_sha256)
    print(json.dumps(dict(status=status,terminal=terminal),sort_keys=True))
    if status!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':
    main()
