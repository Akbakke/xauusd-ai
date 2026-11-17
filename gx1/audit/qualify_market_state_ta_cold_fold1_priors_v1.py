"""One-use Fold1 S/F endpoint prior fit on the mature cold TRAIN prefix.

Only target owners are read. No model, normalization, optimizer or later labels
enter the prior fit; the qualified later coverage is bound as an axis witness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_future_displacement_candidate_v1 as future
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts import entry_anchor_span_feature_inputs_v1 as overlay
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
STAGE = 'ta_m5_cold_fold1_endpoint_priors_20261117_001'
ROOT = RUNS / STAGE.upper()
PLAN = REPO / 'configs/research' / (STAGE.upper() + '.json')
POLICY = REPO / 'NEXT_RUN_POLICY.json'
SOURCE_BASE = 'a11439c4f2a293db59d1b0ae182a7b035cc9e6a8'
B_ROOT = RUNS / 'TA_M5_COLD_FULL_LATER_INPUTS_20261117_001'
B_REFS = {
    'plan': dict(path=str(REPO / 'configs/research/TA_M5_COLD_FULL_LATER_INPUTS_20261117_001.json'),
                 sha256='ccf3437645cd63f5defd896a996fb6b8ee2db27e373fa09705787e06d41cb836'),
    'view': dict(path=str(B_ROOT / 'FEATURE_VIEW.json'),
                 sha256='a10d152df67308348f72636c544f7b7299f6f226ad1d876fa96cd48d6b6971f4'),
    'result': dict(path=str(B_ROOT / 'RESULT.json'),
                   sha256='614c5b766e564fa0466a3f68f72f94300674b9823d67f34490c8e986fd01b36d'),
    'coverage': dict(path=str(B_ROOT / 'COVERAGE.json'),
                     sha256='ca030948f3f450d19e7af6244d881db7c5cc7cb10157a53aba368d5bfa55ea2c'),
    'terminal': dict(path=str(B_ROOT / 'TERMINAL.json'),
                     sha256='ab7a81b2f25d268e7ee537d294be91915a6cab36c8a8cd87a7473e8c0f34d4e5'),
}
VIEW_SHA = '7cdcc64b80ba8a1ac92aae24ac7fcd2b62252da31cf96e57ad7a86fc943c3293'
LOSS_SHA = '63cb7337ee45806cd10f2f4b15416997e7a8b0f579acc233b577b2b147a48f0d'
FIT_ROWS = 29574
PARENT_SHA = '1c5a8c36a94b8fa39d613f0bc122cf869dc9fa3c2f87dc64c0506acd2f55a90f'
ISSUE_SHA = 'ebb3489c45b01b6fd8e77bf27529d650beb6883ba84b57ee936dc549ec74e427'
WALL_SECONDS = 600
SOURCE_PATHS = (
    'gx1/audit/qualify_market_state_ta_cold_fold1_priors_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_future_displacement_candidate_v1.py',
    'gx1/contracts/market_state_five_endpoint_prior_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/market_state_time_query_scorer_v2.py',
    'gx1/contracts/market_state_time_query_target_v1.py',
    'gx1/contracts/market_state_time_query_side_normalized_loss_v1.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/entry_anchor_span_feature_inputs_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_qualify_market_state_ta_cold_fold1_priors_v1.py',
)


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_COLD_FOLD1_PRIOR_' + reason)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def file_ref(path):
    path = Path(path)
    need(path.is_absolute() and path.is_file() and not path.is_symlink(), 'FILE_REF')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest())


def checked_json(ref):
    need(type(ref) is dict and set(ref) == {'path', 'sha256'}
         and file_ref(ref['path']) == ref, 'INPUT_REF')
    return json.loads(Path(ref['path']).read_text())


def target_refs():
    refs = {('state_' + name):dict(path=str(path), sha256=digest)
            for name, (path, digest) in state.REFS.items()}
    refs.update({('future_' + name):dict(path=str(path), sha256=digest)
                 for name, (path, digest) in future.REFS.items()})
    return refs


def verify_b_receipts(view, result, coverage, terminal):
    panel = overlay.cold_full_later_panel()
    need(view['contract_sha256'] == VIEW_SHA
         and view['prefix']['rows'] == FIT_ROWS
         and view['prefix']['cutoff_ns'] == panel['prefix_cutoff_ns']
         and view['full_axes']['rows'] == 118249
         and result['status'] == 'COMPLETE'
         and result['plan'] == B_REFS['plan']
         and result['view'] == B_REFS['view']
         and result['coverage_path'] == B_REFS['coverage']['path']
         and result['source_unchanged'] is True
         and result['test_data_used'] is False
         and result['model_forwards'] == result['optimizer_steps'] == 0
         and coverage['schema_version'] == overlay.COLD_FULL_LATER_COVERAGE_SCHEMA
         and coverage['status'] == 'COMPLETE'
         and coverage['qualification'] == B_REFS['result']
         and coverage['view_sha256'] == VIEW_SHA
         and coverage['full_axes'] == view['full_axes']
         and coverage['normalization_prefix'] == view['prefix']
         and coverage['physical_sidecars'] == result['physical_sidecars']
         and coverage['physical_populations'] == result['physical_populations']
         and all(coverage[k] == panel[k] for k in panel if k not in ('prefix_rows','prefix_cutoff_ns'))
         and terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
         and terminal['plan'] == B_REFS['plan']
         and terminal['result'] == B_REFS['result']
         and terminal['coverage'] == B_REFS['coverage']
         and terminal['no_relaunch'] is True
         and terminal['source_unchanged'] is True, 'B_COVERAGE_CHAIN')


def build_plan():
    checked_json(B_REFS['plan'])
    view, result, coverage, terminal = (checked_json(B_REFS[k]) for k in
        ('view', 'result', 'coverage', 'terminal'))
    verify_b_receipts(view, result, coverage, terminal)
    profile = objective.m5_equal_observation_profile(fold_id=1)
    need(objective.contract(ta_only=True, dedicated_m5_side=True)['contract_sha256'] == LOSS_SHA
         and profile['fit_rows'] == FIT_ROWS
         and profile['parent_axis_sha256'] == PARENT_SHA
         and profile['issue_axis_sha256'] == ISSUE_SHA, 'CURRENT_CONTRACT')
    scope = dict(fold_id=1, fit_rows=FIT_ROWS, fit_parent_rows_sha256=PARENT_SHA,
        fit_decision_ns_sha256=ISSUE_SHA, cold_view_sha256=VIEW_SHA,
        loss_contract_sha256=LOSS_SHA, later_coverage_rows=coverage['score_rows'],
        later_coverage_sha256=B_REFS['coverage']['sha256'],
        target_arrays_as_forward_inputs=False, model_constructions=0,
        model_forwards=0, optimizer_steps=0, normalization_fits=0,
        test_data_used=False, economic_edge_proven=False)
    return dict(schema_version='gx1_ta_m5_cold_fold1_prior_plan_v1', stage=STAGE,
        source_repo=str(REPO), source_base_commit=SOURCE_BASE,
        source_owners={p:file_ref(REPO / p) for p in SOURCE_PATHS},
        b_refs=B_REFS, target_refs=target_refs(), scope=scope,
        execution=dict(runner='scripts/gx1_capped_run.sh', job_class='audit',
            memory='4G', swap='512M', timeout_seconds=WALL_SECONDS,
            max_invocations=1, training_authorized=False, test_authorized=False),
        output_root=str(ROOT))


def inspect_plan(plan_sha, *, execute):
    plan_ref = dict(path=str(PLAN), sha256=plan_sha)
    plan = checked_json(plan_ref)
    need(plan == build_plan(), 'PLAN_IDENTITY')
    need(all(file_ref(ref['path']) == ref for ref in plan['target_refs'].values()),
         'TARGET_FILE_HASH')
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    branch = subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    clean = subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True) == ''
    ancestor = subprocess.run(['git','merge-base','--is-ancestor',SOURCE_BASE,head],
        cwd=REPO,capture_output=True).returncode == 0
    need(Path.cwd().resolve() == REPO and branch == 'work/gx1-current' and ancestor,
         'SOURCE_BRANCH_CWD')
    need(not ROOT.exists(), 'ONE_USE_ROOT_EXISTS')
    if execute:
        need(clean, 'CLEAN_SOURCE')
        policy_ref = file_ref(POLICY)
        policy = checked_json(policy_ref)
        need(all(policy.get(k) is False for k in
             ('training_enabled','native_training_enabled','native_launch_authorized',
              'full_val_allowed','full_epoch_training_allowed')), 'TRAINING_CLOSED')
        gate = policy.get(STAGE, {})
        need(gate.get('status') == 'ADMITTED' and gate.get('plan') == plan_ref
             and gate.get('output_root') == str(ROOT)
             and gate.get('scope') == plan['scope']
             and gate.get('execution_authorized') is True
             and gate.get('data_reads_authorized') is True
             and gate.get('baseline_estimation_authorized') is True
             and gate.get('training_authorized') is False
             and gate.get('test_authorized') is False
             and gate.get('broker_access_allowed') is False
             and gate.get('model_forwards_authorized') == 0
             and gate.get('optimizer_steps_authorized') == 0
             and gate.get('max_invocations') == gate.get('remaining_invocations') == 1
             and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
             and gate.get('relaunch_allowed') is False, 'EXACT_POLICY_ADMISSION')
        return plan, plan_ref, policy_ref, head, clean
    return plan, plan_ref, None, head, clean


def validate_axes(state_bound, future_bound, coverage):
    need(np.array_equal(state_bound.parent_rows, future_bound.parent_rows)
         and np.array_equal(state_bound.decision_ns, future_bound.decision_ns),
         'TARGET_JOIN')
    fit = state_bound.mature_oof_axes(fold_id=1)
    need(fit['joined_fit_rows'] == FIT_ROWS
         and fit['fit_parent_rows_sha256'] == PARENT_SHA
         and fit['fit_decision_ns_sha256'] == ISSUE_SHA
         and fit['old_joined_fit_rows'] == coverage['score_parent_start']
         and fit['joined_fit_rows'] + fit['discarded_joined_fit_rows'] == fit['old_joined_fit_rows']
         and len(fit['score_parent_rows']) >= coverage['score_rows']
         and np.array_equal(state_bound.parent_rows[coverage['score_parent_start']:
             coverage['score_parent_start'] + coverage['score_rows']],
             fit['score_parent_rows'][:coverage['score_rows']])
         and overlay._axis_sha(state_bound.parent_rows) == coverage['full_axes']['parent_rows_sha256']
         and overlay._axis_sha(state_bound.decision_ns) == coverage['full_axes']['decision_ns_sha256']
         and overlay._axis_sha(state_bound.parent_rows[coverage['score_parent_start']:
             coverage['score_parent_start'] + coverage['score_rows']])
             == coverage['selected_parent_rows_sha256']
         and overlay._axis_sha(state_bound.decision_ns[coverage['score_parent_start']:
             coverage['score_parent_start'] + coverage['score_rows']])
             == coverage['selected_decision_ns_sha256']
         and int(fit['fit_decision_ns'][-1]) + 1440 * state.MINUTE_NS
             < coverage['score_start_ns'], 'EXACT_MATURE_AXES')
    return fit


def validate_prior(priors, fit):
    need(objective.require_endpoint_priors(priors,fold_id=1,ta_only=True,
         dedicated_m5_side=True) is priors
         and priors['fit_rows'] == len(fit['fit_parent_rows'])
         and priors['fit_parent_rows_sha256'] == fit['fit_parent_rows_sha256']
         and priors['fit_decision_ns_sha256'] == fit['fit_decision_ns_sha256']
         and priors['loss_contract_sha256'] == LOSS_SHA, 'NEW_PRIOR_BINDING')
    return priors


def fit_prior(state_bound, future_bound, fit):
    value = objective.fit_endpoint_priors(state_bound, future_bound,
        axes={'fit_parent_rows':fit['fit_parent_rows'],
              'fit_decision_ns':fit['fit_decision_ns']},
        fold_id=1, ta_only=True, dedicated_m5_side=True)
    return validate_prior(value, fit)


def publish(name, value):
    data = encoded(value)
    need(len(data) <= 2 * 1024**2, 'JSON_BUDGET')
    path = ROOT / name
    _publish_bytes(path, data)
    return file_ref(path)


def bookend(plan_ref, policy_ref, head, plan):
    need(file_ref(PLAN) == plan_ref and file_ref(POLICY) == policy_ref
         and subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip() == head
         and subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True) == ''
         and all(file_ref(REPO / p) == ref for p, ref in plan['source_owners'].items())
         and all(file_ref(ref['path']) == ref for ref in
             (*plan['b_refs'].values(), *plan['target_refs'].values())),
         'SOURCE_INPUT_BOOKEND')


def terminal_record(status, *, plan_ref, claim, head, prior, result, failure, source_unchanged):
    need((status == 'COMPLETE' and prior is not None and result is not None and failure is None)
         or (status == 'FAILED_CONSUMED' and failure is not None), 'TERMINAL_STATE')
    return dict(status=status, exit_code=0 if status == 'COMPLETE' else 1,
        no_relaunch=True, source_head=head, plan=plan_ref, claim=claim,
        endpoint_priors=prior, result=result, failure=failure,
        source_unchanged=source_unchanged)


def run(plan_sha):
    began = time.monotonic()
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    plan, plan_ref, policy_ref, head, _ = inspect_plan(plan_sha, execute=True)
    need(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'AMBIENT_TIMER')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ROOT.mkdir()
    claim = publish('CLAIM.json', dict(plan=plan_ref, policy=policy_ref,
        source_head=head, cap=cap, scope=plan['scope']))
    prior = result = failure = None
    status = 'FAILED_CONSUMED'
    old_alarm = signal.getsignal(signal.SIGALRM)
    def expired(*_):raise TimeoutError('TA_COLD_FOLD1_PRIOR_600_SECONDS')
    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(1, WALL_SECONDS - 30 - (time.monotonic() - began)))
        coverage = checked_json(B_REFS['coverage'])
        state_bound = state.load_bound_five_endpoint_targets()
        future_bound = future.load_bound_future_displacement()
        fit = validate_axes(state_bound, future_bound, coverage)
        priors = fit_prior(state_bound, future_bound, fit)
        bookend(plan_ref, policy_ref, head, plan)
        prior = publish('ENDPOINT_PRIORS.json', priors)
        result = publish('RESULT.json', dict(status='COMPLETE', plan=plan_ref,
            claim=claim, policy=policy_ref, source_head=head, source_unchanged=True,
            endpoint_priors=prior, fold_id=1, fit_rows=FIT_ROWS,
            fit_parent_rows_sha256=PARENT_SHA, fit_decision_ns_sha256=ISSUE_SHA,
            loss_contract_sha256=LOSS_SHA, cold_view=B_REFS['view'],
            later_coverage=B_REFS['coverage'], later_result=B_REFS['result'],
            target_refs=plan['target_refs'], model_constructions=0,
            model_forwards=0, optimizer_steps=0, normalization_fits=0,
            test_data_used=False, economic_edge_proven=False))
        status = 'COMPLETE'
    except BaseException as exc:
        failure = publish('FAILED.json', dict(status='FAILED_CONSUMED', plan=plan_ref,
            claim=claim, error_type=type(exc).__name__, error=str(exc),
            endpoint_priors=prior, result=result))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        terminal = publish('TERMINAL.json', terminal_record(status, plan_ref=plan_ref,
            claim=claim, head=head, prior=prior, result=result, failure=failure,
            source_unchanged=(subprocess.check_output(['git','rev-parse','HEAD'],
                cwd=REPO,text=True).strip()==head)))
    return terminal, status


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    if args.preflight:
        plan, plan_ref, _, head, clean = inspect_plan(args.plan_sha256, execute=False)
        print(json.dumps(dict(status='PREFLIGHT_PASS' if clean else 'PREFLIGHT_SOURCE_DIRTY',
            plan=plan_ref, source_head=head, source_clean=clean, output_root_absent=True,
            scope=plan['scope'], claim_created=False), sort_keys=True))
        return
    terminal, status = run(args.plan_sha256)
    print(json.dumps(dict(status=status, terminal=terminal), sort_keys=True))
    if status != 'COMPLETE':raise SystemExit(1)


if __name__ == '__main__':
    main()
