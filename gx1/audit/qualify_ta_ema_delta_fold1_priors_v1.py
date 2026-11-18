"""One-use target-only Fold1 endpoint priors for the qualified 318/256 EMA view.

The old prior is sealed to a different loss contract. This successor fits the
current contract on the same mature 29,574 FIT axes; later labels are never
passed to the fitter. No model, normalization or optimizer is constructed.
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

from gx1.audit import qualify_market_state_ta_cold_fold1_priors_v1 as old
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_future_displacement_candidate_v1 as future
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts import entry_anchor_span_feature_inputs_v1 as overlay
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
STAGE = 'ta_m5_ema_delta_fold1_priors_20261118_001'
ROOT = RUNS / STAGE.upper()
PLAN = REPO / 'configs/research' / (STAGE.upper() + '.json')
POLICY = REPO / 'NEXT_RUN_POLICY.json'
GATHER_ROOT = RUNS / 'TA_M5_EMA_DELTA_FULL_GATHER_20261118_001'
VIEW_ROOT = RUNS / 'TA_M5_EMA_DELTA_FEATURE_VIEW_20261118_001'
GATHER_SHA = {
    'result': 'f1589bff55919ac9a1bcb0c9ed99ae7848c51e5dd34bf453d9bd6d92d34b8f2a',
    'coverage': 'd3c95a7a16a3a63c3276ba5f90dfa3b99a93957edea302cf990d521c3c20325c',
    'terminal': '57ff529479000918ac382268e857d518c8d89f7ebe8111bad16aedb4a6149ce3',
}
VIEW_FILE_SHA = '638b3706ebc4566439599e27826cd00322d8e0156166333e507a4707bd1bac25'
VIEW_CONTRACT_SHA = '1bf7066a7595040ee541b83ca92765c500bc3619898553bca1714ed692fa3921'
LOSS_SHA = '2bcdacf11c849ff60da490af984ec065abf18948540f8494d6a8805517203f20'
FIT_ROWS = old.FIT_ROWS
WALL_SECONDS = 600
SOURCE_PATHS = tuple(dict.fromkeys((
    'gx1/audit/qualify_ta_ema_delta_fold1_priors_v1.py',
    'gx1/audit/qualify_ta_ema_delta_full_gather_v1.py',
    'gx1/contracts/entry_ema_delta_feature_view_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    *old.SOURCE_PATHS,
    'tests/test_qualify_ta_ema_delta_fold1_priors_v1.py',
)))


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_EMA_DELTA_FOLD1_PRIOR_' + reason)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def ref(path):
    path = Path(path)
    need(path.is_absolute() and path.is_file() and not path.is_symlink(), 'FILE_REF')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def read(binding):
    need(type(binding) is dict and set(binding) == {'path', 'sha256'}
         and ref(binding['path']) == binding, 'INPUT_REF')
    return json.loads(Path(binding['path']).read_text())


def input_refs():
    refs = {
        'gather_result': {'path': str(GATHER_ROOT / 'RESULT.json'), 'sha256': GATHER_SHA['result']},
        'gather_coverage': {'path': str(GATHER_ROOT / 'COVERAGE.json'), 'sha256': GATHER_SHA['coverage']},
        'gather_terminal': {'path': str(GATHER_ROOT / 'TERMINAL.json'), 'sha256': GATHER_SHA['terminal']},
        'view': {'path': str(VIEW_ROOT / 'FEATURE_VIEW.json'), 'sha256': VIEW_FILE_SHA},
    }
    need(all(ref(binding['path']) == binding for binding in refs.values()), 'PREDECESSOR_REFS')
    return refs


def validate_receipts(refs, result, coverage, terminal, view):
    panel = overlay.cold_full_later_panel()
    need(result['status'] == coverage['status'] == terminal['status'] == 'COMPLETE'
         and terminal['exit_code'] == 0 and terminal['source_unchanged'] is True
         and terminal['result'] == refs['gather_result']
         and terminal['coverage'] == refs['gather_coverage']
         and result['view'] == refs['view']
         and result['source_unchanged'] is True
         and result['full_raw_and_normalized_selected_values'] is True
         and result['fit']['rows'] == FIT_ROWS
         and result['fit']['parent_rows_sha256'] == old.PARENT_SHA
         and result['fit']['decision_ns_sha256'] == old.ISSUE_SHA
         and result['later']['rows'] == panel['score_rows']
         and result['scope']['selected_parent_rows'] == FIT_ROWS + panel['score_rows']
         and result['scope']['target_arrays_decoded'] == 0
         and result['target_arrays_decoded'] == 0
         and result['model_forwards'] == result['optimizer_steps'] == 0
         and result['test_data_used'] is False
         and view['contract_sha256'] == coverage['view_sha256'] == VIEW_CONTRACT_SHA
         and view['prefix']['rows'] == FIT_ROWS
         and view['prefix']['cutoff_ns'] == panel['prefix_cutoff_ns']
         and coverage['schema_version'] == overlay.COLD_FULL_LATER_COVERAGE_SCHEMA
         and coverage['full_axes'] == view['full_axes']
         and coverage['normalization_prefix'] == view['prefix']
         and coverage['qualification'] == result['materialized_result']
         and all(coverage[k] == panel[k] for k in panel
                 if k not in ('prefix_rows', 'prefix_cutoff_ns')),
         'QUALIFIED_INPUT_CHAIN')
    return panel


def build_plan():
    refs = input_refs()
    result, coverage, terminal, view = (read(refs[k]) for k in
        ('gather_result', 'gather_coverage', 'gather_terminal', 'view'))
    panel = validate_receipts(refs, result, coverage, terminal, view)
    profile = objective.m5_equal_observation_profile(fold_id=1)
    need(objective.contract(ta_only=True, dedicated_m5_side=True)['contract_sha256'] == LOSS_SHA
         and profile['fit_rows'] == FIT_ROWS
         and profile['parent_axis_sha256'] == old.PARENT_SHA
         and profile['issue_axis_sha256'] == old.ISSUE_SHA, 'CURRENT_CONTRACT')
    targets = old.target_refs()
    need(all(ref(item['path']) == item for item in targets.values()), 'TARGET_REFS')
    scope = {'fold_id': 1, 'fit_rows': FIT_ROWS,
             'fit_parent_rows_sha256': old.PARENT_SHA,
             'fit_decision_ns_sha256': old.ISSUE_SHA,
             'feature_view_sha256': VIEW_CONTRACT_SHA,
             'loss_contract_sha256': LOSS_SHA,
             'later_coverage_rows': panel['score_rows'],
             'later_coverage_sha256': refs['gather_coverage']['sha256'],
             'later_target_values_in_fit': False,
             'model_constructions': 0, 'model_forwards': 0,
             'normalization_fits': 0, 'optimizer_steps': 0,
             'test_data_used': False, 'economic_edge_proven': False}
    return {'schema_version': 'gx1_ta_ema_delta_fold1_prior_plan_v1',
            'stage': STAGE, 'output_root': str(ROOT),
            'source_base_commit': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'source_owners': {name: ref(REPO / name) for name in SOURCE_PATHS},
            'inputs': refs, 'target_refs': targets, 'scope': scope,
            'execution': {'runner': 'scripts/gx1_capped_run.sh',
                          'job_class': 'audit', 'memory': '4G', 'swap': '512M',
                          'timeout_seconds': WALL_SECONDS, 'kill_after_seconds': 10}}


def inspect(plan_sha, *, execute):
    plan_ref = ref(PLAN)
    need(plan_ref['sha256'] == plan_sha and not ROOT.exists(), 'PLAN_OR_ONE_USE_ROOT')
    plan = read(plan_ref)
    expected = build_plan()
    expected['source_base_commit'] = plan.get('source_base_commit')
    need(plan == expected and plan['scope'] == expected['scope'], 'PLAN_IDENTITY')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
    clean = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True) == ''
    ancestor = subprocess.run(['git', 'merge-base', '--is-ancestor',
        plan['source_base_commit'], head], cwd=REPO, capture_output=True).returncode == 0
    need(Path.cwd().resolve() == REPO and branch == 'work/gx1-current' and ancestor,
         'SOURCE_BRANCH_CWD')
    if not execute:
        return plan, plan_ref, None, head, clean
    need(clean, 'SOURCE_CLEAN')
    policy_ref = ref(POLICY)
    policy = read(policy_ref)
    need(all(policy.get(k) is False for k in ('training_enabled',
         'native_training_enabled', 'native_launch_authorized',
         'full_epoch_training_allowed', 'full_val_allowed')),
         'TRAINING_CLOSED')
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
         and gate.get('relaunch_allowed') is False, 'EXACT_ONE_USE_ADMISSION')
    return plan, plan_ref, policy_ref, head, clean


def publish(name, value):
    data = encoded(value)
    need(len(data) <= 2 * 1024**2, 'JSON_BUDGET')
    _publish_bytes(ROOT / name, data)
    return ref(ROOT / name)


def bookend(plan, plan_ref, policy_ref, head):
    need(ref(PLAN) == plan_ref and ref(POLICY) == policy_ref
         and subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head
         and subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True) == ''
         and all(ref(REPO / name) == item for name, item in plan['source_owners'].items())
         and all(ref(item['path']) == item for item in (*plan['inputs'].values(),
                                                        *plan['target_refs'].values())),
         'SOURCE_INPUT_BOOKEND')


def run(plan_sha):
    began = time.monotonic()
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    plan, plan_ref, policy_ref, head, _ = inspect(plan_sha, execute=True)
    need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'AMBIENT_TIMER')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ROOT.mkdir()
    claim = publish('CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
                                   'source_head': head, 'cap': cap, 'scope': plan['scope']})
    prior = result = failure = None
    status = 'FAILED_CONSUMED'
    previous_alarm = signal.getsignal(signal.SIGALRM)
    try:
        def expired(*_):
            raise TimeoutError('TA_EMA_DELTA_FOLD1_PRIOR_TIMEOUT')
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL,
            max(1, WALL_SECONDS - 30 - (time.monotonic() - began)))
        coverage = read(plan['inputs']['gather_coverage'])
        state_bound = state.load_bound_five_endpoint_targets()
        future_bound = future.load_bound_future_displacement()
        fit = old.validate_axes(state_bound, future_bound, coverage)
        value = objective.fit_endpoint_priors(state_bound, future_bound,
            axes={'fit_parent_rows': fit['fit_parent_rows'],
                  'fit_decision_ns': fit['fit_decision_ns']},
            fold_id=1, ta_only=True, dedicated_m5_side=True)
        need(objective.require_endpoint_priors(value, fold_id=1, ta_only=True,
             dedicated_m5_side=True) is value
             and value['fit_rows'] == FIT_ROWS
             and value['fit_parent_rows_sha256'] == old.PARENT_SHA
             and value['fit_decision_ns_sha256'] == old.ISSUE_SHA
             and value['loss_contract_sha256'] == LOSS_SHA, 'CURRENT_PRIOR_BINDING')
        bookend(plan, plan_ref, policy_ref, head)
        prior = publish('ENDPOINT_PRIORS.json', value)
        result = publish('RESULT.json', {'status': 'COMPLETE',
            'plan': plan_ref, 'claim': claim, 'policy': policy_ref,
            'source_head': head, 'source_unchanged': True,
            'endpoint_priors': prior, 'fold_id': 1, 'fit_rows': FIT_ROWS,
            'fit_parent_rows_sha256': old.PARENT_SHA,
            'fit_decision_ns_sha256': old.ISSUE_SHA,
            'loss_contract_sha256': LOSS_SHA,
            'feature_view': plan['inputs']['view'],
            'gather_result': plan['inputs']['gather_result'],
            'later_coverage': plan['inputs']['gather_coverage'],
            'target_refs': plan['target_refs'],
            'later_target_values_in_fit': False,
            'model_constructions': 0, 'model_forwards': 0,
            'normalization_fits': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'economic_edge_proven': False})
        status = 'COMPLETE'
    except BaseException as exc:
        failure = publish('FAILED.json', {'status': 'FAILED_CONSUMED',
            'plan': plan_ref, 'claim': claim,
            'error_type': type(exc).__name__, 'error': str(exc)[:1800],
            'endpoint_priors': prior, 'result': result})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        terminal = publish('TERMINAL.json', {
            'status': status, 'exit_code': 0 if status == 'COMPLETE' else 1,
            'no_relaunch': True, 'source_head': head,
            'plan': plan_ref, 'claim': claim, 'endpoint_priors': prior,
            'result': result, 'failure': failure,
            'source_unchanged': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head})
    return terminal, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--build-plan', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    if args.build_plan:
        need(args.plan_sha256 is None and not PLAN.exists() and not ROOT.exists(), 'BUILD_SCOPE')
        _publish_bytes(PLAN, encoded(build_plan()))
        print(json.dumps(ref(PLAN), sort_keys=True))
        return
    need(args.plan_sha256 is not None, 'PLAN_SHA_REQUIRED')
    if args.preflight:
        plan, plan_ref, _, head, clean = inspect(args.plan_sha256, execute=False)
        print(json.dumps({'status': 'PREFLIGHT_PASS' if clean else 'PREFLIGHT_SOURCE_DIRTY',
            'plan': plan_ref, 'source_head': head, 'source_clean': clean,
            'output_root_absent': True, 'scope': plan['scope'], 'claim_created': False},
            sort_keys=True))
        return
    terminal, status = run(args.plan_sha256)
    print(json.dumps({'status': status, 'terminal': terminal}, sort_keys=True))
    if status != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
