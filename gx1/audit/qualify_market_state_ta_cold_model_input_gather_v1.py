"""One capped cold FIT/SCORE raw feature gather under a frozen prefix normalization binding.

No model is constructed or called. Passing here does not establish post-normalized
model tensors, learning, generalization, or economic edge.
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
import torch

from gx1.audit.market_state_ta_apr_jun_campaign_inputs_v1 import (
    REPO, COLD_FIT_ROWS, COLD_SCORE_START, COLD_SCORE_ROWS, COLD_VIEW_SHA256,
    axis_hash, cold_campaign_input_refs, prepare, read, ref,
)
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    project_ta_input_normalization_contract,
)
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

STAGE = 'ta_m5_cold_model_input_gather_20261117_001'
PLAN = REPO / 'configs/research/TA_M5_COLD_MODEL_INPUT_GATHER_20261117_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/TA_M5_COLD_MODEL_INPUT_GATHER_20261117_001')
TARGETS = Path('/home/andre2/GX1_RUNS/TA_M5_COST_PATH_TARGETS_20261117_001/FIT_TARGETS.npz')
SOURCE_BASE = '7fdd2bc6b2c334a83d45a7b63a8460375c88b6c0'
WALL_SECONDS = 900
BATCH = 16
FEATURE_KEYS = frozenset(('seq_x', 'snap_x', 'ctx_cont', 'ctx_cat',
    'seq_m5', 'seq_m15', 'seq_h1', 'seq_h4', 'seq_d1'))
SOURCES = (
    'gx1/audit/qualify_market_state_ta_cold_model_input_gather_v1.py',
    'gx1/audit/market_state_ta_apr_jun_campaign_inputs_v1.py',
    'gx1/audit/qualify_market_state_mature_prefix_normalization_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/entry_anchor_span_feature_inputs_v1.py',
    'gx1/contracts/entry_model_native_input_normalization_v1.py',
    'gx1/contracts/entry_anchor_span_normalization_v1.py',
    'gx1/contracts/entry_native_direction_prefix_units_v1.py',
    'gx1/contracts/entry_exit_feature_surface_v1.py',
    'gx1/contracts/market_state_ta_fold2_m5_fit_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_qualify_market_state_ta_cold_model_input_gather_v1.py',
)


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_COLD_MODEL_INPUT_GATHER_' + reason)


def inputs():
    return {**cold_campaign_input_refs(), 'fit_targets': ref(TARGETS)}


def build_plan():
    refs = inputs()
    return {'schema_version': 'gx1_ta_m5_cold_model_input_gather_plan_v1',
        'stage': STAGE, 'source_base_commit': SOURCE_BASE,
        'source_owners': {name: ref(REPO / name) for name in SOURCES},
        'inputs': refs, 'output_root': str(ROOT),
        'scope': {'view_sha256': COLD_VIEW_SHA256,
            'fit_rows': COLD_FIT_ROWS, 'score_parent_start': COLD_SCORE_START,
            'score_rows': COLD_SCORE_ROWS, 'batch_size': BATCH,
            'fit_tail_rows': COLD_FIT_ROWS % BATCH,
            'score_tail_rows': COLD_SCORE_ROWS % BATCH,
            'batch_names': ['FIT_B16', 'FIT_TAIL6', 'SCORE_B16', 'SCORE_TAIL1'],
            'raw_values_not_normalized': True, 'normalization_fits': 0,
            'model_constructions': 0, 'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'economic_edge_proven': False},
        'execution': {'runner': 'scripts/gx1_capped_run.sh', 'class': 'audit',
            'memory': '4G', 'swap': '512M', 'max_seconds': WALL_SECONDS,
            'max_invocations': 1, 'training_authorized': False,
            'test_authorized': False, 'broker_access_allowed': False}}


def inspect(plan_sha, *, execute):
    plan_ref = ref(PLAN)
    plan = read(plan_ref)
    need(plan_ref['sha256'] == plan_sha and plan == build_plan(), 'PLAN_IDENTITY')
    need(not ROOT.exists(), 'ONE_USE_ROOT')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
    clean = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True) == ''
    ancestor = subprocess.run(['git', 'merge-base', '--is-ancestor', SOURCE_BASE, head],
        cwd=REPO, capture_output=True).returncode == 0
    need(Path.cwd().resolve() == REPO and branch == 'work/gx1-current' and ancestor,
         'SOURCE_BRANCH_CWD')
    if not execute:
        return plan, plan_ref, None, head, clean
    need(clean, 'CLEAN_SOURCE')
    policy_ref = ref(REPO / 'NEXT_RUN_POLICY.json')
    policy = read(policy_ref)
    need(all(policy.get(key) is False for key in ('training_enabled',
        'native_training_enabled', 'native_launch_authorized',
        'full_epoch_training_allowed', 'full_val_allowed'))
        and policy['exit_training_and_evaluation_this_wave'] is False,
        'TRAINING_AND_TEST_CLOSED')
    gate = policy.get(STAGE, {})
    need(gate.get('status') == 'ADMITTED' and gate.get('plan') == plan_ref
         and gate.get('scope') == plan['scope']
         and gate.get('execution_authorized') is True
         and gate.get('data_reads_authorized') is True
         and gate.get('training_authorized') is False
         and gate.get('test_authorized') is False
         and gate.get('broker_access_allowed') is False
         and gate.get('model_forwards_authorized') == 0
         and gate.get('optimizer_steps_authorized') == 0
         and gate.get('max_invocations') == gate.get('remaining_invocations') == 1
         and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
         and gate.get('relaunch_allowed') is False, 'ONE_USE_ADMISSION')
    return plan, plan_ref, policy_ref, head, clean


def verify_axes(context, target_ref):
    axes, selected, view = context['axes'], context['selected'], context['view']
    prefix = view['prefix']
    need(context['fit_rows'] == COLD_FIT_ROWS and context['score_rows'] == COLD_SCORE_ROWS
         and context['fit_fold_id'] == context['score_fold_id'] == 1
         and context['score_partition'] == 'score'
         and view['contract_sha256'] == COLD_VIEW_SHA256
         and len(selected) == COLD_SCORE_ROWS
         and np.array_equal(selected, np.arange(COLD_SCORE_START,
             COLD_SCORE_START + COLD_SCORE_ROWS, dtype=np.int64))
         and axis_hash(axes['parent_rows'][:COLD_FIT_ROWS]) == prefix['parent_rows_sha256']
         and axis_hash(axes['decision_ns'][:COLD_FIT_ROWS]) == prefix['decision_ns_sha256'],
         'FIT_SCORE_SCOPE')
    coverage = read(context['input_refs']['coverage'])
    need(axis_hash(axes['parent_rows'][selected]) == coverage['selected_parent_rows_sha256']
         and axis_hash(axes['decision_ns'][selected]) == coverage['selected_decision_ns_sha256'],
         'LATER_AXES')
    with np.load(target_ref['path'], allow_pickle=False) as archive:
        parents, issues = archive['parent_rows'], archive['issue_ns']
    need(len(parents) == len(issues) == 47338
         and np.array_equal(axes['parent_rows'][:COLD_FIT_ROWS], parents[:COLD_FIT_ROWS])
         and np.array_equal(axes['decision_ns'][:COLD_FIT_ROWS], issues[:COLD_FIT_ROWS])
         and np.array_equal(axes['parent_rows'][selected], parents[COLD_SCORE_START:47338])
         and np.array_equal(axes['decision_ns'][selected], issues[COLD_SCORE_START:47338]),
         'COST_TARGET_AXIS_ONLY')
    packet = context['packet']
    need(view['prefix_normalization']['binding'] == context['input_refs']['packet']
         and packet['unobserved_units_profile'] ==
             packet['base_normalization']['anchor_span_extension']['unobserved_units_profile'],
         'FOLD1_UNITS_BINDING')
    projected = project_ta_input_normalization_contract(packet['base_normalization'],
        unobserved_units_profile=packet['unobserved_units_profile'])
    need(projected['contract_sha256'] ==
         view['prefix_normalization']['ta_base_normalization_sha256'],
         'TA_PREFIX_NORMALIZATION')
    return projected['contract_sha256']


def summarize_batch(name, features, rows, issues):
    need(set(features) == FEATURE_KEYS and len(rows) == len(issues) > 0, 'FEATURE_KEYS_AXIS')
    details = {}
    for key in sorted(features):
        tensor = features[key]
        dtype = torch.int64 if key == 'ctx_cat' else torch.float32
        need(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu'
             and tensor.dtype == dtype and tensor.shape[0] == len(rows)
             and bool(torch.isfinite(tensor).all()),
             'FEATURE_TENSOR_' + key)
        array = tensor.numpy()
        details[key] = {'shape': list(array.shape), 'dtype': str(array.dtype),
            'sha256': hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
    return {'name': name, 'rows': len(rows),
        'parent_rows_sha256': axis_hash(rows), 'decision_ns_sha256': axis_hash(issues),
        'features': details, 'raw_values_not_normalized': True}


def gather_four(context):
    axes, selected = context['axes'], context['selected']
    cases = (('FIT_B16', 'TRAIN', np.arange(BATCH, dtype=np.int64)),
        ('FIT_TAIL6', 'TRAIN', np.arange(COLD_FIT_ROWS - COLD_FIT_ROWS % BATCH,
                                     COLD_FIT_ROWS, dtype=np.int64)),
        ('SCORE_B16', 'SCORE', selected[:BATCH]),
        ('SCORE_TAIL1', 'SCORE', selected[-1:]))
    reports = []
    for name, label, positions in cases:
        rows, issues = axes['parent_rows'][positions], axes['decision_ns'][positions]
        reports.append(summarize_batch(name, context['features'](label, rows, issues),
                                       rows, issues))
    need([x['rows'] for x in reports] == [16, 6, 16, 1]
         and all({k: tuple(v['shape'][1:]) for k, v in x['features'].items()}
                 == {k: tuple(v['shape'][1:]) for k, v in reports[0]['features'].items()}
                 for x in reports[1:]), 'FOUR_BATCH_SCHEMA')
    return reports


def publish(name, value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    need(len(raw) <= 4 * 1024**2, 'JSON_BUDGET')
    _publish_bytes(ROOT / name, raw)
    return ref(ROOT / name)


def bookend(plan_ref, policy_ref, head, plan):
    need(ref(PLAN) == plan_ref and ref(REPO / 'NEXT_RUN_POLICY.json') == policy_ref
         and subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head
         and subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True) == ''
         and all(ref(REPO / path) == binding for path, binding in plan['source_owners'].items())
         and inputs() == plan['inputs'], 'SOURCE_INPUT_BOOKEND')


def run(plan_sha):
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    plan, plan_ref, policy_ref, head, _ = inspect(plan_sha, execute=True)
    need(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'AMBIENT_TIMER')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ROOT.mkdir()
    claim = publish('CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
        'source_head': head, 'cap': cap, 'scope': plan['scope']})
    started = time.monotonic()
    context = result = failure = None
    status = 'FAILED_CONSUMED'
    old_alarm = signal.getsignal(signal.SIGALRM)
    def expired(*_): raise TimeoutError('TA_COLD_MODEL_INPUT_GATHER_TIMEOUT')
    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS - 30)
        cold_refs = cold_campaign_input_refs()
        context = prepare(device=torch.device('cpu'), cold_inputs=cold_refs)
        need(context['input_refs'] == cold_refs, 'PREPARED_INPUT_REFS')
        norm_sha = verify_axes(context, plan['inputs']['fit_targets'])
        batches = gather_four(context)
        context['release']()
        context = None
        bookend(plan_ref, policy_ref, head, plan)
        result = publish('RESULT.json', {'status': 'COMPLETE', 'plan': plan_ref,
            'claim': claim, 'policy': policy_ref, 'source_head': head,
            'source_unchanged': True, 'inputs': plan['inputs'],
            'view_sha256': COLD_VIEW_SHA256, 'ta_prefix_normalization_sha256': norm_sha,
            'batches': batches, 'raw_values_not_normalized': True,
            'model_constructions': 0, 'model_forwards': 0,
            'optimizer_steps': 0, 'normalization_fits': 0,
            'test_data_used': False, 'economic_edge_proven': False,
            'elapsed_seconds': time.monotonic() - started})
        status = 'COMPLETE'
    except BaseException as error:
        failure = publish('FAILED.json', {'status': 'FAILED_CONSUMED',
            'plan': plan_ref, 'claim': claim,
            'error': type(error).__name__ + ':' + str(error)[:1800], 'result': result})
    finally:
        if context is not None:
            try:
                context['release']()
            except BaseException as error:
                if failure is None:
                    failure = publish('FAILED.json', {'status': 'FAILED_CONSUMED',
                        'plan': plan_ref, 'claim': claim,
                        'error': type(error).__name__ + ':' + str(error)[:1800],
                        'result': result})
                status = 'FAILED_CONSUMED'
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        terminal = publish('TERMINAL.json', {'status': status,
            'exit_code': 0 if status == 'COMPLETE' else 1,
            'plan': plan_ref, 'claim': claim, 'result': result, 'failure': failure,
            'source_unchanged': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                cwd=REPO, text=True).strip() == head,
            'no_relaunch': True, 'remaining_invocations': 0})
    return status, terminal


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    if args.preflight:
        plan, plan_ref, _, head, clean = inspect(args.plan_sha256, execute=False)
        print(json.dumps({'status': 'PREFLIGHT_PASS' if clean else 'PREFLIGHT_SOURCE_DIRTY',
            'plan': plan_ref, 'source_head': head, 'source_clean': clean,
            'scope': plan['scope'], 'claim_created': False}, sort_keys=True))
        return
    status, terminal = run(args.plan_sha256)
    print(json.dumps({'status': status, 'terminal': terminal}, sort_keys=True))
    if status != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
