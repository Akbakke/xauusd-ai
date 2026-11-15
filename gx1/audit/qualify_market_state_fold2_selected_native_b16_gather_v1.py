"""One capped, read-only Fold 2 full-GC B16 input gather; no model work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from gx1.audit import verify_market_state_native_F_cpu_zero_step_v1 as initial
from gx1.contracts import market_state_fold2_selected_native_private_v1 as fold2
from gx1.contracts import market_state_fold2_selected_native_batches_v1 as batches

REPO = fold2.old.CANONICAL_REPO
POLICY = REPO / 'NEXT_RUN_POLICY.json'
STAGE = 'market_state_fold2_selected_native_b16_gather_20261115_001'
PLAN_PATH = REPO / 'configs/research/MARKET_STATE_FOLD2_SELECTED_NATIVE_B16_GATHER_20261115_001.json'
OUT_ROOT = Path('/home/andre2/GX1_RUNS') / STAGE.upper()
WORKER_PATH = 'gx1/audit/qualify_market_state_fold2_selected_native_b16_gather_v1.py'
MEMORY_BYTES = 12 * 1024**3
SWAP_BYTES = 512 * 1024**2
WALL_SECONDS = 1800
SOURCE_PATHS = tuple(sorted(fold2.required_source_paths() | {WORKER_PATH}))


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('FOLD2_SELECTED_B16_GATHER_' + reason)


def ref(path: Path) -> dict:
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def planned_command(plan_sha256: str) -> list[str]:
    return ['timeout', '--signal=TERM', '--kill-after=10s', '1800s',
            'scripts/gx1_capped_run.sh', '--class', 'producer', '--mem', '12G',
            '--swap', '512M', '--', '.venv/bin/python', '-m',
            'gx1.audit.qualify_market_state_fold2_selected_native_b16_gather_v1',
            '--plan', str(PLAN_PATH), '--plan-sha256', plan_sha256]


def clean_head() -> str:
    need(subprocess.check_output(['git', 'branch', '--show-current'],
         cwd=REPO, text=True).strip() == 'work/gx1-current', 'BRANCH')
    need(not subprocess.check_output(['git', 'status', '--porcelain=v1',
                                      '--untracked-files=all'], cwd=REPO, text=True),
         'SOURCE_DIRTY')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                   cwd=REPO, text=True).strip()


def inspect(plan_ref: dict, *, require_policy: bool) -> dict:
    from gx1.contracts import entry_native_direction_research_campaign_v1 as outer
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    head = clean_head()
    need(plan_ref == ref(PLAN_PATH) and not OUT_ROOT.exists(), 'PLAN_OR_OUTPUT')
    plan = json.loads(PLAN_PATH.read_text())
    need(set(plan) == {'schema_version', 'stage', 'status', 'source_base_head',
                       'source_owners', 'candidate_recipe', 'scope', 'limits',
                       'output_root', 'command_template'}
         and plan['schema_version'] == 'gx1_fold2_selected_native_b16_gather_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['output_root'] == str(OUT_ROOT)
         and plan['scope'] == {'fold_id': 2, 'partition': 'fit',
                               'parent_rows': list(fold2.old.PARENT_ROWS),
                               'batch_size': 16, 'feature_batches': 2,
                               'target_mode': 'selected_native_endpoints',
                               'model_constructions': 0, 'model_forwards': 0,
                               'optimizer_steps': 0, 'test_data_used': False}
         and plan['limits'] == {'memory_max': MEMORY_BYTES,
                                'memory_high': MEMORY_BYTES,
                                'swap': SWAP_BYTES, 'wall_seconds': WALL_SECONDS}
         and plan['command_template'] == planned_command('{plan_sha256}')
         and subprocess.run(['git', 'merge-base', '--is-ancestor',
                             plan['source_base_head'], head], cwd=REPO,
                            capture_output=True).returncode == 0,
         'PLAN_SCOPE')
    need(set(plan['source_owners']) == set(SOURCE_PATHS)
         and all(ref(REPO / name)['sha256'] == digest
                 for name, digest in plan['source_owners'].items()),
         'SOURCE_OWNERS')
    need(plan['candidate_recipe'] == ref(fold2.RECIPE_PATH), 'CANDIDATE_RECIPE')
    recipe = outer.require_research_launch_recipe(plan['candidate_recipe'],
             source_repo=REPO, source_commit=head)
    need(recipe['run_scope'] == fold2.SCOPE
         and recipe['fold_id'] == 2
         and recipe['selected_state_endpoints'] is True
         and recipe['max_private_updates'] == 4
         and outer.guarded_research_power_limit(recipe) == 250,
         'CANDIDATE_SCOPE')
    policy_ref = ref(POLICY)
    if require_policy:
        policy = json.loads(POLICY.read_text())
        gate = policy.get(STAGE)
        need(policy['training_enabled'] is False
             and policy['native_training_enabled'] is False
             and policy['native_launch_authorized'] is False
             and isinstance(gate, dict)
             and gate.get('status') == gate.get('decision') == 'ADMITTED'
             and gate.get('plan') == plan_ref
             and gate.get('worker') == ref(REPO / WORKER_PATH)
             and gate.get('max_invocations') == 1
             and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
             and gate.get('execution_authorized') is True
             and gate.get('data_reads_authorized') is True
             and gate.get('model_forwards_authorized') == 0
             and gate.get('backward_calls_authorized') == 0
             and gate.get('optimizer_steps_authorized') == 0
             and gate.get('training_authorized') is False
             and gate.get('test_data_used') is False
             and gate.get('relaunch_allowed') is False,
             'CLOSED_READ_ONLY_POLICY')
    return {'plan': plan, 'recipe': recipe, 'policy_ref': policy_ref,
            'source_head': head}


def preflight(plan_ref: dict) -> dict:
    checked = inspect(plan_ref, require_policy=False)
    return {'status': 'READY_NO_CLAIM_NO_ARRAY_DECODE',
            'source_head': checked['source_head'],
            'candidate_recipe': checked['plan']['candidate_recipe'],
            'source_owners_checked': len(checked['plan']['source_owners']),
            'model_constructions': 0, 'model_forwards': 0,
            'optimizer_steps': 0, 'test_data_used': False}


def run(plan_ref: dict) -> dict:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_research_producer_execution,
        require_capped_lock_ancestry,
    )
    import torch
    started = time.monotonic()
    checked = inspect(plan_ref, require_policy=True)
    cap = require_capped_cpu_research_producer_execution()
    cap['lock'] = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == MEMORY_BYTES
         and cap['swap'] == SWAP_BYTES and cap['pids'] == 64,
         'CAP')
    need(not torch.cuda.is_initialized(), 'NO_CUDA_BEFORE')
    head = checked['source_head']
    OUT_ROOT.mkdir(parents=True, exist_ok=False)
    claim = None
    try:
        claim = initial.old._write(OUT_ROOT, 'CLAIM.json', {
            'schema_version': 'gx1_fold2_selected_native_b16_gather_claim_v1',
            'plan': plan_ref, 'policy': checked['policy_ref'],
            'source_head': head, 'test_data_used': False},
            limit=initial.old.MAX_JSON_BYTES)
        old_handler = signal.getsignal(signal.SIGALRM)
        def timeout(_signum, _frame):
            raise TimeoutError('FOLD2_SELECTED_B16_GATHER_WALL')
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS - 100)
        try:
            view, packet, constructor, raw, reports, provider, provider_plan = (
                batches.load_two_batches(source_owners=checked['recipe']['source_owners']))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        rows = np.asarray([row for _, batch_rows, _ in raw for row in batch_rows],
                          dtype=np.int64)
        issues = np.asarray([issue for _, _, batch_issues in raw for issue in batch_issues],
                            dtype=np.int64)
        need(rows.tolist() == list(fold2.old.PARENT_ROWS)
             and len(raw) == 2 and all(len(batch_rows) == 16 for _, batch_rows, _ in raw)
             and bool(np.all(issues[1:] > issues[:-1]))
             and all(set(features) == set(raw[0][0]) for features, _, _ in raw)
             and all(len(report['causal_lanes']) == 5 for report in reports)
             and not torch.cuda.is_initialized(), 'PHYSICAL_B16_GATHER')
        need(clean_head() == head and ref(PLAN_PATH) == plan_ref
             and ref(POLICY) == checked['policy_ref']
             and all(ref(REPO / name)['sha256'] == digest
                     for name, digest in checked['plan']['source_owners'].items()),
             'SOURCE_BOOKEND')
        result_ref = initial.old._write(OUT_ROOT, 'RESULT.json', {
            'schema_version': 'gx1_fold2_selected_native_b16_gather_result_v1',
            'status': 'COMPLETE', 'decision': 'PASS_FULL_GC_FOLD2_SELECTED_B16_GATHER',
            'plan': plan_ref, 'claim': claim, 'policy': checked['policy_ref'],
            'source_head': head, 'cap': cap,
            'elapsed_seconds': time.monotonic() - started,
            'view_sha256': view['contract_sha256'],
            'parent_rows_sha256': hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),
            'decision_ns_sha256': hashlib.sha256(issues.astype('<i8').tobytes()).hexdigest(),
            'batch_reports': reports, 'feature_batch_count': 2,
            'model_constructions': 0, 'model_forwards': 0,
            'optimizer_steps': 0, 'cuda_initialized': False,
            'test_data_used': False}, limit=initial.old.MAX_JSON_BYTES)
        terminal = initial.old._write(OUT_ROOT, 'TERMINAL.json', {
            'schema_version': 'gx1_fold2_selected_native_b16_gather_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'claim': claim, 'result': result_ref, 'feature_batch_count': 2,
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'no_relaunch_under_this_plan': True},
            limit=initial.old.MAX_JSON_BYTES)
        return {'result': result_ref, 'terminal': terminal}
    except BaseException as exc:
        failure = initial.old._write(OUT_ROOT, 'FAILED.json', {
            'schema_version': 'gx1_fold2_selected_native_b16_gather_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'elapsed_seconds': time.monotonic() - started,
            'no_relaunch_under_this_plan': True}, limit=initial.old.MAX_JSON_BYTES)
        initial.old._write(OUT_ROOT, 'TERMINAL.json', {
            'schema_version': 'gx1_fold2_selected_native_b16_gather_terminal_v1',
            'status': 'FAILED', 'exit_code': 1, 'plan': plan_ref,
            'claim': claim, 'failure': failure, 'feature_batch_count_upper_bound': 2,
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'no_relaunch_under_this_plan': True},
            limit=initial.old.MAX_JSON_BYTES)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    plan_ref = {'path': args.plan, 'sha256': args.plan_sha256}
    report = preflight(plan_ref) if args.preflight else run(plan_ref)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
