"""One capped TRAIN-only fit of the twelve added span statistics in folds 3-5.

The completed fold-1/2 normalization is reused. Full sidecars are sliced only
at each immutable mature TRAIN cutoff; all 253 old statistics remain exact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from gx1.audit import qualify_market_state_ta_anchor_span_normalization_v1 as owner
from gx1.audit import qualify_market_state_ta_fivefold_span_extension_v1 as spans
from gx1.contracts import entry_anchor_span_normalization_v1 as extension
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry,
)
from gx1.contracts.immutable_event_authority_v1 import _fsync_directory, _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
STAGE = 'market_state_ta_fivefold_span_normalization_20261114_001'
PLAN = REPO / 'configs/research' / (STAGE.upper() + '.json')
ROOT = RUNS / STAGE.upper()
POLICY = REPO / 'NEXT_RUN_POLICY.json'
FOLDS = (3, 4, 5)
SURFACE_LANES = {'signal': 'local', 'mtf_m5': 'M5', 'mtf_m15': 'M15',
                 'mtf_h1': 'H1', 'mtf_h4': 'H4', 'mtf_d1': 'D1'}
SOURCE_PATHS = frozenset({
    'gx1/audit/qualify_market_state_ta_fivefold_span_normalization_v1.py',
    'gx1/audit/qualify_market_state_ta_anchor_span_normalization_v1.py',
    'gx1/audit/qualify_market_state_ta_fivefold_span_extension_v1.py',
    'gx1/audit/qualify_market_state_ta_anchor_span_inputs_v1.py',
    'gx1/contracts/entry_anchor_span_normalization_v1.py',
    'gx1/contracts/entry_model_native_input_normalization_v1.py',
    'gx1/contracts/entry_model_native_training_objective_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/features/entry_specialist_feature_groups_v1.py',
    'gx1/models/entry_v10/entry_v10_input_normalization.py',
    'scripts/gx1_capped_run.sh',
})
SCOPE = {'fold_ids': list(FOLDS), 'surface_fit_calls': 18,
         'new_field_statistics': 36, 'old_field_fit_calls': 0,
         'old_statistics_refitted': False, 'sidecar_array_decodes': 24,
         'model_constructions': 0, 'model_forwards': 0,
         'optimizer_steps': 0, 'checkpoint_reads': 0,
         'target_labels_used': False, 'test_data_used': False}
CUTOFF_LIMIT = 1767225600000000000
MEMORY_MAX = 4 * 1024**3
SWAP_MAX = 512 * 1024**2
WALL_SECONDS = 300
INTERNAL_SECONDS = 280


def need(ok: bool, label: str) -> None:
    if not ok:
        raise RuntimeError('FIVEFOLD_SPAN_NORMALIZATION_' + label)


def ref(path: Path) -> dict:
    return spans.ref(path)


def checked(item: dict) -> dict:
    return spans.checked(item)


def planned_command(plan_sha: str) -> list[str]:
    return ['timeout', '--signal=TERM', '--kill-after=10s', '300s',
            'scripts/gx1_capped_run.sh', '--class', 'audit', '--mem', '4G',
            '--swap', '512M', '--', '.venv/bin/python', '-m',
            'gx1.audit.qualify_market_state_ta_fivefold_span_normalization_v1',
            '--plan', str(PLAN), '--plan-sha256', plan_sha]


def clean_head() -> str:
    need(Path.cwd().resolve() == REPO, 'WORKDIR')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], text=True)
    need(branch == 'work/gx1-current' and not dirty and len(head) == 40,
         'CLEAN_SOURCE')
    return head


def publish(name: str, value: dict) -> dict:
    need('/' not in name and name not in ('', '.', '..'), 'OUTPUT_NAME')
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    need(len(raw) <= 4 * 1024**2, 'JSON_BUDGET')
    _publish_bytes(ROOT / name, raw)
    return ref(ROOT / name)


def preflight(plan_path: Path, plan_sha: str, *, admitted: bool = False) -> tuple:
    head = clean_head()
    need(plan_path == PLAN and len(plan_sha) == 64 and spans.sha(PLAN) == plan_sha
         and not ROOT.exists(), 'PLAN_IDENTITY')
    plan = json.loads(PLAN.read_text())
    coverage = checked(plan['coverage'])
    physical = checked(plan['physical_result'])
    terminal = checked(plan['physical_terminal'])
    need(plan['schema_version'] == 'gx1_ta_fivefold_span_normalization_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['output_root'] == str(ROOT)
         and plan['command_template'] == planned_command('{plan_sha256}')
         and plan['scope'] == SCOPE and plan['max_invocations'] == 1
         and plan['test_data_used'] is False
         and coverage['fit_proof_checks_total'] == 30
         and coverage['score_end_exclusive_utc'] == '2026-01-01T00:00:00+00:00'
         and physical['status'] == 'COMPLETE_SIX_LANES_INPUT_ONLY'
         and physical['source_unchanged'] is True
         and physical['normalization_fitted'] is False
         and physical['model_fit'] is False
         and physical['test_data_used'] is False
         and terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
         and terminal['test_data_used'] is False and terminal['no_relaunch'] is True
         and terminal['result'] == plan['physical_result']
         and set(plan['folds']) == {str(i) for i in FOLDS}
         and len(plan['source_base_head']) == 40
         and subprocess.run(['git', 'merge-base', '--is-ancestor',
                             plan['source_base_head'], head],
                            capture_output=True).returncode == 0,
         'PLAN_SCOPE')
    need(set(plan['source_owners']) == SOURCE_PATHS
         and all(spans.sha(REPO / path) == digest
                 for path, digest in plan['source_owners'].items()),
         'SOURCE_OWNERS')
    blocks = {}
    for lane in spans.LANES:
        item = physical['lanes'][lane]['full_artifact']
        need(item == plan['sidecars'][lane], 'SIDECAR_REF:' + lane)
        blocks[lane] = spans.archive(item,
            expected_range=coverage['lanes'][lane]['required_superrange'])
    parents = {}
    for fold_id in FOLDS:
        fold = plan['folds'][str(fold_id)]
        packet = checked(fold['inputs']['parent'])
        parent = packet['base_normalization']
        binding = extension.FOLD_BINDINGS.get(parent['contract_sha256'])
        cutoff = packet['prefix_qualification']['proof']['prefix']['cutoff_ns']
        need(binding is not None and binding[0] == fold_id
             and fold['inputs']['parent'] == binding[1]
             and fold['inputs']['physical_result'] == binding[2] == plan['physical_result']
             and fold['cutoff_ns'] == cutoff and 0 < cutoff < CUTOFF_LIMIT
             and fold['output_root'] == str(ROOT / ('FOLD_' + str(fold_id)))
             and set(fold['populations']) == set(SURFACE_LANES),
             'FOLD_BINDING:' + str(fold_id))
        norm.require_input_normalization_contract(
            parent, expected_field_names={s: v['field_names']
                                          for s, v in parent['surfaces'].items()},
            expected_ctx_cat_names=parent['ctx_cat']['field_names'],
            expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
            unobserved_units_profile=None)
        for surface, lane in SURFACE_LANES.items():
            item = fold['populations'][surface]
            block = blocks[lane]
            n = parent['surfaces'][surface]['fit_row_count']
            start = coverage['lanes'][lane]['required_superrange'][0]
            rows = block['row_indices'][:n]
            available = (block['times_ns'] + owner.io.BAR_NS[
                'M5' if lane == 'local' else lane] * 10**9)
            proof = next(x for x in coverage['lanes'][lane]['fit_proof_checks']
                         if x['fold_id'] == fold_id)
            need(item['lane'] == lane and item['artifact'] == plan['sidecars'][lane]
                 and item['artifact_stat'] == owner.io.stat(item['artifact']['path'])
                 and item['prefix_from_full_sidecar'] is True
                 and item['new_field_names'] == extension.span_names(surface)
                 and item['row_bounds'] == [start, start + n]
                 and item['population'] == {
                     'selected_unique_row_count': n,
                     'selected_row_indices_sha256': owner.io.row_sha(rows, lane),
                     'available_ns_sha256': owner.io.axis_sha(available[:n])}
                 and proof['fit_input_rows'] == n and proof['row_hash_match'] is True
                 and n < len(available) and np.all(available[:n] < cutoff)
                 and available[n] >= cutoff,
                 'FIT_PREFIX:' + str(fold_id) + ':' + lane)
        parents[str(fold_id)] = parent
    policy_ref = None
    if admitted:
        cap = require_capped_cpu_audit_execution()
        require_capped_lock_ancestry()
        need(cap['memory_max'] == cap['memory_high'] == MEMORY_MAX
             and cap['swap'] == SWAP_MAX, 'CAP_LIMITS')
        policy_ref = ref(POLICY)
        policy = checked(policy_ref)
        gate = policy.get(STAGE, {})
        need(all(policy[key] is False for key in
                 ('training_enabled', 'native_training_enabled',
                  'full_epoch_training_allowed', 'full_val_allowed',
                  'native_launch_authorized'))
             and gate.get('status') == gate.get('decision') == 'ADMITTED'
             and gate.get('plan') == {'path': str(PLAN), 'sha256': plan_sha}
             and gate.get('scope') == SCOPE
             and gate.get('output_root') == str(ROOT)
             and gate.get('execution_authorized') is True
             and gate.get('data_reads_authorized') is True
             and gate.get('normalization_fits_authorized') == 18
             and gate.get('training_authorized') is False
             and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
             and gate.get('max_invocations') == 1
             and gate.get('relaunch_allowed') is False
             and gate.get('memory_max') == '4G'
             and gate.get('swap_max') == '512M'
             and gate.get('wall_seconds') == WALL_SECONDS
             and gate.get('test_data_used') is False,
             'POLICY_GATE')
    return plan, parents, head, policy_ref


def bookend(plan: dict, head: str, plan_sha: str, policy_ref: dict) -> None:
    need(clean_head() == head and spans.sha(PLAN) == plan_sha
         and ref(POLICY) == policy_ref
         and all(spans.sha(REPO / path) == digest
                 for path, digest in plan['source_owners'].items())
         and all(ref(Path(item['path'])) == item for item in
                 [plan['coverage'], plan['physical_result'], plan['physical_terminal'],
                  *plan['sidecars'].values(),
                  *(plan['folds'][str(i)]['inputs']['parent'] for i in FOLDS)]),
         'SOURCE_BOOKEND')


def wall_limit(_signum, _frame) -> None:
    raise TimeoutError('FIVEFOLD_SPAN_NORMALIZATION_INTERNAL_WALL_LIMIT')


def run(plan_path: Path, plan_sha: str) -> dict:
    started = time.monotonic()
    plan, parents, head, policy_ref = preflight(plan_path, plan_sha, admitted=True)
    need(policy_ref is not None, 'ADMISSION_REQUIRED')
    ROOT.mkdir(parents=True, exist_ok=False)
    _fsync_directory(ROOT.parent)
    completed = {}
    old_handler = signal.getsignal(signal.SIGALRM)
    claim = None
    try:
        signal.signal(signal.SIGALRM, wall_limit)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_SECONDS)
        claim = publish('CLAIM.json', {
            'schema_version': 'gx1_ta_fivefold_span_normalization_claim_v1',
            'plan': {'path': str(PLAN), 'sha256': plan_sha},
            'policy': policy_ref, 'source_head': head,
            'invocation': 1, 'test_data_used': False})
        for fold_id in FOLDS:
            fold = plan['folds'][str(fold_id)]
            out = Path(fold['output_root'])
            out.mkdir()
            measured = owner.measure(fold, parent=parents[str(fold_id)],
                                     cutoff=fold['cutoff_ns'], output_root=out)
            need(measured['surface_fit_calls'] == 6
                 and measured['new_statistics'] == 12
                 and measured['old_field_fit_calls'] == 0
                 and measured['normalization_qualified'] is True
                 and measured['old_field_preservation']['all_old_stat_vectors_exact'] is True
                 and measured['units_contract_sha256'] is None,
                 'FOLD_FIT:' + str(fold_id))
            completed[str(fold_id)] = {'measurement': measured,
                'cutoff_ns': fold['cutoff_ns'],
                'parent': fold['inputs']['parent']}
        bookend(plan, head, plan_sha, policy_ref)
        need(time.monotonic() - started < INTERNAL_SECONDS, 'WALL_BOOKEND')
        result = publish('RESULT.json', {
            'schema_version': 'gx1_ta_fivefold_span_normalization_result_v1',
            'status': 'COMPLETE_FOLDS_3_TO_5_NEW_STATS_ONLY',
            'plan': {'path': str(PLAN), 'sha256': plan_sha},
            'claim': claim, 'policy': policy_ref,
            'source_head': head, 'source_unchanged': True,
            'folds': completed, 'counts': SCOPE,
            'elapsed_seconds': time.monotonic() - started,
            'model_fit': False, 'test_data_used': False})
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminal = publish('TERMINAL.json', {
            'schema_version': 'gx1_ta_fivefold_span_normalization_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0,
            'result': result, 'claim': claim,
            'plan': {'path': str(PLAN), 'sha256': plan_sha},
            'no_relaunch': True, 'test_data_used': False})
        return {'result': result, 'terminal': terminal}
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        failed = publish('FAILED.json', {
            'schema_version': 'gx1_ta_fivefold_span_normalization_failure_v1',
            'plan': {'path': str(PLAN), 'sha256': plan_sha},
            'claim': claim, 'completed_folds': completed,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'elapsed_seconds': time.monotonic() - started,
            'no_relaunch': True, 'test_data_used': False})
        publish('TERMINAL.json', {
            'schema_version': 'gx1_ta_fivefold_span_normalization_terminal_v1',
            'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'failure': failed, 'claim': claim,
            'completed_folds': len(completed),
            'plan': {'path': str(PLAN), 'sha256': plan_sha},
            'no_relaunch': True, 'test_data_used': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        plan, _, head, _ = preflight(args.plan, args.plan_sha256)
        print(json.dumps({'status': 'READY_NO_CLAIM_NO_FIT',
                          'plan': {'path': str(PLAN), 'sha256': args.plan_sha256},
                          'source_head': head, 'folds': list(FOLDS),
                          'planned_command': planned_command(args.plan_sha256),
                          'test_data_used': False}, sort_keys=True))
        return 0
    print(json.dumps(run(args.plan, args.plan_sha256), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
