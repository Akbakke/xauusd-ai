"""One-use TRAIN-only frozen references for future M5-tick market states.

Reads one COMPLETE candidate M1 tape and the pinned native model issue axis.
No target rebuild, model fit, TEST access, Entry, Exit, or trading authority.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from gx1.audit import market_state_time_query_reference_core_v1 as core
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as binding
from gx1.contracts.market_state_time_query_target_v1 import TimeQueryTargetTape
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry,
)
from gx1.contracts.immutable_event_authority_v1 import _fsync_directory, _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
STAGE = 'market_state_time_query_references_20261107_001'
SCHEMA = 'gx1_market_state_time_query_references_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
POLICY = REPO / 'NEXT_RUN_POLICY.json'
TAPE_ROOT = RUNS / 'MARKET_STATE_TIME_QUERY_TAPE_20261107_001'
TAPE_REFS = {
    'candidate_plan': (REPO / 'configs/research/MARKET_STATE_TIME_QUERY_TAPE_20261107_001.json',
                       '3dcf7c67a03a9fbba3f8074e17210bbcab3b28f8a8193f8ca4cba29d192423f0'),
    'candidate_claim': (TAPE_ROOT / 'CLAIM.json',
                        '2ddc50df15e1208110050967efe2d93a4ad6e845b085cd51360d471cc983c933'),
    'candidate_result': (TAPE_ROOT / 'RESULT.json',
                         'd36f84f645a66d208e53583255f6d0dbb688ef154f4994b80cf9a0c14c683a86'),
    'candidate_terminal': (TAPE_ROOT / 'TERMINAL.json',
                           '2635f51cdb5c592913cd583c1b5cf73b7bd6e64e8f3089ee3d31026e978e3cff'),
    'candidate_tape': (TAPE_ROOT / 'TRAIN_CANDIDATE_TAPE.npz',
                       '86aff221bf63532b67efe7c5770daa986bdebe916f149dd99961dacde1043b8b'),
}
INPUT_REFS = {
    **{name: {'path': str(path), 'sha256': sha} for name, (path, sha) in TAPE_REFS.items()},
    **{'binding_' + name: {'path': path, 'sha256': sha}
       for name, (path, sha) in binding.REFS.items()},
}
SOURCE_PATHS = frozenset({
    'gx1/audit/verify_market_state_time_query_references_v1.py',
    'gx1/audit/market_state_time_query_reference_core_v1.py',
    'gx1/contracts/market_state_time_query_target_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_five_endpoint_contract_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
})
NUMPY_VERSION = '2.2.6'
EXECUTION = {
    'class': 'audit', 'runner': 'scripts/gx1_capped_run.sh',
    'memory': '4G', 'swap': '512M', 'timeout_seconds': 1800,
    'kill_after_seconds': 10,
    'module': 'gx1.audit.verify_market_state_time_query_references_v1',
    'temporary_target_tapes': 0, 'published_target_tapes': 0, 'model_fits': 0,
}
LIMITS = {
    'memory_max': 4 * 1024**3, 'memory_high': 4 * 1024**3,
    'swap': 512 * 1024**2, 'pids': 64, 'numerical_threads': 1,
    'internal_wall_seconds': 1680, 'chunk_rows': 8192,
    'max_output_bytes': 16 * 1024**2,
}
CONTRACT = {
    'scope': 'five_chronological_TRAIN_folds_on_exact_118249_native_model_issues',
    'query': 'all_future_M5_ticks_k_1_to_native_K_per_TF',
    'native_K': [1, 3, 12, 48, 288],
    'prior': 'frozen_earlier_mature_TF_by_k_plus_one_activity_side_and_separate_singleton',
    'persistence': 'exact_causal_M1_state_at_issue_activity_or_certified_side_only',
    'comparison': 'paired_same_future_cells_prior_and_selective_persistence',
    'fit_axes': 'existing_mature_oof_axes_excludes_last_D1_day_before_each_fit_boundary',
    'score_axes': 'existing_native_model_score_issues_only',
    'overlap': 'weekly_raw_query_cell_sums_descriptive_not_per_issue_normalized_or_independent',
    'primary_direction_reference': 'M5_k1_certified_side_Brier',
    'candidate_tape_is_not_selected_training_target': True,
    'test_data_used': False,
}
EXPECTED_FIT_ROWS = (29574, 47338, 64615, 82152, 100224)
EXPECTED_SCORE_ROWS = (17753, 17288, 17537, 18072, 17750)  # exact bound model axes
EXPECTED_SCORE_TOTAL = 88400
TAPE_KEYS = frozenset(('validity_ns', 'm1_source_row', 'allowed_set',
                       'path_observed', 'gap_kind'))


def _need(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError('MARKET_STATE_TIME_QUERY_REFERENCES_' + code)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(name: str) -> dict:
    return INPUT_REFS[name]


def _checked_bytes(name: str, max_bytes: int) -> bytes:
    path = Path(_ref(name)['path'])
    size = path.stat().st_size
    _need(0 < size <= max_bytes, 'REF_SIZE:' + name)
    raw = path.read_bytes()
    _need(len(raw) == size and hashlib.sha256(raw).hexdigest() == _ref(name)['sha256'],
          'REF_HASH:' + name)
    return raw


def _clean_head() -> str:
    branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    changed = subprocess.check_output(['git', 'status', '--porcelain',
                                       '--untracked-files=all'], text=True)
    _need(branch == 'work/gx1-current' and len(head) == 40 and not changed,
          'CLEAN_CURRENT')
    return head


def _small_chain() -> None:
    plan = json.loads(_checked_bytes('candidate_plan', 1024**2))
    claim = json.loads(_checked_bytes('candidate_claim', 1024**2))
    result = json.loads(_checked_bytes('candidate_result', 1024**2))
    terminal = json.loads(_checked_bytes('candidate_terminal', 1024**2))
    _need(plan.get('stage') == 'market_state_time_query_tape_20261107_001'
          and claim.get('plan') == _ref('candidate_plan')
          and result.get('plan') == _ref('candidate_plan')
          and result.get('claim') == _ref('candidate_claim')
          and result.get('candidate_tape', {}).get('path') == _ref('candidate_tape')['path']
          and result['candidate_tape'].get('sha256') == _ref('candidate_tape')['sha256']
          and result.get('status') == 'COMPLETE_CANDIDATE_TAPE_ENDPOINT_PARITY'
          and result.get('model_fits') == 0 and result.get('test_data_used') is False
          and result.get('target_selection') is False
          and all(x.get('bit_mismatches') == x.get('m1_source_row_mismatches') == 0
                  for x in result.get('selected_endpoint_parity', {}).values())
          and len(result.get('selected_endpoint_parity', {})) == 5
          and terminal.get('status') == 'COMPLETE' and terminal.get('exit_code') == 0
          and terminal.get('result') == _ref('candidate_result')
          and terminal.get('claim') == _ref('candidate_claim')
          and terminal.get('candidate_tape') == result['candidate_tape']
          and terminal.get('test_data_used') is False,
          'CANDIDATE_CHAIN')
    _need(Path(_ref('candidate_tape')['path']).stat().st_size
          == result['candidate_tape']['bytes'] == 4169474,
          'CANDIDATE_TAPE_SIZE')
    # Check every small receipt before CLAIM without reading model/target payloads.
    for name in ('mask_result', 'mask_terminal', 'mask_plan',
                 'selected_plan', 'target_source_plan', 'model_plan'):
        _checked_bytes('binding_' + name, 4 * 1024**2)
    for name in ('mask', 'model_axes'):
        _need(Path(_ref('binding_' + name)['path']).stat().st_size > 0,
              'BINDING_PAYLOAD_PRESENT:' + name)


def planned_command(plan_sha256: str) -> list[str]:
    return ['timeout', '--signal=TERM', '--kill-after=10s', '1800s',
            EXECUTION['runner'], '--class', 'audit', '--mem', '4G',
            '--swap', '512M', '--', '.venv/bin/python', '-m', EXECUTION['module'],
            '--plan', str(PLAN_PATH), '--plan-sha256', plan_sha256,
            '--out-dir', str(OUT_ROOT)]


def preflight(plan_path: Path, digest: str, out_dir: Path,
              *, admitted: bool = False) -> tuple[dict, dict, str, dict | None]:
    head = _clean_head()
    _need(plan_path == PLAN_PATH and out_dir == OUT_ROOT and not out_dir.exists()
          and len(digest) == 64 and _sha(plan_path) == digest,
          'PLAN_IDENTITY')
    plan = json.loads(plan_path.read_text())
    _small_chain()
    _need(plan.get('schema_version') == SCHEMA
          and plan.get('stage') == STAGE
          and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and plan.get('output_root') == str(OUT_ROOT)
          and plan.get('max_invocations') == 1
          and plan.get('input_refs') == INPUT_REFS
          and plan.get('source_owners', {}).keys() == SOURCE_PATHS
          and all(_sha(REPO / name) == digest_ for name, digest_ in
                  plan['source_owners'].items())
          and plan.get('execution') == EXECUTION
          and plan.get('limits') == LIMITS
          and plan.get('metric_contract') == CONTRACT
          and plan.get('numpy_version') == NUMPY_VERSION == np.__version__
          and plan.get('expected_model_rows') == 118249
          and plan.get('expected_fit_rows') == list(EXPECTED_FIT_ROWS)
          and plan.get('expected_score_rows') == list(EXPECTED_SCORE_ROWS)
          and plan.get('command_template') == planned_command('{plan_sha256}')
          and plan.get('read_only') is True and plan.get('test_data_used') is False
          and isinstance(plan.get('source_base_head'), str)
          and len(plan['source_base_head']) == 40
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
                              plan['source_base_head'], head],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0,
          'PLAN_SCOPE')
    plan_ref = {'path': str(plan_path), 'sha256': digest}
    policy_ref = None
    if admitted:
        cap = require_capped_cpu_audit_execution()
        lock = require_capped_lock_ancestry()
        _need(cap['memory_max'] == cap['memory_high'] == LIMITS['memory_max']
              and cap['swap'] == LIMITS['swap'] and cap['pids'] == LIMITS['pids']
              and cap['numerical_threads'] == 1
              and lock['scope'] == cap['cgroup_path'][len('/sys/fs/cgroup'):],
              'CAP_LIMITS')
        policy_ref = {'path': str(POLICY), 'sha256': _sha(POLICY)}
        policy = json.loads(POLICY.read_text())
        gate = policy.get(STAGE, {})
        tape_gate = policy.get('market_state_time_query_tape_20261107_001', {})
        _need(_sha(POLICY) == policy_ref['sha256']
              and all(policy.get(k) is False for k in
                      ('training_enabled', 'native_training_enabled',
                       'native_launch_authorized', 'full_val_allowed'))
              and tape_gate.get('status') == 'CONSUMED_COMPLETE_CANDIDATE_M1_TAPE_ZERO_ENDPOINT_MISMATCH'
              and tape_gate.get('candidate_tape') == _ref('candidate_tape') | {'bytes': 4169474}
              and gate.get('status') == gate.get('decision') == 'ADMITTED'
              and gate.get('plan') == plan_ref
              and gate.get('output_root') == str(OUT_ROOT)
              and gate.get('execution_authorized') is True
              and gate.get('data_reads_authorized') is True
              and gate.get('target_tape_builds_authorized') == 0
              and gate.get('target_publish_authorized') is False
              and gate.get('model_fits_authorized') == 0
              and gate.get('training_authorized') is False
              and gate.get('test_authorized') is False
              and gate.get('checkpoint_reads_authorized') is False
              and gate.get('broker_access_allowed') is False
              and gate.get('test_data_used') is False
              and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
              and gate.get('max_invocations') == 1
              and gate.get('relaunch_allowed') is False
              and gate.get('memory_max') == '4G' and gate.get('swap_max') == '512M'
              and gate.get('wall_seconds') == 1800,
              'POLICY_GATE')
    return plan, plan_ref, head, policy_ref


def _load_candidate_tape() -> TimeQueryTargetTape:
    raw = _checked_bytes('candidate_tape', 32 * 1024**2)
    with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
        _need(set(saved.files) == TAPE_KEYS, 'TAPE_MEMBERS')
        axis = np.array(saved['validity_ns'], copy=True)
        source = np.array(saved['m1_source_row'], copy=True)
        bits = np.array(saved['allowed_set'], copy=True)
        path = np.array(saved['path_observed'], copy=True)
        gaps = np.array(saved['gap_kind'], copy=True)
    _need(axis.shape == source.shape == gaps.shape == (591682,)
          and bits.shape == path.shape == (591682, 5)
          and axis.dtype == source.dtype == np.int64
          and bits.dtype == gaps.dtype == np.uint8 and path.dtype == np.bool_
          and np.array_equal(source, np.arange(5193205, 5193205 + 591682))
          and np.isin(gaps, (0, 1, 2)).all()
          and int((gaps == 1).sum()) == 413
          and int((gaps == 2).sum()) == 95
          and np.all(path <= np.isin(bits, (1, 2, 4))),
          'TAPE_SCHEMA')
    report = json.loads(_checked_bytes('candidate_result', 1024**2))['tape_report']
    _need(all(int((bits[:, j] == bit).sum()) == report['allowed_counts'][tf][str(bit)]
              for j, tf in enumerate(core.TIMEFRAMES) for bit in (1, 2, 4, 5, 7))
          and all(int(path[:, j].sum()) == report['path_confirmed_counts'][tf]
                  for j, tf in enumerate(core.TIMEFRAMES)),
          'TAPE_REPORT_PARITY')
    return TimeQueryTargetTape(axis, source, bits)


def _fold_references(tape: TimeQueryTargetTape, deadline: float) -> dict:
    bound = binding.load_bound_five_endpoint_targets()
    _need(len(bound.decision_ns) == 118249
          and np.all(np.diff(bound.decision_ns) > 0), 'MODEL_AXIS')
    folds = []
    for fold_id in range(1, 6):
        _need(time.monotonic() + 60 < deadline,
              'DEADLINE_FOLD:' + str(fold_id))
        axes = bound.mature_oof_axes(fold_id=fold_id)
        fit = axes['fit_decision_ns']
        score = axes['score_decision_ns']
        _need(len(fit) == EXPECTED_FIT_ROWS[fold_id - 1]
              and len(score) == EXPECTED_SCORE_ROWS[fold_id - 1]
              and len(axes['fit_parent_rows']) == len(fit)
              and len(axes['score_parent_rows']) == len(score)
              and axes['official_fit_cutoff_ns'] == binding.FOLD_BOUNDARIES_NS[fold_id - 1]
              and axes['score_end_exclusive_ns'] == binding.FOLD_BOUNDARIES_NS[fold_id],
              'FOLD_AXES:' + str(fold_id))
        ref = core.score_reference_fold(
            tape, fit, score,
            fit_lower_ns=binding.TRAIN_START_NS,
            fold_boundary_ns=axes['official_fit_cutoff_ns'],
            score_upper_ns=axes['score_end_exclusive_ns'],
            chunk_rows=LIMITS['chunk_rows'])
        _need(ref['fit_issues'] == len(fit)
              and ref['score_issues'] == len(score)
              and ref['fit_issue_axis_sha256'] == axes['fit_decision_ns_sha256']
              and ref['score_issue_axis_sha256'] == axes['score_decision_ns_sha256']
              and ref['model_fits'] == 0 and ref['test_data_used'] is False
              and set(ref['per_tf_k']) == set(core.TIMEFRAMES)
              and set(ref['weekly_by_issue_monday_utc']) == set(core.TIMEFRAMES)
              and set(ref['score_source_reuse']) == set(core.TIMEFRAMES)
              and all(len(ref['per_tf_k'][tf]) == int(k)
                      for tf, k in zip(core.TIMEFRAMES, core.NATIVE_K)),
              'FOLD_RESULT:' + str(fold_id))
        m5 = ref['per_tf_k']['M5']['1']
        folds.append({
            'fold_id': fold_id,
            'fit_rows': len(fit), 'score_rows': len(score),
            'fit_parent_rows_sha256': axes['fit_parent_rows_sha256'],
            'fit_issue_axis_sha256': axes['fit_decision_ns_sha256'],
            'score_parent_rows_sha256': axes['score_parent_rows_sha256'],
            'score_issue_axis_sha256': axes['score_decision_ns_sha256'],
            'official_fit_cutoff_ns': axes['official_fit_cutoff_ns'],
            'effective_fit_cutoff_ns': axes['effective_fit_cutoff_ns'],
            'score_end_exclusive_ns': axes['score_end_exclusive_ns'],
            'm5_native_endpoint': m5,
            'reference': ref,
        })
    _need(sum(f['score_rows'] for f in folds) == EXPECTED_SCORE_TOTAL,
          'OOF_SCORE_TOTAL')
    m5_keys = (
        'activity_known', 'side_known', 'singleton',
        'activity_prior_brier_sum', 'side_prior_brier_sum',
        'three_class_prior_brier_sum',
        'activity_persistence_count', 'activity_persistence_brier_sum',
        'activity_prior_paired_brier_sum', 'side_persistence_count',
        'side_persistence_brier_sum', 'side_prior_paired_brier_sum',
    )
    totals = {key: sum(f['m5_native_endpoint']['score'][key] for f in folds)
              for key in m5_keys}
    for score, count in (('activity_prior_brier_sum', 'activity_known'),
                         ('side_prior_brier_sum', 'side_known'),
                         ('three_class_prior_brier_sum', 'singleton'),
                         ('activity_persistence_brier_sum', 'activity_persistence_count'),
                         ('activity_prior_paired_brier_sum', 'activity_persistence_count'),
                         ('side_persistence_brier_sum', 'side_persistence_count'),
                         ('side_prior_paired_brier_sum', 'side_persistence_count')):
        totals[score.replace('_sum', '_mean')] = totals[score] / totals[count] if totals[count] else None
    return {
        'bound_model_issue_rows': len(bound.decision_ns),
        'bound_model_issue_axis_sha256': hashlib.sha256(bound.decision_ns.tobytes()).hexdigest(),
        'oof_score_rows': EXPECTED_SCORE_TOTAL,
        'folds': folds,
        'm5_k1_totals': totals,
        'scope_limit': 'Weekly sums are descriptive raw query-cell aggregates, not per-issue-normalized uncertainty or model gain.',
    }


def _write(root: Path, name: str, value: dict) -> dict:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _need(len(payload) <= LIMITS['max_output_bytes'], 'OUTPUT_BYTES')
    _publish_bytes(root / name, payload)
    return {'path': str(root / name), 'sha256': hashlib.sha256(payload).hexdigest(),
            'bytes': len(payload)}


def _wall_limit(_signum, _frame) -> None:
    raise TimeoutError('MARKET_STATE_TIME_QUERY_REFERENCES_INTERNAL_WALL_LIMIT')


def run(plan_path: Path, digest: str, out_dir: Path) -> dict:
    started = time.monotonic()
    plan, plan_ref, head, policy_ref = preflight(plan_path, digest, out_dir,
                                                  admitted=True)
    deadline = started + LIMITS['internal_wall_seconds']
    _need(policy_ref is not None and deadline > time.monotonic() + 60,
          'WALL_BUDGET_PRECLAIM')
    out_dir.mkdir(parents=True, exist_ok=False)
    old_handler = signal.getsignal(signal.SIGALRM)
    claim = None
    try:
        signal.signal(signal.SIGALRM, _wall_limit)
        signal.setitimer(signal.ITIMER_REAL, deadline - time.monotonic())
        _fsync_directory(out_dir.parent)
        claim = _write(out_dir, 'CLAIM.json', {
            'schema_version': 'gx1_market_state_time_query_references_claim_v1',
            'invocation': 1, 'plan': plan_ref, 'policy': policy_ref,
            'source_head': head, 'read_only': True, 'test_data_used': False})
        tape = _load_candidate_tape()
        refs = _fold_references(tape, deadline)
        _need(_clean_head() == head and _sha(PLAN_PATH) == digest
              and _sha(POLICY) == policy_ref['sha256']
              and all(_sha(REPO / name) == digest_ for name, digest_ in
                      plan['source_owners'].items())
              and all(_sha(Path(ref['path'])) == ref['sha256']
                      for ref in INPUT_REFS.values()),
              'SOURCE_BOOKEND')
        result = _write(out_dir, 'RESULT.json', {
            'schema_version': 'gx1_market_state_time_query_references_result_v1',
            'status': 'COMPLETE_FROZEN_TRAIN_REFERENCES_NO_MODEL',
            'plan': plan_ref, 'policy': policy_ref, 'claim': claim,
            'source_head': head, 'elapsed_seconds': time.monotonic() - started,
            'input_refs': INPUT_REFS, 'metric_contract': CONTRACT,
            'candidate_tape': _ref('candidate_tape'),
            'references': refs,
            'read_only': True, 'temporary_target_tapes': 0,
            'published_target_tapes': 0, 'model_fits': 0,
            'test_data_used': False, 'target_selection': False,
            'limit': 'TRAIN priors and causal selective persistence only; no prediction, native fit, TEST, joint episode probability or economic value.'})
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminal = _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_time_query_references_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0,
            'plan': plan_ref, 'claim': claim, 'result': result,
            'no_relaunch_under_this_plan': True, 'test_data_used': False})
        return {'result': result, 'terminal': terminal}
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        failed = _write(out_dir, 'FAILED.json', {
            'schema_version': 'gx1_market_state_time_query_references_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'no_relaunch_under_this_plan': True})
        _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_time_query_references_terminal_v1',
            'status': 'FAILED', 'exit_code': 1, 'plan': plan_ref,
            'claim': claim, 'failure': failed,
            'no_relaunch_under_this_plan': True, 'test_data_used': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        plan, ref, head, _ = preflight(args.plan, args.plan_sha256, args.out_dir)
        print(json.dumps({
            'status': 'READY_NO_CLAIM_NO_PAYLOAD', 'plan': ref,
            'source_head': head, 'model_rows': plan['expected_model_rows'],
            'planned_command': planned_command(args.plan_sha256),
            'claim_created': False, 'payload_read': False,
            'test_data_used': False}, sort_keys=True))
        return 0
    print(json.dumps(run(args.plan, args.plan_sha256, args.out_dir), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
