"""One-use capped CPU audit of predeclared five-fold TRAIN state priors.

Only earlier mature TRAIN targets fit each constant prior. Later TRAIN folds
are scored once; this never reads TEST or authorizes a model fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.contracts import market_state_five_endpoint_target_binding_v1 as target
from gx1.contracts.market_state_five_endpoint_contract_v1 import TIMEFRAMES, proper_score_sums
from gx1.contracts.market_state_five_endpoint_prior_v1 import fit_activity_side_prior
from gx1.contracts.immutable_event_authority_v1 import _fsync_directory, _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
STAGE = 'market_state_five_endpoint_prior_20261107_001'
SCHEMA = 'gx1_market_state_five_endpoint_prior_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
POLICY = REPO / 'NEXT_RUN_POLICY.json'
SOURCE_PATHS = frozenset({
    'gx1/audit/measure_market_state_five_endpoint_prior_v1.py',
    'gx1/contracts/market_state_five_endpoint_prior_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_five_endpoint_contract_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'scripts/gx1_capped_run.sh',
})
LIMITS = {'memory_max': 2 * 1024**3, 'memory_high': 2 * 1024**3,
          'swap': 512 * 1024**2, 'max_wall_seconds': 600}
SMOOTHING = 'laplace_plus_one_separate_binary_activity_and_conditional_side'
REFERENCES = ['fit_prefix_prior', 'always_active_activity_brier',
              'always_flat_activity_brier']
EXECUTION = {'class': 'audit', 'runner': 'scripts/gx1_capped_run.sh',
             'memory': '2G', 'swap': '512M', 'timeout_seconds': 600,
             'kill_after_seconds': 10,
             'module': 'gx1.audit.measure_market_state_five_endpoint_prior_v1',
             'model_forwards': 0, 'optimizer_steps': 0}


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('FIVE_ENDPOINT_PRIOR_AUDIT_' + reason)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _ref(path: Path) -> dict:
    return {'path': str(path), 'sha256': _sha(path)}


def _write(root: Path, name: str, value: dict) -> dict:
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _need(len(data) <= 256 * 1024 and json.loads(data) == value, 'PUBLICATION')
    path = root / name
    _publish_bytes(path, data)
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}


def _clean_head() -> str:
    _need(Path.cwd().resolve() == REPO
          and subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
          == 'work/gx1-current'
          and subprocess.check_output(['git', 'status', '--porcelain'], text=True) == '',
          'CLEAN_SOURCE')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def _receipt_preflight() -> None:
    """Hash small pinned receipts and their named keys, without NPZ decoding."""
    refs = target.REFS
    def read(name: str) -> dict:
        path, digest = refs[name]
        file = Path(path)
        _need(file.is_file() and 0 < file.stat().st_size <= 4 * 1024 * 1024
              and _sha(file) == digest, 'REF:' + name)
        return json.loads(file.read_text())
    result, terminal = read('mask_result'), read('mask_terminal')
    corrected, selected = read('mask_plan'), read('selected_plan')
    source, model = read('target_source_plan'), read('model_plan')
    def ref(name: str) -> dict:
        return {'path': refs[name][0], 'sha256': refs[name][1]}
    _need(result.get('row_masks') == ref('mask')
          and result.get('plan') == ref('mask_plan')
          and result.get('decision') == 'PASS_SEVEN_LOCAL_SIDE_DEMOTIONS_SOURCE_ONLY'
          and result.get('status') == terminal.get('status') == 'COMPLETE'
          and result.get('changed_cells') == 7
          and result.get('test_data_used') is False
          and terminal.get('result') == ref('mask_result')
          and terminal.get('row_masks') == ref('mask')
          and terminal.get('exit_code') == 0
          and terminal.get('test_data_used') is False
          and corrected.get('inputs', {}).get('selected', {}).get('plan') == ref('selected_plan')
          and corrected.get('test_data_used') is False
          and selected.get('inputs', {}).get('full_plan') == ref('target_source_plan')
          and selected.get('test_data_used') is False
          and source.get('test_data_used') is False
          and source.get('lead_minutes') == [5, 15, 60, 240, 1440]
          and source.get('train_start_utc') == '2024-05-01T00:00:00+00:00'
          and source.get('train_end_utc_exclusive') == '2026-01-01T00:00:00+00:00'
          and model.get('axes', {}).get('source') == ref('model_axes')
          and model.get('axes', {}).get('rows') == 118249
          and model.get('axes', {}).get('decision_rule') == 'bar_start_ns_plus_300000000000'
          and model.get('base_lineage', {}).get('m5_prebuilt_path') == source.get('m5', {}).get('source', {}).get('path')
          and model.get('base_lineage', {}).get('m5_prebuilt_sha256') == source.get('m5', {}).get('source', {}).get('sha256'),
          'RECEIPT_CHAIN')
    for name, limit in (('mask', 4 * 1024 * 1024), ('model_axes', 32 * 1024 * 1024)):
        path = Path(refs[name][0])
        _need(path.is_file() and 0 < path.stat().st_size <= limit
              and _sha(path) == refs[name][1], 'PAYLOAD_HASH:' + name)


def preflight_read_only(plan_path: Path, plan_sha256: str, out_dir: Path,
                        *, admitted: bool = False) -> tuple[dict, dict, str, dict | None]:
    head = _clean_head()
    _need(plan_path == PLAN_PATH and out_dir == OUT_ROOT
          and plan_path.is_file() and not out_dir.exists()
          and len(plan_sha256) == 64 and _sha(plan_path) == plan_sha256,
          'PLAN_IDENTITY')
    plan = json.loads(plan_path.read_text())
    expected_refs = {name: {'path': path, 'sha256': digest}
                     for name, (path, digest) in target.REFS.items()}
    _need(plan.get('schema_version') == SCHEMA and plan.get('stage') == STAGE
          and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and plan.get('output_root') == str(OUT_ROOT)
          and plan.get('max_invocations') == 1
          and plan.get('fold_ids') == [1, 2, 3, 4, 5]
          and plan.get('smoothing') == SMOOTHING
          and plan.get('references') == REFERENCES
          and plan.get('execution') == EXECUTION
          and plan.get('input_refs') == expected_refs
          and plan.get('limits') == LIMITS
          and plan.get('model_forwards_authorized') == 0
          and plan.get('optimizer_steps_authorized') == 0
          and plan.get('training_authorized') is False
          and plan.get('test_data_used') is False
          and isinstance(plan.get('source_base_head'), str)
          and len(plan['source_base_head']) == 40
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
                              plan['source_base_head'], head],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0,
          'PLAN_SCOPE')
    owners = plan.get('source_owners')
    _need(isinstance(owners, dict) and set(owners) == SOURCE_PATHS
          and all(_sha(REPO / path) == digest for path, digest in owners.items()),
          'SOURCE_OWNERS')
    _receipt_preflight()
    plan_ref = {'path': str(plan_path), 'sha256': plan_sha256}
    policy_ref = None
    if admitted:
        from gx1.contracts.gx1_capped_execution_v1 import (
            require_capped_cpu_audit_execution, require_capped_lock_ancestry)
        cap = require_capped_cpu_audit_execution()
        require_capped_lock_ancestry()
        _need(all(cap.get(key) == value for key, value in LIMITS.items()
                  if key != 'max_wall_seconds') and cap.get('pids') <= 64,
              'CAP_LIMITS')
        policy_ref = _ref(POLICY)
        policy = json.loads(POLICY.read_text())
        _need(_sha(POLICY) == policy_ref['sha256'], 'POLICY_STABLE_READ')
        gate = policy.get(STAGE, {})
        _need(policy.get('training_enabled') is False
              and policy.get('full_val_allowed') is False
              and gate.get('decision') == 'ADMITTED'
              and gate.get('plan') == plan_ref
              and gate.get('execution_authorized') is True
              and gate.get('data_reads_authorized') is True
              and gate.get('baseline_estimation_authorized') is True
              and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
              and gate.get('max_invocations') == 1
              and gate.get('model_forwards_authorized') == 0
              and gate.get('optimizer_steps_authorized') == 0
              and gate.get('training_authorized') is False
              and gate.get('test_authorized') is False
              and gate.get('output_root') == str(OUT_ROOT),
              'POLICY_GATE')
    return plan, plan_ref, head, policy_ref


def _fold_report(bound: target.BoundFiveEndpointTargets, fold_id: int) -> dict:
    lower = target.FOLD_BOUNDARIES_NS[fold_id - 1]
    upper = target.FOLD_BOUNDARIES_NS[fold_id]
    fit = (bound.decision_ns >= target.TRAIN_START_NS) & (bound.decision_ns < lower)
    score = (bound.decision_ns >= lower) & (bound.decision_ns < upper)
    _need(bool(fit.any()) and bool(score.any()) and not bool((fit & score).any()),
          'FOLD_ROWS')
    fit_truth = bound.gather(bound.parent_rows[fit], bound.decision_ns[fit],
                             fold_id=fold_id, partition='fit')
    prior = fit_activity_side_prior(fit_truth['allowed'], fit_truth['eligible'],
                                    fit_truth['mature'])
    _need(prior['smoothing'] == SMOOTHING, 'PRIOR_IDENTITY')
    score_truth = bound.gather(bound.parent_rows[score], bound.decision_ns[score],
                               fold_id=fold_id, partition='score')
    predictions = np.broadcast_to(prior['probabilities'],
                                  (int(score.sum()), len(TIMEFRAMES), 3))
    score_sums = proper_score_sums(predictions, score_truth['allowed'],
                                   score_truth['eligible'], score_truth['mature'])
    eligible = score_truth['eligible'] & score_truth['mature']
    analytic = {}
    coverage = {}
    for j, tf in enumerate(TIMEFRAMES):
        ready = eligible[:, j] & (score_truth['allowed'][:, j] != 7)
        active = ready & (score_truth['allowed'][:, j] != 2)
        flat = ready & (score_truth['allowed'][:, j] == 2)
        _need(int(ready.sum()) == score_sums['per_tf'][tf]['activity']['count'],
              'SCORE_SUPPORT:' + tf)
        analytic[tf] = {'count': int(ready.sum()),
                        'always_flat_brier_sum': int(active.sum()),
                        'always_active_brier_sum': int(flat.sum())}
        coverage[tf] = {'rows': int(score.sum()),
                        'eligible': int(score_truth['eligible'][:, j].sum()),
                        'mature': int(score_truth['mature'][:, j].sum()),
                        'eligible_mature': int(eligible[:, j].sum())}
    return {'fold_id': fold_id, 'fit_rows': int(fit.sum()),
            'score_rows': int(score.sum()),
            'fit_first_issue_ns': int(bound.decision_ns[fit][0]),
            'fit_last_issue_ns': int(bound.decision_ns[fit][-1]),
            'score_first_issue_ns': int(bound.decision_ns[score][0]),
            'score_last_issue_ns': int(bound.decision_ns[score][-1]),
            'prior_probabilities': prior['probabilities'].tolist(),
            'fit_counts': prior['counts'], 'score_sums': score_sums['per_tf'],
            'score_coverage': coverage, 'analytic_activity': analytic}


def _measure() -> dict:
    bound = target.load_bound_five_endpoint_targets()
    folds = [_fold_report(bound, fold_id) for fold_id in range(1, 6)]
    pooled = {}
    for tf in TIMEFRAMES:
        pooled[tf] = {}
        for name in ('activity', 'conditional_side', 'three_class'):
            values = [fold['score_sums'][tf][name] for fold in folds]
            pooled[tf][name] = {
                'count': sum(item['count'] for item in values),
                'brier_sum': sum(item['brier_sum'] for item in values),
                'brier_mean': (sum(item['brier_sum'] for item in values)
                               / sum(item['count'] for item in values))
                              if sum(item['count'] for item in values) else None}
            if name != 'activity':
                pooled[tf][name]['log_sum'] = sum(item['log_sum'] for item in values)
                pooled[tf][name]['log_mean'] = (
                    pooled[tf][name]['log_sum'] / pooled[tf][name]['count']
                    if pooled[tf][name]['count'] else None)
    pooled_analytic = {}
    for tf in TIMEFRAMES:
        rows = [fold['analytic_activity'][tf] for fold in folds]
        count = sum(item['count'] for item in rows)
        flat_sum = sum(item['always_flat_brier_sum'] for item in rows)
        active_sum = sum(item['always_active_brier_sum'] for item in rows)
        _need(count == pooled[tf]['activity']['count']
              and flat_sum + active_sum == count, 'ANALYTIC_SUPPORT:' + tf)
        pooled_analytic[tf] = {'count': count,
                               'always_flat_brier_sum': flat_sum,
                               'always_active_brier_sum': active_sum,
                               'always_flat_brier_mean': flat_sum / count if count else None,
                               'always_active_brier_mean': active_sum / count if count else None}
    return {'scope_limit': 'Five chronological TRAIN prior references only; no model predictions, OOF model score, learning, generalization to sealed TEST, or trading value.',
            'target_rows_joined': len(bound.parent_rows),
            'unjoined_target_rows': bound.unjoined_target_rows,
            'smoothing': SMOOTHING, 'folds': folds,
            'pooled_score': pooled, 'pooled_analytic_activity': pooled_analytic,
            'model_forwards': 0,
            'optimizer_steps': 0, 'test_data_used': False}


def run(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    plan, plan_ref, head, policy_ref = preflight_read_only(
        plan_path, plan_sha256, out_dir, admitted=True)
    _need(policy_ref is not None and _sha(POLICY) == policy_ref['sha256'],
          'POLICY_BEFORE_CLAIM')
    out_dir.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    claim = None
    try:
        _fsync_directory(out_dir.parent)
        claim = _write(out_dir, 'CLAIM.json', {
            'schema_version': 'gx1_market_state_five_endpoint_prior_claim_v1',
            'plan': plan_ref, 'policy': policy_ref, 'source_head': head,
            'invocation': 1, 'fold_ids': plan['fold_ids']})
        data = _measure()
        _need(_sha(plan_path) == plan_sha256
              and _sha(POLICY) == policy_ref['sha256']
              and _clean_head() == head, 'FINAL_SOURCE_IDENTITY')
        result = _write(out_dir, 'RESULT.json', {
            'schema_version': 'gx1_market_state_five_endpoint_prior_result_v1',
            'status': 'COMPLETE', 'decision': 'PASS_TRAIN_PRIOR_REFERENCE_ONLY',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'elapsed_seconds': time.monotonic() - began, **data})
        terminal = _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_five_endpoint_prior_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'claim': claim, 'result': result, 'model_forwards': 0,
            'optimizer_steps': 0, 'test_data_used': False,
            'no_relaunch_under_this_plan': True,
            'output_inventory': ['CLAIM.json', 'RESULT.json', 'TERMINAL.json']})
        return {'result': result, 'terminal': terminal}
    except Exception as exc:
        failure = _write(out_dir, 'FAILED.json', {
            'schema_version': 'gx1_market_state_five_endpoint_prior_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'no_relaunch_under_this_plan': True})
        _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_five_endpoint_prior_terminal_v1',
            'status': 'FAILED', 'exit_code': 1, 'plan': plan_ref,
            'claim': claim, 'failure': failure, 'model_forwards': 0,
            'optimizer_steps': 0, 'test_data_used': False,
            'no_relaunch_under_this_plan': True,
            'output_inventory': (['CLAIM.json', 'FAILED.json', 'TERMINAL.json']
                                 if claim else ['FAILED.json', 'TERMINAL.json'])})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        plan, ref, head, _ = preflight_read_only(args.plan, args.plan_sha256,
                                                args.out_dir)
        print(json.dumps({'status': 'READY_NO_CLAIM_NO_PAYLOAD',
                          'plan': ref, 'source_head': head,
                          'source_owners_checked': len(plan['source_owners']),
                          'receipt_refs_checked': 8,
                          'claim_created': False, 'model_forwards': 0,
                          'optimizer_steps': 0, 'test_data_used': False},
                         sort_keys=True))
        return 0
    print(json.dumps(run(args.plan, args.plan_sha256, args.out_dir), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
