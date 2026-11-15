"""One-use physical Fold2 fixed 25-cell S prior and later reference score.

Only a +1 smoothed constant prior is fitted on earlier mature FIT labels.
No model, feature gather, optimizer, TEST, broker, Entry or trading.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit import build_market_state_time_query_tape_v1 as producer
from gx1.audit import measure_market_state_fold2_exact_prior_later_v1 as exact
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.contracts.market_state_fold2_multiminute_path_v1 import (
    QUERY_SHA256, gather_fixed_path, score_fixed_path,
)
from gx1.contracts.market_state_fold2_multiminute_prior_v1 import (
    fit_exact_fold2_prior, require_exact_fold2_prior,
)
from gx1.contracts.market_state_minute_query_candidate_v1 import MinuteQueryTargetTape
from gx1.contracts.market_state_time_query_native_F_objective_v1 import _weeks

REPO = exact.REPO
POLICY = exact.POLICY
STAGE = 'market_state_fold2_fixed_25_cell_prior_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_FOLD2_FIXED_25_CELL_PRIOR_20261115_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_FOLD2_FIXED_25_CELL_PRIOR_20261115_001')
SCOPE = {
    'fold_id': 2, 'fit_rows': 47338,
    'fit_parent_rows_sha256': exact.PARENT_SHA,
    'fit_decision_ns_sha256': exact.ISSUE_SHA,
    'later_selected_rows': 4096, 'query_sha256': QUERY_SHA256,
    'query_cells': 25, 'model_forwards': 0, 'optimizer_steps': 0,
    'test_data_used': False,
}
LIMITS = {
    'class': 'audit', 'memory': '4G', 'swap': '512M',
    'wall_seconds': 600, 'max_invocations': 1,
}
CODE = (
    'gx1/audit/fit_market_state_fold2_fixed_25_cell_prior_v1.py',
    'gx1/audit/measure_market_state_fold2_exact_prior_later_v1.py',
    'gx1/audit/build_market_state_time_query_tape_v1.py',
    'gx1/contracts/market_state_fold2_multiminute_prior_v1.py',
    'gx1/contracts/market_state_fold2_multiminute_path_v1.py',
    'gx1/contracts/market_state_five_endpoint_prior_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_minute_query_candidate_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/market_state_five_endpoint_contract_v1.py',
    'gx1/contracts/market_state_fold2_multiminute_side_loss_v1.py',
    'gx1/contracts/market_state_time_query_scorer_v2.py',
    'gx1/contracts/market_state_time_query_target_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
)


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('FOLD2_FIXED_25_CELL_PRIOR_' + reason)


def terminal_payload(*, status: str, exit_code: int, claim: dict,
                     result: dict | None, head: str,
                     reason: str | None = None) -> dict:
    need(status in ('COMPLETE', 'FAILED_CONSUMED')
         and exit_code == (0 if status == 'COMPLETE' else 1)
         and (result is not None) == (status == 'COMPLETE')
         and (reason is None) == (status == 'COMPLETE'), 'TERMINAL_ARGS')
    value = {
        'schema_version': 'gx1_fold2_fixed_25_cell_prior_terminal_v1',
        'status': status, 'exit_code': exit_code, 'stage': STAGE,
        'claim': claim, 'result': result, 'source_head': head,
        'test_data_used': False,
    }
    if reason is not None:
        value['reason'] = reason
    return value


def preflight(plan_sha: str) -> tuple:
    need(Path.cwd().resolve() == REPO
         and exact.git('branch', '--show-current') == 'work/gx1-current'
         and exact.git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = exact.git('rev-parse', 'HEAD')
    need(not ROOT.exists() and exact.sha(PLAN) == plan_sha, 'PLAN_ROOT')
    plan = json.loads(PLAN.read_text())
    need(plan.get('schema_version') == 'gx1_fold2_fixed_25_cell_prior_plan_v1'
         and plan.get('stage') == STAGE
         and plan.get('status') == 'ADMITTED_BASELINE_ONLY'
         and plan.get('source_base_head') == '7d7977646b1981a855d7d59981364f10ba07c47d'
         and plan.get('scope') == SCOPE and plan.get('limits') == LIMITS
         and plan.get('output_root') == str(ROOT), 'PLAN')
    need(subprocess.run(('git', 'merge-base', '--is-ancestor',
                         plan['source_base_head'], head), cwd=REPO).returncode == 0,
         'ANCESTOR')
    need(set(plan.get('source_owners', {})) == set(CODE)
         and all(exact.sha(REPO / name) == digest
                 for name, digest in plan['source_owners'].items()),
         'SOURCE_HASH')
    need(set(plan.get('input_refs', {})) == {
             'coverage', 'later_input_result', 'later_input_terminal',
             'target_tape', 'prior_source_review', 'path_review'}
         and all(Path(ref['path']).is_file()
                 and exact.sha(Path(ref['path'])) == ref['sha256']
                 for ref in plan['input_refs'].values()), 'INPUT_HASH')
    refs = plan['input_refs']
    coverage = json.loads(Path(refs['coverage']['path']).read_text())
    later = json.loads(Path(refs['later_input_result']['path']).read_text())
    terminal = json.loads(Path(refs['later_input_terminal']['path']).read_text())
    prior_review = json.loads(Path(refs['prior_source_review']['path']).read_text())
    path_review = json.loads(Path(refs['path_review']['path']).read_text())
    need(coverage['status'] == later['status'] == terminal['status'] == 'COMPLETE'
         and terminal['exit_code'] == 0
         and terminal['result'] == refs['later_input_result']
         and coverage['qualification'] == refs['later_input_result']
         and later['selected_parent_rows'] == 4096
         and later['target_arrays_decoded'] == 0
         and prior_review['status'] == 'SOURCE_ONLY_PASS_25_CELL_PRIOR_NO_PHYSICAL_FIT'
         and path_review['status'].startswith('COMPLETE_CONSUMED')
         and path_review['query_sha256'] == QUERY_SHA256, 'RECEIPTS')
    bound = state.load_bound_five_endpoint_targets()
    fit, later_rows, later_issues = exact.axes(coverage, bound)
    need(len(fit['fit_parent_rows']) == 47338
         and len(later_rows) == len(later_issues) == 4096
         and np.array_equal(
             fit['fit_decision_ns'],
             bound.decision_ns[np.searchsorted(
                 bound.parent_rows, fit['fit_parent_rows'])]), 'AXES')
    policy = json.loads(POLICY.read_text())
    expected = {
        'status': 'ADMITTED_BASELINE_ONLY', 'plan': exact.ref(PLAN),
        'output_root': str(ROOT), 'max_invocations': 1,
        'claimed_invocations': 0, 'completed_invocations': 0,
        'execution_authorized': True, 'data_reads_authorized': True,
        'baseline_estimation_authorized': True,
        'model_forwards_authorized': 0, 'optimizer_steps_authorized': 0,
        'training_authorized': False, 'test_authorized': False,
        'broker_access_allowed': False, 'relaunch_allowed': False,
    }
    need(policy['training_enabled'] is False
         and policy['native_training_enabled'] is False
         and policy[STAGE] == expected, 'POLICY')
    return head, exact.ref(POLICY), plan, coverage, fit, later_rows, later_issues


def _tape(ref: dict) -> MinuteQueryTargetTape:
    with np.load(io.BytesIO(Path(ref['path']).read_bytes()),
                 allow_pickle=False) as saved:
        need(set(saved.files) == producer.TAPE_KEYS, 'TAPE_KEYS')
        return MinuteQueryTargetTape(
            np.array(saved['validity_ns'], copy=True),
            np.array(saved['m1_source_row'], copy=True),
            np.array(saved['allowed_set'], copy=True),
            np.array(saved['path_observed'], copy=True),
            np.array(saved['gap_kind'], copy=True))


def _sums(path: dict, raw_grid: np.ndarray) -> list:
    raw = np.broadcast_to(raw_grid[None], path['allowed'].shape + (2,))
    return score_fixed_path(raw, path)['per_checkpoint']


def _slice(path: dict, selected: np.ndarray) -> dict:
    length = len(path['issue_ns'])
    return {key: (value[selected] if isinstance(value, np.ndarray)
                  and value.ndim > 0 and value.shape[0] == length else value)
            for key, value in path.items()}


def _check_week_partition(overall: list, weekly: dict) -> None:
    for q in range(5):
        for tf in ('M5', 'M15', 'H1', 'H4', 'D1'):
            for metric in ('activity', 'conditional_side', 'three_class'):
                full = overall[q]['per_tf'][tf][metric]
                for key in ('count', 'brier_sum', 'log_sum'):
                    if key not in full:
                        continue
                    total = sum(w[q]['per_tf'][tf][metric][key]
                                for w in weekly.values())
                    need(np.isclose(total, full[key], rtol=0, atol=1e-7),
                         'WEEK_PARTITION')


def measure(head: str, policy_ref: dict, plan: dict, coverage: dict,
            fit: dict, later_rows: np.ndarray,
            later_issues: np.ndarray) -> tuple[dict, dict]:
    tape = _tape(plan['input_refs']['target_tape'])
    fit_path = gather_fixed_path(
        tape, fit['fit_decision_ns'],
        fold_upper_ns=1735689600000000000)
    frozen = fit_exact_fold2_prior(
        fit_path, fit['fit_parent_rows'],
        expected_parent_rows_sha256=exact.PARENT_SHA,
        expected_issue_ns_sha256=exact.ISSUE_SHA)
    raw_grid = require_exact_fold2_prior(
        frozen, parent_rows_sha256=exact.PARENT_SHA,
        issue_ns_sha256=exact.ISSUE_SHA)
    score_path = gather_fixed_path(
        tape, later_issues,
        fold_upper_ns=coverage['score_end_exclusive_ns'])
    overall = _sums(score_path, raw_grid)
    weeks = _weeks(later_issues)
    weekly = {str(w): _sums(_slice(score_path, weeks == w), raw_grid)
              for w in np.unique(weeks)}
    _check_week_partition(overall, weekly)
    need(len(weekly) > 0 and len(overall) == 5
         and all(set(point['per_tf']) == {'M5','M15','H1','H4','D1'}
                 for point in overall), 'SCORE')
    result = {
        'schema_version': 'gx1_fold2_fixed_25_cell_prior_result_v1',
        'status': 'COMPLETE', 'stage': STAGE, 'source_head': head,
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'plan': exact.ref(PLAN), 'policy': policy_ref,
        'prior': None, 'query_sha256': QUERY_SHA256,
        'fit_rows': 47338, 'later_rows': 4096,
        'fit_parent_rows_sha256': exact.PARENT_SHA,
        'fit_decision_ns_sha256': exact.ISSUE_SHA,
        'later_parent_rows_sha256': coverage['selected_parent_rows_sha256'],
        'later_issue_ns_sha256': coverage['selected_decision_ns_sha256'],
        'later_utc_weeks': len(weekly),
        'per_checkpoint_prior_proper_sums': overall,
        'by_utc_week': weekly,
        'model_forwards': 0, 'optimizer_steps': 0,
        'test_data_used': False, 'trade_selection': False,
    }
    return frozen, result


def run(plan_sha: str) -> None:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    head, policy_ref, plan, coverage, fit, rows, issues = preflight(plan_sha)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    ROOT.mkdir(parents=False, exist_ok=False)
    _publish_bytes(ROOT / 'CLAIM.json', exact.encode({
        'schema_version': 'gx1_fold2_fixed_25_cell_prior_claim_v1',
        'stage': STAGE, 'plan': exact.ref(PLAN),
        'policy': policy_ref, 'source_head': head, 'test_data_used': False}))
    claim = exact.ref(ROOT / 'CLAIM.json')
    started = time.monotonic()
    try:
        frozen, result = measure(
            head, policy_ref, plan, coverage, fit, rows, issues)
        _publish_bytes(ROOT / 'PRIOR.json', exact.encode(frozen))
        result['prior'] = exact.ref(ROOT / 'PRIOR.json')
        result['claim'] = claim
        result['elapsed_seconds'] = time.monotonic() - started
        _publish_bytes(ROOT / 'RESULT.json', exact.encode(result))
        _publish_bytes(ROOT / 'TERMINAL.json', exact.encode(
            terminal_payload(
                status='COMPLETE', exit_code=0, claim=claim,
                result=exact.ref(ROOT / 'RESULT.json'), head=head)))
    except BaseException as exc:
        _publish_bytes(ROOT / 'TERMINAL.json', exact.encode(
            terminal_payload(
                status='FAILED_CONSUMED', exit_code=1, claim=claim,
                result=None, head=head,
                reason=type(exc).__name__ + ':' + str(exc))))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha', required=True)
    run(parser.parse_args().plan_sha)


if __name__ == '__main__':
    main()
