"""One-use inner-TRAIN chronology diagnosis for F-M5 tree overfit.

Five fixed CPU fits use only each outer fold's mature TRAIN prefix. No outer
score label, sealed TEST, native model, Entry, Exit, broker, or trading route.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from gx1.audit import run_market_state_f_m5_tabular_oof_v1 as old
from gx1.contracts import market_state_current_tabular_span_overlay_v1 as overlay
from gx1.contracts import market_state_future_displacement_candidate_v1 as future
from gx1.contracts import market_state_tabular_lightgbm_v1 as model
from gx1.contracts.market_state_tabular_target_score_v1 import mature_fold_training_targets
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_research_producer_execution
from gx1.contracts.immutable_event_authority_v1 import _fsync_directory

REPO = old.REPO
RUNS = old.RUNS
STAGE = 'market_state_f_m5_inner_train_transfer_20261114_001'
SCHEMA = 'gx1_market_state_f_m5_inner_train_transfer_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
POLICY = REPO / 'NEXT_RUN_POLICY.json'
FOLD_IDS = old.FOLD_IDS
CHECKPOINTS = [1, 2, 4, 8, 16, 32, 64]
LEAD_NS = 5 * 60 * 1_000_000_000
SPLIT_RULE = {'population': 'own_outer_fold_mature_F_M5_known_side_TRAIN_only',
              'validation_boundary': 'known_side_issue_at_floor_80_percent_index',
              'inner_train_label_validity': 'issue_plus_5_minutes_strictly_before_boundary',
              'inner_validation_issue': 'at_or_after_boundary',
              'prior': 'Laplace_up_count_on_inner_train_only',
              'no_outer_score_target_or_model_selection': True}
EXECUTION = {**old.EXECUTION, 'module': 'gx1.audit.run_market_state_f_m5_inner_train_transfer_v1',
             'fixed_tabular_fits': 5, 'timeout_seconds': 1800}
LIMITS = {**old.LIMITS, 'internal_wall_seconds': 1770, 'timeout_seconds': 1800}
RECIPE = {**old.RECIPE, 'boost_rounds': 64,
          'prediction_checkpoints': CHECKPOINTS,
          'validation_source': 'inner_TRAIN_chronology_only'}
DECISION_RULE = {'metric': 'inner_validation_F_M5_certified_side_Brier',
                 'baseline': 'inner_TRAIN_Laplace_prior',
                 'candidate_rounds': CHECKPOINTS,
                 'selection': 'strictly_lower_Brier_than_prior_else_zero_round_prior',
                 'tie_break': 'earliest_round',
                 'outer_score_selection': False,
                 'no_rank_or_edge_promotion': True}
REPORT_REFS = {
    'original_oof_result': {'path': str(old.OUT_ROOT / 'RESULT.json'),
                            'sha256': 'ac5dac12791c092380f318d20552529c690fe1928be0a9b6aec9a1626cc9b3c9'},
    'train_score_diagnosis': {'path': str(REPO / 'docs/TA_F_M5_SAVED_MODEL_TRAIN_SCORE_DIAGNOSIS_20261114.json'),
                              'sha256': '78c9a43dc75cc5696779b4b626828b13c6d63a0f686e8f366188639c3d87f58c'},
    'tree_path_diagnosis': {'path': str(REPO / 'docs/TA_F_M5_SAVED_TREE_PATH_DIAGNOSIS_20261114.json'),
                            'sha256': 'fb2e3d1afb1d59ebf08f687ed6ddaaedfbea836c51eb81647a3b2e2b4b5cc6ff'},
}
INPUT_REFS = {**old.INPUT_REFS, **REPORT_REFS}
SOURCE_PATHS = old.SOURCE_PATHS | frozenset({
    'gx1/audit/run_market_state_f_m5_inner_train_transfer_v1.py',
})


def need(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError('F_M5_INNER_TRAIN_TRANSFER_' + code)


def checked_inputs() -> tuple[dict, dict]:
    chain, schema, _ = old._inputs()
    for item in REPORT_REFS.values():
        need(old._sha(Path(item['path'])) == item['sha256'], 'REPORT_REF')
    result = old._checked_json(REPORT_REFS['original_oof_result'])
    need(result['status'] == 'COMPLETE_FIVE_FOLDS'
         and result['fixed_tabular_fits'] == 5
         and result['test_data_used'] is False
         and result['native_training'] is False, 'PRIOR_RESULT')
    need(future.state.LEAD_MINUTES[0] == 5
         and model.BOOST_ROUNDS == CHECKPOINTS[-1]
         and model.EXPECTED_WIDTH == 4003, 'FIXED_CONTRACT')
    return chain, schema


def planned_command(plan_sha: str) -> list[str]:
    return ['timeout', '--signal=TERM', '--kill-after=10s', '1800s',
            'scripts/gx1_capped_run.sh', '--class', 'producer', '--mem', '12G',
            '--swap', '512M', '--', '.venv/bin/python', '-m', EXECUTION['module'],
            '--plan', str(PLAN_PATH), '--plan-sha256', plan_sha,
            '--out-dir', str(OUT_ROOT)]


def _policy_gate(policy: dict, plan_ref: dict, out_dir: Path) -> None:
    gate = policy.get(STAGE, {})
    need(policy.get('training_enabled') is False
         and policy.get('native_training_enabled') is False
         and policy.get('native_launch_authorized') is False
         and policy.get('full_val_allowed') is False
         and gate.get('status') == gate.get('decision') == 'ADMITTED'
         and gate.get('plan') == plan_ref
         and gate.get('output_root') == str(out_dir)
         and gate.get('execution_authorized') is True
         and gate.get('data_reads_authorized') is True
         and gate.get('research_fit_authorized') is True
         and gate.get('fixed_tabular_fits_authorized') == 5
         and gate.get('native_training_authorized') is False
         and gate.get('training_authorized') is False
         and gate.get('test_authorized') is False
         and gate.get('broker_access_allowed') is False
         and gate.get('test_data_used') is False
         and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
         and gate.get('remaining_invocations') == gate.get('max_invocations') == 1
         and gate.get('relaunch_allowed') is False
         and gate.get('memory_max') == EXECUTION['memory']
         and gate.get('swap_max') == EXECUTION['swap']
         and gate.get('wall_seconds') == EXECUTION['timeout_seconds'], 'POLICY_GATE')


def preflight(plan_path: Path, plan_sha: str, out_dir: Path, *,
              admitted: bool = False, physical_view: bool = False) -> tuple[dict, dict, str, dict | None]:
    head = old.base._clean_head()
    need(plan_path == PLAN_PATH and out_dir == OUT_ROOT and not out_dir.exists()
         and len(plan_sha) == 64 and old._sha(plan_path) == plan_sha, 'PLAN_IDENTITY')
    plan = json.loads(plan_path.read_text())
    chain, schema = checked_inputs()
    old.source._validate_cohort_contract(plan.get('cohorts'))
    need(plan.get('schema_version') == SCHEMA
         and plan.get('stage') == STAGE
         and plan.get('status') == 'PREPARED_NOT_ADMITTED'
         and plan.get('output_root') == str(OUT_ROOT)
         and plan.get('max_invocations') == 1
         and plan.get('fold_ids') == FOLD_IDS
         and plan.get('cohorts') == chain['cohorts']
         and plan.get('input_refs') == INPUT_REFS
         and plan.get('split_rule') == SPLIT_RULE
         and plan.get('checkpoints') == CHECKPOINTS
         and isinstance(plan.get('expected_splits'), list)
         and len(plan['expected_splits']) == 5
         and plan.get('execution') == EXECUTION
         and plan.get('limits') == LIMITS
         and plan.get('recipe') == RECIPE
         and plan.get('decision_rule') == DECISION_RULE
         and plan.get('command_template') == planned_command('{plan_sha256}')
         and plan.get('fixed_tabular_fits_authorized') == 5
         and plan.get('native_training_authorized') is False
         and plan.get('outer_score_targets_used') is False
         and plan.get('test_data_used') is False
         and isinstance(plan.get('source_base_head'), str)
         and len(plan['source_base_head']) == 40
         and subprocess.run(['git', 'merge-base', '--is-ancestor',
                             plan['source_base_head'], head],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0,
         'PLAN_SCOPE')
    owners = plan.get('source_owners')
    need(isinstance(owners, dict) and set(owners) == SOURCE_PATHS
         and all(old._sha(REPO / path) == digest for path, digest in owners.items()),
         'SOURCE_OWNERS')
    provider = old.source._provider_module()
    info = provider.validate_metadata(chain['provider_plan'])
    model._import_lightgbm()
    if physical_view:
        view, _ = overlay.build(provider, chain['provider_plan'], info,
                                old_plan=old._checked_json(old.OLD_ANCHOR_PLAN),
                                span_result=old._checked_json(old.SPAN_RESULT),
                                old_schema=schema)
        need(view.width == 4003 and len(view.parent_rows) == schema['rows'], 'VIEW_SCOPE')
        rows = np.asarray([0, len(view.parent_rows) // 2,
                           len(view.parent_rows) - 1], dtype=np.int64)
        sample = view.take(rows, max_rows=3)
        need(sample.shape == (3, 4003) and sample.dtype == np.float32
             and not np.isinf(sample).any(), 'VIEW_SAMPLE')
        s_bound = old.base.target.load_bound_five_endpoint_targets()
        f_bound = future.load_bound_future_displacement()
        need(np.array_equal(view.parent_rows, f_bound.parent_rows)
             and np.array_equal(view.decision_ns, f_bound.decision_ns),
             'PREFLIGHT_TARGET_AXES')
        for fold_id in FOLD_IDS:
            cohort = mature_fold_training_targets(s_bound, fold_id=fold_id)['cohort']
            fit = f_bound.gather(cohort['fit_parent_rows'],
                                 cohort['fit_decision_ns'],
                                 fold_id=fold_id, partition='fit')
            selected = (fit['eligible'][:, 0] & fit['mature'][:, 0]
                        & np.isin(fit['allowed'][:, 0], (1, 4)))
            train_idx, val_idx, boundary = inner_split(
                cohort['fit_decision_ns'], fit['validity_ns'][:, 0], selected)
            values = fit['allowed'][:, 0]
            expected = {'fold_id': fold_id, 'inner_boundary_ns': boundary,
                        'inner_train_known_side_n': len(train_idx),
                        'inner_validation_known_side_n': len(val_idx),
                        'purged_known_side_n': int(selected.sum()) - len(train_idx) - len(val_idx),
                        'train_up_n': int((values[train_idx] == 4).sum()),
                        'validation_up_n': int((values[val_idx] == 4).sum())}
            need(expected == plan['expected_splits'][fold_id - 1],
                 'PREFLIGHT_SPLIT:' + str(fold_id))
    plan_ref = {'path': str(plan_path), 'sha256': plan_sha}
    policy_ref = None
    if admitted:
        cap = require_capped_cpu_research_producer_execution()
        need(cap['memory_max'] == cap['memory_high'] == LIMITS['memory_max']
             and cap['swap'] == LIMITS['swap']
             and cap['pids'] == LIMITS['pids']
             and cap['cpu_affinity'] == list(range(8))
             and cap['numerical_threads'] == 1
             and isinstance(cap['lock'], dict), 'CAP_LIMITS')
        policy_ref = {'path': str(POLICY), 'sha256': old._sha(POLICY)}
        _policy_gate(json.loads(POLICY.read_text()), plan_ref, out_dir)
        need(old._sha(POLICY) == policy_ref['sha256'], 'POLICY_STABLE_READ')
    return plan, plan_ref, head, policy_ref


def inner_split(issues: np.ndarray, validity: np.ndarray,
                selected: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Purged chronology from only one outer fold's mature TRAIN F labels."""
    need(isinstance(issues, np.ndarray) and issues.dtype == np.int64
         and isinstance(validity, np.ndarray) and validity.dtype == np.int64
         and isinstance(selected, np.ndarray) and selected.dtype == np.bool_
         and issues.ndim == validity.ndim == selected.ndim == 1
         and len(issues) == len(validity) == len(selected) > 0
         and np.all(np.diff(issues) > 0)
         and np.array_equal(validity, issues + LEAD_NS), 'SPLIT_INPUT')
    known = np.flatnonzero(selected)
    need(len(known) >= 1000, 'SPLIT_SUPPORT')
    boundary = int(issues[known[(4 * len(known)) // 5]])
    train = np.flatnonzero(selected & (validity < boundary))
    validation = np.flatnonzero(selected & (issues >= boundary))
    need(len(train) >= 500 and len(validation) >= 200
         and np.all(validity[train] < boundary)
         and np.all(issues[validation] >= boundary)
         and train[-1] < validation[0], 'SPLIT_CHRONOLOGY')
    return train, validation, boundary


def choose_round(records: list[dict], prior_brier: float) -> int:
    need(len(records) == len(CHECKPOINTS)
         and [item['rounds'] for item in records] == CHECKPOINTS
         and all(np.isfinite(item['brier']) for item in records)
         and np.isfinite(prior_brier), 'SELECTION_INPUT')
    best = min(records, key=lambda item: (item['brier'], item['rounds']))
    return int(best['rounds']) if best['brier'] < prior_brier else 0


def terminal_payload(status: str, plan_ref: dict, claim: dict,
                     receipt: dict, fold_count: int) -> dict:
    need(status in ('COMPLETE', 'FAILED') and 0 <= fold_count <= 5,
         'TERMINAL_PAYLOAD')
    return {'schema_version': 'gx1_f_m5_inner_train_transfer_terminal_v1',
            'status': status, 'exit_code': 0 if status == 'COMPLETE' else 1,
            'plan': plan_ref, 'claim': claim,
            'result' if status == 'COMPLETE' else 'failure': receipt,
            'fold_count': fold_count, 'test_data_used': False,
            'no_relaunch_under_this_plan': True}


def _bookend(plan: dict, plan_sha: str, policy_sha: str, head: str) -> None:
    need(old.base._clean_head() == head and old._sha(PLAN_PATH) == plan_sha
         and old._sha(POLICY) == policy_sha
         and all(old._sha(REPO / path) == digest
                 for path, digest in plan['source_owners'].items()), 'SOURCE_BOOKEND')
    for item in INPUT_REFS.values():
        need(old._sha(Path(item['path'])) == item['sha256'], 'INPUT_BOOKEND')


def _wall_limit(_signum, _frame) -> None:
    raise TimeoutError('F_M5_INNER_TRAIN_TRANSFER_INTERNAL_WALL_LIMIT')


def run(plan_path: Path, plan_sha: str, out_dir: Path) -> dict:
    started = time.monotonic()
    plan, plan_ref, head, policy_ref = preflight(plan_path, plan_sha, out_dir,
                                                  admitted=True)
    need(policy_ref is not None, 'POLICY_REQUIRED')
    deadline = started + LIMITS['internal_wall_seconds']
    need(deadline > time.monotonic()
         and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
         'WALL_BUDGET_PRECLAIM')
    out_dir.mkdir(parents=True, exist_ok=False)
    old.base.PUBLISHED_BYTES = 0
    claim = None
    completed: list[dict] = []
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _wall_limit)
        signal.setitimer(signal.ITIMER_REAL, deadline - time.monotonic())
        _fsync_directory(out_dir.parent)
        claim = old.base._write_json(out_dir, 'CLAIM.json', {
            'schema_version': 'gx1_f_m5_inner_train_transfer_claim_v1',
            'invocation': 1, 'plan': plan_ref, 'policy': policy_ref,
            'source_head': head, 'fold_ids': FOLD_IDS,
            'fixed_tabular_fits': 5, 'test_data_used': False})
        chain, schema = checked_inputs()
        provider = old.source._provider_module()
        info = provider.validate_metadata(chain['provider_plan'])
        view, _ = overlay.build(provider, chain['provider_plan'], info,
                                old_plan=old._checked_json(old.OLD_ANCHOR_PLAN),
                                span_result=old._checked_json(old.SPAN_RESULT),
                                old_schema=schema)
        matrix, current_sha, old_sha = old._current_matrix(view, deadline)
        qualified = old._checked_json(old.CURRENT_VIEW_REVIEW)
        need(current_sha == qualified['current_matrix_float32_c_order_sha256']
             and old_sha == schema['matrix_float32_c_order_sha256']
             and matrix.shape == (schema['rows'], 4003), 'MATRIX_BEFORE_FIT')
        s_bound = old.base.target.load_bound_five_endpoint_targets()
        f_bound = future.load_bound_future_displacement()
        need(np.array_equal(view.parent_rows, f_bound.parent_rows)
             and np.array_equal(view.decision_ns, f_bound.decision_ns),
             'TARGET_MODEL_AXES')
        frozen = []
        payload = []
        for fold_id in FOLD_IDS:
            old.base._time_guard(deadline)
            cohort = mature_fold_training_targets(s_bound, fold_id=fold_id)['cohort']
            declaration = plan['cohorts'][fold_id - 1]
            need(all(cohort[key] == declaration[key] for key in declaration)
                 and np.array_equal(view.parent_rows[:cohort['joined_fit_rows']],
                                    cohort['fit_parent_rows'])
                 and np.array_equal(view.decision_ns[:cohort['joined_fit_rows']],
                                    cohort['fit_decision_ns']),
                 'COHORT_AXES:' + str(fold_id))
            fit = f_bound.gather(cohort['fit_parent_rows'],
                                 cohort['fit_decision_ns'],
                                 fold_id=fold_id, partition='fit')
            selected = (fit['eligible'][:, 0] & fit['mature'][:, 0]
                        & np.isin(fit['allowed'][:, 0], (1, 4)))
            train_idx, val_idx, boundary = inner_split(
                cohort['fit_decision_ns'], fit['validity_ns'][:, 0], selected)
            y = (fit['allowed'][:, 0] == 4).astype(np.uint8)
            train_y, val_y = y[train_idx], y[val_idx]
            need(0 < train_y.sum() < len(train_y)
                 and 0 < val_y.sum() < len(val_y),
                 'TWO_CLASS:' + str(fold_id))
            prior = float((int(train_y.sum()) + 1) / (len(train_y) + 2))
            record = {'fold_id': fold_id, 'inner_boundary_ns': boundary,
                      'outer_fit_rows': cohort['joined_fit_rows'],
                      'inner_train_known_side_n': len(train_idx),
                      'inner_validation_known_side_n': len(val_idx),
                      'purged_known_side_n': int(selected.sum()) - len(train_idx) - len(val_idx),
                      'train_parent_rows_sha256': old.features._axis_hash(
                          cohort['fit_parent_rows'][train_idx]),
                      'validation_parent_rows_sha256': old.features._axis_hash(
                          cohort['fit_parent_rows'][val_idx]),
                      'train_up_n': int(train_y.sum()),
                      'validation_up_n': int(val_y.sum()),
                      'inner_train_prior_up': prior,
                      'first_validation_issue_ns': int(cohort['fit_decision_ns'][val_idx[0]]),
                      'last_train_target_validity_ns': int(fit['validity_ns'][train_idx[-1], 0]),
                      'outer_score_labels_used': False, 'test_data_used': False}
            need(record['last_train_target_validity_ns'] < boundary
                 and boundary <= record['first_validation_issue_ns'],
                 'FROZEN_MATURITY:' + str(fold_id))
            actual = {key: record[key] for key in (
                'fold_id', 'inner_boundary_ns', 'inner_train_known_side_n',
                'inner_validation_known_side_n', 'purged_known_side_n',
                'train_up_n', 'validation_up_n')}
            need(actual == plan['expected_splits'][fold_id - 1],
                 'EXPECTED_SPLIT:' + str(fold_id))
            frozen.append(record)
            payload.append((train_idx, val_idx, train_y, val_y))
        frozen_ref = old.base._write_json(out_dir, 'INNER_SPLITS_PRIORS_FROZEN.json', {
            'schema_version': 'gx1_f_m5_inner_train_splits_priors_v1',
            'plan': plan_ref, 'claim': claim, 'split_rule': SPLIT_RULE,
            'folds': frozen, 'before_any_model_fit': True,
            'outer_score_labels_used': False, 'test_data_used': False})
        lgb = model._import_lightgbm()
        for fold_id in FOLD_IDS:
            old.base._time_guard(deadline)
            _bookend(plan, plan_sha, policy_ref['sha256'], head)
            train_idx, val_idx, train_y, val_y = payload[fold_id - 1]
            fold = frozen[fold_id - 1]
            x_train = np.ascontiguousarray(matrix[train_idx], dtype=np.float32)
            x_val = np.ascontiguousarray(matrix[val_idx], dtype=np.float32)
            categories = model._categorical_indices(view.feature_names,
                                                     view.categorical,
                                                     x_train, x_val)
            dataset = lgb.Dataset(x_train, label=train_y,
                                  feature_name=list(view.feature_names),
                                  categorical_feature=categories,
                                  free_raw_data=True)
            booster = lgb.train(dict(model.RECIPE), dataset,
                                num_boost_round=model.BOOST_ROUNDS)
            q = fold['inner_train_prior_up']
            prior_brier = float(np.mean(np.square(q - val_y)))
            checkpoints = []
            for rounds in CHECKPOINTS:
                p = np.asarray(booster.predict(x_val, num_iteration=rounds,
                                               num_threads=1), dtype=np.float64)
                metrics = old._metrics(val_y, p, q)
                need(metrics['n'] == len(val_y)
                     and abs(metrics['prior_brier'] - prior_brier) < 1e-14,
                     'CHECKPOINT_SCORE:' + str(fold_id))
                checkpoints.append({'rounds': rounds, 'auc': metrics['auc'],
                                    'brier': metrics['brier'],
                                    'prior_brier': prior_brier,
                                    'prior_minus_model': prior_brier - metrics['brier'],
                                    'mean_probability': metrics['mean_probability'],
                                    'up_fraction': metrics['up_fraction']})
            selected_round = choose_round(checkpoints, prior_brier)
            train_prediction = np.asarray(booster.predict(x_train,
                                             num_iteration=64,
                                             num_threads=1), dtype=np.float64)
            train_metrics = old._metrics(train_y, train_prediction, q)
            model_ref = old.base._write_bytes(out_dir,
                              f'MODEL_INNER_FOLD_{fold_id}_F_M5_SIDE.txt',
                              booster.model_to_string().encode())
            receipt = old.base._write_json(out_dir, f'FOLD_{fold_id}.json', {
                'schema_version': 'gx1_f_m5_inner_train_fold_v1',
                'fold_id': fold_id, 'frozen_split': fold,
                'train_metrics_at_64': train_metrics,
                'validation_checkpoints': checkpoints,
                'selected_round_from_inner_validation': selected_round,
                'model': model_ref, 'recipe': RECIPE,
                'outer_score_labels_used': False, 'test_data_used': False})
            completed.append(receipt)
            del x_train, x_val, dataset, booster, train_prediction
        provider.validate_metadata(chain['provider_plan'])
        _bookend(plan, plan_sha, policy_ref['sha256'], head)
        need(len(completed) == 5, 'FIVE_FOLDS')
        decisions = [old._checked_json(item)['selected_round_from_inner_validation']
                     for item in completed]
        result = old.base._write_json(out_dir, 'RESULT.json', {
            'schema_version': 'gx1_f_m5_inner_train_transfer_result_v1',
            'status': 'COMPLETE_FIVE_INNER_TRAIN_FITS',
            'plan': plan_ref, 'policy': policy_ref, 'claim': claim,
            'source_head': head, 'elapsed_seconds': time.monotonic() - started,
            'input_refs': INPUT_REFS, 'current_matrix_sha256': current_sha,
            'old_projection_sha256': old_sha,
            'splits_priors_frozen': frozen_ref, 'folds': completed,
            'selected_rounds': decisions, 'fixed_tabular_fits': 5,
            'outer_score_labels_used': False, 'test_data_used': False,
            'native_training': False, 'entry_exit_or_trading_authority': False})
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminal = old.base._write_json(out_dir, 'TERMINAL.json',
                                        terminal_payload('COMPLETE', plan_ref,
                                                         claim, result, 5))
        return {'result': result, 'terminal': terminal}
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        failure = old.base._write_json(out_dir, 'FAILED.json', {
            'schema_version': 'gx1_f_m5_inner_train_transfer_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'completed_folds': completed, 'error_type': type(exc).__name__,
            'error': str(exc)[:2048],
            'elapsed_seconds': time.monotonic() - started,
            'no_relaunch_under_this_plan': True})
        old.base._write_json(out_dir, 'TERMINAL.json',
                             terminal_payload('FAILED', plan_ref, claim,
                                              failure, len(completed)))
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
    parser.add_argument('--physical-view', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        plan, plan_ref, head, _ = preflight(args.plan, args.plan_sha256,
                                           args.out_dir,
                                           physical_view=args.physical_view)
        print(json.dumps({'status': 'READY_NO_CLAIM_NO_TARGET_PAYLOAD',
                          'plan': plan_ref, 'source_head': head,
                          'fixed_tabular_fits': 5,
                          'physical_view_checked': args.physical_view,
                          'planned_command': planned_command(args.plan_sha256),
                          'claim_created': False, 'target_payload_read': args.physical_view,
                          'model_fits': 0, 'test_data_used': False},
                         sort_keys=True))
        return 0
    print(json.dumps(run(args.plan, args.plan_sha256, args.out_dir), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
