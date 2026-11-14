"""One-use fivefold F-M5 side diagnostic on the existing causal tabular view.

This CPU worker fits one fixed LightGBM head per mature TRAIN prefix. It has no
native trainer, sealed TEST, Entry, Exit, broker, or trading route.
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

from gx1.audit import run_market_state_tabular_oof_v1 as base
from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as source
from gx1.contracts import market_state_future_displacement_candidate_v1 as future
from gx1.contracts import market_state_tabular_lightgbm_v1 as model
from gx1.contracts import market_state_tabular_weekly_review_v1 as review
from gx1.contracts.market_state_tabular_target_score_v1 import mature_fold_training_targets
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_research_producer_execution
from gx1.contracts.immutable_event_authority_v1 import _fsync_directory

REPO = base.REPO
RUNS = base.RUNS
STAGE = 'market_state_f_m5_tabular_oof_20261114_001'
SCHEMA = 'gx1_market_state_f_m5_tabular_oof_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
POLICY = REPO / 'NEXT_RUN_POLICY.json'
FEATURE_SCHEMA = {'path': str(base.OUT_ROOT / 'FEATURE_SCHEMA.json'),
                  'sha256': 'a9ba2702805a89de9e1a5e703ee2487de4425013f6a1b9f60b5d38edfbb1a22a'}
TARGET_AUDIT = {'path': str(REPO / 'docs/TA_M5_F_ONLY_FIVEFOLD_TARGET_AUDIT_20261114.json'),
                'sha256': '6c06bd481d7466e9eb861f86f15906d8a17f587d599cca26d171ad542bda0518'}
INPUT_REFS = {**base.INPUT_REFS,
              **{'future_' + name: {'path': str(path), 'sha256': digest}
                 for name, (path, digest) in future.REFS.items()},
              'tabular_feature_schema': FEATURE_SCHEMA, 'fivefold_target_audit': TARGET_AUDIT}
SOURCE_PATHS = base.SOURCE_PATHS | frozenset({
    'gx1/audit/run_market_state_f_m5_tabular_oof_v1.py',
    'gx1/contracts/market_state_future_displacement_candidate_v1.py',
})
FOLD_IDS = base.FOLD_IDS
SCORE_AXES = base.SCORE_AXES
HEAD_COUNT = 5
LIMITS = base.LIMITS
EXECUTION = {**base.EXECUTION, 'module': 'gx1.audit.run_market_state_f_m5_tabular_oof_v1',
             'fixed_tabular_fits': HEAD_COUNT}
RECIPE = {'target': 'F_M5_certified_conditional_side_only',
          'library': 'lightgbm', 'version': model.LIGHTGBM_VERSION,
          'boost_rounds': model.BOOST_ROUNDS, 'params': dict(model.RECIPE),
          'early_stopping': False, 'class_weighting': False,
          'score_driven_selection': False}
DECISION_RULE = {**base.DECISION_RULE,
                 'primary': 'F_M5_certified_side_Brier_train_prior_minus_model',
                 'auc': 'supplementary_tie_aware',
                 'target_strata': 'retrospective_only_not_causal_gates'}


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('MARKET_STATE_F_M5_TABULAR_OOF_' + reason)


def _sha(path: Path) -> str:
    return base._sha(path)


def _checked_json(item: dict) -> dict:
    return base._checked_json(item)


def _inputs() -> tuple[dict, dict, dict]:
    chain = base._reference_chain()
    for name, item in INPUT_REFS.items():
        if name.startswith('future_'):
            path = Path(item['path'])
            _need(path.is_file() and not path.is_symlink()
                  and _sha(path) == item['sha256'], 'FUTURE_REF:' + name)
    schema = _checked_json(FEATURE_SCHEMA)
    audit = _checked_json(TARGET_AUDIT)
    _need(schema.get('schema_version') == 'gx1_market_state_tabular_feature_schema_v1'
          and schema.get('no_future_targets_in_features') is True
          and len(audit.get('folds', [])) == 5, 'REUSED_SCHEMA_AND_AUDIT')
    return chain, schema, audit


def _bookend(plan: dict, plan_sha: str, policy_sha: str, head: str) -> None:
    _need(base._clean_head() == head and _sha(PLAN_PATH) == plan_sha
          and _sha(POLICY) == policy_sha
          and all(_sha(REPO / path) == digest
                  for path, digest in plan['source_owners'].items()), 'SOURCE_BOOKEND')
    for name, item in INPUT_REFS.items():
        _need(_sha(Path(item['path'])) == item['sha256'], 'INPUT_BOOKEND:' + name)


def _policy_gate(policy: dict, plan_ref: dict, out_dir: Path) -> None:
    gate = policy.get(STAGE, {})
    _need(policy.get('training_enabled') is False
          and policy.get('native_training_enabled') is False
          and policy.get('native_launch_authorized') is False
          and policy.get('full_val_allowed') is False
          and gate.get('status') == gate.get('decision') == 'ADMITTED'
          and gate.get('plan') == plan_ref
          and gate.get('execution_authorized') is True
          and gate.get('data_reads_authorized') is True
          and gate.get('research_fit_authorized') is True
          and gate.get('fixed_tabular_fits_authorized') == HEAD_COUNT
          and gate.get('native_training_authorized') is False
          and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
          and gate.get('max_invocations') == 1
          and gate.get('relaunch_allowed') is False
          and gate.get('memory_max') == EXECUTION['memory']
          and gate.get('swap_max') == EXECUTION['swap']
          and gate.get('wall_seconds') == EXECUTION['timeout_seconds']
          and gate.get('test_data_used') is False
          and gate.get('checkpoint_reads_authorized') is False
          and gate.get('broker_access_allowed') is False
          and gate.get('test_authorized') is False
          and gate.get('output_root') == str(out_dir), 'POLICY_GATE')


def planned_command(plan_sha: str) -> list[str]:
    return ['timeout', '--signal=TERM', '--kill-after=10s', '10800s',
            'scripts/gx1_capped_run.sh', '--class', 'producer', '--mem', '12G',
            '--swap', '512M', '--', '.venv/bin/python', '-m', EXECUTION['module'],
            '--plan', str(PLAN_PATH), '--plan-sha256', plan_sha,
            '--out-dir', str(OUT_ROOT)]


def preflight(plan_path: Path, plan_sha: str, out_dir: Path, *,
              admitted: bool = False) -> tuple[dict, dict, str, dict | None]:
    head = base._clean_head()
    _need(plan_path == PLAN_PATH and out_dir == OUT_ROOT and not out_dir.exists()
          and len(plan_sha) == 64 and _sha(plan_path) == plan_sha, 'PLAN_IDENTITY')
    plan = json.loads(plan_path.read_text())
    chain, _, audit = _inputs()
    source._validate_cohort_contract(plan.get('cohorts'))
    _need(plan.get('cohorts') == chain['cohorts']
          and plan.get('schema_version') == SCHEMA
          and plan.get('stage') == STAGE
          and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and plan.get('output_root') == str(OUT_ROOT)
          and plan.get('max_invocations') == 1
          and plan.get('fold_ids') == FOLD_IDS
          and plan.get('score_axes') == SCORE_AXES
          and plan.get('command_template') == planned_command('{plan_sha256}')
          and plan.get('input_refs') == INPUT_REFS
          and plan.get('execution') == EXECUTION
          and plan.get('limits') == LIMITS
          and plan.get('recipe') == RECIPE
          and plan.get('decision_rule') == DECISION_RULE
          and plan.get('target_audit_counts') == [
              {'fold_id': i + 1,
               'fit_f_known_n': audit['folds'][i]['fit']['F_m5_known_side_n'],
               'score_f_known_n': audit['folds'][i]['score']['F_m5_known_side_n']}
              for i in range(5)]
          and plan.get('research_fit_authorized') is True
          and plan.get('fixed_tabular_fits_authorized') == HEAD_COUNT
          and plan.get('native_training_authorized') is False
          and plan.get('test_data_used') is False
          and all(audit['folds'][i]['score']['rows'] == SCORE_AXES[i]['score_rows']
                  for i in range(5))
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
    _need(Path(__file__).resolve() == REPO / 'gx1/audit/run_market_state_f_m5_tabular_oof_v1.py',
          'IMPORT_PATH')
    source._provider_module()
    model._import_lightgbm()
    plan_ref = {'path': str(plan_path), 'sha256': plan_sha}
    policy_ref = None
    if admitted:
        cap = require_capped_cpu_research_producer_execution()
        _need(cap['memory_max'] == cap['memory_high'] == LIMITS['memory_max']
              and cap['swap'] == LIMITS['swap']
              and cap['pids'] == LIMITS['pids']
              and cap['cpu_affinity'] == list(range(8))
              and cap['numerical_threads'] == 1
              and isinstance(cap['lock'], dict), 'CAP_LIMITS')
        policy_ref = {'path': str(POLICY), 'sha256': _sha(POLICY)}
        policy = json.loads(POLICY.read_text())
        _need(_sha(POLICY) == policy_ref['sha256'], 'POLICY_STABLE_READ')
        _policy_gate(policy, plan_ref, out_dir)
    return plan, plan_ref, head, policy_ref


def _selected(truth: dict) -> np.ndarray:
    return (truth['eligible'][:, 0] & truth['mature'][:, 0]
            & np.isin(truth['allowed'][:, 0], (1, 4)))


def _prior_record(f_truth: dict, *, fold_id: int, expected_count: int) -> dict:
    mask = _selected(f_truth)
    labels = (f_truth['allowed'][mask, 0] == 4).astype(np.uint8)
    up = int(labels.sum())
    _need(len(labels) == expected_count and 0 < up < len(labels),
          'F_TRAIN_SUPPORT:' + str(fold_id))
    return {'fold_id': fold_id, 'known_side_n': len(labels), 'up_n': up,
            'down_n': len(labels) - up, 'laplace_up_probability': (up + 1) / (len(labels) + 2)}


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    positives = int(y.sum())
    negatives = len(y) - positives
    _need(positives > 0 and negatives > 0, 'AUC_TWO_CLASS')
    order = np.argsort(p, kind='mergesort')
    sorted_p = p[order]
    sorted_y = y[order]
    _, first, count = np.unique(sorted_p, return_index=True, return_counts=True)
    average_ranks = first + (count + 1) / 2
    positive_rank_sum = float(np.dot(average_ranks, np.add.reduceat(sorted_y, first)))
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _metrics(y: np.ndarray, p: np.ndarray, prior: float) -> dict:
    _need(len(y) > 0 and len(y) == len(p) and np.isfinite(p).all()
          and np.all((p > 0) & (p < 1)), 'SCORE_PROBABILITIES')
    return {'n': len(y), 'up_fraction': float(y.mean()), 'mean_probability': float(p.mean()),
            'auc': _auc(y, p),
            'brier': float(np.square(p - y).mean()),
            'prior_brier': float(np.square(prior - y).mean()),
            'log_loss': float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
            'prior_log_loss': float(-(y * np.log(prior)
                                      + (1 - y) * np.log1p(-prior)).mean())}


def _prob_tensor(p: np.ndarray) -> np.ndarray:
    output = np.zeros((len(p), 5, 3), dtype=np.float64)
    output[:, :, 0] = (1 - p)[:, None]
    output[:, :, 2] = p[:, None]
    return output


def _fold(plan: dict, s_bound, f_bound, view, matrix: np.ndarray,
          prior: dict, fold_id: int, out_dir: Path, deadline: float) -> tuple[dict, dict]:
    base._time_guard(deadline)
    cohort = mature_fold_training_targets(s_bound, fold_id=fold_id)['cohort']
    declaration = plan['cohorts'][fold_id - 1]
    axes = plan['score_axes'][fold_id - 1]
    _need(all(cohort[key] == declaration[key] for key in declaration)
          and axes == {'fold_id': fold_id,
                       'score_rows': len(cohort['score_parent_rows']),
                       'score_parent_rows_sha256': cohort['score_parent_rows_sha256'],
                       'score_decision_ns_sha256': cohort['score_decision_ns_sha256']}
          and np.array_equal(view.parent_rows[:cohort['joined_fit_rows']], cohort['fit_parent_rows'])
          and np.array_equal(view.decision_ns[:cohort['joined_fit_rows']], cohort['fit_decision_ns'])
          and np.array_equal(view.parent_rows[cohort['old_joined_fit_rows']:
                                               cohort['old_joined_fit_rows'] + axes['score_rows']],
                             cohort['score_parent_rows'])
          and np.array_equal(view.decision_ns[cohort['old_joined_fit_rows']:
                                               cohort['old_joined_fit_rows'] + axes['score_rows']],
                             cohort['score_decision_ns']), 'FOLD_AXES:' + str(fold_id))
    fit = f_bound.gather(cohort['fit_parent_rows'], cohort['fit_decision_ns'],
                         fold_id=fold_id, partition='fit')
    score = f_bound.gather(cohort['score_parent_rows'], cohort['score_decision_ns'],
                           fold_id=fold_id, partition='score')
    _need(fit['mature'].all() and _prior_record(
        fit, fold_id=fold_id, expected_count=prior['known_side_n']) == prior,
        'F_PRIOR:' + str(fold_id))
    fit_mask = _selected(fit)
    fit_y = (fit['allowed'][fit_mask, 0] == 4).astype(np.uint8)
    score_matrix = matrix[cohort['old_joined_fit_rows']:
                          cohort['old_joined_fit_rows'] + axes['score_rows']]
    fit_matrix = np.ascontiguousarray(matrix[:cohort['joined_fit_rows']][fit_mask])
    categories = model._categorical_indices(view.feature_names, view.categorical,
                                              fit_matrix, score_matrix)
    lgb = model._import_lightgbm()
    dataset = lgb.Dataset(fit_matrix, label=fit_y,
                          feature_name=list(view.feature_names),
                          categorical_feature=categories, free_raw_data=True)
    booster = lgb.train(dict(model.RECIPE), dataset, num_boost_round=model.BOOST_ROUNDS)
    prediction = np.asarray(booster.predict(score_matrix, num_threads=1), dtype=np.float64)
    _need(prediction.shape == (axes['score_rows'],)
          and np.isfinite(prediction).all()
          and np.all((prediction > 0) & (prediction < 1)), 'PREDICTION:' + str(fold_id))
    selected = _selected(score)
    label = (score['allowed'][selected, 0] == 4).astype(np.uint8)
    p0 = prior['laplace_up_probability']
    metrics = _metrics(label, prediction[selected], p0)
    _need(metrics['n'] == plan['target_audit_counts'][fold_id - 1]['score_f_known_n'],
          'F_SCORE_SUPPORT:' + str(fold_id))
    s_score = s_bound.gather(cohort['score_parent_rows'], cohort['score_decision_ns'],
                             fold_id=fold_id, partition='score')
    overlap = selected & s_score['eligible'][:, 0] & s_score['mature'][:, 0] & np.isin(
        s_score['allowed'][:, 0], (1, 4))
    only = selected & ~overlap
    strata = {'F_only': _metrics((score['allowed'][only, 0] == 4).astype(np.uint8),
                                 prediction[only], p0),
              'S_F_overlap': _metrics((score['allowed'][overlap, 0] == 4).astype(np.uint8),
                                      prediction[overlap], p0)}
    model_ref = base._write_bytes(out_dir, f'MODEL_FOLD_{fold_id}_F_M5_SIDE.txt',
                                  booster.model_to_string().encode())
    stream = io.BytesIO()
    np.savez_compressed(stream, parent_rows=cohort['score_parent_rows'],
                        decision_ns=cohort['score_decision_ns'],
                        p_up=prediction)
    pred_ref = base._write_bytes(out_dir, f'PREDICTIONS_FOLD_{fold_id}.npz',
                                 stream.getvalue())
    record = base._write_json(out_dir, f'FOLD_{fold_id}.json', {
        'schema_version': 'gx1_market_state_f_m5_tabular_fold_v1',
        'fold_id': fold_id, 'fit_rows': cohort['joined_fit_rows'],
        'fit_known_side_n': prior['known_side_n'], 'score_rows': axes['score_rows'],
        'score_known_side_n': metrics['n'], 'score_axes': axes,
        'prior': prior, 'metrics': metrics, 'retrospective_strata': strata,
        'model': model_ref, 'predictions': pred_ref, 'recipe': RECIPE,
        'test_data_used': False})
    weekly_input = {'fold_id': fold_id, 'decision_ns': cohort['score_decision_ns'],
                    'allowed': score['allowed'], 'eligible': score['eligible'],
                    'mature': score['mature'],
                    'model_probabilities': _prob_tensor(prediction),
                    'prior_probabilities': _prob_tensor(np.full(len(prediction), p0))}
    return record, weekly_input


def _wall_limit(_signum, _frame) -> None:
    raise TimeoutError('MARKET_STATE_F_M5_TABULAR_OOF_INTERNAL_WALL_LIMIT')


def run(plan_path: Path, plan_sha: str, out_dir: Path) -> dict:
    started = time.monotonic()
    plan, plan_ref, head, policy_ref = preflight(plan_path, plan_sha, out_dir, admitted=True)
    _need(policy_ref is not None, 'POLICY_REQUIRED')
    deadline = started + LIMITS['internal_wall_seconds']
    remaining = deadline - time.monotonic()
    _need(remaining > 0 and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
          'WALL_BUDGET_PRECLAIM')
    out_dir.mkdir(parents=True, exist_ok=False)
    base.PUBLISHED_BYTES = 0
    claim = None
    completed = []
    weekly_inputs = []
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _wall_limit)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        _fsync_directory(out_dir.parent)
        claim = base._write_json(out_dir, 'CLAIM.json', {
            'schema_version': 'gx1_market_state_f_m5_tabular_claim_v1',
            'invocation': 1, 'plan': plan_ref, 'policy': policy_ref,
            'source_head': head, 'fold_ids': FOLD_IDS,
            'fixed_tabular_fits': HEAD_COUNT, 'test_data_used': False})
        chain, schema, audit = _inputs()
        provider = source._provider_module()
        provider_plan = chain['provider_plan']
        info = provider.validate_metadata(provider_plan)
        prepared = source._prepare_inputs(provider, provider_plan, info, chain['gc_layout'])
        source._validate_axes(plan['cohorts'], prepared['axes'])
        view = base._view(prepared)
        s_bound = base.target.load_bound_five_endpoint_targets()
        f_bound = future.load_bound_future_displacement()
        _need(np.array_equal(s_bound.parent_rows, f_bound.parent_rows)
              and np.array_equal(s_bound.decision_ns, f_bound.decision_ns)
              and np.array_equal(view.parent_rows, f_bound.parent_rows)
              and np.array_equal(view.decision_ns, f_bound.decision_ns), 'BOUND_MODEL_AXES')
        priors = []
        for fold_id in FOLD_IDS:
            base._time_guard(deadline)
            cohort = mature_fold_training_targets(s_bound, fold_id=fold_id)['cohort']
            truth = f_bound.gather(cohort['fit_parent_rows'], cohort['fit_decision_ns'],
                                   fold_id=fold_id, partition='fit')
            _need(truth['mature'].all(), 'F_FIT_MATURITY:' + str(fold_id))
            prior = _prior_record(truth, fold_id=fold_id,
                                  expected_count=audit['folds'][fold_id - 1]['fit']['F_m5_known_side_n'])
            priors.append(prior)
        prior_ref = base._write_json(out_dir, 'PRIOR_FROZEN.json', {
            'schema_version': 'gx1_market_state_f_m5_tabular_prior_v1',
            'plan': plan_ref, 'claim': claim, 'folds': priors,
            'before_any_model_fit': True, 'test_data_used': False})
        matrix, matrix_sha = base._matrix(view, deadline)
        _need(matrix_sha == schema['matrix_float32_c_order_sha256']
              and matrix.shape == (schema['rows'], schema['width']),
              'REUSED_FEATURE_IDENTITY')
        feature_ref = base._write_json(out_dir, 'FEATURE_SCHEMA.json', {
            'schema_version': 'gx1_market_state_f_m5_tabular_feature_identity_v1',
            'reused_schema': FEATURE_SCHEMA, 'matrix_float32_c_order_sha256': matrix_sha,
            'no_future_targets_in_features': True})
        for fold_id in FOLD_IDS:
            _bookend(plan, plan_sha, policy_ref['sha256'], head)
            record, weekly = _fold(plan, s_bound, f_bound, view, matrix,
                                   priors[fold_id - 1], fold_id, out_dir, deadline)
            completed.append(record)
            weekly_inputs.append(weekly)
        base._time_guard(deadline)
        provider.validate_metadata(provider_plan)
        _bookend(plan, plan_sha, policy_ref['sha256'], head)
        _need(len(completed) == len(weekly_inputs) == 5, 'FIVE_FOLDS_REQUIRED')
        weekly = review.paired_m5_weekly_review(weekly_inputs)
        weekly_ref = base._write_json(out_dir, 'WEEKLY_REVIEW.json', weekly)
        result = base._write_json(out_dir, 'RESULT.json', {
            'schema_version': 'gx1_market_state_f_m5_tabular_result_v1',
            'status': 'COMPLETE_FIVE_FOLDS', 'plan': plan_ref, 'policy': policy_ref,
            'claim': claim, 'source_head': head, 'elapsed_seconds': time.monotonic() - started,
            'input_refs': INPUT_REFS, 'prior_frozen': prior_ref,
            'feature_schema': feature_ref, 'folds': completed,
            'fixed_tabular_fits': HEAD_COUNT, 'score_axes': SCORE_AXES,
            'decision_rule': DECISION_RULE, 'weekly_review': weekly_ref,
            'diagnostic_decision': weekly['decision'], 'native_training': False,
            'test_data_used': False, 'entry_exit_or_trading_authority': False})
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminal = base._write_json(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_f_m5_tabular_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'claim': claim, 'result': result, 'fold_count': 5,
            'fixed_tabular_fits': HEAD_COUNT, 'test_data_used': False,
            'no_relaunch_under_this_plan': True})
        return {'result': result, 'terminal': terminal}
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        failed = base._write_json(out_dir, 'FAILED.json', {
            'schema_version': 'gx1_market_state_f_m5_tabular_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'completed_folds': completed, 'error_type': type(exc).__name__,
            'error': str(exc)[:2048], 'elapsed_seconds': time.monotonic() - started,
            'no_relaunch_under_this_plan': True})
        base._write_json(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_f_m5_tabular_terminal_v1',
            'status': 'FAILED', 'exit_code': 1, 'plan': plan_ref,
            'claim': claim, 'failure': failed,
            'fold_count': len(completed), 'test_data_used': False,
            'no_relaunch_under_this_plan': True})
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
        plan, plan_ref, head, _ = preflight(args.plan, args.plan_sha256, args.out_dir)
        print(json.dumps({'status': 'READY_NO_CLAIM_NO_PAYLOAD', 'plan': plan_ref,
                          'source_head': head, 'cohorts_checked': len(plan['cohorts']),
                          'fixed_recipe': RECIPE, 'planned_command': planned_command(args.plan_sha256),
                          'claim_created': False, 'payload_read': False,
                          'test_data_used': False}, sort_keys=True))
        return 0
    print(json.dumps(run(args.plan, args.plan_sha256, args.out_dir), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
