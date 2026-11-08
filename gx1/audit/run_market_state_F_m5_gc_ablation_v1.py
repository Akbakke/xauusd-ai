"""One fixed chronological M5 F LightGBM comparison: all inputs versus no GC.

Research only. Uses the existing causal tabular view, mature fold axes, fixed
LightGBM recipe and pinned F target. No native update, TEST or trading route.
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
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_five_endpoint_prior_v1 as prior
from gx1.contracts import market_state_tabular_lightgbm_v1 as light
from gx1.contracts.market_state_tabular_target_score_v1 import binary_fit_targets
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_research_producer_execution

REPO, RUNS, POLICY = base.REPO, base.RUNS, base.POLICY
STAGE = 'market_state_F_m5_gc_ablation_20261108_001'
PLAN_PATH = REPO / 'configs/research/MARKET_STATE_F_M5_GC_ABLATION_20261108_001.json'
OUT_ROOT = RUNS / 'MARKET_STATE_F_M5_GC_ABLATION_20261108_001'
SOURCE_PATHS = base.SOURCE_PATHS | frozenset({
    'gx1/audit/run_market_state_F_m5_gc_ablation_v1.py',
    'gx1/audit/run_market_state_tabular_oof_v1.py',
    'gx1/contracts/market_state_future_displacement_candidate_v1.py',
    'gx1/contracts/market_state_tabular_target_score_v1.py',
})
INPUTS = {**base.INPUT_REFS,
          'F_result': {'path': str(future.REFS['result'][0]), 'sha256': future.REFS['result'][1]},
          'F_terminal': {'path': str(future.REFS['terminal'][0]), 'sha256': future.REFS['terminal'][1]},
          'F_tape': {'path': str(future.REFS['tape'][0]), 'sha256': future.REFS['tape'][1]}}
LIMIT = {'memory_max': 12 * 1024**3, 'memory_high': 12 * 1024**3,
         'swap': 512 * 1024**2, 'pids': 64, 'wall_seconds': 3600}
FOLDS = (1, 2, 3, 4, 5)
ARMS = ('full', 'without_gc')
HEADS = ('activity', 'conditional_side')
FITS = len(FOLDS) * len(ARMS) * len(HEADS)
DECISION_RULE = {
    'primary': 'F_M5_conditional_side_Brier_prior_minus_full',
    'gc_increment': 'F_M5_conditional_side_Brier_without_gc_minus_full',
    'positive_folds_at_least': 4, 'pooled_improvement_strictly_positive': True,
    'utc_week_bootstrap_replicates': 2000, 'utc_week_bootstrap_seed': 20261108,
    'ci_level': 0.95, 'ci_lower_bound_strictly_positive': True,
    'score_driven_recipe_selection': False,
}
OUTPUT_LIMIT = 128 * 1024**2
WRITTEN = 0


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('F_M5_GC_ABLATION_' + reason)


def sha(path: Path) -> str:
    return base._sha(path)


def ref(path: Path) -> dict:
    return {'path': str(path), 'sha256': sha(path)}


def write(root: Path, name: str, data: bytes) -> dict:
    global WRITTEN
    need(root == OUT_ROOT and '/' not in name and data
         and (name in ('FAILED.json', 'TERMINAL.json') or WRITTEN + len(data) <= OUTPUT_LIMIT),
         'WRITE_SCOPE')
    result = base._write_bytes(root, name, data)
    WRITTEN += len(data)
    return result


def write_json(name: str, value: dict) -> dict:
    return write(OUT_ROOT, name,
                 (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


def feature_indices(names: tuple[str, ...]) -> dict[str, np.ndarray]:
    need(len(names) == light.EXPECTED_WIDTH and len(set(names)) == len(names), 'FEATURE_NAMES')
    gc = np.asarray([j for j, name in enumerate(names) if 'gc_' in name], np.int64)
    other = np.asarray([j for j, name in enumerate(names) if 'gc_' not in name], np.int64)
    need(len(gc) == 2410 and len(other) == light.EXPECTED_WIDTH - len(gc)
         and all(names[j].startswith('signal.') for j in gc)
         and all('gc_' not in names[j] for j in other), 'GC_COLUMN_PARTITION')
    return {'full': np.arange(len(names), dtype=np.int64), 'without_gc': other,
            'gc': gc}


def fixed_recipe() -> dict:
    return {'library': 'lightgbm', 'version': light.LIGHTGBM_VERSION,
            'params': dict(light.RECIPE), 'boost_rounds': light.BOOST_ROUNDS,
            'timeframe': 'M5', 'heads': list(HEADS), 'arms': list(ARMS),
            'fixed_fits': FITS, 'early_stopping': False,
            'class_weighting': False, 'score_driven_selection': False}


def planned_command(digest: str) -> list[str]:
    return ['scripts/gx1_capped_run.sh', '--class', 'producer', '--mem', '12G',
            '--swap', '512M', '--', '/usr/bin/timeout', '--signal=TERM',
            '--kill-after=10s', '3600s', '.venv/bin/python', '-m',
            'gx1.audit.run_market_state_F_m5_gc_ablation_v1', '--plan',
            str(PLAN_PATH), '--plan-sha256', digest]


def inspect(plan_ref: dict, *, admitted: bool) -> tuple[dict, dict | None, str]:
    head = base._clean_head()
    need(plan_ref == ref(PLAN_PATH) and not OUT_ROOT.exists()
         and subprocess.check_output(('git', 'branch', '--show-current'),
                                     cwd=REPO, text=True).strip() == 'work/gx1-current',
         'PLAN_IDENTITY')
    plan = json.loads(PLAN_PATH.read_text())
    chain = base._reference_chain()
    need(set(plan) == {'schema_version', 'stage', 'status', 'source_base_head',
                       'source_owners', 'input_refs', 'cohorts', 'score_axes',
                       'scope', 'recipe', 'decision_rule', 'limits', 'command_template', 'output_root'}
         and plan['schema_version'] == 'gx1_market_state_F_m5_gc_ablation_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['output_root'] == str(OUT_ROOT)
         and plan['source_owners'] == {path: sha(REPO / path) for path in SOURCE_PATHS}
         and plan['input_refs'] == INPUTS
         and plan['cohorts'] == chain['cohorts']
         and plan['score_axes'] == base.SCORE_AXES
         and plan['recipe'] == fixed_recipe()
         and plan['decision_rule'] == DECISION_RULE
         and plan['limits'] == LIMIT
         and plan['command_template'] == planned_command('{plan_sha256}')
         and plan['scope'] == {'fold_ids': list(FOLDS), 'timeframe': 'M5',
                               'target': future.contract()['contract_sha256'],
                               'arms': list(ARMS), 'heads': list(HEADS),
                               'fixed_fits': FITS, 'boost_rounds': light.BOOST_ROUNDS,
                               'primary': 'paired_M5_F_conditional_side_Brier_prior_minus_full',
                               'secondary_gc': 'paired_M5_F_conditional_side_Brier_without_gc_minus_full',
                               'label_threshold': future.HALF_ATR_T,
                               'no_hpo_or_score_selected_recipe': True,
                               'native_training': False, 'test_data_used': False}
         and subprocess.run(('git', 'merge-base', '--is-ancestor',
                             plan['source_base_head'], head), cwd=REPO,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0,
         'PLAN_SCOPE')
    for key, item in INPUTS.items():
        path = Path(item['path'])
        need(path.is_file() and not path.is_symlink() and sha(path) == item['sha256'],
             'INPUT_REF:' + key)
    need(Path(__file__).resolve() == REPO / 'gx1/audit/run_market_state_F_m5_gc_ablation_v1.py',
         'IMPORT_PATH')
    source._provider_module()
    light._import_lightgbm()
    bound = future.load_bound_future_displacement()
    old = state.load_bound_five_endpoint_targets()
    need(np.array_equal(bound.parent_rows, old.parent_rows)
         and np.array_equal(bound.decision_ns, old.decision_ns), 'TARGET_AXES')
    policy_ref = None
    if admitted:
        cap = require_capped_cpu_research_producer_execution()
        need(cap['memory_max'] == cap['memory_high'] == LIMIT['memory_max']
             and cap['swap'] == LIMIT['swap'] and cap['pids'] == LIMIT['pids']
             and isinstance(cap['lock'], dict), 'CAP')
        policy_ref = ref(POLICY)
        policy = json.loads(POLICY.read_text())
        gate = policy.get(STAGE, {})
        need(policy.get('training_enabled') is True
             and policy.get('full_epoch_training_allowed') is False
             and policy.get('native_training_enabled') is False
             and policy.get('native_launch_authorized') is False
             and gate.get('status') == 'ADMITTED'
             and gate.get('plan') == plan_ref
             and gate.get('max_invocations') == 1
             and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
             and gate.get('fixed_fits_authorized') == FITS
             and gate.get('output_root') == str(OUT_ROOT)
             and gate.get('memory_max') == '12G'
             and gate.get('swap_max') == '512M'
             and gate.get('wall_seconds') == 3600
             and gate.get('execution_authorized') is True
             and gate.get('data_reads_authorized') is True
             and gate.get('research_fit_authorized') is True
             and gate.get('native_training_authorized') is False
             and gate.get('test_authorized') is False
             and gate.get('checkpoint_reads_authorized') is False
             and gate.get('broker_access_allowed') is False
             and gate.get('test_data_used') is False
             and gate.get('relaunch_allowed') is False,
             'POLICY')
    return plan, policy_ref, head


def preflight_feature_view(plan: dict) -> dict:
    cap = require_capped_cpu_research_producer_execution()
    need(cap['memory_max'] == cap['memory_high'] == LIMIT['memory_max']
         and cap['swap'] == LIMIT['swap'] and cap['pids'] == LIMIT['pids']
         and isinstance(cap['lock'], dict), 'PREFLIGHT_CAP')
    chain = base._reference_chain()
    provider = source._provider_module()
    provider_plan = chain['provider_plan']
    info = provider.validate_metadata(provider_plan)
    prepared = source._prepare_inputs(provider, provider_plan, info, chain['gc_layout'])
    try:
        source._validate_axes(plan['cohorts'], prepared['axes'])
        view = base._view(prepared)
        columns = feature_indices(view.feature_names)
        target = future.load_bound_future_displacement()
        need(np.array_equal(target.parent_rows, view.parent_rows)
             and np.array_equal(target.decision_ns, view.decision_ns),
             'PREFLIGHT_FEATURE_TARGET_AXES')
        state_bound = state.load_bound_five_endpoint_targets()
        for fold in FOLDS:
            cohort = state_bound.mature_oof_axes(fold_id=fold)
            need(all(cohort[key] == plan['cohorts'][fold - 1][key]
                     for key in plan['cohorts'][fold - 1]),
                 'PREFLIGHT_COHORT:' + str(fold))
            truth = target.gather(cohort['fit_parent_rows'], cohort['fit_decision_ns'],
                                  fold_id=fold, partition='fit')
            binary_fit_targets(truth['allowed'], truth['eligible'], truth['mature'])
        provider.validate_metadata(provider_plan)
        return {'feature_rows': len(view.parent_rows), 'feature_width': view.width,
                'gc_columns': len(columns['gc']), 'cohorts': len(FOLDS)}
    finally:
        source._release_pages(provider, prepared)


def weeks(issues: np.ndarray) -> np.ndarray:
    days = issues // (24 * 60 * 60 * 1_000_000_000)
    monday = days - (days + 3) % 7
    return (np.datetime64('1970-01-01', 'D') + monday.astype('timedelta64[D]')).astype(str)


def score(prob: dict[str, np.ndarray], truth: dict, prior_prob: tuple[float, float]) -> dict:
    allowed = truth['allowed'][:, 0]
    ready = truth['eligible'][:, 0] & truth['mature'][:, 0]
    masks = {'activity': ready & (allowed != 7),
             'conditional_side': ready & np.isin(allowed, (1, 4))}
    labels = {'activity': np.isin(allowed, (1, 4, 5)).astype(np.float64),
              'conditional_side': (allowed == 4).astype(np.float64)}
    result = {}
    for head in HEADS:
        mask = masks[head]
        y = labels[head][mask]
        result[head] = {'count': int(mask.sum()), 'positive': int(y.sum())}
        for arm in ARMS:
            prediction = prob[arm][:, HEADS.index(head)][mask]
            need(np.isfinite(prediction).all()
                 and np.all((prediction > 0) & (prediction < 1)), 'PROBABILITY')
            result[head][arm + '_brier_sum'] = float(np.square(prediction - y).sum(dtype=np.float64))
        result[head]['prior_brier_sum'] = float(np.square(prior_prob[HEADS.index(head)] - y).sum(dtype=np.float64))
    return result


def fit_head(lgb, train: np.ndarray, score_matrix: np.ndarray, mask: np.ndarray,
             label: np.ndarray, names: list[str], categories: list[int]):
    selected = np.ascontiguousarray(train[mask], dtype=np.float32)
    y = label[mask]
    counts = np.bincount(y, minlength=2)
    need(len(y) >= 2 and counts[0] > 0 and counts[1] > 0, 'TWO_CLASSES')
    dataset = lgb.Dataset(selected, label=y, feature_name=names,
                          categorical_feature=categories, free_raw_data=True)
    booster = lgb.train(dict(light.RECIPE), dataset, num_boost_round=light.BOOST_ROUNDS)
    prediction = np.asarray(booster.predict(score_matrix, num_threads=1), np.float64)
    need(prediction.shape == (len(score_matrix),)
         and np.isfinite(prediction).all()
         and np.all((prediction > 0) & (prediction < 1)), 'FIT_PREDICTION')
    return booster, prediction, {'rows': len(y), 'negative': int(counts[0]),
                                 'positive': int(counts[1])}


def time_guard(deadline: float) -> None:
    need(time.monotonic() + 120 < deadline, 'WALL_RESERVE')


def timeout(_signum, _frame) -> None:
    raise TimeoutError('F_M5_GC_ABLATION_WALL')


def run(plan_ref: dict) -> dict:
    global WRITTEN
    began = time.monotonic()
    plan, policy_ref, head = inspect(plan_ref, admitted=True)
    need(policy_ref is not None, 'POLICY_REQUIRED')
    deadline = began + LIMIT['wall_seconds'] - 100
    need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'SIGNAL_STATE')
    OUT_ROOT.mkdir(parents=True, exist_ok=False)
    WRITTEN = 0
    base.PUBLISHED_BYTES = 0
    claim = None
    completed = []
    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, LIMIT['wall_seconds'] - 100)
        claim = write_json('CLAIM.json', {'schema_version': 'gx1_F_m5_gc_ablation_claim_v1',
                          'plan': plan_ref, 'policy': policy_ref, 'source_head': head,
                          'fixed_fits': FITS, 'test_data_used': False})
        chain = base._reference_chain()
        provider = source._provider_module()
        provider_plan = chain['provider_plan']
        info = provider.validate_metadata(provider_plan)
        prepared = source._prepare_inputs(provider, provider_plan, info, chain['gc_layout'])
        source._validate_axes(plan['cohorts'], prepared['axes'])
        view = base._view(prepared)
        indices = feature_indices(view.feature_names)
        target = future.load_bound_future_displacement()
        need(np.array_equal(target.parent_rows, view.parent_rows)
             and np.array_equal(target.decision_ns, view.decision_ns), 'FEATURE_TARGET_AXES')
        prior_rows = []
        axes = []
        state_bound = state.load_bound_five_endpoint_targets()
        for fold in FOLDS:
            cohort = state_bound.mature_oof_axes(fold_id=fold)
            need(all(cohort[key] == plan['cohorts'][fold - 1][key]
                     for key in plan['cohorts'][fold - 1]), 'COHORT:' + str(fold))
            truth = target.gather(cohort['fit_parent_rows'], cohort['fit_decision_ns'],
                                  fold_id=fold, partition='fit')
            binary = binary_fit_targets(truth['allowed'], truth['eligible'], truth['mature'])
            fitted = prior.fit_activity_side_prior(truth['allowed'], truth['eligible'], truth['mature'])
            p = fitted['probabilities'][0]
            prior_rows.append({'fold_id': fold, 'counts': fitted['counts']['M5'],
                               'activity': float(p[0] + p[2]),
                               'conditional_side': float(p[2] / (p[0] + p[2]))})
            axes.append((cohort, binary))
        prior_ref = write_json('PRIOR_FROZEN.json', {'schema_version': 'gx1_F_m5_gc_prior_v1',
                               'plan': plan_ref, 'folds': prior_rows,
                               'before_any_fit': True, 'test_data_used': False})
        matrix, digest = base._matrix(view, deadline)
        schema_ref = write_json('FEATURE_SCHEMA.json', {
            'schema_version': 'gx1_F_m5_gc_feature_schema_v1', 'rows': len(view.parent_rows),
            'width': view.width, 'names_sha256': hashlib.sha256(
                json.dumps(view.feature_names).encode()).hexdigest(),
            'matrix_sha256': digest, 'gc_column_count': len(indices['gc']),
            'without_gc_column_count': len(indices['without_gc']),
            'gc_indices_sha256': hashlib.sha256(indices['gc'].tobytes()).hexdigest(),
            'no_future_targets_in_features': True})
        lgb = light._import_lightgbm()
        category_map = {item['index'] for item in view.categorical}
        for fold, (cohort, binary) in zip(FOLDS, axes):
            time_guard(deadline)
            truth = target.gather(cohort['score_parent_rows'], cohort['score_decision_ns'],
                                  fold_id=fold, partition='score')
            fit_rows = cohort['joined_fit_rows']
            score_begin = cohort['old_joined_fit_rows']
            score_end = score_begin + len(cohort['score_parent_rows'])
            need(np.array_equal(view.parent_rows[:fit_rows], cohort['fit_parent_rows'])
                 and np.array_equal(view.parent_rows[score_begin:score_end], cohort['score_parent_rows']),
                 'FOLD_VIEW')
            prediction = {arm: np.empty((score_end-score_begin, 2), np.float64) for arm in ARMS}
            weights = {}
            fit_counts = {}
            for arm in ARMS:
                cols = indices[arm]
                train = np.ascontiguousarray(matrix[:fit_rows, cols], np.float32)
                score_matrix = np.ascontiguousarray(matrix[score_begin:score_end, cols], np.float32)
                names = [view.feature_names[j] for j in cols]
                categories = [i for i, j in enumerate(cols) if int(j) in category_map]
                need(len(categories) == len(view.categorical), 'CATEGORICAL_ABLATION')
                for j, (head, mask_key, label_key) in enumerate((
                    ('activity', 'activity_mask', 'activity_label'),
                    ('conditional_side', 'certified_side_mask', 'side_up_label'))):
                    booster, pred, counts = fit_head(lgb, train, score_matrix,
                        binary[mask_key][:, 0], binary[label_key][:, 0], names, categories)
                    prediction[arm][:, j] = pred
                    fit_counts[arm + ':' + head] = counts
                    weights[arm + ':' + head] = write(
                        OUT_ROOT, f'MODEL_FOLD_{fold}_{arm}_{head}.txt',
                        booster.model_to_string().encode())
                del train, score_matrix
            frozen = prior_rows[fold-1]
            p = (frozen['activity'], frozen['conditional_side'])
            overall = score(prediction, truth, p)
            week_axis = weeks(truth['issue_ns'])
            weekly = {str(week): score(prediction_subset(prediction, week_axis == week),
                                      subset_truth(truth, week_axis == week), p)
                      for week in np.unique(week_axis)}
            for head in HEADS:
                need(sum(x[head]['count'] for x in weekly.values()) == overall[head]['count']
                     and all(abs(sum(x[head][key] for x in weekly.values()) - overall[head][key]) < 1e-6
                             for key in ('full_brier_sum', 'without_gc_brier_sum', 'prior_brier_sum')),
                     'WEEK_PARTITION')
            stream = io.BytesIO()
            np.savez_compressed(stream, parent_rows=cohort['score_parent_rows'],
                                issue_ns=cohort['score_decision_ns'],
                                full=prediction['full'], without_gc=prediction['without_gc'])
            pred_ref = write(OUT_ROOT, f'PREDICTIONS_FOLD_{fold}.npz', stream.getvalue())
            completed.append(write_json(f'FOLD_{fold}.json', {
                'schema_version': 'gx1_F_m5_gc_fold_v1', 'fold_id': fold,
                'fit_rows': fit_rows, 'score_rows': score_end-score_begin,
                'fit_counts': fit_counts, 'weights': weights, 'predictions': pred_ref,
                'overall': overall, 'by_utc_week': weekly,
                'fit_parent_rows_sha256': cohort['fit_parent_rows_sha256'],
                'score_parent_rows_sha256': cohort['score_parent_rows_sha256'],
                'test_data_used': False}))
        provider.validate_metadata(provider_plan)
        source_bookend(plan, plan_ref, policy_ref, head)
        result = write_json('RESULT.json', {
            'schema_version': 'gx1_F_m5_gc_result_v1', 'status': 'COMPLETE',
            'plan': plan_ref, 'policy': policy_ref, 'claim': claim,
            'prior_frozen': prior_ref, 'feature_schema': schema_ref,
            'folds': completed, 'fixed_fits': FITS,
            'recipe': fixed_recipe(), 'decision_rule': DECISION_RULE,
            'quality_decision': 'PENDING_INDEPENDENT_WEEK_REVIEW',
            'elapsed_seconds': time.monotonic()-began,
            'native_training': False, 'test_data_used': False,
            'scope_limit': 'Fixed M5 F tabular diagnostic only; not five-TF native prediction, Entry, or economic edge.'})
        terminal = write_json('TERMINAL.json', {
            'schema_version': 'gx1_F_m5_gc_terminal_v1', 'status': 'COMPLETE',
            'exit_code': 0, 'plan': plan_ref, 'claim': claim, 'result': result,
            'fold_count': len(completed), 'fixed_fits': FITS,
            'test_data_used': False, 'no_relaunch_under_this_plan': True})
        return {'result': result, 'terminal': terminal}
    except BaseException as exc:
        failed = write_json('FAILED.json', {
            'schema_version': 'gx1_F_m5_gc_failure_v1', 'plan': plan_ref,
            'claim': claim, 'completed_folds': completed,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'elapsed_seconds': time.monotonic()-began,
            'no_relaunch_under_this_plan': True})
        write_json('TERMINAL.json', {
            'schema_version': 'gx1_F_m5_gc_terminal_v1', 'status': 'FAILED',
            'exit_code': 1, 'plan': plan_ref, 'claim': claim, 'failure': failed,
            'fold_count': len(completed), 'test_data_used': False,
            'no_relaunch_under_this_plan': True})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def prediction_subset(prediction: dict, mask: np.ndarray) -> dict:
    return {key: value[mask] for key, value in prediction.items()}


def subset_truth(truth: dict, mask: np.ndarray) -> dict:
    return {key: value[mask] if isinstance(value, np.ndarray) and len(value) == len(mask)
            else value for key, value in truth.items()}


def source_bookend(plan: dict, plan_ref: dict, policy_ref: dict, head: str) -> None:
    need(base._clean_head() == head and ref(PLAN_PATH) == plan_ref
         and ref(POLICY) == policy_ref
         and all(sha(REPO/path) == digest for path, digest in plan['source_owners'].items())
         and all(sha(Path(item['path'])) == item['sha256'] for item in INPUTS.values()),
         'BOOKEND')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    plan_ref = {'path': args.plan, 'sha256': args.plan_sha256}
    if args.preflight:
        plan, _, head = inspect(plan_ref, admitted=False)
        view_report = preflight_feature_view(plan)
        print(json.dumps({'status': 'READY_NO_CLAIM_NO_FIT',
                          'plan': plan_ref, 'source_head': head,
                          'fixed_fits': FITS, 'view': view_report,
                          'cohort_count': len(plan['cohorts']),
                          'claim_created': False, 'model_fits': 0,
                          'test_data_used': False}, sort_keys=True))
    else:
        print(json.dumps(run(plan_ref), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
