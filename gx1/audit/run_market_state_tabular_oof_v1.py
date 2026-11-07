"""One-use, capped CPU research fit of five causal tabular market-state folds.

This worker has no native trainer, TEST, Entry, Exit, broker, or trading route.
A separate immutable plan and policy admission are required before payloads.
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

from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as source
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as target
from gx1.contracts import market_state_tabular_feature_view_v1 as features
from gx1.contracts import market_state_tabular_lightgbm_v1 as model
from gx1.contracts import market_state_tabular_weekly_review_v1 as review
from gx1.contracts.market_state_tabular_target_score_v1 import (
    mature_fold_training_targets, paired_mature_fold_scores,
)
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_research_producer_execution,
)
from gx1.contracts.immutable_event_authority_v1 import _fsync_directory, _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
STAGE = 'market_state_tabular_lightgbm_oof_20261107_001'
SCHEMA = 'gx1_market_state_tabular_lightgbm_oof_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
POLICY = REPO / 'NEXT_RUN_POLICY.json'
SOURCE_ADAPTER_PATH = 'gx1/audit/qualify_market_state_mature_prefix_normalization_v1.py'
SOURCE_ADAPTER_SHA = 'bc609287e99eead126804220f5d77b24afd30e45a2937ba62bdcdea673f7dda2'
MATURE_PLAN = {'path': str(source.PLAN_PATH),
               'sha256': 'b77795e606b35424d6189b7866337ca74c9f7dfc22371168452f5a6f6d665029'}
MATURE_RESULT = {'path': str(source.OUT_ROOT / 'RESULT.json'),
                 'sha256': '4b2772e893cdca98af91b04daa817d5b289c234f9dcd5cab5e2bac0f680c6833'}
MATURE_TERMINAL = {'path': str(source.OUT_ROOT / 'TERMINAL.json'),
                   'sha256': 'c88cca2d44189ec0284ebfea7f55e431759a43ecce49c801f59a880dff88f1bc'}
INPUT_REFS = {**source.INPUT_REFS,
              **{'target_' + name: {'path': path, 'sha256': digest}
                 for name, (path, digest) in target.REFS.items()},
              'mature_plan': MATURE_PLAN, 'mature_result': MATURE_RESULT,
              'mature_terminal': MATURE_TERMINAL}
SOURCE_PATHS = source.SOURCE_PATHS | frozenset({
    'gx1/audit/run_market_state_tabular_oof_v1.py',
    'gx1/contracts/market_state_tabular_feature_view_v1.py',
    'gx1/contracts/market_state_tabular_target_score_v1.py',
    'gx1/contracts/market_state_tabular_lightgbm_v1.py',
    'gx1/contracts/market_state_tabular_weekly_review_v1.py',
    'gx1/contracts/market_state_five_endpoint_prior_v1.py',
})
FOLD_IDS = [1, 2, 3, 4, 5]
SCORE_AXES = [
    {'fold_id': 1, 'score_rows': 17753,
     'score_parent_rows_sha256': 'afa02322eca78c7c1161f969d94d1693bba99cb9565626dcfe2efa81992eabf7',
     'score_decision_ns_sha256': '22a1b7282f908836380755186c919ec01a5061cd681e9723e5655078c7d6fe6c'},
    {'fold_id': 2, 'score_rows': 17288,
     'score_parent_rows_sha256': '07939f4346aa6bf1528961f8ff432a13682d1d13f432b0173fc1f217a52d2b2a',
     'score_decision_ns_sha256': 'cd13bc2bc6ebd373406f83a311a5666fad57e642a8a71b5800bd33b72201edc5'},
    {'fold_id': 3, 'score_rows': 17537,
     'score_parent_rows_sha256': 'a9df5d08104962bef0fb64a2a318da7ab29183f021b07cf3b38d1126b4a30281',
     'score_decision_ns_sha256': '36c16dc493482a12a20b394f102c25a4878cb07b53b83cc4a6373329e732c96b'},
    {'fold_id': 4, 'score_rows': 18072,
     'score_parent_rows_sha256': 'ec5766d4b9fa355555332580ecedaaff94aafda9960e29ac36f8f3d0067f9f31',
     'score_decision_ns_sha256': '1397529e7b438e5021f109360c4ec62e8ad30029eeecf835c5c9b466947a78f0'},
    {'fold_id': 5, 'score_rows': 17750,
     'score_parent_rows_sha256': '5b66d1fd5bc818281a70b6e5f3f95d848efee39d2be3223479252a2cdf57c2ab',
     'score_decision_ns_sha256': '61b79b53e5caba6581911b36d543349f6a1a545335ad7cccf49e861d58c2f06a'},
]
HEAD_COUNT = 50
MAX_CHUNK_ROWS = 2048
MAX_OUTPUT_BYTES = 256 * 1024**2
PUBLISHED_BYTES = 0
LIMITS = {'memory_max': 12 * 1024**3, 'memory_high': 12 * 1024**3,
          'swap': 512 * 1024**2, 'pids': 64, 'cpu_affinity': '0-7',
          'numerical_threads': 1, 'internal_wall_seconds': 10500}
EXECUTION = {'class': 'producer', 'runner': 'scripts/gx1_capped_run.sh',
             'memory': '12G', 'swap': '512M', 'timeout_seconds': 10800,
             'kill_after_seconds': 10,
             'module': 'gx1.audit.run_market_state_tabular_oof_v1',
             'folds': 5, 'fixed_tabular_fits': HEAD_COUNT}
DECISION_RULE = {
    'primary': 'M5_certified_side_Brier_prior_minus_model',
    'pooled_improvement_strictly_positive': True,
    'positive_folds_at_least': 4,
    'confidence_interval': 'UTC_ISO_week_block_paired_percentile_bootstrap',
    'bootstrap_replicates': review.BOOTSTRAP_REPS,
    'bootstrap_seed': review.SEED,
    'ci_level': 0.95,
    'ci_lower_bound_strictly_positive': True,
    'review': 'independent_after_all_five_folds',
    'score_driven_recipe_selection': False,
}


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('MARKET_STATE_TABULAR_OOF_' + reason)


def _sha(path: Path) -> str:
    return source._sha(path)


def _checked_json(item: dict) -> dict:
    path = Path(item['path'])
    _need(path.is_absolute() and path.is_file() and not path.is_symlink()
          and 0 < path.stat().st_size <= 4 * 1024**2
          and _sha(path) == item['sha256'], 'INPUT_REF:' + str(path))
    return json.loads(path.read_text())


def _reference_chain() -> dict:
    chain = source._receipt_chain()
    for name, item in INPUT_REFS.items():
        if name.startswith('target_') and name not in ('target_mask', 'target_model_axes'):
            _checked_json(item)
    mature_plan = _checked_json(MATURE_PLAN)
    result = _checked_json(MATURE_RESULT)
    terminal = _checked_json(MATURE_TERMINAL)
    _need(result.get('status') == 'COMPLETE_FIVE_MATURE_INPUT_QUALIFICATIONS_CONDITIONAL_FITS_NO_MODEL'
          and result.get('plan') == MATURE_PLAN
          and result.get('cohorts') == mature_plan.get('cohorts')
          and result.get('test_data_used') is False
          and result.get('model_constructions') == result.get('optimizer_steps') == 0
          and terminal.get('status') == 'COMPLETE'
          and terminal.get('exit_code') == 0
          and terminal.get('result') == MATURE_RESULT
          and terminal.get('test_data_used') is False,
          'MATURE_RECEIPTS')
    return {**chain, 'cohorts': mature_plan['cohorts']}


def _write_bytes(root: Path, name: str, payload: bytes) -> dict:
    global PUBLISHED_BYTES
    _need(root.is_relative_to(RUNS) or root.parent == RUNS,
          'OUTPUT_OUTSIDE_RUNS')
    _need('/' not in name and name not in ('', '.', '..') and len(payload) > 0,
          'OUTPUT_NAME')
    _need(name in ('FAILED.json', 'TERMINAL.json')
          or PUBLISHED_BYTES + len(payload) <= MAX_OUTPUT_BYTES,
          'OUTPUT_BUDGET')
    _publish_bytes(root / name, payload)
    PUBLISHED_BYTES += len(payload)
    return {'path': str(root / name), 'sha256': hashlib.sha256(payload).hexdigest()}


def _write_json(root: Path, name: str, value: dict) -> dict:
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    return _write_bytes(root, name, data)


def _write_predictions(root: Path, fold_id: int, cohort: dict,
                       probabilities: np.ndarray) -> dict:
    stream = io.BytesIO()
    np.savez_compressed(stream,
        parent_rows=cohort['score_parent_rows'],
        decision_ns=cohort['score_decision_ns'],
        probabilities=probabilities.astype(np.float64, copy=False))
    return _write_bytes(root, f'PREDICTIONS_FOLD_{fold_id}.npz', stream.getvalue())


def _clean_head() -> str:
    return source._clean_head()


def _source_bookend(plan: dict, plan_sha256: str, policy_sha256: str, head: str) -> None:
    _need(_clean_head() == head and _sha(PLAN_PATH) == plan_sha256
          and _sha(POLICY) == policy_sha256
          and all(_sha(REPO / path) == digest for path, digest in plan['source_owners'].items())
          and _sha(REPO / SOURCE_ADAPTER_PATH) == SOURCE_ADAPTER_SHA,
          'SOURCE_BOOKEND')
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
          and gate.get('output_root') == str(out_dir),
          'POLICY_GATE')


def planned_command(plan_sha256: str) -> list[str]:
    return ['timeout', '--signal=TERM', '--kill-after=10s', '10800s',
            EXECUTION['runner'], '--class', 'producer', '--mem', '12G',
            '--swap', '512M', '--', '.venv/bin/python', '-m',
            EXECUTION['module'], '--plan', str(PLAN_PATH),
            '--plan-sha256', plan_sha256, '--out-dir', str(OUT_ROOT)]


def preflight(plan_path: Path, plan_sha256: str, out_dir: Path, *,
              admitted: bool = False) -> tuple[dict, dict, str, dict | None]:
    head = _clean_head()
    _need(plan_path == PLAN_PATH and out_dir == OUT_ROOT and not out_dir.exists()
          and len(plan_sha256) == 64 and _sha(plan_path) == plan_sha256,
          'PLAN_IDENTITY')
    plan = json.loads(plan_path.read_text())
    chain = _reference_chain()
    source._validate_cohort_contract(plan.get('cohorts'))
    _need(plan['cohorts'] == chain['cohorts']
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
          and plan.get('recipe') == model.fixed_recipe()
          and plan.get('decision_rule') == DECISION_RULE
          and plan.get('research_fit_authorized') is True
          and plan.get('fixed_tabular_fits_authorized') == HEAD_COUNT
          and plan.get('native_training_authorized') is False
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
          and owners[SOURCE_ADAPTER_PATH] == SOURCE_ADAPTER_SHA
          and all(_sha(REPO / path) == digest for path, digest in owners.items()),
          'SOURCE_OWNERS')
    _need(Path(__file__).resolve() == REPO / 'gx1/audit/run_market_state_tabular_oof_v1.py',
          'IMPORT_PATH')
    source._provider_module()
    model._import_lightgbm()
    plan_ref = {'path': str(plan_path), 'sha256': plan_sha256}
    policy_ref = None
    if admitted:
        cap = require_capped_cpu_research_producer_execution()
        _need(cap['memory_max'] == cap['memory_high'] == LIMITS['memory_max']
              and cap['swap'] == LIMITS['swap']
              and cap['pids'] == LIMITS['pids']
              and cap['cpu_affinity'] == list(range(8))
              and cap['numerical_threads'] == 1
              and isinstance(cap['lock'], dict),
              'CAP_LIMITS')
        policy_ref = {'path': str(POLICY), 'sha256': _sha(POLICY)}
        policy = json.loads(POLICY.read_text())
        _need(_sha(POLICY) == policy_ref['sha256'], 'POLICY_STABLE_READ')
        _policy_gate(policy, plan_ref, out_dir)
    return plan, plan_ref, head, policy_ref


def _time_guard(deadline: float) -> None:
    _need(time.monotonic() + 300 < deadline, 'DEADLINE_RESERVE')


def _matrix(view: features.MarketStateTabularFeatureView,
            deadline: float) -> tuple[np.ndarray, str]:
    _need(features.MAX_CHUNK_ROWS == MAX_CHUNK_ROWS, 'FEATURE_CHUNK_LIMIT')
    count = len(view.parent_rows)
    matrix = np.empty((count, view.width), dtype=np.float32)
    digest = hashlib.sha256()
    for start in range(0, count, MAX_CHUNK_ROWS):
        _time_guard(deadline)
        stop = min(start + MAX_CHUNK_ROWS, count)
        positions = np.arange(start, stop, dtype=np.int64)
        chunk = view.take(positions, max_rows=MAX_CHUNK_ROWS)
        _need(chunk.dtype == np.float32
              and chunk.shape == (stop - start, view.width), 'FEATURE_CHUNK')
        matrix[start:stop] = chunk
        digest.update(memoryview(np.ascontiguousarray(chunk)))
    _need(not np.isinf(matrix).any(), 'FEATURE_INFINITY')
    return matrix, digest.hexdigest()


def _view(prepared: dict) -> features.MarketStateTabularFeatureView:
    axes = prepared['axes']
    current = prepared['current']
    return features.MarketStateTabularFeatureView(
        parent_rows=axes['parent_rows'], bar_start_ns=axes['bar_start_ns'],
        decision_ns=axes['decision_ns'], expected_axes=prepared['expected_axes'],
        signal=current['snap'], ctx_cont=current['ctx_cont'],
        ctx_cat=current['ctx_cat'], multi_tf_sources=prepared['cache'])


def _fold(plan: dict, bound: target.BoundFiveEndpointTargets,
          view: features.MarketStateTabularFeatureView, matrix: np.ndarray,
          priors: dict[int, dict], fold_id: int, out_dir: Path,
          deadline: float) -> dict:
    _time_guard(deadline)
    fit = mature_fold_training_targets(bound, fold_id=fold_id)
    cohort = fit['cohort']
    declaration = plan['cohorts'][fold_id - 1]
    score_axis = plan['score_axes'][fold_id - 1]
    _need(score_axis == {
              'fold_id': fold_id,
              'score_rows': len(cohort['score_parent_rows']),
              'score_parent_rows_sha256': cohort['score_parent_rows_sha256'],
              'score_decision_ns_sha256': cohort['score_decision_ns_sha256']},
          'SCORE_AXIS:' + str(fold_id))
    _need(fit['prior'] == priors[fold_id]
          and all(cohort[key] == declaration[key] for key in declaration)
          and np.array_equal(view.parent_rows[:cohort['joined_fit_rows']],
                             cohort['fit_parent_rows'])
          and np.array_equal(view.decision_ns[:cohort['joined_fit_rows']],
                             cohort['fit_decision_ns'])
          and np.array_equal(view.parent_rows[cohort['old_joined_fit_rows']:
                             cohort['old_joined_fit_rows'] + len(cohort['score_parent_rows'])],
                             cohort['score_parent_rows'])
          and np.array_equal(view.decision_ns[cohort['old_joined_fit_rows']:
                             cohort['old_joined_fit_rows'] + len(cohort['score_decision_ns'])],
                             cohort['score_decision_ns']),
          'FOLD_AXES:' + str(fold_id))
    trained = model.fit_predict_fold(
        matrix[:cohort['joined_fit_rows']],
        matrix[cohort['old_joined_fit_rows']:
               cohort['old_joined_fit_rows'] + len(cohort['score_parent_rows'])],
        fit['binary'], feature_names=view.feature_names,
        categorical=view.categorical)
    _need(len(trained['boosters']) == 10
          and trained['recipe'] == model.fixed_recipe(), 'TEN_FIXED_HEADS')
    score = paired_mature_fold_scores(
        bound, fold_id=fold_id,
        prediction_parent_rows=cohort['score_parent_rows'],
        prediction_decision_ns=cohort['score_decision_ns'],
        probabilities=trained['probabilities'],
        prior_record=priors[fold_id])
    _time_guard(deadline)
    weights = {}
    for key in (tf + ':' + head for tf in features.TIMEFRAMES
                for head in ('activity', 'conditional_side')):
        _need(key in trained['boosters'], 'MODEL_HEAD:' + key)
        weights[key] = _write_bytes(
            out_dir, f'MODEL_FOLD_{fold_id}_{key.replace(":", "_").upper()}.txt',
            trained['boosters'][key].model_to_string().encode('utf-8'))
    predictions = _write_predictions(out_dir, fold_id, cohort,
                                     trained['probabilities'])
    return _write_json(out_dir, f'FOLD_{fold_id}.json', {
        'schema_version': 'gx1_market_state_tabular_lightgbm_fold_v1',
        'fold_id': fold_id, 'fit_rows': cohort['joined_fit_rows'],
        'score_rows': len(cohort['score_parent_rows']),
        'fit_parent_rows_sha256': cohort['fit_parent_rows_sha256'],
        'fit_decision_ns_sha256': cohort['fit_decision_ns_sha256'],
        'score_parent_rows_sha256': cohort['score_parent_rows_sha256'],
        'score_decision_ns_sha256': cohort['score_decision_ns_sha256'],
        'fit_counts': trained['fit_counts'], 'model_weights': weights,
        'predictions': predictions, 'paired_scores': score,
        'fixed_recipe': trained['recipe'],
        'test_data_used': False})


def _wall_limit(_signum, _frame) -> None:
    raise TimeoutError('MARKET_STATE_TABULAR_OOF_INTERNAL_WALL_LIMIT')


def run(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    global PUBLISHED_BYTES
    started = time.monotonic()
    plan, plan_ref, head, policy_ref = preflight(
        plan_path, plan_sha256, out_dir, admitted=True)
    _need(policy_ref is not None, 'POLICY_REQUIRED')
    deadline = started + LIMITS['internal_wall_seconds']
    remaining = deadline - time.monotonic()
    _need(remaining > 0 and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
          'WALL_BUDGET_PRECLAIM')
    out_dir.mkdir(parents=True, exist_ok=False)
    PUBLISHED_BYTES = 0
    claim = None
    completed = []
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _wall_limit)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        _fsync_directory(out_dir.parent)
        claim = _write_json(out_dir, 'CLAIM.json', {
            'schema_version': 'gx1_market_state_tabular_lightgbm_claim_v1',
            'invocation': 1, 'plan': plan_ref, 'policy': policy_ref,
            'source_head': head, 'fold_ids': FOLD_IDS,
            'fixed_tabular_fits': HEAD_COUNT,
            'native_training': False, 'test_data_used': False})
        chain = _reference_chain()
        provider_plan = chain['provider_plan']
        provider = source._provider_module()
        info = provider.validate_metadata(provider_plan)
        prepared = source._prepare_inputs(
            provider, provider_plan, info, chain['gc_layout'])
        source._validate_axes(plan['cohorts'], prepared['axes'])
        view = _view(prepared)
        bound = target.load_bound_five_endpoint_targets()
        _need(np.array_equal(bound.parent_rows, view.parent_rows)
              and np.array_equal(bound.decision_ns, view.decision_ns),
              'BOUND_MODEL_AXES')
        prior_rows = []
        priors = {}
        for fold_id in FOLD_IDS:
            _time_guard(deadline)
            fit = mature_fold_training_targets(bound, fold_id=fold_id)
            _need(all(fit['cohort'][key] == plan['cohorts'][fold_id - 1][key]
                      for key in plan['cohorts'][fold_id - 1]),
                  'PRIOR_COHORT:' + str(fold_id))
            priors[fold_id] = fit['prior']
            prior_rows.append(fit['prior'])
        prior_ref = _write_json(out_dir, 'PRIOR_FROZEN.json', {
            'schema_version': 'gx1_market_state_tabular_prior_frozen_v1',
            'plan': plan_ref, 'claim': claim, 'folds': prior_rows,
            'before_any_model_fit': True, 'test_data_used': False})
        matrix, matrix_sha = _matrix(view, deadline)
        feature_ref = _write_json(out_dir, 'FEATURE_SCHEMA.json', {
            'schema_version': 'gx1_market_state_tabular_feature_schema_v1',
            'rows': len(view.parent_rows), 'width': view.width,
            'feature_names': view.feature_names,
            'categorical': view.categorical,
            'matrix_float32_c_order_sha256': matrix_sha,
            'model_parent_rows_sha256': prepared['expected_axes']['parent_rows_sha256'],
            'model_decision_ns_sha256': prepared['expected_axes']['decision_ns_sha256'],
            'no_future_targets_in_features': True})
        for fold_id in FOLD_IDS:
            _source_bookend(plan, plan_sha256, policy_ref['sha256'], head)
            completed.append(_fold(plan, bound, view, matrix, priors,
                                   fold_id, out_dir, deadline))
        _time_guard(deadline)
        provider.validate_metadata(provider_plan)
        _source_bookend(plan, plan_sha256, policy_ref['sha256'], head)
        _need(len(completed) == 5, 'FIVE_FOLDS_REQUIRED')
        result = _write_json(out_dir, 'RESULT.json', {
            'schema_version': 'gx1_market_state_tabular_lightgbm_result_v1',
            'status': 'COMPLETE_FIVE_FOLDS_PENDING_INDEPENDENT_WEEK_BLOCK_REVIEW',
            'plan': plan_ref, 'policy': policy_ref, 'claim': claim,
            'source_head': head, 'elapsed_seconds': time.monotonic() - started,
            'input_refs': INPUT_REFS, 'prior_frozen': prior_ref,
            'feature_schema': feature_ref, 'folds': completed,
            'fixed_tabular_fits': HEAD_COUNT,
            'score_axes': SCORE_AXES, 'decision_rule': DECISION_RULE,
            'quality_decision': 'PENDING_INDEPENDENT_WEEK_BLOCK_REVIEW',
            'native_training': False, 'test_data_used': False,
            'entry_exit_or_trading_authority': False})
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminal = _write_json(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_tabular_lightgbm_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'claim': claim, 'result': result, 'fold_count': 5,
            'fixed_tabular_fits': HEAD_COUNT,
            'test_data_used': False, 'no_relaunch_under_this_plan': True})
        return {'result': result, 'terminal': terminal}
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        failed = _write_json(out_dir, 'FAILED.json', {
            'schema_version': 'gx1_market_state_tabular_lightgbm_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'completed_folds': completed, 'error_type': type(exc).__name__,
            'error': str(exc)[:2048],
            'elapsed_seconds': time.monotonic() - started,
            'no_relaunch_under_this_plan': True})
        _write_json(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_tabular_lightgbm_terminal_v1',
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
        plan, plan_ref, head, _ = preflight(
            args.plan, args.plan_sha256, args.out_dir)
        print(json.dumps({
            'status': 'READY_NO_CLAIM_NO_PAYLOAD',
            'plan': plan_ref, 'source_head': head,
            'cohorts_checked': len(plan['cohorts']),
            'fixed_recipe': model.fixed_recipe(),
            'planned_command': planned_command(args.plan_sha256),
            'claim_created': False, 'payload_read': False,
            'test_data_used': False},
            sort_keys=True))
        return 0
    print(json.dumps(run(args.plan, args.plan_sha256, args.out_dir),
                     sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
