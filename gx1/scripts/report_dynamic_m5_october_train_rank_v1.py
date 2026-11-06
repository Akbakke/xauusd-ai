"""One descriptive in-sample M5 +5 rank audit of saved candidate TRAIN scores."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from gx1.contracts.entry_direction_block_rank_v1 import block_rank_pairs
from gx1.contracts.entry_dynamic_mtf_state_targets_v1 import (
    TIMEFRAMES, load_dynamic_mtf_state_series, query_dynamic_mtf_states,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.scripts.report_dynamic_m5_october_age_pair_v1 import _array
from gx1.scripts.report_dynamic_m5_october_entry_direction_join_v1 import direction_score

SCHEMA = 'gx1_dynamic_m5_october_train_rank_v1'
POLICY_KEY = 'dynamic_m5_october_train_rank_20261106_001'
REPO = Path('/home/andre2/src/GX1_CURRENT')
DAY_NS = 86_400_000_000_000
DRAWS = 10_000
SEED = 20260911


def require(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError('OCTOBER_TRAIN_RANK_' + code)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bound_json(ref: dict) -> dict:
    require(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'REF_SCHEMA')
    path = Path(ref['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and 'TEST' not in path.parts and sha(path) == ref['sha256'], 'REF_BYTES')
    value = json.loads(path.read_text())
    require(type(value) is dict, 'REF_OBJECT')
    return value


def auc_with_day_interval(y: np.ndarray, score: np.ndarray, days: np.ndarray) -> dict:
    """One row-weighted AUC and fixed whole UTC-day bootstrap against chance."""
    require(y.dtype == np.int8 and y.ndim == 1 and score.dtype == np.float64
            and days.dtype == np.int64 and score.shape == days.shape == y.shape
            and len(y) > 0 and np.unique(y).tolist() == [0, 1]
            and np.isfinite(score).all(), 'AUC_DOMAIN')
    unique, block = np.unique(days, return_inverse=True)
    require(len(unique) >= 2, 'AUC_DAYS')
    pairs = block_rank_pairs(y, score, block.astype(np.int64))
    point = float(roc_auc_score(y, score))
    rng = np.random.Generator(np.random.PCG64(SEED))
    choice = rng.integers(len(unique), size=(DRAWS, len(unique)))
    weights = np.zeros((DRAWS, len(unique)), np.int16)
    np.add.at(weights, (np.arange(DRAWS)[:, None], choice), 1)
    denominator = (weights @ pairs['positive']) * (weights @ pairs['negative'])
    numerator = ((weights @ pairs['twice_wins']) * weights).sum(axis=1)
    valid = denominator > 0
    delta = numerator[valid] / (2 * denominator[valid]) - 0.5
    interval = ([float(v) for v in np.quantile(delta, [.025, .975], method='linear')]
                if valid.any() else None)
    return {'row_auc': point, 'minus_half_utc_day_95_interval': interval,
            'outcome_utc_days': len(unique),
            'bootstrap': {'unit': 'whole_UTC_outcome_day', 'draws': DRAWS, 'seed': SEED,
                          'valid_draws': int(valid.sum()),
                          'interval': 'central_two_sided_percentile_95'}}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    require(args.plan.is_absolute() and args.plan.is_file() and not args.plan.is_symlink()
            and sha(args.plan) == args.plan_sha256, 'PLAN_BYTES')
    plan = json.loads(args.plan.read_text())
    require(plan['schema_version'] == SCHEMA + '_plan'
            and plan['status'] == 'DESCRIPTIVE_RULE_BOUND_BEFORE_THIS_TRAIN_RANK_AFTER_FIT_AND_OCTOBER_EXPOSURE'
            and plan['script_sha256'] == sha(Path(__file__))
            and plan['parent_range'] == [0, 29849]
            and plan['train_cutoff_ns_exclusive'] == 1727740800000000000
            and plan['lead_minutes'] == 5
            and plan['expected_observed'] == 16798
            and plan['expected_brier'] == 0.6042147567186823
            and plan['expected_prior_brier'] == 0.6090713086128303
            and plan['expected_fitted_updates'] == 1866
            and plan['bootstrap_draws'] == DRAWS and plan['bootstrap_seed'] == SEED
            and plan['capped_class'] == 'audit'
            and plan['memory_bytes'] == 4294967296
            and plan['swap_bytes'] == 536870912
            and plan['max_wall_seconds'] == 3600
            and plan['usd_budget'] == 0 and plan['model_calls_authorized'] == 0
            and plan['test_data_used'] is False and plan['no_action_selection'] is True
            and plan['decision_authority'] == 'NONE'
            and plan['broker_access_allowed'] is False
            and plan['trading_allowed'] is False, 'PLAN_SCOPE')
    output, claim_path = Path(plan['output_path']), Path(plan['claim_path'])
    require(output.is_absolute() and output.is_relative_to(Path('/home/andre2/GX1_RUNS'))
            and claim_path == output.parent / 'CLAIM.json'
            and not output.exists() and not claim_path.exists(), 'NO_RELAUNCH_PATHS')
    policy = json.loads((REPO / 'NEXT_RUN_POLICY.json').read_text())
    stage = policy[POLICY_KEY]
    require(policy['training_enabled'] is False
            and stage['execution_authorized'] is True
            and stage['plan'] == {'path': str(args.plan), 'sha256': args.plan_sha256}
            and stage['claim_path'] == str(claim_path)
            and stage['claimed_invocations'] == stage['completed_invocations'] == 0
            and stage['training_enabled'] is False
            and stage['model_calls_authorized'] == 0
            and stage['test_data_used'] is False
            and stage['decision_authority'] == 'NONE', 'POLICY_ADMISSION')
    claim = {'schema_version': SCHEMA + '_claim', 'status': 'CLAIMED_ONCE_BEFORE_TRAIN_SCORE_AND_TARGET_READ',
             'created_utc': datetime.now(timezone.utc).isoformat(),
             'plan': {'path': str(args.plan), 'sha256': args.plan_sha256},
             'source_script_sha256': plan['script_sha256'], 'model_calls': 0, 'test_data_used': False}
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    _publish_bytes(claim_path, (json.dumps(claim, sort_keys=True, indent=2) + '\n').encode())
    claim_ref = {'path': str(claim_path), 'sha256': sha(claim_path)}

    fit_review = bound_json(plan['fit_review'])
    recipe = bound_json(plan['fit_recipe'])
    final = bound_json(plan['final_train_measurement'])
    manifest = bound_json(plan['predictions_manifest'])
    target_meta = bound_json(plan['target_manifest'])
    require(fit_review['arm'] == 'candidate'
            and fit_review['schema_version'] == 'gx1_october_native_fit_terminal_review_v1'
            and fit_review['status'] == 'COMPLETE_CONSUMED_OCTOBER_FIT'
            and fit_review['physical_optimizer_updates'] == plan['expected_fitted_updates']
            and fit_review['training_parent_rows'] == plan['parent_range'][1]
            and fit_review['held_out_features_read'] is False
            and fit_review['held_out_outcomes_read'] is False
            and fit_review['test_data_used'] is False
            and fit_review['recipe'] == plan['fit_recipe']
            and fit_review['final_train_measurement'] == plan['final_train_measurement'], 'FIT_BINDING')
    require(recipe['arm'] == 'candidate'
            and recipe['fold']['prefix_rows'] == plan['parent_range'][1]
            and recipe['fold']['train_cutoff_ns'] == plan['train_cutoff_ns_exclusive']
            and recipe['target_manifest'] == plan['target_manifest']
            and recipe['test_data_used'] is False, 'RECIPE_SCOPE')
    saved = final['model_metrics']['per_lead']
    require(final['schema_version'] == 'gx1_dynamic_mtf_state_measurement_v1'
            and final['role'] == 'FINAL_TRAIN'
            and final['class_names'] == ['DOWN', 'FLAT', 'UP']
            and final['timeframes'] == list(TIMEFRAMES)
            and final['lead_minutes'] == list(range(1, 1441))
            and final['predictions'] == plan['predictions_manifest']
            and final['target_manifest'] == plan['target_manifest']
            and final['target_sources'] == recipe['target_sources']
            and final['fold'] == recipe['fold']
            and final['model_metrics']['parents'] == plan['parent_range'][1]
            and saved['observed'][0][4] == plan['expected_observed']
            and saved['brier'][0][4] == plan['expected_brier']
            and saved['prior_brier'][0][4] == plan['expected_prior_brier']
            and final['test_data_used'] is False, 'FINAL_TRAIN_SCOPE')
    require(manifest['schema_version'] == 'gx1_dynamic_mtf_state_prediction_arrays_v1'
            and manifest['class_names'] == ['DOWN', 'FLAT', 'UP']
            and manifest['target_manifest'] == plan['target_manifest']
            and manifest['target_sources'] == recipe['target_sources']
            and manifest['cutoff_ns'] == plan['train_cutoff_ns_exclusive']
            and manifest['timeframes'] == list(TIMEFRAMES)
            and manifest['lead_minutes'] == list(range(1, 1441))
            and manifest['test_data_used'] is False, 'PREDICTION_BINDING')
    require(target_meta['metadata']['cutoff_ns'] == plan['train_cutoff_ns_exclusive']
            and target_meta['metadata']['parents'] == plan['parent_range'][1]
            and target_meta['metadata']['source_refs'] == recipe['target_sources']
            and target_meta['metadata']['test_data_used'] is False, 'TARGET_BINDING')
    series = load_dynamic_mtf_state_series(plan['target_manifest'],
        expected_source_refs=recipe['target_sources'],
        expected_cutoff_ns=plan['train_cutoff_ns_exclusive'])
    parents = np.arange(*plan['parent_range'], dtype=np.int64)
    require(np.array_equal(series.arrays['parent_rows'], parents), 'TARGET_PARENT_AXIS')
    issue = np.asarray(series.arrays['issue_ns'], np.int64).copy()
    query = query_dynamic_mtf_states(series, parent_rows=parents, issue_ns=issue,
        cutoff_ns=plan['train_cutoff_ns_exclusive'], lead_minutes=np.array([5], np.int64))
    tf = TIMEFRAMES.index('M5')
    truth = query['state'][:, tf, 0]
    observed = query['observed'][:, tf, 0]
    validity = query['validity_ns'][:, 0]
    require(int(observed.sum()) == plan['expected_observed']
            and np.isin(truth[observed], (0, 1, 2)).all()
            and np.all(truth[~observed] == -1)
            and np.all(validity[observed] < plan['train_cutoff_ns_exclusive']), 'TARGET_MARGIN')
    parent = Path(plan['predictions_manifest']['path']).parent
    saved_parent = _array(manifest['arrays']['parent_rows'], parent=parent)
    saved_issue = _array(manifest['arrays']['issue_ns'], parent=parent)
    require(np.array_equal(saved_parent, parents) and np.array_equal(saved_issue, issue), 'SAVED_AXIS')
    probabilities = _array(manifest['arrays']['probabilities'], parent=parent)
    require(probabilities.shape == (len(parents), 5, 1440, 3), 'PROBABILITY_SHAPE')
    p = np.asarray(probabilities[:, tf, plan['lead_minutes'] - 1, :], dtype=np.float64)
    require(p.shape == (len(parents), 3), 'PROBABILITY_SLICE')
    p_observed = p[observed]
    y = truth[observed]
    require(np.isfinite(p_observed).all() and np.all((p_observed >= 0) & (p_observed <= 1))
            and np.all(np.abs(p_observed.sum(axis=1) - 1) <= 3 * np.finfo(np.float32).eps),
            'PROBABILITY_DOMAIN')
    brier = float(np.square(p_observed - np.eye(3, dtype=np.float64)[y]).sum(axis=1).mean())
    require(abs(brier - plan['expected_brier']) < 1e-6, 'SAVED_BRIER_PARITY')
    nonflat = y != 1
    direction = (y[nonflat] == 2).astype(np.int8)
    score = direction_score(p_observed[nonflat])
    days = (validity[observed][nonflat] // DAY_NS).astype(np.int64)
    auc = auc_with_day_interval(direction, score, days)
    require(time.monotonic() - started < plan['max_wall_seconds'], 'WALL_LIMIT')
    result = {'schema_version': SCHEMA, 'status': 'DESCRIPTIVE_IN_SAMPLE_TRAIN_RANK_NO_ENTRY_GATE',
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'plan': {'path': str(args.plan), 'sha256': args.plan_sha256}, 'claim': claim_ref,
        'train_parent_rows': len(parents), 'm5_plus5_observed': int(observed.sum()),
        'm5_plus5_class_counts': {name: int(np.count_nonzero(y == code))
                                  for code, name in enumerate(('DOWN', 'FLAT', 'UP'))},
        'direction_rows': len(direction), 'm5_plus5_brier_reproduced': brier,
        'm5_plus5_prior_brier_saved': plan['expected_prior_brier'],
        'rank': auc,
        'source_scope': 'Saved final TRAIN candidate forecast and saved pre-October TRAIN target only; no model forward. Prediction file SHA streamed then M5+5 slice memory-mapped. No TEST.',
        'independent_edge': 'UNKNOWN', 'economic_edge': 'UNKNOWN',
        'model_calls': 0, 'optimizer_updates': 0, 'test_data_used': False,
        'no_action_selection': True, 'elapsed_seconds': time.monotonic() - started,
        'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    require(not output.exists() and sha(claim_path) == claim_ref['sha256'], 'OUTPUT_NO_REPLACE')
    _publish_bytes(output, (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps({'status': result['status'], 'output_path': str(output),
                      'output_sha256': sha(output), 'train_auc': auc['row_auc'],
                      'minus_half_row_95ci': auc['minus_half_utc_day_95_interval']},
                     sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
