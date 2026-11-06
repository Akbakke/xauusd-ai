"""Descriptive +5 state-forecast/Entry-side rank join; saved predictions only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from gx1.contracts.entry_causal_m1_outcomes_v1 import query_dynamic_m1_prefix
from gx1.contracts.entry_direction_block_rank_v1 import block_rank_pairs
from gx1.contracts.entry_observed_market_v1 import (
    build_entry_observed_market_targets, read_dynamic_m5_target_source,
    require_entry_observed_market_scope,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.scripts.report_dynamic_m5_october_age_pair_v1 import _arm

SCHEMA = 'gx1_dynamic_m5_october_entry_direction_join_v1'
POLICY_KEY = 'dynamic_m5_october_entry_direction_join_20261106_001'
REPO = Path('/home/andre2/src/GX1_CURRENT')
DAY_NS = 86_400_000_000_000
MINUTE_NS = 60_000_000_000
DRAWS = 10_000
SEED = 20260911


def require(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError('OCTOBER_ENTRY_DIRECTION_JOIN_' + code)


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
            and sha(path) == ref['sha256'], 'REF_BYTES')
    value = json.loads(path.read_text())
    require(type(value) is dict, 'REF_OBJECT')
    return value


def direction_score(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    require(p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all()
            and np.all((p >= 0) & (p <= 1))
            and np.all(np.abs(p.sum(axis=1) - 1) <= 3 * np.finfo(np.float32).eps),
            'ISSUED_PROBABILITIES')
    denom = p[:, 0] + p[:, 2]
    require(np.all(denom > 0), 'DIRECTION_DENOMINATOR')
    return p[:, 2] / denom


def paired_auc(y: np.ndarray, days: np.ndarray, scores: dict[str, np.ndarray]) -> dict:
    require(y.dtype == np.int8 and y.ndim == 1 and len(y) > 0
            and np.unique(y).tolist() == [0, 1]
            and all(z.dtype == np.float64 and z.shape == y.shape and np.isfinite(z).all()
                    for z in scores.values()), 'AUC_DOMAIN')
    unique, block, count = np.unique(days, return_inverse=True, return_counts=True)
    require(len(unique) >= 2, 'AUC_DAYS')
    rng = np.random.Generator(np.random.PCG64(SEED))
    choices = rng.integers(len(unique), size=(DRAWS, len(unique)))
    weight = np.zeros((DRAWS, len(unique)), np.int16)
    np.add.at(weight, (np.arange(DRAWS)[:, None], choices), 1)
    points, draws = {}, {'row': {}, 'equal_day': {}}
    for name, score in scores.items():
        pairs = block_rank_pairs(y, score, block.astype(np.int64))
        points[name] = {'row': float(roc_auc_score(y, score))}
        unit = 1.0 / count
        denominator = (unit @ pairs['positive']) * (unit @ pairs['negative'])
        points[name]['equal_day'] = float((unit @ pairs['twice_wins'] @ unit) / (2 * denominator))
        for mode, w in (('row', weight), ('equal_day', weight / count)):
            denominator = (w @ pairs['positive']) * (w @ pairs['negative'])
            numerator = ((w @ pairs['twice_wins']) * w).sum(axis=1)
            draws[mode][name] = np.divide(numerator, 2 * denominator,
                out=np.full(DRAWS, np.nan), where=denominator > 0)
    intervals = {}
    for comparison in ('candidate_minus_control', 'candidate_minus_half', 'control_minus_half'):
        intervals[comparison] = {}
        for mode in ('row', 'equal_day'):
            a = draws[mode]['candidate'] if comparison.startswith('candidate') else draws[mode]['control']
            b = draws[mode]['control'] if comparison.endswith('control') else 0.5
            delta = a - b
            intervals[comparison][mode] = ([float(v) for v in np.quantile(delta, [.025, .975], method='linear')]
                                           if np.isfinite(delta).all() else None)
    return {'outcome_utc_days': len(unique), 'auc': points,
            'candidate_minus_control_and_half_utc_day_intervals': intervals,
            'bootstrap': {'unit': 'whole_UTC_outcome_day', 'draws': DRAWS,
                          'seed': SEED, 'interval': 'central_two_sided_percentile_95'}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    require(args.plan.is_absolute() and args.plan.is_file()
            and sha(args.plan) == args.plan_sha256, 'PLAN_BYTES')
    plan = json.loads(args.plan.read_text())
    require(plan['schema_version'] == SCHEMA + '_plan'
            and plan['status'] == 'DESCRIPTIVE_RULE_BOUND_BEFORE_THIS_JOIN_AFTER_PRIOR_DEVELOPMENT_EXPOSURE'
            and plan['script_sha256'] == sha(Path(__file__))
            and plan['parent_range'] == [29849, 36172]
            and plan['lead_minutes'] == 5
            and plan['october_cutoff_ns_exclusive'] == 1730419200000000000
            and plan['expected_fold_mature'] == 6322
            and plan['expected_executable_terminal'] == 6297
            and plan['max_wall_seconds'] == 3600
            and plan['model_calls_authorized'] == 0
            and plan['decision_authority'] == 'NONE'
            and plan['no_action_selection'] is True
            and plan['test_data_used'] is False, 'PLAN_SCOPE')
    output = Path(plan['output_path'])
    claim_path = Path(plan['claim_path'])
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
            and stage['model_calls_authorized'] == 0
            and stage['test_data_used'] is False, 'POLICY_ADMISSION')
    mask_report = bound_json(plan['entry_mask_report'])
    mask_plan = bound_json(mask_report['plan'])
    require(mask_report['status'] == 'COMPLETE_READ_ONLY_OCTOBER_MASK_CENSUS'
            and mask_report['eligible_by_october_cutoff'][4] == 6322
            and mask_report['observed_by_side']['terminal_open_observed'][4] == [6297, 6297]
            and mask_plan['input_recipe'] == plan['input_recipe']
            and mask_plan['physical_target_receipt'] == plan['physical_target_receipt']
            and mask_plan['october_axis'] == plan['october_axis']
            and mask_plan['parent_range'] == plan['parent_range']
            and mask_plan['october_cutoff_ns_exclusive'] == plan['october_cutoff_ns_exclusive'],
            'MASK_PRECONDITION')
    claim = {'schema_version': SCHEMA + '_claim', 'status': 'CLAIMED_ONCE_BEFORE_ENTRY_OUTCOMES',
             'created_utc': datetime.now(timezone.utc).isoformat(),
             'plan': {'path': str(args.plan), 'sha256': args.plan_sha256},
             'source_script_sha256': plan['script_sha256'],
             'model_calls': 0, 'test_data_used': False}
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    _publish_bytes(claim_path, (json.dumps(claim, sort_keys=True, indent=2) + '\n').encode())
    claim_ref = {'path': str(claim_path), 'sha256': sha(claim_path)}
    control, candidate = (_arm(plan[name]) for name in ('control_review', 'candidate_review'))
    for key in ('candidate_plan', 'observer_axes', 'target_manifest', 'target_sources',
                'prefix_normalization', 'prefix_prior', 'reference_plan', 'fold'):
        require(control['recipe'][key] == candidate['recipe'][key], 'MATCHED_RECIPE:' + key)
    require(control['review']['arm'] == 'control' and candidate['review']['arm'] == 'candidate'
            and np.array_equal(control['arrays']['parent_rows'], candidate['arrays']['parent_rows'])
            and np.array_equal(control['arrays']['issue_ns'], candidate['arrays']['issue_ns']),
            'MATCHED_SAVED_SCORE')
    parents = np.asarray(control['arrays']['parent_rows'], np.int64).copy()
    issue = np.asarray(control['arrays']['issue_ns'], np.int64).copy()
    scores = {name: direction_score(np.asarray(arm['arrays']['probabilities'][:, 0, 4, :]).copy())
              for name, arm in (('control', control), ('candidate', candidate))}
    del control, candidate
    axis = bound_json(plan['october_axis'])
    require(np.array_equal(parents, np.arange(*plan['parent_range'], dtype=np.int64))
            and hashlib.sha256(parents.astype('<i8', copy=False).tobytes()).hexdigest()
                == axis['evaluation']['parent_rows_sha256']
            and hashlib.sha256(issue.astype('<i8', copy=False).tobytes()).hexdigest()
                == axis['evaluation']['decision_ns_sha256'], 'SAVED_OCTOBER_AXIS')
    receipt = bound_json(plan['physical_target_receipt'])['splits']['train']
    recipe = bound_json(plan['input_recipe'])
    require(recipe['files']['train_cost_authority'] == plan['cost_authority'], 'DECLARED_COST_BINDING')
    scope = require_entry_observed_market_scope(recipe)
    require(scope['physical_sources']['train']['parquet'] == receipt['source_parquet']
            and scope['physical_sources']['train']['manifest'] == receipt['source_manifest']
            and scope['train_control_cutoff'] == '2026-01-01T00:00:00+00:00'
            and scope['economics']['historical_cost_truth_qualified'] is False,
            'ENTRY_SOURCE_AND_COST')
    source = read_dynamic_m5_target_source(scope, split='train')
    bars = pd.DatetimeIndex(pd.to_datetime(source['frame']['time'], utc=True)).as_unit('ns').asi8
    require(len(bars) == 118249
            and hashlib.sha256(bars.astype('<i8', copy=False).tobytes()).hexdigest()
                == axis['full_axes']['bar_start_ns_sha256']
            and np.array_equal(bars[parents] + 5 * MINUTE_NS, issue),
            'ACTUAL_ENTRY_CLOCK')
    wanted = issue + 5 * MINUTE_NS
    fold_mature = wanted < plan['october_cutoff_ns_exclusive']
    require(int(fold_mature.sum()) == 6322, 'FOLD_CUTOFF')
    rows = parents[fold_mature]
    horizons = np.full(len(rows), 5, dtype=np.int64)
    observations = query_dynamic_m1_prefix(source['index'], fill_surface=source['fill'],
        query_entry_rows=rows, horizon_minutes=horizons,
        endpoint_kind='terminal_open_inclusive')
    require(np.array_equal(observations['query_entry_rows'], rows)
            and np.array_equal(observations['horizon_minutes'], horizons)
            and np.array_equal(observations['terminal_time_ns'], wanted[fold_mature])
            and np.all(observations['terminal_time_ns'] < plan['october_cutoff_ns_exclusive']),
            'ENDPOINT_QUERY_CLOCK')
    observed = observations['terminal_gross_observed']
    require(observed.shape == (len(rows), 2) and observed.dtype == np.bool_
            and np.array_equal(observed[:, 0], observed[:, 1])
            and int(observed[:, 0].sum()) == 6297, 'EXECUTABLE_MASK')
    net = build_entry_observed_market_targets(
        gross_return_bps=torch.tensor(observations['terminal_gross_return_bps'][observed[:, 0]], dtype=torch.float64),
        elapsed_wall_clock_seconds=torch.tensor(observations['elapsed_wall_seconds'][observed[:, 0]], dtype=torch.int64),
        economics=scope['economics'])['target_bps'].numpy()
    require(net.shape == (6297, 3) and np.isfinite(net).all() and np.all(net[:, 2] == 0),
            'NET_MARKOUT_DOMAIN')
    contrast = net[:, 0] - net[:, 1]
    ties = contrast == 0
    require(int((~ties).sum()) > 0, 'NON_TIE_OUTCOMES')
    y = (contrast[~ties] > 0).astype(np.int8)
    days = observations['terminal_time_ns'][observed[:, 0]][~ties] // DAY_NS
    joined_scores = {name: value[fold_mature][observed[:, 0]][~ties]
                     for name, value in scores.items()}
    auc = paired_auc(y, days, joined_scores)
    require(time.monotonic() - started < plan['max_wall_seconds'], 'WALL_LIMIT')
    result = {'schema_version': SCHEMA, 'status': 'DESCRIPTIVE_DEVELOPMENT_ONLY_NO_ENTRY_GATE',
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'plan': {'path': str(args.plan), 'sha256': args.plan_sha256},
        'claim': claim_ref,
        'parent_rows': 6323, 'fold_mature': int(fold_mature.sum()),
        'executable_terminal_observed': int(observed[:, 0].sum()),
        'exact_long_short_markout_ties': int(ties.sum()),
        'direction_rows': len(y), 'long_net_better': int(y.sum()),
        'short_net_better': int((y == 0).sum()),
        'score_semantics': 'Issued M5 +5 P(UP)/(P(DOWN)+P(UP)) ranks relative executable LONG-versus-SHORT net markout; it is not calibrated probability of an Entry outcome.',
        'cost_authority': plan['cost_authority'],
        'declared_economics_sha256': scope['economics']['economics_sha256'],
        'historical_cost_truth_qualified': False,
        'source_read_scope': 'Existing owner reads full pre-2026 TRAIN physical target columns and quote index during source validation; only October parent rows and +5 terminal times before 2024-11-01 enter this join. No TEST source is read.',
        'rank': auc, 'no_action_selection': True,
        'pre_mfe_mae_measured': False, 'flat_action_measured': False,
        'independent_edge': 'UNKNOWN', 'economic_edge': 'UNKNOWN',
        'model_calls': 0, 'optimizer_updates': 0, 'test_data_used': False,
        'elapsed_seconds': time.monotonic() - started,
        'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    out = output
    require(not out.exists() and sha(claim_path) == claim_ref['sha256'], 'OUTPUT_NO_REPLACE')
    _publish_bytes(out, (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps({'status': result['status'], 'output_path': str(out),
                      'output_sha256': sha(out), 'rank': auc['auc'],
                      'candidate_minus_control_row_95ci': auc['candidate_minus_control_and_half_utc_day_intervals']['candidate_minus_control']['row']},
                     sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
