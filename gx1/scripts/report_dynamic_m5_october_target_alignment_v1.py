"""Descriptive October +5 state-label versus executable Entry-side alignment.

No fitting, model scoring, action selection, TEST read, or edge gate. The two
labels have different price and time semantics; this report only measures their
association on the same observed development rows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import torch

from gx1.contracts.entry_causal_m1_outcomes_v1 import query_dynamic_m1_prefix
from gx1.contracts.entry_dynamic_mtf_state_targets_v1 import (
    TIMEFRAMES, load_dynamic_mtf_state_series, query_dynamic_mtf_states,
)
from gx1.contracts.entry_observed_market_v1 import (
    build_entry_observed_market_targets, read_dynamic_m5_target_source,
    require_entry_observed_market_scope,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

SCHEMA = 'gx1_dynamic_m5_october_target_alignment_v1'
POLICY_KEY = 'dynamic_m5_october_target_alignment_20261106_001'
REPO = Path('/home/andre2/src/GX1_CURRENT')
MINUTE_NS = 60_000_000_000
DAY_NS = 86_400_000_000_000
DRAWS = 10_000
SEED = 20260911
CLASS_NAMES = ('DOWN', 'FLAT', 'UP')
SIDE_NAMES = ('SHORT', 'LONG')


def require(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError('OCTOBER_TARGET_ALIGNMENT_' + code)


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


def joint_counts(state: np.ndarray, state_observed: np.ndarray,
                 entry_side: np.ndarray) -> dict:
    """Count the declared three states against relative LONG/SHORT outcomes."""
    require(state.dtype == np.int8 and state.ndim == 1
            and state_observed.dtype == np.bool_ and state_observed.shape == state.shape
            and entry_side.dtype == np.int8 and entry_side.shape == state.shape
            and np.isin(state[state_observed], (0, 1, 2)).all()
            and np.isin(entry_side, (-1, 0, 1)).all(), 'JOINT_DOMAIN')
    entry_known = entry_side >= 0
    common = state_observed & entry_known
    table = {
        name: {side: int(np.count_nonzero(common & (state == i) & (entry_side == j)))
               for j, side in enumerate(SIDE_NAMES)}
        for i, name in enumerate(CLASS_NAMES)
    }
    require(sum(sum(row.values()) for row in table.values()) == int(common.sum()), 'JOINT_SUM')
    return {
        'table': table,
        'common_observed': int(common.sum()),
        'state_unknown_entry_known': int(np.count_nonzero(~state_observed & entry_known)),
        'entry_unknown_state_known': int(np.count_nonzero(state_observed & ~entry_known)),
        'both_unknown': int(np.count_nonzero(~state_observed & ~entry_known)),
        'entry_known': int(entry_known.sum()),
        'state_known': int(state_observed.sum()),
    }


def day_bootstrap(state: np.ndarray, entry_side: np.ndarray, days: np.ndarray) -> dict:
    """Fixed whole UTC outcome-day resampling of P(LONG|UP)-P(LONG|DOWN)."""
    require(state.dtype == np.int8 and entry_side.dtype == np.int8
            and days.dtype == np.int64 and state.ndim == entry_side.ndim == days.ndim == 1
            and len(state) == len(entry_side) == len(days) > 0
            and np.isin(state, (0, 1, 2)).all() and np.isin(entry_side, (0, 1)).all(),
            'BOOT_DOMAIN')
    unique, block = np.unique(days, return_inverse=True)
    require(len(unique) >= 2, 'BOOT_DAYS')
    totals, long_counts = {}, {}
    for code, name in ((0, 'DOWN'), (2, 'UP')):
        rows = state == code
        totals[name] = np.bincount(block[rows], minlength=len(unique)).astype(np.int64)
        long_counts[name] = np.bincount(block[rows & (entry_side == 1)],
                                        minlength=len(unique)).astype(np.int64)
        require(int(totals[name].sum()) > 0, 'BOOT_CLASS_' + name)
    rate = {name: float(long_counts[name].sum() / totals[name].sum())
            for name in ('DOWN', 'UP')}
    rng = np.random.Generator(np.random.PCG64(SEED))
    choices = rng.integers(len(unique), size=(DRAWS, len(unique)))
    weight = np.zeros((DRAWS, len(unique)), np.int16)
    np.add.at(weight, (np.arange(DRAWS)[:, None], choices), 1)
    up_den, down_den = weight @ totals['UP'], weight @ totals['DOWN']
    valid = (up_den > 0) & (down_den > 0)
    delta = np.full(DRAWS, np.nan, np.float64)
    delta[valid] = ((weight[valid] @ long_counts['UP']) / up_den[valid]
                    - (weight[valid] @ long_counts['DOWN']) / down_den[valid])
    interval = ([float(v) for v in np.quantile(delta[valid], [.025, .975], method='linear')]
                if valid.any() else None)
    return {
        'long_better_rate_by_state': rate,
        'up_minus_down_long_better_rate': rate['UP'] - rate['DOWN'],
        'up_minus_down_utc_day_95_interval': interval,
        'outcome_utc_days': len(unique),
        'bootstrap': {'unit': 'whole_UTC_outcome_day', 'draws': DRAWS, 'seed': SEED,
                      'valid_draws': int(valid.sum()), 'invalid_draws': int((~valid).sum()),
                      'interval': 'central_two_sided_percentile_95'},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    require(args.plan.is_absolute() and args.plan.is_file()
            and not args.plan.is_symlink() and sha(args.plan) == args.plan_sha256,
            'PLAN_BYTES')
    plan = json.loads(args.plan.read_text())
    require(plan['schema_version'] == SCHEMA + '_plan'
            and plan['status'] == 'DESCRIPTIVE_RULE_BOUND_BEFORE_THIS_ALIGNMENT_AFTER_PRIOR_OCTOBER_EXPOSURE'
            and plan['script_sha256'] == sha(Path(__file__))
            and plan['parent_range'] == [29849, 36172]
            and plan['october_cutoff_ns_exclusive'] == 1730419200000000000
            and plan['lead_minutes'] == 5
            and plan['expected_state_observed'] == 3359
            and plan['expected_state_classes'] == {'DOWN': 1334, 'FLAT': 490, 'UP': 1535}
            and plan['expected_entry_fold_mature'] == 6322
            and plan['expected_entry_executable'] == 6297
            and plan['expected_entry_long_better'] == 3225
            and plan['expected_entry_short_better'] == 3072
            and plan['expected_entry_ties'] == 0
            and plan['bootstrap_draws'] == DRAWS and plan['bootstrap_seed'] == SEED
            and plan['capped_class'] == 'audit'
            and plan['memory_bytes'] == 4294967296
            and plan['swap_bytes'] == 536870912
            and plan['max_wall_seconds'] == 3600
            and plan['usd_budget'] == 0
            and plan['model_calls_authorized'] == 0
            and plan['no_action_selection'] is True
            and plan['decision_authority'] == 'NONE'
            and plan['test_data_used'] is False
            and plan['broker_access_allowed'] is False
            and plan['trading_allowed'] is False, 'PLAN_SCOPE')
    for name in ('state_report', 'entry_report', 'entry_mask_report', 'state_manifest',
                 'input_recipe', 'physical_target_receipt', 'october_axis', 'cost_authority'):
        require(type(plan[name]) is dict and set(plan[name]) == {'path', 'sha256'}
                and Path(plan[name]['path']).is_absolute(), 'PLAN_REF_' + name)
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
            and stage['training_enabled'] is False
            and stage['test_data_used'] is False
            and stage['decision_authority'] == 'NONE', 'POLICY_ADMISSION')
    # The claim precedes all saved target reports, state arrays, and Entry outcomes.
    claim = {'schema_version': SCHEMA + '_claim', 'status': 'CLAIMED_ONCE_BEFORE_TARGET_OUTCOMES',
             'created_utc': datetime.now(timezone.utc).isoformat(),
             'plan': {'path': str(args.plan), 'sha256': args.plan_sha256},
             'source_script_sha256': plan['script_sha256'],
             'model_calls': 0, 'test_data_used': False}
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    _publish_bytes(claim_path, (json.dumps(claim, sort_keys=True, indent=2) + '\n').encode())
    claim_ref = {'path': str(claim_path), 'sha256': sha(claim_path)}

    state_report = bound_json(plan['state_report'])
    entry_report = bound_json(plan['entry_report'])
    mask_report = bound_json(plan['entry_mask_report'])
    state_manifest = bound_json(plan['state_manifest'])
    cost_policy = bound_json(plan['cost_authority'])
    axis = bound_json(plan['october_axis'])
    entry_plan = bound_json(entry_report['plan'])
    mask_plan = bound_json(mask_report['plan'])
    require(state_report['status'] == 'DESCRIPTIVE_DEVELOPMENT_ONLY_NO_PASS_GATE'
            and state_report['test_data_used'] is False
            and state_report['target_manifest'] == plan['state_manifest']
            and state_report['score_cutoff_ns_exclusive'] == plan['october_cutoff_ns_exclusive']
            and state_report['timeframes']['M5']['native_lead_minutes'] == plan['lead_minutes']
            and state_report['timeframes']['M5']['observed'] == plan['expected_state_observed']
            and state_report['timeframes']['M5']['class_counts'] == plan['expected_state_classes'],
            'STATE_REPORT_MARGIN')
    require(entry_report['status'] == 'DESCRIPTIVE_DEVELOPMENT_ONLY_NO_ENTRY_GATE'
            and entry_report['test_data_used'] is False
            and entry_report['model_calls'] == 0
            and entry_report['parent_rows'] == plan['parent_range'][1] - plan['parent_range'][0]
            and entry_report['fold_mature'] == plan['expected_entry_fold_mature']
            and entry_report['executable_terminal_observed'] == plan['expected_entry_executable']
            and entry_report['long_net_better'] == plan['expected_entry_long_better']
            and entry_report['short_net_better'] == plan['expected_entry_short_better']
            and entry_report['exact_long_short_markout_ties'] == plan['expected_entry_ties']
            and entry_report['cost_authority'] == plan['cost_authority']
            and entry_plan['input_recipe'] == plan['input_recipe']
            and entry_plan['physical_target_receipt'] == plan['physical_target_receipt']
            and entry_plan['october_axis'] == plan['october_axis']
            and entry_plan['entry_mask_report'] == plan['entry_mask_report']
            and entry_plan['parent_range'] == plan['parent_range']
            and entry_plan['october_cutoff_ns_exclusive'] == plan['october_cutoff_ns_exclusive'],
            'ENTRY_REPORT_BINDING')
    require(mask_report['status'] == 'COMPLETE_READ_ONLY_OCTOBER_MASK_CENSUS'
            and mask_report['eligible_by_october_cutoff'][4] == plan['expected_entry_fold_mature']
            and mask_report['observed_by_side']['terminal_open_observed'][4]
                == [plan['expected_entry_executable']] * 2
            and mask_plan['input_recipe'] == plan['input_recipe']
            and mask_plan['physical_target_receipt'] == plan['physical_target_receipt']
            and mask_plan['october_axis'] == plan['october_axis'], 'ENTRY_MASK_BINDING')
    require(state_manifest['metadata']['cutoff_ns'] == plan['october_cutoff_ns_exclusive']
            and state_manifest['metadata']['parents'] == plan['parent_range'][1]
            and state_manifest['metadata']['test_data_used'] is False,
            'STATE_MANIFEST_SCOPE')
    parents = np.arange(*plan['parent_range'], dtype=np.int64)
    source_refs = state_manifest['metadata']['source_refs']
    series = load_dynamic_mtf_state_series(plan['state_manifest'],
        expected_source_refs=source_refs,
        expected_cutoff_ns=plan['october_cutoff_ns_exclusive'])
    stored_parents = series.arrays['parent_rows']
    positions = np.searchsorted(stored_parents, parents)
    require(np.all(positions < len(stored_parents))
            and np.array_equal(stored_parents[positions], parents), 'STATE_PARENT_AXIS')
    issue = np.asarray(series.arrays['issue_ns'][positions], np.int64).copy()
    require(np.all(np.diff(issue) > 0)
            and hashlib.sha256(parents.astype('<i8', copy=False).tobytes()).hexdigest()
                == axis['evaluation']['parent_rows_sha256']
            and hashlib.sha256(issue.astype('<i8', copy=False).tobytes()).hexdigest()
                == axis['evaluation']['decision_ns_sha256'], 'STATE_OCTOBER_AXIS')
    state_query = query_dynamic_mtf_states(series, parent_rows=parents, issue_ns=issue,
        cutoff_ns=plan['october_cutoff_ns_exclusive'], lead_minutes=np.array([5], np.int64))
    tf = TIMEFRAMES.index('M5')
    state = state_query['state'][:, tf, 0].copy()
    state_observed = state_query['observed'][:, tf, 0].copy()
    state_reason = state_query['reason'][:, tf, 0].copy()
    wanted = state_query['validity_ns'][:, 0].copy()
    classes = {name: int(np.count_nonzero(state_observed & (state == code)))
               for code, name in enumerate(CLASS_NAMES)}
    require(np.array_equal(wanted, issue + plan['lead_minutes'] * MINUTE_NS)
            and int(state_observed.sum()) == plan['expected_state_observed']
            and classes == plan['expected_state_classes']
            and np.all(state[~state_observed] == -1), 'STATE_QUERY_MARGIN')
    del series, state_query
    gc.collect()

    recipe = bound_json(plan['input_recipe'])
    receipt = bound_json(plan['physical_target_receipt'])['splits']['train']
    require(recipe['files']['train_cost_authority'] == plan['cost_authority']
            and cost_policy['historical_cost_truth_qualified'] is False
            and cost_policy['test_data_used'] is False, 'DECLARED_COST_BINDING')
    scope = require_entry_observed_market_scope(recipe)
    require(scope['physical_sources']['train']['parquet'] == receipt['source_parquet']
            and scope['physical_sources']['train']['manifest'] == receipt['source_manifest']
            and scope['train_control_cutoff'] == '2026-01-01T00:00:00+00:00'
            and scope['economics']['historical_cost_truth_qualified'] is False
            and scope['economics']['cost_authority_sha256'] == cost_policy['authority_sha256'],
            'ENTRY_SOURCE_AND_COST')
    source = read_dynamic_m5_target_source(scope, split='train')
    bars = pd.DatetimeIndex(pd.to_datetime(source['frame']['time'], utc=True)).as_unit('ns').asi8
    require(len(bars) == 118249
            and hashlib.sha256(bars.astype('<i8', copy=False).tobytes()).hexdigest()
                == axis['full_axes']['bar_start_ns_sha256']
            and np.array_equal(bars[parents] + 5 * MINUTE_NS, issue),
            'ACTUAL_ENTRY_CLOCK')
    mature = wanted < plan['october_cutoff_ns_exclusive']
    require(int(mature.sum()) == plan['expected_entry_fold_mature'], 'FOLD_CUTOFF')
    observations = query_dynamic_m1_prefix(source['index'], fill_surface=source['fill'],
        query_entry_rows=parents[mature], horizon_minutes=np.full(int(mature.sum()), 5, np.int64),
        endpoint_kind='terminal_open_inclusive')
    require(np.array_equal(observations['query_entry_rows'], parents[mature])
            and np.array_equal(observations['terminal_time_ns'], wanted[mature])
            and np.all(observations['terminal_time_ns'] < plan['october_cutoff_ns_exclusive']),
            'ENDPOINT_CLOCK')
    terminal_observed = observations['terminal_gross_observed']
    require(terminal_observed.shape == (int(mature.sum()), 2)
            and terminal_observed.dtype == np.bool_
            and np.array_equal(terminal_observed[:, 0], terminal_observed[:, 1])
            and int(terminal_observed[:, 0].sum()) == plan['expected_entry_executable'],
            'ENTRY_OBSERVED_MARGIN')
    net = build_entry_observed_market_targets(
        gross_return_bps=torch.tensor(
            observations['terminal_gross_return_bps'][terminal_observed[:, 0]], dtype=torch.float64),
        elapsed_wall_clock_seconds=torch.tensor(
            observations['elapsed_wall_seconds'][terminal_observed[:, 0]], dtype=torch.int64),
        economics=scope['economics'])['target_bps'].numpy()
    require(net.shape == (plan['expected_entry_executable'], 3)
            and np.isfinite(net).all() and np.all(net[:, 2] == 0), 'NET_MARKOUT_DOMAIN')
    contrast = net[:, 0] - net[:, 1]
    ties = contrast == 0
    require(int(ties.sum()) == plan['expected_entry_ties'], 'ENTRY_TIES')
    side = np.full(len(parents), -1, np.int8)
    eligible_positions = np.flatnonzero(mature)
    observed_positions = eligible_positions[terminal_observed[:, 0]]
    side[observed_positions[~ties]] = (contrast[~ties] > 0).astype(np.int8)
    require(int(np.count_nonzero(side == 1)) == plan['expected_entry_long_better']
            and int(np.count_nonzero(side == 0)) == plan['expected_entry_short_better'],
            'ENTRY_SIDE_MARGIN')
    counts = joint_counts(state, state_observed, side)
    common = state_observed & (side >= 0)
    effect = day_bootstrap(state[common], side[common], wanted[common] // DAY_NS)
    unknown_reasons = {str(int(code)): int(count) for code, count in
        zip(*np.unique(state_reason[~state_observed], return_counts=True))}
    require(time.monotonic() - started < plan['max_wall_seconds'], 'WALL_LIMIT')
    result = {
        'schema_version': SCHEMA, 'status': 'DESCRIPTIVE_TARGET_ALIGNMENT_ONLY_NO_ENTRY_GATE',
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'plan': {'path': str(args.plan), 'sha256': args.plan_sha256}, 'claim': claim_ref,
        'population': {'parent_rows': len(parents), 'fold_mature': int(mature.sum()),
                       'state_observed': int(state_observed.sum()),
                       'state_unknown': int((~state_observed).sum()),
                       'state_unknown_reason_counts': unknown_reasons,
                       'entry_executable_non_tie': int(np.count_nonzero(side >= 0)),
                       'entry_unknown_or_censored': int(np.count_nonzero(side < 0)),
                       'entry_exact_ties': int(ties.sum())},
        'joint': counts, 'association': effect,
        'state_semantics': 'Local M5 state at issue+5 from closed M1 midpoint OHLC and causal 0.5 ATR; independent of issue price.',
        'entry_semantics': 'Relative declared-net executable LONG versus SHORT markout from issue to exact terminal-open BID/ASK at issue+5.',
        'source_read_scope': 'State series is saved October TRAIN target. Existing Entry owner opens full pre-2026 TRAIN physical target columns and quote index; only October parent rows and +5 terminal before 2024-11-01 enter the query. No TEST source is read.',
        'cost_authority': plan['cost_authority'],
        'declared_economics_sha256': scope['economics']['economics_sha256'],
        'historical_cost_truth_qualified': False,
        'prebound_after_separate_october_outcomes_inspected': True,
        'no_action_selection': True, 'pre_mfe_mae_measured': False,
        'independent_edge': 'UNKNOWN', 'economic_edge': 'UNKNOWN',
        'model_calls': 0, 'optimizer_updates': 0, 'test_data_used': False,
        'elapsed_seconds': time.monotonic() - started,
        'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    require(not output.exists() and sha(claim_path) == claim_ref['sha256'], 'OUTPUT_NO_REPLACE')
    _publish_bytes(output, (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps({'status': result['status'], 'output_path': str(output),
                      'output_sha256': sha(output), 'joint': counts['table'],
                      'up_minus_down_long_better_rate': effect['up_minus_down_long_better_rate'],
                      'up_minus_down_utc_day_95_interval': effect['up_minus_down_utc_day_95_interval']},
                     sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
