"""One read-only October Entry-path mask census; no model, target fit or TEST."""
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

from gx1.contracts.entry_causal_m1_outcomes_v1 import query_dynamic_m1_prefix
from gx1.contracts.entry_observed_market_v1 import (
    read_dynamic_m5_target_source, require_entry_observed_market_scope,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

SCHEMA = 'gx1_dynamic_m5_october_entry_mask_coverage_v1'
POLICY_KEY = 'dynamic_m5_october_entry_mask_coverage_20261106_001'
REPO = Path('/home/andre2/src/GX1_CURRENT')
MINUTE_NS = 60_000_000_000
M5_NS = 5 * MINUTE_NS
LEADS = np.arange(1, 481, dtype=np.int64)
CHUNK_PARENTS = 32


def require(condition: bool, code: str) -> None:
    if not condition:
        raise RuntimeError('OCTOBER_ENTRY_MASK_' + code)


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


def add_reasons(destination: dict[str, np.ndarray], reasons: np.ndarray,
                horizons: np.ndarray) -> None:
    codes, inverse = np.unique(reasons, return_inverse=True)
    for i, code in enumerate(codes):
        key = str(code)
        if key not in destination:
            destination[key] = np.zeros(len(LEADS), np.int64)
        destination[key] += np.bincount(horizons[inverse == i] - 1, minlength=len(LEADS))


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
            and plan['status'] == 'ONE_READ_ONLY_OCTOBER_CENSUS'
            and plan['script_sha256'] == sha(Path(__file__))
            and plan['parent_range'] == [29849, 36172]
            and plan['october_cutoff_ns_exclusive'] == 1730419200000000000
            and plan['lead_minutes_inclusive_range'] == [1, 480]
            and plan['chunk_parents'] == CHUNK_PARENTS
            and plan['max_wall_seconds'] == 3600
            and plan['test_data_used'] is False
            and plan['model_calls_authorized'] == 0, 'PLAN_SCOPE')
    policy = json.loads((REPO / 'NEXT_RUN_POLICY.json').read_text())
    stage = policy[POLICY_KEY]
    require(policy['training_enabled'] is False
            and stage['execution_authorized'] is True
            and stage['plan'] == {'path': str(args.plan), 'sha256': args.plan_sha256}
            and stage['claimed_invocations'] == stage['completed_invocations'] == 0
            and stage['model_calls_authorized'] == 0
            and stage['test_data_used'] is False, 'POLICY_ADMISSION')
    receipt = bound_json(plan['physical_target_receipt'])['splits']['train']
    axis = bound_json(plan['october_axis'])
    recipe = bound_json(plan['input_recipe'])
    require(receipt['rows'] == axis['full_axes']['rows'] == 118249
            and receipt['physical_clock_sha256'] == axis['full_axes']['bar_start_ns_sha256']
            and axis['evaluation']['parent_row_range'] == plan['parent_range']
            and axis['evaluation']['rows'] == 6323
            and receipt['dynamic_m5_supervision']['dense_future_paths_materialized'] is False,
            'METADATA_AXIS')
    scope = require_entry_observed_market_scope(recipe)
    require(scope['physical_sources']['train']['parquet'] == receipt['source_parquet']
            and scope['physical_sources']['train']['manifest'] == receipt['source_manifest']
            and scope['train_control_cutoff'] == '2026-01-01T00:00:00+00:00',
            'SOURCE_SCOPE')
    source = read_dynamic_m5_target_source(scope, split='train')
    frame = source['frame']
    bars = pd.DatetimeIndex(pd.to_datetime(frame['time'], utc=True)).as_unit('ns').asi8
    require(len(bars) == 118249 and hashlib.sha256(bars.astype('<i8', copy=False).tobytes()).hexdigest()
            == axis['full_axes']['bar_start_ns_sha256'], 'ACTUAL_FULL_CLOCK')
    first, stop = plan['parent_range']
    decision = bars[first:stop] + M5_NS
    require(len(decision) == 6323 and decision[0] == axis['evaluation']['first_decision_ns']
            and decision[-1] == axis['evaluation']['last_decision_ns']
            and hashlib.sha256(decision.astype('<i8', copy=False).tobytes()).hexdigest()
                == axis['evaluation']['decision_ns_sha256'], 'ACTUAL_OCTOBER_CLOCK')
    counts = {key: np.zeros((len(LEADS), 2), np.int64) for key in
              ('terminal_open_observed', 'terminal_open_path_observed', 'closed_prefix_path_observed')}
    reason_counts: dict[str, dict[str, np.ndarray]] = {key: {} for key in
              ('terminal_open_reason', 'terminal_open_path_reason', 'closed_prefix_path_reason')}
    right_censored = np.zeros(len(LEADS), np.int64)
    for left in range(first, stop, CHUNK_PARENTS):
        right = min(stop, left + CHUNK_PARENTS)
        rows = np.repeat(np.arange(left, right, dtype=np.int64), len(LEADS))
        horizons = np.tile(LEADS, right - left)
        wanted = bars[rows] + M5_NS + horizons * MINUTE_NS
        eligible = wanted < plan['october_cutoff_ns_exclusive']
        right_censored += np.bincount(horizons[~eligible] - 1, minlength=len(LEADS))
        if not eligible.any():
            continue
        rows, horizons = rows[eligible], horizons[eligible]
        terminal = query_dynamic_m1_prefix(source['index'], fill_surface=source['fill'],
            query_entry_rows=rows, horizon_minutes=horizons,
            endpoint_kind='terminal_open_inclusive')
        closed = query_dynamic_m1_prefix(source['index'], fill_surface=source['fill'],
            query_entry_rows=rows, horizon_minutes=horizons,
            endpoint_kind='closed_m1_prefix')
        require(np.array_equal(terminal['query_entry_rows'], rows)
                and np.array_equal(closed['query_entry_rows'], rows)
                and np.array_equal(terminal['horizon_minutes'], horizons)
                and np.array_equal(closed['horizon_minutes'], horizons)
                and np.array_equal(terminal['terminal_time_ns'], wanted[eligible])
                and np.array_equal(closed['terminal_time_ns'], wanted[eligible])
                and np.all(terminal['terminal_time_ns'] < plan['october_cutoff_ns_exclusive']),
                'QUERY_CLOCK')
        for key, observed in (
            ('terminal_open_observed', terminal['terminal_gross_observed']),
            ('terminal_open_path_observed', terminal['path_observed']),
            ('closed_prefix_path_observed', closed['path_observed']),
        ):
            require(observed.shape == (len(rows), 2) and observed.dtype == np.bool_,
                    'OBSERVED_SHAPE:' + key)
            for side in range(2):
                counts[key][:, side] += np.bincount(horizons - 1,
                    weights=observed[:, side].astype(np.int64), minlength=len(LEADS)).astype(np.int64)
        for key, reason in (
            ('terminal_open_reason', terminal['terminal_censor_reason']),
            ('terminal_open_path_reason', terminal['path_censor_reason']),
            ('closed_prefix_path_reason', closed['path_censor_reason']),
        ):
            add_reasons(reason_counts[key], reason, horizons)
        require(time.monotonic() - started < plan['max_wall_seconds'], 'WALL_LIMIT')
    allowed = 6323 - right_censored
    require(np.all((allowed >= 0) & (allowed <= 6323)), 'RIGHT_CENSOR')
    for key, value in counts.items():
        require(np.all(value[:, 0] == value[:, 1]) and np.all(value[:, 0] <= allowed),
                'COUNT_DOMAIN:' + key)
    for key, mapping in reason_counts.items():
        require(np.array_equal(sum(mapping.values(), np.zeros(len(LEADS), np.int64)), allowed),
                'REASON_PARTITION:' + key)
    result = {'schema_version': SCHEMA, 'status': 'COMPLETE_READ_ONLY_OCTOBER_MASK_CENSUS',
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'plan': {'path': str(args.plan), 'sha256': args.plan_sha256},
        'parent_range': plan['parent_range'], 'october_cutoff_ns_exclusive': plan['october_cutoff_ns_exclusive'],
        'lead_minutes': LEADS.tolist(), 'right_censored_by_october_cutoff': right_censored.tolist(),
        'eligible_by_october_cutoff': allowed.tolist(),
        'observed_by_side': {key: value.tolist() for key, value in counts.items()},
        'reason_counts': {key: {code: value.tolist() for code, value in sorted(mapping.items())}
                          for key, mapping in reason_counts.items()},
        'source_read_scope': 'Existing owner reads full pre-2026 TRAIN physical target columns and quote index during source validation; only October parent rows and terminal times before 2024-11-01 enter the reported endpoint/path query census. No TEST source is read.',
        'model_calls': 0, 'optimizer_updates': 0, 'test_data_used': False,
        'independent_edge': 'UNKNOWN', 'economic_edge': 'UNKNOWN',
        'elapsed_seconds': time.monotonic() - started,
        'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    out = Path(plan['output_path'])
    require(out.is_absolute() and not out.exists(), 'OUTPUT_NO_REPLACE')
    out.parent.mkdir(parents=True, exist_ok=True)
    _publish_bytes(out, (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
    summary = {str(h): {'eligible': int(allowed[h-1]), 'terminal': int(counts['terminal_open_observed'][h-1, 0]),
                        'terminal_path': int(counts['terminal_open_path_observed'][h-1, 0]),
                        'closed_path': int(counts['closed_prefix_path_observed'][h-1, 0])}
               for h in (1, 5, 15, 60, 240, 480)}
    print(json.dumps({'status': result['status'], 'output_path': str(out),
                      'output_sha256': sha(out), 'summary': summary}, sort_keys=True))


if __name__ == '__main__':
    main()
