"""One-use TRAIN-only quote check of selected S M5 side against Entry geometry.

A hindsight S-side is a target diagnostic, never a selected trade or PnL.
No model, TEST, broker, network, or Exit policy is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from gx1.contracts import entry_causal_m1_outcomes_v1 as entry
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as target
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'market_state_s_m5_quote_alignment_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_S_M5_QUOTE_ALIGNMENT_20261115_001.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_S_M5_QUOTE_ALIGNMENT_20261115_001')
M1 = Path('/home/andre2/GX1_DATA/data/data/prebuilt/HISTORY2009W_BOOTSTRAP_20260927/artifacts/pretest_direct_m1_quotes_source/m1_quotes.parquet')
M1_SHA = '54b66c5e9b5a0125f089afcc9d3327c79af9c3543eebdb2644281ed9d3c2734c'
FOLD = 2
FIT_ROWS = 47338
PARENT_SHA = 'db43609ad6bd10e8d1a13b216243855c499189f8d2f08ef0a67d49de66e29fe9'
ISSUE_SHA = '380184a9c95a02f2716326765783746b1834371a7266545113fb1eae6722da47'
MINUTE_NS = 60_000_000_000
SLIPPAGE_BPS = 4.0
LONG_FINANCING_BPS_ANNUAL = 569.0
SECONDS_PER_YEAR = 31557600.0
COLUMNS = ('time', 'bid_open', 'bid_high', 'bid_low',
           'ask_open', 'ask_high', 'ask_low')
CODE = (
    'gx1/audit/measure_market_state_s_m5_quote_alignment_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/entry_causal_m1_outcomes_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
)


def need(value: object, code: str) -> None:
    if not bool(value):
        raise RuntimeError('S_M5_QUOTE_' + code)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ref(path: Path) -> dict:
    return {'path': str(path), 'sha256': sha(path)}


def encode(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def git(*args: str) -> str:
    return subprocess.check_output(('git', *args), cwd=REPO, text=True).strip()


def exact_axes() -> tuple[object, dict]:
    bound = target.load_bound_five_endpoint_targets()
    axes = bound.mature_oof_axes(fold_id=FOLD)
    need(axes['joined_fit_rows'] == FIT_ROWS
         and axes['fit_parent_rows_sha256'] == PARENT_SHA
         and axes['fit_decision_ns_sha256'] == ISSUE_SHA,
         'AXES')
    truth = bound.gather(axes['fit_parent_rows'], axes['fit_decision_ns'],
                         fold_id=FOLD, partition='fit')
    need(len(truth['issue_ns']) == FIT_ROWS
         and bool(truth['mature'].all())
         and bool(np.all(truth['issue_ns'] < target.FOLD_BOUNDARIES_NS[FOLD - 1]))
         and bool(np.all(truth['issue_ns'] + 5 * MINUTE_NS
                         < target.FOLD_BOUNDARIES_NS[FOLD - 1])), 'MATURE')
    return axes, truth


def preflight(expected_plan_sha: str) -> tuple[dict, dict, str]:
    need(Path.cwd().resolve() == REPO and git('branch', '--show-current') == 'work/gx1-current'
         and git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = git('rev-parse', 'HEAD')
    need(sha(PLAN) == expected_plan_sha and not ROOT.exists(), 'PLAN_OUTPUT')
    plan = json.loads(PLAN.read_text())
    need(plan == {
        'schema_version': 'gx1_s_m5_quote_alignment_plan_v1',
        'stage': STAGE, 'status': 'PREPARED_NOT_ADMITTED',
        'source_base_head': plan['source_base_head'],
        'input_refs': plan['input_refs'], 'source_owners': plan['source_owners'],
        'scope': {'fold_id': FOLD, 'fit_rows': FIT_ROWS,
                  'fit_parent_rows_sha256': PARENT_SHA,
                  'fit_decision_ns_sha256': ISSUE_SHA,
                  'timeframe': 'M5', 'query_horizon_minutes': 5,
                  'entry_timeframe': 'M5', 'exit_evaluation': False,
                  'test_data_used': False},
        'economics': {'entry': 'exact_m1_bid_ask_open_at_issue',
                      'terminal': 'exact_m1_bid_ask_open_at_issue_plus_5_minutes',
                      'path': 'five_contiguous_m1_bars',
                      'roundtrip_slippage_bps': SLIPPAGE_BPS,
                      'long_financing_bps_annual': LONG_FINANCING_BPS_ANNUAL,
                      'short_financing_bps_annual': 0.0,
                      'seconds_per_year': SECONDS_PER_YEAR,
                      'spread_counted_in_bid_ask': True,
                      'cost_status': 'declared_projection_not_historical_broker_fills'},
        'limits': {'class': 'audit', 'memory': '4G', 'swap': '512M',
                   'wall_seconds': 1800, 'model_forwards': 0,
                   'optimizer_steps': 0, 'max_invocations': 1},
        'output_root': str(ROOT),
    }, 'PLAN_SCHEMA')
    need(subprocess.run(('git', 'merge-base', '--is-ancestor',
                         plan['source_base_head'], head), cwd=REPO).returncode == 0,
         'ANCESTOR')
    need(set(plan['source_owners']) == set(CODE)
         and all(sha(REPO / name) == digest
                 for name, digest in plan['source_owners'].items()), 'CODE_HASH')
    need(plan['input_refs'] == {
        'm1': {'path': str(M1), 'sha256': M1_SHA},
        'mask': {'path': target.REFS['mask'][0],
                 'sha256': target.REFS['mask'][1]},
        'model_axes': {'path': target.REFS['model_axes'][0],
                       'sha256': target.REFS['model_axes'][1]},
    } and sha(M1) == M1_SHA, 'INPUT_HASH')
    _, truth = exact_axes()
    need(len(truth['allowed']) == FIT_ROWS, 'TRUTH')
    policy_ref = ref(POLICY)
    policy = json.loads(POLICY.read_text())
    gate = policy.get(STAGE, {})
    need(policy.get('training_enabled') is False
         and policy.get('native_training_enabled') is False
         and policy.get('native_launch_authorized') is False
         and gate == {'status': 'ADMITTED',
                      'plan': ref(PLAN), 'output_root': str(ROOT),
                      'max_invocations': 1, 'claimed_invocations': 0,
                      'completed_invocations': 0,
                      'execution_authorized': True,
                      'data_reads_authorized': True,
                      'model_forwards_authorized': 0,
                      'optimizer_steps_authorized': 0,
                      'training_authorized': False,
                      'test_authorized': False,
                      'broker_access_allowed': False,
                      'relaunch_allowed': False}, 'POLICY')
    return plan, policy_ref, head


def summary(values: np.ndarray) -> dict:
    need(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), 'SUMMARY')
    return {'n': int(len(values)), 'mean_bps': float(values.mean()),
            'median_bps': float(np.median(values)),
            'p10_bps': float(np.quantile(values, 0.1)),
            'p90_bps': float(np.quantile(values, 0.9)),
            'positive': int((values > 0).sum()),
            'nonpositive': int((values <= 0).sum())}


def measure(truth: dict) -> dict:
    decisions = truth['issue_ns']
    lo = pd.Timestamp(int(decisions[0]), unit='ns', tz='UTC')
    hi = pd.Timestamp(int(decisions[-1] + 6 * MINUTE_NS), unit='ns', tz='UTC')
    need(hi < pd.Timestamp('2025-01-01', tz='UTC'), 'READ_RANGE')
    time_type = pa.timestamp('ns', tz='UTC')
    predicate = ((ds.field('time') >= pa.scalar(lo.to_pydatetime(), type=time_type))
                 & (ds.field('time') < pa.scalar(hi.to_pydatetime(), type=time_type)))
    table = ds.dataset(M1, format='parquet').to_table(columns=list(COLUMNS), filter=predicate)
    m1 = table.to_pandas()
    need(len(m1) > 0 and m1['time'].is_monotonic_increasing
         and m1['time'].is_unique
         and m1['time'].min() >= lo and m1['time'].max() < hi, 'M1_SOURCE')
    prepared = entry.prepare_causal_m1_quote_source(m1)
    starts = pd.to_datetime(decisions - 5 * MINUTE_NS, utc=True)
    fills = entry.build_entry_m1_fill_surface(m5_decision_times=starts,
                                               closed_m1=prepared)
    outcome = entry.causal_m1_outcomes_at_horizon(fill_surface=fills,
                                                  closed_m1=prepared,
                                                  horizon_m5_bars=1)
    valid = outcome['outcome_valid'].to_numpy(dtype=np.bool_)
    bits = truth['allowed'][:, 0]
    known = np.isin(bits, (1, 4)) & truth['eligible'][:, 0]
    need(len(valid) == FIT_ROWS and len(bits) == FIT_ROWS
         and np.isin(bits, (1, 2, 4, 5, 7)).all(), 'JOIN')
    gross_long = outcome['long_executable_pnl_bps'].to_numpy(dtype=np.float64)
    gross_short = outcome['short_executable_pnl_bps'].to_numpy(dtype=np.float64)
    finance = LONG_FINANCING_BPS_ANNUAL * 300 / SECONDS_PER_YEAR
    net_long = gross_long - SLIPPAGE_BPS - finance
    net_short = gross_short - SLIPPAGE_BPS
    chosen = known & valid
    need(bool(chosen.any()), 'NO_CERTIFIED_QUOTE')
    hindsight = np.where(bits[chosen] == 4, net_long[chosen], net_short[chosen])
    mfe_long = outcome['long_mfe_bps'].to_numpy(dtype=np.float64)
    mae_long = outcome['long_mae_bps'].to_numpy(dtype=np.float64)
    mfe_short = outcome['short_mfe_bps'].to_numpy(dtype=np.float64)
    mae_short = outcome['short_mae_bps'].to_numpy(dtype=np.float64)
    hindsight_mfe = np.where(bits[chosen] == 4, mfe_long[chosen], mfe_short[chosen])
    hindsight_mae = np.where(bits[chosen] == 4, mae_long[chosen], mae_short[chosen])
    groups = {'certified_down': bits == 1, 'flat': bits == 2,
              'certified_up': bits == 4, 'active_side_unknown': bits == 5,
              'all_unknown': bits == 7}
    return {
        'fit_rows': FIT_ROWS, 'm1_rows_loaded': len(m1),
        'm1_first_utc': m1['time'].iloc[0].isoformat(),
        'm1_last_utc': m1['time'].iloc[-1].isoformat(),
        'quote_coverage': {name: {'target_rows': int(mask.sum()),
                                  'exact_contiguous_quote_rows': int((mask & valid).sum())}
                           for name, mask in groups.items()},
        'all_exact_contiguous_quote_rows': int(valid.sum()),
        'certified_side_exact_quote_rows': int(chosen.sum()),
        'hindsight_s_side_5m_net': summary(hindsight),
        'same_rows_always_long_5m_net': summary(net_long[chosen]),
        'same_rows_always_short_5m_net': summary(net_short[chosen]),
        'same_rows_s_side_mfe_bps': summary(hindsight_mfe),
        'same_rows_s_side_mae_bps': summary(hindsight_mae),
        'hindsight_s_side_negative_net_count': int((hindsight < 0).sum()),
        'hindsight_s_side_nonpositive_mfe_count': int((hindsight_mfe <= 0).sum()),
        'interpretation_limit': 'Target-known hindsight subset; labels are future information. Fixed 5m terminal is a diagnostic, not a hold or Exit rule. M1 extrema do not establish intraminute or pre-MFE order, fill, realized PnL, or edge.'
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    need(args.preflight != args.run, 'MODE')
    plan, policy_ref, head = preflight(args.plan_sha256)
    if args.preflight:
        x, y = entry._entry_notional_pnl_bps(
            entry_bid=np.array([99.0]), entry_ask=np.array([100.0]),
            exit_bid=np.array([101.0]), exit_ask=np.array([102.0]))
        need(abs(float(x[0]) - 100.0) < 1e-12
             and abs(float(y[0]) + 10000 * 3 / 99) < 1e-12,
             'SYNTHETIC_SUCCESS')
        ok = summary(np.array([1.0, -2.0], dtype=np.float64))
        need(ok['n'] == 2 and ok['positive'] == 1
             and ok['nonpositive'] == 1, 'SYNTHETIC_REPORT_SUCCESS')
        try:
            summary(np.array([np.nan], dtype=np.float64))
        except RuntimeError as exc:
            need(str(exc) == 'S_M5_QUOTE_SUMMARY', 'SYNTHETIC_REPORT_FAILURE')
        else:
            need(False, 'SYNTHETIC_REPORT_FAILURE')
        print(json.dumps({'preflight': 'PASS_NO_CLAIM_NO_M1_DECODE',
                          'fit_rows': FIT_ROWS, 'source_head': head}, sort_keys=True))
        return 0
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    ROOT.mkdir(parents=False, exist_ok=False)
    claim = {'schema_version': 'gx1_s_m5_quote_alignment_claim_v1',
             'stage': STAGE, 'plan': ref(PLAN), 'policy': policy_ref,
             'source_head': head, 'test_data_used': False}
    _publish_bytes(ROOT / 'CLAIM.json', encode(claim))
    t0 = time.monotonic()
    try:
        _, truth = exact_axes()
        result = {'schema_version': 'gx1_s_m5_quote_alignment_result_v1',
                  'status': 'COMPLETE', 'stage': STAGE,
                  'plan': ref(PLAN), 'policy': policy_ref,
                  'claim': ref(ROOT / 'CLAIM.json'),
                  'source_head': head, 'metrics': measure(truth),
                  'model_forwards': 0, 'optimizer_steps': 0,
                  'test_data_used': False, 'elapsed_seconds': time.monotonic() - t0}
        need(git('rev-parse', 'HEAD') == head
             and git('status', '--porcelain') == ''
             and ref(POLICY) == policy_ref and sha(M1) == M1_SHA,
             'FINAL_SOURCE')
        _publish_bytes(ROOT / 'RESULT.json', encode(result))
        terminal = {'schema_version': 'gx1_s_m5_quote_alignment_terminal_v1',
                    'status': 'COMPLETE', 'exit_code': 0, 'stage': STAGE,
                    'claim': ref(ROOT / 'CLAIM.json'),
                    'result': ref(ROOT / 'RESULT.json'),
                    'source_head': head, 'test_data_used': False}
        _publish_bytes(ROOT / 'TERMINAL.json', encode(terminal))
    except BaseException as exc:
        terminal = {'schema_version': 'gx1_s_m5_quote_alignment_terminal_v1',
                    'status': 'FAILED_CONSUMED', 'exit_code': 1, 'stage': STAGE,
                    'claim': ref(ROOT / 'CLAIM.json'), 'result': None,
                    'source_head': head, 'reason': type(exc).__name__ + ':' + str(exc),
                    'test_data_used': False}
        _publish_bytes(ROOT / 'TERMINAL.json', encode(terminal))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
