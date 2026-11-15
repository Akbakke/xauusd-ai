"""One-use TRAIN-only S/M5 quote-path diagnostic on fixed Fold2 axes.

S is a future market-state label. Its hindsight side is never a trade
selection, Entry target, model score, Exit rule, or claim of economic Edge.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch

from gx1.audit.measure_market_state_fold2_exact_prior_later_v1 import axes
from gx1.audit.measure_market_state_s_m5_quote_alignment_v1 import (
    encode, git, ref, sha, summary,
)
from gx1.contracts import entry_causal_m1_outcomes_v1 as path
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts.entry_observed_market_v1 import (
    build_entry_observed_market_targets, read_dynamic_m5_target_source,
    require_entry_observed_market_scope,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'market_state_s_m5_path_cost_alignment_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_S_M5_PATH_COST_ALIGNMENT_20261115_001.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_S_M5_PATH_COST_ALIGNMENT_20261115_001')
COVERAGE = REPO / 'configs/research/MARKET_STATE_FULL_GC_FOLD2_LATER_COVERAGE_20261115_001.json'
PRIOR = Path('/home/andre2/GX1_RUNS/MARKET_STATE_FOLD2_EXACT_PRIOR_LATER_4096_20261115_001')
QUOTE = Path('/home/andre2/GX1_RUNS/MARKET_STATE_S_M5_QUOTE_ALIGNMENT_20261115_001')
QUOTE_PLAN = REPO / 'configs/research/MARKET_STATE_S_M5_QUOTE_ALIGNMENT_20261115_001.json'
OLD_MASK_PLAN = REPO / 'configs/research/DYNAMIC_M5_OCTOBER_ENTRY_MASK_COVERAGE_20261106_001.json'
RECIPE = Path('/home/andre2/GX1_RUNS/NATIVE_M5_COMPACT_SINGLE_FULL_NATIVE_EPOCH_20261020_108/RECIPE.json')
PHYSICAL = Path('/home/andre2/GX1_RUNS/NATIVE_M5_ACTUAL_NATIVE_BINDINGS_20261019_057/PHYSICAL_TARGET_RECEIPT.json')
M1 = Path('/home/andre2/GX1_DATA/data/data/prebuilt/HISTORY2009W_BOOTSTRAP_20260927/artifacts/pretest_direct_m1_quotes_source/m1_quotes.parquet')
M1_SHA = '54b66c5e9b5a0125f089afcc9d3327c79af9c3543eebdb2644281ed9d3c2734c'
HORIZONS = (1, 5, 15, 60, 240, 480)
MINUTE_NS = 60_000_000_000
M5_NS = 5 * MINUTE_NS
FIT_CUTOFF = state.FOLD_BOUNDARIES_NS[1]
LATER_CUTOFF = state.FOLD_BOUNDARIES_NS[2]
ECONOMICS_SHA = '81b7d1c7311d831753e273840ed5e732819c4c4371aa832b76b7f9ec1118b999'
CODE = (
    'gx1/audit/measure_market_state_s_m5_path_cost_alignment_v1.py',
    'gx1/audit/measure_market_state_fold2_exact_prior_later_v1.py',
    'gx1/audit/measure_market_state_s_m5_quote_alignment_v1.py',
    'gx1/contracts/entry_causal_m1_outcomes_v1.py',
    'gx1/contracts/entry_observed_market_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
)
INPUTS = {
    'coverage': (COVERAGE, 'ff18b8fcde8b09c41cbec5c216c532ef688a6be29767e456475cef941495c640'),
    'prior_result': (PRIOR / 'RESULT.json', '252b397f794bb7663e7836b941ba69f33fcc58f31bb6e0796af448b408140b07'),
    'prior_terminal': (PRIOR / 'TERMINAL.json', '304b7204faab52d9440f659a4e46bcd9da2f4de856963dcf37adb8a265a6b497'),
    'quote_result': (QUOTE / 'RESULT.json', '44dd9eb7b60de6bf4d2b7c75a69fecc834df94db51d2724939aa0e83521b952b'),
    'quote_plan': (QUOTE_PLAN, 'a1a1dcb0ca3a3666220d43ad2a9952f75f8b3f9fa4072d3bb6bd682edda60006'),
    'recipe': (RECIPE, '83dbdaec44aac242db740cc43bd1314679aba2668eb53da4bc5e1a972afc82cd'),
    'physical_receipt': (PHYSICAL, 'dd95d9ecb43f309635e61ba268ec4db6518a3ee23550aa0c94330ff303a5063d'),
    'old_mask_plan': (OLD_MASK_PLAN, '6aad1e0a2cebddd405b034f932815d8cb434e4a7ec7709c56764cb7216b0fc35'),
    'm1': (M1, M1_SHA),
    'state_mask': (Path(state.REFS['mask'][0]), state.REFS['mask'][1]),
    'model_axes': (Path(state.REFS['model_axes'][0]), state.REFS['model_axes'][1]),
}


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('S_M5_PATH_COST_' + reason)


def summarized(values: np.ndarray) -> dict | None:
    return summary(np.asarray(values, dtype=np.float64)) if len(values) else None


def checked_report(value: dict) -> None:
    need(value['fit']['rows'] == 47338
         and value['later']['rows'] == 4096
         and set(value['fit']['horizons']) == set(value['later']['horizons'])
             == set(map(str, HORIZONS)), 'REPORT_COHORT')
    for name in ('fit', 'later'):
        cohort = value[name]
        for horizon in cohort['horizons'].values():
            n = horizon['eligible_rows']
            counts = horizon['counts']
            need(0 <= counts['path_observed'] <= counts['terminal_observed'] <= n
                 and 0 <= counts['known_s_joint'] <= counts['known_s_terminal']
                    <= counts['known_s_eligible'] <= n, 'REPORT_COUNTS')
            for key in ('hindsight_s_side_net_bps', 'same_rows_long_net_bps',
                        'same_rows_short_net_bps'):
                item = horizon[key]
                need((item is None and counts['known_s_terminal'] == 0)
                     or (item is not None
                         and item['n'] == counts['known_s_terminal']),
                     'REPORT_NET')
            for key in ('hindsight_s_side_mfe_bps',
                        'hindsight_s_side_pre_mfe_mae_upper_bps'):
                item = horizon[key]
                need((item is None and counts['known_s_joint'] == 0)
                     or (item is not None and item['n'] == counts['known_s_joint']),
                     'REPORT_PATH')


def inspect(plan_sha: str) -> tuple[dict, dict, str, object, dict, dict, dict]:
    need(Path.cwd().resolve() == REPO
         and git('branch', '--show-current') == 'work/gx1-current'
         and git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = git('rev-parse', 'HEAD')
    need(not ROOT.exists() and sha(PLAN) == plan_sha, 'PLAN_OUTPUT')
    plan = json.loads(PLAN.read_text())
    need(set(plan) == {'schema_version', 'stage', 'status', 'source_base_head',
                       'input_refs', 'source_owners', 'scope', 'limits', 'output_root'}
         and plan['schema_version'] == 'gx1_s_m5_path_cost_alignment_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['scope'] == {
             'fold_id': 2, 'fit_rows': 47338, 'later_rows': 4096,
             'horizon_minutes': list(HORIZONS),
             'fit_cutoff_ns_exclusive': FIT_CUTOFF,
             'later_cutoff_ns_exclusive': LATER_CUTOFF,
             'entry_timeframe': 'M5', 'exit_evaluation': False,
             'model_decision_used': False, 'test_data_used': False}
         and plan['limits'] == {
             'class': 'audit', 'memory': '4G', 'swap': '512M',
             'wall_seconds': 1800, 'max_invocations': 1,
             'model_forwards': 0, 'optimizer_steps': 0,
             'training_authorized': False}
         and plan['output_root'] == str(ROOT), 'PLAN_SCHEMA')
    need(subprocess.run(('git', 'merge-base', '--is-ancestor',
                         plan['source_base_head'], head), cwd=REPO).returncode == 0,
         'ANCESTOR')
    need(set(plan['source_owners']) == set(CODE)
         and all(sha(REPO / rel) == digest
                 for rel, digest in plan['source_owners'].items()), 'CODE_HASH')
    expected = {name: {'path': str(p), 'sha256': digest}
                for name, (p, digest) in INPUTS.items()}
    need(plan['input_refs'] == expected
         and all(sha(p) == digest for p, digest in INPUTS.values()), 'INPUT_HASH')
    coverage = json.loads(COVERAGE.read_text())
    prior = json.loads((PRIOR / 'RESULT.json').read_text())
    prior_terminal = json.loads((PRIOR / 'TERMINAL.json').read_text())
    quote = json.loads((QUOTE / 'RESULT.json').read_text())
    quote_plan = json.loads(QUOTE_PLAN.read_text())
    old_mask_plan = json.loads(OLD_MASK_PLAN.read_text())
    physical = json.loads(PHYSICAL.read_text())
    need(prior['status'] == prior_terminal['status'] == 'COMPLETE'
         and prior_terminal['exit_code'] == 0
         and prior_terminal['result'] == expected['prior_result']
         and prior['fit_rows'] == 47338 and prior['selected_score_rows'] == 4096
         and prior['selected_parent_rows_sha256']
             == coverage['selected_parent_rows_sha256']
         and prior['selected_decision_ns_sha256']
             == coverage['selected_decision_ns_sha256']
         and quote['status'] == 'COMPLETE'
         and quote['metrics']['fit_rows'] == 47338
         and quote['metrics']['hindsight_s_side_5m_net']['n'] == 16071
         and quote['plan'] == expected['quote_plan']
         and quote_plan['economics']['long_financing_bps_annual'] == 569.0
         and quote_plan['economics']['roundtrip_slippage_bps'] == 4.0
         and old_mask_plan['input_recipe'] == expected['recipe']
         and old_mask_plan['physical_target_receipt'] == expected['physical_receipt']
         and physical['splits']['train']['rows'] == 118249
         and physical['splits']['train']['dynamic_m5_supervision']['dense_future_paths_materialized'] is False,
         'RECEIPT_CHAIN')
    state_bound = state.load_bound_five_endpoint_targets()
    fit, later_rows, later_issues = axes(coverage, state_bound)
    recipe = json.loads(RECIPE.read_text())
    scope = require_entry_observed_market_scope(recipe)
    need(scope['target_m1_source_sha256'] == M1_SHA
         and scope['economics']['economics_sha256'] == ECONOMICS_SHA
         and scope['economics']['round_trip_execution_cost_bps'] == 4.0
         and scope['economics']['annual_financing_cost_bps'] == [532.0, 0.0]
         and scope['physical_sources']['train']['parquet']
             == physical['splits']['train']['source_parquet']
         and scope['physical_sources']['train']['manifest']
             == physical['splits']['train']['source_manifest']
         and scope['train_control_cutoff'] == '2026-01-01T00:00:00+00:00',
         'ENTRY_SOURCE_COST')
    gate = json.loads(POLICY.read_text()).get(STAGE)
    need(json.loads(POLICY.read_text())['training_enabled'] is False
         and gate == {
             'status': 'ADMITTED', 'plan': ref(PLAN),
             'output_root': str(ROOT), 'max_invocations': 1,
             'claimed_invocations': 0, 'completed_invocations': 0,
             'execution_authorized': True, 'data_reads_authorized': True,
             'model_forwards_authorized': 0, 'optimizer_steps_authorized': 0,
             'training_authorized': False, 'test_authorized': False,
             'broker_access_allowed': False, 'relaunch_allowed': False},
         'POLICY')
    return plan, ref(POLICY), head, state_bound, fit, {
        'rows': later_rows, 'issues': later_issues}, scope


def measure_cohort(source: dict, bound: object, *, rows: np.ndarray,
                   issues: np.ndarray, partition: str, cutoff: int,
                   economics: dict) -> dict:
    need(partition in ('fit', 'score') and len(rows) == len(issues)
         and len(rows) > 0 and int(rows.max()) < len(source['frame']),
         'COHORT_INPUT')
    bar_ns = pd.DatetimeIndex(pd.to_datetime(
        source['frame']['time'].iloc[rows], utc=True)).as_unit('ns').asi8
    need(np.array_equal(bar_ns + M5_NS, issues), 'PHYSICAL_DECISION_CLOCK')
    truth = bound.gather(rows, issues, fold_id=2, partition=partition)
    bits = truth['allowed'][:, 0]
    known = np.isin(bits, (1, 4)) & truth['eligible'][:, 0] & truth['mature'][:, 0]
    result = {'rows': len(rows), 'parent_rows_sha256': None,
              'decision_ns_sha256': None, 's_m5_known_side_rows': int(known.sum()),
              's_m5_bits': {str(bit): int((bits == bit).sum()) for bit in (1, 2, 4, 5, 7)},
              'horizons': {}}
    for horizon in HORIZONS:
        allowed = issues + horizon * MINUTE_NS < cutoff
        qrows, qbits, qknown, qissues = rows[allowed], bits[allowed], known[allowed], issues[allowed]
        need(len(qrows) > 0, 'EMPTY_HORIZON')
        observation = path.query_dynamic_m1_prefix(
            source['index'], fill_surface=source['fill'],
            query_entry_rows=qrows,
            horizon_minutes=np.full(len(qrows), horizon, dtype=np.int64),
            endpoint_kind='terminal_open_inclusive')
        need(np.array_equal(observation['query_entry_rows'], qrows)
             and np.array_equal(observation['terminal_time_ns'],
                                qissues + horizon * MINUTE_NS)
             and bool(np.all(observation['terminal_time_ns'] < cutoff))
             and observation['endpoint_kind'] == 'terminal_open_inclusive',
             'QUERY_BINDING')
        term = observation['terminal_gross_observed']
        pathmask = observation['path_observed']
        need(term.shape == pathmask.shape == (len(qrows), 2)
             and np.array_equal(term[:, 0], term[:, 1])
             and np.array_equal(pathmask[:, 0], pathmask[:, 1])
             and bool(np.all(~pathmask[:, 0] | term[:, 0])), 'OBSERVATION_MASK')
        terminal = term[:, 0]
        joint = terminal & pathmask[:, 0]
        known_term = qknown & terminal
        known_joint = qknown & joint
        net = np.full((len(qrows), 2), np.nan, dtype=np.float64)
        if terminal.any():
            measured = build_entry_observed_market_targets(
                gross_return_bps=torch.tensor(
                    observation['terminal_gross_return_bps'][terminal],
                    dtype=torch.float64),
                elapsed_wall_clock_seconds=torch.tensor(
                    observation['elapsed_wall_seconds'][terminal],
                    dtype=torch.int64),
                economics=economics)
            net[terminal] = measured['target_bps'][:, :2].numpy()
        side = np.where(qbits == 4, 0, 1)
        selected_net = net[np.flatnonzero(known_term), side[known_term]]
        mfe = observation['mfe_bps'][np.flatnonzero(known_joint), side[known_joint]]
        mae = observation['mae_before_first_mfe_upper_bps'][
            np.flatnonzero(known_joint), side[known_joint]]
        need(bool(np.isfinite(selected_net).all())
             and bool(np.isfinite(mfe).all())
             and bool(np.isfinite(mae).all())
             and bool(np.all(mae >= 0)), 'OUTCOME_FINITE')
        selected_weeks = pd.to_datetime(qissues[known_term], unit='ns', utc=True).to_period('W-SUN').astype(str)
        weekly = {str(week): summarized(selected_net[selected_weeks == week])
                  for week in np.unique(selected_weeks)}
        result['horizons'][str(horizon)] = {
            'query_horizon_minutes': horizon,
            'right_censored_rows': int((~allowed).sum()),
            'eligible_rows': len(qrows),
            'counts': {
                'terminal_observed': int(terminal.sum()),
                'path_observed': int(pathmask[:, 0].sum()),
                'known_s_eligible': int(qknown.sum()),
                'known_s_terminal': int(known_term.sum()),
                'known_s_joint': int(known_joint.sum())},
            'terminal_censor_reasons': {
                str(k): int(v) for k, v in zip(
                    *np.unique(observation['terminal_censor_reason'], return_counts=True))},
            'path_censor_reasons': {
                str(k): int(v) for k, v in zip(
                    *np.unique(observation['path_censor_reason'], return_counts=True))},
            'hindsight_s_side_net_bps': summarized(selected_net),
            'same_rows_long_net_bps': summarized(net[known_term, 0]),
            'same_rows_short_net_bps': summarized(net[known_term, 1]),
            'hindsight_s_side_mfe_bps': summarized(mfe),
            'hindsight_s_side_pre_mfe_mae_upper_bps': summarized(mae),
            'by_utc_week_hindsight_s_side_net_bps': weekly,
        }
    return result


def run(plan_sha: str) -> None:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    plan, policy_ref, head, bound, fit, later, scope = inspect(plan_sha)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    ROOT.mkdir(parents=False, exist_ok=False)
    claim = {'schema_version': 'gx1_s_m5_path_cost_claim_v1',
             'stage': STAGE, 'plan': ref(PLAN), 'policy': policy_ref,
             'source_head': head, 'test_data_used': False}
    _publish_bytes(ROOT / 'CLAIM.json', encode(claim))
    t0 = time.monotonic()
    try:
        source = read_dynamic_m5_target_source(scope, split='train')
        fit_result = measure_cohort(
            source, bound, rows=fit['fit_parent_rows'],
            issues=fit['fit_decision_ns'], partition='fit',
            cutoff=FIT_CUTOFF, economics=scope['economics'])
        later_result = measure_cohort(
            source, bound, rows=later['rows'], issues=later['issues'],
            partition='score', cutoff=LATER_CUTOFF,
            economics=scope['economics'])
        fit_result['parent_rows_sha256'] = fit['fit_parent_rows_sha256']
        fit_result['decision_ns_sha256'] = fit['fit_decision_ns_sha256']
        coverage = json.loads(COVERAGE.read_text())
        later_result['parent_rows_sha256'] = coverage['selected_parent_rows_sha256']
        later_result['decision_ns_sha256'] = coverage['selected_decision_ns_sha256']
        result = {
            'schema_version': 'gx1_s_m5_path_cost_result_v1',
            'status': 'COMPLETE', 'stage': STAGE,
            'plan': ref(PLAN), 'policy': policy_ref,
            'claim': ref(ROOT / 'CLAIM.json'), 'source_head': head,
            'observed_utc': datetime.now(timezone.utc).isoformat(),
            'economics': scope['economics'],
            'comparison_quote_audit_annual_long_financing_bps': 569.0,
            'horizon_minutes': list(HORIZONS),
            'fit': fit_result, 'later': later_result,
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'trade_selection': False,
            'elapsed_seconds': time.monotonic() - t0}
        checked_report(result)
        need(git('rev-parse', 'HEAD') == head
             and git('status', '--porcelain') == ''
             and ref(POLICY) == policy_ref
             and all(sha(p) == digest for p, digest in INPUTS.values()),
             'FINAL_SOURCE')
        _publish_bytes(ROOT / 'RESULT.json', encode(result))
        _publish_bytes(ROOT / 'TERMINAL.json', encode({
            'schema_version': 'gx1_s_m5_path_cost_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'stage': STAGE,
            'claim': ref(ROOT / 'CLAIM.json'),
            'result': ref(ROOT / 'RESULT.json'),
            'source_head': head, 'test_data_used': False}))
    except BaseException as exc:
        _publish_bytes(ROOT / 'TERMINAL.json', encode({
            'schema_version': 'gx1_s_m5_path_cost_terminal_v1',
            'status': 'FAILED_CONSUMED', 'exit_code': 1, 'stage': STAGE,
            'claim': ref(ROOT / 'CLAIM.json'), 'result': None,
            'source_head': head,
            'reason': type(exc).__name__ + ':' + str(exc),
            'test_data_used': False}))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    need(args.preflight != args.run, 'MODE')
    if args.preflight:
        _, _, head, _, fit, later, scope = inspect(args.plan_sha256)
        need(len(fit['fit_parent_rows']) == 47338
             and len(later['rows']) == 4096, 'PREFLIGHT_AXES')
        toy = {
            'fit': {'rows': 47338, 'horizons': {str(h): {
                'eligible_rows': 2,
                'counts': {'terminal_observed': 2, 'path_observed': 1,
                           'known_s_eligible': 1, 'known_s_terminal': 1,
                           'known_s_joint': 1},
                'hindsight_s_side_net_bps': summary(np.array([1.0])),
                'same_rows_long_net_bps': summary(np.array([1.0])),
                'same_rows_short_net_bps': summary(np.array([-1.0])),
                'hindsight_s_side_mfe_bps': summary(np.array([2.0])),
                'hindsight_s_side_pre_mfe_mae_upper_bps': summary(np.array([0.5]))}
                for h in HORIZONS}},
            'later': {'rows': 4096, 'horizons': {str(h): {
                'eligible_rows': 2,
                'counts': {'terminal_observed': 2, 'path_observed': 1,
                           'known_s_eligible': 1, 'known_s_terminal': 1,
                           'known_s_joint': 1},
                'hindsight_s_side_net_bps': summary(np.array([1.0])),
                'same_rows_long_net_bps': summary(np.array([1.0])),
                'same_rows_short_net_bps': summary(np.array([-1.0])),
                'hindsight_s_side_mfe_bps': summary(np.array([2.0])),
                'hindsight_s_side_pre_mfe_mae_upper_bps': summary(np.array([0.5]))}
                for h in HORIZONS}}}
        checked_report(toy)
        try:
            toy['later']['horizons']['5']['counts']['known_s_joint'] = 2
            checked_report(toy)
        except RuntimeError as exc:
            need(str(exc) == 'S_M5_PATH_COST_REPORT_COUNTS', 'SYNTHETIC_ERROR')
        else:
            need(False, 'SYNTHETIC_ERROR')
        need(scope['economics']['economics_sha256'] == ECONOMICS_SHA,
             'PREFLIGHT_COST')
        print(json.dumps({'preflight': 'PASS_NO_CLAIM_NO_PRICE_ARRAY_DECODE',
                          'fit_rows': 47338, 'later_rows': 4096,
                          'source_head': head}, sort_keys=True))
    else:
        run(args.plan_sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
