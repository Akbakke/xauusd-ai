"""One capped, TRAIN-only signed-net versus causal GC/price reference audit.

Reuses the certified market-state axes, XAU source loader, and GC input reader.
Publishes one minimal local signed target tape for reuse; no model or TEST.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from gx1.audit import build_full_train_market_state_endpoint_candidate_v1 as base
from gx1.audit import market_state_signed_net_reference_core_v1 as core
from gx1.audit import review_market_state_net_side_v1 as net
from gx1.audit import score_market_state_gc_flow_reference_v1 as gc
from gx1.audit import verify_market_state_gray_physical_v1 as gray
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as state_target
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as binding
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = base.REPO
STAGE = 'market_state_signed_net_gc_reference_20261108_001'
PLAN_PATH = REPO / 'configs/research/MARKET_STATE_SIGNED_NET_GC_REFERENCE_20261108_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_SIGNED_NET_GC_REFERENCE_20261108_001')
TAPE_NAME = 'SIGNED_TARGET_TAPE.npz'
SOURCE_PATHS = (
    'gx1/audit/measure_market_state_signed_net_gc_reference_v1.py',
    'gx1/audit/market_state_signed_net_reference_core_v1.py',
    'gx1/audit/review_market_state_net_side_v1.py',
    'gx1/audit/score_market_state_gc_flow_reference_v1.py',
    'gx1/audit/market_state_m5_gc_flow_reference_core_v1.py',
    'gx1/audit/verify_market_state_gray_physical_v1.py',
    'gx1/audit/build_full_train_market_state_endpoint_candidate_v1.py',
    'gx1/contracts/entry_dynamic_mtf_state_targets_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_time_query_target_v1.py',
    'gx1/features/entry_specialist_feature_groups_v1.py',
    'gx1/features/swing_structure_v1.py',
    'gx1/features/gc_order_flow_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'scripts/gx1_capped_run.sh',
)
INPUT_PATHS = {
    'net_plan': net.PLAN_PATH,
    'net_result': net.ROOT / 'RESULT.json',
    'net_terminal': net.ROOT / 'TERMINAL.json',
    'gc_plan': gc.PLAN_PATH,
    'gc_result': gc.OUT_ROOT / 'RESULT.json',
    'gc_terminal': gc.OUT_ROOT / 'TERMINAL.json',
}
LIMITS = {'memory_max': '4G', 'swap_max': '512M', 'wall_seconds': 1800,
          'model_forwards': 0, 'optimizer_steps': 0}
PRICE_COLUMNS = {'ema5_distance': 8, 'swing_high_age': 14,
                 'swing_low_age': 15, 'swing_impulse_present': 17}
PRICE_NAMES = {'ema5_distance': 'close_distance_from_ema5_bps',
               'swing_high_age': 'bars_since_swing_high',
               'swing_low_age': 'bars_since_swing_low',
               'swing_impulse_present': 'swing_impulse_present'}


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('MARKET_STATE_SIGNED_NET_GC_' + reason)


def ref(path: Path) -> dict:
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def current_state() -> tuple[str, str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(('git', *args), cwd=REPO, text=True).strip()
    return git('rev-parse', 'HEAD'), git('branch', '--show-current'), git('status', '--porcelain')


def inspect(plan_ref: dict) -> tuple[dict, dict, dict, dict, dict, dict]:
    head, branch, dirty = current_state()
    need(branch == 'work/gx1-current' and not dirty, 'CLEAN_SOURCE')
    need(plan_ref['path'] == str(PLAN_PATH)
         and ref(PLAN_PATH) == plan_ref, 'PLAN_IDENTITY')
    plan = json.loads(PLAN_PATH.read_text())
    need(set(plan) == {'schema_version', 'stage', 'status', 'source_base_head',
                       'source_owners', 'inputs', 'scope', 'limits', 'output_root'}
         and plan['schema_version'] == 'gx1_market_state_signed_net_gc_reference_plan_v1'
         and plan['stage'] == STAGE
         and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['output_root'] == str(ROOT)
         and plan['limits'] == LIMITS
         and plan['scope'] == {'timeframes': list(state_target.TIMEFRAMES),
                               'native_lead_minutes': list(base.LEADS),
                               'model_rows': 118249, 'm5_issues': 118706,
                               'fold_ids': [1, 2, 3, 4, 5],
                               'price_columns': PRICE_COLUMNS,
                               'core_contract_sha256': core.contract()['contract_sha256'],
                               'test_data_used': False}
         and subprocess.run(('git', 'merge-base', '--is-ancestor',
                             plan['source_base_head'], head), cwd=REPO,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0,
         'PLAN_SCOPE')
    owners = plan['source_owners']
    need(set(owners) == set(SOURCE_PATHS)
         and all(ref(REPO / name)['sha256'] == owners[name] for name in SOURCE_PATHS),
         'SOURCE_OWNERS')
    need(set(plan['inputs']) == set(INPUT_PATHS)
         and all(plan['inputs'][key] == ref(path)
                 for key, path in INPUT_PATHS.items()), 'INPUT_REFS')
    net_plan = plan['inputs']['net_plan']
    _, full_plan, saved, cohorts, masks, _ = net.inspect(net_plan)
    prior = json.loads(Path(plan['inputs']['net_result']['path']).read_text())
    prior_terminal = json.loads(Path(plan['inputs']['net_terminal']['path']).read_text())
    gc_result = json.loads(Path(plan['inputs']['gc_result']['path']).read_text())
    gc_terminal = json.loads(Path(plan['inputs']['gc_terminal']['path']).read_text())
    need(prior['status'] == prior_terminal['status'] == 'COMPLETE'
         and prior_terminal['result'] == plan['inputs']['net_result']
         and gc_result['status'] == gc_terminal['status'] == 'COMPLETE'
         and gc_terminal['result'] == plan['inputs']['gc_result']
         and gc_result['test_data_used'] is False
         and prior['test_data_used'] is False, 'PRIOR_TERMINALS')
    meta = gc._source_metadata()
    return plan, full_plan, saved, cohorts, masks, {'net': prior, 'gc': gc_result,
                                                     'gc_meta': meta}


def check_policy(plan_ref: dict) -> dict:
    policy_ref = ref(REPO / 'NEXT_RUN_POLICY.json')
    policy = json.loads(Path(policy_ref['path']).read_text())
    gate = policy.get(STAGE, {})
    need(gate.get('status') == 'ADMITTED'
         and gate.get('plan') == plan_ref
         and gate.get('worker') == ref(REPO / SOURCE_PATHS[0])
         and gate.get('output_root') == str(ROOT)
         and gate.get('max_invocations') == 1
         and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
         and gate.get('execution_authorized') is True
         and gate.get('data_reads_authorized') is True
         and gate.get('relaunch_allowed') is False
         and gate.get('memory_max') == '4G'
         and gate.get('swap_max') == '512M'
         and gate.get('wall_seconds') == 1800
         and gate.get('model_forwards_authorized') is False
         and gate.get('optimizer_steps_authorized') is False
         and gate.get('checkpoint_reads_authorized') is False
         and gate.get('broker_access_allowed') is False
         and gate.get('training_authorized') is False
         and gate.get('test_data_used') is False
         and policy.get('training_enabled') is False
         and policy.get('native_training_enabled') is False
         and policy.get('native_launch_authorized') is False,
         'POLICY')
    return policy_ref


def _price_inputs(meta: dict) -> tuple[dict, dict]:
    split = gc._read_json(gc.REFS['split_manifest'], 64 * 1024**2)
    names = split['feature_contract']['ctx_cont_names']
    need(all(names[col] == PRICE_NAMES[name] for name, col in PRICE_COLUMNS.items()),
         'PRICE_FIELD_NAMES')
    before = gc._file_stat(gc.CURRENT_MMAP)
    need(before == meta['current_stat'], 'PRICE_MMAP_BEFORE')
    backing = np.memmap(gc.CURRENT_MMAP, mode='r', dtype='<f4', shape=(118249, 2542))
    price = {name: np.array(backing[:, col], dtype=np.float32, copy=True)
             for name, col in PRICE_COLUMNS.items()}
    del backing
    need(gc._file_stat(gc.CURRENT_MMAP) == before, 'PRICE_MMAP_AFTER')
    ema = price['ema5_distance']
    ema_valid = np.isfinite(ema)
    swing_present = price['swing_impulse_present']
    need(np.isfinite(swing_present).all()
         and np.isin(swing_present, (0.0, 1.0)).all(), 'SWING_PRESENT')
    high, low = price['swing_high_age'], price['swing_low_age']
    swing_valid = (swing_present == 1.0) & np.isfinite(high) & np.isfinite(low)
    need(np.all(high[swing_valid] >= 0) and np.all(low[swing_valid] >= 0)
         and np.all(high[swing_valid] != low[swing_valid]), 'SWING_AGE')
    ema_sign = np.zeros(len(ema), dtype=np.int8)
    swing_sign = np.zeros(len(ema), dtype=np.int8)
    ema_sign[ema_valid] = np.sign(ema[ema_valid]).astype(np.int8)
    # A later confirmed high than low yields positive age_low-age_high.
    swing_sign[swing_valid] = np.sign(low[swing_valid] - high[swing_valid]).astype(np.int8)
    return {'ema': ema_sign, 'ema_valid': ema_valid,
            'swing': swing_sign, 'swing_valid': swing_valid}, {
                'ema_observed': int(ema_valid.sum()),
                'confirmed_swing_order_observed': int(swing_valid.sum()),
                'price_columns': PRICE_COLUMNS,
                'source_mmap_stat': before}


def _gc_inputs(meta: dict) -> tuple[dict, dict]:
    values, hashes = gc._extract_fields(meta)
    parity = gc._flow_parity(values)
    names = ('gc_m5_observed_buy_known', 'gc_m5_observed_sell_known',
             'gc_m5_observed_unknown')
    present = np.logical_and.reduce([values[name + '_observed'] for name in names])
    buy, sell, unknown = (values[name].astype(np.float64) for name in names)
    total = buy + sell + unknown
    valid = present & np.isfinite(total) & (total > 0)
    pressure = np.zeros(len(total), np.float64)
    pressure[valid] = (buy[valid] - sell[valid]) / total[valid]
    need(np.isfinite(pressure[valid]).all() and np.all(np.abs(pressure[valid]) <= 1),
         'GC_PRESSURE')
    sign = np.sign(pressure).astype(np.int8)
    return {'gc': sign, 'gc_valid': valid, 'gc_raw': pressure}, {
        'field_hashes': hashes, 'trade_arithmetic': parity,
        'pressure_observed': int(valid.sum())}


def _signed_tape(full_plan: dict, saved: dict, cohorts: dict,
                 masks: dict, prior: dict) -> tuple[dict, dict, dict]:
    schedule, authority = base.read_closure(full_plan)
    m1, m5, issues, m5_source_rows, stats = base.load_sources(full_plan, authority)
    need(np.array_equal(issues, saved['issue_ns'])
         and np.array_equal(m5_source_rows, saved['m5_source_row']), 'ISSUE_AXIS')
    clock = state_target._clock(m1, state_target.EXIT_DECISION_BAR_SECONDS)
    validity = clock.asi8 + base.MINUTE_NS
    known, _ = state_target._known_closures(
        clock, full_plan['m1']['source'], authority, schedule)
    unknown_gap = gray.unknown_gap_mask(validity, known)
    need(int(unknown_gap.sum()) == authority['unknown_source_gap_count'], 'GAPS')
    ohlc = state_target._ohlc(m1)
    atr = state_target.build_causal_tf_atr(m5, cutoff_ns=pd.Timestamp(base.TRAIN_END).value)
    shape = (len(issues), len(state_target.TIMEFRAMES))
    sign = np.zeros(shape, dtype=np.int8)
    magnitude = np.zeros(shape, dtype=np.float64)
    valid = saved['complete_range_atr_valid'].copy()
    need(valid.shape == shape and masks['allowed_set'].shape == shape, 'TAPE_SHAPE')
    for col, tf in enumerate(state_target.TIMEFRAMES):
        chosen = np.flatnonzero(valid[:, col])
        source = saved['m1_source_row'][chosen, col]
        rows = source - gray.SOURCE_ROW_OFFSET
        starts = gray.require_endpoint_and_window(
            issues[chosen], rows, source, validity, base.LEADS[col],
            state_target.LOCAL_WINDOW_M1_BARS[tf], unknown_gap)
        net_move = ohlc['close'][rows] - ohlc['open'][starts]
        need(np.isfinite(net_move).all(), 'NET:' + tf)
        surface = atr['surfaces'][tf]
        at = np.searchsorted(surface['available_ns'], validity[rows], side='right') - 1
        need(np.all(at >= 0), 'ATR_INDEX:' + tf)
        scale = surface['atr'][at]
        need(np.isfinite(scale).all() and np.all(scale > 0), 'ATR_SCALE:' + tf)
        physical_range = gray.raw_window_range(
            ohlc['open'], ohlc['high'], ohlc['low'], ohlc['close'],
            unknown_gap, state_target.LOCAL_WINDOW_M1_BARS[tf])[rows]
        gray.require_ratio_parity(physical_range, scale,
                                  saved['complete_range_atr'][chosen, col])
        sign[chosen, col] = np.sign(net_move).astype(np.int8)
        magnitude[chosen, col] = np.abs(net_move) / scale
        for cohort in gray.COHORTS:
            selected = cohorts[cohort][chosen, col]
            counts = [int((sign[chosen[selected], col] == s).sum()) for s in (-1, 0, 1)]
            need(counts == prior['counts'][tf][cohort]['net_down_zero_up'],
                 'PRIOR_SIGN:' + tf + ':' + cohort)
    need(np.all(magnitude[valid] >= 0)
         and np.array_equal(sign[valid] == 0, magnitude[valid] == 0),
         'MAGNITUDE_SIGN')
    for path, old_stat in stats.items():
        need(base.regular(Path(path)) == old_stat, 'RAW_BOOKEND')
    tape = {'issue_ns': issues, 'target_valid': valid,
            'signed_net': sign, 'abs_net_over_atr': magnitude}
    info = {'m1_rows': len(validity), 'm5_issues': len(issues),
            'target_valid_by_tf': {tf: int(valid[:, j].sum())
                                   for j, tf in enumerate(state_target.TIMEFRAMES)},
            'target_nonzero_by_tf': {tf: int((valid[:, j] & (sign[:, j] != 0)).sum())
                                     for j, tf in enumerate(state_target.TIMEFRAMES)},
            'prior_15_cohort_sign_partitions_matched': True,
            'ratio_parity_cells': int(valid.sum())}
    return tape, info, stats


def _measure(full_plan: dict, saved: dict, cohorts: dict,
             masks: dict, prior: dict) -> tuple[dict, dict, dict]:
    gc_meta = prior['gc_meta']
    price, price_info = _price_inputs(gc_meta)
    flow, flow_info = _gc_inputs(gc_meta)
    tape, tape_info, stats = _signed_tape(full_plan, saved, cohorts, masks,
                                          prior['net'])
    bound = binding.load_bound_five_endpoint_targets()
    parent = bound.parent_rows
    decision = bound.decision_ns
    gc._axis_identity(parent, decision, gc_meta['provider'], gc_meta['registry'])
    at = np.searchsorted(tape['issue_ns'], decision)
    need(len(parent) == 118249 and np.all(at < len(tape['issue_ns']))
         and np.array_equal(tape['issue_ns'][at], decision), 'MODEL_ISSUE_JOIN')
    folds = []
    for fid in range(1, 6):
        axis = bound.mature_oof_axes(fold_id=fid)
        frow, srow = axis['fit_parent_rows'], axis['score_parent_rows']
        fissue, sissue = axis['fit_decision_ns'], axis['score_decision_ns']
        need(np.array_equal(decision[frow], fissue)
             and np.array_equal(decision[srow], sissue), 'FOLD_AXIS')
        tf_report = {}
        for j, tf in enumerate(state_target.TIMEFRAMES):
            fidx, sidx = at[frow], at[srow]
            fit_valid = tape['target_valid'][fidx, j] & (
                fissue + base.LEADS[j] * base.MINUTE_NS < binding.FOLD_BOUNDARIES_NS[fid - 1])
            score_valid = tape['target_valid'][sidx, j] & (
                sissue + base.LEADS[j] * base.MINUTE_NS < binding.FOLD_BOUNDARIES_NS[fid])
            tf_report[tf] = core.score_fold(
                fit_target=tape['signed_net'][fidx, j], fit_target_valid=fit_valid,
                fit_ema=price['ema'][frow], fit_ema_valid=price['ema_valid'][frow],
                fit_swing=price['swing'][frow], fit_swing_valid=price['swing_valid'][frow],
                fit_gc=flow['gc'][frow], fit_gc_valid=flow['gc_valid'][frow],
                score_target=tape['signed_net'][sidx, j], score_target_valid=score_valid,
                score_magnitude_atr=tape['abs_net_over_atr'][sidx, j],
                score_issues=sissue,
                score_ema=price['ema'][srow], score_ema_valid=price['ema_valid'][srow],
                score_swing=price['swing'][srow], score_swing_valid=price['swing_valid'][srow],
                score_gc=flow['gc'][srow], score_gc_valid=flow['gc_valid'][srow],
                score_gc_raw=flow['gc_raw'][srow])
        folds.append({'fold_id': fid, 'fit_rows': len(frow), 'score_rows': len(srow),
                      'fit_parent_rows_sha256': axis['fit_parent_rows_sha256'],
                      'score_parent_rows_sha256': axis['score_parent_rows_sha256'],
                      'timeframes': tf_report})
    need(gc._file_stat(gc.CURRENT_MMAP) == gc_meta['current_stat']
         and gc._file_stat(gc.SOURCE_MMAP) == gc_meta['source_stat'],
         'MMAP_BOOKEND')
    return tape, {'core_contract': core.contract(), 'tape_info': tape_info,
                  'price_info': price_info, 'gc_info': flow_info,
                  'folds': folds, 'model_forwards': 0, 'optimizer_steps': 0,
                  'test_data_used': False}, stats


def run(plan_ref: dict) -> dict:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    start_state = current_state()
    plan, full_plan, saved, cohorts, masks, prior = inspect(plan_ref)
    policy_ref = check_policy(plan_ref)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and cap['pids'] == 64
         and cap['numerical_threads'] == 1
         and lock['scope'] == cap['cgroup_path'][len('/sys/fs/cgroup'):], 'CAP')
    need(ROOT.is_dir() and {p.name for p in ROOT.iterdir()} == {'CLAIM.json'},
         'OUTPUT_CLAIM_ONLY')
    claim_ref = ref(ROOT / 'CLAIM.json')
    claim = json.loads(Path(claim_ref['path']).read_text())
    need(claim == {'schema_version': 'gx1_market_state_signed_net_gc_claim_v1',
                   'plan': plan_ref, 'policy': policy_ref, 'source_head': start_state[0],
                   'worker': ref(REPO / SOURCE_PATHS[0]), 'test_data_used': False},
         'CLAIM')
    started = time.monotonic()
    tape, data, stats = _measure(full_plan, saved, cohorts, masks, prior)
    need(current_state() == start_state and ref(PLAN_PATH) == plan_ref
         and ref(Path(policy_ref['path'])) == policy_ref
         and ref(ROOT / 'CLAIM.json') == claim_ref
         and {p.name for p in ROOT.iterdir()} == {'CLAIM.json'}, 'FINAL_SOURCE')
    for path, old_stat in stats.items():
        need(base.regular(Path(path)) == old_stat, 'FINAL_RAW_STAT')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **tape)
    tape_bytes = buffer.getvalue()
    need(len(tape_bytes) < 64 * 1024**2, 'TAPE_SIZE')
    _publish_bytes(ROOT / TAPE_NAME, tape_bytes)
    tape_ref = {'path': str(ROOT / TAPE_NAME),
                'sha256': hashlib.sha256(tape_bytes).hexdigest()}
    result = {'schema_version': 'gx1_market_state_signed_net_gc_result_v1',
              'status': 'COMPLETE', 'decision': 'TARGET_AND_CAUSAL_REFERENCE_ONLY',
              'plan': plan_ref, 'policy': policy_ref, 'claim': claim_ref,
              'source_head': start_state[0], 'inputs': plan['inputs'],
              'source_owners': plan['source_owners'], 'target_tape': tape_ref,
              **data, 'elapsed_seconds': time.monotonic() - started,
              'cap': {'execution': cap, 'lock': lock}}
    _publish_bytes(ROOT / 'RESULT.json', base.encode(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    plan_ref = {'path': args.plan, 'sha256': args.plan_sha256}
    if args.preflight:
        need(not ROOT.exists(), 'PREFLIGHT_OUTPUT_EXISTS')
        plan, _, saved, _, _, _ = inspect(plan_ref)
        check_policy(plan_ref)
        print(json.dumps({'preflight': 'PASS_NO_CLAIM_NO_RAW_READ',
                          'm5_issues': len(saved['issue_ns']),
                          'source_owners': len(plan['source_owners'])}, sort_keys=True))
    else:
        run(plan_ref)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
