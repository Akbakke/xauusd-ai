"""One capped TRAIN audit of decision-anchored future XAU return and GC flow.

Uses exact closed M5 prices at t and t+native lead. No M1 raw reread, model,
optimizer, TEST, Entry, Exit, broker or trading authority.
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
import pyarrow.parquet as pq

from gx1.audit import build_full_train_market_state_endpoint_candidate_v1 as base
from gx1.audit import market_state_signed_net_reference_core_v1 as math_core
from gx1.audit import measure_market_state_signed_net_gc_reference_v1 as prior
from gx1.audit import score_market_state_gc_flow_reference_v1 as gc
from gx1.audit import verify_market_state_gray_physical_v1 as gray
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as state_target
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as binding
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = base.REPO
STAGE = 'market_state_decision_return_gc_20261108_002'
PLAN_PATH = REPO / 'configs/research/MARKET_STATE_DECISION_RETURN_GC_20261108_002.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_DECISION_RETURN_GC_20261108_002')
TAPE_NAME = 'DECISION_RETURN_TAPE.npz'
SOURCE_PATHS = (
    'gx1/audit/measure_market_state_decision_return_gc_v1.py',
    'gx1/audit/measure_market_state_signed_net_gc_reference_v1.py',
    'gx1/audit/market_state_signed_net_reference_core_v1.py',
    'gx1/audit/market_state_m5_gc_flow_reference_core_v1.py',
    'gx1/audit/score_market_state_gc_flow_reference_v1.py',
    'gx1/audit/build_full_train_market_state_endpoint_candidate_v1.py',
    'gx1/audit/verify_market_state_gray_physical_v1.py',
    'gx1/contracts/entry_dynamic_mtf_state_targets_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/features/entry_specialist_feature_groups_v1.py',
    'gx1/features/swing_structure_v1.py',
    'scripts/gx1_capped_run.sh',
)
INPUT_PATHS = {
    'full_train_plan': Path(gray.INPUTS['full_plan']['path']),
    'local_reference_result': prior.ROOT / 'RESULT.json',
    'local_reference_terminal': prior.ROOT / 'TERMINAL.json',
    'gc_reference_result': gc.OUT_ROOT / 'RESULT.json',
}
LIMITS = {'memory_max': '4G', 'swap_max': '512M', 'wall_seconds': 1800,
          'model_forwards': 0, 'optimizer_steps': 0}
LEADS = tuple(int(x) for x in base.LEADS)


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('MARKET_STATE_DECISION_RETURN_GC_' + reason)


def ref(path: Path) -> dict:
    return prior.ref(path)


def state() -> tuple[str, str, str]:
    return prior.current_state()


def inspect(plan_ref: dict) -> tuple[dict, dict, dict]:
    head, branch, dirty = state()
    need(branch == 'work/gx1-current' and not dirty
         and plan_ref == ref(PLAN_PATH), 'SOURCE_AND_PLAN')
    plan = json.loads(PLAN_PATH.read_text())
    need(set(plan) == {'schema_version', 'stage', 'status', 'source_base_head',
                       'source_owners', 'inputs', 'scope', 'limits', 'output_root'}
         and plan['schema_version'] == 'gx1_market_state_decision_return_gc_plan_v1'
         and plan['stage'] == STAGE
         and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['output_root'] == str(ROOT)
         and plan['limits'] == LIMITS
         and plan['scope'] == {'timeframes': list(state_target.TIMEFRAMES),
                               'native_lead_minutes': list(LEADS),
                               'target': 'close_XAU_M5_at_exact_v_minus_close_XAU_M5_at_t',
                               'magnitude_scale': 'last_completed_native_TF_ATR_at_t',
                               'fold_ids': [1, 2, 3, 4, 5],
                               'math_core_sha256': math_core.contract()['contract_sha256'],
                               'test_data_used': False}
         and subprocess.run(('git', 'merge-base', '--is-ancestor',
                             plan['source_base_head'], head), cwd=REPO,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0,
         'PLAN_SCOPE')
    need(set(plan['source_owners']) == set(SOURCE_PATHS)
         and all(ref(REPO / name)['sha256'] == plan['source_owners'][name]
                 for name in SOURCE_PATHS), 'SOURCE_OWNERS')
    need(set(plan['inputs']) == set(INPUT_PATHS)
         and all(plan['inputs'][key] == ref(path)
                 for key, path in INPUT_PATHS.items()), 'INPUT_REFS')
    prior_result = json.loads(Path(plan['inputs']['local_reference_result']['path']).read_text())
    terminal = json.loads(Path(plan['inputs']['local_reference_terminal']['path']).read_text())
    gc_result = json.loads(Path(plan['inputs']['gc_reference_result']['path']).read_text())
    need(prior_result['status'] == terminal['status'] == gc_result['status'] == 'COMPLETE'
         and terminal['result'] == plan['inputs']['local_reference_result']
         and prior_result['test_data_used'] is False
         and gc_result['test_data_used'] is False, 'PRIOR_RECEIPTS')
    full_plan = base.read_json(plan['inputs']['full_train_plan'])
    need(full_plan['m5']['source']['path']
         and full_plan['m5']['manifest']['path'], 'M5_BINDING')
    meta = gc._source_metadata()
    return plan, full_plan, meta


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
         and policy.get('native_launch_authorized') is False, 'POLICY')
    return policy_ref


def exact_future_close_move(issues: np.ndarray, close: np.ndarray,
                            lead_minutes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use only the closed decision price and exact later M5 close."""
    need(issues.dtype == np.int64 and issues.ndim == 1 and len(issues) > 0
         and np.all(np.diff(issues) > 0) and np.all(issues % (5 * base.MINUTE_NS) == 0)
         and close.dtype == np.float64 and close.shape == issues.shape
         and np.isfinite(close).all() and np.all(close > 0)
         and type(lead_minutes) is int and lead_minutes > 0
         and lead_minutes % 5 == 0, 'FUTURE_CLOSE_INPUT')
    v = issues + lead_minutes * base.MINUTE_NS
    need(np.all(v > issues), 'FUTURE_CLOCK')
    at = np.searchsorted(issues, v)
    present = at < len(issues)
    present[present] &= issues[at[present]] == v[present]
    move = np.zeros(len(issues), np.float64)
    gap = np.zeros(len(issues), bool)
    where = np.flatnonzero(present)
    move[where] = close[at[where]] - close[where]
    gap[where] = at[where] - where != lead_minutes // 5
    return present, move, gap


def _future_tape(full_plan: dict) -> tuple[dict, dict, tuple[tuple[str, tuple], ...]]:
    m5_ref = full_plan['m5']['source']
    manifest = base.read_json(full_plan['m5']['manifest'])
    need(manifest['output_parquet'] == m5_ref['path']
         and manifest['output_parquet_sha256'] == m5_ref['sha256']
         and manifest['rows'] == 1140982
         and manifest['timeframe'] == 'M5', 'M5_MANIFEST')
    _, before = base.checked(m5_ref, max_bytes=None)
    parquet = pq.ParquetFile(m5_ref['path'])
    need(parquet.metadata.num_rows == 1140982
         and parquet.metadata.num_row_groups > base.M5_GROUPS[1]
         and str(parquet.schema_arrow.field('time').type) == 'timestamp[ns, tz=UTC]',
         'M5_HEADER')
    m5 = parquet.read_row_groups(list(range(*base.M5_GROUPS)),
                                 columns=['time', 'open', 'high', 'low', 'close'],
                                 use_threads=False).to_pandas()
    clock = state_target._clock(m5, 300)
    keep, issues, source_rows = base.bound_m5_issue_clock(
        np.asarray(clock.asi8, dtype=np.int64),
        pd.Timestamp(base.TRAIN_START).value,
        pd.Timestamp(base.TRAIN_END).value, np)
    bounded_clock = np.asarray(clock.asi8[keep], dtype=np.int64)
    m5 = m5.loc[keep].reset_index(drop=True)
    bounded_close = m5['close'].to_numpy(dtype=np.float64, copy=True)
    need(len(issues) == len(source_rows) == 118706
         and np.all(source_rows >= 0) and np.all(source_rows < len(bounded_close))
         and np.array_equal(bounded_clock[source_rows] + base.M5_NS, issues),
         'ISSUE_PRICE_AXIS')
    close = bounded_close[source_rows]
    need(np.isfinite(close).all() and np.all(close > 0)
         and np.all(np.diff(issues) > 0), 'ISSUE_PRICE_VALUES')
    atr = state_target.build_causal_tf_atr(m5,
                                           cutoff_ns=pd.Timestamp(base.TRAIN_END).value)
    shape = (len(issues), len(LEADS))
    sign = np.zeros(shape, np.int8)
    magnitude = np.zeros(shape, np.float64)
    valid = np.zeros(shape, bool)
    exact_endpoint = np.zeros(shape, bool)
    gap_between = np.zeros(shape, bool)
    for col, tf in enumerate(state_target.TIMEFRAMES):
        present, move, gap = exact_future_close_move(issues, close, LEADS[col])
        exact_endpoint[:, col] = present
        surface = atr['surfaces'][tf]
        atr_at = np.searchsorted(surface['available_ns'], issues, side='right') - 1
        scale = np.zeros(len(issues), np.float64)
        has_atr = atr_at >= 0
        scale[has_atr] = surface['atr'][atr_at[has_atr]]
        eligible = present & np.isfinite(scale) & (scale > 0)
        valid[:, col] = eligible
        where = np.flatnonzero(eligible)
        sign[where, col] = np.sign(move[where]).astype(np.int8)
        magnitude[where, col] = np.abs(move[where]) / scale[where]
        gap_between[where, col] = gap[where]
        need(np.array_equal(sign[where, col] == 0,
                            magnitude[where, col] == 0), 'SIGN_MAGNITUDE:' + tf)
    need(base.regular(Path(m5_ref['path'])) == before, 'M5_BOOKEND')
    tape = {'issue_ns': issues, 'target_valid': valid,
            'future_sign': sign, 'abs_future_over_atr_t': magnitude,
            'exact_endpoint': exact_endpoint, 'gap_between': gap_between}
    info = {'m5_issues': len(issues), 'source_m5_rows': manifest['rows'],
            'target_valid_by_tf': {tf: int(valid[:, j].sum())
                                   for j, tf in enumerate(state_target.TIMEFRAMES)},
            'exact_endpoint_by_tf': {tf: int(exact_endpoint[:, j].sum())
                                     for j, tf in enumerate(state_target.TIMEFRAMES)},
            'gap_between_by_tf': {tf: int(gap_between[:, j].sum())
                                  for j, tf in enumerate(state_target.TIMEFRAMES)},
            'target_nonzero_by_tf': {tf: int((valid[:, j] & (sign[:, j] != 0)).sum())
                                     for j, tf in enumerate(state_target.TIMEFRAMES)}}
    return tape, info, ((m5_ref['path'], before),)


def _measure(full_plan: dict, meta: dict) -> tuple[dict, dict, tuple]:
    price, price_info = prior._price_inputs(meta)
    flow, flow_info = prior._gc_inputs(meta)
    tape, info, stats = _future_tape(full_plan)
    bound = binding.load_bound_five_endpoint_targets()
    parent, decision = bound.parent_rows, bound.decision_ns
    gc._axis_identity(parent, decision, meta['provider'], meta['registry'])
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
            fvalid = tape['target_valid'][fidx, j] & (
                fissue + LEADS[j] * base.MINUTE_NS < binding.FOLD_BOUNDARIES_NS[fid - 1])
            svalid = tape['target_valid'][sidx, j] & (
                sissue + LEADS[j] * base.MINUTE_NS < binding.FOLD_BOUNDARIES_NS[fid])
            scored = math_core.score_fold(
                fit_target=tape['future_sign'][fidx, j], fit_target_valid=fvalid,
                fit_ema=price['ema'][frow], fit_ema_valid=price['ema_valid'][frow],
                fit_swing=price['swing'][frow], fit_swing_valid=price['swing_valid'][frow],
                fit_gc=flow['gc'][frow], fit_gc_valid=flow['gc_valid'][frow],
                score_target=tape['future_sign'][sidx, j], score_target_valid=svalid,
                score_magnitude_atr=tape['abs_future_over_atr_t'][sidx, j],
                score_issues=sissue,
                score_ema=price['ema'][srow], score_ema_valid=price['ema_valid'][srow],
                score_swing=price['swing'][srow], score_swing_valid=price['swing_valid'][srow],
                score_gc=flow['gc'][srow], score_gc_valid=flow['gc_valid'][srow],
                score_gc_raw=flow['gc_raw'][srow])
            del scored['schema_version'], scored['contract_sha256']
            scored['by_abs_F_over_ATR_t'] = scored.pop('by_abs_D_over_ATR')
            scored['score_exact_price_endpoints'] = int(tape['exact_endpoint'][sidx, j].sum())
            scored['score_gap_between'] = int((svalid & tape['gap_between'][sidx, j]).sum())
            tf_report[tf] = scored
        folds.append({'fold_id': fid, 'fit_rows': len(frow), 'score_rows': len(srow),
                      'fit_parent_rows_sha256': axis['fit_parent_rows_sha256'],
                      'score_parent_rows_sha256': axis['score_parent_rows_sha256'],
                      'timeframes': tf_report})
    need(gc._file_stat(gc.CURRENT_MMAP) == meta['current_stat']
         and gc._file_stat(gc.SOURCE_MMAP) == meta['source_stat'], 'MMAP_BOOKEND')
    return tape, {'target_definition': 'F_s(t,v)=M5_close(v)-M5_close(t)',
                  'target_price_time': 'exact_closed_M5_decision_clock',
                  'target_magnitude': 'abs_F_over_last_closed_native_TF_ATR_at_t',
                  'math_core_reused_sha256': math_core.contract()['contract_sha256'],
                  'tape_info': info, 'price_info': price_info, 'gc_info': flow_info,
                  'folds': folds, 'model_forwards': 0, 'optimizer_steps': 0,
                  'test_data_used': False}, stats


def run(plan_ref: dict) -> dict:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    beginning = state()
    plan, full_plan, meta = inspect(plan_ref)
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
    need(claim == {'schema_version': 'gx1_market_state_decision_return_gc_claim_v1',
                   'plan': plan_ref, 'policy': policy_ref,
                   'source_head': beginning[0], 'worker': ref(REPO / SOURCE_PATHS[0]),
                   'test_data_used': False}, 'CLAIM')
    started = time.monotonic()
    tape, data, stats = _measure(full_plan, meta)
    need(state() == beginning and ref(PLAN_PATH) == plan_ref
         and ref(Path(policy_ref['path'])) == policy_ref
         and ref(ROOT / 'CLAIM.json') == claim_ref
         and {p.name for p in ROOT.iterdir()} == {'CLAIM.json'}, 'FINAL_SOURCE')
    for path, old_stat in stats:
        need(base.regular(Path(path)) == old_stat, 'FINAL_M5_STAT')
    out = io.BytesIO()
    np.savez_compressed(out, **tape)
    encoded = out.getvalue()
    need(len(encoded) < 64 * 1024**2, 'TAPE_SIZE')
    _publish_bytes(ROOT / TAPE_NAME, encoded)
    tape_ref = {'path': str(ROOT / TAPE_NAME),
                'sha256': hashlib.sha256(encoded).hexdigest()}
    result = {'schema_version': 'gx1_market_state_decision_return_gc_result_v1',
              'status': 'COMPLETE', 'decision': 'FUTURE_DIRECTION_REFERENCE_ONLY',
              'plan': plan_ref, 'policy': policy_ref, 'claim': claim_ref,
              'source_head': beginning[0], 'inputs': plan['inputs'],
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
        plan, _, _ = inspect(plan_ref)
        check_policy(plan_ref)
        print(json.dumps({'preflight': 'PASS_NO_CLAIM_NO_PRICE_READ',
                          'source_owners': len(plan['source_owners'])}, sort_keys=True))
    else:
        run(plan_ref)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
