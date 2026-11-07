"""One bounded TRAIN-only comparison of observed local net and state side.

Reuses sealed gray-zone/source readers. No new target, fit, model, or TEST.
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

from gx1.audit import build_full_train_market_state_endpoint_candidate_v1 as base
from gx1.audit import verify_market_state_gray_physical_v1 as gray
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = base.REPO
STAGE = 'market_state_net_side_review_20261107_001'
PLAN_PATH = REPO / 'configs/research/MARKET_STATE_NET_SIDE_REVIEW_20261107_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_NET_SIDE_REVIEW_20261107_001')
SOURCE_PATHS = {
    'worker': 'gx1/audit/review_market_state_net_side_v1.py',
    'gray_worker': 'gx1/audit/verify_market_state_gray_physical_v1.py',
    'source_loader': 'gx1/audit/build_full_train_market_state_endpoint_candidate_v1.py',
    'target': 'gx1/contracts/entry_dynamic_mtf_state_targets_v1.py',
    'cap': 'gx1/contracts/gx1_capped_execution_v1.py',
    'publish': 'gx1/contracts/immutable_event_authority_v1.py',
    'capped_wrapper': 'scripts/gx1_capped_run.sh',
}
REF_PATHS = {
    'gray_plan': REPO / 'configs/research/MARKET_STATE_GRAY_PHYSICAL_20261107_001.json',
    'gray_result': Path('/home/andre2/GX1_RUNS/MARKET_STATE_GRAY_PHYSICAL_20261107_001/RESULT.json'),
    'gray_terminal': Path('/home/andre2/GX1_RUNS/MARKET_STATE_GRAY_PHYSICAL_20261107_001/TERMINAL.json'),
    'selected_result': Path('/home/andre2/GX1_RUNS/MARKET_STATE_LOCAL_SIDE_EXCEPTIONS_20261107_001/RESULT.json'),
    'selected_terminal': Path('/home/andre2/GX1_RUNS/MARKET_STATE_LOCAL_SIDE_EXCEPTIONS_20261107_001/TERMINAL.json'),
    'selected_masks': Path('/home/andre2/GX1_RUNS/MARKET_STATE_LOCAL_SIDE_EXCEPTIONS_20261107_001/ROW_MASKS.npz'),
}
KEYS = {'activity_active', 'activity_observed', 'allowed_set', 'issue_ns',
        'lead_minutes', 'm1_source_row', 'm5_source_row', 'side_observed',
        'side_up', 'singleton_observed'}


def need(ok: object, code: str) -> None:
    if not bool(ok):
        raise RuntimeError('MARKET_STATE_NET_SIDE_' + code)


def ref(path: Path) -> dict:
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def state() -> tuple[str, str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(('git', *args), cwd=REPO, text=True).strip()
    return git('rev-parse', 'HEAD'), git('branch', '--show-current'), git('status', '--porcelain')


def summarize(sign: np.ndarray, bits: np.ndarray, crossings: np.ndarray) -> dict:
    """Separate observable net sign from certified local state side."""
    need(sign.dtype == np.int8 and bits.dtype == np.uint8
         and crossings.dtype == np.int32
         and sign.ndim == bits.ndim == crossings.ndim == 1
         and sign.shape == bits.shape == crossings.shape
         and np.all(crossings >= 0)
         and np.isin(sign, (-1, 0, 1)).all()
         and np.isin(bits, (1, 2, 4, 5, 7)).all(), 'SUMMARY_DOMAIN')
    known = np.isin(bits, (1, 4))
    same = known & (((bits == 1) & (sign == -1)) | ((bits == 4) & (sign == 1)))
    opposite = known & (((bits == 1) & (sign == 1)) | ((bits == 4) & (sign == -1)))
    return {'n': int(len(sign)),
            'net_down_zero_up': [int((sign == -1).sum()), int((sign == 0).sum()),
                                 int((sign == 1).sum())],
            'certified_side': int(known.sum()),
            'certified_same_net': int(same.sum()),
            'certified_opposite_net': int(opposite.sum()),
            'certified_zero_net': int((known & (sign == 0)).sum()),
            'start_price_close_crossing_rows': int((crossings > 0).sum()),
            'certified_opposite_with_start_crossing': int((opposite & (crossings > 0)).sum()),
            'active_side_unknown': int((bits == 5).sum()),
            'flat': int((bits == 2).sum()), 'all_unknown': int((bits == 7).sum())}


def inspect(plan_ref: dict) -> tuple[dict, dict, dict, dict, dict, dict]:
    need(plan_ref['path'] == str(PLAN_PATH), 'PLAN_PATH')
    plan = base.read_json(plan_ref)
    need(set(plan) == {'schema_version', 'stage', 'status', 'inputs',
                       'source_owners', 'scope', 'limits', 'output_dir'}
         and plan['schema_version'] == 'gx1_market_state_net_side_review_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['output_dir'] == str(ROOT)
         and plan['scope'] == {'train_start_utc': base.TRAIN_START,
                               'train_end_utc_exclusive': base.TRAIN_END,
                               'timeframes': list(target.TIMEFRAMES),
                               'm5_issues': 118706, 'test_data_used': False}
         and plan['limits'] == {'memory_max': '4G', 'swap_max': '512M',
                                'wall_seconds': 1800, 'model_forwards': 0,
                                'optimizer_steps': 0}, 'PLAN')
    need(set(plan['inputs']) == set(REF_PATHS)
         and set(plan['source_owners']) == set(SOURCE_PATHS), 'PLAN_KEYS')
    for key, path in REF_PATHS.items():
        need(plan['inputs'][key]['path'] == str(path), 'INPUT_PATH_' + key)
        base.checked(plan['inputs'][key], max_bytes=64 * 1024 * 1024)
    for key, rel in SOURCE_PATHS.items():
        item = plan['source_owners'][key]
        need(item['path'] == str(REPO / rel), 'OWNER_PATH_' + key)
        base.checked(item, max_bytes=4 * 1024 * 1024)
    old_plan = base.read_json(plan['inputs']['gray_plan'])
    gray_result = base.read_json(plan['inputs']['gray_result'])
    gray_terminal = base.read_json(plan['inputs']['gray_terminal'])
    need(old_plan['stage'] == gray.STAGE and old_plan['inputs'] == gray.INPUTS
         and old_plan['output_dir'] == str(gray.FULL_ROOT.parent / gray.STAGE.upper())
         and old_plan['test_data_used'] is False
         and gray_result['plan'] == plan['inputs']['gray_plan']
         and gray_terminal['result'] == plan['inputs']['gray_result']
         and gray_terminal['status'] == 'COMPLETE'
         and gray_terminal['exit_code'] == 0
         and gray_terminal['test_data_used'] is False, 'GRAY_RECEIPT')
    # Historical cap-owner bytes are checked at the producer commit; today's
    # cap-owner bytes are independently pinned by this plan and checked at run.
    old_cap = old_plan['source_owners']['capped_owner']
    old_claim = base.read_json(gray_result['claim'])
    need(old_cap['path'] == str(REPO / SOURCE_PATHS['cap'])
         and gray_result['claim'] == gray_terminal['claim']
         and old_claim['source_head'] == gray_result['source_head'],
         'HISTORICAL_CAP_REF')
    old_bytes = subprocess.check_output(
        ('git', 'show', gray_result['source_head'] + ':' + SOURCE_PATHS['cap']),
        cwd=REPO)
    need(hashlib.sha256(old_bytes).hexdigest() == old_cap['sha256'],
         'HISTORICAL_CAP_BYTES')
    full_plan = base.read_json(gray.INPUTS['full_plan'])
    single_result = base.read_json(gray.INPUTS['single_result'])
    _, saved, cohorts = gray.load_masks(single_result)
    selected_result = base.read_json(plan['inputs']['selected_result'])
    terminal = base.read_json(plan['inputs']['selected_terminal'])
    need(gray_result['status'] == selected_result['status'] == terminal['status'] == 'COMPLETE'
         and gray_result['test_data_used'] is False
         and selected_result['test_data_used'] is False
         and terminal['test_data_used'] is False and terminal['exit_code'] == 0
         and gray_result['model_forwards'] == gray_result['optimizer_steps'] == 0
         and selected_result['model_forwards'] == selected_result['optimizer_steps'] == 0
         and selected_result['row_masks'] == plan['inputs']['selected_masks']
         and terminal['row_masks'] == plan['inputs']['selected_masks']
         and terminal['result'] == plan['inputs']['selected_result']
         and selected_result['changed_cells'] == 7, 'PRIOR_RECEIPTS')
    raw, _ = base.checked(plan['inputs']['selected_masks'], max_bytes=64 * 1024 * 1024)
    need(raw is not None, 'MASK_BYTES')
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        need(set(z.files) == KEYS, 'MASK_KEYS')
        masks = {k: np.array(z[k]) for k in KEYS}
    bits = masks['allowed_set']
    need(bits.dtype == np.uint8 and bits.shape == (118706, 5)
         and np.array_equal(masks['issue_ns'], saved['issue_ns'])
         and np.array_equal(masks['m1_source_row'], saved['m1_source_row'])
         and np.array_equal(masks['m5_source_row'], saved['m5_source_row'])
         and np.array_equal(masks['lead_minutes'], saved['lead_minutes'])
         and np.isin(bits, (1, 2, 4, 5, 7)).all()
         and np.all(bits[cohorts['gray_0_5_to_1_0']] == 5)
         and np.all(bits[cohorts['below_0_5']] == 2)
         and np.array_equal(masks['side_observed'], np.isin(bits, (1, 4)))
         and np.array_equal(masks['side_up'], bits == 4)
         and np.array_equal(masks['activity_observed'], np.isin(bits, (1, 2, 4, 5)))
         and np.array_equal(masks['activity_active'], np.isin(bits, (1, 4, 5)))
         and np.array_equal(masks['singleton_observed'], np.isin(bits, (1, 2, 4))),
         'SELECTED_MASK_PARITY')
    return plan, full_plan, saved, cohorts, masks, gray_result


def check_policy(plan_ref: dict, plan: dict) -> tuple[dict, dict]:
    policy_ref = ref(REPO / 'NEXT_RUN_POLICY.json')
    policy = base.read_json(policy_ref)
    gate = policy.get(STAGE)
    need(isinstance(gate, dict) and gate.get('status') == 'ADMITTED'
         and gate.get('plan') == plan_ref
         and gate.get('worker') == plan['source_owners']['worker']
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
    return policy_ref, gate


def run(plan_ref: dict) -> dict:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    original = state()
    need(original[1:] == ('work/gx1-current', ''), 'SOURCE_TREE')
    plan, full_plan, saved, cohorts, masks, prior = inspect(plan_ref)
    policy_ref, _ = check_policy(plan_ref, plan)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and cap['pids'] == 64
         and cap['numerical_threads'] == 1
         and lock['scope'] == cap['cgroup_path'][len('/sys/fs/cgroup'):], 'CAP')
    need(ROOT.is_dir() and not ROOT.is_symlink()
         and {p.name for p in ROOT.iterdir()} == {'CLAIM.json'}, 'OUTPUT')
    claim_ref = ref(ROOT / 'CLAIM.json')
    claim = base.read_json(claim_ref)
    need(claim == {'schema_version': 'gx1_market_state_net_side_review_claim_v1',
                   'stage': STAGE, 'plan': plan_ref, 'policy': policy_ref,
                   'worker': plan['source_owners']['worker'],
                   'source_head': original[0], 'test_data_used': False}, 'CLAIM')
    started = time.monotonic()
    schedule, authority = base.read_closure(full_plan)
    m1, _, issues, m5_source_rows, stats = base.load_sources(full_plan, authority)
    need(np.array_equal(issues, saved['issue_ns'])
         and np.array_equal(m5_source_rows, saved['m5_source_row']), 'ISSUE_AXIS')
    clock = target._clock(m1, target.EXIT_DECISION_BAR_SECONDS)
    validity = clock.asi8 + base.MINUTE_NS
    known, _ = target._known_closures(clock, full_plan['m1']['source'], authority, schedule)
    unknown = gray.unknown_gap_mask(validity, known)
    need(int(unknown.sum()) == authority['unknown_source_gap_count'], 'GAPS')
    ohlc = target._ohlc(m1)
    opening, closing = ohlc['open'], ohlc['close']
    report = {}
    for col, tf in enumerate(target.TIMEFRAMES):
        complete = saved['complete_range_atr_valid'][:, col]
        chosen = np.flatnonzero(complete)
        source = saved['m1_source_row'][chosen, col]
        rows = source - gray.SOURCE_ROW_OFFSET
        starts = gray.require_endpoint_and_window(
            issues[chosen], rows, source, validity,
            base.LEADS[col], target.LOCAL_WINDOW_M1_BARS[tf], unknown)
        net = closing[rows] - opening[starts]
        need(np.isfinite(net).all(), 'NET_' + tf)
        sign = np.sign(net).astype(np.int8)
        crossings, _ = gray.close_diagnostics(
            closing, starts, rows, opening[starts], net,
            target.LOCAL_WINDOW_M1_BARS[tf])
        bits = masks['allowed_set'][chosen, col]
        report[tf] = {}
        for cohort in gray.COHORTS:
            selected = cohorts[cohort][chosen, col]
            item = summarize(sign[selected], bits[selected], crossings[selected])
            need(item['net_down_zero_up'] == prior['total'][tf][cohort]['net_down_zero_up']
                 and item['n'] == prior['total'][tf][cohort]['n'],
                 'PRIOR_NET_' + tf + '_' + cohort)
            report[tf][cohort] = item
    need(state() == original and ROOT.is_dir()
         and {p.name for p in ROOT.iterdir()} == {'CLAIM.json'}
         and ref(ROOT / 'CLAIM.json') == claim_ref, 'FINAL_SOURCE')
    for item in plan['inputs'].values():
        base.checked(item, max_bytes=64 * 1024 * 1024)
    for path, stat in stats.items():
        need(base.regular(Path(path)) == stat, 'FINAL_RAW_STAT')
    result = {'schema_version': 'gx1_market_state_net_side_review_result_v1',
              'status': 'COMPLETE', 'decision': 'TARGET_COMPARISON_ONLY_NO_FIT',
              'plan': plan_ref, 'policy': policy_ref,
              'claim': claim_ref, 'source_head': original[0],
              'inputs': plan['inputs'], 'source_owners': plan['source_owners'],
              'counts': report, 'm1_rows': len(validity), 'm5_issues': len(issues),
              'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False,
              'elapsed_seconds': time.monotonic() - started,
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
        plan, _, saved, cohorts, _, _ = inspect(plan_ref)
        check_policy(plan_ref, plan)
        print(json.dumps({'preflight': 'PASS_NO_CLAIM_NO_RAW_READ',
                          'm5_issues': len(saved['issue_ns']),
                          'gray_cells': int(cohorts['gray_0_5_to_1_0'].sum())}, sort_keys=True))
    else:
        run(plan_ref)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
