"""One-use, no-fit GC M5 observed mid-response screen on selected Fold 2 S/F.

Hypothesis fixed before scoring: positive observed GC M5 book mid-response
ranks UP above DOWN at each native S/F endpoint. The primary cell is S/M5.
Same-row XAU EMA5 distance is a price rank control, not incremental value.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit import measure_market_state_gc_book_tw_fold2_v1 as book
from gx1.audit import measure_market_state_fold2_exact_prior_later_v1 as prior
from gx1.audit import score_market_state_gc_flow_reference_v1 as source
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_future_displacement_candidate_v1 as future
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = book.REPO
POLICY = book.POLICY
STAGE = 'market_state_gc_mid_response_fold2_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_GC_MID_RESPONSE_FOLD2_20261115_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_GC_MID_RESPONSE_FOLD2_20261115_001')
FIELD = 'gc_m5_observed_book_mid_response_ticks'
FIELD_COLS = (822, 823)
CONTROL = book.CONTROL
CONTROL_COL = book.CONTROL_COL
EXPECTED_FIELD_HASH = 'b82a5fa4603979eab03d56389e927f8d528bfc37dd5fa25113de99b2b29ea8e5'
INPUTS = book.INPUTS
CODE = tuple(sorted(set(book.CODE) | {
    'gx1/audit/measure_market_state_gc_mid_response_fold2_v1.py',
}))


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('GC_MID_RESPONSE_FOLD2_' + reason)


def terminal(status: str, *, claim: dict, result: dict | None, head: str,
             reason: str | None = None) -> dict:
    need(status in ('COMPLETE', 'FAILED_CONSUMED')
         and (result is not None) == (status == 'COMPLETE')
         and (reason is not None) == (status == 'FAILED_CONSUMED'),
         'TERMINAL_ARGUMENTS')
    value = {'schema_version': 'gx1_gc_mid_response_fold2_terminal_v1',
             'status': status, 'exit_code': 0 if status == 'COMPLETE' else 1,
             'stage': STAGE, 'claim': claim, 'result': result,
             'source_head': head, 'test_data_used': False}
    if reason is not None:
        value['reason'] = reason
    return value


def inspect(plan_sha: str) -> tuple:
    need(Path.cwd().resolve() == REPO
         and book.git('branch', '--show-current') == 'work/gx1-current'
         and book.git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = book.git('rev-parse', 'HEAD')
    need(not ROOT.exists() and book.sha(PLAN) == plan_sha, 'PLAN_OUTPUT')
    plan = json.loads(PLAN.read_text())
    need(set(plan) == {'schema_version', 'stage', 'status', 'source_base_head',
                       'input_refs', 'source_owners', 'scope', 'limits', 'output_root'}
         and plan['schema_version'] == 'gx1_gc_mid_response_fold2_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['scope'] == {
             'fold_id': 2, 'fit_rows': 47338, 'later_rows': 4096,
             'timeframes': ['M5', 'M15', 'H1', 'H4', 'D1'],
             'tasks': ['state', 'future'], 'primary_cell': 'state_M5',
             'field': FIELD, 'field_columns': list(FIELD_COLS),
             'price_control': CONTROL, 'price_control_column': CONTROL_COL,
             'positive_orientation': 'UP', 'test_data_used': False}
         and plan['limits'] == {
             'class': 'audit', 'memory': '4G', 'swap': '512M',
             'wall_seconds': 1800, 'max_invocations': 1,
             'model_forwards': 0, 'optimizer_steps': 0,
             'training_authorized': False}
         and plan['output_root'] == str(ROOT), 'PLAN')
    need(subprocess.run(('git', 'merge-base', '--is-ancestor',
                         plan['source_base_head'], head), cwd=REPO).returncode == 0,
         'ANCESTOR')
    need(set(plan['source_owners']) == set(CODE)
         and all(book.sha(REPO / path) == digest
                 for path, digest in plan['source_owners'].items()), 'SOURCE_HASH')
    expected = {name: {'path': str(path), 'sha256': digest}
                for name, (path, digest) in INPUTS.items()}
    need(plan['input_refs'] == expected
         and all(book.sha(path) == digest for path, digest in INPUTS.values()),
         'INPUT_HASH')
    old = json.loads((book.OLD_GC_ROOT / 'RESULT.json').read_text())
    old_terminal = json.loads((book.OLD_GC_ROOT / 'TERMINAL.json').read_text())
    baseline = json.loads((prior.ROOT / 'RESULT.json').read_text())
    baseline_terminal = json.loads((prior.ROOT / 'TERMINAL.json').read_text())
    need(old['status'] == old_terminal['status'] == 'COMPLETE'
         and old_terminal['exit_code'] == 0
         and old_terminal['result'] == expected['old_gc_result']
         and old['model_forwards'] == old['optimizer_steps'] == 0
         and old['test_data_used'] is False
         and baseline['status'] == baseline_terminal['status'] == 'COMPLETE'
         and baseline_terminal['exit_code'] == 0
         and baseline_terminal['result'] == expected['prior_result']
         and baseline['fit_rows'] == 47338
         and baseline['selected_score_rows'] == 4096
         and baseline['frozen_priors'] == expected['prior_frozen']
         and baseline['test_data_used'] is False, 'RECEIPTS')
    meta = book.legacy_source_metadata()
    manifest = json.loads(Path(source.REFS['split_manifest']['path']).read_text())
    fields = manifest['feature_contract']['ctx_cont_names']
    registry = meta['registry']['field_report'][FIELD]
    need(len(fields) == book.SHAPE[1]
         and fields[FIELD_COLS[0]] == FIELD
         and fields[FIELD_COLS[1]] == FIELD + '_observed'
         and fields[CONTROL_COL] == CONTROL
         and registry['decision_rows'] == book.SHAPE[0]
         and registry['finite_observed'] == 117628
         and registry['masked_float32_train_sha256'] == EXPECTED_FIELD_HASH,
         'FIELD_BINDING')
    s = state.load_bound_five_endpoint_targets()
    f = future.load_bound_future_displacement()
    need(np.array_equal(s.parent_rows, f.parent_rows)
         and np.array_equal(s.decision_ns, f.decision_ns), 'TARGET_JOIN')
    axis = source._axis_identity(s.parent_rows, s.decision_ns,
                                 meta['provider'], meta['registry'])
    need(axis == old['axis_sha256'], 'AXIS_REPLAY')
    fit, later_rows, later_issues = prior.axes(
        json.loads(prior.COVERAGE.read_text()), s)
    policy = json.loads(POLICY.read_text())
    need(policy.get('training_enabled') is False
         and policy.get('native_training_enabled') is False
         and policy.get('native_launch_authorized') is False
         and policy.get(STAGE) == {
             'status': 'ADMITTED', 'plan': book.ref(PLAN),
             'output_root': str(ROOT), 'max_invocations': 1,
             'claimed_invocations': 0, 'completed_invocations': 0,
             'execution_authorized': True, 'data_reads_authorized': True,
             'model_forwards_authorized': 0, 'optimizer_steps_authorized': 0,
             'training_authorized': False, 'test_authorized': False,
             'broker_access_allowed': False, 'relaunch_allowed': False},
         'POLICY')
    return meta, book.ref(POLICY), head, s, f, fit, later_rows, later_issues, registry


def extract(meta: dict, registry: dict) -> tuple:
    before = source._file_stat(source.CURRENT_MMAP)
    need(before == meta['current_stat'], 'CURRENT_STAT')
    mmap = np.memmap(source.CURRENT_MMAP, mode='r', dtype='<f4',
                     shape=book.SHAPE)
    gc = np.array(mmap[:, FIELD_COLS[0]], dtype=np.float32, copy=True)
    mask_raw = np.array(mmap[:, FIELD_COLS[1]], dtype=np.float32, copy=True)
    price = np.array(mmap[:, CONTROL_COL], dtype=np.float32, copy=True)
    del mmap
    need(source._file_stat(source.CURRENT_MMAP) == before
         and np.isfinite(mask_raw).all()
         and np.isin(mask_raw, (0.0, 1.0)).all(), 'EXTRACT')
    observed = mask_raw == 1.0
    digest = source._masked_hash(gc, observed)
    need(digest == EXPECTED_FIELD_HASH
         and digest == registry['masked_float32_train_sha256']
         and int(observed.sum()) == registry['finite_observed']
         and int((~observed).sum()) == registry['missing']
         and np.isfinite(gc[observed]).all(), 'FIELD_BYTES')
    return gc.astype(np.float64), observed, price.astype(np.float64), {
        'field': FIELD, 'value_col': FIELD_COLS[0], 'mask_col': FIELD_COLS[1],
        'masked_float32_train_sha256': digest,
        'observed': int(observed.sum()), 'missing': int((~observed).sum()),
        'price_control': CONTROL, 'price_control_col': CONTROL_COL,
        'price_control_finite': int(np.isfinite(price).sum()),
        'mmap_stat': before}


def run(plan_sha: str) -> None:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    meta, policy_ref, head, s, f, fit, later_rows, later_issues, registry = inspect(plan_sha)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    ROOT.mkdir(parents=False, exist_ok=False)
    _publish_bytes(ROOT / 'CLAIM.json', book.encode({
        'schema_version': 'gx1_gc_mid_response_fold2_claim_v1',
        'stage': STAGE, 'plan': book.ref(PLAN), 'policy': policy_ref,
        'source_head': head, 'test_data_used': False}))
    claim = book.ref(ROOT / 'CLAIM.json')
    started = time.monotonic()
    try:
        gc, observed, price, field = extract(meta, registry)
        fit_out = book.cohort(s, f, rows=fit['fit_parent_rows'],
                              issues=fit['fit_decision_ns'], partition='fit',
                              gc=gc, observed=observed, price=price)
        later_out = book.cohort(s, f, rows=later_rows, issues=later_issues,
                                partition='score', gc=gc, observed=observed,
                                price=price)
        result = {'schema_version': 'gx1_gc_mid_response_fold2_result_v1',
                  'status': 'COMPLETE', 'stage': STAGE,
                  'plan': book.ref(PLAN), 'policy': policy_ref,
                  'claim': claim, 'source_head': head,
                  'observed_utc': datetime.now(timezone.utc).isoformat(),
                  'source_field': field, 'fit': fit_out, 'later': later_out,
                  'model_forwards': 0, 'optimizer_steps': 0,
                  'probability_fit': False, 'test_data_used': False,
                  'trade_selection': False,
                  'elapsed_seconds': time.monotonic() - started}
        need(book.git('rev-parse', 'HEAD') == head
             and book.git('status', '--porcelain') == ''
             and book.ref(POLICY) == policy_ref
             and all(book.sha(path) == digest for path, digest in INPUTS.values()),
             'FINAL_SOURCE')
        _publish_bytes(ROOT / 'RESULT.json', book.encode(result))
        _publish_bytes(ROOT / 'TERMINAL.json', book.encode(
            terminal('COMPLETE', claim=claim,
                     result=book.ref(ROOT / 'RESULT.json'), head=head)))
    except BaseException as exc:
        _publish_bytes(ROOT / 'TERMINAL.json', book.encode(
            terminal('FAILED_CONSUMED', claim=claim, result=None, head=head,
                     reason=type(exc).__name__ + ':' + str(exc))))
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan-sha256', required=True)
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    need(args.preflight != args.run, 'MODE')
    if args.preflight:
        meta, _, head, _, _, fit, rows, _, registry = inspect(args.plan_sha256)
        need(fit['joined_fit_rows'] == 47338 and len(rows) == 4096,
             'PREFLIGHT_ROWS')
        gc, observed, price, field = extract(meta, registry)
        need(field['observed'] == 117628
             and len(gc) == len(price) == len(observed) == book.SHAPE[0],
             'PREFLIGHT_FIELD')
        toy = book.score_signal(np.array([-1., 0., 1., 2.]),
                                np.array([1., -1., -1., 1.]),
                                np.array([False, False, True, True]))
        need(toy['gc_auc'] == 1.0 and toy['constant_prior_auc'] == 0.5,
             'SYNTHETIC_SCORE_SUCCESS')
        for status, result, reason in (
            ('COMPLETE', {'path': 'synthetic', 'sha256': '0'*64}, None),
            ('FAILED_CONSUMED', None, 'synthetic_failure'),
        ):
            item = terminal(status, claim={'path': 'synthetic', 'sha256': '1'*64},
                            result=result, head=head, reason=reason)
            need(item['status'] == status
                 and (item['exit_code'] == 0) == (status == 'COMPLETE')
                 and (item['result'] is not None) == (status == 'COMPLETE'),
                 'SYNTHETIC_TERMINAL')
        print(json.dumps({'preflight': 'PASS_NO_CLAIM_NO_TARGET_SCORE',
                          'source_head': head, 'fit_rows': 47338,
                          'later_rows': 4096,
                          'field_digest': field['masked_float32_train_sha256']},
                         sort_keys=True))
    else:
        run(args.plan_sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
