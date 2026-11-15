"""No-fit causal GC book-imbalance rank screen on exact Fold2 S/F axes.

One predeclared M5 book field, one XAU EMA-distance control, ten S/F target
cells, fit and later development. No probability fit, model, Entry, TEST or trade.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

from gx1.audit import measure_market_state_fold2_exact_prior_later_v1 as prior
from gx1.audit import score_market_state_gc_flow_reference_v1 as source
from gx1.audit.market_state_m5_gc_flow_reference_core_v1 import _auc
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_future_displacement_candidate_v1 as future
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'market_state_gc_book_tw_fold2_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_GC_BOOK_TW_FOLD2_20261115_001.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_GC_BOOK_TW_FOLD2_20261115_001')
FIELD = 'gc_m5_observed_book_tw_imbalance'
CONTROL = 'close_distance_from_ema5_bps'
FIELD_COLS = (814, 815)
CONTROL_COL = 8
SHAPE = (118249, 2542)
OLD_GC_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_GC_FLOW_REFERENCE_20261107_001')
INPUTS = {
    **{name: (Path(ref['path']), ref['sha256']) for name, ref in source.REFS.items()},
    'coverage': prior.INPUTS['coverage'],
    'prior_result': (prior.ROOT / 'RESULT.json',
                     '252b397f794bb7663e7836b941ba69f33fcc58f31bb6e0796af448b408140b07'),
    'prior_terminal': (prior.ROOT / 'TERMINAL.json',
                       '304b7204faab52d9440f659a4e46bcd9da2f4de856963dcf37adb8a265a6b497'),
    'prior_frozen': (prior.ROOT / 'FROZEN_PRIORS.json',
                     'bd8ecf1d47d067d8ef3aee1f4bd5fa2241e618f615edc4792b95dbe194fa0eb4'),
    'old_gc_result': (OLD_GC_ROOT / 'RESULT.json',
                      '256710c697cba9324cb4635aa323e99f4319ddbd1c418fa54ec549f04fde4946'),
    'old_gc_terminal': (OLD_GC_ROOT / 'TERMINAL.json',
                        '18c979db1685d25dd30deb337e65fea6da0168c6fea98137a8f1371dc1a8d8c7'),
    'state_mask': (Path(state.REFS['mask'][0]), state.REFS['mask'][1]),
    'model_axes': (Path(state.REFS['model_axes'][0]), state.REFS['model_axes'][1]),
    'future_tape': (Path(future.REFS['tape'][0]), future.REFS['tape'][1]),
}
CODE = (
    'gx1/audit/measure_market_state_gc_book_tw_fold2_v1.py',
    'gx1/audit/measure_market_state_fold2_exact_prior_later_v1.py',
    'gx1/audit/score_market_state_gc_flow_reference_v1.py',
    'gx1/audit/market_state_m5_gc_flow_reference_core_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_future_displacement_candidate_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/features/entry_specialist_feature_groups_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
)


def need(value: object, reason: str) -> None:
    if not bool(value):
        raise RuntimeError('GC_BOOK_TW_FOLD2_' + reason)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict:
    return {'path': str(path), 'sha256': sha(path)}


def encode(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def git(*args: str) -> str:
    return subprocess.check_output(('git', *args), cwd=REPO, text=True).strip()


def score_signal(gc: np.ndarray, price: np.ndarray, up: np.ndarray) -> dict:
    need(gc.ndim == price.ndim == up.ndim == 1
         and gc.shape == price.shape == up.shape and len(gc) > 0
         and np.isfinite(gc).all() and np.isfinite(price).all()
         and up.dtype == np.bool_, 'SCORE_DOMAIN')
    out = {'n': len(gc), 'up': int(up.sum()), 'down': int((~up).sum()),
           'gc_auc': _auc(gc, up)['value'],
           'price_control_auc': _auc(price, up)['value'],
           'constant_prior_auc': _auc(np.zeros(len(gc), dtype=np.float64), up)['value'],
           'price_sign_strata': {}}
    for name, mask in (('price_nonpositive', price <= 0),
                       ('price_positive', price > 0)):
        out['price_sign_strata'][name] = {
            'n': int(mask.sum()), 'up': int(up[mask].sum()),
            'gc_auc': _auc(gc[mask], up[mask])['value'] if mask.any() else None}
    return out



def legacy_source_metadata() -> dict:
    """Bind sealed 051 fields without consulting today's expanded native profile."""
    def load(name: str, limit: int = 4 * 1024**2) -> dict:
        return source._read_json(source.REFS[name], limit)

    provider = load('provider_plan')
    binding = load('decode_binding')
    source_plan = load('source_plan')
    source_terminal = load('source_terminal')
    source_outer = load('source_outer_terminal')
    reconstruction = load('source_reconstruction')
    feature_manifest = load('feature_manifest')
    split = load('split_manifest', 64 * 1024**2)
    registry = load('gc_registry')
    complete_plan = load('complete_plan')
    complete = load('complete_result')
    complete_terminal = load('complete_terminal')
    refs = source.REFS
    need(provider.get('decode_reuse_binding') == refs['decode_binding']
         and provider.get('m5_feature_manifest') == refs['feature_manifest']
         and provider.get('source_reconstruction') == refs['source_reconstruction']
         and provider.get('axes', {}).get('source')
             == {'path': str(Path(state.REFS['model_axes'][0])),
                 'sha256': state.REFS['model_axes'][1]}
         and provider['axes'].get('rows') == SHAPE[0]
         and provider['axes'].get('decision_rule')
             == 'bar_start_ns_plus_300000000000'
         and binding.get('source_plan') == refs['source_plan']
         and binding.get('source_worker') == refs['source_worker']
         and binding.get('terminal') == refs['source_terminal']
         and binding.get('outer_terminal') == refs['source_outer_terminal']
         and binding.get('schema_version')
             == 'gx1_native_direction_completed_decode_reuse_v1'
         and binding.get('data_fit_or_support_qualified') is False
         and source_terminal.get('source_unchanged') is True
         and source_terminal.get('test_data_used') is False
         and source_outer.get('source_and_external_bookends_unchanged') is True
         and source_plan.get('worker') == refs['source_worker'],
         'LEGACY_LINEAGE')
    need(complete.get('status')
             == 'COMPLETE_FIVE_RAW_QUALIFICATIONS_FOUR_BOOLEAN_PREFIX_NORMALIZATIONS_NO_MODEL_ADMISSION'
         and complete.get('input_provider_plan') == refs['provider_plan']
         and complete.get('five_complete_raw_parities_before_first_fit') is True
         and complete.get('test_data_used') is False
         and complete.get('plan') == refs['complete_plan']
         and complete_terminal.get('status') == 'COMPLETE'
         and complete_terminal.get('exit_code') == 0
         and complete_terminal.get('source_unchanged') is True
         and complete_terminal.get('result') == refs['complete_result']
         and complete_plan.get('input_provider_plan') == refs['provider_plan'],
         'LEGACY_COMPLETE')
    for path, shape in ((source.CURRENT_MMAP, SHAPE),
                        (source.SOURCE_MMAP, (1140763, SHAPE[1]))):
        stat = source._file_stat(path)
        need(stat == binding['scratch_files'][str(path)]
             and stat == complete['reused_scratch'][str(path)]
             and not stat['st_mode'] & 0o222
             and stat['st_size'] == shape[0] * shape[1] * 4,
             'MMAP_STAT:' + path.name)
    checks = reconstruction.get('checks', {})
    need(reconstruction.get('decision') == 'PASS'
         and reconstruction.get('rows') == SHAPE[0]
         and reconstruction.get('feature_surface_rows') == 1140763
         and reconstruction.get('manifest_path') == refs['split_manifest']['path']
         and reconstruction.get('manifest_sha256') == refs['split_manifest']['sha256']
         and all(checks.get(key) is True for key in (
             'every_emitted_timestamp_maps_exactly_once_to_feature_surface',
             'every_sequence_equals_exact_source_surface_history_bit_identical',
             'every_snapshot_equals_exact_source_surface_current_row_bit_identical',
             'source_timestamps_strictly_increasing')),
         'LEGACY_CLOCK')
    need(feature_manifest.get('ctx_cont_dim') == SHAPE[1]
         and len(split['feature_contract']['ctx_cont_names']) == SHAPE[1]
         and split['extra']['gc_native_context']['registry_sha256']
             == refs['gc_registry']['sha256']
         and registry.get('schema_version') == 'gx1_native_gc_context_registry_v1'
         and registry.get('outcomes_accessed') is False
         and registry.get('train_decision_rows') == SHAPE[0]
         and registry.get('train_parent_clock_binding', {}).get('bar_clock_sha256')
             == provider['axes']['bar_start_ns_sha256'],
         'LEGACY_FIELDS')
    return {'provider': provider, 'binding': binding, 'registry': registry,
            'current_stat': source._file_stat(source.CURRENT_MMAP),
            'source_stat': source._file_stat(source.SOURCE_MMAP)}


def inspect(plan_sha: str) -> tuple[dict, dict, str, object, object, dict, np.ndarray, np.ndarray, dict]:
    need(Path.cwd().resolve() == REPO
         and git('branch', '--show-current') == 'work/gx1-current'
         and git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = git('rev-parse', 'HEAD')
    need(not ROOT.exists() and sha(PLAN) == plan_sha, 'PLAN_OUTPUT')
    plan = json.loads(PLAN.read_text())
    need(set(plan) == {'schema_version', 'stage', 'status', 'source_base_head',
                       'input_refs', 'source_owners', 'scope', 'limits', 'output_root'}
         and plan['schema_version'] == 'gx1_gc_book_tw_fold2_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['scope'] == {
             'fold_id': 2, 'fit_rows': 47338, 'later_rows': 4096,
             'timeframes': ['M5', 'M15', 'H1', 'H4', 'D1'],
             'tasks': ['state', 'future'], 'field': FIELD,
             'field_columns': list(FIELD_COLS), 'price_control': CONTROL,
             'price_control_column': CONTROL_COL,
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
         and all(sha(REPO / rel) == digest
                 for rel, digest in plan['source_owners'].items()), 'SOURCE_HASH')
    expected = {name: {'path': str(path), 'sha256': digest}
                for name, (path, digest) in INPUTS.items()}
    need(plan['input_refs'] == expected
         and all(sha(path) == digest for path, digest in INPUTS.values()),
         'INPUT_HASH')
    old = json.loads((OLD_GC_ROOT / 'RESULT.json').read_text())
    old_terminal = json.loads((OLD_GC_ROOT / 'TERMINAL.json').read_text())
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
    meta = legacy_source_metadata()
    names = meta['binding']['scratch_files']
    need(str(source.CURRENT_MMAP) in names, 'MMAP_BINDING')
    manifest = json.loads(Path(source.REFS['split_manifest']['path']).read_text())
    fields = manifest['feature_contract']['ctx_cont_names']
    registry = meta['registry']['field_report'][FIELD]
    need(len(fields) == SHAPE[1]
         and fields[FIELD_COLS[0]] == FIELD
         and fields[FIELD_COLS[1]] == FIELD + '_observed'
         and fields[CONTROL_COL] == CONTROL
         and registry['decision_rows'] == SHAPE[0]
         and registry['finite_observed'] == 117628
         and registry['masked_float32_train_sha256']
             == '792d56e2c63753daa24bb5d71b2f162a31539cde018e6df60f59a330fa93da4c',
         'FIELD_BINDING')
    state_bound = state.load_bound_five_endpoint_targets()
    future_bound = future.load_bound_future_displacement()
    need(np.array_equal(state_bound.parent_rows, future_bound.parent_rows)
         and np.array_equal(state_bound.decision_ns, future_bound.decision_ns),
         'TARGET_JOIN')
    axis = source._axis_identity(state_bound.parent_rows, state_bound.decision_ns,
                                 meta['provider'], meta['registry'])
    need(axis == old['axis_sha256'], 'AXIS_REPLAY')
    fit, later_rows, later_issues = prior.axes(
        json.loads(prior.COVERAGE.read_text()), state_bound)
    policy = json.loads(POLICY.read_text())
    need(policy.get('training_enabled') is False
         and policy.get('native_training_enabled') is False
         and policy.get('native_launch_authorized') is False
         and policy.get(STAGE) == {
             'status': 'ADMITTED', 'plan': ref(PLAN),
             'output_root': str(ROOT), 'max_invocations': 1,
             'claimed_invocations': 0, 'completed_invocations': 0,
             'execution_authorized': True, 'data_reads_authorized': True,
             'model_forwards_authorized': 0, 'optimizer_steps_authorized': 0,
             'training_authorized': False, 'test_authorized': False,
             'broker_access_allowed': False, 'relaunch_allowed': False},
         'POLICY')
    return meta, ref(POLICY), head, state_bound, future_bound, fit, later_rows, later_issues, registry


def extract(meta: dict, registry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    before = source._file_stat(source.CURRENT_MMAP)
    need(before == meta['current_stat'], 'CURRENT_STAT')
    mmap = np.memmap(source.CURRENT_MMAP, mode='r', dtype='<f4', shape=SHAPE)
    gc = np.array(mmap[:, FIELD_COLS[0]], dtype=np.float32, copy=True)
    mask_raw = np.array(mmap[:, FIELD_COLS[1]], dtype=np.float32, copy=True)
    price = np.array(mmap[:, CONTROL_COL], dtype=np.float32, copy=True)
    del mmap
    need(source._file_stat(source.CURRENT_MMAP) == before
         and np.isfinite(mask_raw).all()
         and np.isin(mask_raw, (0.0, 1.0)).all(), 'EXTRACT')
    observed = mask_raw == 1.0
    digest = source._masked_hash(gc, observed)
    need(digest == registry['masked_float32_train_sha256']
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


def cohort(bound_state: object, bound_future: object, *,
           rows: np.ndarray, issues: np.ndarray, partition: str,
           gc: np.ndarray, observed: np.ndarray, price: np.ndarray) -> dict:
    s = bound_state.gather(rows, issues, fold_id=2, partition=partition)
    f = bound_future.gather(rows, issues, fold_id=2, partition=partition)
    need(np.array_equal(s['parent_rows'], f['parent_rows'])
         and np.array_equal(s['issue_ns'], f['issue_ns'])
         and np.array_equal(s['parent_rows'], rows)
         and np.array_equal(s['issue_ns'], issues), 'COHORT_JOIN')
    weeks = objective._weeks(issues)
    out = {'rows': len(rows), 'tasks': {}}
    for name, truth in (('state', s), ('future', f)):
        per_tf = {}
        for col, tf in enumerate(objective.TIMEFRAMES):
            bits = truth['allowed'][:, col]
            known = truth['eligible'][:, col] & truth['mature'][:, col] & np.isin(bits, (1, 4))
            good = known & observed[rows] & np.isfinite(price[rows])
            need(bool(good.any()), 'EMPTY:' + name + ':' + tf)
            scores = gc[rows[good]]
            controls = price[rows[good]]
            up = bits[good] == 4
            measured = score_signal(scores, controls, up)
            by_week = {}
            for week in np.unique(weeks[good]):
                chosen = good & (weeks == week)
                by_week[str(week)] = score_signal(
                    gc[rows[chosen]], price[rows[chosen]], bits[chosen] == 4)
            need(sum(x['n'] for x in by_week.values()) == measured['n'], 'WEEKS')
            per_tf[tf] = {
                'known_side': int(known.sum()),
                'joint_observed': int(good.sum()),
                'gc_missing_on_known_side': int((known & ~observed[rows]).sum()),
                'price_nonfinite_on_known_gc': int((known & observed[rows] & ~np.isfinite(price[rows])).sum()),
                'overall': measured, 'by_utc_issue_week': by_week}
        out['tasks'][name] = per_tf
    return out


def run(plan_sha: str) -> None:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    meta, policy_ref, head, s, f, fit, later_rows, later_issues, registry = inspect(plan_sha)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    ROOT.mkdir(parents=False, exist_ok=False)
    _publish_bytes(ROOT / 'CLAIM.json', encode({
        'schema_version': 'gx1_gc_book_tw_fold2_claim_v1',
        'stage': STAGE, 'plan': ref(PLAN), 'policy': policy_ref,
        'source_head': head, 'test_data_used': False}))
    started = time.monotonic()
    try:
        gc, observed, price, field = extract(meta, registry)
        fit_out = cohort(s, f, rows=fit['fit_parent_rows'],
                         issues=fit['fit_decision_ns'], partition='fit',
                         gc=gc, observed=observed, price=price)
        later_out = cohort(s, f, rows=later_rows, issues=later_issues,
                           partition='score', gc=gc, observed=observed, price=price)
        value = {'schema_version': 'gx1_gc_book_tw_fold2_result_v1',
                 'status': 'COMPLETE', 'stage': STAGE,
                 'plan': ref(PLAN), 'policy': policy_ref,
                 'claim': ref(ROOT / 'CLAIM.json'), 'source_head': head,
                 'observed_utc': datetime.now(timezone.utc).isoformat(),
                 'source_field': field, 'fit': fit_out, 'later': later_out,
                 'model_forwards': 0, 'optimizer_steps': 0,
                 'probability_fit': False, 'test_data_used': False,
                 'trade_selection': False,
                 'elapsed_seconds': time.monotonic() - started}
        need(git('rev-parse', 'HEAD') == head
             and git('status', '--porcelain') == ''
             and ref(POLICY) == policy_ref
             and all(sha(path) == digest for path, digest in INPUTS.values()),
             'FINAL_SOURCE')
        _publish_bytes(ROOT / 'RESULT.json', encode(value))
        _publish_bytes(ROOT / 'TERMINAL.json', encode({
            'schema_version': 'gx1_gc_book_tw_fold2_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0,
            'stage': STAGE, 'claim': ref(ROOT / 'CLAIM.json'),
            'result': ref(ROOT / 'RESULT.json'), 'source_head': head,
            'test_data_used': False}))
    except BaseException as exc:
        _publish_bytes(ROOT / 'TERMINAL.json', encode({
            'schema_version': 'gx1_gc_book_tw_fold2_terminal_v1',
            'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'stage': STAGE, 'claim': ref(ROOT / 'CLAIM.json'),
            'result': None, 'source_head': head,
            'reason': type(exc).__name__ + ':' + str(exc),
            'test_data_used': False}))
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
        need(field['observed'] == 117628 and len(gc) == len(price) == len(observed) == SHAPE[0],
             'PREFLIGHT_FIELD')
        toy = score_signal(np.array([-1., 0., 1., 2.]),
                           np.array([1., -1., -1., 1.]),
                           np.array([False, False, True, True]))
        need(toy['gc_auc'] == 1.0 and toy['constant_prior_auc'] == 0.5,
             'SYNTHETIC_SUCCESS')
        try:
            score_signal(np.array([float('nan')]), np.array([0.]),
                         np.array([True]))
        except RuntimeError as exc:
            need(str(exc) == 'GC_BOOK_TW_FOLD2_SCORE_DOMAIN', 'SYNTHETIC_FAILURE')
        else:
            need(False, 'SYNTHETIC_FAILURE')
        print(json.dumps({'preflight': 'PASS_NO_CLAIM_NO_TARGET_SCORE',
                          'source_head': head, 'fit_rows': 47338,
                          'later_rows': 4096, 'field_digest': field['masked_float32_train_sha256']},
                         sort_keys=True))
    else:
        run(args.plan_sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
