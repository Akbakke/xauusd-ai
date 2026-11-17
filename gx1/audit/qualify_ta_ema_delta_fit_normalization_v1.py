"""One-use Fold1 FIT normalization of six qualified additive EMA-delta sidecars.

Only the restored field is fitted. Existing 317/255 statistics are copied by
name; later rows are checked for coverage but never included in the fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from gx1.audit import qualify_market_state_ta_anchor_span_inputs_v1 as axes
from gx1.contracts import entry_anchor_span_normalization_v1 as extension
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
STAGE = 'ta_m5_ema_delta_fit_normalization_20261118_001'
ROOT = RUNS / STAGE.upper()
PLAN = REPO / 'configs/research' / (STAGE.upper() + '.json')
POLICY = REPO / 'NEXT_RUN_POLICY.json'
MATERIALIZED = RUNS / 'TA_M5_EMA_DELTA_SIDECARS_20261117_002'
MATERIALIZED_RESULT = MATERIALIZED / 'OUTPUT/RESULT.json'
MATERIALIZED_TERMINAL = MATERIALIZED / 'TERMINAL.json'
FOLD1_PLAN = REPO / 'configs/research/MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_NORMALIZATION_20261112_001.json'
PARENT = RUNS / 'MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_NORMALIZATION_20261112_001/EXTENDED_BASE_NORMALIZATION.json'
FOLD1_PLAN_SHA256 = '07653339e511831d34f17f3ad12120e4a1ffea48795bb8d1ccf0daa0a68210e3'
PARENT_SHA256 = '71e201b690101b7a0748fe8594fe35cae6d88288f74deee46897a54adfda69c1'
SPAN_RESULT = RUNS / 'MARKET_STATE_TA_FOLD2_SPAN_REUSE_20261113_002/RESULT.json'
LANES = {'signal': 'local', 'mtf_m5': 'M5', 'mtf_m15': 'M15',
         'mtf_h1': 'H1', 'mtf_h4': 'H4', 'mtf_d1': 'D1'}
SOURCES = (
    'gx1/audit/qualify_ta_ema_delta_fit_normalization_v1.py',
    'gx1/audit/qualify_market_state_ta_anchor_span_inputs_v1.py',
    'gx1/contracts/entry_anchor_span_normalization_v1.py',
    'gx1/contracts/entry_model_native_input_normalization_v1.py',
    'gx1/contracts/entry_native_direction_prefix_units_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/features/entry_specialist_feature_groups_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_qualify_ta_ema_delta_fit_normalization_v1.py',
)
SCOPE = {'fold_id': 1, 'fit_issue_rows': 29574, 'surface_fit_calls': 6,
         'new_field_statistics': 6, 'old_field_fit_calls': 0,
         'model_constructions': 0, 'model_forwards': 0, 'optimizer_steps': 0,
         'test_data_used': False, 'later_statistics_fit': False}


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_EMA_DELTA_NORMALIZATION_' + reason)


def ref(path):
    path = Path(path)
    need(path.is_absolute() and path.is_file() and not path.is_symlink(), 'FILE_REF')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def read(binding):
    need(ref(binding['path']) == binding, 'INPUT_REF')
    return json.loads(Path(binding['path']).read_text())


def _sha_array(value, dtype):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def input_refs():
    result = read(ref(MATERIALIZED_RESULT))
    manifest_ref = result['manifest']
    manifest = read(manifest_ref)
    need(manifest_ref['path'] == str(MATERIALIZED / 'OUTPUT/MANIFEST.json')
         and result['status'] == manifest['status'] == 'COMPLETE'
         and result['schema_version'] == 'entry_ta_ema_delta_materialization_result_v1'
         and manifest['schema_version'] == 'entry_ta_ema_delta_sidecars_v1'
         and result['source'] == manifest['source']
         and result['surfaces'] == manifest['surfaces']
         and result['cutoff_ns'] == manifest['cutoff_ns'] == 1735603200000000000
         and result['fit_rows'] == 29574
         and result['later_parent_start'] == 29849
         and result['later_rows'] == 17489
         and result['test_data_used'] is manifest['test_data_used'] is False
         and result['model_forwards'] == manifest['model_forwards'] == 0
         and result['optimizer_steps'] == manifest['optimizer_steps'] == 0
         and manifest['fields'] == {'local': extension.ema_delta_names('signal')[0],
                                    'mtf': extension.ema_delta_names('mtf_m5')[0]},
         'MATERIALIZED_METADATA')
    terminal = read(ref(MATERIALIZED_TERMINAL))
    need(terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
         and terminal['result'] == ref(MATERIALIZED_RESULT)
         and terminal['source_unchanged'] is True, 'MATERIALIZED_TERMINAL')
    need(ref(FOLD1_PLAN)['sha256'] == FOLD1_PLAN_SHA256
         and ref(PARENT)['sha256'] == PARENT_SHA256, 'HISTORICAL_PARENT_REFS')
    old_plan = read(ref(FOLD1_PLAN))
    need(set(old_plan['populations']) == set(LANES)
         and old_plan['scope']['parent_rows'] == 29574, 'FOLD1_POPULATIONS')
    need(manifest['span_reuse_result'] == ref(SPAN_RESULT), 'SPAN_RESULT_REF')
    span_result = read(manifest['span_reuse_result'])
    need(span_result['status'] == 'COMPLETE' and span_result['test_data_used'] is False,
         'SPAN_RESULT_STATUS')
    bindings = {'materialized_result': ref(MATERIALIZED_RESULT),
                'materialized_manifest': manifest_ref,
                'materialized_terminal': ref(MATERIALIZED_TERMINAL),
                'fold1_plan': ref(FOLD1_PLAN), 'parent_normalization': ref(PARENT)}
    for surface, lane in LANES.items():
        item = result['surfaces'][lane]
        need(item['artifact']['path'] == str(MATERIALIZED / 'OUTPUT' / (lane + '_EMA_DELTA.npz'))
             and ref(item['artifact']['path']) == item['artifact']
             and ref(item['span_artifact']['path']) == item['span_artifact']
             and span_result['surfaces'][lane]['artifact'] == item['span_artifact']
             and old_plan['populations'][surface]['lane'] == lane,
             'SIDECAR_BINDING:' + lane)
        bindings[lane + '_sidecar'] = item['artifact']
        bindings[lane + '_span'] = item['span_artifact']
    return bindings


def build_plan():
    return {'schema_version': 'gx1_ta_ema_delta_fit_normalization_plan_v1',
            'stage': STAGE, 'output_root': str(ROOT),
            'source_base_commit': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'source_owners': {name: ref(REPO / name) for name in SOURCES},
            'inputs': input_refs(), 'scope': dict(SCOPE),
            'execution': {'runner': 'scripts/gx1_capped_run.sh',
                          'job_class': 'audit', 'memory': '4G', 'swap': '512M',
                          'module': 'gx1.audit.qualify_ta_ema_delta_fit_normalization_v1',
                          'timeout_seconds': 300, 'kill_after_seconds': 10}}


def inspect(plan_sha, *, admitted):
    plan_ref = ref(PLAN)
    need(plan_ref['sha256'] == plan_sha and not ROOT.exists(), 'PLAN_OR_ONE_USE_ROOT')
    plan = read(plan_ref)
    expected = build_plan()
    expected['source_base_commit'] = plan.get('source_base_commit')
    need(plan == expected, 'PLAN_IDENTITY')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
    clean = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True) == ''
    ancestor = subprocess.run(['git', 'merge-base', '--is-ancestor',
        plan['source_base_commit'], head], cwd=REPO, capture_output=True).returncode == 0
    need(Path.cwd().resolve() == REPO and branch == 'work/gx1-current' and ancestor,
         'SOURCE_BRANCH_CWD')
    if not admitted:
        return plan, plan_ref, None, head, clean
    need(clean, 'SOURCE_CLEAN')
    policy_ref = ref(POLICY)
    policy = read(policy_ref)
    need(all(policy.get(key) is False for key in (
        'training_enabled', 'native_training_enabled', 'native_launch_authorized',
        'full_epoch_training_allowed', 'full_val_allowed')),
        'TRAINING_CLOSED')
    gate = policy.get(STAGE, {})
    need(gate.get('status') == 'ADMITTED' and gate.get('plan') == plan_ref
         and gate.get('scope') == plan['scope']
         and gate.get('execution_authorized') is True
         and gate.get('data_reads_authorized') is True
         and gate.get('normalization_fits_authorized') == 6
         and gate.get('model_forwards_authorized') == 0
         and gate.get('training_authorized') is gate.get('test_authorized') is False
         and gate.get('max_invocations') == gate.get('remaining_invocations') == 1
         and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
         and gate.get('relaunch_allowed') is False,
         'EXACT_ONE_USE_ADMISSION')
    return plan, plan_ref, policy_ref, head, clean


def fit_surface(surface, *, sidecar, span, population, parent_rows, cutoff_ns):
    """Use only the exact Fold1 prefix of a fully bound physical sidecar."""
    lane = LANES[surface]
    with np.load(sidecar['path'], allow_pickle=False) as archive:
        need(set(archive.files) == {'row_indices', 'times_ns', 'delta_5_atr'},
             'SIDECAR_MEMBERS:' + lane)
        rows, times, values = (archive[key] for key in
                               ('row_indices', 'times_ns', 'delta_5_atr'))
    with np.load(span['path'], allow_pickle=False) as archive:
        need(set(archive.files) == {'row_indices', 'times_ns',
                                   'native_bar_positions', 'anchor_span_bars'},
             'SPAN_MEMBERS:' + lane)
        span_rows, span_times = archive['row_indices'], archive['times_ns']
    need(rows.dtype == times.dtype == span_rows.dtype == span_times.dtype == np.int64
         and values.dtype == np.float32
         and rows.ndim == times.ndim == values.ndim == 1
         and rows.shape == times.shape == values.shape == span_rows.shape == span_times.shape
         and len(rows) > parent_rows
         and np.array_equal(rows, span_rows) and np.array_equal(times, span_times)
         and np.isfinite(values).all() and np.all(np.diff(rows) == 1)
         and np.all(np.diff(times) > 0), 'SIDECAR_SPAN_AXES:' + lane)
    first, stop = population['row_bounds']
    selected_rows = np.arange(first, stop, dtype=np.int64)
    at = np.searchsorted(rows, selected_rows)
    need(len(selected_rows) == parent_rows
         and np.all(at < len(rows))
         and np.array_equal(rows[at], selected_rows), 'FIT_PREFIX_ROWS:' + lane)
    selected_times = times[at]
    duration = axes.BAR_NS['M5' if lane == 'local' else lane] * 10**9
    need(axes.row_sha(selected_rows, lane) == population['population']['selected_row_indices_sha256']
         and axes.axis_sha(selected_times + duration) == population['population']['available_ns_sha256']
         and np.all(selected_times + duration < cutoff_ns), 'FIT_PREFIX_CLOCK:' + lane)
    chosen = np.ascontiguousarray(values[at, None], dtype=np.float32)
    stat = norm.fit_surface_normalization(chosen, surface=surface,
        field_names=extension.ema_delta_names(surface), allow_constant_train_fields=False)
    transformed = norm.apply_surface_normalization(chosen, stat)
    need(np.isfinite(transformed).all(), 'NORMALIZED_FINITE:' + lane)
    return stat, {'rows': parent_rows, 'row_indices_sha256': axes.row_sha(selected_rows, lane),
                  'available_ns_sha256': axes.axis_sha(selected_times + duration),
                  'raw_values_sha256': _sha_array(chosen[:, 0], '<f4'),
                  'normalized_values_sha256': _sha_array(transformed[:, 0], '<f4'),
                  'last_available_ns': int((selected_times + duration)[-1]),
                  'statistics_sha256': stat['stats_sha256']}


def publish(name, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _publish_bytes(ROOT / name, raw)
    return ref(ROOT / name)


def bookend(plan, plan_ref, policy_ref, head):
    need(ref(PLAN) == plan_ref and ref(POLICY) == policy_ref
         and all(ref(binding['path']) == binding for binding in plan['inputs'].values())
         and all(ref(binding['path']) == binding for binding in plan['source_owners'].values())
         and subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head
         and subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True) == '',
         'SOURCE_INPUT_BOOKEND')


def terminal_record(status, *, plan_ref, claim, result=None, error=None):
    need((status == 'COMPLETE' and result is not None and error is None)
         or (status == 'FAILED_CONSUMED' and result is None and error is not None),
         'TERMINAL_STATE')
    return {'status': status, 'exit_code': 0 if status == 'COMPLETE' else 1,
            'plan': plan_ref, 'claim': claim, 'result': result,
            'error': error, 'relaunch_allowed': False,
            'remaining_invocations': 0, 'test_data_used': False}


def run(plan_sha):
    cap = require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    plan, plan_ref, policy_ref, head, _ = inspect(plan_sha, admitted=True)
    ROOT.mkdir()
    claim = publish('CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
        'source_commit': head, 'cap': cap, 'scope': plan['scope']})
    result_ref = None
    try:
        refs = plan['inputs']
        materialized = read(refs['materialized_result'])
        manifest = read(refs['materialized_manifest'])
        old_plan = read(refs['fold1_plan'])
        old = read(refs['parent_normalization'])
        need(materialized['surfaces'] == manifest['surfaces']
             and materialized['source'] == manifest['source'], 'MATERIALIZED_IDENTITY')
        cutoff = units.mature_fold1_prefix_units_profile()['fit_prefix']['cutoff_ns']
        need(cutoff == axes.CUTOFF, 'FOLD1_CUTOFF')
        stats, proof = {}, {}
        for surface, lane in LANES.items():
            item = materialized['surfaces'][lane]
            with np.load(refs[lane + '_sidecar']['path'], allow_pickle=False) as archive:
                rows, times, values = (archive[key] for key in
                                       ('row_indices', 'times_ns', 'delta_5_atr'))
            need(_sha_array(rows, '<i8') == item['row_indices_sha256']
                 and _sha_array(times, '<i8') == item['times_ns_sha256']
                 and _sha_array(values, '<f4') == item['delta_5_atr_sha256']
                 and len(rows) == item['rows']
                 and int(times[0]) == item['first_time_ns']
                 and int(times[-1]) == item['last_time_ns']
                 and item['old_slope_parity_rows'] == len(rows)
                 and item['span_artifact'] == refs[lane + '_span'],
                 'MATERIALIZED_SURFACE:' + lane)
            population = old_plan['populations'][surface]
            count = old['surfaces'][surface]['fit_row_count']
            need(count == population['population']['selected_unique_row_count'],
                 'PARENT_FIT_ROWS:' + lane)
            stats[surface], proof[surface] = fit_surface(surface,
                sidecar=refs[lane + '_sidecar'], span=refs[lane + '_span'],
                population=population, parent_rows=count, cutoff_ns=cutoff)
        new = extension.extend_ema_delta_normalization(old, stats,
            physical_result=refs['materialized_result'])
        profile = units.ema_delta_fold1_prefix_units_profile()
        need(extension.old_field_preservation(old, new)['all_old_stat_vectors_exact'] is True,
             'OLD_STAT_PRESERVATION')
        ta = norm.project_ta_input_normalization_contract(new,
            unobserved_units_profile=profile)
        need([ta['surfaces'][surface]['field_count'] for surface in norm.EXPECTED_SURFACES]
             == [318, 132, 256, 256, 256, 256, 256], 'CURRENT_TA_WIDTHS')
        refs_out = {name: publish(name + '.json', value) for name, value in (
            ('NEW_FIELD_STATISTICS', stats), ('FIT_PROOF', proof),
            ('EXTENDED_BASE_NORMALIZATION', new), ('TA_INPUT_NORMALIZATION', ta),
            ('NEW_UNITS_PROFILE', profile))}
        bookend(plan, plan_ref, policy_ref, head)
        result_ref = publish('RESULT.json', {'status': 'COMPLETE', 'plan': plan_ref,
            'claim': claim, 'inputs': refs, 'artifacts': refs_out,
            'base_contract_sha256': new['contract_sha256'],
            'ta_contract_sha256': ta['contract_sha256'],
            'units_contract_sha256': profile['contract_sha256'],
            'fit_proof': proof, 'surface_fit_calls': 6,
            'old_field_fit_calls': 0, 'model_forwards': 0,
            'test_data_used': False, 'source_unchanged': True})
        terminal = publish('TERMINAL.json', terminal_record('COMPLETE',
            plan_ref=plan_ref, claim=claim, result=result_ref))
    except BaseException as exc:
        terminal = publish('TERMINAL.json', terminal_record('FAILED_CONSUMED',
            plan_ref=plan_ref, claim=claim,
            error=type(exc).__name__ + ':' + str(exc)[:1000]))
        raise
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        plan, plan_ref, _, head, clean = inspect(args.plan_sha256, admitted=False)
        print(json.dumps({'status': 'PREFLIGHT_PASS' if clean else 'PREFLIGHT_SOURCE_DIRTY',
            'plan': plan_ref, 'source_head': head, 'source_clean': clean,
            'output_root_absent': True, 'scope': plan['scope'], 'claim_created': False},
            sort_keys=True))
    else:
        print(json.dumps({'status': 'COMPLETE', 'terminal': run(args.plan_sha256)},
            sort_keys=True))


if __name__ == '__main__':
    main()
