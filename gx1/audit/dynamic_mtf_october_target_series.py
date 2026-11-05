"""Publish one strictly pre-November October development target series; no model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit.dynamic_mtf_train_prefix_targets import _copy_fit_arrays
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry


REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / 'configs/research/DYNAMIC_M5_OCTOBER_TARGET_SERIES_20261105_001.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
POLICY_KEY = 'dynamic_m5_october_target_series_20261105_001'


def _require(ok, reason):
    if not ok:
        raise RuntimeError('DYNAMIC_M5_OCTOBER_SERIES_' + reason)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _ref(path):
    return {'path': str(path), 'sha256': _sha(path)}


def _read(ref):
    _require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}
             and _ref(ref['path']) == ref, 'SOURCE_REF')
    return json.loads(Path(ref['path']).read_text())


def _axis_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<i8').tobytes()).hexdigest()


def _publish(root, name, value):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    _require(len(raw) <= 1048576, 'OUTPUT_SIZE:' + name)
    path = root / name
    _publish_bytes(path, raw)
    return _ref(path)


def run():
    require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    _require(Path.cwd().resolve() == REPO
             and subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'work/gx1-current'
             and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == '', 'CLEAN_REPO')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    plan_ref = _ref(PLAN)
    plan = _read(plan_ref)
    policy_bytes = POLICY.read_bytes()
    policy = json.loads(policy_bytes)
    state = policy.get(POLICY_KEY)
    _require(plan['schema_version'] == 'gx1_dynamic_m5_october_target_series_plan_v1'
             and plan['status'] == 'PREPARED_NOT_ADMITTED' and plan['max_invocations'] == 1
             and plan['output_root'] == '/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_TARGET_SERIES_20261105_001'
             and plan['execution']['command'] ==
                 'bash scripts/gx1_capped_run.sh --class audit --mem 2G --swap 512M -- timeout --signal=TERM --kill-after=10s 600s .venv/bin/python -m gx1.audit.dynamic_mtf_october_target_series'
             and plan['restrictions'] == {'model_construction': False, 'optimizer_updates': False,
                 'test_split': False, 'broker_or_network': False, 'spending_usd': 0,
                 'post_october_outcomes': False, 'raw_market_source_reads': False}
             and isinstance(state, dict) and state.get('status') == 'ADMITTED_ONCE'
             and state.get('plan') == plan_ref and state.get('operator_approved') is True
             and state.get('remaining_invocations') == 1
             and policy['training_enabled'] is False and policy['full_val_allowed'] is False,
             'EXPLICIT_ADMISSION_REQUIRED')
    _require(plan['standing_authorization'] == policy['no_cost_local_standing_authorization_20261105']['operator_instruction'],
             'STANDING_AUTHORIZATION')
    _require(all(_sha(REPO / path) == digest for path, digest in plan['source_files_sha256'].items()), 'SOURCE_CHANGED')
    axis = _read(plan['axis_receipt'])
    candidate = _read(plan['candidate_design'])
    original = _read(plan['original_series'])
    prior = _read(plan['prior_target_result'])
    terminal = _read(plan['prior_target_terminal'])
    _require(axis['fit_prefix'] == plan['fit_prefix'] and axis['evaluation'] == plan['evaluation_axis']
             and candidate['population']['axis_receipt'] == plan['axis_receipt']
             and prior['original_series'] == plan['original_series']
             and prior['fit_series'] == plan['fit_series']
             and prior['candidate_design'] == plan['candidate_design']
             and prior['parent_range'] == plan['evaluation_axis']['parent_row_range']
             and prior['source_row_slice'] == [0, 182360]
             and prior['model_calls'] == prior['post_october_outcomes_decoded'] == 0
             and terminal['exit_code'] == 0 and terminal['result'] == plan['prior_target_result']
             and original['metadata']['test_data_used'] is False
             and original['metadata']['definition']['contract_sha256'] == plan['target_contract_sha256'],
             'QUALIFIED_PREDECESSOR')
    root = Path(plan['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    try:
        source = target.read_dynamic_mtf_state_series_metadata(
            plan['original_series'], expected_source_refs=plan['source_refs'],
            expected_cutoff_ns=original['metadata']['cutoff_ns'])
        stop = plan['evaluation_axis']['parent_row_range'][1]
        cutoff = plan['evaluation_axis']['cutoff_ns_exclusive']
        arrays = _copy_fit_arrays(source, Path(plan['original_series']['path']).parent,
                                  prefix_rows=stop, source_rows=None, cutoff_ns=cutoff)
        _require(len(arrays['validity_ns']) == 182360 and len(arrays['parent_rows']) == stop,
                 'OCTOBER_SERIES_SIZE')
        fit = plan['fit_prefix']
        held = plan['evaluation_axis']
        for begin, end, expected in ((0, fit['rows'], fit),
                                     (fit['rows'], stop, held)):
            _require(_axis_sha(arrays['parent_rows'][begin:end]) == expected['parent_rows_sha256']
                     and _axis_sha(arrays['issue_ns'][begin:end]) == expected['decision_ns_sha256']
                     and _axis_sha(arrays['issue_ns'][begin:end] - 5 * target.MINUTE_NS)
                         == expected['bar_start_ns_sha256'], 'OCTOBER_AXIS')
        _require(int(arrays['issue_ns'][fit['rows']]) == held['first_decision_ns']
                 and int(arrays['issue_ns'][stop - 1]) == held['last_decision_ns']
                 and np.all(arrays['validity_ns'] < cutoff), 'OCTOBER_CUTOFF')
        meta = {key: source['metadata'][key] for key in ('schema_version', 'definition',
            'source_refs', 'source_row_offset', 'target_rule_locked', 'test_data_used')}
        meta.update(cutoff_ns=cutoff, rows=len(arrays['validity_ns']), parents=stop)
        series = target._series(arrays, meta)
        series_ref = target.save_dynamic_mtf_state_series(series, root / 'STATE_SERIES')
        _require(_sha(POLICY) == hashlib.sha256(policy_bytes).hexdigest()
                 and _ref(PLAN) == plan_ref
                 and all(_sha(REPO / path) == digest for path, digest in plan['source_files_sha256'].items())
                 and subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == head
                 and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == '',
                 'SOURCE_OR_POLICY_CHANGED')
        result = _publish(root, 'RESULT.json', {
            'schema_version': 'gx1_dynamic_m5_october_target_series_result_v1',
            'plan': plan_ref, 'series': series_ref, 'source': plan['original_series'],
            'prior_target_result': plan['prior_target_result'], 'axis_receipt': plan['axis_receipt'],
            'source_rows': len(arrays['validity_ns']), 'parent_rows': stop,
            'fit_parent_rows': fit['rows'], 'development_parent_rows': held['rows'],
            'cutoff_ns_exclusive': cutoff, 'model_calls': 0,
            'post_october_outcomes_decoded': 0, 'test_split_used': False,
            'independent_edge': 'UNKNOWN'})
        _publish(root, 'TERMINAL.json', {'plan': plan_ref, 'result': result,
            'exit_code': 0, 'elapsed_seconds': time.monotonic() - began,
            'source_unchanged': True, 'model_calls': 0,
            'no_relaunch_under_this_plan': True})
        return result
    except Exception as exc:
        _publish(root, 'FAILED.json', {'plan': plan_ref, 'error_type': type(exc).__name__,
                 'error': str(exc)[:2048], 'no_relaunch_under_this_plan': True})
        raise


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
