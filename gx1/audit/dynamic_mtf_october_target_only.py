"""One chronological October development target coverage check, no model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit.dynamic_mtf_m5_train_age_censor import first_gap_m5_support
from gx1.audit.dynamic_mtf_train_prefix_targets import _copy_fit_arrays
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry


REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / 'configs/research/DYNAMIC_M5_OCTOBER_TARGET_ONLY_20261105_001.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
POLICY_KEY = 'dynamic_m5_october_target_only_20261105_001'


def _require(ok, reason):
    if not ok:
        raise RuntimeError('DYNAMIC_M5_OCTOBER_TARGET_' + reason)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _ref(path):
    return {'path': str(path), 'sha256': _sha(path)}


def _read_ref(ref):
    _require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'} and _ref(ref['path']) == ref,
             'SOURCE_REF')
    return json.loads(Path(ref['path']).read_text())


def _axis_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<i8').tobytes()).hexdigest()


def _publish(root, name, value, limit):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _require(len(raw) <= limit, 'OUTPUT_SIZE:' + name)
    path = root / name
    _publish_bytes(path, raw)
    return _ref(path)


def endpoint_support(axis, issues, states, observed, reason, cutoff_ns, *, tf, lead):
    """Exact mature endpoint and complete-local-window counts for one TF."""
    _require(axis.dtype == issues.dtype == np.int64 and states.dtype == np.int8
             and observed.dtype == np.bool_ and reason.dtype == np.uint8
             and axis.ndim == issues.ndim == states.ndim == observed.ndim == reason.ndim == 1
             and len(axis) == len(states) == len(observed) == len(reason) > 0
             and len(issues) > 0 and np.all(np.diff(axis) > 0) and np.all(np.diff(issues) > 0)
             and np.all(axis < cutoff_ns) and np.all(issues < cutoff_ns)
             and lead == target.LOCAL_WINDOW_M1_BARS[tf], 'ENDPOINT_INPUT')
    wanted = issues + lead * target.MINUTE_NS
    mature = wanted < cutoff_ns
    rows = np.searchsorted(axis, wanted[mature])
    present = (rows < len(axis)) & (axis[np.minimum(rows, len(axis) - 1)] == wanted[mature])
    selected = rows[present]
    blocked = target.ATR_UNAVAILABLE | target.LOCAL_SPAN_INCOMPLETE | target.UNKNOWN_SOURCE_GAP
    complete = (reason[selected] & blocked) == 0
    labels = states[selected[complete]]
    counts = {name: int(np.count_nonzero(labels == value)) for name, value in
              (('UNKNOWN', target.UNKNOWN), ('DOWN', target.DOWN), ('FLAT', target.FLAT), ('UP', target.UP))}
    _require(sum(counts.values()) == len(labels)
             and np.array_equal(observed[selected], states[selected] != target.UNKNOWN),
             'ENDPOINT_CLASS_MASK')
    return {'timeframe': tf, 'lead_minutes': lead, 'parents': len(issues),
            'immature': int((~mature).sum()), 'missing_mature_m1_endpoint': int((~present).sum()),
            'exact_mature_m1_endpoint': len(selected),
            'all_known_endpoint_classes': int(observed[selected].sum()),
            'incomplete_local_window_or_atr': int((~complete).sum()),
            'complete_local_windows': len(labels), 'complete_window_endpoint_counts': counts}


def _october_age(series, parent_rows, issues, cutoff):
    result = target.query_historical_dynamic_mtf_confirmed_run_age(
        series, parent_rows=parent_rows, issue_ns=issues, cutoff_ns=cutoff)
    mask = result['current_confirmed_run_age_observed']
    values = result['current_confirmed_run_age_minutes']
    _require(mask.shape == values.shape == (len(parent_rows), 5)
             and np.all(mask <= result['current_observed'] & result['current_path_observed'])
             and np.all(values[~mask] == 0) and np.all(values[mask] >= 1), 'AGE_MASK')
    return {'parents': len(parent_rows), 'cutoff_ns_exclusive': cutoff,
            'clock_basis': 'historical_event_time_only_not_online_receipt',
            'age_definition': result['age_definition'],
            'timeframes': {tf: {'confirmed_age_observed': int(mask[:, i].sum()),
                                'current_state_observed': int(result['current_observed'][:, i].sum()),
                                'current_state_and_path_observed': int((result['current_observed'][:, i] & result['current_path_observed'][:, i]).sum())}
                           for i, tf in enumerate(target.TIMEFRAMES)},
            'model_consumption_qualified': False, 'independent_edge': 'UNKNOWN'}


def run():
    require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    _require(Path.cwd().resolve() == REPO
             and subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'work/gx1-current'
             and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == '', 'CLEAN_REPO')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    plan_ref = _ref(PLAN)
    plan = json.loads(PLAN.read_text())
    policy_bytes = POLICY.read_bytes()
    policy = json.loads(policy_bytes)
    state = policy.get(POLICY_KEY)
    _require(plan['schema_version'] == 'gx1_dynamic_m5_october_target_only_plan_v1'
             and plan['status'] == 'PREPARED_NOT_ADMITTED' and plan['max_invocations'] == 1
             and plan['output_root'] == '/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_TARGET_ONLY_20261105_001'
             and plan['execution']['command'] == 'bash scripts/gx1_capped_run.sh --class audit --mem 2G --swap 512M -- timeout --signal=TERM --kill-after=10s 600s .venv/bin/python -m gx1.audit.dynamic_mtf_october_target_only'
             and plan['restrictions'] == {'model_construction': False, 'optimizer_updates': False,
                 'test': False, 'broker_or_network': False, 'spending_usd': 0,
                 'post_october_outcomes': False, 'raw_market_source_reads': False}
             and isinstance(state, dict) and state.get('status') == 'ADMITTED_ONCE'
             and state.get('plan') == plan_ref and state.get('operator_approved') is True
             and state.get('remaining_invocations') == 1
             and policy['training_enabled'] is False and policy['full_val_allowed'] is False,
             'EXPLICIT_ADMISSION_REQUIRED')
    _require(plan['standing_authorization'] == policy['no_cost_local_standing_authorization_20261105']['operator_instruction'],
             'STANDING_AUTHORIZATION')
    _require(all(_sha(REPO / path) == digest for path, digest in plan['source_files_sha256'].items()),
             'SOURCE_CHANGED')
    axis = _read_ref(plan['axis_receipt'])
    candidate = _read_ref(plan['candidate_design'])
    original = _read_ref(plan['original_series'])
    fit = _read_ref(plan['fit_series'])
    _require(axis['evaluation_outcomes_decoded'] == 0
             and axis['fit_prefix'] == plan['fit_prefix']
             and axis['evaluation'] == plan['evaluation_axis']
             and candidate['population']['axis_receipt'] == plan['axis_receipt']
             and candidate['status'] == 'ONE_CANDIDATE_LOCKED_SOURCE_DESIGN_ONLY_NO_MODEL_ADMISSION'
             and original['metadata']['test_data_used'] is False
             and original['metadata']['definition']['contract_sha256'] == plan['target_contract_sha256']
             and fit['metadata']['cutoff_ns'] == plan['fit_prefix']['cutoff_ns']
             and fit['metadata']['parents'] == plan['fit_prefix']['rows'], 'SOURCE_SCOPE')
    root = Path(plan['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        metadata = target.read_dynamic_mtf_state_series_metadata(
            plan['original_series'], expected_source_refs=plan['source_refs'],
            expected_cutoff_ns=original['metadata']['cutoff_ns'])
        _require(metadata['metadata']['parents'] == 64890 and metadata['metadata']['rows'] == 326055,
                 'ORIGINAL_SERIES_POPULATION')
        arrays = _copy_fit_arrays(metadata, Path(plan['original_series']['path']).parent,
                                  prefix_rows=plan['evaluation_axis']['parent_row_range'][1],
                                  source_rows=None, cutoff_ns=plan['evaluation_axis']['cutoff_ns_exclusive'])
        old = target.load_dynamic_mtf_state_series(plan['fit_series'],
            expected_source_refs=plan['source_refs'], expected_cutoff_ns=plan['fit_prefix']['cutoff_ns'])
        fit_rows = len(old.arrays['validity_ns'])
        _require(all(np.array_equal(old.arrays[name], arrays[name][:len(old.arrays[name])])
                     for name in target.SERIES_ARRAY_DTYPES)
                 and fit_rows == 150714, 'FIT_PREFIX_ARRAY_IDENTITY')
        first, last = plan['evaluation_axis']['parent_row_range']
        _require(first == plan['fit_prefix']['rows'] and last == len(arrays['parent_rows'])
                 and _axis_sha(arrays['parent_rows'][first:last]) == plan['evaluation_axis']['parent_rows_sha256']
                 and _axis_sha(arrays['issue_ns'][first:last]) == plan['evaluation_axis']['decision_ns_sha256']
                 and _axis_sha(arrays['issue_ns'][first:last] - 5 * target.MINUTE_NS)
                     == plan['evaluation_axis']['bar_start_ns_sha256']
                 and int(arrays['issue_ns'][first]) == plan['evaluation_axis']['first_decision_ns']
                 and int(arrays['issue_ns'][last - 1]) == plan['evaluation_axis']['last_decision_ns'],
                 'OCTOBER_AXIS_IDENTITY')
        meta = {key: original['metadata'][key] for key in ('schema_version','definition','source_refs',
            'source_row_offset','target_rule_locked','test_data_used')}
        meta.update(cutoff_ns=plan['evaluation_axis']['cutoff_ns_exclusive'],
                    rows=len(arrays['validity_ns']), parents=last)
        series = target._series(arrays, meta)
        issues = arrays['issue_ns'][first:last]
        endpoints = [endpoint_support(arrays['validity_ns'], issues, arrays['state'][:, col],
                     arrays['observed'][:, col], arrays['reason'][:, col], meta['cutoff_ns'], tf=tf,
                     lead=target.LOCAL_WINDOW_M1_BARS[tf]) for col, tf in enumerate(target.TIMEFRAMES)]
        age = _october_age(series, arrays['parent_rows'][first:last], issues, meta['cutoff_ns'])
        event = first_gap_m5_support(arrays['validity_ns'], issues, arrays['issue_source_index'][first:last],
                    arrays['state'][:, 0], arrays['observed'][:, 0], arrays['path_observed'][:, 0], meta['cutoff_ns'])
        _require(age['parents'] == event['parents'] == len(issues) == 6323
                 and all(x['parents'] == 6323 for x in endpoints), 'REPORT_POPULATION')
        endpoint_ref = _publish(root, 'ENDPOINT_COVERAGE.json', {'cutoff_ns_exclusive':meta['cutoff_ns'],
            'parent_range':[first,last], 'timeframes':endpoints, 'target_rule_locked':False,
            'independent_edge':'UNKNOWN'}, plan['max_json_output_bytes'])
        age_ref = _publish(root, 'AGE_INPUT_COVERAGE.json', age, plan['max_json_output_bytes'])
        event_ref = _publish(root, 'M5_FIRST_GAP.json', event, plan['max_json_output_bytes'])
        _require(_sha(POLICY) == hashlib.sha256(policy_bytes).hexdigest()
                 and _ref(PLAN) == plan_ref
                 and all(_sha(REPO / path) == digest for path, digest in plan['source_files_sha256'].items())
                 and subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == head
                 and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == '',
                 'SOURCE_OR_POLICY_CHANGED')
        result = {'schema_version':'gx1_dynamic_m5_october_target_only_result_v1',
                  'plan':plan_ref, 'candidate_design':plan['candidate_design'],
                  'original_series':plan['original_series'],'fit_series':plan['fit_series'],
                  'source_row_slice':[0,len(arrays['validity_ns'])], 'parent_range':[first,last],
                  'endpoint_coverage':endpoint_ref,'age_input_coverage':age_ref,'m5_first_gap':event_ref,
                  'model_calls':0,'post_october_outcomes_decoded':0,'test_data_used':False,
                  'independent_edge':'UNKNOWN'}
        result_ref = _publish(root,'RESULT.json',result,plan['max_json_output_bytes'])
        _publish(root,'TERMINAL.json',{'plan':plan_ref,'result':result_ref,'exit_code':0,
                 'source_unchanged':True,'elapsed_seconds':time.monotonic()-started,
                 'no_relaunch_under_this_plan':True,'model_calls':0,'test_data_used':False},
                 plan['max_json_output_bytes'])
        return result_ref
    except Exception as exc:
        _publish(root,'FAILED.json',{'plan':plan_ref,'error_type':type(exc).__name__,
                 'error':str(exc)[:2048],'no_relaunch_under_this_plan':True},plan['max_json_output_bytes'])
        raise


if __name__ == '__main__':
    print(json.dumps(run(),sort_keys=True))
