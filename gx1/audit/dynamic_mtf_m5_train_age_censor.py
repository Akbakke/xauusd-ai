"""One bounded TRAIN-only check of confirmed age and M5 +5 first-gap censor.

Consumes the previously published, verified pre-October target series. This is
an aggregate data audit, not a label change, model fit or October outcome read.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry


REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / 'configs/research/DYNAMIC_M5_TRAIN_AGE_CENSOR_20261105_001.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
POLICY_KEY = 'dynamic_m5_train_age_censor_20261105_001'
HORIZON = 5


def _require(ok, reason):
    if not ok:
        raise RuntimeError('DYNAMIC_M5_TRAIN_AGE_CENSOR_' + reason)


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _ref(path):
    return {'path': str(path), 'sha256': _sha_file(path)}


def _read_ref(ref):
    _require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'} and _ref(ref['path']) == ref,
             'SOURCE_REF')
    return json.loads(Path(ref['path']).read_text())


def _axis_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<i8').tobytes()).hexdigest()


def _publish(root, name, value, max_bytes):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _require(len(raw) <= max_bytes, 'OUTPUT_SIZE:' + name)
    path = root / name
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    path.chmod(0o444)
    return _ref(path)


def first_gap_m5_support(axis, issues, positions, state, observed, path_observed, cutoff_ns):
    """Stop each +5 path at its first absent/unknown M1 close or cutoff."""
    minute = target.MINUTE_NS
    _require(axis.dtype == issues.dtype == positions.dtype == np.int64
             and all(x.ndim == 1 for x in (axis, issues, positions, state, observed, path_observed))
             and len(axis) == len(state) == len(observed) == len(path_observed)
             and len(issues) == len(positions) > 0 and len(axis) > 0
             and np.all(np.diff(axis) > 0) and np.all(axis < cutoff_ns)
             and np.all(issues < cutoff_ns) and np.all(np.diff(issues) > 0)
             and np.all((positions >= -1) & (positions < len(axis)))
             and state.dtype == np.int8 and observed.dtype == path_observed.dtype == np.bool_,
             'INPUT_SHAPE_CLOCK')
    starts = {'missing_issue_clock': 0, 'unknown_state': 0, 'unknown_path': 0}
    classes = {name: 0 for name in target.STATE_NAMES}
    gaps = {'none': 0, 'cutoff': 0, 'missing_m1': 0, 'unknown_state': 0, 'unknown_path': 0}
    followed = [0] * (HORIZON + 1)
    events = {'first_exit': [0] * HORIZON, 'first_exit_then_return': [0] * HORIZON}
    absent = {'first_exit': 0, 'first_exit_then_return': 0}
    censored = {'first_exit': 0, 'first_exit_then_return': 0}
    at_risk = {'first_exit': 0, 'first_exit_then_return': 0}
    event_times = {'first_exit': set(), 'first_exit_then_return': set()}
    known_starts = 0
    for issue, start_pos in zip(issues, positions):
        issue, start_pos = int(issue), int(start_pos)
        if start_pos < 0:
            starts['missing_issue_clock'] += 1
            continue
        _require(int(axis[start_pos]) == issue, 'ISSUE_SOURCE_CLOCK')
        if not observed[start_pos]:
            starts['unknown_state'] += 1
            continue
        if not path_observed[start_pos]:
            starts['unknown_path'] += 1
            continue
        first_state = int(state[start_pos])
        _require(first_state in (target.DOWN, target.FLAT, target.UP), 'START_CLASS')
        classes[target.STATE_NAMES[first_state]] += 1
        known_starts += 1
        seen = exit_lead = return_lead = 0
        reason = 'none'
        for lead in range(1, HORIZON + 1):
            wanted = issue + lead * minute
            if wanted >= cutoff_ns:
                reason = 'cutoff'
                break
            row = int(np.searchsorted(axis, wanted))
            if row >= len(axis) or int(axis[row]) != wanted:
                reason = 'missing_m1'
                break
            if not observed[row]:
                reason = 'unknown_state'
                break
            if not path_observed[row]:
                reason = 'unknown_path'
                break
            seen = lead
            current = int(state[row])
            _require(current in (target.DOWN, target.FLAT, target.UP), 'FUTURE_CLASS')
            if not exit_lead and current != first_state:
                exit_lead = lead
            elif exit_lead and not return_lead and current == first_state:
                return_lead = lead
        _require((reason == 'none') == (seen == HORIZON), 'FOLLOWUP_PARTITION')
        gaps[reason] += 1
        followed[seen] += 1
        for name, event_lead in (('first_exit', exit_lead), ('first_exit_then_return', return_lead)):
            at_risk[name] += event_lead if event_lead else seen
            if event_lead:
                events[name][event_lead - 1] += 1
                event_times[name].add(issue + event_lead * minute)
            elif seen == HORIZON:
                absent[name] += 1
            else:
                censored[name] += 1
    _require(known_starts + sum(starts.values()) == len(issues)
             and sum(classes.values()) == known_starts
             and sum(followed) == known_starts
             and sum(gaps.values()) == known_starts
             and all(sum(events[name]) + absent[name] + censored[name] == known_starts for name in events)
             and sum(events['first_exit_then_return']) <= sum(events['first_exit']),
             'COUNT_PARTITION')
    return {'timeframe': 'M5', 'lead_minutes': HORIZON, 'parents': len(issues),
            'known_start_state_and_path': known_starts, 'unknown_start_reasons': starts,
            'known_start_classes': classes, 'first_unobservable_future_reason': gaps,
            'consecutive_observed_future_minutes': {str(i): n for i, n in enumerate(followed)},
            'full_five_minute_followup': followed[HORIZON],
            'events': {name: {'event_parent_issues': sum(events[name]),
                             'event_lead_counts': {str(i + 1): n for i, n in enumerate(events[name])},
                             'confirmed_absent_full_horizon': absent[name],
                             'right_censored_before_horizon': censored[name],
                             'at_risk_parent_minutes': at_risk[name],
                             'distinct_event_minutes': len(event_times[name]),
                             'distinct_event_utc_days': len({ns // (1440 * minute) for ns in event_times[name]})}
                       for name in events},
            'semantics': 'known issue state and path; exact consecutive M1 close at +1..+5; first missing clock, unknown state/path or cutoff right-censors unresolved events; no absent event claimed before full follow-up',
            'target_rule_locked': False, 'independent_edge': 'UNKNOWN'}


def _endpoint_source_check(arrays, review, cutoff_ns):
    """Reconcile the saved +5 endpoint counts without re-reading raw quotes."""
    entries = [x for x in review['native_leads'] if x['timeframe'] == 'M5' and x['lead_minutes'] == HORIZON]
    _require(len(entries) == 1, 'PRESERVED_ENDPOINT_ENTRY')
    expected = entries[0]
    wanted = arrays['issue_ns'] + HORIZON * target.MINUTE_NS
    mature = wanted < cutoff_ns
    axis = arrays['validity_ns']
    rows = np.searchsorted(axis, wanted[mature])
    present = (rows < len(axis)) & (axis[np.minimum(rows, len(axis) - 1)] == wanted[mature])
    selected = rows[present]
    reason = arrays['reason'][selected, 0]
    blocked = target.ATR_UNAVAILABLE | target.LOCAL_SPAN_INCOMPLETE | target.UNKNOWN_SOURCE_GAP
    complete = (reason & blocked) == 0
    labels = arrays['state'][selected[complete], 0]
    counts = {name: int(np.count_nonzero(labels == value)) for name, value in
              (('UNKNOWN', target.UNKNOWN), ('DOWN', target.DOWN),
               ('FLAT', target.FLAT), ('UP', target.UP))}
    actual = {'parents': len(arrays['issue_ns']), 'immature': int((~mature).sum()),
              'missing_mature_m1_endpoint': int((~present).sum()),
              'exact_mature_m1_endpoint': len(selected),
              'all_known_endpoint_classes': int(arrays['observed'][selected, 0].sum()),
              'incomplete_local_window_or_atr': int((~complete).sum()),
              'complete_local_windows': len(labels),
              'complete_window_endpoint_counts': counts}
    _require(all(expected[key] == value for key, value in actual.items()), 'PRESERVED_ENDPOINT_COUNTS')
    return actual


def _age_coverage(series):
    a = series.arrays
    result = target.query_historical_dynamic_mtf_confirmed_run_age(
        series, parent_rows=a['parent_rows'], issue_ns=a['issue_ns'], cutoff_ns=series.metadata['cutoff_ns'])
    mask = result['current_confirmed_run_age_observed']
    age = result['current_confirmed_run_age_minutes']
    _require(mask.shape == age.shape == (len(a['parent_rows']), len(target.TIMEFRAMES))
             and np.all(mask <= (result['current_observed'] & result['current_path_observed']))
             and np.all(age[~mask] == 0) and np.all(age[mask] >= 1), 'AGE_MASK')
    per_tf = {}
    for col, tf in enumerate(target.TIMEFRAMES):
        values = age[mask[:, col], col]
        per_tf[tf] = {'parents': len(a['parent_rows']),
                      'current_state_observed': int(result['current_observed'][:, col].sum()),
                      'current_state_and_path_observed': int((result['current_observed'][:, col] & result['current_path_observed'][:, col]).sum()),
                      'confirmed_age_observed': len(values),
                      'confirmed_age_unobserved': int((~mask[:, col]).sum()),
                      'first_confirmed_change_row_count': int((values == 1).sum()),
                      'age_observed_m1_rows_min': int(values.min()) if len(values) else None,
                      'age_observed_m1_rows_median': float(np.median(values)) if len(values) else None,
                      'age_observed_m1_rows_p90': float(np.quantile(values, .9)) if len(values) else None,
                      'age_observed_m1_rows_max': int(values.max()) if len(values) else None}
    return {'parents': len(a['parent_rows']), 'fit_cutoff_ns_exclusive': series.metadata['cutoff_ns'],
            'timeframes': per_tf, 'age_definition': result['age_definition'],
            'clock_basis': 'historical_event_time_only_not_online_receipt',
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
    _require(plan['schema_version'] == 'gx1_dynamic_m5_train_age_censor_plan_v1'
             and plan['status'] == 'PREPARED_NOT_ADMITTED' and plan['max_invocations'] == 1
             and plan['output_root'] == '/home/andre2/GX1_RUNS/DYNAMIC_M5_TRAIN_AGE_CENSOR_20261105_001'
             and plan['execution']['command'] == 'bash scripts/gx1_capped_run.sh --class audit --mem 2G --swap 512M -- timeout --signal=TERM --kill-after=10s 600s .venv/bin/python -m gx1.audit.dynamic_mtf_m5_train_age_censor'
             and plan['restrictions'] == {'model_construction': False, 'optimizer_updates': False,
                 'test': False, 'broker_or_network': False, 'spending_usd': 0,
                 'october_outcomes': False, 'raw_market_source_reads': False}
             and isinstance(state, dict) and state.get('status') == 'ADMITTED_ONCE'
             and state.get('plan') == plan_ref and state.get('operator_approved') is True
             and state.get('remaining_invocations') == 1
             and policy['training_enabled'] is False and policy['full_val_allowed'] is False,
             'EXPLICIT_ADMISSION_REQUIRED')
    _require(plan['standing_authorization'] == policy['no_cost_local_standing_authorization_20261105']['operator_instruction'],
             'STANDING_AUTHORIZATION')
    _require(all(_sha_file(REPO / path) == digest for path, digest in plan['source_files_sha256'].items()),
             'SOURCE_CHANGED')
    source = _read_ref(plan['train_result'])
    review = _read_ref(plan['target_review'])
    _require(_ref(plan['input_prefix_normalization']['path']) == plan['input_prefix_normalization'],
             'QUALIFIED_INPUT_PREFIX')
    _require(source['series'] == plan['series'] and source['fit_prefix'] == plan['fit_prefix']['rows']
             and source['fit_cutoff_ns'] == plan['fit_prefix']['cutoff_ns']
             and source['test_data_used'] is False and source['model_calls'] == 0
             and review['target_contract_sha256'] == plan['target_contract_sha256']
             and review['parents'] == plan['fit_prefix']['rows']
             and review['target_rule_locked'] is False, 'SOURCE_SCOPE')
    root = Path(plan['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        manifest = target.read_dynamic_mtf_state_series_metadata(
            plan['series'], expected_source_refs=plan['source_refs'],
            expected_cutoff_ns=plan['fit_prefix']['cutoff_ns'])
        _require(manifest['metadata']['rows'] == 150714
                 and manifest['metadata']['parents'] == plan['fit_prefix']['rows']
                 and manifest['metadata']['target_rule_locked'] is False,
                 'SERIES_METADATA')
        series = target.load_dynamic_mtf_state_series(
            plan['series'], expected_source_refs=plan['source_refs'],
            expected_cutoff_ns=plan['fit_prefix']['cutoff_ns'])
        a = series.arrays
        fit = plan['fit_prefix']
        _require(len(a['parent_rows']) == fit['rows']
                 and _axis_sha(a['parent_rows']) == fit['parent_rows_sha256']
                 and _axis_sha(a['issue_ns']) == fit['decision_ns_sha256']
                 and _axis_sha(a['issue_ns'] - 300 * 1_000_000_000) == fit['bar_start_ns_sha256']
                 and int(a['issue_ns'][-1]) + HORIZON * target.MINUTE_NS == fit['cutoff_ns'],
                 'PREFIX_AXIS_IDENTITY')
        endpoint = _endpoint_source_check(a, review, fit['cutoff_ns'])
        age = _age_coverage(series)
        event = first_gap_m5_support(
            a['validity_ns'], a['issue_ns'], a['issue_source_index'],
            a['state'][:, 0], a['observed'][:, 0], a['path_observed'][:, 0], fit['cutoff_ns'])
        _require(event['parents'] == age['parents'] == fit['rows'], 'POPULATION_IDENTITY')
        event['preserved_m5_plus_5_endpoint_reconciliation'] = endpoint
        age_ref = _publish(root, 'AGE_COVERAGE.json', age, plan['max_json_output_bytes'])
        event_ref = _publish(root, 'M5_FIRST_GAP.json', event, plan['max_json_output_bytes'])
        _require(_sha_file(POLICY) == hashlib.sha256(policy_bytes).hexdigest()
                 and _ref(PLAN) == plan_ref
                 and all(_sha_file(REPO / path) == digest for path, digest in plan['source_files_sha256'].items())
                 and subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == head
                 and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip() == '',
                 'SOURCE_OR_POLICY_CHANGED')
        result = {'schema_version': 'gx1_dynamic_m5_train_age_censor_result_v1',
                  'plan': plan_ref, 'train_series': plan['series'], 'target_review': plan['target_review'],
                  'age_coverage': age_ref, 'm5_first_gap': event_ref,
                  'parents': fit['rows'], 'target_rule_locked': False,
                  'target_arrays_decoded': True, 'october_outcomes_decoded': 0,
                  'model_calls': 0, 'test_data_used': False, 'independent_edge': 'UNKNOWN'}
        result_ref = _publish(root, 'RESULT.json', result, plan['max_json_output_bytes'])
        _publish(root, 'TERMINAL.json', {'plan': plan_ref, 'result': result_ref,
                 'exit_code': 0, 'source_unchanged': True,
                 'elapsed_seconds': time.monotonic() - started,
                 'no_relaunch_under_this_plan': True, 'model_calls': 0,
                 'october_outcomes_decoded': 0, 'test_data_used': False}, plan['max_json_output_bytes'])
        return result_ref
    except Exception as exc:
        _publish(root, 'FAILED.json', {'plan': plan_ref, 'error_type': type(exc).__name__,
                 'error': str(exc)[:2048], 'no_relaunch_under_this_plan': True},
                 plan['max_json_output_bytes'])
        raise


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
