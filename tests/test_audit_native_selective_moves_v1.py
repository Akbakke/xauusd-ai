"""Synthetic first-warning, causal-path and one-use auditor checks; no market data."""
from __future__ import annotations
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from gx1.audit import audit_native_selective_moves_v1 as audit

T = audit.TICK


def path_case(*, omitted=(), both=False, no_hit=False, closed=()):
    starts = np.asarray([i*T for i in range(6) if i not in omitted], np.int64)
    prices = np.tile(np.array([100., 100.4, 99.6, 100.]), (len(starts), 1))
    if not no_hit:
        at = np.flatnonzero(starts == 2*T)
        if len(at):
            prices[at[0], 1] = 102.
            if both: prices[at[0], 2] = 98.
    closed_mask = np.zeros(8, np.bool_)
    closed_mask[list(closed)] = True
    table = audit.gap_table(starts, grid_start=0, upper=8*T, closed=closed_mask)
    return starts, prices, table


def event(*, atr=1., atr_available=T, end=6*T, **kwargs):
    starts, prices, table = path_case(**kwargs)
    return audit.scan_event(starts, prices, anchor=T, end=end, target_start=4*T,
                            atr=atr, atr_available=atr_available, table=table)


def test_first_cross_uses_known_anchor_and_five_minute_interval():
    e = event()
    assert e['kind'] == audit.UP
    assert (e['hit_start'], e['hit_end']) == (2*T, 3*T)
    assert e['anchor_price'] == 100. and e['pretarget_hit'] is True
    assert e['complete'] is True


def test_same_first_bar_crossing_both_sides_never_gets_an_invented_order():
    assert event(both=True)['kind'] == audit.BOTH


def test_gap_before_cross_is_unknown():
    e = event(omitted=(1,))
    assert e['kind'] == audit.UNKNOWN and e['reason'] == 'PREFIX_GAP_BEFORE_HIT'


def test_gap_after_observed_cross_keeps_known_first_hit_but_not_complete_end():
    e = event(omitted=(3,))
    assert e['kind'] == audit.UP and not e['complete'] and e['end_row'] == -1


def test_manifest_declared_weekly_closure_is_the_only_gap_exception():
    assert event(omitted=(1,), closed=(1,))['kind'] == audit.UP


def test_observed_price_in_declared_closure_is_rejected():
    with pytest.raises(ValueError, match='PRICE_INSIDE_DECLARED_WEEKLY_CLOSURE'):
        path_case(closed=(1,))


def test_none_requires_complete_expected_open_window():
    assert event(no_hit=True)['kind'] == audit.NONE
    assert event(no_hit=True, omitted=(1,))['kind'] == audit.UNKNOWN


@pytest.mark.parametrize('atr', [0., -1., np.nan, np.inf])
def test_invalid_atr_is_unknown_without_substituting_a_future_or_new_scale(atr):
    assert event(atr=atr)['kind'] == audit.UNKNOWN


def test_atr_available_after_anchor_is_unknown():
    assert event(atr_available=2*T)['reason'] == 'MISSING_OR_NONCAUSAL_ATR'


def test_outcome_end_at_or_after_train_cutoff_is_unknown():
    assert event(end=8*T)['kind'] == audit.UNKNOWN
    assert event(end=9*T)['kind'] == audit.UNKNOWN


def test_anchor_price_comes_from_last_already_closed_m5_bar():
    starts, prices, table = path_case()
    prices[1, 3] = 1000.
    e = audit.scan_event(starts, prices, anchor=T, end=6*T, target_start=4*T,
                         atr=1., atr_available=T, table=table)
    assert e['anchor_price'] == 100.


def test_first_warning_is_chronological_even_if_later_score_is_stronger():
    scores = np.array([.61, .99, .40, .81])
    begins = np.array([0, 2])
    assert audit.first_warnings(scores, begins, .60).tolist() == [0, 3]
    assert audit.first_warnings(scores, begins, .90).tolist() == [1, -1]


def test_clock_groups_keep_first_issue_anchor_and_do_not_drop_invalid_clock_rows():
    issues = np.array([T, 2*T, 3*T, 4*T])
    begins, ends, eligible = audit.clock_groups(issues, np.array([3*T, 3*T, -1, -1]), np.array([4*T, 4*T, -1, -1]))
    assert begins.tolist() == [0, 2, 3] and ends.tolist() == [2, 3, 4]
    assert eligible.tolist() == [True, True, False, False]


def test_same_target_clock_with_conflicting_end_is_rejected():
    with pytest.raises(ValueError, match='GROUP_VALIDITY_CONSTANT'):
        audit.clock_groups(np.array([T, 2*T]), np.array([3*T, 3*T]), np.array([4*T, 5*T]))


def test_timely_wrong_late_unknown_none_are_exhaustive_and_precision_keeps_unknown():
    kinds = np.array([audit.UP, audit.DOWN, audit.UP, audit.NONE, audit.UNKNOWN, audit.BOTH])
    c = audit.classify_warnings(kinds, np.array([2*T]*6), np.array([T, T, 3*T, T, T, T]),
                                np.array([audit.UP]*6), np.array([True]*6))
    assert c['correct_timely'].sum() == c['wrong_direction'].sum() == c['late_correct_direction'].sum() == 1
    assert c['no_hit'].sum() == c['unknown'].sum() == c['both_same_bar'].sum() == 1


def tiny_analysis(*, both=False):
    starts, prices, table = path_case(both=both)
    issues = np.array([T, 2*T, 3*T, 4*T], np.int64)
    # Earliest warning is DOWN; later UP must never replace it using the outcome.
    side = np.tile(np.array([-10., 10., 10., 10.], np.float32)[:, None], (1, 5))
    capture = {'issue_ns': issues, 'activity_logits': np.full((4, 5), 10., np.float32), 'side_logits': side}
    target = {'target_start_ns': np.full((4, 5), 4*T, np.int64), 'validity_ns': np.full((4, 5), 6*T, np.int64),
              'atr14_asof': np.ones((4, 5)), 'atr_asof_available_ns': np.tile(issues[:, None], (1, 5))}
    prior = {'per_task': {task: {tf: {'probabilities': [1-p, p]} for tf in audit.TF}
                          for task, p in (('activity', .9), ('side', .6))}}
    return audit.analyse(capture, target, starts, prices, table, prior)


def test_all_five_timeframes_and_all_thresholds_are_reported_without_winner():
    r = tiny_analysis()
    assert tuple(r) == audit.TF
    for tf in audit.TF:
        row = r[tf]
        assert row['issue_rows'] == row['group_rows_total'] == 4
        assert row['unique_target_window_groups'] == 1
        assert row['score_distribution']['count'] == 4 and row['predicted_UP_fraction'] == .75
        assert set(row['thresholds']) == {'0.50', '0.60', '0.70', '0.80', '0.90'}
        for cell in row['thresholds'].values():
            m = cell['model']
            assert m['classification']['wrong_direction'] == 1
            assert m['captured_known_moves'] == 0 and m['missed_known_moves'] == 1
            assert m['selected_precision_lower'] == m['selected_precision_upper'] == 0
            assert cell['always_UP_at_model_warning_issues']['captured_known_moves'] == 1
            assert cell['always_DOWN_at_model_warning_issues']['selected_warnings'] == m['selected_warnings'] == 1
            assert cell['constant_TRAIN_prior_at_model_warning_issues']['captured_known_moves'] == 1
            assert m['PnL'] == m['costs'] == m['MAE_MFE'] == 'NOT_MEASURED'
        assert row['thresholds']['0.50']['constant_TRAIN_prior_same_threshold']['selected_warnings'] == 1
        assert row['thresholds']['0.60']['constant_TRAIN_prior_same_threshold']['selected_warnings'] == 0


def test_same_bar_ambiguity_stays_in_precision_denominator_bounds():
    cell = tiny_analysis(both=True)['M5']['thresholds']['0.50']['model']
    assert cell['classification']['both_same_bar'] == 1
    assert cell['selected_precision_lower'] == 0 and cell['selected_precision_upper'] == 1
    assert cell['known_moves'] == 0 and cell['known_move_coverage'] is None


def test_full_clock_requires_actual_price_and_capture_match_not_just_same_count(monkeypatch):
    monkeypatch.setattr(audit, 'ROWS', 3)
    monkeypatch.setattr(audit, 'START', 0)
    monkeypatch.setattr(audit, 'UPPER', 10*T)
    starts = np.array([0, T, 2*T])
    c = {'issue_ns': starts+T, 'activity_logits': np.zeros((3, 5), np.float32), 'side_logits': np.zeros((3, 5), np.float32)}
    target = {'issue_ns': starts+T}
    audit.validate_full_clock(c, target, starts)
    bad = copy.deepcopy(c); bad['issue_ns'][1] += T
    with pytest.raises(ValueError, match='HUNDRED_PERCENT_TRAIN_CLOCK'):
        audit.validate_full_clock(bad, target, starts)
    bad = copy.deepcopy(c); bad['side_logits'] = np.zeros((3, 4), np.float32)
    with pytest.raises(ValueError, match='ALL_FIVE_FINITE_HEADS'):
        audit.validate_full_clock(bad, target, starts)


def test_npz_header_validation_has_no_payload_load_and_requires_exact_inventory(monkeypatch, tmp_path):
    path = tmp_path/'a.npz'; np.savez(path, x=np.ones((2, 5), np.float32))
    h = {'x': {'shape': [2, 5], 'dtype': '<f4'}}
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('header check decoded payload'))
    audit.archive_headers({'path': str(path)}, h, exact_inventory={'x'})
    with pytest.raises(ValueError, match='NPZ_EXACT_KEYS'):
        audit.archive_headers({'path': str(path)}, h, exact_inventory={'x', 'extra'})
    with pytest.raises(ValueError, match='NPY_HEADER'):
        audit.archive_headers({'path': str(path)}, {'x': {'shape': [3, 5], 'dtype': '<f4'}})


def test_loaded_array_bytes_must_match_metadata(tmp_path):
    path = tmp_path/'a.npz'; a = np.ones((2, 5), np.float32); np.savez(path, x=a, unused=np.ones(1))
    h = {'x': {'shape': [2, 5], 'dtype': '<f4', 'sha256': audit.array_sha(a)}}
    assert list(audit.load_fields({'path': str(path)}, ('x',), h)) == ['x']
    h['x']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='PAYLOAD_ARRAY_IDENTITY'):
        audit.load_fields({'path': str(path)}, ('x',), h)


def metadata_fixture():
    capture = {'path': '/synthetic/CAPTURE.npz', 'sha256': 'a'*64}
    identity = {'original_MAIN_result': {'path': '/main/result', 'sha256': 'b'*64},
                'original_MAIN_plan': {'path': '/main/plan', 'sha256': 'c'*64},
                'checkpoint_receipt': {'path': '/main/final', 'sha256': 'd'*64}, 'model_state_sha256': 'e'*64}
    prereg = {'model_identity': identity}
    plan = {'inputs': {'main_result': identity['original_MAIN_result'], 'main_plan': identity['original_MAIN_plan'],
                       'checkpoint_receipt': identity['checkpoint_receipt'], 'preregistration': audit.PREREG}}
    r = {'rows': audit.ROWS, 'batch_rows': 256, 'batch_count': 1385, 'capture': capture,
         'capture_array_headers': {k: {'shape': [audit.ROWS, 5] if k.endswith('_logits') else [audit.ROWS],
                                     'dtype': '<f4' if k.endswith('_logits') else '<i8', 'sha256': 'f'*64} for k in audit.CAPTURE_KEYS},
         'reference_producer': identity['original_MAIN_result'], 'original_FINAL_components': {'model_state': 'e'*64},
         'model_state_restored': {'model_state_sha256': 'e'*64}, 'original_PASS1_batch_hashes_all_bitexact': True,
         'original_PASS1_full_output_and_proper_scores_bitexact': True}
    result = {'status': 'COMPLETE_NATIVE_FULL_TRAIN_CAPTURE', 'source_unchanged': True, 'test_data_used': False, 'report': r}
    return result, capture, plan, prereg


def test_complete_capture_metadata_binds_original_main_final_and_prereg():
    audit.validate_capture_metadata(*metadata_fixture())


@pytest.mark.parametrize('bad', ['sparse', 'four_tf', 'wrong_dtype', 'changed_model', 'not_bitexact', 'TEST', 'wrong_prereg'])
def test_capture_metadata_fails_closed_for_wrong_scope_or_producer(bad):
    result, capture, plan, prereg = metadata_fixture(); r = result['report']
    if bad == 'sparse': r['rows'] = 20727
    if bad == 'four_tf': r['capture_array_headers']['side_logits']['shape'][1] = 4
    if bad == 'wrong_dtype': r['capture_array_headers']['side_logits']['dtype'] = '<f8'
    if bad == 'changed_model': r['model_state_restored']['model_state_sha256'] = '0'*64
    if bad == 'not_bitexact': r['original_PASS1_batch_hashes_all_bitexact'] = False
    if bad == 'TEST': result['test_data_used'] = True
    if bad == 'wrong_prereg': plan['inputs']['preregistration'] = {'path': '/other', 'sha256': '0'*64}
    with pytest.raises(ValueError): audit.validate_capture_metadata(result, capture, plan, prereg)


def lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, 'ROOT', tmp_path/'OUT')
    binding = {'path': str(tmp_path/'PLAN.json'), 'sha256': '1'*64}
    pref = {'path': str(tmp_path/'POLICY.json'), 'sha256': '2'*64}
    monkeypatch.setattr(audit, 'require_capped_cpu_audit_execution', lambda: {'memory_max': 4*1024**3,
        'memory_high': 4*1024**3, 'swap': 512*1024**2, 'numerical_threads': 1, 'cgroup_path': '/sys/fs/cgroup/synthetic'})
    monkeypatch.setattr(audit, 'require_capped_lock_ancestry', lambda: {'scope': '/synthetic'})
    monkeypatch.setattr(audit, 'inspect', lambda *a, **k: ({}, binding, pref))
    monkeypatch.setattr(audit, 'checked_inputs', lambda p: None)
    monkeypatch.setattr(audit, 'source_clean', lambda p: True)
    original_ref = audit.ref
    monkeypatch.setattr(audit, 'ref', lambda p: pref if p == audit.POLICY else binding if p == audit.PLAN else original_ref(p))
    return binding, pref


def test_success_publishes_consumed_receipts_with_zero_models_and_fits(monkeypatch, tmp_path):
    lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(audit, 'calculate', lambda p: {'synthetic': True})
    output = audit.run('1'*64)
    terminal = json.loads((audit.ROOT/'TERMINAL.json').read_text())
    result = json.loads((audit.ROOT/'RESULT.json').read_text())
    assert terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
    assert terminal['remaining_invocations'] == 0 and terminal['relaunch_allowed'] is False
    assert terminal['result'] == output['result']
    assert result['model_forwards'] == result['checkpoint_loads'] == result['optimizer_updates'] == result['fits'] == 0
    with pytest.raises(FileExistsError): audit.run('1'*64)


def test_failure_keeps_consumed_claim_and_terminal_without_success_result(monkeypatch, tmp_path):
    lifecycle(monkeypatch, tmp_path)
    def failed(_): raise ValueError('synthetic')
    monkeypatch.setattr(audit, 'calculate', failed)
    with pytest.raises(ValueError, match='synthetic'): audit.run('1'*64)
    terminal = json.loads((audit.ROOT/'TERMINAL.json').read_text())
    assert terminal['status'] == 'FAILED_CONSUMED' and terminal['remaining_invocations'] == 0
    assert terminal['result'] is None and (audit.ROOT/'CLAIM.json').is_file()
    assert not (audit.ROOT/'RESULT.json').exists()


def test_preflight_has_no_claim_or_decode(monkeypatch, tmp_path, capsys):
    lifecycle(monkeypatch, tmp_path)
    monkeypatch.setattr(audit, 'calculate', lambda p: pytest.fail('preflight decoded outcomes'))
    monkeypatch.setattr(sys, 'argv', ['audit', '--plan-sha256', '1'*64, '--preflight'])
    audit.main()
    out = json.loads(capsys.readouterr().out)
    assert out['array_payloads_decoded'] == out['parquet_rows_decoded'] == out['models'] == 0
    assert out['claim_created'] is False and not audit.ROOT.exists()


def test_gate_keeps_all_training_and_test_authority_closed():
    gate = audit.gate_expected({'path': '/plan', 'sha256': 'a'*64}, True)
    assert gate['execution_authorized'] is True and gate['remaining_invocations'] == 1
    assert gate['training_authorized'] is gate['test_authorized'] is gate['relaunch_allowed'] is False


def test_late_warning_never_gets_a_backwards_warning_to_hit_markout():
    starts, prices, table = path_case()
    issues = np.array([T, 2*T, 3*T, 4*T], np.int64)
    events = audit.event_arrays(issues, np.full(4, 4*T), np.full(4, 6*T), np.ones(4), issues,
                                starts, prices, table)
    r = audit.warning_summary(events, np.array([2]), np.full(4, audit.UP), issues, starts, prices)
    assert r['classification']['late_correct_direction'] == 1
    assert r['gross_mid_return_warning_close_to_hitbar_close_bps'] is None


@pytest.mark.parametrize('target_start', [6*T, 7*T])
def test_target_start_at_or_after_end_stays_unknown_even_with_observed_hit(target_start):
    starts, prices, table = path_case()
    e = audit.scan_event(starts, prices, anchor=T, end=6*T, target_start=target_start,
                         atr=1., atr_available=T, table=table)
    assert e['kind'] == audit.UNKNOWN and e['reason'] == 'INVALID_OR_IMMATURE_WINDOW'



def past_case(*, omitted=(), closed=()):
    starts = np.asarray([i*T for i in range(8) if i not in omitted], np.int64)
    prices = np.tile(np.array([100., 101., 90., 100.]), (len(starts), 1))
    for start, close in ((T, 99.), (3*T, 97.)):
        at = np.flatnonzero(starts == start)
        if len(at): prices[at[0], 3] = close
    closed_mask = np.zeros(10, np.bool_); closed_mask[list(closed)] = True
    table = audit.gap_table(starts, grid_start=0, upper=10*T, closed=closed_mask)
    return starts, prices, table


def test_anchor_past_direction_uses_only_already_available_closes():
    starts, prices, table = past_case()
    assert audit.anchor_past_direction(starts, prices, anchor=4*T, duration=2*T, table=table) == audit.DOWN
    prices[starts >= 4*T, 3] = 100000.
    assert audit.anchor_past_direction(starts, prices, anchor=4*T, duration=2*T, table=table) == audit.DOWN


def test_missing_open_past_bar_or_missing_history_stays_unknown_past():
    starts, prices, table = past_case(omitted=(2,))
    assert audit.anchor_past_direction(starts, prices, anchor=4*T, duration=2*T, table=table) == audit.UNKNOWN
    starts, prices, table = past_case()
    assert audit.anchor_past_direction(starts, prices, anchor=4*T, duration=5*T, table=table) == audit.UNKNOWN


def test_missing_bar_at_past_reference_cannot_silently_substitute_an_older_quote():
    starts, prices, table = past_case(omitted=(1,))
    assert audit.anchor_past_direction(starts, prices, anchor=4*T, duration=2*T, table=table) == audit.UNKNOWN


def test_declared_weekly_closure_is_exempt_in_past_direction_too():
    starts, prices, table = past_case(omitted=(2,), closed=(2,))
    assert audit.anchor_past_direction(starts, prices, anchor=4*T, duration=2*T, table=table) == audit.DOWN


def test_exact_zero_past_direction_has_no_epsilon_or_invented_side():
    starts, prices, table = past_case()
    prices[starts == 3*T, 3] = 99.
    assert audit.anchor_past_direction(starts, prices, anchor=4*T, duration=2*T, table=table) == audit.NONE


def test_event_strata_are_exhaustive_and_separate_reversals_from_continuations():
    kinds = np.array([audit.UP, audit.DOWN, audit.UP, audit.UP, audit.NONE, audit.UNKNOWN, audit.BOTH])
    past = np.array([audit.DOWN, audit.UP, audit.UP, audit.NONE, audit.UNKNOWN, audit.UP, audit.UP])
    strata = audit.event_strata(kinds, past)
    assert strata['REVERSAL'].tolist() == [True, True, False, False, False, False, False]
    assert strata['CONTINUATION'].sum() == 1 and strata['NO_PAST_DIRECTION'].sum() == 1
    assert strata['UNKNOWN_PAST'].sum() == 1 and strata['NO_KNOWN_DIRECTIONAL_EVENT'].sum() == 2
    assert sum(int(v.sum()) for v in strata.values()) == len(kinds)


def test_strict_anchor_warning_is_distinct_from_later_warning_before_first_hit():
    starts, prices, table = path_case()
    issues = np.array([T, 2*T, 3*T, 4*T], np.int64)
    events = audit.event_arrays(issues, np.full(4, 4*T), np.full(4, 6*T), np.ones(4), issues,
                                starts, prices, table)
    first = audit.warning_summary(events, np.array([0]), np.full(4, audit.UP), issues, starts, prices)
    later = audit.warning_summary(events, np.array([1]), np.full(4, audit.UP), issues, starts, prices)
    assert first['correct_timely_at_anchor'] == 1 and first['correct_later_before_1ATR_hit'] == 0
    assert later['correct_timely_at_anchor'] == 0 and later['correct_later_before_1ATR_hit'] == 1
    assert 'latent physical turn onset' in first['timing_limit']


def test_reversal_stratum_reports_timely_warning_progress_and_headroom():
    starts, prices, table = past_case()
    issues = np.array([4*T, 5*T, 6*T], np.int64)
    prices[starts >= 4*T] = [97., 97.4, 96.6, 97.]
    prices[starts == 5*T, 1] = 99.
    events = audit.event_arrays(issues, np.full(3, 6*T), np.full(3, 8*T), np.ones(3), issues,
                                starts, prices, table)
    report = audit.warning_summary(events, np.array([0]), np.full(3, audit.UP), issues, starts, prices)
    reverse = report['event_strata']['REVERSAL']
    assert reverse['eligible_known_moves'] == reverse['captured_known_moves'] == 1
    assert reverse['correct_timely_at_anchor'] == 1
    assert reverse['correct_timely_progress_ATR']['mean'] == 0.
    assert reverse['correct_timely_headroom_ATR']['mean'] == 1.
    assert report['warning_relation_to_past_at_GROUP_ANCHOR']['OPPOSES_PAST']['selected_warnings'] == 1
