"""Focused cold coverage/source/report gates; no run, model or producer."""
from copy import deepcopy

import pytest

from gx1.audit import qualify_market_state_ta_cold_full_later_inputs_v1 as cold
from tests.test_entry_anchor_span_feature_inputs_v1 import cold_full_metadata_case


@pytest.fixture(scope='module')
def retained_proof_case():
    view, _ = cold.cold_views()
    span, gather, st, gt, fold2_view = (cold.checked_json(cold.INPUTS[k]) for k in
        ('span_result', 'fold2_gather', 'span_terminal', 'fold2_gather_terminal', 'fold2_view'))
    return view, span, gather, st, gt, fold2_view


def test_cold_rebind_matches_current_function_and_full_physical_fold2_source(retained_proof_case):
    view, span, gather, st, gt, fold2_view = retained_proof_case
    sidecars, populations = cold.reuse_proof(span, gather, st, gt, fold2_view, view)
    assert view['contract_sha256'] == cold.COLD_VIEW_SHA
    assert len(sidecars) == len(populations) == 6
    assert gather['counts']['parent_rows'] == 47338
    assert cold.overlay.cold_full_later_panel()['score_parent_start'] + 17489 == 47338


@pytest.mark.parametrize('fault',['terminal','all_rows','physical_fields','sidecar_path','overlap'])
def test_cold_reuse_proof_rejects_wrong_receipt_or_physical_identity(fault, retained_proof_case):
    view, span, gather, st, gt, fold2_view = retained_proof_case
    view, span, gather, st, gt, fold2_view = deepcopy((view, span, gather, st, gt, fold2_view))
    if fault == 'terminal':st['exit_code'] = 1
    elif fault == 'all_rows':gather['counts']['parent_rows'] -= 1
    elif fault == 'physical_fields':view['physical_field_names_sha256'] = '0'*64
    elif fault == 'sidecar_path':span['surfaces']['M5']['artifact']['path'] += '.other'
    else:span['surfaces']['D1']['all_overlap_columns_bitexact'] = False
    with pytest.raises(RuntimeError, match='TA_COLD_FULL_LATER_'):
        cold.reuse_proof(span, gather, st, gt, fold2_view, view)


def test_coverage_record_uses_all_17489_rows_and_exact_tail():
    view, expected, axes, pops, sidecars, positions = cold_full_metadata_case()
    plan = dict(panel=cold.overlay.cold_full_later_panel(),
                physical_populations=pops, physical_sidecars=sidecars)
    result = dict(path='/home/andre2/GX1_RUNS/SYNTHETIC_COLD/RESULT.json', sha256='f'*64)
    actual = cold.coverage_record(plan, view, axes, result)
    assert actual['schema_version'] == cold.overlay.COLD_FULL_LATER_COVERAGE_SCHEMA
    assert actual['status'] == 'COMPLETE'
    assert actual['selected_parent_rows_sha256'] == expected['selected_parent_rows_sha256']
    assert actual['selected_decision_ns_sha256'] == expected['selected_decision_ns_sha256']
    assert positions[-1] == 47337 and actual['tail_rows'] == 1
    plan['physical_sidecars']['M5']['sha256'] = 'g'*64
    with pytest.raises(RuntimeError, match='LATER_COLD_SIDECAR_REF'):
        cold.coverage_record(plan, view, axes, result)


def test_terminal_reports_success_and_consumed_failure_without_relaunch():
    ref = dict(path='/home/andre2/GX1_RUNS/SYNTHETIC/RESULT.json', sha256='f'*64)
    kwargs = dict(plan_ref=ref, claim=ref, head='a'*40, result=ref,
                  coverage=ref, failure=None, source_unchanged=True)
    complete = cold.terminal_record('COMPLETE', **kwargs)
    assert complete['exit_code'] == 0 and complete['no_relaunch'] is True
    failed = cold.terminal_record('FAILED_CONSUMED', **{**kwargs, 'coverage':None, 'failure':ref})
    assert failed['exit_code'] == 1 and failed['result'] == ref and failed['no_relaunch'] is True
    with pytest.raises(RuntimeError, match='TERMINAL_STATE'):
        cold.terminal_record('COMPLETE', **{**kwargs, 'coverage':None})
    with pytest.raises(RuntimeError, match='TERMINAL_STATE'):
        cold.terminal_record('FAILED_CONSUMED', **kwargs)


def test_preflight_hashes_actual_axis_and_all_six_sidecars():
    plan = cold.build_plan()
    cold.require_physical_refs(plan)
    tampered = deepcopy(plan)
    tampered['physical_sidecars']['D1']['sha256'] = '0'*64
    with pytest.raises(RuntimeError, match='PHYSICAL_INPUT_SHA'):
        cold.require_physical_refs(tampered)
