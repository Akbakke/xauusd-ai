"""Focused prior fit and receipt guards; no audit execution or model."""
from copy import deepcopy

import pytest

from gx1.audit import qualify_market_state_ta_cold_fold1_priors_v1 as cold
from tests.test_market_state_time_query_native_F_objective import _endpoint_fixture


def test_synthetic_mature_fold1_fit_uses_current_dedicated_contract():
    state_bound, future_bound = _endpoint_fixture()
    fit = state_bound.mature_oof_axes(fold_id=1)
    prior = cold.fit_prior(state_bound, future_bound, fit)
    assert prior['fit_rows'] == 2
    assert prior['loss_contract_sha256'] == cold.LOSS_SHA
    assert prior['fit_parent_rows_sha256'] == fit['fit_parent_rows_sha256']
    assert prior['fit_decision_ns_sha256'] == fit['fit_decision_ns_sha256']
    assert set(prior['tasks']) == {'state', 'future'}


def test_stale_sealed_prior_and_wrong_fit_axis_fail_closed():
    state_bound, future_bound = _endpoint_fixture()
    fit = state_bound.mature_oof_axes(fold_id=1)
    current = cold.fit_prior(state_bound, future_bound, fit)
    old = cold.objective._seal({**{k:v for k,v in current.items() if k != 'contract_sha256'},
                                'loss_contract_sha256':'f5a79eacd89aefd878c6cc823e616456fead2e7748c5e589b2a2d8f7a73b30cb'})
    with pytest.raises(ValueError, match='NATIVE_F_ENDPOINT_PRIOR_BINDING'):
        cold.validate_prior(old, fit)
    wrong = dict(fit, fit_parent_rows=fit['fit_parent_rows'][1:])
    with pytest.raises(ValueError, match='EXACT_FIT_AXIS'):
        cold.fit_prior(state_bound, future_bound, wrong)


def test_completed_b_receipt_chain_is_exact_and_wrong_coverage_fails():
    view, result, coverage, terminal = (cold.checked_json(cold.B_REFS[k]) for k in
        ('view', 'result', 'coverage', 'terminal'))
    cold.verify_b_receipts(view, result, coverage, terminal)
    tampered = deepcopy(coverage)
    tampered['score_rows'] -= 1
    with pytest.raises(RuntimeError, match='B_COVERAGE_CHAIN'):
        cold.verify_b_receipts(view, result, tampered, terminal)
    tampered_terminal = deepcopy(terminal)
    tampered_terminal['coverage']['sha256'] = '0'*64
    with pytest.raises(RuntimeError, match='B_COVERAGE_CHAIN'):
        cold.verify_b_receipts(view, result, coverage, tampered_terminal)


def test_plan_binds_actual_target_refs_and_no_model_work():
    plan = cold.build_plan()
    assert plan['b_refs'] == cold.B_REFS
    assert plan['target_refs'] == cold.target_refs()
    assert plan['scope']['fit_rows'] == 29574
    assert plan['scope']['later_coverage_rows'] == 17489
    assert plan['scope']['model_constructions'] == 0
    assert plan['scope']['optimizer_steps'] == 0
    assert plan['scope']['test_data_used'] is False


def test_terminal_success_and_consumed_failure_are_exclusive():
    ref = dict(path='/home/andre2/GX1_RUNS/SYNTHETIC/RESULT.json', sha256='f'*64)
    kwargs = dict(plan_ref=ref, claim=ref, head='a'*40,
                  prior=ref, result=ref, failure=None, source_unchanged=True)
    assert cold.terminal_record('COMPLETE', **kwargs)['exit_code'] == 0
    failed = cold.terminal_record('FAILED_CONSUMED',
        **{**kwargs,'failure':ref,'result':None})
    assert failed['exit_code'] == 1 and failed['no_relaunch'] is True
    with pytest.raises(RuntimeError, match='TERMINAL_STATE'):
        cold.terminal_record('COMPLETE', **{**kwargs,'prior':None})
