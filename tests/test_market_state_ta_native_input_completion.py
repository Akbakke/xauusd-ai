"""Focused reuse/census/publication tests; no market payload or predictor."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gx1.audit import complete_market_state_ta_native_input_path_v1 as owner


def test_scope_reuse_allows_only_the_exact_single_stage_assignment():
    old = b'FOLD = {"fit_rows": 29574}\n' + owner.OLD_ASSIGNMENT + b'\nBATCH_SIZE = 16\n'
    new = old.replace(owner.OLD_ASSIGNMENT, owner.NEW_ASSIGNMENT, 1)
    assert owner.require_stage_only_transition(old, new)['all_other_bytes_identical'] is True
    with pytest.raises(RuntimeError, match='SCOPE_STAGE_ONLY_BYTE_PARITY'):
        owner.require_stage_only_transition(old, new.replace(b'BATCH_SIZE = 16', b'BATCH_SIZE = 32'))
    with pytest.raises(RuntimeError, match='SCOPE_STAGE_ONLY_BYTE_PARITY'):
        owner.require_stage_only_transition(old + owner.OLD_ASSIGNMENT, new)


def test_interval_reconstruction_preserves_unconsumed_gaps_and_rejects_overlap():
    rows = owner.rows_from_intervals([[2, 5], [8, 10]], row_count=12)
    assert rows.tolist() == [2, 3, 4, 8, 9]
    for bad in ([[2, 5], [4, 8]], [[2, 5], [5, 8]], [[-1, 2]], [[10, 13]]):
        with pytest.raises(RuntimeError, match='INTERVAL_ORDER'):
            owner.rows_from_intervals(bad, row_count=12)


def test_raw_census_reports_missingness_without_imputation_and_hashes_exact_subset():
    matrix = np.array([[0., 1., np.nan], [2., 3., 4.], [5., 6., np.inf], [7., 8., 9.]], dtype=np.float32)
    rows = np.array([0, 2, 3], dtype=np.int64)
    columns = np.array([0, 2], dtype=np.int64)
    report = owner.raw_census(matrix, rows, columns, ['first', 'last'], batch_rows=2, digest_columns=[1])
    assert report['finite_rows'] == [3, 1]
    assert report['missing_rows'] == [0, 2]
    assert report['raw_min'] == [0., 9.]
    assert report['raw_max'] == [7., 9.]
    assert report['raw_nonzero_rows'] == [2, 1]
    assert report['checked_values'] == 6
    assert report['selected_fields_sha256'] == hashlib.sha256(np.ascontiguousarray(matrix[rows, 2:3]).tobytes()).hexdigest()


def _evidence():
    phase = {'audit_scope': {}, 'candidate_view_sha256': 'view-sha',
        'model_constructions': 1, 'model_forwards': 1, 'limits_of_evidence': ['macro provenance remains bounded'],
        **{key: {} for key in ('gather', 'normalization', 'family_consumption',
            'local_formula_alias_bridge', 'local_context_formula_evidence')}}
    diagnostic = {'diagnostic': {'measurements': {'actual2010': {'mismatched_cells': 0},
        'actual2010_producer': {'producer_vs_cache': {'float32_bit_mismatched_cells': 0}}},
        'source_comparison': {'actual_first_ns': 1276466400000000000}}}
    view = {'contract_sha256': 'view-sha', 'entry_execution_profile': {'name': 'TA'}}
    return phase, diagnostic, view


def test_composition_keeps_reused_forward_and_new_zero_work_distinct():
    phase, diagnostic, view = _evidence()
    mtf = {'all_consumed_mtf_rows_covered': True}
    result = owner.compose_input_audit(phase, diagnostic, view, mtf, {'path': '/bound/census', 'sha256': 'a' * 64})
    assert result['model_constructions'] == result['model_forwards'] == 1
    assert result['evidence_execution']['this_invocation']['model_forwards'] == 0
    assert result['evidence_execution']['reused_tensor_phase'] == owner.PHASE_REF
    assert result['local_trendline_formula_evidence']['replayed_in_completion'] is False
    assert result['limits_of_evidence'][0] == phase['limits_of_evidence'][0]
    assert 'Normalized extrema' in result['limits_of_evidence'][1]
    phase['model_forwards'] = 0
    with pytest.raises(RuntimeError, match='COMPOSED_EVIDENCE_SCOPE'):
        owner.compose_input_audit(phase, diagnostic, view, mtf, {})


@pytest.mark.parametrize('fail_after_tf', [False, True])
def test_run_publishes_complete_or_preserves_partial_failure_with_real_writers(tmp_path, monkeypatch, fail_after_tf):
    from gx1.contracts import gx1_capped_execution_v1 as guards
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
    output = tmp_path / 'run'
    output.mkdir()
    candidate = output / 'CANDIDATE_VIEW.json'
    candidate.write_text('{}\n')
    plan_path, policy_path = tmp_path / 'plan.json', tmp_path / 'policy.json'
    for path in (plan_path, policy_path):
        path.write_text('{}\n')
    monkeypatch.setattr(owner, 'OUT_ROOT', output)
    monkeypatch.setattr(owner, 'PLAN_PATH', plan_path)
    monkeypatch.setattr(owner, 'POLICY_PATH', policy_path)
    monkeypatch.setattr(guards, 'require_capped_cpu_research_producer_execution', lambda: dict(owner.LIMITS))
    monkeypatch.setattr(owner.io, '_clean_head', lambda: 'test-head')
    plan_ref, policy_ref = owner._ref(plan_path), owner._ref(policy_path)
    plan = {'source_owners': {}, 'source_reuse_binding': {}, 'formula_inputs': {},
        'inputs': {'candidate_view': owner._ref(candidate)}, 'this_invocation': dict(owner.THIS_INVOCATION)}
    phase, diagnostic, view = _evidence()
    provider = SimpleNamespace(validate_metadata=lambda _plan: {'metadata': {}, 'reuse': {'scratch_files': {}}})
    provider_plan = {'payloads': {}}
    prepared = {'cache': object()}
    released = []
    def inspect(ref, *, require_policy):
        assert ref == plan_ref and require_policy is True
        owner.need({p.name for p in output.iterdir()} == {'CANDIDATE_VIEW.json'}, 'UNCONSUMED_OUTPUT_ROOT')
        return plan, policy_ref, 'test-head', {}, view, phase, diagnostic, {}
    monkeypatch.setattr(owner, 'inspect', inspect)
    monkeypatch.setattr(runner, '_prepared_fold1', lambda *_args, **_kwargs:
        (provider, provider_plan, prepared, np.array([0, 1], np.int64), np.array([10, 20], np.int64)))
    monkeypatch.setattr(mature, '_release_pages', lambda _provider, value: released.append(value))
    def census(*_args, publish_measurement):
        ref = publish_measurement('RAW_CENSUS_AND_FORMULA_UNIONS.json', {'census_complete': True})
        return {}, {}, ref
    monkeypatch.setattr(owner, 'census_and_unions', census)
    def gaps(*_args, publish_measurement):
        ref = publish_measurement('MTF_HISTORY_M5.json', {'mismatched_cells': 0, 'additional_rows': 2})
        if fail_after_tf:
            raise RuntimeError('injected later TF failure')
        return {'all_consumed_mtf_rows_covered': True, 'measurement_refs': {'M5': ref}}
    monkeypatch.setattr(owner, 'complete_mtf_gaps', gaps)
    prior_calls = []
    def priors(rows, issues):
        prior_calls.append((rows.copy(), issues.copy()))
        return {'fit_rows': owner.fold.FIT_ROWS, 'test_fixture': True}
    monkeypatch.setattr(owner, '_fit_priors', priors)
    if fail_after_tf:
        with pytest.raises(RuntimeError, match='injected later TF failure'):
            owner.run(plan_ref)
        failure = json.loads((output / 'FAILED.json').read_text())
        terminal = json.loads((output / 'TERMINAL.json').read_text())
        assert terminal['status'] == 'FAILED_CONSUMED'
        assert terminal['failure'] == owner._ref(output / 'FAILED.json')
        assert failure['completed_measurements']['MTF_HISTORY_M5.json'] == owner._ref(output / 'MTF_HISTORY_M5.json')
        assert not (output / 'RESULT.json').exists() and not (output / 'ENDPOINT_PRIORS.json').exists()
        assert prior_calls == []
    else:
        returned = owner.run(plan_ref)
        result = json.loads((output / 'RESULT.json').read_text())
        terminal = json.loads((output / 'TERMINAL.json').read_text())
        assert result['status'] == terminal['status'] == 'COMPLETE'
        assert terminal['result'] == returned['result'] == owner._ref(output / 'RESULT.json')
        assert terminal['endpoint_priors'] == result['endpoint_priors'] == owner._ref(output / 'ENDPOINT_PRIORS.json')
        assert result['model_constructions'] == result['model_forwards'] == result['gather_calls'] == 0
        assert result['input_audit']['model_forwards'] == 1
        assert result['endpoint_prior_fit_rows'] == owner.fold.FIT_ROWS
        assert result['raw_census'] == owner._ref(output / 'RAW_CENSUS_AND_FORMULA_UNIONS.json')
        assert len(prior_calls) == 1 and not (output / 'FAILED.json').exists()
    assert released == [prepared]
    assert terminal['no_relaunch_under_this_plan'] is True
    assert terminal['claim'] == owner._ref(output / 'CLAIM.json')
    with pytest.raises(RuntimeError, match='UNCONSUMED_OUTPUT_ROOT'):
        owner.run(plan_ref)
