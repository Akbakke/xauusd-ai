"""S-only private metadata must stay distinct from fit and dual-head scopes."""
import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts import market_state_fold2_multiminute_private_campaign_v1 as scope
from gx1.contracts.market_state_fold2_multiminute_private_v1 import (
    MODEL_CONTRACT_SHA256, LOSS_CONTRACT_SHA256, numerical_recipe,
)
from gx1.contracts.market_state_fold2_multiminute_side_loss_v1 import contract as loss
from gx1.contracts.market_state_time_query_model_v2 import factorized_time_query_model_contract


def _ref(path: Path) -> dict:
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _json_ref(path: Path, value: dict) -> dict:
    path.write_text(json.dumps(value, sort_keys=True))
    return _ref(path)


def test_numerical_identity_matches_live_model_and_loss_contracts():
    assert MODEL_CONTRACT_SHA256 == factorized_time_query_model_contract()['contract_sha256']
    assert LOSS_CONTRACT_SHA256 == loss()['contract_sha256']
    recipe = numerical_recipe()
    assert recipe['research_objective_sha256'] == MODEL_CONTRACT_SHA256
    assert recipe['fold2_25_cell_s_loss_contract_sha256'] == LOSS_CONTRACT_SHA256
    assert recipe['batch_size'] == 16 and recipe['max_updates'] == 2


def test_exact_private_gate_has_no_full_fit_authority():
    recipe = {'run_scope': scope.SCOPE, 'admission_stage': scope.STAGE}
    gate = {'decision': 'ADMITTED', 'status': 'ADMITTED',
            'fold2_25_cell_s_private_authorized': True, 'fold_id': 2,
            'sample_parent_rows': list(scope.PARENT_ROWS),
            'model_forwards_authorized': 6, 'backward_calls_authorized': 4,
            'optimizer_steps_authorized': 4,
            'genuine_predictor_fit_authorized': False, 'test_data_used': False}
    scope.require_gate(gate, recipe)
    for key, wrong in (('fold2_25_cell_s_private_authorized', False),
                       ('genuine_predictor_fit_authorized', True),
                       ('optimizer_steps_authorized', 5), ('test_data_used', True)):
        bad = {**gate, key: wrong}
        with pytest.raises(Exception, match='POLICY_GATE'):
            scope.require_gate(bad, recipe)


@pytest.mark.parametrize('complete', [False, True])
def test_cursor_checks_immutable_terminal_result(tmp_path, complete):
    recipe = {'path': str(scope.RECIPE_PATH), 'sha256': 'a' * 64}
    order_sha = 'b' * 64
    updates, position = (4, 32) if complete else (0, 0)
    state_ref = None
    if complete:
        checkpoint = tmp_path / 'FULL_STATE.pt'
        checkpoint.write_bytes(b'synthetic state reference')
        state_ref = _ref(checkpoint)
    report = {
        'schema_version': scope.RESULT_SCHEMA, 'recipe': recipe,
        'parent_order_sha256': order_sha, 'parent_cursor': position,
        'physical_optimizer_updates': updates, 'final_state': state_ref,
        'decision': scope.PASS_DECISION if complete else 'FAILED',
        'bit_exact_resume': complete, 'test_data_used': False,
    }
    result_ref = _json_ref(tmp_path / 'RESULT.json', report)
    cursor = scope.build_time_query_private_cursor(
        recipe_ref=recipe, outcome='COMPLETE' if complete else 'FAILED',
        optimizer_updates=updates, parent_cursor=position,
        parent_order_sha256=order_sha, full_state=state_ref,
        result=result_ref)
    scope.require_cursor(cursor, expected_recipe=recipe, verify_files=True,
                         batch_size=16)
    wrong = {**cursor, 'optimizer_updates': updates + 1}
    with pytest.raises(Exception, match='CURSOR_FIELDS|CURSOR_SCOPE'):
        scope.require_cursor(wrong, expected_recipe=recipe, verify_files=False,
                             batch_size=16)
    if complete:
        report['bit_exact_resume'] = False
        result_ref = _json_ref(tmp_path / 'RESULT.json', report)
        rebound = scope.build_time_query_private_cursor(
            recipe_ref=recipe, outcome='COMPLETE', optimizer_updates=4,
            parent_cursor=32, parent_order_sha256=order_sha,
            full_state=state_ref, result=result_ref)
        with pytest.raises(Exception, match='CURSOR_RESULT'):
            scope.require_cursor(rebound, expected_recipe=recipe,
                                 verify_files=True, batch_size=16)


def test_feature_fit_cutoff_does_not_censor_s_target_day():
    import numpy as np
    from gx1.contracts.market_state_fold2_multiminute_path_v1 import (
        QUERY_MINUTES, gather_fixed_path,
    )
    from gx1.contracts.market_state_minute_query_candidate_v1 import MinuteQueryTargetTape
    from gx1.contracts.market_state_time_query_target_v1 import MINUTE_NS

    issue = np.array([scope.FEATURE_FIT_CUTOFF_NS - 5 * MINUTE_NS], np.int64)
    validity = np.unique(issue[0] + QUERY_MINUTES.reshape(-1) * MINUTE_NS)
    bits = np.full((len(validity), 5), 2, np.uint8)
    tape = MinuteQueryTargetTape(
        validity, np.arange(len(validity), dtype=np.int64), bits,
        np.ones_like(bits, dtype=np.bool_), np.zeros(len(validity), np.uint8))
    actual = gather_fixed_path(tape, issue,
        fold_upper_ns=scope.S_TARGET_FOLD_UPPER_NS)
    wrong = gather_fixed_path(tape, issue,
        fold_upper_ns=scope.FEATURE_FIT_CUTOFF_NS)
    assert actual['mature'][0, -1, -1] and actual['eligible'][0, -1, -1]
    assert not wrong['mature'][0, -1, -1]


@pytest.mark.parametrize('complete', [False, True])
def test_new_scope_publishes_complete_or_failed_terminal(tmp_path, monkeypatch, complete):
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.scripts import run_market_state_time_query_private_window_v1 as runner

    output = tmp_path / 'private-research'
    output.mkdir()
    recipe = {'path': str(scope.RECIPE_PATH), 'sha256': 'a' * 64}
    window = {'recipe': recipe,
              'campaign_cursor_path': str(tmp_path / 'cursor' / 'pointer.json')}
    progress_path = tmp_path / 'progress' / 'invocation.json'
    counters = {'json_bytes': 0, 'optimizer_updates': 4 if complete else 0}
    final_state = None
    if complete:
        path = tmp_path / 'FULL_STATE.pt'
        path.write_bytes(b'synthetic checkpoint, never used for fit')
        final_state = _ref(path)

    def progress(**kwargs):
        return {'outcome': kwargs['outcome'],
                'checkpoint_pointer': kwargs['checkpoint_pointer']}

    def require_progress(value, **kwargs):
        cursor_ref = value['checkpoint_pointer']
        cursor = json.loads(Path(cursor_ref['path']).read_text())
        scope.require_cursor(cursor, expected_recipe=recipe, verify_files=True,
                             batch_size=16)
        assert kwargs['verify_file'] is True
        return value

    monkeypatch.setattr(authority, 'build_research_progress', progress)
    monkeypatch.setattr(authority, 'require_research_progress', require_progress)
    payload = {'schema_version': scope.RESULT_SCHEMA,
               'decision': scope.PASS_DECISION if complete else 'FAILED',
               'recipe': recipe, 'parent_order_sha256': 'b' * 64,
               'parent_cursor': 32 if complete else 0,
               'physical_optimizer_updates': counters['optimizer_updates'],
               'final_state': final_state, 'bit_exact_resume': complete,
               'test_data_used': False}
    result = runner._publish_terminal(
        output=output, progress_path=progress_path,
        window=window, plan={'plan_sha256': 'c' * 64},
        invocation={'kind': 'synthetic-metadata-only'},
        payload=payload, counters=counters, maximum=100000, scope=scope)
    assert Path(result['result']['path']).is_file()
    assert Path(result['checkpoint_pointer']['path']).is_file()
    assert json.loads(progress_path.read_text())['outcome'] == (
        'COMPLETE' if complete else 'FAILED')


@pytest.mark.parametrize('complete', [False, True])
def test_canonical_progress_validates_new_scope_cursor(tmp_path, complete):
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority

    recipe = {'path': str(scope.RECIPE_PATH), 'sha256': 'a' * 64}
    order_sha = 'b' * 64
    updates, position = (4, 32) if complete else (0, 0)
    state_ref = None
    if complete:
        state_path = tmp_path / 'FULL_STATE.pt'
        state_path.write_bytes(b'synthetic private state')
        state_ref = _ref(state_path)
    result_ref = _json_ref(tmp_path / 'RESULT.json', {
        'schema_version': scope.RESULT_SCHEMA, 'recipe': recipe,
        'parent_order_sha256': order_sha, 'parent_cursor': position,
        'physical_optimizer_updates': updates, 'final_state': state_ref,
        'decision': scope.PASS_DECISION if complete else 'FAILED',
        'bit_exact_resume': complete, 'test_data_used': False,
    })
    cursor = scope.build_time_query_private_cursor(
        recipe_ref=recipe, outcome='COMPLETE' if complete else 'FAILED',
        optimizer_updates=updates, parent_cursor=position,
        parent_order_sha256=order_sha, full_state=state_ref,
        result=result_ref)
    cursor_ref = _json_ref(tmp_path / 'RESUME_CURSOR.json', cursor)
    invocation = {
        'kind': authority.KIND, 'invocation_sha256': 'c' * 64,
        'direction_research_recipe': recipe, 'batch_size': 16,
        'optimizer_step_budget': 4,
        'checkpoint': {'pointer_path': cursor_ref['path']},
    }
    plan = {'plan_sha256': 'd' * 64}
    progress = authority.build_research_progress(
        window={'recipe': recipe}, plan=plan, invocation=invocation,
        checkpoint_pointer=cursor_ref, optimizer_updates=updates,
        parent_cursor=position, parent_order_sha256=order_sha,
        outcome='COMPLETE' if complete else 'FAILED')
    checked = authority.require_research_progress(
        progress, plan_sha256=plan['plan_sha256'], invocation=invocation,
        expected_selection_receipt_sha256=None, verify_file=True)
    assert checked['completed_units'] == updates
    assert checked['total_units'] == 4
