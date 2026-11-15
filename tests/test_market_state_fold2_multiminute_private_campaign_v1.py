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
