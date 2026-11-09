"""Focused zero-update policy/result boundaries and independent score oracle."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.contracts import market_state_native_F_private_campaign_v1 as scope
from gx1.scripts import run_market_state_native_F_private_window_v1 as runner


def test_scores_use_only_known_rows_and_stable_logits():
    z = np.array([np.log(3), -np.log(3), 1000., -1000.])
    known = np.array([True, True, False, False])
    result = runner._m5_import_score(z, known, np.array([1., 0., 0., 1.]))
    assert result['count'] == 2 and result['up_count'] == 1
    assert result['brier'] == pytest.approx(0.0625, abs=1e-15)
    assert result['log'] == pytest.approx(-np.log(0.75), abs=1e-15)
    assert result['mean_p'] == pytest.approx(0.5, abs=1e-15)
    with pytest.raises(RuntimeError, match='IMPORT_SCORE_INPUT'):
        runner._m5_import_score(z, np.zeros(4, dtype=bool), z)


def _gate():
    return dict(decision='ADMITTED', status='ADMITTED', ta_m5_import_authorized=True,
        fold_id=1, parent_selection=scope.TA_PARENT_SELECTION, selected_fit_rows=32,
        model_constructions_authorized=3, optimizer_constructions_authorized=3,
        model_forwards_authorized=6, backward_calls_authorized=0,
        optimizer_steps_authorized=0, training_enabled=False, native_training_enabled=False,
        genuine_predictor_fit_authorized=False, genuine_updates_authorized=False,
        genuine_backward_authorized=False, test_data_used=False)


@pytest.mark.parametrize('key', ['optimizer_steps_authorized', 'backward_calls_authorized',
    'genuine_updates_authorized', 'genuine_backward_authorized', 'training_enabled',
    'native_training_enabled', 'genuine_predictor_fit_authorized', 'test_data_used'])
def test_import_gate_never_grants_training(key):
    gate = _gate()
    recipe = dict(run_scope=scope.TA_M5_IMPORT_SCOPE, admission_stage=scope.TA_M5_IMPORT_STAGE)
    scope.require_ta_m5_import_gate(gate, recipe)
    gate[key] = 1 if type(gate[key]) is int else True
    with pytest.raises(scope.campaign.RandomAccessCampaignError):
        scope.require_ta_m5_import_gate(gate, recipe)


@pytest.mark.parametrize('final_readout', [False, True])
@pytest.mark.parametrize('success', [False, True])
def test_result_cursor_accepts_honest_success_or_failure(tmp_path, success, final_readout):
    path = scope.TA_M5_FINAL_IMPORT_RECIPE_PATH if final_readout else scope.TA_M5_IMPORT_RECIPE_PATH
    recipe = dict(path=str(path), sha256='a' * 64)
    checkpoint = tmp_path / 'STATE.pt'
    checkpoint.write_bytes(b'synthetic-state')
    final = dict(path=str(checkpoint), sha256=scope.campaign.file_sha256(checkpoint)) if success else None
    report = dict(schema_version=scope.TA_M5_IMPORT_RESULT_SCHEMA, recipe=recipe,
        parent_order_sha256='b' * 64, parent_cursor=0, test_data_used=False,
        physical_optimizer_updates=0, final_state=final,
        decision='PASS_TA_M5_EQUAL_WEIGHT_IMPORT' if success else 'FAILED',
        bit_exact_resume=success)
    path = tmp_path / 'RESULT.json'
    path.write_text(json.dumps(report))
    value = dict(schema_version=scope.TA_M5_IMPORT_CURSOR_SCHEMA, recipe=recipe,
        outcome='COMPLETE' if success else 'FAILED', optimizer_updates=0, parent_cursor=0,
        parent_order_sha256='b' * 64, full_state=final, test_data_used=False,
        result=dict(path=str(path), sha256=scope.campaign.file_sha256(path)))
    value['cursor_sha256'] = scope.campaign.canonical_sha256(value)
    assert scope.require_ta_m5_import_cursor(value, expected_recipe=recipe) == value
    for key, bad in [('optimizer_updates', 1), ('parent_cursor', 16), ('test_data_used', True)]:
        changed = copy.deepcopy(value)
        changed[key] = bad
        changed.pop('cursor_sha256')
        changed['cursor_sha256'] = scope.campaign.canonical_sha256(changed)
        with pytest.raises(scope.campaign.RandomAccessCampaignError):
            scope.require_ta_m5_import_cursor(changed, expected_recipe=recipe)
    if success:
        report['bit_exact_resume'] = False
        path.write_text(json.dumps(report))
        value['result']['sha256'] = scope.campaign.file_sha256(path)
        value.pop('cursor_sha256')
        value['cursor_sha256'] = scope.campaign.canonical_sha256(value)
        with pytest.raises(scope.campaign.RandomAccessCampaignError, match='CURSOR_RESULT'):
            scope.require_ta_m5_import_cursor(value, expected_recipe=recipe)
