"""Narrow TRAIN-only time-query fit boundaries; no model or GPU run."""
from __future__ import annotations

import pytest
import torch

from gx1.contracts import market_state_time_query_fold1_fit_v1 as scope
from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner


def test_numerical_and_frozen_fold1_reference_match_selected_scope():
    recipe = scope.fold1_numerical_recipe()
    assert recipe['batch_size'] == 16
    assert recipe['max_updates'] == 1849
    assert recipe['seed'] == 20260911
    assert recipe['research_objective_sha256'] == scope.MODEL_CONTRACT_SHA256
    assert recipe['time_query_loss_contract_sha256'] == scope.LOSS_CONTRACT_SHA256
    report = scope._reference_report()
    fold = report['references']['folds'][0]
    assert fold['fit_rows'] == 29574
    assert fold['score_rows'] == 17753
    assert fold['official_fit_cutoff_ns'] == 1727740800000000000
    assert fold['score_end_exclusive_ns'] == scope.FOLD['oof_cutoff_ns']


def test_state_cursor_counts_skips_and_final_six_row_batch_exactly():
    order = torch.arange(scope.FIT_ROWS, dtype=torch.int64)
    recipe = scope.fold1_numerical_recipe()
    assert state._progress(0, 0, 0, order, recipe) == 0
    assert state._progress(16, 0, 1, order, recipe) == 1
    assert state._progress(scope.FIT_ROWS, 1848, 1, order, recipe) == 1849
    with pytest.raises(RuntimeError, match='EXACT_CONSUMED_BATCH_CURSOR'):
        state._progress(scope.FIT_ROWS - 1, 1848, 1, order, recipe)
    with pytest.raises(RuntimeError, match='EXACT_CONSUMED_BATCH_CURSOR'):
        state._progress(scope.FIT_ROWS, 1848, 0, order, recipe)
    with pytest.raises(RuntimeError, match='EXACT_CONSUMED_BATCH_CURSOR'):
        state._progress(16, 1, 1, order, recipe)


def test_score_only_resume_requires_completed_fit_checkpoint_and_measurement():
    identity = {'contract_sha256': scope.VIEW_SHA256}
    prior = {
        'outcome': 'RESUMABLE', 'parent_order_sha256': 'a' * 64,
        'feature_view_identity': identity, 'oof_cursor': 0,
        'phase_state': 'SCORING', 'parent_cursor': scope.FIT_ROWS,
        'optimizer_updates': 1848,
        'full_state': {'path': '/runs/states/FINAL_FOLD1_FIT_STATE.pt',
                       'sha256': 'b' * 64},
        'final_train_measurement': {'path': '/runs/FINAL_TRAIN.json',
                                    'sha256': 'c' * 64},
    }
    assert runner._require_prior_cursor(
        prior, order_sha='a' * 64, view_identity=identity) is True
    with pytest.raises(RuntimeError, match='SCORE_ONLY_RESUME_FROZEN_FIT'):
        runner._require_prior_cursor(
            {**prior, 'final_train_measurement': None},
            order_sha='a' * 64, view_identity=identity)
    with pytest.raises(RuntimeError, match='SCORE_ONLY_RESUME_FROZEN_FIT'):
        runner._require_prior_cursor(
            {**prior, 'full_state': {'path': '/runs/states/w1_p029574.pt',
                                     'sha256': 'b' * 64}},
            order_sha='a' * 64, view_identity=identity)
    with pytest.raises(RuntimeError, match='RESUME_CURSOR_IDENTITY'):
        runner._require_prior_cursor(
            {**prior, 'oof_cursor': 1},
            order_sha='a' * 64, view_identity=identity)


def test_private_probe_gate_cannot_authorize_fold1_predictor_fit():
    recipe = {'run_scope': scope.SCOPE, 'admission_stage': scope.STAGE}
    admitted = {
        'decision': 'ADMITTED', 'status': 'ADMITTED',
        'time_query_fold1_fit_authorized': True,
        'fold_id': 1, 'fit_rows': scope.FIT_ROWS,
        'score_rows': scope.SCORE_ROWS,
        'max_optimizer_updates': scope.MAX_UPDATES,
        'genuine_predictor_fit_authorized': True,
        'test_data_used': False,
    }
    scope.require_fit_gate(admitted, recipe)
    with pytest.raises(RuntimeError, match='POLICY_GATE'):
        scope.require_fit_gate({
            **admitted, 'genuine_predictor_fit_authorized': False,
        }, recipe)
    with pytest.raises(RuntimeError, match='POLICY_GATE'):
        scope.require_fit_gate({
            **admitted, 'time_query_fold1_fit_authorized': False,
        }, recipe)
