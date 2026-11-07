"""Synthetic factorized proper scores and matched aggregation; no market reads."""
from __future__ import annotations

import numpy as np
import pytest

from gx1.contracts import market_state_time_query_loss_v1 as old_loss
from gx1.contracts.market_state_time_query_scorer_v1 import TimeQueryMatchedScorer
from gx1.contracts.market_state_time_query_scorer_v2 import (
    FactorizedTimeQueryMatchedScorer, factorized_probabilities,
    factorized_time_query_scorer_contract, proper_score_sums,
)
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
from tests.test_market_state_time_query_scorer_v1 import _block, _issue


def _raw(batch=1, queries=1, activity=np.log(4.0), side=np.log(3.0)):
    return np.broadcast_to(np.array([activity, side], dtype=np.float32),
                           (batch, queries, 5, 2)).copy()


def _feed(scorer, issue, *, activity: float, side: float, label: int):
    scorer.begin_batch(issue, fold_id=1)
    for start in range(1, 289, 16):
        block = _block(issue, start, min(16, 289 - start))
        raw = _raw(len(issue), block['allowed'].shape[1], activity, side)
        if start == 1:
            block['allowed'][0, 0, 0] = label
            block['eligible'][0, 0, 0] = True
        classes, _, _ = factorized_probabilities(raw)
        if isinstance(scorer, FactorizedTimeQueryMatchedScorer):
            block.pop('model_p')
            scorer.add_block(start, raw_logits=raw, **block)
        else:
            block['model_p'] = classes
            scorer.add_block(start, **block)
    scorer.end_batch()
    return scorer.result()


def test_raw_proper_score_matches_old_probability_score_on_ordinary_logits():
    raw = np.array([
        [[1.3, -2.2], [-1.0, 0.4], [0.5, 1.1], [2.0, -0.8], [0.0, 0.0]],
        [[-0.2, 1.5], [0.7, -0.6], [1.0, 0.3], [-1.2, 0.9], [0.8, -1.8]],
    ], dtype=np.float32)
    allowed = np.array([[1, 2, 4, 5, 7], [5, 4, 1, 2, 7]], dtype=np.uint8)
    ready = np.ones((2, 5), dtype=np.bool_)
    query = np.broadcast_to(np.asarray(NATIVE_K, dtype=np.int64), (2, 5)).copy()
    classes, _, _ = factorized_probabilities(raw)
    got = proper_score_sums(raw, allowed, ready, ready, query)
    old = old_loss.proper_score_sums(classes, allowed, ready, ready, query)
    assert got['contract_sha256'] == factorized_time_query_scorer_contract()['contract_sha256']
    assert got['query_k_sha256'] == old['query_k_sha256']
    for tf in got['per_tf']:
        for metric in got['per_tf'][tf]:
            for key, value in got['per_tf'][tf][metric].items():
                reference = old['per_tf'][tf][metric][key]
                if isinstance(value, float):
                    assert value == pytest.approx(reference, abs=2e-7)
                else:
                    assert value == reference


def test_underflow_retains_conditional_side_and_finite_proper_log():
    raw = np.zeros((1, 5, 2), dtype=np.float32)
    raw[0, 0] = [-1000.0, 0.0]
    allowed = np.full((1, 5), 7, dtype=np.uint8)
    allowed[0, 0] = 4
    ready = np.ones((1, 5), dtype=np.bool_)
    query = np.asarray(NATIVE_K, dtype=np.int64)[None, :]
    classes, _, side_up = factorized_probabilities(raw)
    assert classes[0, 0].tolist() == [0.0, 1.0, 0.0]
    assert side_up[0, 0] == 0.5
    with pytest.raises(ValueError, match='UNDEFINED_CONDITIONAL_SIDE'):
        old_loss.proper_score_sums(classes, allowed, ready, ready, query)
    got = proper_score_sums(raw, allowed, ready, ready, query)
    assert got['per_tf']['M5']['conditional_side']['count'] == 1
    assert got['per_tf']['M5']['conditional_side']['brier_sum'] == pytest.approx(.25)
    assert got['per_tf']['M5']['conditional_side']['log_sum'] == pytest.approx(np.log(2))
    assert got['per_tf']['M5']['three_class']['brier_sum'] == pytest.approx(2.0)
    assert got['per_tf']['M5']['three_class']['log_sum'] == pytest.approx(1000 + np.log(2))
    assert got['per_tf']['M5']['conditional_side']['zero_truth_probability_rows'] == 0


def test_matched_v2_parity_and_extreme_underflow_without_floor():
    issue = np.array([_issue(1)], dtype=np.int64)
    ordinary_old = _feed(TimeQueryMatchedScorer(), issue,
                         activity=float(np.log(4)), side=float(np.log(3)), label=4)
    ordinary_new = _feed(FactorizedTimeQueryMatchedScorer(), issue,
                         activity=float(np.log(4)), side=float(np.log(3)), label=4)
    for metric in ('certified_side', 'activity'):
        for key, value in ordinary_new['primary_M5_k1'][metric].items():
            assert value == pytest.approx(ordinary_old['primary_M5_k1'][metric][key])
    assert ordinary_new['contract_sha256'] == factorized_time_query_scorer_contract()['contract_sha256']
    assert ordinary_new['issue_axis'] == ordinary_old['issue_axis']
    assert ordinary_new['per_tf_k']['M5']['1']['side_known'] == 1

    extreme = _feed(FactorizedTimeQueryMatchedScorer(), issue,
                    activity=-1000.0, side=0.0, label=4)
    assert extreme['primary_M5_k1']['certified_side']['model_brier_sum'] == pytest.approx(.25)
    assert extreme['primary_M5_k1']['activity']['model_brier_sum'] == pytest.approx(1.0)
    assert extreme['per_tf_k']['M5']['1']['three_class_model_brier_sum'] == pytest.approx(2.0)


def test_unknown_and_immature_are_not_side_scored():
    raw = np.zeros((2, 5, 2), dtype=np.float32)
    allowed = np.array([[5, 7, 4, 1, 2], [4, 4, 4, 4, 4]], dtype=np.uint8)
    eligible = np.ones((2, 5), dtype=np.bool_)
    mature = eligible.copy()
    mature[1] = False
    query = np.broadcast_to(np.asarray(NATIVE_K, dtype=np.int64), (2, 5)).copy()
    report = proper_score_sums(raw, allowed, eligible, mature, query)['per_tf']
    assert report['M5']['activity']['count'] == 1
    assert report['M5']['conditional_side']['count'] == 0
    assert report['M15']['activity']['count'] == 0
    assert report['H1']['conditional_side']['count'] == 1
    assert report['H4']['conditional_side']['count'] == 1
    assert report['D1']['conditional_side']['count'] == 0


def test_unrepresentable_log_sum_fails_closed_and_bad_block_does_not_advance():
    raw = np.zeros((1, 5, 2), dtype=np.float64)
    raw[0, 0] = [-np.finfo(np.float64).max, np.finfo(np.float64).max]
    allowed = np.full((1, 5), 7, dtype=np.uint8)
    allowed[0, 0] = 1
    ready = np.ones((1, 5), dtype=np.bool_)
    query = np.asarray(NATIVE_K, dtype=np.int64)[None, :]
    with pytest.raises(ValueError, match='NONFINITE_PROPER_LOG'):
        proper_score_sums(raw, allowed, ready, ready, query)
    issue = np.array([_issue(1)], dtype=np.int64)
    scorer = FactorizedTimeQueryMatchedScorer()
    scorer.begin_batch(issue, fold_id=1)
    block = _block(issue, 1, 1)
    block.pop('model_p')
    bad = _raw()
    bad[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='RAW_LOGITS'):
        scorer.add_block(1, raw_logits=bad, **block)
    scorer.add_block(1, raw_logits=_raw(), **block)


def test_explicit_side_rejects_mismatch_before_accumulator_change():
    issue = np.array([_issue(1)], dtype=np.int64)
    scorer = TimeQueryMatchedScorer()
    scorer.begin_batch(issue, fold_id=1)
    block = _block(issue, 1, 1)
    block['allowed'][0, 0, 0] = 4
    block['eligible'][0, 0, 0] = True
    supplied = np.full(block['allowed'].shape, .5, dtype=np.float64)
    with pytest.raises(ValueError, match='MODEL_SIDE_PARITY'):
        scorer.add_block(1, model_side_up=supplied, **block)
    supplied[0, 0, 0] = .75
    scorer.add_block(1, model_side_up=supplied, **block)
    assert scorer._active['next_step'] == 2
