"""Synthetic streaming-score algebra; no market data, model fit or TEST."""
from __future__ import annotations

import numpy as np
import pytest

from gx1.contracts.market_state_five_endpoint_target_binding_v1 import FOLD_BOUNDARIES_NS
from gx1.contracts.market_state_time_query_scorer_v1 import (
    TimeQueryMatchedScorer, time_query_scorer_contract,
)
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TICK_NS


def _issue(day: int) -> int:
    return FOLD_BOUNDARIES_NS[0] + day * 24 * 60 * 60 * 1_000_000_000


def _block(issue_ns, start: int, length: int):
    issue_ns = np.asarray(issue_ns, dtype=np.int64)
    shape = (len(issue_ns), length, 5)
    allowed = np.full(shape, 7, dtype=np.uint8)
    eligible = np.zeros(shape, dtype=bool)
    mature = np.zeros(shape, dtype=bool)
    model = np.broadcast_to(np.array([.2, .2, .6]), shape + (3,)).copy()
    prior_three = np.broadcast_to(np.array([1 / 3] * 3), shape + (3,))
    activity = np.broadcast_to(np.array(.6), shape)
    side = np.broadcast_to(np.array(.5), shape)
    steps = np.arange(start, start + length, dtype=np.int64)
    query_k = np.broadcast_to(np.minimum(steps[:, None], np.asarray(NATIVE_K)[None, :])
                              [None, :, :], shape).copy()
    validity_ns = issue_ns[:, None, None] + query_k * TICK_NS
    mature[:] = validity_ns < FOLD_BOUNDARIES_NS[1]
    return dict(query_k=query_k, validity_ns=validity_ns,
                model_p=model, prior_activity=activity,
                prior_side_up=side, prior_three=prior_three,
                allowed=allowed, eligible=eligible, mature=mature)


def _label(block, start, row, k, tf, bit, probability):
    offset = k - start
    if not 0 <= offset < block['allowed'].shape[1]:
        return
    block['allowed'][row, offset, tf] = bit
    block['eligible'][row, offset, tf] = True
    block['model_p'][row, offset, tf] = probability


def _feed(scorer, issues, edits=None):
    scorer.begin_batch(np.array(issues, dtype=np.int64), fold_id=1)
    for start in range(1, 289, 16):
        block = _block(issues, start, min(16, 289 - start))
        if edits is not None:
            edits(block, start)
        scorer.add_block(start, **block)
    scorer.end_batch()


def test_contract_and_exact_m5_primary_with_bit5_activity_only():
    scorer = TimeQueryMatchedScorer()
    issues = [_issue(1), _issue(8)]
    def edits(block, start):
        _label(block, start, 0, 1, 0, 4, [0, .2, .8])
        _label(block, start, 1, 1, 0, 5, [.4, .2, .4])
    _feed(scorer, issues, edits)
    got = scorer.result()
    assert got['contract_sha256'] == time_query_scorer_contract()['contract_sha256']
    primary = got['primary_M5_k1']
    assert primary['certified_side']['count'] == 1
    assert primary['certified_side']['model_brier_sum'] == pytest.approx(0)
    assert primary['certified_side']['prior_brier_sum'] == pytest.approx(.25)
    assert primary['certified_side']['prior_minus_model'] == pytest.approx(.25)
    assert primary['activity']['count'] == 2
    assert primary['activity']['model_brier_sum'] == pytest.approx(.08)
    assert primary['activity']['prior_brier_sum'] == pytest.approx(.32)
    assert got['per_tf_k']['M5']['1']['active_side_unknown'] == 1
    assert got['per_tf_k']['M5']['1']['singleton'] == 1
    assert got['per_tf_k']['M5']['1']['side_known'] == 1
    assert got['per_tf_k']['M5']['1']['three_class_model_brier_sum'] == pytest.approx(.08)
    assert got['per_tf_k']['M5']['1']['three_class_prior_brier_sum'] == pytest.approx(2 / 3)
    assert got['per_tf_k']['M5']['1']['issues'] == 2
    assert got['per_tf_k']['D1']['288']['issues'] == 2
    assert got['issue_axis']['1']['count'] == 2
    assert got['test_data_used'] is False


def test_query_dependence_reduced_within_issue_before_week_and_fold():
    scorer = TimeQueryMatchedScorer()
    issues = [_issue(1), _issue(8)]
    def edits(block, start):
        _label(block, start, 0, 1, 2, 4, [0, .2, .8])
        _label(block, start, 0, 2, 2, 1, [0, .2, .8])
        _label(block, start, 1, 1, 2, 1, [.8, .2, 0])
    _feed(scorer, issues, edits)
    fold = scorer.result()['fold_after_issue_TF_reduction']['1']['H1']['conditional_side']
    assert fold['issue_count'] == 2 and fold['cell_count'] == 3
    assert fold['issue_mean_prior_minus_model'] == pytest.approx(0)
    assert fold['cell_mean_prior_minus_model'] == pytest.approx(-1 / 12)
    weeks = scorer.result()['week_after_issue_TF_reduction']['1']
    assert weeks['2024-09-30']['H1']['conditional_side']['issue_mean_prior_minus_model'] == pytest.approx(-.25)
    assert weeks['2024-10-07']['H1']['conditional_side']['issue_mean_prior_minus_model'] == pytest.approx(.25)
    assert scorer.result()['per_tf_k']['H1']['1']['side_known'] == 2
    assert scorer.result()['per_tf_k']['H1']['2']['side_known'] == 1
    assert scorer.result()['per_tf_k']['H1']['3']['side_known'] == 0


def test_immature_and_unknown_never_score():
    scorer = TimeQueryMatchedScorer()
    def edits(block, start):
        if start == 1:
            block['allowed'][0, 0, 0] = 1
            block['model_p'][0, 0, 0] = [.8, .2, 0]
    _feed(scorer, [FOLD_BOUNDARIES_NS[1] - TICK_NS], edits)
    got = scorer.result()
    assert got['primary_M5_k1']['certified_side']['count'] == 0
    assert got['primary_M5_k1']['activity']['count'] == 0
    assert got['primary_M5_k1']['certified_side']['prior_minus_model'] is None


def test_missing_duplicate_and_inactive_steps_fail_closed():
    issue = np.array([_issue(1)], dtype=np.int64)
    scorer = TimeQueryMatchedScorer()
    scorer.begin_batch(issue, fold_id=1)
    with pytest.raises(ValueError, match='STEP_ORDER'):
        scorer.add_block(2, **_block(issue, 2, 1))
    scorer.add_block(1, **_block(issue, 1, 1))
    with pytest.raises(ValueError, match='STEP_ORDER'):
        scorer.add_block(1, **_block(issue, 1, 1))
    with pytest.raises(ValueError, match='INCOMPLETE_QUERY_GRID'):
        scorer.end_batch()
    with pytest.raises(ValueError, match='UNFINISHED'):
        scorer.result()

    scorer = TimeQueryMatchedScorer()
    scorer.begin_batch(issue, fold_id=1)
    block = _block(issue, 1, 2)
    block['allowed'][0, 1, 0] = 4  # Capped M5 k=1 is present at step 2.
    block['eligible'][0, 1, 0] = True
    scorer.add_block(1, **block)  # Scorer ignores the inactive M5 slot.
    for start in range(3, 289, 16):
        scorer.add_block(start, **_block(issue, start, min(16, 289 - start)))
    scorer.end_batch()
    assert scorer.result()['per_tf_k']['M5']['1']['activity_known'] == 0


def test_score_input_and_fold_axis_guards():
    scorer = TimeQueryMatchedScorer()
    with pytest.raises(ValueError, match='ISSUE_AXIS'):
        scorer.begin_batch(np.array([FOLD_BOUNDARIES_NS[0] - TICK_NS], dtype=np.int64),
                           fold_id=1)
    issue = np.array([_issue(1)], dtype=np.int64)
    scorer.begin_batch(issue, fold_id=1)
    bad = _block(issue, 1, 1)
    bad['prior_three'] = np.zeros((1, 1, 5, 3), dtype=np.float64)
    with pytest.raises(ValueError, match='PROBABILITY_SUM'):
        scorer.add_block(1, **bad)
    bad = _block(issue, 1, 1)
    bad['model_p'][0, 0, 0] = [0, 1, 0]
    bad['allowed'][0, 0, 0] = 4
    bad['eligible'][0, 0, 0] = True
    with pytest.raises(ValueError, match='UNDEFINED_CONDITIONAL_SIDE'):
        scorer.add_block(1, **bad)


def test_query_and_validity_must_match_the_model_offsets():
    issue = np.array([_issue(1)], dtype=np.int64)
    scorer = TimeQueryMatchedScorer()
    scorer.begin_batch(issue, fold_id=1)
    wrong = _block(issue, 1, 1)
    wrong['query_k'][0, 0, 2] = 2
    with pytest.raises(ValueError, match='QUERY_K'):
        scorer.add_block(1, **wrong)
    wrong = _block(issue, 1, 1)
    wrong['validity_ns'][0, 0, 2] += TICK_NS
    with pytest.raises(ValueError, match='VALIDITY_CLOCK'):
        scorer.add_block(1, **wrong)
    wrong = _block(issue, 1, 1)
    wrong['mature'][0, 0, 2] = False
    with pytest.raises(ValueError, match='MASKS'):
        scorer.add_block(1, **wrong)
