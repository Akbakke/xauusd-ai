"""Synthetic fold/mask/prior checks; no physical target or model fit."""
import hashlib

import numpy as np
import pytest

from gx1.audit.market_state_time_query_reference_core_v1 import score_reference_fold
from gx1.contracts.market_state_time_query_target_v1 import (
    MINUTE_NS, TimeQueryTargetTape,
)


def _fixture(*, missing_now=False):
    minutes = np.array([m for m in range(31) if not (missing_now and m == 15)],
                       dtype=np.int64)
    axis = minutes * MINUTE_NS
    allowed = np.full((len(minutes), 5), 7, dtype=np.uint8)
    for minute, bit in ((5, 1), (10, 5), (15, 2), (20, 1), (25, 4)):
        at = np.flatnonzero(minutes == minute)
        if len(at):
            allowed[at[0], 0] = bit
    tape = TimeQueryTargetTape(axis, np.arange(len(axis), dtype=np.int64), allowed)
    fit = np.array([0, 5, 10], dtype=np.int64) * MINUTE_NS
    score = np.array([15, 20], dtype=np.int64) * MINUTE_NS
    return tape, fit, score


def _run(tape, fit, score, *, upper=30, chunk=8):
    return score_reference_fold(tape, fit, score, fit_lower_ns=0,
                                fold_boundary_ns=15 * MINUTE_NS,
                                score_upper_ns=upper * MINUTE_NS,
                                chunk_rows=chunk)


def test_bit5_contributes_activity_but_not_side_or_three_class_prior():
    tape, fit, score = _fixture()
    result = _run(tape, fit, score)
    m5 = result['per_tf_k']['M5']['1']
    assert m5['fit']['issues'] == 3
    assert m5['fit']['mature'] == 2  # t=10 +5 is immature at the fit boundary.
    assert m5['fit']['activity_known'] == 2
    assert m5['fit']['active'] == 2
    assert m5['fit']['side_known'] == m5['fit']['singleton'] == 1
    assert m5['prior']['activity'] == pytest.approx(0.75)
    assert m5['prior']['conditional_up'] == pytest.approx(1 / 3)
    assert m5['prior']['three_class'] == pytest.approx([0.5, 0.25, 0.25])
    assert m5['score']['activity_known'] == m5['score']['side_known'] == 2
    assert m5['score']['activity_prior_brier_sum'] == pytest.approx(0.125)
    assert m5['score']['side_prior_brier_sum'] == pytest.approx(5 / 9)
    assert m5['score']['three_class_prior_brier_sum'] == pytest.approx(1.25)
    assert result['fit_issue_axis_sha256'] == hashlib.sha256(fit.tobytes()).hexdigest()
    assert result['score_issue_axis_sha256'] == hashlib.sha256(score.tobytes()).hexdigest()


def test_causal_persistence_abstains_on_flat_or_unknown_side_and_pairs_prior():
    tape, fit, score = _fixture()
    m5 = _run(tape, fit, score)['per_tf_k']['M5']['1']['score']
    assert m5['activity_persistence_count'] == 2  # FLAT and DOWN now.
    assert m5['activity_persistence_brier_sum'] == 1  # FLAT -> active is wrong.
    assert m5['activity_prior_paired_brier_sum'] == pytest.approx(0.125)
    assert m5['side_persistence_count'] == 1  # FLAT now has no side.
    assert m5['side_persistence_brier_sum'] == 1  # DOWN -> UP is wrong.
    assert m5['side_prior_paired_brier_sum'] == pytest.approx(4 / 9)


def test_missing_now_minute_abstains_without_changing_future_or_prior():
    tape, fit, score = _fixture(missing_now=True)
    m5 = _run(tape, fit, score)['per_tf_k']['M5']['1']['score']
    assert m5['side_known'] == m5['activity_known'] == 2
    assert m5['activity_persistence_count'] == 1
    assert m5['activity_persistence_brier_sum'] == 0
    assert m5['side_persistence_count'] == 1


def test_immature_score_query_is_unknown_before_future_source_reason():
    tape, fit, score = _fixture()
    m5 = _run(tape, fit, score, upper=25)['per_tf_k']['M5']['1']['score']
    assert m5['issues'] == 2
    assert m5['mature'] == 1
    assert m5['exact_m1'] == 1
    assert m5['all_unknown'] == 1
    assert m5['activity_known'] == m5['side_known'] == 1


def test_fold_issue_partition_and_chunk_invariance():
    tape, fit, score = _fixture()
    with pytest.raises(ValueError, match='ISSUE_PARTITION'):
        _run(tape, np.r_[fit, 15 * MINUTE_NS], score)
    with pytest.raises(ValueError, match='ISSUE_PARTITION'):
        _run(tape, fit, np.r_[10 * MINUTE_NS, score])
    a, b = _run(tape, fit, score, chunk=1), _run(tape, fit, score, chunk=8)
    assert a == b


def test_utc_issue_week_partition_and_unique_future_source_reuse():
    tape, fit, score = _fixture()
    base = int(np.datetime64('2024-05-05T23:40:00', 'ns').astype(np.int64))
    shifted = TimeQueryTargetTape(tape.validity_ns + base,
                                  tape.source_row.copy(), tape.allowed_bits.copy())
    got = score_reference_fold(
        shifted, fit + base, score + base, fit_lower_ns=base,
        fold_boundary_ns=base + 15 * MINUTE_NS,
        score_upper_ns=base + 30 * MINUTE_NS, chunk_rows=1)
    m5 = got['weekly_by_issue_monday_utc']['M5']
    assert list(m5) == ['2024-04-29', '2024-05-06']
    assert [m5[k]['unique_issue_rows'] for k in m5] == [1, 1]
    assert [m5[k]['side_known'] for k in m5] == [1, 1]
    assert [m5[k]['side_persistence_count'] for k in m5] == [0, 1]
    assert sum(m5[k]['side_prior_brier_sum'] for k in m5) == pytest.approx(5 / 9)
    reuse = got['score_source_reuse']
    assert reuse['M5']['exact_query_cells'] == 2
    assert reuse['M5']['unique_exact_m1_source_rows'] == 2
    assert reuse['M5']['unique_informative_m1_source_rows'] == 2
    # M15 k=1 and k=2 both encounter source minute 25; count it only once.
    assert reuse['M15']['exact_query_cells'] == 3
    assert reuse['M15']['unique_exact_m1_source_rows'] == 2
    assert reuse['M15']['exact_cells_per_unique_source'] == pytest.approx(1.5)
    assert reuse['M15']['informative_query_cells'] == 0
    assert reuse['M15']['unique_informative_m1_source_rows'] == 0
