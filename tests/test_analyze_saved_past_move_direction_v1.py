import numpy as np
import pytest

from gx1.scripts import analyze_saved_past_move_direction_v1 as audit


def target_fixture():
    m = audit.MINUTE
    decision = np.array([10, 15, 20, 25], dtype=np.int64) * m
    terminal = decision[:, None] + np.arange(1, 61, dtype=np.int64)[None, :] * m
    return {'parent_rows': np.arange(4, dtype=np.int64), 'decision_ns': decision,
        'terminal_time_ns': terminal, 'observed': np.ones((4, 60), dtype=np.bool_),
        'sign': np.ones((4, 60), dtype=np.int8)}


def test_past_move_cohort_requires_exact_older_parent_and_observed_values():
    t = target_fixture(); rows = np.array([0, 1, 2, 3], dtype=np.int64)
    eligible, sign = audit.previous_move_cohort(t, rows, 100 * audit.MINUTE)
    assert eligible.tolist() == [False, False, True, True]
    assert sign[eligible].tolist() == [1, 1]
    t['observed'][0, 4] = False
    assert audit.previous_move_cohort(t, rows, 100 * audit.MINUTE)[0].tolist() == [False, False, False, True]
    t['sign'][1, 4] = 0
    assert not audit.previous_move_cohort(t, rows, 100 * audit.MINUTE)[0].any()


def test_past_move_rejects_wrong_previous_clock_and_respects_outcome_cutoff():
    t = target_fixture(); rows = np.array([2, 3], dtype=np.int64)
    assert audit.previous_move_cohort(t, rows, 26 * audit.MINUTE)[0].tolist() == [True, False]
    t['terminal_time_ns'][0, 4] += audit.MINUTE
    with pytest.raises(RuntimeError, match='PAST_TERMINAL_CLOCK'):
        audit.previous_move_cohort(t, rows, 100 * audit.MINUTE)
