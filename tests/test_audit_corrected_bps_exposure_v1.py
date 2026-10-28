import copy
import numpy as np
import pytest

from gx1.scripts import audit_corrected_bps_exposure_v1 as audit


def case():
    n = 12; minute = 60_000_000_000; base = 1735700000000000000
    decision = base + np.arange(n, dtype=np.int64) * 7 * 86400 * 10**9
    parents = np.arange(n, dtype=np.int64)
    old = np.tile(np.array([-1, 1, -1], dtype=np.int8), (n, 1)); old[::2] *= -1
    arm = {'parent_rows': parents, 'bar_start_ns': decision - 5 * minute, 'decision_ns': decision,
        'terminal_time_ns': decision[:, None] + np.array([1, 5, 15], np.int64)[None, :] * minute,
        'observed': np.ones((n, 3), bool), 'sign': old, 'logit_up': np.linspace(-.1, .1, n*3, dtype=np.float32).reshape(n, 3),
        'context__one': np.arange(n, dtype=np.int8)}
    arms = {'one': copy.deepcopy(arm), 'bps': copy.deepcopy(arm), 'three': copy.deepcopy(arm)}
    full_sign = np.tile(np.array([-1, 1], np.int8), (n, 30)); full_sign[:, audit.COLS] = old
    target = {'parent_rows': parents, 'bar_start_ns': arm['bar_start_ns'].copy(), 'decision_ns': decision.copy(),
        'terminal_time_ns': decision[:, None] + np.arange(1, 61, dtype=np.int64)[None, :] * minute,
        'observed': np.ones((n, 60), bool), 'sign': full_sign}
    target['sign'][0, 4] = 0
    return arms, target


def test_corrected_join_removes_false_sign_and_rejects_leak_or_context_change():
    arms, target = case(); sign, counts = audit.validate_join(arms, target)
    assert sign.shape == (12, 3) and counts == [0, 1, 0]
    wrong = copy.deepcopy(target); wrong['sign'][1, 4] *= -1
    with pytest.raises(RuntimeError, match='NO_OPPOSITE_SIGN_FLIPS'): audit.validate_join(arms, wrong)
    wrong = copy.deepcopy(arms); wrong['bps']['context__one'][0] += 1
    with pytest.raises(RuntimeError, match='ORIGINAL_PAIR:bps'): audit.validate_join(wrong, target)
    wrong = copy.deepcopy(target); wrong['decision_ns'][0] += 1
    with pytest.raises(RuntimeError, match='CORRECTED_AXIS:decision_ns'): audit.validate_join(arms, wrong)


def test_evaluate_keeps_same_rows_and_zero_differences_for_equal_arms():
    arms, target = case(); sign, _ = audit.validate_join(arms, target)
    result = audit.evaluate(arms, sign, replicates=100)
    assert [row['horizon_minutes'] for row in result['panels']] == [1, 5, 15]
    assert result['panels'][1]['eligible'] == 11
    assert all(bounds['low'] == bounds['high'] == 0 for row in result['paired_intervals']
        for bounds in row['intervals_candidate_minus_one'].values())
