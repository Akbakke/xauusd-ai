import subprocess
import sys

import numpy as np
import pytest

from gx1.scripts import compare_entry_direction_matched_exposure_v1 as audit


def brute_metrics(y, z):
    pos, neg = z[y == 1], z[y == 0]
    auc = np.mean((pos[:, None] > neg) + 0.5 * (pos[:, None] == neg))
    p = 1 / (1 + np.exp(-z))
    return auc, np.mean(np.logaddexp(0, z) - y * z), np.mean((p - y) ** 2)


def test_bootstrap_equals_explicit_duplicate_rows_with_ties_and_unequal_blocks():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1], dtype=np.int8)
    old = np.array([0., 0., 1., 1., -1., 2., 2., 0., 0.])
    new = np.array([1., 0., 1., 2., -1., -1., 2., 0., 1.])
    groups = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    weights = np.array([[1, 1, 1], [2, 0, 1], [0, 1, 2]], dtype=np.int64)
    actual = audit.paired_draws(y, old, new, groups, weights)
    for i, w in enumerate(weights):
        take = np.repeat(np.arange(len(y)), w[groups])
        one, three = brute_metrics(y[take], old[take]), brute_metrics(y[take], new[take])
        assert actual['one_auc'][i] == pytest.approx(one[0])
        assert actual['three_auc'][i] == pytest.approx(three[0])
        for metric, ix in [('auc', 0), ('logloss', 1), ('brier', 2)]:
            assert actual[metric][i] == pytest.approx(three[ix] - one[ix], abs=1e-14)


def test_constant_logit_shift_preserves_auc_but_can_change_probability_scores():
    y = np.array([0, 1, 0, 1], np.int8)
    z = np.array([0., 0., -1., 1.])
    result = audit.paired_draws(y, z, z + 1., np.array([0, 0, 1, 1], np.int64),
                                np.array([[1, 1], [2, 0]], np.int64))
    np.testing.assert_array_equal(result['auc'], [0., 0.])
    assert np.any(np.abs(result['logloss']) > 1e-6)
    assert np.any(np.abs(result['brier']) > 1e-6)
    same = audit.paired_draws(y, z, z.copy(), np.array([0, 0, 1, 1], np.int64),
                              np.array([[1, 1]], np.int64))
    for metric in ('auc', 'logloss', 'brier'):
        np.testing.assert_array_equal(same[metric], [0.])


def test_undefined_single_class_or_empty_bootstrap_fails_without_replacement():
    y = np.array([0, 0, 1, 1], np.int8)
    z = np.array([0., 1., 1., 2.])
    groups = np.array([0, 0, 1, 1], np.int64)
    with pytest.raises(RuntimeError, match='SINGLE_CLASS_BOOTSTRAP'):
        audit.paired_draws(y, z, z, groups, np.array([[2, 0]], np.int64))
    with pytest.raises(RuntimeError, match='EMPTY_BOOTSTRAP'):
        audit.paired_draws(y, z, z, groups, np.array([[0, 0, 3]], np.int64))


def fixture():
    minute = 60_000_000_000
    day = audit.DAY_NS
    decision = np.concatenate([np.int64(d) * day + np.arange(4, dtype=np.int64) * 5 * minute
                               for d in (20000, 20007, 20014)])
    sign = np.tile(np.array([-1, 1, -1, 1], np.int8), 3)[:, None] * np.ones((1, 3), np.int8)
    observed = np.ones((12, 3), dtype=np.bool_)
    observed[:4] = False
    sign[:4] = 0
    sign[5] = 0
    observed[6] = False
    sign[6] = 0
    values = np.tile(np.array([0., 0., -1., 1.], np.float32), 3)[:, None] * np.ones((1, 3), np.float32)
    one = {'parent_rows': np.arange(12, dtype=np.int64), 'decision_ns': decision,
           'bar_start_ns': decision - 5 * minute,
           'terminal_time_ns': decision[:, None] + np.array([1, 5, 15], np.int64) * minute,
           'observed': observed, 'sign': sign, 'logit_up': values,
           'context__synthetic': np.zeros(12, dtype=np.int8)}
    three = {k: v.copy() for k, v in one.items()}
    three['logit_up'] += np.float32(1.)
    return one, three, decision[-1] + 5 * minute


def test_all_parent_calendar_is_fixed_before_masks_and_cutoff(monkeypatch):
    one, three, cutoff = fixture()
    calls = []
    def fixed_weights(blocks, *, seed, np, replicates):
        calls.append((blocks, seed, replicates))
        return np.array([[1, 1, 1], [0, 2, 1], [2, 1, 0]], np.int64)
    monkeypatch.setattr(audit.saved, 'block_weights', fixed_weights)
    result = audit.score_and_compare(one, three, cutoff=cutoff, replicates=3, minimum_blocks=3)
    assert calls == [(3, 20261027, 3), (3, 20261028, 3)]
    panel = next(x for x in result['panels'] if x['horizon_minutes'] == 5)
    assert panel['counts'] == {'parents': 12, 'eligible': 5, 'observed_mature_zero': 1,
                               'mature_missing': 5, 'immature': 1}
    assert all(g['blocks'] == 3 for g in panel['group_intervals'])
    assert panel['three_minus_one']['auc'] == 0
    assert panel['three_minus_one']['logloss'] != 0
    assert not result['primary_auc_improvement_supported_separately']


@pytest.mark.parametrize('key', ['parent_rows', 'decision_ns', 'terminal_time_ns', 'sign',
                                 'observed', 'context__synthetic'])
def test_mismatched_parent_clock_target_mask_or_context_is_rejected_before_scoring(key):
    one, three, cutoff = fixture()
    if three[key].dtype == np.bool_:
        three[key].flat[0] = not three[key].flat[0]
    else:
        three[key].flat[0] += 1
    with pytest.raises(RuntimeError, match='PAIR_EXACT_JOIN'):
        audit.score_and_compare(one, three, cutoff=cutoff, replicates=3, minimum_blocks=3)


def test_guard_allows_only_the_pinned_git_queries_without_model_imports():
    program = """\
import sys
from pathlib import Path
from gx1.scripts import compare_entry_direction_matched_exposure_v1 as a
assert 'torch' not in sys.modules
reads, denied = a._install_guard([], set(), Path('/tmp/synthetic-output'))
head, branch, status = a._git_state()
assert len(head) == 40 and branch == 'work/gx1-current'
assert not reads and not denied
try:
    open('/tmp/unbound-prediction.npz', 'rb')
except RuntimeError as exc:
    assert 'UNBOUND_PAYLOAD' in str(exc)
else:
    raise AssertionError('unbound payload was not rejected')
"""
    run = subprocess.run([sys.executable, '-B', '-c', program], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
