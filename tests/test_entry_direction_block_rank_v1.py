import numpy as np
import pytest
from gx1.contracts.entry_direction_block_rank_v1 import block_rank_pairs, weighted_rank_pairs


def brute(y, z):
    a, b = z[y == 1], z[y == 0]
    return int((a[:, None] > b).sum()) * 2 + int((a[:, None] == b).sum()), len(a) * len(b)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_resampled_pair_equals_explicit_duplicate_rows(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 60, dtype=np.int8)
    z = rng.integers(-3, 4, 60).astype(np.float64)
    b = np.tile(np.arange(6), 10)
    before = [x.tobytes() for x in (y, z, b)]
    pair = block_rank_pairs(y, z, b)
    w = rng.multinomial(6, np.full(6, 1 / 6), size=30)
    got = weighted_rank_pairs(pair, w)
    for i, weights in enumerate(w):
        take = np.repeat(np.arange(len(y)), weights[b])
        assert (got[0][i], got[1][i]) == brute(y[take], z[take])
    assert before == [x.tobytes() for x in (y, z, b)]


def test_all_tied_scores_have_half_auc_with_unequal_blocks():
    y = np.array([0, 1, 1, 0, 1], np.int8)
    p = block_rank_pairs(y, np.zeros(5), np.array([0, 0, 1, 1, 1]))
    a, d = weighted_rank_pairs(p, [[1, 1], [3, 2], [0, 1]])
    np.testing.assert_array_equal(a, d)


def test_one_class_has_zero_comparable_pairs():
    p = block_rank_pairs(np.ones(4, np.int8), np.arange(4, dtype=np.float64),
                         np.array([0, 0, 1, 1]))
    a, d = weighted_rank_pairs(p, [[1, 1]])
    assert a[0] == d[0] == 0


def test_pool_within_fold_pairs_excludes_cross_fold_prior_ranking():
    y = np.array([0, 1, 0, 1], np.int8)
    z = np.array([0., 1., 10., 11.])
    counts = [weighted_rank_pairs(block_rank_pairs(y[i:i+2], z[i:i+2],
              np.zeros(2, dtype=np.int64)), [[1]]) for i in (0, 2)]
    assert sum(x[0][0] for x in counts) / (2 * sum(x[1][0] for x in counts)) == 1
    assert brute(y, z)[0] / (2 * brute(y, z)[1]) == .75


@pytest.mark.parametrize("fault", ["labels", "scores", "blocks", "gaps"])
def test_rejects_invalid_arrays(fault):
    y, z, b = np.array([0, 1], np.int8), np.array([0., 1.]), np.array([0, 1])
    if fault == "labels": y[0] = -1
    if fault == "scores": z[0] = np.nan
    if fault == "blocks": b[0] = -1
    if fault == "gaps": b[1] = 2
    with pytest.raises(ValueError): block_rank_pairs(y, z, b)


@pytest.mark.parametrize("weights", [[[-1, 2]], [[.5, 1.5]], [[0, 0]], [[1]], [[np.nan, 1]]])
def test_rejects_invalid_weights(weights):
    p = block_rank_pairs(np.array([0, 1], np.int8), np.array([0., 1.]), np.array([0, 1]))
    with pytest.raises(ValueError): weighted_rank_pairs(p, weights)
