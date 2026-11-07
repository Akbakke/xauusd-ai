"""Fixed prior respects partial targets, maturity and empty support."""
import numpy as np
import pytest

from gx1.contracts.market_state_five_endpoint_prior_v1 import fit_activity_side_prior


def test_active_unknown_side_counts_activity_without_side_vote():
    allowed = np.full((4, 5), 7, dtype=np.uint8)
    allowed[:, 0] = [5, 1, 4, 2]
    ready = np.ones((4, 5), dtype=np.bool_)
    result = fit_activity_side_prior(allowed, ready, ready)
    assert result["counts"]["M5"] == {"activity": 4, "active": 3,
                                       "certified_side": 2, "up": 1}
    np.testing.assert_allclose(result["probabilities"][0], [1/3, 1/3, 1/3])
    assert result["counts"]["M15"]["activity"] == 0
    np.testing.assert_allclose(result["probabilities"][1], [.25, .5, .25])


def test_unknown_ineligible_and_immature_cells_do_not_enter_prior():
    allowed = np.full((5, 5), 7, dtype=np.uint8)
    allowed[:, 1] = [1, 4, 5, 2, 7]
    eligible = np.ones((5, 5), dtype=np.bool_)
    mature = np.ones((5, 5), dtype=np.bool_)
    mature[0, 1] = False
    eligible[1, 1] = False
    result = fit_activity_side_prior(allowed, eligible, mature)
    assert result["counts"]["M15"] == {"activity": 2, "active": 1,
                                        "certified_side": 0, "up": 0}
    np.testing.assert_allclose(result["probabilities"][1], [.25, .5, .25])


@pytest.mark.parametrize("bit,expected", [
    (2, [.125, .75, .125]),
    (5, [.375, .25, .375]),
    (7, [.25, .5, .25]),
])
def test_all_flat_active_or_unknown_stays_finite_simplex(bit, expected):
    allowed = np.full((2, 5), bit, dtype=np.uint8)
    ready = np.ones((2, 5), dtype=np.bool_)
    result = fit_activity_side_prior(allowed, ready, ready)
    np.testing.assert_allclose(result["probabilities"], np.tile(expected, (5, 1)))
    assert np.isfinite(result["probabilities"]).all()
    assert np.allclose(result["probabilities"].sum(axis=-1), 1)
    assert all(counts["certified_side"] == 0 for counts in result["counts"].values())


@pytest.mark.parametrize("bad", [
    np.full((2, 5), 3, dtype=np.uint8),
    np.full((2, 4), 2, dtype=np.uint8),
    np.full((0, 5), 2, dtype=np.uint8),
    np.full((2, 5), 2, dtype=np.int64),
])
def test_invalid_allowed_fails_closed(bad):
    ready = np.ones(bad.shape, dtype=np.bool_)
    with pytest.raises(ValueError, match="FIVE_ENDPOINT_PRIOR_INPUT"):
        fit_activity_side_prior(bad, ready, ready)
