import numpy as np

from gx1.audit.measure_market_state_fold2_endpoint_alias_v1 import alias_cell
from gx1.audit.verify_market_state_m1_duration_v1 import MINUTE_NS


def setup(bits0, gap_at=()):
    axis = np.arange(6, dtype=np.int64) * MINUTE_NS
    bits = np.broadcast_to(np.array(bits0, dtype=np.uint8)[:, None], (6, 5)).copy()
    gap = np.zeros(6, bool)
    gap[list(gap_at)] = True
    issues = np.array([0], np.int64)
    truth = {'allowed': np.full((1, 5), 4, np.uint8),
             'eligible': np.ones((1, 5), bool),
             'mature': np.ones((1, 5), bool)}
    return alias_cell(axis, bits, gap, issues, truth, 0, 5)


def test_certified_opposite_state_hidden_between_equal_endpoints():
    x = setup([4, 4, 1, 4, 4, 4])
    assert x['same_certified_current_and_endpoint'] == 1
    assert x['same_with_certified_different_intermediate'] == 1
    assert x['same_with_certified_opposite_side_intermediate'] == 1
    assert x['same_without_unknown_gap_with_different_intermediate'] == 1


def test_no_supported_intermediate_change_is_not_promoted():
    x = setup([4, 4, 4, 4, 4, 4])
    assert x['same_with_certified_different_intermediate'] == 0
    assert x['different_intermediate_fraction_of_same'] == 0


def test_unknown_gap_is_separate_from_certified_change():
    x = setup([4, 4, 1, 4, 4, 4], gap_at=[3])
    assert x['same_with_certified_different_intermediate'] == 1
    assert x['same_without_unknown_source_gap'] == 0
    assert x['same_without_unknown_gap_with_different_intermediate'] == 0
