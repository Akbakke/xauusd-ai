import numpy as np
import pytest

from gx1.scripts.audit_direction_d1_trendline_replay_v1 import compare_fields


def test_cache_comparison_detects_float_bits_and_missingness():
    a = np.array([[0., np.nan], [1., 2.]], dtype=np.float32)
    ts = np.array([10, 20], dtype=np.int64)
    assert compare_fields(a, a.copy(), ('x', 'y'), ts)['all_finite_float32_bits_and_missingness_equal']
    b = a.copy(); b[0, 0] = -0.; b[0, 1] = 0.; b[1, 1] = 3.
    got = compare_fields(a, b, ('x', 'y'), ts)
    assert got['mismatched_cells'] == 3
    assert got['per_field'][0]['first_mismatch_timestamp_ns'] == 10
    assert got['per_field'][1]['mismatched_rows'] == 2


def test_cache_comparison_rejects_invalid_geometry():
    with pytest.raises(RuntimeError, match='COMPARE_SHAPE'):
        compare_fields(np.ones((2, 1)), np.ones((3, 1)), ('x',), np.array([1, 2]))
    with pytest.raises(RuntimeError, match='COMPARE_SHAPE'):
        compare_fields(np.array([[np.inf]]), np.array([[np.inf]]), ('x',), np.array([1]))
