import numpy as np
import pytest

from gx1.audit.dynamic_mtf_train_prefix_inputs import _d1_prefix_masks


def test_d1_screen_uses_only_bound_mask_columns_strictly_before_cutoff():
    times = np.array([100, 200, 300, 400], dtype=np.int64)
    # The unexamined feature column is a price. At and beyond cutoff even
    # malformed mask values must have no effect on the prefix measurement.
    values = np.array([[3., 0., 1.], [4., 0., 0.], [5., 1., 1.], [6., 2., -1.]], np.float32)
    got = _d1_prefix_masks(times, values, [1, 2], 300)
    np.testing.assert_array_equal(got, np.array([[0, 1], [0, 0]], dtype=np.uint8))
    values[1, 2] = 2.
    with pytest.raises(ValueError, match='D1_MASK_DOMAIN'):
        _d1_prefix_masks(times, values, [1, 2], 300)
