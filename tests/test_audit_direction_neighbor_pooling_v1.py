import numpy as np
import pytest

from gx1.scripts.audit_direction_neighbor_pooling_v1 import pool_logits


def test_pooling_uses_only_adjacent_heads_and_handles_boundaries():
    values = np.zeros((2, 60), dtype=np.float32)
    values[0, 4] = 3
    values[1, 0] = 2
    actual = pool_logits(values)
    assert np.array_equal(actual[0, :7], np.array([0, 0, 0, 1, 1, 1, 0]))
    assert np.array_equal(actual[1, :4], np.array([1, 2/3, 0, 0]))
    assert not np.any(actual[:, 7:])


def test_pooling_rejects_wrong_shape_and_nonfinite():
    with pytest.raises(RuntimeError, match='LOGITS'): pool_logits(np.zeros((3, 59)))
    values = np.zeros((1, 60)); values[0, 4] = np.nan
    with pytest.raises(RuntimeError, match='LOGITS'): pool_logits(values)
