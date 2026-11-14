"""Exact insertion keeps every legacy float32 bit in name order."""
import numpy as np
import pytest

from gx1.contracts import market_state_current_tabular_span_overlay_v1 as overlay


def test_two_named_spans_insert_without_changing_old_bits():
    old = np.array([[0., -0., np.nan], [1., 2., 3.]], dtype=np.float32)
    extra = np.array([[4., 5.], [6., 7.]], dtype=np.float32)
    names = ['a', overlay.SPAN_MTF[0], 'b', overlay.SPAN_MTF[1], 'c']
    out = overlay.ordered_insert(old, extra, names, overlay.SPAN_MTF)
    assert out.shape == (2, 5)
    assert np.array_equal(out[:, [0, 2, 4]].view('<u4'), old.view('<u4'))
    assert out[:, [1, 3]].tolist() == extra.tolist()
    with pytest.raises(RuntimeError, match='INSERT_SHAPE'):
        overlay.ordered_insert(old, extra, names[:-1], overlay.SPAN_MTF)
