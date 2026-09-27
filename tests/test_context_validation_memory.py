"""Native M1 validation keeps full causal checks with column-sized scratch."""
import numpy as np
import pandas as pd
import pytest
from gx1.scripts import augment_forward_outcome_v2 as owner
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS

@pytest.mark.parametrize('values', [
    [np.nan, np.nan, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0],
    [np.nan, 1.0, np.inf, 2.0], [np.nan] * 4,
])
def test_columnwise_warmup_matches_matrix_reference(values):
    frame = pd.DataFrame({'a': values, 'b': ['bad', '2', '3', '4'], 'payload': [7, 8, 9, 10]})
    frame.attrs['source'] = 'exact'
    before = frame.copy(deep=True)
    invalid = ~np.isfinite(frame[['a', 'b']].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)).all(axis=1)
    first = int(np.argmax(~invalid)) if (~invalid).any() else len(frame)
    if first == len(frame) or invalid[first:].any():
        with pytest.raises(RuntimeError):
            owner.trim_causal_context_warmup_prefix(frame, ['a', 'b'])
    else:
        result = owner.trim_causal_context_warmup_prefix(frame, ['a', 'b'])
        pd.testing.assert_frame_equal(result, frame.iloc[first:])
        assert result.attrs['source'] == 'exact'
        assert np.shares_memory(result['payload'].values, frame['payload'].values)
    pd.testing.assert_frame_equal(frame, before)


def test_finalize_has_identical_values_without_stacking(monkeypatch):
    frame = pd.DataFrame({'payload': np.arange(5, dtype=np.float64)})
    cols = {name: np.array([np.nan, 1, 2, 3, 4], dtype=np.float32) for name in MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS}
    def forbidden(*args, **kwargs):
        raise AssertionError('full feature validation matrix allocated')
    monkeypatch.setattr(owner.np, 'column_stack', forbidden)
    result = owner.finalize_attach_columns(frame, cols)
    assert result.attrs['causal_context_warmup_rows'] == 1
    assert np.shares_memory(result['payload'].values, frame['payload'].values)
    for name, values in cols.items():
        np.testing.assert_array_equal(result[name].values, values)
