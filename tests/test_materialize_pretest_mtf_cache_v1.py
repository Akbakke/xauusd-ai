from __future__ import annotations

import numpy as np
import pytest

from gx1.scripts.materialize_pretest_mtf_cache_v1 import copy_safe_prefix


def test_copy_safe_prefix_copies_only_requested_prefix(tmp_path):
    path = tmp_path / "values.npy"
    np.save(path, np.arange(12, dtype=np.int64))

    copied = copy_safe_prefix(
        array_path=path,
        safe_length=8,
        expected_shape_tail=(),
        dtype=np.dtype(np.int64),
        label="TEST",
    )

    assert copied.tolist() == list(range(8))


def test_copy_safe_prefix_rejects_bad_shape_before_copy(tmp_path):
    path = tmp_path / "values.npy"
    np.save(path, np.zeros((8, 2), dtype=np.float32))

    with pytest.raises(RuntimeError, match="ARRAY_SHAPE_INVALID"):
        copy_safe_prefix(
            array_path=path,
            safe_length=8,
            expected_shape_tail=(3,),
            dtype=np.dtype(np.float32),
            label="TEST",
        )
