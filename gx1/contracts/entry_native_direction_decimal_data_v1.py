"""Separate in-memory consumer for the source-verified decimal direction target.

The original dense-target owner remains byte-identical and still enforces its
legacy anchors. File authentication, execution admission and model admission
belong to the caller; this module alone establishes none of those properties.
"""
from __future__ import annotations

import numpy as np
from gx1.contracts.entry_native_direction_dense_v1 import HORIZONS_MINUTES
from gx1.contracts.entry_native_direction_dense_data_v1 import (
    AXES, TARGETS, DENSE_MEMBERS, _require, bind_dense_targets,
)

DECIMAL_TARGET_IDENTITY = 'exact_original_decimal_midpoint_v1'



def bind_decimal_grid_dense_targets(*, axes, legacy, original_dense, corrected_dense, target_identity):
    """Validate the separate corrected 1..60 target without changing the old binder.

    This is only an in-memory structural join. The caller must first verify the
    exact archive bytes, the independently source-verified materialization
    receipt and the admitted execution scope. Consistent signs alone do not
    prove decimal-price provenance or the magnitude of a move. This function
    opens no files, carries no execution authority and is not a training route.
    """
    _require(target_identity == DECIMAL_TARGET_IDENTITY, 'DECIMAL_IDENTITY')
    bind_dense_targets(axes=axes, legacy=legacy, dense=original_dense)
    _require(isinstance(corrected_dense, dict)
        and set(corrected_dense) == set(DENSE_MEMBERS), 'DECIMAL_MEMBERS')
    size = len(axes['parent_rows'])
    for key in AXES:
        value = corrected_dense[key]
        _require(isinstance(value, np.ndarray) and value.dtype == np.int64
            and value.shape == (size,) and np.array_equal(value, axes[key]), 'DECIMAL_AXIS:' + key)
    dtypes = {'terminal_time_ns': np.int64, 'terminal_source_row': np.int64,
        'observed': np.bool_, 'mid_move': np.float64, 'mid_return_bps': np.float64, 'sign': np.int8}
    for key in TARGETS:
        value = corrected_dense[key]
        _require(isinstance(value, np.ndarray) and value.dtype == dtypes[key]
            and value.shape == (size, len(HORIZONS_MINUTES)), 'DECIMAL_TARGET_HEADER:' + key)
    for key in ('terminal_time_ns', 'terminal_source_row', 'observed'):
        _require(np.array_equal(corrected_dense[key], original_dense[key]), 'DECIMAL_PRESERVE:' + key)
    observed = corrected_dense['observed']
    sign = corrected_dense['sign']
    move = corrected_dense['mid_move']
    bps = corrected_dense['mid_return_bps']
    _require(np.isin(sign, (-1, 0, 1)).all() and np.isfinite(move).all() and np.isfinite(bps).all()
        and np.array_equal(np.sign(move).astype(np.int8), sign)
        and np.array_equal(np.sign(bps).astype(np.int8), sign)
        and np.all(sign[~observed] == 0) and np.all(move[~observed] == 0)
        and np.all(bps[~observed] == 0), 'DECIMAL_TARGET_DOMAIN')
    result = {key: corrected_dense[key] for key in ('terminal_time_ns', 'observed', 'sign', 'mid_return_bps')}
    for value in result.values():
        value.setflags(write=False)
    return result
