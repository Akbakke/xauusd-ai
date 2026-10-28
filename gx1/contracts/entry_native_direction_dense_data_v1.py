"""Exact, in-memory join of original parents and 1..60-minute targets.

This module reads no files and grants no access. Caller must bind and verify
source bytes, fold cutoffs and execution admission before reading real data.
"""
from __future__ import annotations

import numpy as np
from gx1.contracts.entry_native_direction_dense_v1 import HORIZONS_MINUTES

MINUTE_NS = 60 * 10**9
MAX_NS = 2**63 - 1
AXES = ('parent_rows', 'bar_start_ns', 'decision_ns', 'dynamic_entry_m1_row')
TARGETS = ('terminal_time_ns', 'terminal_source_row', 'observed', 'mid_move', 'mid_return_bps', 'sign')
DENSE_MEMBERS = AXES + TARGETS


def _require(ok, reason):
    if not ok:
        raise RuntimeError('NATIVE_DIRECTION_DENSE_DATA_' + reason)


def bind_dense_targets(*, axes, legacy, dense):
    """Validate exact parents, clocks, masks, sign and original anchor bytes."""
    _require(isinstance(axes, dict) and isinstance(legacy, dict) and isinstance(dense, dict)
        and set(AXES).issubset(axes) and set(AXES + TARGETS).issubset(legacy)
        and set(dense) == set(DENSE_MEMBERS), 'MEMBERS')
    parents = axes['parent_rows']
    _require(isinstance(parents, np.ndarray) and parents.dtype == np.int64 and parents.ndim == 1 and len(parents) > 0, 'PARENTS')
    size = len(parents)
    for key in AXES:
        original, old, new = axes[key], legacy[key], dense[key]
        _require(all(isinstance(v, np.ndarray) and v.dtype == np.int64 and v.shape == (size,)
            for v in (original, old, new)) and np.array_equal(original, old) and np.array_equal(original, new), 'AXIS:' + key)
    decision, bars = axes['decision_ns'], axes['bar_start_ns']
    _require(np.all(parents >= 0) and np.all(parents[1:] > parents[:-1])
        and np.all(axes['dynamic_entry_m1_row'] >= 0) and np.all(bars >= 0)
        and np.all(bars <= MAX_NS - 5 * MINUTE_NS) and np.all(decision > 0) and np.all(decision[1:] > decision[:-1])
        and np.all(decision <= MAX_NS - 60 * MINUTE_NS)
        and np.array_equal(decision, bars + 5 * MINUTE_NS), 'AXIS_CLOCK')
    dtypes = {'terminal_time_ns': np.int64, 'terminal_source_row': np.int64,
        'observed': np.bool_, 'mid_move': np.float64, 'mid_return_bps': np.float64, 'sign': np.int8}
    anchor = np.array([0, 4, 14], dtype=np.int64)
    for key in TARGETS:
        a, b = dense[key], legacy[key]
        _require(isinstance(a, np.ndarray) and a.dtype == dtypes[key] and a.shape == (size, 60)
            and isinstance(b, np.ndarray) and b.dtype == dtypes[key] and b.shape == (size, 3), 'TARGET_HEADER:' + key)
        aa, bb = np.ascontiguousarray(a[:, anchor]), np.ascontiguousarray(b)
        _require(aa.tobytes() == bb.tobytes(), 'ANCHOR_BYTES:' + key)
    expected = decision[:, None] + np.asarray(HORIZONS_MINUTES, dtype=np.int64)[None, :] * MINUTE_NS
    obs, sign = dense['observed'], dense['sign']
    move, bps, source = dense['mid_move'], dense['mid_return_bps'], dense['terminal_source_row']
    _require(np.array_equal(dense['terminal_time_ns'], expected)
        and np.isin(sign, (-1, 0, 1)).all() and np.isfinite(move).all() and np.isfinite(bps).all()
        and np.array_equal(np.sign(move).astype(np.int8), sign)
        and np.array_equal(np.sign(bps).astype(np.int8), sign)
        and np.all(source[obs] >= 0) and np.all(source[~obs] == -1)
        and np.all(sign[~obs] == 0) and np.all(move[~obs] == 0) and np.all(bps[~obs] == 0), 'TARGET_DOMAIN')
    result = {key: dense[key] for key in ('terminal_time_ns', 'observed', 'sign')}
    for value in result.values():
        value.setflags(write=False)
    return result
