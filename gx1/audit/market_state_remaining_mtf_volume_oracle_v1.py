"""Independent closed-TF oracle for MTF cache columns 23–25 only.

The source is an OANDA price-update count per closed native TF bar, never
traded contracts or exchange volume. The caller owns TF resampling, immutable
source verification, TRAIN selection and exact cache clock alignment. This
module reads no data files and grants no job or model authority.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
import math

import numpy as np

CACHE_COLUMNS = (23, 24, 25)
FIELD_NAMES = ("vol_z_20", "vol_ratio_5_20", "vol_pct_96")
WARMUP_ROWS = (19, 19, 95)
SOURCE_UNIT = "positive_integer_price_updates_per_closed_tf_bar"
OUTPUT_UNIT = "dimensionless"


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("MTF_PRICE_UPDATE_VOLUME_ORACLE_" + reason)


def _checked_inputs(price_update_counts, available_ns, train_end_exclusive_ns):
    raw = np.asarray(price_update_counts)
    clock = np.asarray(available_ns)
    _require(raw.ndim == clock.ndim == 1 and len(raw) > 0
             and len(raw) == len(clock), "SHAPE")
    _require(raw.dtype.kind in "iuf" and raw.dtype.kind != "b",
             "PRICE_UPDATE_COUNT_DTYPE")
    if raw.dtype.kind in "iu":
        _require(bool(np.all(raw <= 2**53)), "PRICE_UPDATE_COUNT_EXACTNESS")
    counts = raw.astype(np.float64)
    _require(bool(np.isfinite(counts).all())
             and bool(np.all(counts > 0))
             and bool(np.all(counts <= 2**53))
             and bool(np.equal(counts, np.floor(counts)).all()),
             "PRICE_UPDATE_COUNT_DOMAIN")
    _require(clock.dtype == np.dtype("int64")
             and bool(np.all(clock > 0))
             and bool(np.all(clock[1:] > clock[:-1])),
             "AVAILABLE_CLOCK")
    if train_end_exclusive_ns is not None:
        _require(type(train_end_exclusive_ns) is int
                 and train_end_exclusive_ns > 0
                 and bool(np.all(clock < train_end_exclusive_ns)),
                 "TRAIN_ONLY_CUTOFF")
    return counts


def independent_closed_tf_price_update_oracle(
    price_update_counts, *, available_ns, train_end_exclusive_ns=None,
) -> dict:
    """Calculate three causal native-bar statistics without producer imports.

    Complete constant 20-bar windows have z=0. Percentile uses the current
    bar and midranks ties; a constant 96-bar window therefore has rank 0.5.
    Only source-complete windows are available: 19/19/95 warmup rows remain NaN
    with a false availability mask. No missing count is turned into zero.
    """
    counts = _checked_inputs(price_update_counts, available_ns,
                             train_end_exclusive_ns)
    n = len(counts)
    values = np.full((n, 3), np.nan, dtype=np.float32)
    available = np.zeros((n, 3), dtype=np.bool_)
    sorted_96: list[float] = []
    for at in range(n):
        current = float(counts[at])
        insort(sorted_96, current)
        if at >= 96:
            old = float(counts[at - 96])
            del sorted_96[bisect_left(sorted_96, old)]
        if at >= 19:
            last_20 = counts[at - 19:at + 1]
            mean_20 = math.fsum(float(x) for x in last_20) / 20.0
            variance_20 = math.fsum(
                (float(x) - mean_20) ** 2 for x in last_20) / 20.0
            z = ((current - mean_20) / math.sqrt(variance_20)
                 if variance_20 > 0.0 else 0.0)
            mean_5 = math.fsum(float(x) for x in counts[at - 4:at + 1]) / 5.0
            values[at, 0] = np.float32(z)
            values[at, 1] = np.float32(mean_5 / mean_20 - 1.0)
            available[at, :2] = True
        if at >= 95:
            below = bisect_left(sorted_96, current)
            tied = bisect_right(sorted_96, current) - below
            values[at, 2] = np.float32((below + 0.5 * tied) / 96.0)
            available[at, 2] = True
    _require(bool(np.array_equal(np.isfinite(values), available))
             and bool(np.all(available[19:, :2]))
             and bool(np.all(available[95:, 2])),
             "WARMUP_AND_FINITE")
    return {
        "cache_columns": CACHE_COLUMNS,
        "field_names": FIELD_NAMES,
        "source_unit": SOURCE_UNIT,
        "output_unit": OUTPUT_UNIT,
        "warmup_rows": WARMUP_ROWS,
        "values": values,
        "warmup_available": available,
    }
