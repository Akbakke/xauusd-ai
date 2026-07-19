"""Strict conversion helpers for nested Parquet array columns."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def stack_nested_array_column(
    values: Iterable[Any],
    dtype: np.dtype,
) -> np.ndarray:
    """Stack one nested array column without padding or shape fallback."""

    items = list(values)
    if not items:
        return np.asarray([], dtype=dtype)
    try:
        return np.stack(items).astype(dtype, copy=False)
    except ValueError:
        return np.stack([np.stack(item) for item in items]).astype(dtype, copy=False)
