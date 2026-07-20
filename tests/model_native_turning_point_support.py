"""Exact bounded turning-point columns for immutable prediction fixtures."""

from __future__ import annotations

import numpy as np

from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_TIMING_OUTPUT_DIM,
    model_native_aux_target_contract_metadata,
)


_LAYOUT = model_native_aux_target_contract_metadata()["turning_point_timing"][
    "layout"
]


def turning_point_prediction_columns(rows: int) -> dict[str, object]:
    ordinal = np.arange(int(rows), dtype=np.float64)
    timing = np.zeros((int(rows), MODEL_NATIVE_TIMING_OUTPUT_DIM), dtype=np.float64)
    columns: dict[str, object] = {}
    for item in _LAYOUT:
        index = int(item["index"])
        values = 0.05 + 0.90 * (np.mod(ordinal + 7.0 * index, 101.0) / 100.0)
        timing[:, index] = values
        columns[str(item["target_column"])] = values
    columns["timing_pred"] = [row.tolist() for row in timing]
    return columns


def turning_point_prediction_row(index: int = 0) -> dict[str, object]:
    columns = turning_point_prediction_columns(max(1, int(index) + 1))
    row: dict[str, object] = {}
    for name, values in columns.items():
        row[name] = values[int(index)]
    return row


__all__ = ["turning_point_prediction_columns", "turning_point_prediction_row"]
