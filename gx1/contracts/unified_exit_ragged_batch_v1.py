"""Schema-neutral ragged collation for sampled Exit transitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

RAGGED_SAMPLE_LAYOUT_SCHEMA_VERSION = "gx1_unified_exit_ragged_sample_layout_v1"
RIGHT_PADDED_ARRAY_SCHEMA_VERSION = "gx1_unified_exit_right_padded_array_v1"


def flatten_sample_counts(sample_counts: Sequence[int]) -> dict[str, Any]:
    """Flatten variable K samples per Entry while preserving scatter ownership."""

    if (
        not isinstance(sample_counts, Sequence)
        or isinstance(sample_counts, (str, bytes))
        or not sample_counts
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in sample_counts
        )
    ):
        raise RuntimeError("UNIFIED_EXIT_RAGGED_SAMPLE_COUNTS_INVALID")
    counts = np.ascontiguousarray(sample_counts, dtype=np.int64)
    total = int(counts.sum())
    if total < 1:
        raise RuntimeError("UNIFIED_EXIT_RAGGED_ZERO_ACTIVE_SAMPLES")
    max_samples = int(counts.max())
    entry_batch_index = np.repeat(
        np.arange(len(counts), dtype=np.int64), counts
    )
    sample_slot = np.concatenate(
        [np.arange(int(count), dtype=np.int64) for count in counts if count]
    )
    valid = np.arange(max_samples, dtype=np.int64)[None, :] < counts[:, None]
    return {
        "schema_version": RAGGED_SAMPLE_LAYOUT_SCHEMA_VERSION,
        "batch_size": len(counts),
        "flat_sample_count": total,
        "max_samples_per_entry": max_samples,
        "sample_counts": counts,
        "entry_batch_index": np.ascontiguousarray(entry_batch_index),
        "sample_slot": np.ascontiguousarray(sample_slot),
        "sample_valid_mask": np.ascontiguousarray(valid, dtype=np.bool_),
    }


def right_pad_arrays(
    arrays: Sequence[np.ndarray],
    *,
    max_rows_cap: int | None = None,
    pad_value: int | float | bool = 0,
) -> dict[str, Any]:
    """Right-pad exact arrays without dtype conversion or row truncation."""

    if (
        not isinstance(arrays, Sequence)
        or isinstance(arrays, (str, bytes))
        or not arrays
    ):
        raise RuntimeError("UNIFIED_EXIT_RAGGED_ARRAYS_INVALID")
    values = list(arrays)
    first = values[0]
    if (
        not isinstance(first, np.ndarray)
        or first.ndim < 1
        or len(first) < 1
        or first.dtype.kind not in "biufc"
    ):
        raise RuntimeError("UNIFIED_EXIT_RAGGED_ARRAYS_INVALID")
    trailing_shape = first.shape[1:]
    dtype = first.dtype
    for value in values:
        if (
            not isinstance(value, np.ndarray)
            or value.ndim != first.ndim
            or len(value) < 1
            or value.shape[1:] != trailing_shape
            or value.dtype != dtype
            or (value.dtype.kind in "fc" and not np.isfinite(value).all())
        ):
            raise RuntimeError("UNIFIED_EXIT_RAGGED_ARRAYS_INCOMPATIBLE")
    lengths = np.ascontiguousarray([len(value) for value in values], dtype=np.int64)
    max_rows = int(lengths.max())
    if (
        max_rows_cap is not None
        and (
            isinstance(max_rows_cap, bool)
            or not isinstance(max_rows_cap, int)
            or max_rows_cap < 1
            or max_rows > max_rows_cap
        )
    ):
        raise RuntimeError("UNIFIED_EXIT_RAGGED_ARRAY_CAP_EXCEEDED")
    padded = np.full(
        (len(values), max_rows, *trailing_shape),
        pad_value,
        dtype=dtype,
    )
    for index, value in enumerate(values):
        padded[index, : len(value)] = value
    row_valid = np.arange(max_rows, dtype=np.int64)[None, :] < lengths[:, None]
    return {
        "schema_version": RIGHT_PADDED_ARRAY_SCHEMA_VERSION,
        "values": np.ascontiguousarray(padded),
        "lengths": lengths,
        "row_valid_mask": np.ascontiguousarray(row_valid, dtype=np.bool_),
        "unpadded_nbytes": int(sum(value.nbytes for value in values)),
        "padded_nbytes": int(padded.nbytes),
    }


def right_pad_monotonic_gathers(
    gathers: Sequence[np.ndarray],
    *,
    source_lengths: Sequence[int],
    target_rows: int | None = None,
) -> dict[str, Any]:
    """Pad gather rows with their final legal index, preserving monotonicity."""

    if len(gathers) != len(source_lengths) or not gathers:
        raise RuntimeError("UNIFIED_EXIT_RAGGED_GATHER_INPUT_INVALID")
    normalized: list[np.ndarray] = []
    for gather, source_length in zip(gathers, source_lengths):
        if (
            not isinstance(gather, np.ndarray)
            or gather.ndim != 1
            or len(gather) < 1
            or gather.dtype != np.dtype(np.int64)
            or isinstance(source_length, bool)
            or not isinstance(source_length, (int, np.integer))
            or int(source_length) < 1
            or np.any(gather < 0)
            or np.any(gather >= int(source_length))
            or np.any(np.diff(gather) < 0)
        ):
            raise RuntimeError("UNIFIED_EXIT_RAGGED_GATHER_INPUT_INVALID")
        normalized.append(gather)
    max_rows = max(len(value) for value in normalized)
    if target_rows is not None:
        if (
            isinstance(target_rows, bool)
            or not isinstance(target_rows, int)
            or target_rows < max_rows
        ):
            raise RuntimeError("UNIFIED_EXIT_RAGGED_GATHER_TARGET_INVALID")
        max_rows = target_rows
    padded = np.empty((len(normalized), max_rows), dtype=np.int64)
    valid = np.zeros((len(normalized), max_rows), dtype=np.bool_)
    for index, gather in enumerate(normalized):
        padded[index] = int(gather[-1])
        padded[index, : len(gather)] = gather
        valid[index, : len(gather)] = True
    return {
        "values": np.ascontiguousarray(padded),
        "lengths": np.asarray([len(value) for value in normalized], dtype=np.int64),
        "row_valid_mask": np.ascontiguousarray(valid),
    }


def scatter_flat_samples(
    flat_values: np.ndarray,
    *,
    layout: Mapping[str, Any],
    fill_value: int | float | bool = 0,
) -> dict[str, np.ndarray]:
    """Scatter flat transition outputs back to [Entry, K, ...] exactly once."""

    if (
        not isinstance(flat_values, np.ndarray)
        or flat_values.ndim < 1
        or int(flat_values.shape[0]) != layout.get("flat_sample_count")
        or layout.get("schema_version") != RAGGED_SAMPLE_LAYOUT_SCHEMA_VERSION
    ):
        raise RuntimeError("UNIFIED_EXIT_RAGGED_SCATTER_INPUT_INVALID")
    entry_index = np.asarray(layout.get("entry_batch_index"), dtype=np.int64)
    sample_slot = np.asarray(layout.get("sample_slot"), dtype=np.int64)
    batch_size = layout.get("batch_size")
    max_samples = layout.get("max_samples_per_entry")
    if (
        entry_index.shape != (len(flat_values),)
        or sample_slot.shape != (len(flat_values),)
        or not isinstance(batch_size, int)
        or not isinstance(max_samples, int)
        or np.any(entry_index < 0)
        or np.any(entry_index >= batch_size)
        or np.any(sample_slot < 0)
        or np.any(sample_slot >= max_samples)
        or len(set(zip(entry_index.tolist(), sample_slot.tolist())))
        != len(flat_values)
    ):
        raise RuntimeError("UNIFIED_EXIT_RAGGED_SCATTER_LAYOUT_INVALID")
    output = np.full(
        (batch_size, max_samples, *flat_values.shape[1:]),
        fill_value,
        dtype=flat_values.dtype,
    )
    output[entry_index, sample_slot] = flat_values
    valid = np.zeros((batch_size, max_samples), dtype=np.bool_)
    valid[entry_index, sample_slot] = True
    if not np.array_equal(valid, np.asarray(layout["sample_valid_mask"], dtype=np.bool_)):
        raise RuntimeError("UNIFIED_EXIT_RAGGED_SCATTER_LAYOUT_INVALID")
    return {"values": np.ascontiguousarray(output), "sample_valid_mask": valid}


def one_forward_one_backward(
    *,
    online_model: Callable[..., torch.Tensor],
    target_model: Callable[..., torch.Tensor],
    online_inputs: Mapping[str, torch.Tensor],
    target_inputs: Mapping[str, torch.Tensor],
    target_builder: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    importance_weight: torch.Tensor,
    grad_accum_steps: int = 1,
    loss_scale: torch.Tensor | float = 1.0,
) -> dict[str, Any]:
    """Run one online call, one frozen-target call and one weighted backward."""

    if (
        not callable(online_model)
        or not callable(target_model)
        or not callable(target_builder)
        or not isinstance(online_inputs, Mapping)
        or not isinstance(target_inputs, Mapping)
        or isinstance(grad_accum_steps, bool)
        or not isinstance(grad_accum_steps, int)
        or grad_accum_steps < 1
    ):
        raise RuntimeError("UNIFIED_EXIT_RAGGED_HARNESS_INPUT_INVALID")
    with torch.no_grad():
        frozen_prediction = target_model(**target_inputs)
        if not isinstance(frozen_prediction, torch.Tensor):
            raise RuntimeError("UNIFIED_EXIT_RAGGED_TARGET_OUTPUT_INVALID")
        targets, valid_mask = target_builder(frozen_prediction)
    prediction = online_model(**online_inputs)
    if (
        not isinstance(prediction, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or not isinstance(valid_mask, torch.Tensor)
        or tuple(prediction.shape) != tuple(targets.shape)
        or tuple(valid_mask.shape) != tuple(prediction.shape)
        or prediction.ndim < 1
        or valid_mask.dtype != torch.bool
        or targets.requires_grad
        or not bool(torch.isfinite(prediction).all().item())
        or not bool(torch.isfinite(targets).all().item())
        or not isinstance(importance_weight, torch.Tensor)
        or tuple(importance_weight.shape) != (int(prediction.shape[0]),)
        or not bool(torch.isfinite(importance_weight).all().item())
        or bool((importance_weight <= 0.0).any().item())
    ):
        raise RuntimeError("UNIFIED_EXIT_RAGGED_HARNESS_TENSOR_INVALID")
    broadcast_weight = importance_weight.reshape(
        int(prediction.shape[0]), *([1] * (prediction.ndim - 1))
    ).to(device=prediction.device, dtype=prediction.dtype)
    weighted_valid = broadcast_weight * valid_mask.to(prediction.dtype)
    normalization = weighted_valid.sum()
    if not bool(torch.isfinite(normalization).item()) or not bool(
        (normalization > 0.0).item()
    ):
        raise RuntimeError("UNIFIED_EXIT_RAGGED_ZERO_VALID_SUPERVISION")
    squared_error = (prediction - targets.to(prediction.dtype)).square()
    raw_loss = (squared_error * weighted_valid).sum() / normalization
    scale = torch.as_tensor(loss_scale, device=prediction.device, dtype=prediction.dtype)
    objective = scale * raw_loss / float(grad_accum_steps)
    if not bool(torch.isfinite(objective).item()):
        raise RuntimeError("UNIFIED_EXIT_RAGGED_OBJECTIVE_NONFINITE")
    objective.backward()
    return {
        "prediction": prediction.detach(),
        "targets": targets.detach(),
        "valid_mask": valid_mask.detach(),
        "raw_loss": raw_loss.detach(),
        "weighted_valid_cells": normalization.detach(),
        "flat_sample_count": int(prediction.shape[0]),
        "online_forward_calls": 1,
        "target_forward_calls": 1,
        "backward_calls": 1,
    }


__all__ = (
    "RAGGED_SAMPLE_LAYOUT_SCHEMA_VERSION",
    "RIGHT_PADDED_ARRAY_SCHEMA_VERSION",
    "flatten_sample_counts",
    "one_forward_one_backward",
    "right_pad_arrays",
    "right_pad_monotonic_gathers",
    "scatter_flat_samples",
)
