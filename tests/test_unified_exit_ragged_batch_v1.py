from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from gx1.contracts.unified_exit_ragged_batch_v1 import (
    flatten_sample_counts,
    one_forward_one_backward,
    right_pad_arrays,
    right_pad_monotonic_gathers,
    scatter_flat_samples,
)


def test_variable_k_layout_and_scatter_preserve_both_sides() -> None:
    layout = flatten_sample_counts([2, 0, 3])
    assert layout["entry_batch_index"].tolist() == [0, 0, 2, 2, 2]
    assert layout["sample_slot"].tolist() == [0, 1, 0, 1, 2]
    flat = np.arange(5 * 2 * 2, dtype=np.float32).reshape(5, 2, 2)
    scattered = scatter_flat_samples(flat, layout=layout, fill_value=-1.0)
    assert scattered["values"].shape == (3, 3, 2, 2)
    assert scattered["sample_valid_mask"].tolist() == [
        [True, True, False],
        [False, False, False],
        [True, True, True],
    ]
    for flat_row, (entry, slot) in enumerate(
        zip(layout["entry_batch_index"], layout["sample_slot"])
    ):
        assert np.array_equal(scattered["values"][entry, slot], flat[flat_row])


def test_right_padding_preserves_every_tail_byte_and_dtype() -> None:
    tails = [
        np.arange(length * 2 * 3, dtype=np.float32).reshape(length, 2, 3)
        for length in (1, 7, 4)
    ]
    padded = right_pad_arrays(tails, max_rows_cap=8)
    assert padded["values"].shape == (3, 7, 2, 3)
    assert padded["lengths"].tolist() == [1, 7, 4]
    for index, tail in enumerate(tails):
        observed = padded["values"][index, : len(tail)]
        assert observed.dtype == tail.dtype
        assert observed.tobytes() == tail.tobytes()
        assert not padded["row_valid_mask"][index, len(tail) :].any()
        assert not padded["values"][index, len(tail) :].any()
    assert padded["unpadded_nbytes"] == sum(value.nbytes for value in tails)
    assert padded["padded_nbytes"] == padded["values"].nbytes
    with pytest.raises(RuntimeError, match="CAP_EXCEEDED"):
        right_pad_arrays(tails, max_rows_cap=6)


def test_monotonic_gather_padding_repeats_final_legal_index() -> None:
    gathers = [
        np.asarray([0, 2, 4], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
    ]
    padded = right_pad_monotonic_gathers(
        gathers, source_lengths=[5, 3], target_rows=4
    )
    assert padded["values"].tolist() == [[0, 2, 4, 4], [1, 1, 1, 1]]
    assert padded["row_valid_mask"].tolist() == [
        [True, True, True, False],
        [True, False, False, False],
    ]


class _TailModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(5, 4)
        self.forward_calls = 0

    def forward(self, *, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        rows = self.projection(values)
        last = lengths.to(torch.long) - 1
        selected = rows[torch.arange(len(rows)), last]
        return selected.reshape(len(rows), 2, 2)


def _sequential_reference(
    *,
    online: _TailModel,
    target: _TailModel,
    tails: list[np.ndarray],
    valid: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    total_weight = sum(
        float(weights[index]) * int(valid[index].sum())
        for index in range(len(tails))
    )
    raw_sum = torch.tensor(0.0)
    backward_calls = 0
    for index, tail in enumerate(tails):
        value = torch.from_numpy(tail).unsqueeze(0)
        length = torch.tensor([len(tail)])
        with torch.no_grad():
            target_value = target(values=value, lengths=length) + 0.25
        prediction = online(values=value, lengths=length)
        row_loss = (
            (prediction - target_value).square()
            * valid[index : index + 1].to(prediction.dtype)
            * weights[index]
        ).sum()
        (row_loss / total_weight).backward()
        raw_sum = raw_sum + row_loss.detach()
        backward_calls += 1
    return raw_sum / total_weight, online.projection.weight.grad.detach(), backward_calls


def test_one_call_harness_matches_sample_loop_loss_and_gradients() -> None:
    torch.manual_seed(7)
    tails = [
        np.random.default_rng(index).normal(size=(length, 5)).astype(np.float32)
        for index, length in enumerate((2, 7, 4, 6))
    ]
    padded = right_pad_arrays(tails, max_rows_cap=8)
    values = torch.from_numpy(padded["values"])
    lengths = torch.from_numpy(padded["lengths"])
    valid = torch.tensor(
        [
            [[True, True], [True, False]],
            [[True, True], [False, True]],
            [[True, False], [True, True]],
            [[True, True], [True, True]],
        ]
    )
    weights = torch.tensor([0.5, 1.0, 2.0, 1.5])
    online_batch = _TailModel()
    target_batch = _TailModel()
    target_batch.load_state_dict(copy.deepcopy(online_batch.state_dict()))
    online_loop = copy.deepcopy(online_batch)
    target_loop = copy.deepcopy(target_batch)
    backward_calls = 0

    def count_backward(gradient):
        nonlocal backward_calls
        backward_calls += 1
        return gradient

    online_batch.projection.weight.register_hook(count_backward)
    result = one_forward_one_backward(
        online_model=online_batch,
        target_model=target_batch,
        online_inputs={"values": values, "lengths": lengths},
        target_inputs={"values": values, "lengths": lengths},
        target_builder=lambda prediction: (prediction + 0.25, valid),
        importance_weight=weights,
    )
    loop_loss, loop_gradient, loop_backward_calls = _sequential_reference(
        online=online_loop,
        target=target_loop,
        tails=tails,
        valid=valid,
        weights=weights,
    )
    assert online_batch.forward_calls == 1
    assert target_batch.forward_calls == 1
    assert online_loop.forward_calls == len(tails)
    assert target_loop.forward_calls == len(tails)
    assert backward_calls == 1
    assert loop_backward_calls == len(tails)
    assert torch.allclose(result["raw_loss"], loop_loss, rtol=1e-6, atol=1e-8)
    assert torch.allclose(
        online_batch.projection.weight.grad,
        loop_gradient,
        rtol=1e-6,
        atol=1e-8,
    )


def test_harness_fails_closed_before_backward_with_zero_valid_cells() -> None:
    model = _TailModel()
    target = copy.deepcopy(model)
    values = torch.zeros((2, 3, 5))
    lengths = torch.tensor([2, 3])
    with pytest.raises(RuntimeError, match="ZERO_VALID_SUPERVISION"):
        one_forward_one_backward(
            online_model=model,
            target_model=target,
            online_inputs={"values": values, "lengths": lengths},
            target_inputs={"values": values, "lengths": lengths},
            target_builder=lambda prediction: (
                prediction,
                torch.zeros_like(prediction, dtype=torch.bool),
            ),
            importance_weight=torch.ones(2),
        )
    assert model.projection.weight.grad is None
