"""Passive capture of the exact B16 joint/source and S-then-F query boundaries."""
from __future__ import annotations

from contextlib import contextmanager


def _need(condition, reason):
    if not condition:
        raise RuntimeError("TA_C171_JOINT_BOUNDARY_" + reason)


@contextmanager
def capture_joint_boundaries(model, *, counts, torch):
    """Yield owned FP32 NumPy copies; never replace module inputs or outputs.

    The joint hidden is read from actual query inputs, never reconstructed.
    Counts record hook invocations, including invalid ones. Partial valid copies
    survive failure; a swallowed hook failure still rejects the context. This
    helper supplies no run admission or historical/predictive interpretation.
    """
    state = getattr(model, "state_model", None)
    backbone = getattr(state, "backbone", None)
    norm = getattr(backbone, "entry_q_joint_norm", None)
    projection = getattr(state, "query_projection", None)
    coordinates = getattr(state, "query_tf_coordinates", None)
    _need(isinstance(model, torch.nn.Module)
          and isinstance(norm, torch.nn.LayerNorm)
          and tuple(norm.normalized_shape) == (512,)
          and isinstance(projection, torch.nn.Sequential)
          and len(projection) == 2
          and isinstance(projection[0], torch.nn.Linear)
          and (projection[0].in_features, projection[0].out_features) == (134, 128)
          and isinstance(projection[1], torch.nn.GELU), "MODULE_LAYOUT")
    _need(isinstance(coordinates, torch.Tensor)
          and tuple(coordinates.shape) == (5, 5)
          and coordinates.dtype == torch.float32
          and coordinates.device.type != "meta"
          and bool(torch.isfinite(coordinates.detach()).all().item()), "TF_COORDINATES")
    _need(torch.equal(coordinates.detach(), torch.eye(
        5, dtype=torch.float32, device=coordinates.device)), "TF_COORDINATES")
    keys = ("joint_source_captures", "joint_hidden_captures", "query_input_observations")
    _need(isinstance(counts, dict)
          and all(type(counts.get(k, 0)) is int and counts.get(k, 0) >= 0
                  for k in keys), "COUNTS")
    for key in keys:
        counts.setdefault(key, 0)
    device = coordinates.device
    tf_reference = coordinates.detach().clone()
    captured, handles = {}, []
    source_calls = query_calls = 0
    hidden_reference = None
    failed = False

    def same_bits(left, right):
        return torch.equal(left.detach().contiguous().view(torch.int32),
                           right.detach().contiguous().view(torch.int32))

    def input_value(inputs, shape, name):
        _need(len(inputs) == 1 and isinstance(inputs[0], torch.Tensor), "INPUT:" + name)
        value = inputs[0]
        _need(tuple(value.shape) == shape and value.dtype == torch.float32
              and value.device == device, "TENSOR:" + name)
        value = value.detach()
        _need(bool(torch.isfinite(value).all().item()), "NONFINITE:" + name)
        return value

    def source_hook(_module, inputs):
        nonlocal source_calls, failed
        counts["joint_source_captures"] += 1
        source_calls += 1
        try:
            _need(source_calls == 1 and query_calls == 0, "SOURCE_ORDER")
            value = input_value(inputs, (16, 512), "joint_source")
            captured["joint_source"] = value.cpu().numpy().copy()
        except BaseException:
            failed = True
            raise

    def query_hook(_module, inputs):
        nonlocal query_calls, hidden_reference, failed
        counts["query_input_observations"] += 1
        query_calls += 1
        if query_calls == 1:
            counts["joint_hidden_captures"] += 1
        try:
            _need(source_calls == 1 and "joint_source" in captured
                  and query_calls in (1, 2), "QUERY_ORDER")
            shape = (16, 1, 5, 134) if query_calls == 1 else (16, 5, 134)
            value = input_value(inputs, shape, "state" if query_calls == 1 else "future")
            grid = value[:, 0] if query_calls == 1 else value
            hidden = grid[:, 0, :128]
            _need(same_bits(grid[:, :, :128], hidden[:, None, :].expand(16, 5, 128)),
                  "HIDDEN_TIMEFRAMES")
            _need(same_bits(grid[:, :, 128:133], tf_reference[None].expand(16, 5, 5)),
                  "TF_INPUT")
            fraction = grid[:, :, 133]
            _need(bool(((fraction > 0) & (fraction <= 1)).all().item())
                  and bool((fraction[:, 0] == 1).all().item())
                  and (query_calls == 1 or bool((fraction == 1).all().item())),
                  "QUERY_FRACTIONS")
            if query_calls == 1:
                hidden_reference = hidden.clone()
                captured["joint_hidden"] = hidden.cpu().numpy().copy()
            else:
                _need(same_bits(hidden, hidden_reference), "HIDDEN_STATE_FUTURE")
        except BaseException:
            failed = True
            raise

    try:
        handles.append(norm.register_forward_pre_hook(source_hook))
        handles.append(projection.register_forward_pre_hook(query_hook))
        yield captured
        _need(not failed and source_calls == 1 and query_calls == 2
              and set(captured) == {"joint_source", "joint_hidden"}, "INCOMPLETE")
    finally:
        for handle in reversed(handles):
            handle.remove()
