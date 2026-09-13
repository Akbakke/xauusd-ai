from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch

from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.unified_exit_legacy_state_v1 import RETIRED_STATIC_EXIT_STATE_KEYS
from gx1.contracts.unified_exit_random_access_model_v1 import (
    RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
    bind_preserved_v7_input_normalization,
    bootstrap_random_access_v2_from_pretrained,
    strict_load_random_access_v2_state,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import canonical_sha256
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_PATH_FEATURE_DIM,
)
from tests.test_entry_v10_ctx_model_shapes import (
    _make_exit_episode_inputs,
    _make_model,
)


def _inputs(batch_size: int = 3) -> dict:
    base = _make_exit_episode_inputs(state_count=1, batch_size=batch_size)
    tail_rows = 5
    lengths = torch.tensor([5, 3, 4][:batch_size], dtype=torch.long)
    path = torch.randn(batch_size, 2, tail_rows, UNIFIED_EXIT_PATH_FEATURE_DIM)
    for index, length in enumerate(lengths.tolist()):
        path[index, :, length:] = 0.0
    return {
        "entry_decision_representation": base["entry_decision_representation"],
        "m1_local_history_x": base["exit_local_history_x"],
        "state_ctx_cat": base["exit_state_ctx_cat"][:, 0],
        "state_ctx_cont": base["exit_state_ctx_cont"][:, 0],
        "trade_path_tail_x": path,
        "trade_path_lengths": lengths,
        "normalized_lifetime_summary_x": torch.randn(batch_size, 2, 7),
        "exit_mtf_histories": base["exit_mtf_histories"],
        "exit_mtf_gathers": {
            name: gather[:, :1] for name, gather in base["exit_mtf_gathers"].items()
        },
        "exit_mtf_history_lengths": base["exit_mtf_history_lengths"],
        "action_valid_mask": torch.ones(batch_size, 2, 2, dtype=torch.bool),
    }


def _slice(inputs: dict, index: int) -> dict:
    sliced = {
        name: value[index : index + 1]
        for name, value in inputs.items()
        if not isinstance(value, dict)
    }
    for key in (
        "exit_mtf_histories",
        "exit_mtf_gathers",
        "exit_mtf_history_lengths",
    ):
        sliced[key] = {
            name: value[index : index + 1] for name, value in inputs[key].items()
        }
    length = int(sliced["trade_path_lengths"][0])
    sliced["trade_path_tail_x"] = sliced["trade_path_tail_x"][:, :, :length]
    return sliced


def test_random_access_model_batch_matches_independent_state_calls() -> None:
    torch.manual_seed(20260911)
    batched_model = _make_model(dropout=0.0).eval()
    loop_model = copy.deepcopy(batched_model).eval()
    inputs = _inputs()
    calls = {"path_gru": 0, "q_head": 0}

    def count_path(_module, _args, _output):
        calls["path_gru"] += 1

    def count_head(_module, _args, _output):
        calls["q_head"] += 1

    path_hook = batched_model.exit_episode_path_gru.register_forward_hook(count_path)
    head_hook = batched_model.head_exit_action.register_forward_hook(count_head)
    try:
        output = batched_model.forward_exit_random_access_batch(**inputs)
        batched = output["exit_action_q_bps"]
    finally:
        path_hook.remove()
        head_hook.remove()
    assert calls == {"path_gru": 1, "q_head": 1}
    for name in (
        "exit_specialist_gate",
        "exit_tf_gate",
        "exit_family_tf_cooperation_gate",
        "exit_family_tf_feature_gate",
    ):
        assert isinstance(output[name], torch.Tensor)
        assert output[name].shape[0] == 3
    loop = torch.cat(
        [
            loop_model.forward_exit_random_access_batch(**_slice(inputs, index))[
                "exit_action_q_bps"
            ]
            for index in range(3)
        ],
        dim=0,
    )
    assert torch.allclose(batched, loop, rtol=1e-5, atol=1e-5)

    batched.square().sum().backward()
    loop.square().sum().backward()
    for name in (
        "exit_random_access_summary_proj.1.weight",
        "exit_random_access_fuse.1.weight",
        "head_exit_action.weight",
        "exit_episode_global_gru.weight_ih_l0",
    ):
        batch_parameter = dict(batched_model.named_parameters())[name]
        loop_parameter = dict(loop_model.named_parameters())[name]
        assert torch.allclose(
            batch_parameter.grad, loop_parameter.grad, rtol=2e-5, atol=2e-6
        )


@pytest.mark.parametrize("compact", [False, True])
def test_market_cache_reuses_only_market_state_across_different_positions(compact):
    torch.manual_seed(20260913)
    model = _make_model(dropout=0.0).eval()
    before = canonical_model_state_sha256(model.state_dict())
    inputs = _inputs()
    cache = {}
    market_names = {"m1_local_history_x", "state_ctx_cat", "state_ctx_cont",
                    "exit_mtf_histories", "exit_mtf_gathers", "exit_mtf_history_lengths"}
    def evaluate(values, keys):
        values = dict(values)
        extra = {}
        if compact:
            missing = {}
            for i, key in enumerate(keys):
                if key not in cache:
                    missing.setdefault(key, i)
            positions = list(missing.values())
            def select(value):
                if not positions:
                    return None
                if isinstance(value, dict):
                    return {name: select(item) for name, item in value.items()}
                return value[positions]
            for name in market_names:
                values[name] = select(values[name])
            extra["_market_state_batch_positions"] = positions
        return model.forward_exit_random_access_batch(
            **values, _market_state_cache=cache, _market_state_keys=keys, **extra)
    with torch.inference_mode():
        with patch.object(model, '_forward_exit_causal_episode',
                          wraps=model._forward_exit_causal_episode) as scan:
            cold = evaluate(inputs, [101, 102, 103])
            assert scan.call_count == 1
            changed = copy.deepcopy(inputs)
            changed['entry_decision_representation'] *= 0.5
            changed['trade_path_tail_x'] *= 1.2
            changed['normalized_lifetime_summary_x'] += 0.25
            actual = evaluate(changed, [101, 102, 103])
            assert scan.call_count == 1  # No repeated market GRUs or gates.
            assert not torch.equal(cold['exit_action_q_bps'], actual['exit_action_q_bps'])
            order = torch.tensor([1, 0, 2])

            def reorder(value):
                if isinstance(value, dict):
                    return {k: reorder(v) for k, v in value.items()}
                return value.index_select(0, order)

            mixed = {k: reorder(v) for k, v in changed.items()}
            mixed_actual = evaluate(mixed, [102, 101, 104])
            assert scan.call_count == 2
            assert scan.call_args.kwargs['exit_local_history_x'].shape[0] == 1
        expected = model.forward_exit_random_access_batch(**changed)
        mixed_expected = model.forward_exit_random_access_batch(**mixed)
    for result, reference in ((actual, expected), (mixed_actual, mixed_expected)):
        assert result.keys() == reference.keys()
        for name in result:
            torch.testing.assert_close(result[name], reference[name], atol=1e-5, rtol=1e-5, msg=name)
        assert torch.equal(result['exit_action_q_bps'].argmax(-1),
                           reference['exit_action_q_bps'].argmax(-1))
    assert set(cache) == {101, 102, 103, 104}
    assert canonical_model_state_sha256(model.state_dict()) == before


@pytest.mark.parametrize('training,grad_enabled', [(True, False), (False, True)])
def test_market_cache_cannot_enter_training_or_gradient_path(training, grad_enabled):
    model = _make_model(dropout=0.0).train(training)
    with torch.set_grad_enabled(grad_enabled):
        with pytest.raises(RuntimeError, match='MARKET_CACHE_REQUIRES_FROZEN_EVAL'):
            model.forward_exit_random_access_batch(
                **_inputs(), _market_state_cache={}, _market_state_keys=[101, 102, 103])


@pytest.mark.parametrize("tail_rows", [5, 512])
def test_shared_val_path_preserves_both_sides_and_encodes_once(tail_rows):
    torch.manual_seed(20260913)
    model = _make_model(dropout=0.0).eval()
    inputs = _inputs()
    lengths = torch.tensor([tail_rows, tail_rows - 2, tail_rows - 1])
    path = torch.randn(3, 1, tail_rows, UNIFIED_EXIT_PATH_FEATURE_DIM).expand(-1, 2, -1, -1).clone()
    for row, length in enumerate(lengths):
        path[row, :, length:] = 0
    inputs.update(trade_path_tail_x=path, trade_path_lengths=lengths)
    before = canonical_model_state_sha256(model.state_dict())
    batches = []
    hook = model.exit_episode_path_gru.register_forward_pre_hook(
        lambda module, args: batches.append(args[0].shape[0]))
    try:
        with torch.inference_mode():
            reference = model.forward_exit_random_access_batch(**inputs)
            shared = model.forward_exit_random_access_batch(**inputs, _share_identical_path=True)
    finally:
        hook.remove()
    assert batches == [6, 3]
    for name in reference:
        torch.testing.assert_close(shared[name], reference[name], atol=1e-4, rtol=0.0, msg=name)
    assert torch.equal(shared["exit_action_q_bps"].argmax(-1), reference["exit_action_q_bps"].argmax(-1))
    assert canonical_model_state_sha256(model.state_dict()) == before
    # Side-specific summaries and heads remain independent.
    assert not torch.equal(shared["exit_random_access_summary_state"][:, 0],
                           shared["exit_random_access_summary_state"][:, 1])
    inputs["trade_path_tail_x"][0, 1, 0, 0] += 1
    with torch.inference_mode(), pytest.raises(RuntimeError, match="SHARED_PATH_REQUIRES_IDENTICAL_FROZEN_EVAL"):
        model.forward_exit_random_access_batch(**inputs, _share_identical_path=True)


@pytest.mark.parametrize("training,grad_enabled", [(True, False), (False, True)])
def test_shared_path_cannot_change_training(training, grad_enabled):
    model = _make_model(dropout=0.0).train(training)
    inputs = _inputs()
    inputs["trade_path_tail_x"][:, 1] = inputs["trade_path_tail_x"][:, 0]
    with torch.set_grad_enabled(grad_enabled), pytest.raises(
        RuntimeError, match="SHARED_PATH_REQUIRES_IDENTICAL_FROZEN_EVAL"
    ):
        model.forward_exit_random_access_batch(**inputs, _share_identical_path=True)


@pytest.mark.parametrize("with_retired_static_exit", [False, True])
def test_v1_bootstrap_is_explicit_once_then_v2_restore_is_strict(
    with_retired_static_exit: bool,
) -> None:
    torch.manual_seed(3)
    model = _make_model(dropout=0.0)
    full = copy.deepcopy(model.state_dict())
    new_prefixes = (
        "unified_exit_random_access_architecture_sha256",
        "exit_random_access_summary_proj.",
        "exit_random_access_fuse.",
    )
    old = {
        name: value.clone()
        for name, value in full.items()
        if not any(name == prefix or name.startswith(prefix) for prefix in new_prefixes)
    }
    old["head_exit_action.weight"] = torch.full_like(
        old["head_exit_action.weight"], 0.125
    )
    with pytest.raises(RuntimeError, match="V2_STATE_KEYSET_INVALID"):
        strict_load_random_access_v2_state(model, old)
    new_before = {
        name: value.clone() for name, value in full.items() if name not in old
    }
    if with_retired_static_exit:
        old.update({
            name: torch.tensor([float(index)])
            for index, name in enumerate(sorted(RETIRED_STATIC_EXIT_STATE_KEYS))
        })
    source_digest = canonical_model_state_sha256(old)
    receipt = bootstrap_random_access_v2_from_pretrained(model, old)
    assert receipt["decision"] == "PASS"
    assert receipt["source_pretrained_state_sha256"] == source_digest
    assert canonical_model_state_sha256(old) == source_digest
    assert receipt["reused_state_key_count"] == len(full) - len(new_before)
    assert receipt["retired_state_keys"] == (
        sorted(RETIRED_STATIC_EXIT_STATE_KEYS) if with_retired_static_exit else []
    )
    for name in set(full) - set(new_before):
        assert torch.equal(model.state_dict()[name], old[name])
    assert receipt["architecture_schema_version"] == RANDOM_ACCESS_MODEL_SCHEMA_VERSION
    assert torch.equal(
        model.state_dict()["head_exit_action.weight"],
        old["head_exit_action.weight"],
    )
    for name, value in new_before.items():
        assert torch.equal(model.state_dict()[name], value)
    v2_state = copy.deepcopy(model.state_dict())
    strict_digest = strict_load_random_access_v2_state(model, v2_state)
    assert strict_digest == receipt["resulting_v2_state_sha256"]


def test_bootstrap_explicitly_binds_preserved_v7_normalization() -> None:
    model = _make_model(dropout=0.0)
    old_contract = copy.deepcopy(model._input_normalization_contract)
    old_contract["schema_version"] = "entry_model_native_input_normalization_v7"
    old_contract["transform"] = "shared_entry_exit_train_only_median_raw_iqr_asinh_v4"
    old_contract.pop("contract_sha256")
    old_contract["contract_sha256"] = canonical_sha256(old_contract)
    old = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not (
            name == "unified_exit_random_access_architecture_sha256"
            or name.startswith("exit_random_access_summary_proj.")
            or name.startswith("exit_random_access_fuse.")
        )
    }
    old["input_norm_contract_sha256"] = torch.tensor(
        list(bytes.fromhex(old_contract["contract_sha256"])), dtype=torch.uint8
    )
    bootstrap_random_access_v2_from_pretrained(model, old)
    assert (
        bind_preserved_v7_input_normalization(model, old_contract)
        == old_contract["contract_sha256"]
    )


@pytest.mark.parametrize("corruption", ["partial_retirement", "unknown_extra", "missing_live", "tensor_shape"])
def test_bootstrap_rejects_unreviewed_key_or_tensor_changes(corruption: str) -> None:
    from gx1.contracts.unified_exit_random_access_model_v1 import _new_state_keys

    model = _make_model(dropout=0.0)
    new_keys = set(_new_state_keys(model))
    old = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if name not in new_keys
    }
    old.update({name: torch.ones(1) for name in RETIRED_STATIC_EXIT_STATE_KEYS})
    if corruption == "partial_retirement":
        del old[sorted(RETIRED_STATIC_EXIT_STATE_KEYS)[0]]
    elif corruption == "unknown_extra":
        old["exit_path_encoder.layers.99.norm1.weight"] = torch.ones(1)
    elif corruption == "missing_live":
        del old["head_exit_action.weight"]
    else:
        old["head_exit_action.weight"] = torch.ones(1)
    before = canonical_model_state_sha256(model.state_dict())
    with pytest.raises(RuntimeError, match="BOOTSTRAP_(KEYSET|TENSOR)_INVALID"):
        bootstrap_random_access_v2_from_pretrained(model, old)
    assert canonical_model_state_sha256(model.state_dict()) == before
