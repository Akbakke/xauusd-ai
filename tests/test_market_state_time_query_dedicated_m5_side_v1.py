"""Small CPU decoder checks; no native backbone forward, data, fit or checkpoint load."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from gx1.contracts.entry_model_native_training_objective_v1 import (
    ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
)
from gx1.contracts.market_state_time_query_model_v2 import (
    FactorizedTimeQueryMarketStateModel, factorized_time_query_model_contract,
)
from gx1.contracts.market_state_time_query_native_F_model_v1 import (
    NativeFutureDualModel, contract as dual_contract,
)
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
from gx1.features.entry_specialist_feature_groups_v1 import (
    FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME, compile_full_ta_entry_execution_profile,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    CtxModelConfig, EntryV10CtxHybridTransformer,
)

WIDTH = 128
# Recorded before this opt-in change from the sealed TA candidate view/recovery recipe.
LEGACY_TA_S_SHA = "f3dc272ae8df5d05d49bbb44a419b88d34fda2d046c9a85eec480b3e1619b747"
LEGACY_TA_DUAL_SHA = "bb55e545467b15556e28442360f641ff0bf4e145d8d394001118ebef910542af"


def _backbone(*, ta_only=True):
    native = EntryV10CtxHybridTransformer.__new__(EntryV10CtxHybridTransformer)
    nn.Module.__init__(native)
    cfg = CtxModelConfig.__new__(CtxModelConfig)
    object.__setattr__(cfg, "d_model", WIDTH)
    object.__setattr__(cfg, "entry_training_phase", ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    native.cfg = cfg
    native.entry_training_phase = cfg.entry_training_phase
    native.entry_q_joint_in = nn.Linear(4 * WIDTH, WIDTH)
    if ta_only:
        profile = compile_full_ta_entry_execution_profile()
        native.entry_execution_profile = {key: profile[key] for key in ("name", "contract_sha256")}
        native.gc_mtf_history = None
    else:
        native.entry_execution_profile = {"name": FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME}
        native.gc_mtf_history = {"normalization": {}}
    return native


def _model(*, dedicated_m5_side=False, ta_only=True):
    return NativeFutureDualModel(FactorizedTimeQueryMarketStateModel(
        _backbone(ta_only=ta_only), dedicated_m5_side=dedicated_m5_side))


def _inputs():
    hidden = torch.linspace(-1.3, 1.7, 3 * WIDTH).reshape(3, WIDTH)
    query = torch.tensor([[NATIVE_K, (1, 1, 2, 7, 21)]], dtype=torch.int64).expand(3, -1, -1)
    return hidden, query


def _decode(model):
    hidden, query = _inputs()
    return (model.state_model.decode_raw_query_grid(hidden, query),
            model.decode_native_F_raw(hidden))


def test_legacy_contract_hashes_and_opt_in_identity():
    assert factorized_time_query_model_contract(ta_only=True)["contract_sha256"] == LEGACY_TA_S_SHA
    assert dual_contract(ta_only=True)["contract_sha256"] == LEGACY_TA_DUAL_SHA
    for ta in (False, True):
        for contract in (factorized_time_query_model_contract, dual_contract):
            assert contract(ta_only=ta) == contract(ta_only=ta, dedicated_m5_side=False)
            assert "dedicated_m5_side" not in contract(ta_only=ta)
    model = _model(dedicated_m5_side=True)
    s = factorized_time_query_model_contract(ta_only=True, dedicated_m5_side=True)
    f = dual_contract(ta_only=True, dedicated_m5_side=True)
    assert s["contract_sha256"] != LEGACY_TA_S_SHA
    assert f["contract_sha256"] != LEGACY_TA_DUAL_SHA
    assert f["S_model_contract_sha256"] == s["contract_sha256"]
    assert model.state_model.research_objective == s
    assert model.research_objective == f
    model._require_binding()


@pytest.mark.parametrize("ta_only", [False, True])
def test_default_preserves_legacy_layer_initialization_rng_state_keys_and_output(ta_only):
    # The original allocation order: backbone, shared projection, S head, F head.
    torch.manual_seed(174)
    backbone = _backbone(ta_only=ta_only)
    projection = nn.Sequential(nn.Linear(WIDTH + 6, WIDTH), nn.GELU())
    state_head = nn.Linear(WIDTH, 2)
    future_head = nn.Linear(WIDTH, 2)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(174)
    model = _model(ta_only=ta_only)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for actual, expected in (
        (model.backbone, backbone), (model.state_model.query_projection, projection),
        (model.state_model.query_head, state_head), (model.future_head, future_head),
    ):
        assert actual.state_dict().keys() == expected.state_dict().keys()
        assert all(torch.equal(actual.state_dict()[key], value)
                   for key, value in expected.state_dict().items())
    assert set(model.state_dict()) == {
        "state_model.backbone.entry_q_joint_in.weight", "state_model.backbone.entry_q_joint_in.bias",
        "state_model.query_projection.0.weight", "state_model.query_projection.0.bias",
        "state_model.query_head.weight", "state_model.query_head.bias",
        "future_head.weight", "future_head.bias", "state_model.query_tf_coordinates",
        "state_model.query_native_k", "state_model.query_objective_sha256",
        "native_F_objective_sha256",
    }
    hidden, query = _inputs()
    tf = torch.eye(5)[None, None].expand(3, 2, -1, -1)
    fraction = query.float().div(torch.tensor(NATIVE_K)).unsqueeze(-1)
    representation = projection(torch.cat((hidden[:, None, None].expand(-1, 2, 5, -1),
                                           tf, fraction), dim=-1))
    actual_s, actual_f = _decode(model)
    assert torch.equal(actual_s, state_head(representation))
    native = projection(torch.cat((hidden[:, None].expand(-1, 5, -1),
                                   torch.eye(5)[None].expand(3, -1, -1),
                                   torch.ones(3, 5, 1)), dim=-1))
    assert torch.equal(actual_f, future_head(native))
    # A default checkpoint still strict-loads with the original key set.
    restored = _model(ta_only=ta_only)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert all(torch.equal(a, b) for a, b in zip(_decode(restored), (actual_s, actual_f)))


def test_opt_in_clones_side_rows_without_shifting_legacy_weights_or_rng():
    torch.manual_seed(96)
    old = _model()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(96)
    new = _model(dedicated_m5_side=True)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    old_parameters = dict(old.named_parameters())
    new_parameters = dict(new.named_parameters())
    assert all(torch.equal(parameter, new_parameters[name])
               for name, parameter in old_parameters.items())
    added = set(new_parameters) - set(old_parameters)
    assert added == {"state_model.m5_side_head.weight", "state_model.m5_side_head.bias",
                     "future_m5_side_head.weight", "future_m5_side_head.bias"}
    assert sum(new_parameters[name].numel() for name in added) == 258
    for head, shared in ((new.state_model.m5_side_head, new.state_model.query_head),
                         (new.future_m5_side_head, new.future_head)):
        assert torch.equal(head.weight, shared.weight[1:2])
        assert torch.equal(head.bias, shared.bias[1:2])
        assert head.weight.untyped_storage().data_ptr() != shared.weight.untyped_storage().data_ptr()
    for a, b in zip(_decode(old), _decode(new)):
        assert torch.equal(a[..., 0], b[..., 0])
        assert torch.equal(a[..., 1:, 1], b[..., 1:, 1])
        torch.testing.assert_close(a[..., 0, 1], b[..., 0, 1], rtol=0, atol=1e-6)
    before = _decode(new)
    with torch.no_grad():
        new.state_model.m5_side_head.bias.add_(1)
        new.future_m5_side_head.bias.sub_(2)
    for a, b, shift in zip(before, _decode(new), (1, -2)):
        assert torch.equal(a[..., 0], b[..., 0])
        assert torch.equal(a[..., 1:, 1], b[..., 1:, 1])
        torch.testing.assert_close(b[..., 0, 1] - a[..., 0, 1],
                                   torch.full_like(a[..., 0, 1], shift), rtol=0, atol=1e-6)


@pytest.mark.parametrize("task", ["state", "future"])
@pytest.mark.parametrize("coordinate", ["m5_side", "other_side", "activity"])
def test_gradient_routes_only_selected_output_head(task, coordinate):
    torch.manual_seed(13)
    model = _model(dedicated_m5_side=True)
    hidden, query = _inputs()
    hidden.requires_grad_(True)
    if task == "state":
        raw = model.state_model.decode_raw_query_grid(hidden, query)[:, 0]
        shared, dedicated = model.state_model.query_head, model.state_model.m5_side_head
        unused = (model.future_head, model.future_m5_side_head)
    else:
        raw = model.decode_native_F_raw(hidden)
        shared, dedicated = model.future_head, model.future_m5_side_head
        unused = (model.state_model.query_head, model.state_model.m5_side_head)
    selected = {"m5_side": raw[:, 0, 1], "other_side": raw[:, 1:, 1],
                "activity": raw[:, :, 0]}[coordinate]
    selected.sum().backward()
    def nonzero(value):
        return value is not None and bool(torch.count_nonzero(value))
    if coordinate == "m5_side":
        assert nonzero(dedicated.weight.grad) and nonzero(dedicated.bias.grad)
        assert not nonzero(shared.weight.grad) and not nonzero(shared.bias.grad)
    else:
        row = 1 if coordinate == "other_side" else 0
        assert nonzero(shared.weight.grad[row]) and nonzero(shared.bias.grad[row])
        assert not nonzero(shared.weight.grad[1 - row])
        assert not nonzero(dedicated.weight.grad) and not nonzero(dedicated.bias.grad)
    assert all(parameter.grad is None for head in unused for parameter in head.parameters())
    assert nonzero(model.state_model.query_projection[0].weight.grad)
    assert nonzero(hidden.grad)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_non_boolean_mode_is_rejected(value):
    for contract in (factorized_time_query_model_contract, dual_contract):
        with pytest.raises(ValueError, match="DEDICATED_M5_SIDE_MODE"):
            contract(ta_only=True, dedicated_m5_side=value)
    with pytest.raises(RuntimeError, match="DEDICATED_M5_SIDE_MODE"):
        FactorizedTimeQueryMarketStateModel(_backbone(), dedicated_m5_side=value)


def test_dedicated_mode_requires_explicit_ta_profile():
    for contract in (factorized_time_query_model_contract, dual_contract):
        with pytest.raises(ValueError, match="DEDICATED_M5_SIDE_MODE"):
            contract(dedicated_m5_side=True)
    with pytest.raises(RuntimeError, match="DEDICATED_M5_SIDE_MODE"):
        _model(ta_only=False, dedicated_m5_side=True)


@pytest.mark.parametrize("task", ["state", "future"])
@pytest.mark.parametrize("damage", ["missing", "shape", "nonfinite", "contract", "unexpected"])
def test_invalid_head_or_contract_binding_fails_closed(task, damage):
    model = _model(dedicated_m5_side=damage != "unexpected")
    owner = model.state_model if task == "state" else model
    name = "m5_side_head" if task == "state" else "future_m5_side_head"
    if damage == "missing":
        delattr(owner, name)
    elif damage == "shape":
        getattr(owner, name).weight = nn.Parameter(torch.zeros(2, WIDTH))
    elif damage == "nonfinite":
        with torch.no_grad():
            getattr(owner, name).bias.fill_(float("nan"))
    elif damage == "contract":
        getattr(owner, "query_objective_sha256" if task == "state"
                else "native_F_objective_sha256")[0] ^= 1
    else:
        setattr(owner, name, nn.Linear(WIDTH, 1))
    with pytest.raises(RuntimeError, match="BINDING|BUFFER"):
        model._require_binding()


def test_dual_forward_still_encodes_once(monkeypatch):
    model = _model(dedicated_m5_side=True)
    hidden, query = _inputs()
    calls = []
    def encode(**features):
        calls.append(set(features))
        return hidden
    monkeypatch.setattr(model, "encode_features", encode)
    values = {key: torch.zeros(3, 1, 1) for key in
              ("seq_x", "snap_x", "ctx_cat", "ctx_cont",
               "seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1")}
    out = model(**values, query_k=query[:, 0])
    assert calls == [set(values)]
    # Match the same Q=1 decoder shape used by forward; different GEMM shapes
    # need not round bit-identically even for the same first query.
    expected_s = model.state_model.decode_raw_query_grid(hidden, query[:, :1])[:, 0]
    expected_f = model.decode_native_F_raw(hidden)
    assert torch.equal(out["state_activity_side_logits"], expected_s)
    assert torch.equal(out["future_activity_side_logits"], expected_f)
