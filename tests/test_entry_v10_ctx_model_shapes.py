#!/usr/bin/env python3
"""Shape and gradient proofs for the one exact ENTRY model-native architecture."""

from __future__ import annotations

import inspect

import pytest
import torch

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
)
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import INPUTS
from gx1.contracts.entry_model_native_tf_input_scale_v1 import (
    MIN_EFFECTIVE_SCALE,
    build_tf_input_scale_contract,
    require_tf_input_scale_state,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EXACT_CTX_CAT_DOMAINS,
    EXACT_SPECIALIST_NAMES,
    EntryV10CtxHybridTransformer,
)
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_PATH_FEATURE_DIM,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V2
from tests.model_native_input_normalization_support import (
    input_normalization_fixture,
)


SEQ_DIM = 16
SEQ_LEN = 4
TF_DIM = len(EXACT_SPECIALIST_NAMES)
INPUT_NORMALIZATION = input_normalization_fixture(
    signal_names=[f"signal_{index}" for index in range(SEQ_DIM)],
    mtf_names=[f"mtf_{index}" for index in range(TF_DIM)],
)


def _specialist_indices() -> dict[str, list[int]]:
    grouped = {name: [] for name in EXACT_SPECIALIST_NAMES}
    for index in range(SEQ_DIM):
        grouped[EXACT_SPECIALIST_NAMES[index % len(EXACT_SPECIALIST_NAMES)]].append(index)
    return grouped


def _multi_tf_specialist_indices(width: int) -> dict[str, list[int]]:
    return {
        name: list(range(position, width, len(EXACT_SPECIALIST_NAMES)))
        for position, name in enumerate(EXACT_SPECIALIST_NAMES)
    }


def _make_model(**overrides) -> EntryV10CtxHybridTransformer:
    kwargs = {
        "seq_input_dim": SEQ_DIM,
        "snap_input_dim": SEQ_DIM,
        "seq_len": SEQ_LEN,
        "dropout": 0.05,
        "multi_tf_num_layers": 1,
        "multi_tf_scale": 0.5,
        "specialist_num_layers": 1,
        "specialist_fusion_scale": 0.25,
        "cross_family_fusion_scale": 0.25,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "m5_seq_dim": TF_DIM,
        "m15_seq_dim": TF_DIM,
        "h1_seq_dim": TF_DIM,
        "h4_seq_dim": TF_DIM,
        "d1_seq_dim": TF_DIM,
        "m5_seq_len": SEQ_LEN,
        "m15_seq_len": SEQ_LEN,
        "h1_seq_len": SEQ_LEN,
        "h4_seq_len": SEQ_LEN,
        "d1_seq_len": SEQ_LEN,
        "specialist_input_indices": _specialist_indices(),
        "specialist_ctx_cont_indices": {
            str(name): list(values)
            for name, values in MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT[
                "ctx_cont_indices"
            ].items()
        },
        "specialist_ctx_cont_nominal_indices": {
            str(name): list(values)
            for name, values in MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT[
                "ctx_cont_nominal_indices"
            ].items()
        },
        "specialist_ctx_cat_indices": {
            str(name): list(values)
            for name, values in MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT[
                "ctx_cat_indices"
            ].items()
        },
        "temporal_alias_signal_indices": [],
        "temporal_alias_ctx_cont_indices": [],
        "input_normalization": INPUT_NORMALIZATION,
    }
    kwargs.update(overrides)
    kwargs.setdefault(
        "multi_tf_specialist_input_indices",
        _multi_tf_specialist_indices(int(kwargs["m5_seq_dim"])),
    )
    return EntryV10CtxHybridTransformer(**kwargs)


def _make_inputs(batch_size: int = 2) -> tuple:
    seq_x = torch.randn(batch_size, SEQ_LEN, SEQ_DIM)
    snap_x = seq_x[:, -1, :].clone()
    ctx_cont = torch.randn(batch_size, MODEL_NATIVE_CTX_CONT_DIM)
    nominal_indices = [
        index
        for values in MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT[
            "ctx_cont_nominal_indices"
        ].values()
        for index in values
    ]
    ctx_cont[:, nominal_indices] = torch.randint(
        0,
        5,
        (batch_size, len(nominal_indices)),
    ).float()
    ctx_cat = torch.stack(
        [
            torch.randint(0, len(domain), (batch_size,))
            for domain in EXACT_CTX_CAT_DOMAINS.values()
        ],
        dim=1,
    )
    return (
        seq_x,
        snap_x,
        ctx_cat,
        ctx_cont,
        {
            f"seq_{tf}": torch.randn(batch_size, SEQ_LEN, TF_DIM)
            for tf in ("m5", "m15", "h1", "h4", "d1")
        },
    )


def _forward(model: EntryV10CtxHybridTransformer, batch_size: int = 2) -> dict:
    seq_x, snap_x, ctx_cat, ctx_cont, mtf = _make_inputs(batch_size)
    return model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)


def _make_exit_feature_inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    _entry_seq, _entry_snap, ctx_cat, ctx_cont, _mtf = _make_inputs(batch_size)
    seq = torch.randn(batch_size, SEQ_LEN * 5, SEQ_DIM)
    return {
        "exit_feature_seq_x": seq,
        "exit_feature_snap_x": seq[:, -1, :].clone(),
        "exit_feature_ctx_cat": ctx_cat,
        "exit_feature_ctx_cont": ctx_cont,
    }


def test_exact_architecture_emits_every_mandatory_head_with_exact_width() -> None:
    model = _make_model().eval()
    out = _forward(model, batch_size=2)
    widths = {
        "direction_logits": 3,
        "raw_direction_logits": 3,
        "model_native_logits": 3,
        "mtf_dir_logits": 3,
        "path_quality_raw": 1,
        "path_quality": 1,
        "mfe_first_n": 1,
        "tradable_logit": 1,
        "bad_path_logit_raw": 1,
        "bad_path_logit": 1,
        "clean_edge_logit": 1,
        "survival_logit": 1,
        "specialist_gate": 8,
        "tf_gate": 5,
        "family_tf_cooperation_gate": 5 * len(EXACT_SPECIALIST_NAMES),
        "trade_logit": 1,
        "side_logits": 2,
        "side_utility": 2,
        "side_bad_path_logit": 2,
        "side_mae": 2,
        "side_validity_logit": 2,
        "trendline_rail_logits": 6,
        "tf_agreement_logit": 1,
        "path_quality_log_var": 1,
        "position_size_logit": 1,
        "dip_pred": 18,
        "forecast_pred": 4,
        "timing_pred": 12,
        "tail_risk_pred": 6,
        "vol_forecast_pred": 3,
        "action_value": 9,
        "expectile_value": 3,
        "action_advantage": 9,
        "public_trade_flat_decision_logits": 2,
        "shared_feature_representation": 128,
    }
    for name, width in widths.items():
        assert out[name].shape == (2, width), name
        assert torch.isfinite(out[name]).all(), name
    assert torch.all(out["timing_pred"] >= 0.0)
    assert torch.all(out["timing_pred"] <= 1.0)
    for gate_name in ("specialist_gate", "tf_gate", "family_tf_cooperation_gate"):
        assert torch.allclose(
            out[gate_name].sum(dim=1),
            torch.ones(2),
            atol=1e-6,
        )
    assert out["family_tf_feature_gate"].shape == (2, 5, TF_DIM)
    assert torch.isfinite(out["family_tf_feature_gate"]).all()


def test_unified_exit_head_consumes_shared_entry_state_and_exact_m1_prefix() -> None:
    model = _make_model().train()
    out = _forward(model, batch_size=2)
    exit_features = _make_exit_feature_inputs(2)
    path = torch.randn(2, 4, UNIFIED_EXIT_PATH_FEATURE_DIM)
    path[1, 2:, :] = 0.0
    exit_out = model.forward_exit_action(
        entry_shared_representation=out["shared_feature_representation"],
        **exit_features,
        exit_path_x=path,
        exit_path_lengths=torch.tensor([4, 2], dtype=torch.long),
        exit_side_index=torch.tensor([0, 1], dtype=torch.long),
    )
    assert exit_out["exit_action_logits"].shape == (2, 2)
    assert exit_out["exit_action_probs"].shape == (2, 2)
    assert exit_out["exit_path_attention"].shape == (2, 1, 4)
    assert torch.isfinite(exit_out["exit_action_logits"]).all()
    assert torch.allclose(
        exit_out["exit_action_probs"].sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )

    torch.nn.functional.cross_entropy(
        exit_out["exit_action_logits"],
        torch.tensor([0, 1], dtype=torch.long),
    ).backward()
    for parameter_name in (
        "seq_proj.weight",
        "exit_path_proj.weight",
        "exit_entry_path_attention.in_proj_weight",
        "head_exit_action.weight",
    ):
        parameter = dict(model.named_parameters())[parameter_name]
        assert parameter.grad is not None, parameter_name
        assert bool(torch.count_nonzero(parameter.grad).item()), parameter_name


def test_unified_exit_head_padding_is_exact_and_cannot_hide_path_values() -> None:
    model = _make_model(dropout=0.0).eval()
    shared = _forward(model, batch_size=1)["shared_feature_representation"]
    exit_features = _make_exit_feature_inputs(1)
    prefix = torch.randn(1, 2, UNIFIED_EXIT_PATH_FEATURE_DIM)
    exact = model.forward_exit_action(
        entry_shared_representation=shared,
        **exit_features,
        exit_path_x=prefix,
        exit_path_lengths=torch.tensor([2], dtype=torch.long),
        exit_side_index=torch.tensor([0], dtype=torch.long),
    )
    padded = torch.zeros(1, 4, UNIFIED_EXIT_PATH_FEATURE_DIM)
    padded[:, :2, :] = prefix
    padded_out = model.forward_exit_action(
        entry_shared_representation=shared,
        **exit_features,
        exit_path_x=padded,
        exit_path_lengths=torch.tensor([2], dtype=torch.long),
        exit_side_index=torch.tensor([0], dtype=torch.long),
    )
    assert torch.allclose(
        exact["exit_action_logits"],
        padded_out["exit_action_logits"],
        atol=1e-6,
    )

    padded[0, 3, 0] = 1.0
    with pytest.raises(
        RuntimeError,
        match="UNIFIED_EXIT_NONZERO_RIGHT_PADDING_FORBIDDEN",
    ):
        model.forward_exit_action(
            entry_shared_representation=shared,
            **exit_features,
            exit_path_x=padded,
            exit_path_lengths=torch.tensor([2], dtype=torch.long),
            exit_side_index=torch.tensor([0], dtype=torch.long),
        )


def test_public_trade_flat_decision_is_post_calibration_argmax_ssot() -> None:
    model = _make_model().eval()
    model.set_direction_calibration(
        2.0,
        torch.tensor([1.25, -0.75, 0.40], dtype=torch.float32),
    )
    out = _forward(model, batch_size=4)
    final_logits = out["direction_logits"]
    expected_final = out["raw_direction_logits"] / 2.0 + torch.tensor(
        [1.25, -0.75, 0.40], dtype=final_logits.dtype
    )
    expected_pair = torch.stack(
        (final_logits[:, :2].max(dim=1).values, final_logits[:, 2]), dim=1
    )
    assert torch.allclose(final_logits, expected_final, atol=1e-6)
    assert torch.allclose(out["public_trade_flat_decision_logits"], expected_pair, atol=1e-6)
    assert torch.equal(
        out["public_trade_flat_decision_logits"].argmax(dim=1) == 0,
        final_logits.argmax(dim=1) != 2,
    )


def test_public_direction_gradient_reaches_every_fused_evidence_head() -> None:
    model = _make_model().train()
    out = _forward(model, batch_size=4)
    loss = torch.nn.functional.cross_entropy(
        out["direction_logits"],
        torch.tensor([0, 1, 2, 0]),
    )
    loss.backward()
    for parameter in (
        model.head_direction.weight,
        model.head_mtf_direction.weight,
        model.head_path_quality.weight,
        model.head_path_quality_log_var.weight,
        model.head_mfe_first_n.weight,
        model.head_tradable.weight,
        model.head_bad_path.weight,
        model.head_clean_edge.weight,
        model.head_survival.weight,
        model.head_trade.weight,
        model.head_side.weight,
        model.head_side_utility.weight,
        model.head_side_bad_path.weight,
        model.head_side_mae.weight,
        model.head_side_validity.weight,
        model.head_trendline_rail.weight,
        model.head_tf_agreement.weight,
        model.head_position_size.weight,
        model.head_dip.weight,
        model.head_forecast.weight,
        model.head_timing.weight,
        model.head_tail_risk.weight,
        model.head_vol_forecast.weight,
        model.head_action_value.weight,
        model.head_expectile_value.weight,
        model.evidence_fusion_in.weight,
        model.evidence_fusion_out.weight,
        model.regime_film[-1].weight,
        model.cross_tf_out.weight,
        model.specialist_out.weight,
        model.family_tf_cooperation_out.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0.0


def test_public_direction_reaches_every_specialist_tf_and_cooperation_branch_after_cold_start() -> None:
    torch.manual_seed(1307)
    model = _make_model().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    targets = torch.tensor([0, 1, 2, 0])

    # The three residual outputs are intentionally zero-initialized.  The first
    # step must move their output projections; the second then proves gradient
    # reachability through every upstream cooperation branch.
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = _forward(model, batch_size=4)
        torch.nn.functional.cross_entropy(out["direction_logits"], targets).backward()
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    out = _forward(model, batch_size=4)
    torch.nn.functional.cross_entropy(out["direction_logits"], targets).backward()
    parameters = [
        model.specialist_cross_attn.layers[0].self_attn.in_proj_weight,
        model.family_axis_attn.layers[0].self_attn.in_proj_weight,
        model.timeframe_axis_attn.layers[0].self_attn.in_proj_weight,
        model.cross_tf_attn.layers[0].self_attn.in_proj_weight,
        model.specialist_gate.weight,
        model.specialist_token_gate.weight,
        model.tf_context_gate.weight,
        model.tf_token_gate.weight,
        model.family_tf_context_gate.weight,
        model.family_tf_token_gate.weight,
        *(projection.weight for projection in model.specialist_proj.values()),
        *(projection.weight for projection in model.mtf_family_proj.values()),
        *(
            encoder.layers[0].self_attn.in_proj_weight
            for encoder in model.mtf_family_encoder.values()
        ),
        *(
            gate.weight
            for gate in model.mtf_feature_context_gate.values()
        ),
    ]
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0.0


def test_each_of_26_evidence_groups_changes_a_direction_class_margin() -> None:
    torch.manual_seed(90210)
    model = _make_model().eval()
    with torch.no_grad():
        out = _forward(model, batch_size=3)
        evidence = {name: out[name] for name, _ in INPUTS}
        baseline = model._fuse_direction_evidence(evidence)
        baseline_centered = baseline - baseline.mean(dim=1, keepdim=True)
        assert len(INPUTS) == 26
        for name, _ in INPUTS:
            ablated = dict(evidence)
            ablated[name] = torch.zeros_like(evidence[name])
            changed = model._fuse_direction_evidence(ablated)
            changed_centered = changed - changed.mean(dim=1, keepdim=True)
            assert not torch.allclose(
                baseline_centered,
                changed_centered,
                atol=1e-9,
                rtol=1e-7,
            ), name


def test_report_only_path_calibration_cannot_change_direction_fusion() -> None:
    model = _make_model().eval()
    seq_x, snap_x, ctx_cat, ctx_cont, mtf = _make_inputs(batch_size=3)
    with torch.no_grad():
        before = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)
        model.set_path_calibration(1.7, 0.4, 2.3, -0.6)
        after = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)
    assert torch.equal(before["path_quality_raw"], after["path_quality_raw"])
    assert torch.equal(before["bad_path_logit_raw"], after["bad_path_logit_raw"])
    assert torch.equal(before["raw_direction_logits"], after["raw_direction_logits"])
    assert torch.equal(before["direction_logits"], after["direction_logits"])
    assert not torch.equal(before["path_quality"], after["path_quality"])
    assert not torch.equal(before["bad_path_logit"], after["bad_path_logit"])


@pytest.mark.parametrize(
    "scale_name",
        (
            "multi_tf_scale",
            "specialist_fusion_scale",
            "cross_family_fusion_scale",
        ),
)
def test_exact_architecture_rejects_zero_representation_scale(scale_name: str) -> None:
    with pytest.raises(RuntimeError, match="MANDATORY_REPRESENTATION_SCALE_INVALID"):
        _make_model(**{scale_name: 0.0})


def test_exact_architecture_requires_all_five_tf_inputs() -> None:
    model = _make_model().eval()
    seq_x, snap_x, ctx_cat, ctx_cont, mtf = _make_inputs()
    del mtf["seq_h4"]
    with pytest.raises(TypeError, match="seq_h4"):
        model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)


@pytest.mark.parametrize(
    "bad_key",
    ("ctx_cat", "ctx_cont"),
)
def test_exact_architecture_rejects_wrong_context_width(bad_key: str) -> None:
    model = _make_model().eval()
    seq_x, snap_x, ctx_cat, ctx_cont, mtf = _make_inputs()
    if bad_key == "ctx_cat":
        ctx_cat = torch.zeros(2, MODEL_NATIVE_CTX_CAT_DIM + 1, dtype=torch.long)
        pattern = "CTX_CAT_DIM_MISMATCH"
    else:
        ctx_cont = torch.zeros(2, MODEL_NATIVE_CTX_CONT_DIM + 1)
        pattern = "CTX_CONT_DIM_MISMATCH"
    with pytest.raises(RuntimeError, match=pattern):
        model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)


@pytest.mark.parametrize(
    ("field_index", "invalid_value"),
    [
        (index, invalid)
        for index, domain in enumerate(EXACT_CTX_CAT_DOMAINS.values())
        for invalid in (-1, domain[-1] + 1)
    ],
)
def test_ctx_cat_field_specific_domains_fail_closed(
    field_index: int,
    invalid_value: int,
) -> None:
    model = _make_model().eval()
    seq_x, snap_x, ctx_cat, ctx_cont, mtf = _make_inputs()
    ctx_cat[:, field_index] = invalid_value
    field = tuple(EXACT_CTX_CAT_DOMAINS)[field_index]

    with pytest.raises(RuntimeError, match=f"field={field}"):
        model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)


def test_ctx_cat_fields_use_separate_domain_sized_embedding_tables() -> None:
    model = _make_model()

    assert len(model.ctx_cat_embeddings) == len(EXACT_CTX_CAT_DOMAINS)
    assert [
        embedding.num_embeddings for embedding in model.ctx_cat_embeddings
    ] == [len(domain) for domain in EXACT_CTX_CAT_DOMAINS.values()]
    assert len(
        {
            embedding.weight.data_ptr()
            for embedding in model.ctx_cat_embeddings
        }
    ) == len(EXACT_CTX_CAT_DOMAINS)
    assert not hasattr(model, "ctx_cat_emb")
    assert not hasattr(model, "specialist_ctx_cont_norm")


def test_exact_architecture_eval_is_deterministic() -> None:
    model = _make_model().eval()
    seq_x, snap_x, ctx_cat, ctx_cont, mtf = _make_inputs(batch_size=1)
    first = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)
    second = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)
    for key in first:
        assert torch.allclose(first[key], second[key]), key


def test_tf_input_scale_cannot_become_zero_or_negative() -> None:
    model = _make_model()
    for tf_name in ("m5", "m15", "h1", "h4", "d1"):
        getattr(model, f"tf_input_scale_{tf_name}").data.fill_(-100.0)
        effective = model._effective_tf_input_scale(tf_name)
        assert torch.isfinite(effective)
        assert float(effective.item()) > 0.0
        assert float(effective.item()) == pytest.approx(
            MIN_EFFECTIVE_SCALE,
            rel=1e-6,
        )


def test_tf_input_scale_contract_is_bound_to_raw_state() -> None:
    model = _make_model()
    init = {
        "m5": 1.0,
        "m15": 1.0,
        "h1": 0.7,
        "h4": 0.5,
        "d1": 0.3,
    }
    raw = {
        name: float(model.state_dict()[f"tf_input_scale_{name}"].item())
        for name in init
    }
    contract = build_tf_input_scale_contract(
        init_effective=init,
        learned_raw=raw,
    )

    assert require_tf_input_scale_state(contract, model.state_dict()) == contract[
        "learned"
    ]
    model.tf_input_scale_h4.data.add_(0.01)
    with pytest.raises(RuntimeError, match="STATE_HASH_MISMATCH"):
        require_tf_input_scale_state(contract, model.state_dict())


def test_input_normalization_buffers_are_persistent_and_hash_bound() -> None:
    model = _make_model()

    model.require_input_normalization_state()
    assert "input_norm_signal_center" in model.state_dict()
    assert "input_norm_contract_sha256" in model.state_dict()

    model.input_norm_signal_center[0].add_(0.01)
    with pytest.raises(RuntimeError, match="STATE_BUFFER_MISMATCH"):
        model.require_input_normalization_state()


def test_input_normalization_applies_one_identical_clip_in_train_and_eval() -> None:
    # Clipping at the exact boundary IS the declared handling: the fit
    # contract caps TRAIN clipping at 2%, so beyond-boundary rows
    # legitimately occur in every split and at serve. Train and eval must
    # apply the one identical clamp.
    model = _make_model()
    center = model.input_norm_signal_center
    scale = model.input_norm_signal_scale
    raw = center.clone().view(1, 1, -1)
    raw[..., 0] = center[0] + scale[0] * 13.0

    model.eval()
    normalized_eval = model._normalize_input_surface(raw, surface="signal")
    assert float(normalized_eval[..., 0].item()) == 12.0

    model.train()
    normalized_train = model._normalize_input_surface(raw, surface="signal")
    assert float(normalized_train[..., 0].item()) == 12.0
    assert torch.equal(normalized_eval, normalized_train)


@pytest.mark.parametrize(
    ("field", "invalid_value", "error"),
    (
        (
            "ema_stack_aligned_v2",
            2.0,
            "MTF_EMA_STACK_DOMAIN_INVALID",
        ),
        ("regime_class_id", 5.0, "CATEGORICAL_VALUE_INVALID"),
    ),
)
def test_mtf_semantic_domains_fail_closed_at_model_boundary(
    field: str,
    invalid_value: float,
    error: str,
) -> None:
    mtf_names = list(MULTI_TF_PER_BAR_FEATURES_V2)
    normalization = input_normalization_fixture(
        signal_names=[f"signal_{index}" for index in range(SEQ_DIM)],
        mtf_names=mtf_names,
    )
    model = _make_model(
        m5_seq_dim=len(mtf_names),
        m15_seq_dim=len(mtf_names),
        h1_seq_dim=len(mtf_names),
        h4_seq_dim=len(mtf_names),
        d1_seq_dim=len(mtf_names),
        input_normalization=normalization,
    ).eval()
    raw = model.input_norm_mtf_m5_center.clone().view(1, 1, -1)
    raw[..., mtf_names.index(field)] = invalid_value

    with pytest.raises(RuntimeError, match=error):
        model._normalize_input_surface(raw, surface="mtf_m5")


def test_model_and_config_expose_no_architecture_disable_switches() -> None:
    model_parameters = inspect.signature(EntryV10CtxHybridTransformer).parameters
    assert not [name for name in model_parameters if name.startswith("enable_")]
    assert model_parameters["dropout"].default is inspect.Parameter.empty
    config_fields = EntryV10CtxHybridTransformer.__init__.__annotations__
    assert not [name for name in config_fields if name.startswith("enable_")]


@pytest.mark.parametrize("invalid_dropout", (-0.01, 1.0, float("nan"), True))
def test_model_rejects_invalid_explicit_dropout(invalid_dropout: object) -> None:
    with pytest.raises(RuntimeError, match="MODEL_DROPOUT_INVALID"):
        _make_model(dropout=invalid_dropout)
