#!/usr/bin/env python3
"""Shape and gradient proofs for the one exact ENTRY model-native architecture."""

from __future__ import annotations

import inspect

import pytest
import torch

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_MTF_CONTEXT_COUNT,
    ENTRY_MTF_CONTEXT_TIMEFRAMES,
    EXIT_FEATURE_SEQUENCE_BARS,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_model_native_tf_input_scale_v1 import (
    MIN_EFFECTIVE_SCALE,
    build_tf_input_scale_contract,
    require_tf_input_scale_state,
)
from gx1.contracts.unified_exit_incremental_carry_v1 import (
    UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256,
    build_unified_exit_incremental_carry_envelope,
    decode_unified_exit_incremental_carry_tensors,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EXACT_CTX_CAT_DOMAINS,
    EXACT_SPECIALIST_NAMES,
    EntryV10CtxHybridTransformer,
    _build_unit_test_entry_v10_ctx_hybrid_transformer,
)
from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as model_module
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_MAX_PATH_BARS,
    UNIFIED_EXIT_PATH_FEATURE_DIM,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT,
)
from gx1.features.htf_features import (
    MULTI_TF_PER_BAR_FEATURES_V4,
    MULTI_TF_V4_GROUP_A_BASE_FEATURES,
)
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


def _specialist_indices(width: int = SEQ_DIM) -> dict[str, list[int]]:
    grouped = {name: [] for name in EXACT_SPECIALIST_NAMES}
    for index in range(width):
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
    return _build_unit_test_entry_v10_ctx_hybrid_transformer(**kwargs)


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
            for tf in (
                timeframe.lower() for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES
            )
        },
    )


def _forward(model: EntryV10CtxHybridTransformer, batch_size: int = 2) -> dict:
    seq_x, snap_x, ctx_cat, ctx_cont, mtf = _make_inputs(batch_size)
    return model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)


def _make_exit_episode_inputs(
    *,
    state_count: int = UNIFIED_EXIT_MAX_PATH_BARS,
    batch_size: int = 1,
) -> dict[str, object]:
    warm_rows = EXIT_FEATURE_SEQUENCE_BARS - 1
    _seq, _snap, ctx_cat, ctx_cont, _mtf = _make_inputs(batch_size)
    history_rows = warm_rows + state_count
    mtf_history_rows = SEQ_LEN + state_count - 1
    return {
        "entry_decision_representation": torch.randn(batch_size, 128),
        "exit_local_history_x": torch.randn(
            batch_size, history_rows, SEQ_DIM
        ),
        "exit_state_ctx_cat": ctx_cat[:, None, :].expand(
            -1, state_count, -1
        ).clone(),
        "exit_state_ctx_cont": ctx_cont[:, None, :].expand(
            -1, state_count, -1
        ).clone(),
        "exit_path_x": torch.randn(
            batch_size,
            2,
            state_count,
            UNIFIED_EXIT_PATH_FEATURE_DIM,
        ),
        "exit_mtf_histories": {
            tf.lower(): torch.randn(
                batch_size, mtf_history_rows, TF_DIM
            )
            for tf in EXIT_MTF_CONTEXT_TIMEFRAMES
        },
        "exit_mtf_gathers": {
            tf.lower(): torch.arange(
                SEQ_LEN - 1, mtf_history_rows, dtype=torch.long
            )
            .view(1, -1)
            .expand(batch_size, -1)
            .contiguous()
            for tf in EXIT_MTF_CONTEXT_TIMEFRAMES
        },
        "exit_mtf_history_lengths": {
            tf.lower(): torch.full(
                (batch_size,), mtf_history_rows, dtype=torch.long
            )
            for tf in EXIT_MTF_CONTEXT_TIMEFRAMES
        },
    }


def test_exact_architecture_emits_every_mandatory_head_with_exact_width() -> None:
    model = _make_model().eval()
    out = _forward(model, batch_size=2)
    widths = {
        "entry_action_q_bps": 3,
        "entry_q_joint_hidden": 128,
        "specialist_gate": 8,
        "tf_gate": ENTRY_MTF_CONTEXT_COUNT,
        "family_tf_cooperation_gate": (
            ENTRY_MTF_CONTEXT_COUNT * len(EXACT_SPECIALIST_NAMES)
        ),
        "side_mae_bps": 2,
        "trendline_event_logits": 4,
        "position_size_logit": 1,
        "dip_pred": 18,
        "forecast_pred": 4,
        "timing_pred": 12,
        "tail_risk_pred": 6,
        "vol_forecast_pred": 3,
        "entry_decision_representation": 128,
        "entry_decision_token_source": 643,
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
    assert out["family_tf_feature_gate"].shape == (
        2,
        ENTRY_MTF_CONTEXT_COUNT,
        TF_DIM,
    )
    assert torch.isfinite(out["family_tf_feature_gate"]).all()


def test_context_inputs_materially_change_entry_action_q() -> None:
    torch.manual_seed(1337)
    model = _make_model(dropout=0.0).eval()
    seq_x, snap_x, ctx_cat, ctx_cont, mtf = _make_inputs(batch_size=1)
    changed_ctx_cat = torch.stack(
        [
            (ctx_cat[:, index] + 1) % len(domain)
            for index, domain in enumerate(EXACT_CTX_CAT_DOMAINS.values())
        ],
        dim=1,
    )
    changed_ctx_cont = torch.zeros_like(ctx_cont)

    with torch.no_grad():
        baseline = model(
            seq_x,
            snap_x,
            ctx_cat=ctx_cat,
            ctx_cont=ctx_cont,
            **mtf,
        )["entry_action_q_bps"]
        changed = model(
            seq_x,
            snap_x,
            ctx_cat=changed_ctx_cat,
            ctx_cont=changed_ctx_cont,
            **mtf,
        )["entry_action_q_bps"]

    assert torch.max(torch.abs(baseline - changed)).item() > 1e-6


def test_each_declared_entry_decision_component_block_moves_token() -> None:
    from gx1.contracts.entry_decision_token_v1 import (
        ENTRY_DECISION_TOKEN_COMPONENTS,
    )

    torch.manual_seed(916)
    model = _make_model(dropout=0.0).eval()
    with torch.no_grad():
        forward = _forward(model, batch_size=2)
        source = forward["entry_decision_token_source"]
        components = {}
        start = 0
        for name, width in ENTRY_DECISION_TOKEN_COMPONENTS:
            components[name] = source[:, start : start + width]
            start += width
        baseline = model._project_entry_decision_token(components)
        assert torch.equal(
            baseline,
            forward["entry_decision_representation"],
        )
        for name, _width in ENTRY_DECISION_TOKEN_COMPONENTS:
            # Replace exactly one block with the same block emitted for another
            # genuine forward row.  Both source values are therefore on the
            # model's own evidence manifold; no synthetic score scale is used.
            perturbed = {
                key: value.clone() for key, value in components.items()
            }
            perturbed[name][0:1] = components[name][1:2]
            assert not torch.equal(
                perturbed[name][0:1], components[name][0:1]
            ), name
            changed = model._project_entry_decision_token(perturbed)
            assert not torch.equal(changed[0:1], baseline[0:1]), name
            assert (
                torch.max(torch.abs(changed[0:1] - baseline[0:1])).item()
                > 0.0
            ), name


def test_cold_start_training_preserves_token_influence_on_exit_margin() -> None:
    from gx1.contracts.entry_decision_token_v1 import (
        ENTRY_DECISION_TOKEN_COMPONENTS,
    )

    torch.manual_seed(917)
    model = _make_model(dropout=0.0).train()
    components = {
        name: torch.randn(2, width)
        for name, width in ENTRY_DECISION_TOKEN_COMPONENTS
    }
    components["local_model_native_representation"][1] = (
        components["local_model_native_representation"][0] + 0.5
    )
    tokens = model._project_entry_decision_token(components)
    exit_inputs = _make_exit_episode_inputs(state_count=3, batch_size=2)
    exit_inputs["entry_decision_representation"] = tokens
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    optimizer.zero_grad(set_to_none=True)
    # A 3-state prefix is the incremental owner's domain; `forward_exit_episode`
    # now requires the complete 512-state pack.  Both routes share the exact
    # same weights and causal scan (proved in
    # test_exit_episode_one_pass_prefix_and_future_append_parity_all_states).
    trained = model.forward_exit_incremental_prefix(**exit_inputs)
    target = torch.zeros_like(trained["exit_action_q_bps"])
    target[0, :, :, 0] = 1.0
    target[1, :, :, 1] = 1.0
    torch.nn.functional.mse_loss(
        trained["exit_action_q_bps"], target
    ).backward()
    assert model.entry_decision_token[1].weight.grad is not None
    assert model.entry_decision_token[1].weight.grad.abs().sum().item() > 0.0
    optimizer.step()

    model.eval()
    with torch.no_grad():
        post = model.forward_exit_incremental_prefix(**exit_inputs)[
            "exit_action_q_bps"
        ]
    margins = post[:, 0, 0, 1] - post[:, 0, 0, 0]
    assert not torch.equal(margins[0], margins[1])



def test_entry_q_gradient_reaches_joint_local_mtf_and_family_representations() -> None:
    torch.manual_seed(1307)
    model = _make_model(dropout=0.0).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    target = torch.tensor(
        [[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = _forward(model, batch_size=3)
        torch.nn.functional.mse_loss(
            output["entry_action_q_bps"], target
        ).backward()
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    output = _forward(model, batch_size=3)
    torch.nn.functional.mse_loss(
        output["entry_action_q_bps"], target
    ).backward()
    for parameter in (
        model.seq_proj.weight,
        model.specialist_out.weight,
        model.cross_tf_out.weight,
        model.family_tf_cooperation_out.weight,
        model.entry_q_joint_in.weight,
        model.head_entry_action_q.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0.0


def test_all_eight_family_routes_reach_joint_entry_and_exit_q_losses() -> None:
    """Prove every declared family is trainable, not merely serialized."""

    torch.manual_seed(1308)
    model = _make_model(dropout=0.0).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    entry_target = torch.tensor(
        [[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )

    # The two neutral zero-initialized fusion outputs intentionally protect a
    # cold model from a hand-authored family/timeframe preference. Warm only
    # through the sole Entry-Q authority before testing every family route;
    # this is a reachability test, not evidence of market utility.
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        entry = _forward(model, batch_size=3)
        torch.nn.functional.mse_loss(
            entry["entry_action_q_bps"], entry_target
        ).backward()
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    entry = _forward(model, batch_size=3)
    exit_output = model.forward_exit_incremental_prefix(
        **_make_exit_episode_inputs(state_count=3, batch_size=3)
    )
    (
        torch.nn.functional.mse_loss(entry["entry_action_q_bps"], entry_target)
        + exit_output["exit_action_q_bps"].square().mean()
    ).backward()

    for name in EXACT_SPECIALIST_NAMES:
        for parameter in (
            model.specialist_proj[name].weight,
            model.mtf_family_proj[name].weight,
            model.exit_episode_family_gru[name].weight_ih_l0,
            model.exit_episode_mtf_family_gru[name].weight_ih_l0,
        ):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum().item() > 0.0, name


def test_position_size_head_has_no_entry_q_decision_authority() -> None:
    model = _make_model().train()
    output = _forward(model, batch_size=4)
    torch.sigmoid(output["position_size_logit"]).mean().backward()
    assert model.head_position_size.weight.grad is not None
    assert model.head_position_size.weight.grad.abs().sum().item() > 0.0
    assert model.head_entry_action_q.weight.grad is None
    assert model.entry_q_joint_in.weight.grad is None



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


@pytest.mark.parametrize("missing_timeframe", ENTRY_MTF_CONTEXT_TIMEFRAMES)
def test_exact_entry_architecture_requires_all_four_tf_inputs(
    missing_timeframe: str,
) -> None:
    model = _make_model().eval()
    seq_x, snap_x, ctx_cat, ctx_cont, mtf = _make_inputs()
    missing_key = f"seq_{missing_timeframe.lower()}"
    del mtf[missing_key]
    with pytest.raises(TypeError, match=missing_key):
        model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)


def test_exact_production_entry_shape_uses_raw_q_authority() -> None:
    tf_names = list(MULTI_TF_PER_BAR_FEATURES_V4)
    tf_width = len(tf_names)
    tf_lengths = {"M5": 16, "M15": 64, "H1": 96, "H4": 96, "D1": 252}
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=MODEL_NATIVE_SIGNAL_DIM,
        snap_input_dim=MODEL_NATIVE_SIGNAL_DIM,
        seq_len=MODEL_NATIVE_SEQ_LEN,
        dropout=0.0,
        multi_tf_num_layers=1,
        multi_tf_scale=0.5,
        specialist_num_layers=1,
        specialist_fusion_scale=0.25,
        cross_family_fusion_scale=0.25,
        ctx_cont_dim=MODEL_NATIVE_CTX_CONT_DIM,
        ctx_cat_dim=MODEL_NATIVE_CTX_CAT_DIM,
        m5_seq_dim=tf_width,
        m15_seq_dim=tf_width,
        h1_seq_dim=tf_width,
        h4_seq_dim=tf_width,
        d1_seq_dim=tf_width,
        m5_seq_len=tf_lengths["M5"],
        m15_seq_len=tf_lengths["M15"],
        h1_seq_len=tf_lengths["H1"],
        h4_seq_len=tf_lengths["H4"],
        d1_seq_len=tf_lengths["D1"],
        specialist_input_indices=_specialist_indices(MODEL_NATIVE_SIGNAL_DIM),
        specialist_ctx_cont_indices={
            str(name): list(values)
            for name, values in MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT[
                "ctx_cont_indices"
            ].items()
        },
        specialist_ctx_cont_nominal_indices={
            str(name): list(values)
            for name, values in MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT[
                "ctx_cont_nominal_indices"
            ].items()
        },
        specialist_ctx_cat_indices={
            str(name): list(values)
            for name, values in MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT[
                "ctx_cat_indices"
            ].items()
        },
        multi_tf_specialist_input_indices=_multi_tf_specialist_indices(tf_width),
        temporal_alias_signal_indices=[],
        temporal_alias_ctx_cont_indices=[],
        input_normalization=input_normalization_fixture(
            signal_names=[
                f"production_signal_{index}"
                for index in range(MODEL_NATIVE_SIGNAL_DIM)
            ],
            mtf_names=tf_names,
            per_tf_seq_lens=tf_lengths,
        ),
    ).eval()
    ctx_cont = torch.randn(1, MODEL_NATIVE_CTX_CONT_DIM)
    nominal_indices = [
        index
        for values in MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT[
            "ctx_cont_nominal_indices"
        ].values()
        for index in values
    ]
    ctx_cont[:, nominal_indices] = 0.0
    ctx_cat = torch.zeros(1, MODEL_NATIVE_CTX_CAT_DIM, dtype=torch.long)
    entry_seq = torch.randn(1, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM)
    entry_mtf = {
        f"seq_{tf.lower()}": torch.zeros(1, tf_lengths[tf], tf_width)
        for tf in ENTRY_MTF_CONTEXT_TIMEFRAMES
    }
    with torch.no_grad():
        entry = model(
            entry_seq,
            entry_seq[:, -1, :],
            ctx_cat=ctx_cat,
            ctx_cont=ctx_cont,
            **entry_mtf,
        )
    assert entry["entry_action_q_bps"].shape == (1, 3)
    assert entry["tf_gate"].shape == (1, ENTRY_MTF_CONTEXT_COUNT)


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


def test_host_only_routing_caches_exactly_mirror_contract_owned_indices() -> None:
    """CUDA-avoidance caches may route, but can never invent ownership."""

    model = _make_model()
    for surface in model._input_normalization_contract["surfaces"]:
        expected = tuple(
            torch.nonzero(
                getattr(model, f"input_norm_{surface}_categorical_mask"),
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )
        assert model._input_norm_categorical_index_tuples[surface] == expected
    for name in EXACT_SPECIALIST_NAMES:
        assert model._specialist_input_index_sets[name] == frozenset(
            getattr(model, f"specialist_idx_{name}").tolist()
        )
        assert model._multi_tf_specialist_index_tuples[name] == tuple(
            getattr(model, f"multi_tf_specialist_idx_{name}").tolist()
        )
        assert model._specialist_ctx_cont_nominal_index_tuples[name] == tuple(
            getattr(model, f"specialist_ctx_cont_nominal_idx_{name}").tolist()
        )
        expected_mtf_nominal_positions = tuple(
            (local_position, global_index)
            for local_position, global_index in enumerate(
                model._multi_tf_specialist_index_tuples[name]
            )
            if global_index in model._multi_tf_categorical_index_set
        )
        assert (
            model._multi_tf_specialist_categorical_positions[name]
            == expected_mtf_nominal_positions
        )
    assert model._generic_snap_index_set == frozenset(
        model.generic_snap_idx.tolist()
    )

    model._input_norm_categorical_index_tuples = {
        **model._input_norm_categorical_index_tuples,
        "signal": (0,),
    }
    with pytest.raises(RuntimeError, match="CATEGORICAL_INDEX_CACHE_MISMATCH"):
        model.require_input_normalization_state()


def test_hot_forward_owners_do_not_materialize_device_index_buffers() -> None:
    """Keep Exit and Entry routes free of CUDA ``tolist`` synchronizations."""

    for owner in (
        EntryV10CtxHybridTransformer._normalize_input_surface,
        EntryV10CtxHybridTransformer._build_family_context_tokens,
        EntryV10CtxHybridTransformer._encode_multi_tf_route,
        EntryV10CtxHybridTransformer._forward_exit_causal_episode,
        EntryV10CtxHybridTransformer.forward_exit_incremental_step,
    ):
        assert ".tolist()" not in inspect.getsource(owner)


def test_input_normalization_applies_identical_non_saturating_asinh_in_train_and_eval() -> None:
    model = _make_model()
    center = model.input_norm_signal_center
    scale = model.input_norm_signal_scale
    raw = center.clone().view(1, 1, -1).repeat(2, 1, 1)
    raw[0, ..., 0] = center[0] + scale[0] * 13.0
    raw[1, ..., 0] = center[0] + scale[0] * 130.0

    model.eval()
    normalized_eval = model._normalize_input_surface(raw, surface="signal")
    torch.testing.assert_close(
        normalized_eval[:, 0, 0],
        torch.asinh(torch.tensor([13.0, 130.0])),
    )
    assert normalized_eval[1, 0, 0] > normalized_eval[0, 0, 0]

    model.train()
    normalized_train = model._normalize_input_surface(raw, surface="signal")
    assert torch.equal(normalized_eval, normalized_train)


def test_entry_exit_and_incremental_replay_share_one_normalization_owner() -> None:
    shared_entry_exit = inspect.getsource(
        EntryV10CtxHybridTransformer._encode_shared_feature_base
    )
    exit_episode = inspect.getsource(
        EntryV10CtxHybridTransformer._forward_exit_causal_episode
    )
    exit_incremental = inspect.getsource(
        EntryV10CtxHybridTransformer.forward_exit_incremental_step
    )

    assert shared_entry_exit.count("self._normalize_input_surface(") == 3
    assert 'surface="signal"' in shared_entry_exit
    assert 'surface="ctx_cont"' in shared_entry_exit
    assert 'surface="signal"' in exit_episode
    assert 'surface="ctx_cont"' in exit_episode
    assert 'surface="signal"' in exit_incremental
    assert 'surface="ctx_cont"' in exit_incremental


@pytest.mark.parametrize(
    ("field", "invalid_value", "error"),
    (
        (
            "ema_stack_aligned_v2",
            2.0,
            "MTF_EMA_STACK_DOMAIN_INVALID",
        ),
        # V30 (2026-08-14): `regime_class_id` was the only MTF semantic
        # categorical; it is retired and MTF_SEMANTIC_CATEGORICAL_DOMAINS is
        # empty, so the EMA-stack ternary is the remaining domain guard.
    ),
)
def test_mtf_semantic_domains_fail_closed_at_model_boundary(
    field: str,
    invalid_value: float,
    error: str,
) -> None:
    mtf_names = list(MULTI_TF_V4_GROUP_A_BASE_FEATURES)
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


def test_exit_episode_one_pass_prefix_and_future_append_parity_all_states() -> None:
    torch.manual_seed(20260814)
    model = _make_model(dropout=0.0).eval()
    inputs = _make_exit_episode_inputs()
    with torch.no_grad():
        episode = model.forward_exit_episode(**inputs)
        same_owner = model.forward_exit_incremental_prefix(**inputs)
    assert episode["exit_action_q_bps"].shape == (
        1,
        2,
        UNIFIED_EXIT_MAX_PATH_BARS,
        2,
    )
    assert torch.equal(
        episode["exit_action_q_bps"], same_owner["exit_action_q_bps"]
    )
    assert episode["exit_episode_lengths"].tolist() == [
        [UNIFIED_EXIT_MAX_PATH_BARS, UNIFIED_EXIT_MAX_PATH_BARS]
    ]
    assert episode["exit_terminal_mask"][:, :, :-1].sum().item() == 0
    assert episode["exit_terminal_mask"][:, :, -1].all()
    assert episode["exit_terminal_reason_index"][:, :, -1].eq(1).all()

    # The shorter online prefix sees exactly the same causal owner state at
    # every included row.  Future state/path/MTF appends cannot alter it.
    for boundary in (1, 127, 128, 129, UNIFIED_EXIT_MAX_PATH_BARS - 1):
        prefix_inputs = {
            **inputs,
            "exit_local_history_x": inputs["exit_local_history_x"][
                :, : EXIT_FEATURE_SEQUENCE_BARS - 1 + boundary
            ],
            "exit_state_ctx_cat": inputs["exit_state_ctx_cat"][:, :boundary],
            "exit_state_ctx_cont": inputs["exit_state_ctx_cont"][:, :boundary],
            "exit_path_x": inputs["exit_path_x"][:, :, :boundary],
            "exit_mtf_histories": {
                tf: values[:, : SEQ_LEN + boundary - 1]
                for tf, values in inputs["exit_mtf_histories"].items()
            },
            "exit_mtf_gathers": {
                tf: values[:, :boundary]
                for tf, values in inputs["exit_mtf_gathers"].items()
            },
            "exit_mtf_history_lengths": {
                tf: torch.full((1,), SEQ_LEN + boundary - 1, dtype=torch.long)
                for tf in inputs["exit_mtf_history_lengths"]
            },
        }
        with torch.no_grad():
            prefix = model.forward_exit_incremental_prefix(**prefix_inputs)
        assert torch.allclose(
            prefix["exit_action_q_bps"],
            episode["exit_action_q_bps"][:, :, :boundary],
            rtol=1e-6,
            atol=1e-6,
        )
        assert prefix["exit_episode_lengths"].tolist() == [
            [boundary, boundary]
        ]
        assert not prefix["exit_terminal_mask"].any()


def test_exit_incremental_hidden_carry_matches_all_512_offline_states() -> None:
    torch.manual_seed(20260815)
    model = _make_model(dropout=0.0).eval()
    inputs = _make_exit_episode_inputs()
    # MTF closes only every fifth M1 state. Repeated gathers therefore exercise
    # the zero-new-row carry path, while the close boundary supplies one row.
    gathers = torch.div(
        torch.arange(UNIFIED_EXIT_MAX_PATH_BARS), 5, rounding_mode="floor"
    ) + (SEQ_LEN - 1)
    mtf_rows = int(gathers[-1].item()) + 1
    inputs["exit_mtf_histories"] = {
        tf: values[:, :mtf_rows]
        for tf, values in inputs["exit_mtf_histories"].items()
    }
    inputs["exit_mtf_gathers"] = {
        tf: gathers.view(1, -1).clone()
        for tf in inputs["exit_mtf_gathers"]
    }
    inputs["exit_mtf_history_lengths"] = {
        tf: torch.full((1,), mtf_rows, dtype=torch.long)
        for tf in inputs["exit_mtf_history_lengths"]
    }
    with torch.no_grad():
        offline = model.forward_exit_episode(**inputs)["exit_action_q_bps"]
        carry = None
        pieces = []
        prior_gather = {tf: -1 for tf in inputs["exit_mtf_gathers"]}
        for state in range(UNIFIED_EXIT_MAX_PATH_BARS):
            new_mtf = {}
            for tf, history in inputs["exit_mtf_histories"].items():
                current = int(inputs["exit_mtf_gathers"][tf][0, state])
                start = 0 if state == 0 else prior_gather[tf] + 1
                new_mtf[tf] = history[:, start : current + 1]
                prior_gather[tf] = current
            local_start = 0 if state == 0 else EXIT_FEATURE_SEQUENCE_BARS - 1 + state
            step, carry = model.forward_exit_incremental_step(
                entry_decision_representation=inputs[
                    "entry_decision_representation"
                ],
                exit_local_rows_x=inputs["exit_local_history_x"][:, local_start : EXIT_FEATURE_SEQUENCE_BARS + state],
                exit_state_ctx_cat=inputs["exit_state_ctx_cat"][:, state],
                exit_state_ctx_cont=inputs["exit_state_ctx_cont"][:, state],
                exit_path_row_x=inputs["exit_path_x"][:, :, state],
                exit_mtf_new_rows=new_mtf,
                carry=carry,
            )
            assert carry.step_count == state + 1
            pieces.append(step["exit_action_q_bps"])
        online = torch.cat(pieces, dim=2)
    assert online.shape == offline.shape
    assert torch.allclose(online, offline, rtol=1e-5, atol=1e-5)


def test_exit_incremental_persisted_carry_restart_matches_uninterrupted_step() -> None:
    torch.manual_seed(20260817)
    model = _make_model(dropout=0.0).eval()
    inputs = _make_exit_episode_inputs()
    first_mtf = {
        tf: history[:, :SEQ_LEN]
        for tf, history in inputs["exit_mtf_histories"].items()
    }
    with torch.no_grad():
        _, first_carry = model.forward_exit_incremental_step(
            entry_decision_representation=inputs[
                "entry_decision_representation"
            ],
            exit_local_rows_x=inputs["exit_local_history_x"][
                :, :EXIT_FEATURE_SEQUENCE_BARS
            ],
            exit_state_ctx_cat=inputs["exit_state_ctx_cat"][:, 0],
            exit_state_ctx_cont=inputs["exit_state_ctx_cont"][:, 0],
            exit_path_row_x=inputs["exit_path_x"][:, :, 0],
            exit_mtf_new_rows=first_mtf,
            carry=None,
        )
    envelope = build_unified_exit_incremental_carry_envelope(
        tensor_state=model.export_exit_incremental_carry_tensor_state(
            first_carry
        ),
        step_count=1,
        last_closed_m1_bar_ts="2026-01-01T00:00:00Z",
        trade_identity="trade-restart",
        side="long",
        bundle_sha256="1" * 64,
        input_normalization_sha256="2" * 64,
        entry_token_snapshot_sha256="3" * 64,
        full_path_chain_sha256="4" * 64,
        input_envelope_sha256="5" * 64,
        previous_carry_envelope_sha256=(
            UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256
        ),
        mtf_last_row_sha256={tf: "6" * 64 for tf in first_mtf},
    )
    restored = model.restore_exit_incremental_carry_tensor_state(
        step_count=1,
        batch_size=1,
        tensors=decode_unified_exit_incremental_carry_tensors(
            envelope, device=torch.device("cpu")
        ),
    )
    second_mtf = {
        tf: history[:, SEQ_LEN : SEQ_LEN + 1]
        for tf, history in inputs["exit_mtf_histories"].items()
    }
    kwargs = {
        "entry_decision_representation": inputs[
            "entry_decision_representation"
        ],
        "exit_local_rows_x": inputs["exit_local_history_x"][
            :, EXIT_FEATURE_SEQUENCE_BARS : EXIT_FEATURE_SEQUENCE_BARS + 1
        ],
        "exit_state_ctx_cat": inputs["exit_state_ctx_cat"][:, 1],
        "exit_state_ctx_cont": inputs["exit_state_ctx_cont"][:, 1],
        "exit_path_row_x": inputs["exit_path_x"][:, :, 1],
        "exit_mtf_new_rows": second_mtf,
    }
    with torch.no_grad():
        uninterrupted, _ = model.forward_exit_incremental_step(
            **kwargs, carry=first_carry
        )
        restarted, _ = model.forward_exit_incremental_step(
            **kwargs, carry=restored
        )
    assert torch.equal(
        uninterrupted["exit_action_q_bps"], restarted["exit_action_q_bps"]
    )


def test_exit_episode_feature_tf_gate_is_genuine_per_field_not_family_broadcast() -> None:
    model = _make_model(dropout=0.0).eval()
    with torch.no_grad():
        for position, gate in enumerate(model.mtf_feature_context_gate.values()):
            gate.bias.copy_(
                torch.linspace(-1.0, 1.0, gate.out_features)
                + position / 100.0
            )
    inputs = _make_exit_episode_inputs(state_count=3)
    with torch.no_grad():
        output = model.forward_exit_incremental_prefix(**inputs)
    feature_gate = output["exit_family_tf_feature_gate"]
    assert feature_gate.shape == (1, 3, 5, TF_DIM)
    for family_name in EXACT_SPECIALIST_NAMES:
        indices = getattr(model, f"multi_tf_specialist_idx_{family_name}")
        if int(indices.numel()) > 1:
            owned = feature_gate[0, :, :, indices]
            assert not torch.equal(
                owned[..., 0], owned[..., 1]
            ), family_name
    source = inspect.getsource(
        EntryV10CtxHybridTransformer._forward_exit_causal_episode
    )
    assert "mtf_feature_context_gate" in source
    assert "current_owned_numeric" in source
    assert "= cooperation_gate[..., family_position]" not in source


def test_exit_episode_batch_right_padding_matches_individually_trimmed_histories() -> None:
    torch.manual_seed(20260816)
    model = _make_model(dropout=0.0).eval()
    left = _make_exit_episode_inputs(state_count=3)
    right = _make_exit_episode_inputs(state_count=3)
    batched = {
        name: torch.cat((left[name], right[name]), dim=0)
        for name in (
            "entry_decision_representation",
            "exit_local_history_x",
            "exit_state_ctx_cat",
            "exit_state_ctx_cont",
            "exit_path_x",
        )
    }
    batched["exit_mtf_histories"] = {}
    batched["exit_mtf_gathers"] = {}
    batched["exit_mtf_history_lengths"] = {}
    for tf in left["exit_mtf_histories"]:
        full_right = right["exit_mtf_histories"][tf]
        left_trimmed = left["exit_mtf_histories"][tf][:, :SEQ_LEN]
        left_padded = torch.cat(
            (left_trimmed, torch.zeros_like(full_right[:, SEQ_LEN:])), dim=1
        )
        batched["exit_mtf_histories"][tf] = torch.cat(
            (left_padded, full_right), dim=0
        )
        batched["exit_mtf_gathers"][tf] = torch.cat(
            (
                torch.full((1, 3), SEQ_LEN - 1, dtype=torch.long),
                right["exit_mtf_gathers"][tf],
            ),
            dim=0,
        )
        batched["exit_mtf_history_lengths"][tf] = torch.tensor(
            [SEQ_LEN, SEQ_LEN + 2], dtype=torch.long
        )

    left["exit_mtf_histories"] = {
        tf: values[:, :SEQ_LEN]
        for tf, values in left["exit_mtf_histories"].items()
    }
    left["exit_mtf_gathers"] = {
        tf: torch.full((1, 3), SEQ_LEN - 1, dtype=torch.long)
        for tf in left["exit_mtf_gathers"]
    }
    left["exit_mtf_history_lengths"] = {
        tf: torch.tensor([SEQ_LEN], dtype=torch.long)
        for tf in left["exit_mtf_history_lengths"]
    }
    with torch.no_grad():
        combined = model.forward_exit_incremental_prefix(**batched)[
            "exit_action_q_bps"
        ]
        left_q = model.forward_exit_incremental_prefix(**left)[
            "exit_action_q_bps"
        ]
        right_q = model.forward_exit_incremental_prefix(**right)[
            "exit_action_q_bps"
        ]
    assert torch.allclose(combined[:1], left_q, rtol=1e-5, atol=1e-5)
    assert torch.allclose(combined[1:], right_q, rtol=1e-5, atol=1e-5)


def test_exit_token_axis_transport_chunk_preserves_outputs_and_gradients() -> None:
    torch.manual_seed(20260814)
    layer = torch.nn.TransformerEncoderLayer(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        batch_first=True,
        norm_first=False,
    )
    encoder = torch.nn.TransformerEncoder(layer, num_layers=1).eval()
    full_input = torch.randn(7, 5, 8, requires_grad=True)
    full = model_module._apply_exit_token_axis_encoder(
        encoder,
        full_input,
        row_chunk_size=64,
    )
    full.square().sum().backward()
    full_input_gradient = full_input.grad.detach().clone()
    parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in encoder.named_parameters()
    }

    encoder.zero_grad(set_to_none=True)
    chunked_input = full_input.detach().clone().requires_grad_(True)
    chunked = model_module._apply_exit_token_axis_encoder(
        encoder,
        chunked_input,
        row_chunk_size=3,
    )
    chunked.square().sum().backward()

    assert torch.allclose(chunked, full, atol=1e-6, rtol=0.0)
    assert torch.allclose(
        chunked_input.grad,
        full_input_gradient,
        atol=1e-6,
        rtol=0.0,
    )
    for name, parameter in encoder.named_parameters():
        assert torch.allclose(
            parameter.grad,
            parameter_gradients[name],
            atol=1e-5,
            rtol=0.0,
        )
