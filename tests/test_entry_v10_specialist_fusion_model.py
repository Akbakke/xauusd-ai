import json
from pathlib import Path

import pytest
import torch

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_MTF_CONTEXT_COUNT,
    ENTRY_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT,
    MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT,
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EntryV10CtxHybridTransformer,
    _build_unit_test_entry_v10_ctx_hybrid_transformer,
)
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import _load_specialist_fusion_contract
from tests.model_native_context_routing_support import (
    context_routing_for_ordered_signal_names,
    ordered_signal_names_for_specialist_indices,
)
from tests.model_native_input_normalization_support import (
    input_normalization_fixture,
)

MTF_DIM = len(MODEL_NATIVE_TRAINING_SPECIALISTS)


def _specialist_indices(
    signal_dim: int = MODEL_NATIVE_SIGNAL_DIM,
) -> dict[str, list[int]]:
    grouped = {name: [] for name in MODEL_NATIVE_TRAINING_SPECIALISTS}
    for index in range(signal_dim):
        grouped[MODEL_NATIVE_TRAINING_SPECIALISTS[index % len(grouped)]].append(index)
    return grouped


def _multi_tf_specialist_indices() -> dict[str, list[int]]:
    return {
        name: [position]
        for position, name in enumerate(MODEL_NATIVE_TRAINING_SPECIALISTS)
    }


def _context_indices() -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    routing = MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT
    return (
        {str(k): list(v) for k, v in routing["ctx_cont_indices"].items()},
        {str(k): list(v) for k, v in routing["ctx_cat_indices"].items()},
    )


def _valid_ctx_cont(batch_size: int) -> torch.Tensor:
    values = torch.randn(batch_size, MODEL_NATIVE_CTX_CONT_DIM)
    nominal_indices = [
        index
        for indices in MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT[
            "ctx_cont_nominal_indices"
        ].values()
        for index in indices
    ]
    values[:, nominal_indices] = torch.randint(
        0,
        5,
        (batch_size, len(nominal_indices)),
    ).float()
    return values


def _synchronized_seq_snap(
    seq_x: torch.Tensor,
    ctx_cont: torch.Tensor,
    context_routing: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_x = seq_x.clone()
    snap_x = seq_x[:, -1, :].clone()
    if context_routing is not None:
        policy = context_routing["temporal_alias_policy"]
        nominal_ctx_indices = {
            index
            for indices in context_routing[
                "ctx_cont_nominal_indices"
            ].values()
            for index in indices
        }
        for signal_index, ctx_index in zip(
            policy["signal_indices"],
            policy["ctx_cont_indices"],
        ):
            if ctx_index in nominal_ctx_indices:
                seq_x[:, :, signal_index] = 0.0
        snap_x[:, policy["signal_indices"]] = ctx_cont[
            :,
            policy["ctx_cont_indices"],
        ]
        seq_x[:, -1, :] = snap_x
    return seq_x, snap_x


def _guard_test_model(context_routing: dict) -> EntryV10CtxHybridTransformer:
    return _build_unit_test_entry_v10_ctx_hybrid_transformer(
        seq_input_dim=MODEL_NATIVE_SIGNAL_DIM,
        snap_input_dim=MODEL_NATIVE_SIGNAL_DIM,
        seq_len=4,
        dropout=0.05,
        multi_tf_num_layers=1,
        multi_tf_scale=0.5,
        specialist_num_layers=1,
        specialist_fusion_scale=0.25,
        cross_family_fusion_scale=0.25,
        ctx_cont_dim=MODEL_NATIVE_CTX_CONT_DIM,
        ctx_cat_dim=MODEL_NATIVE_CTX_CAT_DIM,
        m5_seq_dim=MTF_DIM,
        m15_seq_dim=MTF_DIM,
        h1_seq_dim=MTF_DIM,
        h4_seq_dim=MTF_DIM,
        d1_seq_dim=MTF_DIM,
        m5_seq_len=4,
        m15_seq_len=4,
        h1_seq_len=4,
        h4_seq_len=4,
        d1_seq_len=4,
        specialist_input_indices=_specialist_indices(),
        specialist_ctx_cont_indices=context_routing["ctx_cont_indices"],
        specialist_ctx_cont_nominal_indices=context_routing[
            "ctx_cont_nominal_indices"
        ],
        specialist_ctx_cat_indices=context_routing["ctx_cat_indices"],
        multi_tf_specialist_input_indices=_multi_tf_specialist_indices(),
        temporal_alias_signal_indices=context_routing[
            "temporal_alias_policy"
        ]["signal_indices"],
        temporal_alias_ctx_cont_indices=context_routing[
            "temporal_alias_policy"
        ]["ctx_cont_indices"],
        input_normalization=_input_normalization(_specialist_indices()),
    ).eval()


def test_m5_family_token_has_no_hard_coded_duplicate_current_bar_merge() -> None:
    model = _guard_test_model(_context_routing(_specialist_indices()))

    assert not hasattr(model, "mtf_m5_family_merge_norm")


def _context_routing(
    specialist_indices: dict[str, list[int]],
) -> dict:
    return context_routing_for_ordered_signal_names(
        ordered_signal_names_for_specialist_indices(specialist_indices)
    )


def _ordered_signal_names() -> list[str]:
    return ordered_signal_names_for_specialist_indices(_specialist_indices())


def _input_normalization(
    specialist_indices: dict[str, list[int]],
) -> dict:
    signal_width = sum(len(values) for values in specialist_indices.values())
    if signal_width == MODEL_NATIVE_SIGNAL_DIM:
        signal_names = ordered_signal_names_for_specialist_indices(
            specialist_indices
        )
    else:
        signal_names = ordered_signal_names_for_specialist_indices(
            specialist_indices,
            temporal_alias_signal_fields=(),
        )
    return input_normalization_fixture(
        signal_names=signal_names,
        mtf_names=[f"mtf_{index}" for index in range(MTF_DIM)],
    )


def _audit_payload() -> dict:
    specialist_indices = _specialist_indices()
    return {
        "decision": "PASS",
        "created_utc": "2026-07-16T00:00:00+00:00",
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "signal_field_count": MODEL_NATIVE_SIGNAL_DIM,
        "selected_feature_count": 479,
        "specialist_model_contract": json.loads(
            json.dumps(MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT)
        ),
        "specialist_model_contract_valid": True,
        "specialist_model_contract_failures": [],
        "context_specialist_routing_all_mapped": True,
        "context_specialist_routing_failure_count": 0,
        "context_specialist_routing_failures": [],
        "architecture_contract": {
            "specialist_input_indices": specialist_indices,
            "context_specialist_routing": _context_routing(
                specialist_indices
            ),
            "recommended_fusion": {
                "active_heads": list(SPECIALIST_FUSION_ACTIVE_HEADS),
                "blocked_heads": list(SPECIALIST_FUSION_BLOCKED_HEADS),
            },
        },
    }


def _write_audit(tmp_path: Path, payload: dict | None = None) -> Path:
    path = tmp_path / "ENTRY_MODEL_NATIVE_SPECIALIST_AUDIT_20260716T000000Z.json"
    path.write_text(json.dumps(payload or _audit_payload()), encoding="utf-8")
    return path


def test_entry_v10_exact_model_always_has_specialist_state_and_output() -> None:
    ctx_cont_indices, ctx_cat_indices = _context_indices()
    model = _build_unit_test_entry_v10_ctx_hybrid_transformer(
        seq_input_dim=16,
        snap_input_dim=16,
        seq_len=16,
        dropout=0.05,
        multi_tf_num_layers=1,
        multi_tf_scale=0.5,
        specialist_num_layers=1,
        specialist_fusion_scale=0.25,
        cross_family_fusion_scale=0.25,
        ctx_cont_dim=MODEL_NATIVE_CTX_CONT_DIM,
        ctx_cat_dim=MODEL_NATIVE_CTX_CAT_DIM,
        m5_seq_dim=MTF_DIM,
        m15_seq_dim=MTF_DIM,
        h1_seq_dim=MTF_DIM,
        h4_seq_dim=MTF_DIM,
        d1_seq_dim=MTF_DIM,
        m5_seq_len=16,
        m15_seq_len=16,
        h1_seq_len=16,
        h4_seq_len=16,
        d1_seq_len=16,
        specialist_input_indices=_specialist_indices(16),
        specialist_ctx_cont_indices=ctx_cont_indices,
        specialist_ctx_cont_nominal_indices={
            name: [] for name in MODEL_NATIVE_TRAINING_SPECIALISTS
        },
        specialist_ctx_cat_indices=ctx_cat_indices,
        multi_tf_specialist_input_indices=_multi_tf_specialist_indices(),
        temporal_alias_signal_indices=[],
        temporal_alias_ctx_cont_indices=[],
        input_normalization=_input_normalization(_specialist_indices(16)),
    ).eval()

    assert any("specialist" in key for key in model.state_dict())
    ctx_cont = _valid_ctx_cont(2)
    seq_x, snap_x = _synchronized_seq_snap(
        torch.randn(2, 16, 16),
        ctx_cont,
    )
    out = model(
        seq_x,
        snap_x,
        ctx_cat=torch.zeros(2, MODEL_NATIVE_CTX_CAT_DIM, dtype=torch.long),
        ctx_cont=ctx_cont,
        **{
            f"seq_{tf}": torch.randn(2, 16, MTF_DIM)
            for tf in (
                timeframe.lower() for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES
            )
        },
    )
    assert "specialist_gate" in out
    assert out["model_native_logits"].shape == (2, 3)
    assert not ({"anchor_logits", "delta_logits", "anchor_gate"} & set(out))


def test_entry_v10_specialist_fusion_forward_exact_model_native_contract() -> None:
    specialist_indices = _specialist_indices()
    context_routing = _context_routing(specialist_indices)
    model = _build_unit_test_entry_v10_ctx_hybrid_transformer(
        seq_input_dim=MODEL_NATIVE_SIGNAL_DIM,
        snap_input_dim=MODEL_NATIVE_SIGNAL_DIM,
        seq_len=16,
        dropout=0.05,
        multi_tf_num_layers=1,
        multi_tf_scale=0.5,
        specialist_fusion_scale=0.25,
        cross_family_fusion_scale=0.25,
        ctx_cont_dim=MODEL_NATIVE_CTX_CONT_DIM,
        ctx_cat_dim=MODEL_NATIVE_CTX_CAT_DIM,
        m5_seq_dim=MTF_DIM,
        m15_seq_dim=MTF_DIM,
        h1_seq_dim=MTF_DIM,
        h4_seq_dim=MTF_DIM,
        d1_seq_dim=MTF_DIM,
        m5_seq_len=16,
        m15_seq_len=16,
        h1_seq_len=16,
        h4_seq_len=16,
        d1_seq_len=16,
        specialist_input_indices=specialist_indices,
        specialist_ctx_cont_indices=context_routing["ctx_cont_indices"],
        specialist_ctx_cont_nominal_indices=context_routing[
            "ctx_cont_nominal_indices"
        ],
        specialist_ctx_cat_indices=context_routing["ctx_cat_indices"],
        multi_tf_specialist_input_indices=_multi_tf_specialist_indices(),
        temporal_alias_signal_indices=context_routing[
            "temporal_alias_policy"
        ]["signal_indices"],
        temporal_alias_ctx_cont_indices=context_routing[
            "temporal_alias_policy"
        ]["ctx_cont_indices"],
        specialist_num_layers=1,
        input_normalization=_input_normalization(specialist_indices),
    ).eval()

    ctx_cont = _valid_ctx_cont(2)
    seq_x, snap_x = _synchronized_seq_snap(
        torch.randn(2, 16, MODEL_NATIVE_SIGNAL_DIM),
        ctx_cont,
        context_routing,
    )
    out = model(
        seq_x,
        snap_x,
        ctx_cat=torch.zeros(2, MODEL_NATIVE_CTX_CAT_DIM, dtype=torch.long),
        ctx_cont=ctx_cont,
        **{
            f"seq_{tf}": torch.randn(2, 16, MTF_DIM)
            for tf in (
                timeframe.lower() for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES
            )
        },
    )

    assert model._specialist_names == MODEL_NATIVE_TRAINING_SPECIALISTS
    assert out["direction_logits"].shape == (2, 3)
    assert out["model_native_logits"].shape == (2, 3)
    assert not ({"anchor_logits", "delta_logits", "anchor_gate"} & set(out))
    assert out["specialist_gate"].shape == (2, len(MODEL_NATIVE_TRAINING_SPECIALISTS))
    assert torch.allclose(out["specialist_gate"].sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.isfinite(out["specialist_gate"]).all()
    assert out["tf_gate"].shape == (2, ENTRY_MTF_CONTEXT_COUNT)
    assert out["family_tf_cooperation_gate"].shape == (
        2,
        ENTRY_MTF_CONTEXT_COUNT * len(MODEL_NATIVE_TRAINING_SPECIALISTS),
    )
    assert out["family_tf_feature_gate"].shape == (
        2,
        ENTRY_MTF_CONTEXT_COUNT,
        MTF_DIM,
    )
    assert torch.allclose(out["tf_gate"].sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(
        out["family_tf_cooperation_gate"].sum(dim=1), torch.ones(2), atol=1e-6
    )


def test_all_147_context_fields_move_only_their_pre_cross_owner_token() -> None:
    specialist_indices = _specialist_indices()
    context_routing = _context_routing(specialist_indices)
    model = _build_unit_test_entry_v10_ctx_hybrid_transformer(
        seq_input_dim=MODEL_NATIVE_SIGNAL_DIM,
        snap_input_dim=MODEL_NATIVE_SIGNAL_DIM,
        seq_len=4,
        dropout=0.05,
        multi_tf_num_layers=1,
        multi_tf_scale=0.5,
        specialist_num_layers=1,
        specialist_fusion_scale=0.25,
        cross_family_fusion_scale=0.25,
        ctx_cont_dim=MODEL_NATIVE_CTX_CONT_DIM,
        ctx_cat_dim=MODEL_NATIVE_CTX_CAT_DIM,
        m5_seq_dim=MTF_DIM,
        m15_seq_dim=MTF_DIM,
        h1_seq_dim=MTF_DIM,
        h4_seq_dim=MTF_DIM,
        d1_seq_dim=MTF_DIM,
        m5_seq_len=4,
        m15_seq_len=4,
        h1_seq_len=4,
        h4_seq_len=4,
        d1_seq_len=4,
        specialist_input_indices=specialist_indices,
        specialist_ctx_cont_indices=context_routing["ctx_cont_indices"],
        specialist_ctx_cont_nominal_indices=context_routing[
            "ctx_cont_nominal_indices"
        ],
        specialist_ctx_cat_indices=context_routing["ctx_cat_indices"],
        multi_tf_specialist_input_indices=_multi_tf_specialist_indices(),
        temporal_alias_signal_indices=context_routing[
            "temporal_alias_policy"
        ]["signal_indices"],
        temporal_alias_ctx_cont_indices=context_routing[
            "temporal_alias_policy"
        ]["ctx_cont_indices"],
        input_normalization=_input_normalization(specialist_indices),
    ).eval()
    torch.manual_seed(23)
    ctx_cont = _valid_ctx_cont(1)
    ctx_cat = torch.zeros(1, MODEL_NATIVE_CTX_CAT_DIM, dtype=torch.long)
    with torch.no_grad():
        baseline, _ = model._build_family_context_tokens(ctx_cont, ctx_cat)
        nominal_indices = {
            index
            for values in context_routing["ctx_cont_nominal_indices"].values()
            for index in values
        }
        for owner_position, owner in enumerate(MODEL_NATIVE_TRAINING_SPECIALISTS):
            for index in context_routing["ctx_cont_indices"][owner]:
                changed = ctx_cont.clone()
                if index in nominal_indices:
                    changed[0, index] = (changed[0, index] + 1.0) % 5.0
                else:
                    changed[0, index] += 0.75
                observed, _ = model._build_family_context_tokens(
                    changed,
                    ctx_cat,
                )
                movement = (observed - baseline).abs().amax(dim=2).squeeze(0)
                assert float(movement[owner_position]) > 1e-7
                assert torch.count_nonzero(
                    movement[
                        torch.arange(len(movement)) != owner_position
                    ]
                ).item() == 0
            for index in context_routing["ctx_cat_indices"][owner]:
                changed_cat = ctx_cat.clone()
                changed_cat[0, index] = 1
                observed, _ = model._build_family_context_tokens(
                    ctx_cont,
                    changed_cat,
                )
                movement = (observed - baseline).abs().amax(dim=2).squeeze(0)
                assert float(movement[owner_position]) > 1e-7
                assert torch.count_nonzero(
                    movement[
                        torch.arange(len(movement)) != owner_position
                    ]
                ).item() == 0


@pytest.mark.parametrize("invalid_value", (1.5, -1.0, 5.0))
def test_nominal_ctx_cont_regime_ids_fail_closed_outside_integer_domain(
    invalid_value: float,
) -> None:
    specialist_indices = _specialist_indices(16)
    routing = _context_routing(_specialist_indices())
    ctx_cont_indices, ctx_cat_indices = _context_indices()
    model = _build_unit_test_entry_v10_ctx_hybrid_transformer(
        seq_input_dim=16,
        snap_input_dim=16,
        seq_len=4,
        dropout=0.05,
        multi_tf_num_layers=1,
        multi_tf_scale=0.5,
        specialist_num_layers=1,
        specialist_fusion_scale=0.25,
        cross_family_fusion_scale=0.25,
        ctx_cont_dim=MODEL_NATIVE_CTX_CONT_DIM,
        ctx_cat_dim=MODEL_NATIVE_CTX_CAT_DIM,
        m5_seq_dim=MTF_DIM,
        m15_seq_dim=MTF_DIM,
        h1_seq_dim=MTF_DIM,
        h4_seq_dim=MTF_DIM,
        d1_seq_dim=MTF_DIM,
        m5_seq_len=4,
        m15_seq_len=4,
        h1_seq_len=4,
        h4_seq_len=4,
        d1_seq_len=4,
        specialist_input_indices=specialist_indices,
        specialist_ctx_cont_indices=ctx_cont_indices,
        specialist_ctx_cont_nominal_indices=routing[
            "ctx_cont_nominal_indices"
        ],
        specialist_ctx_cat_indices=ctx_cat_indices,
        multi_tf_specialist_input_indices=_multi_tf_specialist_indices(),
        temporal_alias_signal_indices=[],
        temporal_alias_ctx_cont_indices=[],
        input_normalization=_input_normalization(specialist_indices),
    ).eval()
    ctx_cont = _valid_ctx_cont(1)
    nominal_index = next(
        index
        for values in routing["ctx_cont_nominal_indices"].values()
        for index in values
    )
    ctx_cont[0, nominal_index] = invalid_value

    with pytest.raises(RuntimeError, match="CTX_CONT_NOMINAL_DOMAIN_INVALID"):
        model._build_family_context_tokens(
            ctx_cont,
            torch.zeros(1, MODEL_NATIVE_CTX_CAT_DIM, dtype=torch.long),
        )


def test_temporal_alias_current_snap_copies_are_excluded_from_generic_projection() -> None:
    specialist_indices = _specialist_indices()
    context_routing = _context_routing(specialist_indices)
    aliases = context_routing["temporal_alias_policy"]["signal_indices"]
    ctx_cont_indices, ctx_cat_indices = _context_indices()
    model = _build_unit_test_entry_v10_ctx_hybrid_transformer(
        seq_input_dim=MODEL_NATIVE_SIGNAL_DIM,
        snap_input_dim=MODEL_NATIVE_SIGNAL_DIM,
        seq_len=4,
        dropout=0.05,
        multi_tf_num_layers=1,
        multi_tf_scale=0.5,
        specialist_num_layers=1,
        specialist_fusion_scale=0.25,
        cross_family_fusion_scale=0.25,
        ctx_cont_dim=MODEL_NATIVE_CTX_CONT_DIM,
        ctx_cat_dim=MODEL_NATIVE_CTX_CAT_DIM,
        m5_seq_dim=MTF_DIM,
        m15_seq_dim=MTF_DIM,
        h1_seq_dim=MTF_DIM,
        h4_seq_dim=MTF_DIM,
        d1_seq_dim=MTF_DIM,
        m5_seq_len=4,
        m15_seq_len=4,
        h1_seq_len=4,
        h4_seq_len=4,
        d1_seq_len=4,
        specialist_input_indices=specialist_indices,
        specialist_ctx_cont_indices=ctx_cont_indices,
        specialist_ctx_cont_nominal_indices={
            name: [] for name in MODEL_NATIVE_TRAINING_SPECIALISTS
        },
        specialist_ctx_cat_indices=ctx_cat_indices,
        multi_tf_specialist_input_indices=_multi_tf_specialist_indices(),
        temporal_alias_signal_indices=aliases,
        temporal_alias_ctx_cont_indices=context_routing[
            "temporal_alias_policy"
        ]["ctx_cont_indices"],
        input_normalization=_input_normalization(specialist_indices),
    ).eval()
    generic = model.generic_snap_idx.tolist()
    assert len(aliases) == 82
    assert len(generic) == MODEL_NATIVE_SIGNAL_DIM - 82
    assert set(aliases).isdisjoint(generic)

    snap = torch.randn(2, MODEL_NATIVE_SIGNAL_DIM)
    changed = snap.clone()
    changed[:, aliases] += 100.0
    with torch.no_grad():
        before = model.snap_proj(snap[:, generic])
        after = model.snap_proj(changed[:, generic])
    assert torch.equal(before, after)


def test_forward_fails_closed_on_stale_snap_and_ctx_alias_mismatch() -> None:
    routing = _context_routing(_specialist_indices())
    model = _guard_test_model(routing)
    ctx_cont = _valid_ctx_cont(1)
    seq_x, snap_x = _synchronized_seq_snap(
        torch.randn(1, 4, MODEL_NATIVE_SIGNAL_DIM),
        ctx_cont,
        routing,
    )
    ctx_cat = torch.zeros(1, MODEL_NATIVE_CTX_CAT_DIM, dtype=torch.long)
    mtf = {
        f"seq_{tf}": torch.randn(1, 4, MTF_DIM)
        for tf in (
            timeframe.lower() for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES
        )
    }

    stale_snap = snap_x.clone()
    stale_snap[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="SEQ_LAST_SNAP_NOT_BIT_IDENTICAL"):
        model(
            seq_x,
            stale_snap,
            ctx_cat=ctx_cat,
            ctx_cont=ctx_cont,
            **mtf,
        )

    alias_index = routing["temporal_alias_policy"]["signal_indices"][0]
    mismatched_seq = seq_x.clone()
    mismatched_snap = snap_x.clone()
    mismatched_seq[0, -1, alias_index] += 1.0
    mismatched_snap[0, alias_index] += 1.0
    with pytest.raises(
        RuntimeError,
        match="SNAP_CTX_CONT_ALIAS_NOT_BIT_IDENTICAL",
    ):
        model(
            mismatched_seq,
            mismatched_snap,
            ctx_cat=ctx_cat,
            ctx_cont=ctx_cont,
            **mtf,
        )


def test_entry_v10_specialist_loader_accepts_only_exact_model_native_audit(
    tmp_path: Path,
) -> None:
    specialist_indices, specialist_meta = _load_specialist_fusion_contract(
        _write_audit(tmp_path),
        expected_signal_dim=MODEL_NATIVE_SIGNAL_DIM,
        ordered_signal_names=_ordered_signal_names(),
        contract_mode=MODEL_NATIVE_CONTRACT_MODE,
    )

    assert tuple(specialist_indices) == MODEL_NATIVE_TRAINING_SPECIALISTS
    assert specialist_meta["contract_mode"] == MODEL_NATIVE_CONTRACT_MODE
    assert specialist_meta["signal_field_count"] == MODEL_NATIVE_SIGNAL_DIM
    assert specialist_meta["selected_feature_count"] == 479
    assert specialist_meta["trainable_specialists"] == list(MODEL_NATIVE_TRAINING_SPECIALISTS)
    assert specialist_meta["excluded_specialist_groups"] == {}
    assert specialist_meta["active_heads"] == list(SPECIALIST_FUSION_ACTIVE_HEADS)
    assert specialist_meta["blocked_heads"] == list(SPECIALIST_FUSION_BLOCKED_HEADS)
    assert (
        specialist_meta["context_routing"]
        == _audit_payload()["architecture_contract"][
            "context_specialist_routing"
        ]
    )
    assert specialist_meta["specialist_model_contract"] == json.loads(
        json.dumps(MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT)
    )


def test_entry_v10_specialist_loader_rejects_partial_513_coverage(
    tmp_path: Path,
) -> None:
    payload = _audit_payload()
    payload["architecture_contract"]["specialist_input_indices"] = {
        name: [index]
        for index, name in enumerate(MODEL_NATIVE_TRAINING_SPECIALISTS)
    }

    with pytest.raises(RuntimeError, match="SPECIALIST_INDEX_COVERAGE_INVALID"):
        _load_specialist_fusion_contract(
            _write_audit(tmp_path, payload),
            expected_signal_dim=MODEL_NATIVE_SIGNAL_DIM,
            ordered_signal_names=_ordered_signal_names(),
            contract_mode=MODEL_NATIVE_CONTRACT_MODE,
        )


def test_entry_v10_specialist_loader_rejects_missing_context_routing(
    tmp_path: Path,
) -> None:
    payload = _audit_payload()
    del payload["architecture_contract"]["context_specialist_routing"]

    with pytest.raises(RuntimeError, match="CONTEXT_SPECIALIST_ROUTING_MISSING"):
        _load_specialist_fusion_contract(
            _write_audit(tmp_path, payload),
            expected_signal_dim=MODEL_NATIVE_SIGNAL_DIM,
            ordered_signal_names=_ordered_signal_names(),
            contract_mode=MODEL_NATIVE_CONTRACT_MODE,
        )


def test_entry_v10_specialist_loader_rejects_tampered_model_contract(tmp_path: Path) -> None:
    payload = _audit_payload()
    payload["specialist_model_contract"]["structure_swing_encoder"]["owned_objectives"] = []

    with pytest.raises(RuntimeError, match="SPECIALIST_MODEL_CONTRACT_INVALID"):
        _load_specialist_fusion_contract(
            _write_audit(tmp_path, payload),
            expected_signal_dim=MODEL_NATIVE_SIGNAL_DIM,
            ordered_signal_names=_ordered_signal_names(),
            contract_mode=MODEL_NATIVE_CONTRACT_MODE,
        )


@pytest.mark.parametrize(
    "stale_mode",
    ["", "foundation_seq146", "challenger_seq215", "smart_seq520_candidate"],
)
def test_entry_v10_specialist_loader_rejects_stale_contract_modes(
    tmp_path: Path,
    stale_mode: str,
) -> None:
    with pytest.raises(RuntimeError, match="SPECIALIST_MODEL_NATIVE_CONTRACT_REQUIRED"):
        _load_specialist_fusion_contract(
            _write_audit(tmp_path),
            expected_signal_dim=MODEL_NATIVE_SIGNAL_DIM,
            ordered_signal_names=_ordered_signal_names(),
            contract_mode=stale_mode,
        )
