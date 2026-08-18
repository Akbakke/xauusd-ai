from __future__ import annotations

from collections import Counter
import inspect

import pandas as pd
import pytest
import torch

from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT,
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_BASIC_V1_CONTRACT,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DOMAINS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CAT_INDEX_BY_NAME,
    MODEL_NATIVE_CTX_CAT_MIN_MAX,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_CTX_CONT_INDEX_BY_NAME,
    MODEL_NATIVE_CTX_CONT_REGIME_FIELDS,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_MANDATORY_FULL_STACK_SCHEMA_VERSION,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
    RETIRED_HANDCRAFTED_CTX_CONT_FIELDS,
    RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS,
    RETIRED_SMC_CTX_COMPOSITE_FIELDS,
    RETIRED_MODEL_NATIVE_SIGNAL_FIELDS,
    model_native_context_contract_metadata,
    model_native_mandatory_full_stack_metadata,
    model_native_signal_contract_metadata,
    ordered_model_native_signal_fields,
    require_model_native_manifest,
    require_model_native_signal_contract,
)
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    CTX_CAT_DOMAINS,
)
from gx1.contracts.entry_model_native_state_v2 import RETIRED_RANK_STATE_FIELDS
from gx1.execution import v12_model_native_state_live as state_module
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT,
    MODEL_NATIVE_CTX_CAT_DOMAINS as SPECIALIST_CTX_CAT_DOMAINS,
    classify_entry_specialist_feature,
    model_native_context_temporal_alias_policy,
    specialist_contract_training_allowed_for_mode,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    MODEL_NATIVE_MANDATORY_FAMILY_FEATURES,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    PRICE_DERIVED_FEATURE_NAMES_SHA256,
    PRICE_DERIVED_FORMULA_SHA256,
    PRICE_DERIVED_FORMULA_SCHEMA_VERSION,
    PRICE_DERIVED_FEATURE_NAMES,
    price_derived_contract_metadata,
)
from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
    foundation_structure_contract_metadata,
)
from gx1.features.swing_structure_v1 import swing_structure_contract_metadata
from gx1.features.entry_candle_primitives_v1 import (
    CANDLE_PRIMITIVE_FEATURE_NAMES,
    CANDLE_PRIMITIVE_FEATURE_NAMES_SHA256,
    CANDLE_PRIMITIVE_FEATURE_VERSION,
)
from gx1.features.smc_v1 import smc_primitive_contract_metadata
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EXACT_CTX_CAT_DOMAINS,
    EXACT_SPECIALIST_NAMES,
    EntryV10CtxHybridTransformer,
    _build_unit_test_entry_v10_ctx_hybrid_transformer,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields
from tests.model_native_input_normalization_support import (
    input_normalization_fixture,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
)


# The code-owned family fields plus ranked fixture fields form the exact
# selected surface. The genuine base owner produces these totals.
# The three unsigned tick-volume primitives belong to momentum/flow; the
# hand-composed signed-volume interaction is retired.
# Rule 4: every declared specialist family must stay live in the learned
# path. The per-family totals move with every surface wave, so this binds the
# invariant (all eight families present, none empty, all fields routed) rather
# than a restated count that has gone stale within days (rule 13).
_TEST_MTF_DIM = len(EXACT_SPECIALIST_NAMES)


def _test_multi_tf_specialist_indices() -> dict[str, list[int]]:
    return {
        name: [position]
        for position, name in enumerate(EXACT_SPECIALIST_NAMES)
    }


def _selected_fields() -> list[str]:
    return canonical_model_native_selected_fields()


def _native_manifest() -> dict:
    selected = _selected_fields()
    return {
        "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
        "base_signal_feature_count": len(MODEL_NATIVE_BASE_FIELDS),
        "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
        "selected_features": selected,
        "mandatory_full_stack": model_native_mandatory_full_stack_metadata(),
        "model_native_signal_contract": model_native_signal_contract_metadata(selected),
    }


def _exact_specialist_indices() -> dict[str, list[int]]:
    fields = ordered_model_native_signal_fields(_selected_fields())
    return {
        specialist: [
            index
            for index, field in enumerate(fields)
            if classify_entry_specialist_feature(field) == specialist
        ]
        for specialist in EXACT_SPECIALIST_NAMES
    }


def _exact_model_kwargs(*, ctx_cont_dim: int = MODEL_NATIVE_CTX_CONT_DIM) -> dict:
    ordered_signal_names = list(
        ordered_model_native_signal_fields(_selected_fields())
    )
    temporal_alias_policy = model_native_context_temporal_alias_policy(
        ordered_signal_names
    )
    return {
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": 4,
        "dropout": 0.05,
        "multi_tf_num_layers": 1,
        "multi_tf_scale": 0.5,
        "specialist_num_layers": 1,
        "specialist_fusion_scale": 0.25,
        "cross_family_fusion_scale": 0.25,
        "ctx_cont_dim": ctx_cont_dim,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "m5_seq_dim": _TEST_MTF_DIM,
        "m15_seq_dim": _TEST_MTF_DIM,
        "h1_seq_dim": _TEST_MTF_DIM,
        "h4_seq_dim": _TEST_MTF_DIM,
        "d1_seq_dim": _TEST_MTF_DIM,
        "m5_seq_len": 4,
        "m15_seq_len": 4,
        "h1_seq_len": 4,
        "h4_seq_len": 4,
        "d1_seq_len": 4,
        "specialist_input_indices": _exact_specialist_indices(),
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
        "multi_tf_specialist_input_indices": (
            _test_multi_tf_specialist_indices()
        ),
        "temporal_alias_signal_indices": list(
            temporal_alias_policy["signal_indices"]
        ),
        "temporal_alias_ctx_cont_indices": list(
            temporal_alias_policy["ctx_cont_indices"]
        ),
        "input_normalization": input_normalization_fixture(
            signal_names=ordered_signal_names,
            mtf_names=[
                f"mtf_{index}" for index in range(_TEST_MTF_DIM)
            ],
        ),
    }


def test_model_native_signal_contract_has_exact_derived_width_and_all_groups_live() -> (
    None
):
    selected = _selected_fields()
    fields = ordered_model_native_signal_fields(selected)
    contract = model_native_signal_contract_metadata(selected)

    # Every width is DERIVED from the owner tuples. The surface moved five
    # times in two days; a restated 250/164/86/279 is exactly the defect this
    # contract exists to prevent (rule 13).
    assert len(selected) == MODEL_NATIVE_SELECTED_FEATURE_COUNT
    assert MODEL_NATIVE_SELECTED_FEATURE_COUNT == (
        len(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
        + len(MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS)
    )
    assert len(MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS) == (
        MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT
    )
    assert len(fields) == MODEL_NATIVE_SIGNAL_DIM
    assert MODEL_NATIVE_SIGNAL_DIM == (
        len(MODEL_NATIVE_BASE_FIELDS) + MODEL_NATIVE_SELECTED_FEATURE_COUNT
    )
    # The schema identities are bound to the emitted contract rather than
    # pinned, so a legitimate bump cannot pass while the manifest keeps an
    # older identity.
    assert contract["schema_version"] == MODEL_NATIVE_SIGNAL_SCHEMA_VERSION
    assert contract["contract_mode"] == MODEL_NATIVE_CONTRACT_MODE
    assert MODEL_NATIVE_SIGNAL_SCHEMA_VERSION.startswith(
        "entry_model_native_signal_v"
    )
    assert MODEL_NATIVE_CONTRACT_MODE.startswith(
        "xau_seq513_model_native_direction_v"
    )
    assert MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION.startswith(
        "entry_model_native_seq513_split_manifest_v"
    )
    assert MODEL_NATIVE_MANDATORY_FULL_STACK_SCHEMA_VERSION.startswith(
        "entry_model_native_mandatory_full_stack_v"
    )
    assert fields[: len(MODEL_NATIVE_BASE_FIELDS)] == MODEL_NATIVE_BASE_FIELDS
    assert fields[len(MODEL_NATIVE_BASE_FIELDS) :] == tuple(selected)
    assert not (set(fields) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS))
    assert not (set(fields) & set(RETIRED_MODEL_NATIVE_SIGNAL_FIELDS))
    assert contract["retired_signal_fields"] == list(RETIRED_MODEL_NATIVE_SIGNAL_FIELDS)
    assert tuple(contract["retired_handcrafted_ctx_cont_fields"]) == (
        RETIRED_HANDCRAFTED_CTX_CONT_FIELDS
    )
    assert tuple(contract["retired_operator_ctx_cont_composite_fields"]) == (
        RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS
    )
    assert tuple(contract["retired_smc_ctx_composite_fields"]) == (
        RETIRED_SMC_CTX_COMPOSITE_FIELDS
    )
    assert contract["bridge_dim"] == 0
    assert contract["bridge_source"] is None
    assert contract["anchor_source"] is None
    require_model_native_signal_contract(contract, context="TEST")
    assert require_model_native_manifest(_native_manifest(), context="TEST") == contract

    candle_owner = contract["mandatory_full_stack"]["candle_primitive_owner"]
    assert candle_owner == {
        "owner": "gx1.features.entry_candle_primitives_v1",
        "feature_version": CANDLE_PRIMITIVE_FEATURE_VERSION,
        "feature_count": len(CANDLE_PRIMITIVE_FEATURE_NAMES),
        "ordered_feature_names_sha256": CANDLE_PRIMITIVE_FEATURE_NAMES_SHA256,
        "ordered_feature_names": list(CANDLE_PRIMITIVE_FEATURE_NAMES),
    }
    # 2026-08-15: candle.raw_zero_range_flag retired from the ordered tuple
    # (constant post-warmup on H4/D1 -> hard liveness RED, and unscaleable as
    # a declared constant).
    # 2026-08-18 (V30 wave 2): candle.raw_close_location,
    # candle.raw_range_change_local_geometry and the two
    # candle.raw_*_rejection_depth_local_geometry columns retired -- each an
    # exact function of columns that stay in this owner. Re-derived over the
    # narrowed tuple; the literal is the drift guard, not the source.
    assert CANDLE_PRIMITIVE_FEATURE_NAMES_SHA256 == (
        "c72093fed2e8eef17917bf92fc4a2742ecd6f5a3967834b416216e2ba776f475"
    )
    price_owner = contract["mandatory_full_stack"]["price_derived_owner"]
    assert price_owner == price_derived_contract_metadata()
    assert price_owner["schema_version"] == PRICE_DERIVED_FORMULA_SCHEMA_VERSION
    assert PRICE_DERIVED_FORMULA_SHA256 == (
        "62a2cb37fde1e0b0bc654aa39430deb6c4e309acab9d00139f948057b407bdee"
    )
    assert (
        price_owner["ordered_feature_names_sha256"]
        == PRICE_DERIVED_FEATURE_NAMES_SHA256
    )
    assert contract["mandatory_full_stack"]["smc_primitive_owner"] == (
        smc_primitive_contract_metadata()
    )
    assert contract["mandatory_full_stack"]["foundation_structure_owner"] == (
        foundation_structure_contract_metadata()
    )
    assert contract["mandatory_full_stack"]["swing_structure_owner"] == (
        swing_structure_contract_metadata()
    )

    counts = Counter(classify_entry_specialist_feature(field) for field in fields)
    assert set(counts) == set(EXACT_SPECIALIST_NAMES)
    assert all(counts[specialist] > 0 for specialist in EXACT_SPECIALIST_NAMES)
    assert sum(counts.values()) == MODEL_NATIVE_SIGNAL_DIM
    assert "neutral_bridge_anchor" not in counts
    assert (
        specialist_contract_training_allowed_for_mode(MODEL_NATIVE_CONTRACT_MODE)
        is True
    )


def test_retired_signed_volume_cannot_reenter_ranked_signal_surface() -> None:
    selected = _selected_fields()
    selected[-1] = "signed_vol_z_20"

    with pytest.raises(
        RuntimeError,
        match="retired_model_native_signal_fields",
    ):
        ordered_model_native_signal_fields(selected)


def test_exact_local_ema50_200_evidence_is_mandatory_and_trend_owned() -> None:
    families = dict(MODEL_NATIVE_MANDATORY_FAMILY_FEATURES)
    assert families["price_ema50_200_layer"] == PRICE_DERIVED_FEATURE_NAMES
    assert set(PRICE_DERIVED_FEATURE_NAMES).issubset(
        set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    )
    assert {
        classify_entry_specialist_feature(name) for name in PRICE_DERIVED_FEATURE_NAMES
    } == {"trend_ema_encoder"}


def test_all_foundation_cross_family_evidence_is_mandatory_and_routed() -> None:
    families = dict(MODEL_NATIVE_MANDATORY_FAMILY_FEATURES)
    assert families["foundation_cross_family_layer"] == (
        FOUNDATION_STRUCTURE_FEATURE_NAMES
    )
    assert tuple(
        MODEL_NATIVE_MANDATORY_SELECTED_FIELDS[
            : len(FOUNDATION_STRUCTURE_FEATURE_NAMES)
        ]
    ) == (
        FOUNDATION_STRUCTURE_FEATURE_NAMES
    )
    assert Counter(
        classify_entry_specialist_feature(name)
        for name in FOUNDATION_STRUCTURE_FEATURE_NAMES
    ) == {"structure_swing_encoder": len(FOUNDATION_STRUCTURE_FEATURE_NAMES)}


def test_active_context_contract_always_contains_full_regime_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GX1_REGIME_V4", "0")
    monkeypatch.delenv("GX1_TRUTH_MODE", raising=False)
    monkeypatch.delenv("GX1_RUN_MODE", raising=False)

    context = model_native_context_contract_metadata()
    assert len(MODEL_NATIVE_CTX_CONT_FIELDS) == MODEL_NATIVE_CTX_CONT_DIM
    assert len(MODEL_NATIVE_CTX_CAT_FIELDS) == MODEL_NATIVE_CTX_CAT_DIM
    assert MODEL_NATIVE_CTX_CONT_REGIME_FIELDS
    assert MODEL_NATIVE_CTX_CONT_FIELDS[
        -len(MODEL_NATIVE_CTX_CONT_REGIME_FIELDS) :
    ] == (MODEL_NATIVE_CTX_CONT_REGIME_FIELDS)
    assert "trend_regime_id" not in MODEL_NATIVE_CTX_CAT_FIELDS
    assert tuple(context["ctx_cont_names"]) == MODEL_NATIVE_CTX_CONT_FIELDS
    assert tuple(context["ctx_cat_names"]) == MODEL_NATIVE_CTX_CAT_FIELDS
    assert MODEL_NATIVE_CTX_CONT_INDEX_BY_NAME == {
        name: index for index, name in enumerate(MODEL_NATIVE_CTX_CONT_FIELDS)
    }
    assert (
        MODEL_NATIVE_CTX_CONT_INDEX_BY_NAME["D1_dist_from_ema200_atr"] == 2
    )

    contract = model_native_signal_contract_metadata(_selected_fields())
    contract["ctx_cont_names"] = list(MODEL_NATIVE_CTX_CONT_FIELDS[:-1])
    with pytest.raises(RuntimeError, match="ctx_cont_names order mismatch"):
        require_model_native_signal_contract(contract, context="TEST")


def test_categorical_context_owner_routes_exactly_to_all_active_consumers() -> None:
    expected_fields = tuple(MODEL_NATIVE_CTX_CAT_DOMAINS)
    assert expected_fields == MODEL_NATIVE_CTX_CAT_FIELDS
    assert MODEL_NATIVE_CTX_CAT_INDEX_BY_NAME == {
        name: index for index, name in enumerate(expected_fields)
    }
    assert MODEL_NATIVE_CTX_CAT_MIN_MAX == {
        name: (min(domain), max(domain))
        for name, domain in MODEL_NATIVE_CTX_CAT_DOMAINS.items()
    }
    assert CTX_CAT_DOMAINS is MODEL_NATIVE_CTX_CAT_DOMAINS
    assert SPECIALIST_CTX_CAT_DOMAINS is MODEL_NATIVE_CTX_CAT_DOMAINS
    assert EXACT_CTX_CAT_DOMAINS is MODEL_NATIVE_CTX_CAT_DOMAINS
    assert state_module._MODEL_NATIVE_CTX_CAT_DOMAINS is MODEL_NATIVE_CTX_CAT_MIN_MAX


@pytest.mark.parametrize("forbidden_field", FORBIDDEN_LEGACY_BRIDGE_FIELDS)
def test_model_native_signal_contract_rejects_every_legacy_bridge_field(
    forbidden_field: str,
) -> None:
    selected = _selected_fields()
    selected[-1] = forbidden_field

    with pytest.raises(RuntimeError, match="forbidden_legacy_bridge_fields"):
        ordered_model_native_signal_fields(selected)


def test_retired_smart520_manifest_is_never_soft_migrated() -> None:
    manifest = _native_manifest()
    manifest["manifest_variant"] = "smart_seq520_candidate"
    manifest["base_signal_feature_count"] = 41
    manifest["expected_seq_snap_width"] = 520

    with pytest.raises(RuntimeError, match="RETIRED_SMART520_CONTRACT"):
        require_model_native_manifest(manifest, context="TEST")


def test_manifest_rejects_same_size_same_group_replacement_of_one_mandatory_field() -> (
    None
):
    manifest = _native_manifest()
    selected = list(manifest["selected_features"])
    victim = "ctx_cont._v1h1_ema_diff"
    replacement = "ctx_cont.adversarial_ema_diff_fixture"
    before = Counter(classify_entry_specialist_feature(name) for name in selected)
    selected[selected.index(victim)] = replacement
    after = Counter(classify_entry_specialist_feature(name) for name in selected)

    assert len(selected) == len(set(selected)) == MODEL_NATIVE_SELECTED_FEATURE_COUNT
    assert before == after
    manifest["selected_features"] = selected
    with pytest.raises(RuntimeError, match="missing_mandatory_full_stack_fields"):
        require_model_native_manifest(manifest, context="ADVERSARIAL")


def test_signal_contract_rejects_scattered_mandatory_registry_prefix() -> None:
    # All mandatory fields present, but two swapped out of exact registry
    # order: membership alone must not pass the documented prefix invariant.
    selected = _selected_fields()
    selected[0], selected[1] = selected[1], selected[0]

    with pytest.raises(RuntimeError, match="mandatory_registry_prefix_order_violation"):
        ordered_model_native_signal_fields(selected)


def test_signal_contract_rejects_stale_mandatory_family_metadata() -> None:
    contract = model_native_signal_contract_metadata(_selected_fields())
    contract["mandatory_full_stack"]["family_feature_counts"][
        "raw_mtf_trend_layer"
    ] -= 1

    with pytest.raises(RuntimeError, match="mandatory_full_stack metadata mismatch"):
        require_model_native_signal_contract(contract, context="STALE")


def test_signal_contract_rejects_basic_v1_formula_mutation() -> None:
    contract = model_native_signal_contract_metadata(_selected_fields())
    assert contract["basic_v1_contract"] == MODEL_NATIVE_BASIC_V1_CONTRACT
    contract["basic_v1_contract"]["formula_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="basic_v1_contract metadata mismatch"):
        require_model_native_signal_contract(contract, context="STALE")


def test_signal_contract_rejects_v16_v5_or_unbound_candle_owner() -> None:
    contract = model_native_signal_contract_metadata(_selected_fields())
    contract["schema_version"] = "entry_model_native_signal_v16"
    with pytest.raises(RuntimeError, match="schema_version"):
        require_model_native_signal_contract(contract, context="STALE")

    contract = model_native_signal_contract_metadata(_selected_fields())
    contract["contract_mode"] = "xau_seq513_model_native_direction_v5"
    with pytest.raises(RuntimeError, match="contract_mode"):
        require_model_native_signal_contract(contract, context="STALE")

    contract = model_native_signal_contract_metadata(_selected_fields())
    del contract["mandatory_full_stack"]["candle_primitive_owner"]
    with pytest.raises(RuntimeError, match="mandatory_full_stack metadata mismatch"):
        require_model_native_signal_contract(contract, context="STALE")

    contract = model_native_signal_contract_metadata(_selected_fields())
    del contract["mandatory_full_stack"]["price_derived_owner"]
    with pytest.raises(RuntimeError, match="mandatory_full_stack metadata mismatch"):
        require_model_native_signal_contract(contract, context="STALE")

    contract = model_native_signal_contract_metadata(_selected_fields())
    del contract["mandatory_full_stack"]["smc_primitive_owner"]
    with pytest.raises(RuntimeError, match="mandatory_full_stack metadata mismatch"):
        require_model_native_signal_contract(contract, context="STALE")

    contract = model_native_signal_contract_metadata(_selected_fields())
    del contract["mandatory_full_stack"]["foundation_structure_owner"]
    with pytest.raises(RuntimeError, match="mandatory_full_stack metadata mismatch"):
        require_model_native_signal_contract(contract, context="STALE")

    contract = model_native_signal_contract_metadata(_selected_fields())
    del contract["mandatory_full_stack"]["swing_structure_owner"]
    with pytest.raises(RuntimeError, match="mandatory_full_stack metadata mismatch"):
        require_model_native_signal_contract(contract, context="STALE")


def test_manifest_rejects_stale_top_level_mandatory_family_metadata() -> None:
    manifest = _native_manifest()
    manifest["mandatory_full_stack"]["family_features"]["raw_mtf_trend_layer"].append(
        "ctx_cont.unknown_stale_trend_member"
    )

    with pytest.raises(RuntimeError, match="MANDATORY_FULL_STACK_METADATA_STALE"):
        require_model_native_manifest(manifest, context="STALE")


def test_model_native_transformer_entry_q_is_direct_and_anchor_free() -> None:
    torch.manual_seed(7)
    model_kwargs = _exact_model_kwargs()
    model = _build_unit_test_entry_v10_ctx_hybrid_transformer(
        **model_kwargs
    ).eval()
    seq_x = torch.randn(2, 4, MODEL_NATIVE_SIGNAL_DIM)
    ordered_signal_names = list(
        ordered_model_native_signal_fields(_selected_fields())
    )
    for name, domain in model_kwargs["input_normalization"]["surfaces"][
        "signal"
    ]["categorical_domains"].items():
        signal_index = ordered_signal_names.index(name)
        seq_x[:, :, signal_index] = torch.randint(
            min(domain),
            max(domain) + 1,
            (2, 4),
        ).float()
    snap_x = seq_x[:, -1, :].clone()
    ctx_cont = torch.randn(2, MODEL_NATIVE_CTX_CONT_DIM)
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
        (2, len(nominal_indices)),
    ).float()
    alias_policy = model_native_context_temporal_alias_policy(
        ordered_model_native_signal_fields(_selected_fields())
    )
    alias_signal_idx = torch.tensor(alias_policy["signal_indices"], dtype=torch.long)
    alias_ctx_idx = torch.tensor(alias_policy["ctx_cont_indices"], dtype=torch.long)
    ctx_cont[:, alias_ctx_idx] = snap_x[:, alias_signal_idx]
    ctx_cat = torch.stack(
        [
            torch.randint(0, len(domain), (2,))
            for domain in EXACT_CTX_CAT_DOMAINS.values()
        ],
        dim=1,
    )

    mtf = {
        f"seq_{tf}": torch.randn(2, 4, _TEST_MTF_DIM)
        for tf in ("m15", "h1", "h4", "d1")
    }
    out = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)

    assert out["entry_action_q_bps"].shape == (2, 3)
    assert torch.isfinite(out["entry_action_q_bps"]).all()
    assert out["entry_q_joint_hidden"].shape == (2, model.cfg.d_model)
    assert "direction_logits" not in out
    assert "raw_direction_logits" not in out
    assert "model_native_logits" not in out
    assert not any("hierarchical_direction" in key for key in out)
    assert "anchor_logits" not in out
    assert "delta_logits" not in out
    assert "anchor_gate" not in out
    assert "anchor_eps" not in model.state_dict()
    assert not hasattr(model, "head_anchor_gate")
    assert not hasattr(model.cfg, "direction_logit_mode")
    assert "head_position_size.weight" in model.state_dict()
    assert out["position_size_logit"].shape == (2, 1)
    assert torch.count_nonzero(model.head_entry_action_q.weight).item() > 0

    with pytest.raises(TypeError, match="seq_m5"):
        model(
            seq_x,
            snap_x,
            ctx_cat=ctx_cat,
            ctx_cont=ctx_cont,
            seq_m5=torch.randn(2, 4, _TEST_MTF_DIM),
            **mtf,
        )


def test_model_native_transformer_has_no_legacy_direction_api() -> None:
    parameters = inspect.signature(EntryV10CtxHybridTransformer).parameters

    assert "direction_logit_mode" not in parameters
    assert "anchor_eps" not in parameters
    assert "enable_anchor_gate" not in parameters
    assert "anchor_gate_init" not in parameters
    assert "enable_position_size_head" not in parameters
    assert "enable_hold_horizon_head" not in parameters


def test_model_native_transformer_rejects_noncanonical_context_dimensions() -> None:
    with pytest.raises(
        RuntimeError,
        match="ENTRY_EXIT_PRODUCTION_ARCHITECTURE_MISMATCH",
    ):
        EntryV10CtxHybridTransformer(
            **_exact_model_kwargs(ctx_cont_dim=MODEL_NATIVE_CTX_CONT_DIM - 1)
        )


def test_state_builder_exposes_only_current_model_native_symbols(tmp_path) -> None:
    selected = _selected_fields()
    contract = model_native_signal_contract_metadata(selected)
    fit_start = pd.Timestamp("2026-01-01", tz="UTC")
    fit_end = pd.Timestamp("2026-01-02", tz="UTC")
    del fit_end  # the retired rank-reference fit window is no longer state.
    state_contract = state_module.ModelNativeStateContract(
        feature_history_start_utc=fit_start,
    )

    from tests.volatility_squeeze_test_support import (
        make_volatility_squeeze_artifact_set,
    )

    squeeze_artifacts = make_volatility_squeeze_artifact_set(tmp_path)
    builder = state_module.ModelNativeStateBuilder(
        ordered_signal_names=list(contract["fields"]),
        state_contract=state_contract,
        signal_contract=contract,
        volatility_squeeze_artifacts=squeeze_artifacts,
    )

    stale_contract = {**contract, "schema_version": "entry_model_native_signal_v16"}
    with pytest.raises(RuntimeError, match="schema_version"):
        state_module.ModelNativeStateBuilder(
            ordered_signal_names=list(contract["fields"]),
            state_contract=state_contract,
            signal_contract=stale_contract,
            volatility_squeeze_artifacts=squeeze_artifacts,
        )

    assert len(builder.ordered_signal_names) == MODEL_NATIVE_SIGNAL_DIM
    assert len(builder._ext_names) == MODEL_NATIVE_SELECTED_FEATURE_COUNT
    assert not hasattr(state_module, "Smart520StateBuilder")
    assert not hasattr(state_module, "Smart520StateContract")
    assert not hasattr(state_module, "SIGNAL_DIM_SMART520")
    # The retired TRAIN rank reference may not re-enter immutable state.
    assert not hasattr(state_module, "TrainRankReferenceV2")
    state_fields = set(vars(state_contract))
    assert state_fields.isdisjoint(RETIRED_RANK_STATE_FIELDS)
