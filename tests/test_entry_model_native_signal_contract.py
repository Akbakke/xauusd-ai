from __future__ import annotations

from collections import Counter
import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_BASE_FIELDS,
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
    MODEL_NATIVE_SIGNAL_DIM,
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
from gx1.contracts.entry_model_native_state_v2 import TrainRankReferenceV2
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
    PRICE_DERIVED_FEATURE_NAMES,
)
from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
)
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
# selected surface. The genuine 34-field base produces these totals.
# 2026-08-09: the three unsigned tick-volume participation fields in the base
# (vol_z_20, vol_ratio_5_20, vol_pct_96) moved from the volatility owner to
# the momentum/flow owner alongside signed_vol_z_20 (declared order-flow in
# gx1.features.volume_features), shifting 44/31 -> 41/34.
# Pre-V29 audited per-specialist counts plus the V29 stage-2 event-family
# additions, derived from the declared owner tuples (rule 13).
from gx1.features.entry_model_native_feature_layers_v1 import (
    LEVEL_REGISTRY_M5_LAYER_FEATURE_NAMES as _V29_LEVEL_NAMES,
    MOMENTUM_EVENT_M5_LAYER_FEATURE_NAMES as _V29_MOMENTUM_NAMES,
    REGIME_FLIP_EVENT_LAYER_FEATURE_NAMES as _V29_REGIME_NAMES,
    SWING_EVENT_LAYER_FEATURE_NAMES as _V29_SWING_NAMES,
    TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES as _V29_TRENDLINE_NAMES,
)
from gx1.features.entry_session_regime_interactions_v1 import (
    SESSION_REGIME_INTERACTION_MANDATORY_FEATURE_NAMES,
)

_EXPECTED_FULL_SPECIALIST_COUNTS = {
    # V30 package 7 (2026-08-13): the chart-geometry mandatory pin fell 18 -> 2
    # and the candle smart3 suffix 32 -> 31 (35 -> 34 here, the suffix plus the
    # three ctx/base candle names this specialist also owns).
    "chart_geometry_encoder": 2 + len(_V29_TRENDLINE_NAMES),
    "momentum_flow_encoder": 34 + len(_V29_MOMENTUM_NAMES),
    "price_action_candle_encoder": 34,
    # V30 package 8B (2026-08-13): 161 is the ctx/base part of this specialist,
    # unchanged by the package.  The session/regime LAYER part is no longer a
    # restated literal: the layer is produced in full but pinned only to its
    # five measured-genuine primitives, so it is derived from the owner tuple
    # (68 -> 5, and the emitted set also lost the hand-written abstain vote).
    "session_regime_encoder": (
        161
        + len(SESSION_REGIME_INTERACTION_MANDATORY_FEATURE_NAMES)
        + len(_V29_REGIME_NAMES)
    ),
    "smc_liquidity_encoder": 68 + len(_V29_LEVEL_NAMES),
    "structure_swing_encoder": 51 + len(_V29_SWING_NAMES),
    # V30 (2026-08-13): 45 = 37 + chart.local_kama_efficiency_30 (package 1)
    # + the three package-2 GAP-2/3 local age fields
    # (chart.local_ema50_200_cross_age_norm,
    # chart.local_price_above_ema{50,200}_age_norm) + the four package-3
    # price-vs-EMA cross events (chart.local_price_x_ema{50,200}_cross_{up,down})
    # — all price-derived layer fields, all trend-owned.
    "trend_ema_encoder": 45,
    "vol_compression_encoder": 41,
}
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


def test_model_native_signal_contract_is_exact_34_plus_479_and_all_groups_live() -> (
    None
):
    selected = _selected_fields()
    fields = ordered_model_native_signal_fields(selected)
    contract = model_native_signal_contract_metadata(selected)

    assert len(selected) == MODEL_NATIVE_SELECTED_FEATURE_COUNT
    assert len(fields) == MODEL_NATIVE_SIGNAL_DIM == MODEL_NATIVE_SIGNAL_DIM
    assert fields[:34] == MODEL_NATIVE_BASE_FIELDS
    assert fields[34:] == tuple(selected)
    assert not (set(fields) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS))
    assert contract["bridge_dim"] == 0
    assert contract["bridge_source"] is None
    assert contract["anchor_source"] is None
    require_model_native_signal_contract(contract, context="TEST")
    assert require_model_native_manifest(_native_manifest(), context="TEST") == contract

    counts = Counter(classify_entry_specialist_feature(field) for field in fields)
    assert counts == _EXPECTED_FULL_SPECIALIST_COUNTS
    assert all(
        counts[specialist] > 0 for specialist in _EXPECTED_FULL_SPECIALIST_COUNTS
    )
    assert "neutral_bridge_anchor" not in counts
    assert (
        specialist_contract_training_allowed_for_mode(MODEL_NATIVE_CONTRACT_MODE)
        is True
    )


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
    assert tuple(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS[:57]) == (
        FOUNDATION_STRUCTURE_FEATURE_NAMES
    )
    assert Counter(
        classify_entry_specialist_feature(name)
        for name in FOUNDATION_STRUCTURE_FEATURE_NAMES
    ) == {
        "structure_swing_encoder": 19,
        "smc_liquidity_encoder": 5,
        "vol_compression_encoder": 5,
        "session_regime_encoder": 28,
    }


def test_active_context_contract_always_contains_full_regime_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GX1_REGIME_V4", "0")
    monkeypatch.delenv("GX1_TRUTH_MODE", raising=False)
    monkeypatch.delenv("GX1_RUN_MODE", raising=False)

    context = model_native_context_contract_metadata()
    # V30 (2026-08-13): 164 = 142 + H4_range_compression_ratio (package 1)
    # + 9 adopted swing V29 ctx fields + 3 momentum-G3 RSI canon scalars
    # (package 2) + 3 quote/spread-dynamics fields (package 4) + the 6
    # emission-only swing additions of package 8A (two missing run counters,
    # two level-intact flags, two normalized swing ages).
    assert len(MODEL_NATIVE_CTX_CONT_FIELDS) == MODEL_NATIVE_CTX_CONT_DIM == 164
    assert len(MODEL_NATIVE_CTX_CAT_FIELDS) == MODEL_NATIVE_CTX_CAT_DIM == 5
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
    victim = "trend.ema_mtf_score"
    replacement = "trend.ema_ranked_replacement_fixture"
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
        "trend_ema_smart_layer"
    ] -= 1

    with pytest.raises(RuntimeError, match="mandatory_full_stack metadata mismatch"):
        require_model_native_signal_contract(contract, context="STALE")


def test_manifest_rejects_stale_top_level_mandatory_family_metadata() -> None:
    manifest = _native_manifest()
    manifest["mandatory_full_stack"]["family_features"]["trend_ema_smart_layer"].append(
        "trend.ema_unknown_stale_member"
    )

    with pytest.raises(RuntimeError, match="MANDATORY_FULL_STACK_METADATA_STALE"):
        require_model_native_manifest(manifest, context="STALE")


def test_model_native_transformer_direction_is_direct_and_anchor_free() -> None:
    torch.manual_seed(7)
    model = _build_unit_test_entry_v10_ctx_hybrid_transformer(
        **_exact_model_kwargs()
    ).eval()
    seq_x = torch.randn(2, 4, MODEL_NATIVE_SIGNAL_DIM)
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

    assert out["direction_logits"].shape == out["model_native_logits"].shape == (2, 3)
    assert torch.isfinite(out["direction_logits"]).all()
    assert torch.equal(out["direction_logits"], out["raw_direction_logits"])
    assert "model_native_evidence_fusion_active" not in out
    assert not any("hierarchical_direction" in key for key in out)
    assert "anchor_logits" not in out
    assert "delta_logits" not in out
    assert "anchor_gate" not in out
    assert "anchor_eps" not in model.state_dict()
    assert not hasattr(model, "head_anchor_gate")
    assert not hasattr(model.cfg, "direction_logit_mode")
    assert "head_position_size.weight" in model.state_dict()
    assert out["position_size_logit"].shape == (2, 1)
    assert torch.count_nonzero(model.head_direction.weight).item() > 0

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


def test_state_builder_exposes_only_model_native_513_symbols(tmp_path) -> None:
    selected = _selected_fields()
    contract = model_native_signal_contract_metadata(selected)
    fit_start = pd.Timestamp("2026-01-01", tz="UTC")
    fit_end = pd.Timestamp("2026-01-02", tz="UTC")
    reference = TrainRankReferenceV2(
        path=tmp_path / "unused-in-init.npz",
        sha256="0" * 64,
        sidecar_sha256="0" * 64,
        sidecar={},
        fit_start_utc=fit_start,
        fit_end_utc=fit_end,
        fit_row_count=1,
        atr_bps_sorted=np.asarray([10.0], dtype=np.float64),
        spread_bps_sorted=np.asarray([1.0], dtype=np.float64),
    )
    state_contract = state_module.ModelNativeStateContract(
        feature_history_start_utc=fit_start,
        rank_fit_start_utc=fit_start,
        rank_fit_end_utc=fit_end,
        rank_reference_npz=reference.path,
        rank_reference=reference,
    )

    builder = state_module.ModelNativeStateBuilder(
        ordered_signal_names=list(contract["fields"]),
        state_contract=state_contract,
        signal_contract=contract,
    )

    assert len(builder.ordered_signal_names) == MODEL_NATIVE_SIGNAL_DIM
    assert len(builder._ext_names) == MODEL_NATIVE_SELECTED_FEATURE_COUNT
    assert not hasattr(state_module, "Smart520StateBuilder")
    assert not hasattr(state_module, "Smart520StateContract")
    assert not hasattr(state_module, "SIGNAL_DIM_SMART520")
