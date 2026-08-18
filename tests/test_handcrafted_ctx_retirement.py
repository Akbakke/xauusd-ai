from __future__ import annotations

import inspect
from pathlib import Path

from gx1.contracts.entry_exit_feature_usefulness_v1 import (
    feature_usefulness_layout,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT,
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
    RETIRED_HANDCRAFTED_CTX_CONT_FIELDS,
    RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS,
    model_native_context_contract_metadata,
    ordered_model_native_signal_fields,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
from gx1.scripts import augment_forward_outcome_v2 as group_a_owner
from gx1.scripts import build_entry_exit_m1_enriched_frame_v1 as native_builder
from gx1.scripts import materialize_entry_model_native_m5_source_v1 as m5_source
from tests.model_native_signal_support import (
    canonical_model_native_selected_fields,
)


ROOT = Path(__file__).resolve().parents[1]


def _ordered_signal_names() -> tuple[str, ...]:
    return ordered_model_native_signal_fields(
        canonical_model_native_selected_fields()
    )


def test_all_55_handcrafted_context_fields_are_retired_without_aliases() -> None:
    retired = set(RETIRED_HANDCRAFTED_CTX_CONT_FIELDS)

    assert len(retired) == 55
    # Dimensions are DERIVED from the owner, never restated: the surface moved
    # five times in two days and every restated count went stale (rule 13).
    assert MODEL_NATIVE_CTX_CONT_DIM == len(MODEL_NATIVE_CTX_CONT_FIELDS)
    assert MODEL_NATIVE_SELECTED_FEATURE_COUNT == len(
        canonical_model_native_selected_fields()
    )
    assert MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT == len(
        MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS
    )
    assert MODEL_NATIVE_SIGNAL_DIM == len(_ordered_signal_names())
    # A retired handcrafted field may not reappear anywhere the model reads:
    # not in ctx_cont, not in the ordered signal surface, not in the mandatory
    # or candidate pools, and not on any per-timeframe MTF lane.
    assert retired.isdisjoint(MODEL_NATIVE_CTX_CONT_FIELDS)
    assert retired.isdisjoint(_ordered_signal_names())
    assert retired.isdisjoint(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    assert retired.isdisjoint(MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS)
    assert retired.isdisjoint(MULTI_TF_PER_BAR_FEATURES_V4)
    assert tuple(
        model_native_context_contract_metadata()[
            "retired_handcrafted_ctx_cont_fields"
        ]
    ) == RETIRED_HANDCRAFTED_CTX_CONT_FIELDS


def test_all_14_operator_context_composites_are_retired_without_aliases() -> None:
    retired = set(RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS)

    assert len(retired) == 14
    assert len(MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS) == 14
    assert retired.isdisjoint(MODEL_NATIVE_CTX_CONT_FIELDS)
    assert retired.isdisjoint(_ordered_signal_names())
    assert tuple(
        model_native_context_contract_metadata()[
            "retired_operator_ctx_cont_composite_fields"
        ]
    ) == RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS


def test_active_python_has_no_retired_field_or_owner_consumer() -> None:
    allowed_retirement_owner = (
        ROOT / "gx1/contracts/entry_model_native_signal_v1.py"
    ).resolve()
    offenders: list[str] = []
    forbidden_owner_symbols = (
        "MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS",
        "MODEL_NATIVE_CTX_CONT_ENTRY_SMART_DERIVED_FIELDS",
        "ENTRY_SMART_CTX_FEATURE_NAMES",
        "add_entry_smart_context_features",
        "attach_group_a_dip_struct_ctx_columns",
    )
    for path in sorted((ROOT / "gx1").rglob("*.py")):
        if path.resolve() == allowed_retirement_owner:
            continue
        source = path.read_text(encoding="utf-8")
        for field in RETIRED_HANDCRAFTED_CTX_CONT_FIELDS:
            if field in source:
                offenders.append(f"{path.relative_to(ROOT)}:{field}")
        for field in RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS:
            if field in source:
                offenders.append(f"{path.relative_to(ROOT)}:{field}")
        for symbol in forbidden_owner_symbols:
            if symbol in source:
                offenders.append(f"{path.relative_to(ROOT)}:{symbol}")
    assert offenders == []
    assert not (ROOT / "gx1/features/entry_smart_context.py").exists()


def test_native_m1_m5_and_group_a_producers_use_current_exact_context() -> None:
    retired = set(RETIRED_HANDCRAFTED_CTX_CONT_FIELDS) | set(
        RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS
    )

    assert set(MODEL_NATIVE_CTX_CONT_FIELDS).issubset(native_builder.OUTPUT_COLUMNS)
    assert retired.isdisjoint(native_builder.OUTPUT_COLUMNS)
    assert set(m5_source.RANKER_OWNED_DERIVATIONS) == set(
        MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS
    )
    assert retired.isdisjoint(m5_source.ENRICHED_COLUMNS)
    assert retired.isdisjoint(m5_source.OUTPUT_COLUMNS)
    owner_source = inspect.getsource(group_a_owner.attach_group_a_ctx_columns)
    parallel_source = inspect.getsource(
        group_a_owner.attach_group_a_ctx_columns_parallel
    )
    assert "dip" not in owner_source.lower()
    assert "dip" not in parallel_source.lower()


def test_usefulness_layout_covers_only_current_context_without_selection() -> None:
    layout = feature_usefulness_layout(_ordered_signal_names())

    for task in ("entry", "exit"):
        coverage = layout["tasks"][task]["coverage_counts"]
        assert coverage["local_signal"] == MODEL_NATIVE_SIGNAL_DIM
        assert coverage["ctx_cont"] == MODEL_NATIVE_CTX_CONT_DIM
        serialized = repr(layout["tasks"][task])
        assert all(
            field not in serialized
            for field in (
                RETIRED_HANDCRAFTED_CTX_CONT_FIELDS
                + RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS
            )
        )


def test_v31_ctx_cont_retirements_additions_and_renames_are_exact() -> None:
    """V30 wave 2 moved the ctx_cont surface; every move is checked here.

    Non-vacuous against the pre-change contract in both directions: each retired
    name is asserted ABSENT (it was present), each added name is asserted
    PRESENT (it was absent), and each renamed name is asserted present under the
    new spelling and absent under the old one.
    """

    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_REGIME_FIELDS,
        MODEL_NATIVE_CTX_CONT_SESSION_FIELDS,
        MODEL_NATIVE_CTX_CONT_SWING_FIELDS,
        MODEL_NATIVE_CTX_CONT_V2_EXTENSION_FIELDS,
        RETIRED_EXACTLY_RECOVERABLE_CTX_CONT_FIELDS,
    )
    from gx1.features.htf_features import (
        MODEL_NATIVE_CONTEXT_MTF_TIMEFRAMES,
        MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4,
    )
    from gx1.features.swing_structure_v1 import (
        SWING_FEATURE_NAMES_V1,
        SWING_V29_ADDITION_NAMES_V1,
    )

    ctx = set(MODEL_NATIVE_CTX_CONT_FIELDS)

    # 1. The fourteen SWING_V29 duplicates are gone from ctx_cont and every one
    #    of them still exists on the mandatory surface, so rule 4 holds by
    #    construction rather than by claim.
    assert MODEL_NATIVE_CTX_CONT_SWING_FIELDS == tuple(SWING_FEATURE_NAMES_V1)
    mandatory = set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    for name in SWING_V29_ADDITION_NAMES_V1:
        assert name not in ctx, name
        assert name in mandatory, name
    # retracement_from_last_impulse was proposed for retirement and REFUSED
    # (its recovery is an unbounded quotient); it must still be a ctx field.
    assert "retracement_from_last_impulse" in ctx

    # 2. The three exactly-recoverable context values are gone, and the fields
    #    that recover them are still there.
    for name in ("is_ASIA", "minutes_to_next_session_boundary", "dow_cos"):
        assert name not in ctx, name
    assert MODEL_NATIVE_CTX_CONT_SESSION_FIELDS == (
        "minutes_since_session_open",
        "session_change_flag",
    )
    for witness in ("hour_sin", "hour_cos", "dow_sin", "minutes_since_session_open"):
        assert witness in ctx, witness

    # 3. Every retired name is registered, and the registry guard is live.
    assert set(RETIRED_EXACTLY_RECOVERABLE_CTX_CONT_FIELDS) == (
        set(SWING_V29_ADDITION_NAMES_V1)
        | {"is_ASIA", "minutes_to_next_session_boundary", "dow_cos"}
    )
    assert not (ctx & set(RETIRED_EXACTLY_RECOVERABLE_CTX_CONT_FIELDS))
    metadata = model_native_context_contract_metadata()
    assert metadata["retired_exactly_recoverable_ctx_cont_fields"] == list(
        RETIRED_EXACTLY_RECOVERABLE_CTX_CONT_FIELDS
    )
    # The mandatory swing twins must NOT be reachable through the tuple that
    # publishes "handcrafted" retirements, or a later session reads them as
    # deleted everywhere.
    assert not (
        set(RETIRED_HANDCRAFTED_CTX_CONT_FIELDS) & set(SWING_V29_ADDITION_NAMES_V1)
    )

    # 4. The five ema-stack companions are present, one per declared context
    #    timeframe, beside the trend ages they make readable.
    for tf in MODEL_NATIVE_CONTEXT_MTF_TIMEFRAMES:
        assert f"{tf}_ema_stack_aligned_v2" in ctx, tf
        assert f"{tf}_trend_state_age_bars_v2" in ctx, tf
    assert MODEL_NATIVE_CTX_CONT_REGIME_FIELDS[-1] == "d1_dist_change_1bar_atr_v4"

    # 5. The four unit/name renames, in both owners, with position preserved.
    for old, new in (
        ("_v1h1_atr", "_v1h1_atr_bps"),
        ("_v1h4_atr", "_v1h4_atr_bps"),
        ("d1_atr14_canon_v2", "d1_atr14_bps_canon_v2"),
        ("d1_pct_change_5_canon_v2", "d1_change_5_bps_canon_v2"),
    ):
        assert old not in ctx, old
        assert new in ctx, new
        assert old not in MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4, old
        assert new in MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4, new
    # The ctx_cont V2 block must keep the scalar owner's tuple order.
    shared = [
        name
        for name in MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4
        if name in set(MODEL_NATIVE_CTX_CONT_V2_EXTENSION_FIELDS)
    ]
    assert shared == [
        name
        for name in MODEL_NATIVE_CTX_CONT_V2_EXTENSION_FIELDS
        if name in set(MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4)
    ]


def test_v31_m5_source_projects_the_regime_companions_it_cross_checks() -> None:
    """The compact projection must carry BOTH regime fragments on all five lanes.

    Before the repair the projection was a single ``trend_state_age_bars`` entry
    filtered by name suffix, so the five new ``{tf}_ema_stack_aligned_v2``
    columns would have been copied without ever reaching the enriched-vs-projected
    cross-check (or raised M5_SOURCE_OUTPUT_FIELD_UNRESOLVED). This asserts the
    projection, the derived field list and the output schema together.
    """

    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_REGIME_FIELDS,
    )
    from gx1.features.htf_features import MODEL_NATIVE_CONTEXT_MTF_TIMEFRAMES

    outputs = {output for output, _source in m5_source.REGIME_COMPACT_PROJECTION}
    assert outputs == {"trend_state_age_bars", "ema_stack_aligned"}
    assert set(m5_source.REGIME_PROJECTED_FIELDS) == {
        f"{tf}_{output}_v2"
        for tf in MODEL_NATIVE_CONTEXT_MTF_TIMEFRAMES
        for output in outputs
    }
    # The list is derived from the contract, not from a name-suffix filter.
    assert set(m5_source.REGIME_PROJECTED_FIELDS) == (
        set(MODEL_NATIVE_CTX_CONT_REGIME_FIELDS) - {"d1_dist_change_1bar_atr_v4"}
    )
    for name in m5_source.REGIME_PROJECTED_FIELDS:
        assert name in m5_source.OUTPUT_COLUMNS, name
        assert name in m5_source.SOURCE_OWNED_FIELDS, name
    assert not hasattr(m5_source, "TREND_AGE_PROJECTED_FIELDS")
    assert not hasattr(m5_source, "TREND_AGE_COMPACT_PROJECTION")
