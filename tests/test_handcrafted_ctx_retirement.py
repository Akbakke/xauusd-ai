from __future__ import annotations

import inspect
from pathlib import Path

from gx1.contracts.entry_exit_feature_usefulness_v1 import (
    feature_usefulness_layout,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
    RETIRED_HANDCRAFTED_CTX_CONT_FIELDS,
    RETIRED_OPERATOR_CTX_CONT_COMPOSITE_FIELDS,
    model_native_context_contract_metadata,
    ordered_model_native_signal_fields,
)
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
    assert MODEL_NATIVE_CTX_CONT_DIM == 90
    assert MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT == 86
    assert MODEL_NATIVE_SELECTED_FEATURE_COUNT == 250
    assert MODEL_NATIVE_SIGNAL_DIM == 279
    assert retired.isdisjoint(MODEL_NATIVE_CTX_CONT_FIELDS)
    assert retired.isdisjoint(_ordered_signal_names())
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
