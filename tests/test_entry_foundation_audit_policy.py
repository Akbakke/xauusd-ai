from __future__ import annotations

import copy

import pytest

from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_AUDIT_DATA_SPLITS,
    FOUNDATION_AUDIT_POLICY_SHA256,
    foundation_audit_policy_binding,
    foundation_audit_policy_enforcement,
    require_foundation_audit_policy_binding,
    require_foundation_audit_report_policy,
)
from gx1.scripts.audit_entry_foundation_features_v1 import (
    build_parser as feature_parser,
)
from gx1.scripts.audit_entry_foundation_targets_v1 import (
    build_parser as target_parser,
)
from gx1.scripts.audit_entry_specialist_feature_groups_v1 import (
    build_parser as specialist_parser,
)


def test_foundation_audit_policy_has_fixed_identity_and_full_binding() -> None:
    binding = foundation_audit_policy_binding()

    assert FOUNDATION_AUDIT_POLICY_SHA256 == (
        "0bce213331acc609f4b7289d18f30c05388f4e2378b82ceff69c100a0d26bd58"
    )
    assert binding["foundation_audit_policy"]["schema_version"] == (
        "entry_foundation_audit_policy_v2"
    )
    smoke = binding["foundation_audit_policy"]["smoke_edge_pockets"]
    assert smoke["wilson_confidence_level"] == 0.95
    assert smoke["wilson_z_score"] == 1.959963984540054
    assert smoke["min_trade_rows"] == 200
    assert smoke["min_prediction_rows_per_class"] == 100
    assert smoke["min_trade_direction_precision"] == 0.98
    assert smoke["min_class_precision"] == 0.95
    assert smoke["min_trade_precision_wilson_lower"] == 0.95
    assert smoke["min_class_precision_wilson_lower"] == 0.90
    assert smoke["min_context_trade_rows"] == 32
    assert smoke["min_context_trade_direction_precision"] == 0.95
    assert smoke["min_context_trade_precision_wilson_lower"] == 0.85
    assert binding["foundation_audit_policy"]["audit_data_splits"] == list(
        FOUNDATION_AUDIT_DATA_SPLITS
    )
    assert require_foundation_audit_policy_binding(
        binding,
        context="TEST",
    ) == binding

    report = {
        **binding,
        "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
        "foundation_audit_policy_enforcement": (
            foundation_audit_policy_enforcement("target")
        ),
    }
    assert require_foundation_audit_report_policy(
        report,
        audit_kind="target",
        context="TEST",
    )["foundation_audit_policy_enforcement"] == (
        foundation_audit_policy_enforcement("target")
    )


def test_foundation_audit_policy_rejects_threshold_or_hash_forgery() -> None:
    forged = copy.deepcopy(foundation_audit_policy_binding())
    forged["foundation_audit_policy"]["target_quality"][
        "max_majority_rate"
    ] = 1.0
    with pytest.raises(RuntimeError, match="POLICY_PAYLOAD_INVALID"):
        require_foundation_audit_policy_binding(forged, context="TEST")

    forged = copy.deepcopy(foundation_audit_policy_binding())
    forged["foundation_audit_policy_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="POLICY_SHA256_INVALID"):
        require_foundation_audit_policy_binding(forged, context="TEST")

    forged = {
        **foundation_audit_policy_binding(),
        "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
        "foundation_audit_policy_enforcement": (
            foundation_audit_policy_enforcement("feature")
        ),
    }
    forged["foundation_audit_policy_enforcement"]["policy_section"][
        "liveness_epsilon"
    ] = 0.0
    with pytest.raises(RuntimeError, match="POLICY_ENFORCEMENT_INVALID"):
        require_foundation_audit_report_policy(
            forged,
            audit_kind="feature",
            context="TEST",
        )


def test_audit_clis_expose_no_semantic_policy_overrides() -> None:
    feature_help = feature_parser().format_help()
    target_help = target_parser().format_help()
    specialist_help = specialist_parser().format_help()

    for retired in (
        "--data-splits",
        "--liveness-epsilon",
        "--near-constant-std",
        "--min-required-family-active-rate",
        "--min-required-objective-active-rate",
        "--min-required-source-active-rate",
        "--min-required-source-active-count",
        "--parquet-batch-size",
    ):
        assert retired not in feature_help
    assert "--data-splits" not in target_help
    assert "--max-majority-rate" not in target_help
    assert "--data-splits" not in specialist_help
    assert "--contract-mode" not in specialist_help
    for help_text in (feature_help, target_help, specialist_help):
        for split in FOUNDATION_AUDIT_DATA_SPLITS:
            assert f"--{split}-manifest-json" in help_text
            assert f"--{split}-manifest-sha256" in help_text
            assert f"--{split}-parquet-sha256" in help_text
