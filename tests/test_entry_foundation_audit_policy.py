from __future__ import annotations

import copy
import hashlib
import json

import pytest

from gx1.contracts.entry_fitted_q_v1 import (
    ENTRY_FITTED_Q_ACTION_ORDER,
    ENTRY_FITTED_Q_TARGET_UNIT,
)
from gx1.contracts.entry_full_input_liveness_v1 import RARE_EVENT_MINIMUMS
from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_AUDIT_DATA_SPLITS,
    FOUNDATION_AUDIT_POLICY_SCHEMA_VERSION,
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

    # Identity: the published digest must be the canonical hash of the exact
    # published payload, so no binding can advertise a policy it does not
    # carry.  The previous hand-pinned literal duplicated a value no owner
    # declares and went stale twice in one day (v11 -> v13, then again when
    # the liveness owner's rare-event registry moved), so the drift tripwire
    # is carried by the explicit per-threshold assertions below plus the
    # derivation proofs, not by a restated digest.
    assert FOUNDATION_AUDIT_POLICY_SHA256 == hashlib.sha256(
        json.dumps(
            binding["foundation_audit_policy"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert binding["foundation_audit_policy_sha256"] == FOUNDATION_AUDIT_POLICY_SHA256
    assert binding["foundation_audit_policy"]["schema_version"] == (
        FOUNDATION_AUDIT_POLICY_SCHEMA_VERSION
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
    turning = smoke["turning_point_evidence"]
    assert turning["evaluation_horizon_bars"] == 12
    assert turning["min_near_turn_direction_precision"] == 0.98
    assert turning["min_near_turn_precision_wilson_lower"] == 0.90
    assert turning["min_near_turn_timing_precision"] == 0.80
    target_quality = binding["foundation_audit_policy"]["target_quality"]
    # The retired offline-RL per-horizon action-value target policy is replaced
    # by the fitted-Q declarations; the target is no longer serialized in the
    # dataset, so no fixed-horizon reward surface may reappear here.
    assert "offline_rl_target" not in target_quality
    assert "offline_rl_evidence" not in smoke
    assert target_quality["entry_q_action_order"] == list(
        ENTRY_FITTED_Q_ACTION_ORDER
    )
    assert target_quality["entry_q_unit"] == ENTRY_FITTED_Q_TARGET_UNIT
    assert target_quality["entry_q_target_source"] == (
        "frozen_exit_first_state_target_model"
    )
    assert target_quality["entry_q_serialized_in_dataset"] is False
    assert target_quality["static_direction_or_path_horizon_allowed"] is False
    # Direction authority stays the single raw-bps Q argmax: the smoke evidence
    # surface may never publish a second direction source.
    assert turning["direction_authority"] == (
        "unique_raw_entry_action_q_bps_argmax_only"
    )
    specialist = binding["foundation_audit_policy"]["specialist_liveness"]
    assert specialist["train_live_statuses"] == ["LIVE", "ALLOWED_RARE_EVENT"]
    # The rare-event floors are owned by the full-input liveness registry, not
    # restated here: the policy must expose exactly that owner's TRAIN floors
    # for the signal surface, and nothing else may inject a floor.
    assert specialist["rare_event_minimum_active_count"] == {
        field: int(minimums["train"])
        for (surface, field), minimums in sorted(RARE_EVENT_MINIMUMS.items())
        if surface == "signal" and "train" in minimums
    }
    assert specialist["rare_event_minimum_active_count"]
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
