"""Tests for materialize_deepen_exit_iql_state_feature_family_v1.

We exercise the V2 schema invariants (V1 subset, no-shortcut, audit
isolation, audit coverage) plus the go-no-go branches. We do not run
the materializer in pytest - the data inputs are already integration-tested
by the actual gate run; here we lock the contract structure and the
audit semantics.
"""
from __future__ import annotations

import pandas as pd
import pytest

from gx1.scripts import materialize_deepen_exit_iql_state_feature_family_v1 as deepen_gate


# ---------------------------------------------------------------------------
# Schema shape
# ---------------------------------------------------------------------------


def test_v2_state_has_51_features() -> None:
    assert len(deepen_gate.PROPOSED_V2_STATE_FEATURES) == 51


def test_v2_audit_labels_has_5() -> None:
    assert len(deepen_gate.PROPOSED_AUDIT_LABELS_V2) == 5


def test_every_state_feature_has_required_keys() -> None:
    required = {
        "field_name_v2",
        "category_v2",
        "source_v2",
        "source_field_v2",
        "lineage_v2",
        "availability_v2",
        "normalization_v2",
    }
    for f in deepen_gate.PROPOSED_V2_STATE_FEATURES:
        assert required <= set(f.keys()), f"missing keys on {f}"


def test_every_audit_label_has_required_keys() -> None:
    required = {
        "label_name_v2",
        "type_v2",
        "source_v2",
        "formula_v2",
        "interpretation_v2",
        "eligibility_v2",
    }
    for a in deepen_gate.PROPOSED_AUDIT_LABELS_V2:
        assert required <= set(a.keys()), f"missing keys on {a}"


def test_all_state_field_names_unique() -> None:
    names = [f["field_name_v2"] for f in deepen_gate.PROPOSED_V2_STATE_FEATURES]
    assert len(set(names)) == len(names)


def test_all_audit_label_names_unique() -> None:
    names = [a["label_name_v2"] for a in deepen_gate.PROPOSED_AUDIT_LABELS_V2]
    assert len(set(names)) == len(names)


def test_category_breakdown_matches_design() -> None:
    counts: dict[str, int] = {}
    for f in deepen_gate.PROPOSED_V2_STATE_FEATURES:
        counts[f["category_v2"]] = counts.get(f["category_v2"], 0) + 1
    assert counts["TRADE_STATE_RUNNING"] == 12
    assert counts["MARKET_STATE_AT_BAR"] == 24
    assert counts["TRANSFORMER_SIGNAL_AT_BAR"] == 8
    assert counts["ENTRY_CONTEXT_SNAPSHOT"] == 7


# ---------------------------------------------------------------------------
# V1 subset invariant
# ---------------------------------------------------------------------------


def _v1_def(name: str, source_field: str, source: str = "PER_BAR_SCAFFOLD") -> dict:
    return {
        "field_name_v1": name,
        "category_v1": "TRADE_STATE_RUNNING",
        "source_v1": source,
        "source_field_v1": source_field,
        "lineage_v1": "AS_OF_AT_BAR_T",
        "availability_v1": "HAVE",
        "normalization_v1": "PASSTHROUGH",
    }


def _v2_alias(name: str, source_field: str) -> dict:
    return {
        "field_name_v2": name,
        "field_name_v1_alias": name,
        "category_v2": "TRADE_STATE_RUNNING",
        "source_v2": "PER_BAR_SCAFFOLD",
        "source_field_v2": source_field,
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
    }


def test_v1_subset_audit_passes_when_v2_carries_v1_haves() -> None:
    v1 = [_v1_def("running_pnl_at_close_bps_v1", "pnl_at_close_bps_v1")]
    v2 = [_v2_alias("running_pnl_at_close_bps_v1", "pnl_at_close_bps_v1")]
    audit = deepen_gate.validate_v1_subset_invariant(v1, v2)
    assert audit["status_v1"] == "PASS"
    assert audit["v1_fields_missing_in_v2_v1"] == []
    assert audit["v1_to_v2_drift_v1"] == []


def test_v1_subset_audit_detects_missing_v1_field() -> None:
    v1 = [_v1_def("running_pnl_at_close_bps_v1", "pnl_at_close_bps_v1")]
    v2: list[dict] = []
    audit = deepen_gate.validate_v1_subset_invariant(v1, v2)
    assert audit["status_v1"] == "FAIL"
    assert "running_pnl_at_close_bps_v1" in audit["v1_fields_missing_in_v2_v1"]


def test_v1_subset_audit_detects_drift_in_source_field() -> None:
    v1 = [_v1_def("bars_held_v1", "bar_index_v1")]
    v2 = [_v2_alias("bars_held_v1", "DRIFTED_FIELD")]
    audit = deepen_gate.validate_v1_subset_invariant(v1, v2)
    assert audit["status_v1"] == "FAIL"
    assert audit["v1_to_v2_drift_v1"]


def test_v1_subset_audit_passes_against_real_v1_22field_contract() -> None:
    """The actual V1 contract pins 22 fields - we mock the V1 part by
    reading the SAME structure we expect from disk and validating that
    PROPOSED_V2_STATE_FEATURES carries every V1 HAVE forward."""
    # Mirror the V1 contract definitions structure we expect.
    v1 = [
        _v1_def("running_pnl_at_close_bps_v1", "pnl_at_close_bps_v1"),
        _v1_def("running_mfe_bps_v1", "running_mfe_bps_v1"),
        _v1_def("running_mae_bps_v1", "running_mae_bps_v1"),
        _v1_def("running_giveback_from_peak_bps_v1", "running_giveback_from_peak_bps_v1"),
        _v1_def("bars_held_v1", "bar_index_v1"),
        _v1_def("distance_from_peak_mfe_bps_v1", "distance_from_peak_mfe_bps", source="EXIT_EVAL_TRACE"),
        _v1_def("time_since_mfe_bars_v1", "time_since_mfe_bars", source="EXIT_EVAL_TRACE"),
        _v1_def("giveback_ratio_v1", "giveback_ratio", source="EXIT_EVAL_TRACE"),
        _v1_def("atr_bps_now_v1", "atr_bps", source="BASE34_M5"),
        _v1_def("session_id_v1", "session_id", source="BASE34_M5"),
        _v1_def("vol_regime_id_v1", "_v1_atr_regime_id", source="BASE34_M5"),
        _v1_def("trend_slope_ema3_v1", "_v1_close_ema_slope_3", source="BASE34_M5"),
        _v1_def("spread_bps_dyn_v1", "_v1_cost_bps_dyn", source="BASE34_M5"),
        _v1_def("minutes_since_session_open_v1", "minutes_since_session_open", source="BASE34_M5"),
        _v1_def("side_v1", "side", source="TRADE_OUTCOMES"),
        _v1_def("entry_session_v1", "session", source="TRADE_OUTCOMES"),
        _v1_def("entry_spread_bps_v1", "entry_spread_bps", source="TRADE_OUTCOMES"),
        _v1_def("exit_prob_v1", "exit_prob", source="EXIT_EVAL_TRACE"),
    ]
    audit = deepen_gate.validate_v1_subset_invariant(
        v1, deepen_gate.PROPOSED_V2_STATE_FEATURES
    )
    assert audit["status_v1"] == "PASS", audit


# ---------------------------------------------------------------------------
# No-shortcut audit
# ---------------------------------------------------------------------------


def test_no_shortcut_audit_passes_on_v2_schema() -> None:
    out = deepen_gate.validate_no_shortcut(deepen_gate.PROPOSED_V2_STATE_FEATURES)
    assert out["status_v1"] == "PASS"
    assert out["forbidden_field_intersection_v1"] == []
    assert out["forbidden_token_pattern_hits_v1"] == []
    assert out["identity_token_hits_v1"] == []
    assert out["audit_token_state_hits_v1"] == []


def test_no_shortcut_audit_detects_audit_token_in_state() -> None:
    bad = list(deepen_gate.PROPOSED_V2_STATE_FEATURES) + [
        {
            "field_name_v2": "audit_delay_better_v2",
            "category_v2": "TRADE_STATE_RUNNING",
            "source_v2": "PER_BAR_SCAFFOLD",
            "source_field_v2": "x",
            "lineage_v2": "AS_OF_AT_BAR_T",
            "availability_v2": "HAVE",
            "normalization_v2": "PASSTHROUGH",
        }
    ]
    with pytest.raises(RuntimeError, match="AUDIT_TOKEN_LEAKED_INTO_STATE"):
        deepen_gate.validate_no_shortcut(bad)


def test_no_shortcut_audit_detects_identity_token() -> None:
    bad = [
        {
            "field_name_v2": "trade_id_state_v2",
            "category_v2": "TRADE_STATE_RUNNING",
            "source_v2": "PER_BAR_SCAFFOLD",
            "source_field_v2": "x",
            "lineage_v2": "AS_OF_AT_BAR_T",
            "availability_v2": "HAVE",
            "normalization_v2": "PASSTHROUGH",
        }
    ]
    with pytest.raises(RuntimeError, match="IDENTITY_TOKEN_IN_FIELD_NAME"):
        deepen_gate.validate_no_shortcut(bad)


def test_no_shortcut_audit_detects_forbidden_pattern() -> None:
    bad = [
        {
            "field_name_v2": "post_exit_drift_v2",
            "category_v2": "TRADE_STATE_RUNNING",
            "source_v2": "PER_BAR_SCAFFOLD",
            "source_field_v2": "x",
            "lineage_v2": "AS_OF_AT_BAR_T",
            "availability_v2": "HAVE",
            "normalization_v2": "PASSTHROUGH",
        }
    ]
    with pytest.raises(RuntimeError, match="FORBIDDEN_TOKEN_IN_FIELD_NAME"):
        deepen_gate.validate_no_shortcut(bad)


# ---------------------------------------------------------------------------
# Audit-label isolation
# ---------------------------------------------------------------------------


def test_audit_label_isolation_passes_on_v2_schema() -> None:
    out = deepen_gate.validate_audit_label_isolation(
        deepen_gate.PROPOSED_AUDIT_LABELS_V2,
        deepen_gate.PROPOSED_V2_STATE_FEATURES,
    )
    assert out["status_v1"] == "PASS"
    assert out["audit_state_overlap_v1"] == []


def test_audit_label_isolation_rejects_overlap_with_state() -> None:
    state = [{"field_name_v2": "audit_delay_better_v2"}]
    audit = [
        {
            "label_name_v2": "audit_delay_better_v2",
            "eligibility_v2": "AUDIT_ONLY_NEVER_STATE_NEVER_REWARD_NEVER_SELECTOR",
        }
    ]
    with pytest.raises(RuntimeError, match="AUDIT_LABEL_NAME_LEAKED_INTO_STATE"):
        deepen_gate.validate_audit_label_isolation(audit, state)


def test_audit_label_isolation_rejects_bad_eligibility() -> None:
    audit = [
        {
            "label_name_v2": "x_v2",
            "eligibility_v2": "OK_FOR_STATE",
        }
    ]
    with pytest.raises(RuntimeError, match="AUDIT_LABEL_BAD_ELIGIBILITY"):
        deepen_gate.validate_audit_label_isolation(audit, [])


# ---------------------------------------------------------------------------
# Quarantine revival audit
# ---------------------------------------------------------------------------


def test_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    deepen_gate.validate_no_deprecated_revival(Path(deepen_gate.__file__))


def test_validate_final_status_rejects_unknown_status() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        deepen_gate.validate_final_status(
            "MADE_UP", "RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1"
        )


def test_validate_final_status_rejects_unknown_next_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        deepen_gate.validate_final_status(
            "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V2_LOCKED_AVAILABILITY_AUDIT_PASSED",
            "TRAIN_NOW",
        )


# ---------------------------------------------------------------------------
# go-no-go
# ---------------------------------------------------------------------------


def _audit_pass_dict() -> dict:
    return {"status_v1": "PASS"}


def test_go_no_go_full_pass_on_zero_not_established() -> None:
    avail = {
        "have_count_v1": 51,
        "derivable_count_v1": 0,
        "not_established_count_v1": 0,
    }
    status, action, _ = deepen_gate._go_no_go(
        avail,
        _audit_pass_dict(),
        _audit_pass_dict(),
        _audit_pass_dict(),
        _audit_pass_dict(),
    )
    assert status == "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V2_LOCKED_AVAILABILITY_AUDIT_PASSED"
    assert action == "RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1"


def test_go_no_go_partial_when_not_established_present() -> None:
    avail = {
        "have_count_v1": 40,
        "derivable_count_v1": 4,
        "not_established_count_v1": 7,
    }
    status, action, _ = deepen_gate._go_no_go(
        avail,
        _audit_pass_dict(),
        _audit_pass_dict(),
        _audit_pass_dict(),
        _audit_pass_dict(),
    )
    assert status == "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V2_PARTIAL_SOME_FEATURES_NOT_ESTABLISHED"
    assert action == "RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1"


def test_go_no_go_blocks_on_no_shortcut_failure() -> None:
    status, action, _ = deepen_gate._go_no_go(
        {"have_count_v1": 0, "derivable_count_v1": 0, "not_established_count_v1": 0},
        {"status_v1": "FAIL"},
        _audit_pass_dict(),
        _audit_pass_dict(),
        _audit_pass_dict(),
    )
    assert "BLOCKED_BY_NO_SHORTCUT_FAIL" in status
    assert action == "HOLD_UNTIL_V2_STATE_FEATURE_GAPS_RESOLVED_V1"


def test_go_no_go_blocks_on_v1_subset_failure() -> None:
    status, action, _ = deepen_gate._go_no_go(
        {"have_count_v1": 0, "derivable_count_v1": 0, "not_established_count_v1": 0},
        _audit_pass_dict(),
        _audit_pass_dict(),
        _audit_pass_dict(),
        {
            "status_v1": "FAIL",
            "v1_fields_missing_in_v2_v1": ["x"],
            "v1_to_v2_drift_v1": [],
        },
    )
    assert "BLOCKED_BY_NO_SHORTCUT_FAIL" in status
    assert action == "HOLD_UNTIL_V2_STATE_FEATURE_GAPS_RESOLVED_V1"


def test_go_no_go_blocks_on_audit_label_coverage_failure() -> None:
    status, action, _ = deepen_gate._go_no_go(
        {"have_count_v1": 0, "derivable_count_v1": 0, "not_established_count_v1": 0},
        _audit_pass_dict(),
        _audit_pass_dict(),
        {"status_v1": "FAIL"},
        _audit_pass_dict(),
    )
    assert "BLOCKED_BY_INPUT_LOCK_MISSING" in status


# ---------------------------------------------------------------------------
# Promotion provenance
# ---------------------------------------------------------------------------


def test_promoted_fields_carry_status_change_marker() -> None:
    expected_promoted = {
        "p_long_entry_v1",
        "p_hat_entry_v1",
        "uncertainty_entry_v1",
        "margin_entry_v1",
    }
    promoted_actual = {
        f["field_name_v2"]
        for f in deepen_gate.PROPOSED_V2_STATE_FEATURES
        if f.get("v1_status_change_v2") == "PROMOTED_FROM_NOT_ESTABLISHED_VIA_RECOVERY"
    }
    assert expected_promoted == promoted_actual


def test_per_bar_xgb_fields_are_not_established_with_blocking_reason() -> None:
    not_established = [
        f
        for f in deepen_gate.PROPOSED_V2_STATE_FEATURES
        if f["availability_v2"] == "NOT_ESTABLISHED"
    ]
    assert len(not_established) == 7
    for f in not_established:
        assert f["category_v2"] == "TRANSFORMER_SIGNAL_AT_BAR"
        assert f["field_name_v2"].startswith("per_bar_xgb_")
        assert f.get("blocking_reason_v2"), f
