from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_build_clean_as_of_safety_feature_layer_from_source_signals_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/BUILD_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_FROM_SOURCE_SIGNALS_V1_20260428T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_retention_tier_boundaries() -> None:
    assert gate.retention_tier(unsafe_row_blocked=True, good_rows_cut=5) == "GREEN"
    assert gate.retention_tier(unsafe_row_blocked=True, good_rows_cut=6) == "YELLOW"
    assert gate.retention_tier(unsafe_row_blocked=True, good_rows_cut=11) == "ORANGE"
    assert gate.retention_tier(unsafe_row_blocked=True, good_rows_cut=21) == "RED"
    assert gate.retention_tier(unsafe_row_blocked=False, good_rows_cut=0) == "BLOCKED"
    assert gate.retention_tier(unsafe_row_blocked=True, good_rows_cut=0, shortcut_or_leakage=True) == "BLOCKED"


def test_source_signal_inventory_blocks_blueprint_labels_membership_and_row_identity() -> None:
    rows = [
        {"signal_name_v1": "candidate_score_v1", "classification_v1": "AS_OF_SAFE_SOURCE_SIGNAL"},
        {"signal_name_v1": "HISTORICAL_V2_BLUEPRINT", "classification_v1": "BLOCKED_HISTORICAL_ARTIFACT_PROXY"},
        {"signal_name_v1": "bad_label_v1", "classification_v1": "BLOCKED_OUTCOME_OR_HINDSIGHT"},
        {"signal_name_v1": "tail_label_v1", "classification_v1": "BLOCKED_OUTCOME_OR_HINDSIGHT"},
        {"signal_name_v1": "unsafe_audit_v1", "classification_v1": "BLOCKED_OUTCOME_OR_HINDSIGHT"},
        {"signal_name_v1": "candidate_uid_v1", "classification_v1": "BLOCKED_ROW_IDENTITY_OR_ARTIFACT_SHORTCUT"},
        {"signal_name_v1": "is_185_139_teacher_v1", "classification_v1": "BLOCKED_MEMBERSHIP_PROXY"},
        {"signal_name_v1": "is_plus45_diagnostic_v1", "classification_v1": "BLOCKED_MEMBERSHIP_PROXY"},
    ]
    assert gate.validate_source_signal_inventory(rows)
    rows[1]["classification_v1"] = "AS_OF_SAFE_SOURCE_SIGNAL"
    with pytest.raises(RuntimeError, match="FAILED_TO_BLOCK_FORBIDDEN_FIELDS"):
        gate.validate_source_signal_inventory(rows)


def test_candidate_dry_runs_do_not_allow_green_or_blueprint_adapter_ready() -> None:
    rows = [
        {
            "recipe_name_v1": gate.BEST_CLEAN_RECIPE,
            "retention_class_v1": "ORANGE",
            "leakage_or_proxy_risk_v1": False,
            "unsafe_extra_row_blocked_v1": True,
            "uses_historical_v2_blueprint_v1": False,
            "adapter_ready_v1": False,
        },
        {
            "recipe_name_v1": "DIAGNOSTIC_BLUEPRINT_GUARD_REFERENCE_NOT_ALLOWED_V1",
            "retention_class_v1": "BLOCKED",
            "leakage_or_proxy_risk_v1": True,
            "unsafe_extra_row_blocked_v1": True,
            "uses_historical_v2_blueprint_v1": True,
            "adapter_ready_v1": False,
        },
    ]
    assert gate.validate_candidate_recipe_dry_runs(rows)
    rows[0]["retention_class_v1"] = "GREEN"
    with pytest.raises(RuntimeError, match="UNEXPECTED_GREEN_SOURCE_SIGNAL_CANDIDATE"):
        gate.validate_candidate_recipe_dry_runs(rows)


def test_no_forbidden_actions_guard() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"
    blocked = gate.validate_no_forbidden_actions(
        r6=True,
        adapter=True,
        iql=True,
        package=True,
        freeze=True,
        promo=True,
        live=True,
        optuna=True,
        broad_sweep=True,
    )
    assert blocked["status_v1"] == "FAIL"
    assert "R6_FORBIDDEN" in blocked["failures_v1"]
    assert "ADAPTER_BUILD_FORBIDDEN" in blocked["failures_v1"]
    assert "IQL_FORBIDDEN" in blocked["failures_v1"]
    assert "BROAD_SWEEP_FORBIDDEN" in blocked["failures_v1"]


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(gate.FINAL_STATUS, gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("OPEN_ADAPTER_ANYWAY", gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(gate.FINAL_STATUS, "RUN_R6_NOW")


def test_go_no_go_keeps_adapter_r6_iql_blocked() -> None:
    payload = {
        "status_v1": gate.FINAL_STATUS,
        "next_recommended_action_v1": gate.NEXT_ACTION,
        "historical_v2_blueprint_used_as_deployable_input_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "iql_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
    }
    assert gate.validate_go_no_go(payload)
    with pytest.raises(RuntimeError, match="GO_NO_GO_MUST_KEEP_FORBIDDEN_PATHS_BLOCKED"):
        gate.validate_go_no_go(dict(payload, adapter_build_allowed_v1=True))
    with pytest.raises(RuntimeError, match="HISTORICAL_BLUEPRINT_MUST_NOT_BE_DEPLOYABLE_INPUT"):
        gate.validate_go_no_go(dict(payload, historical_v2_blueprint_used_as_deployable_input_v1=True))


def test_materializer_writes_required_outputs_and_keeps_adapter_closed(tmp_path: Path) -> None:
    artifact_root = tmp_path / "BUILD_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_FROM_SOURCE_SIGNALS_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)

    assert summary["baseline_140_94_status_v1"] == "CURRENT_BEST_CAUSAL_BASELINE"
    assert summary["safe_core_rule_id_v1"] == gate.SAFE_CORE_RULE_ID
    assert summary["safe_core_selected_rows_v1"] == 89
    assert summary["safe_core_recovered_original_140_v1"] == 86
    assert summary["safe_core_extra_rows_v1"] == 3
    assert summary["safe_core_bad_tail_audit_only_v1"] == [86, 55]
    assert summary["best_candidate_v1"] == gate.BEST_CLEAN_RECIPE
    assert summary["best_candidate_retention_class_v1"] == "ORANGE"
    assert summary["unsafe_row_blocked_v1"] is True
    assert summary["safe_core_rows_cut_v1"] == 11
    assert summary["historical_v2_blueprint_used_as_deployable_input_v1"] is False
    assert summary["adapter_r6_iql_remain_blocked_v1"] is True
    assert summary["r6_run_v1"] is False
    assert summary["adapter_built_v1"] is False
    assert summary["iql_run_v1"] is False
    assert summary["final_status_v1"] == gate.FINAL_STATUS
    assert summary["next_recommended_action_v1"] == gate.NEXT_ACTION

    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name

    inventory = json.loads((artifact_root / "clean_as_of_safety_layer_source_signal_inventory_v1.json").read_text())
    by_name = {row["signal_name_v1"]: row for row in inventory["rows_v1"]}
    assert by_name["HISTORICAL_V2_BLUEPRINT"]["classification_v1"] == "BLOCKED_HISTORICAL_ARTIFACT_PROXY"
    assert by_name["candidate_uid_v1"]["classification_v1"] == "BLOCKED_ROW_IDENTITY_OR_ARTIFACT_SHORTCUT"

    dry = json.loads((artifact_root / "clean_as_of_safety_layer_candidate_recipe_dry_run_v1.json").read_text())
    best = next(row for row in dry["rows_v1"] if row["recipe_name_v1"] == gate.BEST_CLEAN_RECIPE)
    assert best["unsafe_extra_row_blocked_v1"] is True
    assert best["retention_class_v1"] == "ORANGE"
    assert best["adapter_ready_v1"] is False
    assert all(
        not row["adapter_ready_v1"]
        for row in dry["rows_v1"]
        if row["uses_historical_v2_blueprint_v1"] or row["leakage_or_proxy_risk_v1"]
    )

    go = json.loads(
        (artifact_root / "build_clean_as_of_safety_feature_layer_from_source_signals_go_no_go_v1.json").read_text()
    )
    assert go["status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert go["next_recommended_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert go["adapter_build_allowed_v1"] is False
    assert go["r6_allowed_v1"] is False
    assert go["iql_allowed_v1"] is False
    assert go["historical_v2_blueprint_used_as_deployable_input_v1"] is False
