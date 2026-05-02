from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_build_140_94_safe_core_adapter_input_mapping_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [
            Path("/tmp/BUILD_140_94_SAFE_CORE_ADAPTER_INPUT_MAPPING_V1_20260428T000000Z_LOCK"),
            Path("/tmp/HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1_20260428T085058Z_LOCK"),
        ]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_feature_denylist_blocks_leakage_fields() -> None:
    blocked = [
        "bad_label_v1",
        "tail_label_v1",
        "post_outcome_mfe_v1",
        "safe_recoverable_v1",
        "coverage_proxy_member_v1",
        "lane_selected_v1",
        "selected_by_artifact_v1",
        "candidate_uid_v1",
    ]
    with pytest.raises(RuntimeError, match="FORBIDDEN_SAFE_CORE_ADAPTER_INPUT_FEATURE"):
        gate.validate_no_forbidden_feature_names(blocked)


def test_allowed_adapter_inputs_are_clean() -> None:
    assert gate.validate_no_forbidden_feature_names(
        [
            "tail_repaired_r5_2_oof_candidate_score_v1",
            "asof_signal__r5_1_bad_score_v1",
            "asof_signal__v2_like_bad_tail_v1",
            "asof_low_support_missing_artifact_veto_v1",
            "asof_hard_safety_veto_set_v1",
        ]
    )


def test_reproducibility_requires_exact_hardened_safe_core() -> None:
    payload = {
        "selected_rows_v1": 89,
        "recovered_original_140_rows_v1": 86,
        "extra_rows_v1": 3,
        "bad_count_audit_only_v1": 86,
        "tail_count_audit_only_v1": 55,
        "precision_audit_only_v1": 0.9662921348314607,
        "safety_status_v1": "CLEAN",
        "unsafe_hits_v1": 0,
    }
    assert gate.validate_reproducibility(payload)
    payload["extra_rows_v1"] = 4
    with pytest.raises(RuntimeError, match="SAFE_CORE_INPUT_MAPPING_REPRODUCIBILITY_FAILED"):
        gate.validate_reproducibility(payload)


def test_mapping_dry_run_validator_requires_exact_match() -> None:
    payload = {
        "mapping_dry_run_status_v1": "EXACT_MATCH_WITH_CURRENT_AUDIT_VETO",
        "missed_hardened_safe_core_rows_v1": 0,
        "extra_selected_vs_hardened_safe_core_v1": 0,
    }
    assert gate.validate_mapping_dry_run(payload)
    payload["extra_selected_vs_hardened_safe_core_v1"] = 1
    with pytest.raises(RuntimeError, match="SAFE_CORE_MAPPING_DRY_RUN_EXTRA_ROWS"):
        gate.validate_mapping_dry_run(payload)


def test_no_forbidden_actions_guard_flags_side_effects() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"
    blocked = gate.validate_no_forbidden_actions(
        r6=True,
        adapter=True,
        package=True,
        freeze=True,
        promo=True,
        live=True,
        optuna=True,
    )
    assert blocked["status_v1"] == "FAIL"
    assert "R6_FORBIDDEN" in blocked["failures_v1"]
    assert "ADAPTER_BUILD_FORBIDDEN" in blocked["failures_v1"]
    assert "OPTUNA_FORBIDDEN" in blocked["failures_v1"]


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(
        "140_94_SAFE_CORE_INPUT_MAPPING_BLOCKED_BY_UNMAPPED_VETOES",
        "DEEPEN_140_94_SAFE_CORE_VETO_MAPPING_AUDIT_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("PROMOTE_NOW", "DEEPEN_140_94_SAFE_CORE_VETO_MAPPING_AUDIT_V1")
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status("140_94_SAFE_CORE_INPUT_MAPPING_BLOCKED_BY_UNMAPPED_VETOES", "RUN_R6_NOW")


def test_materializer_writes_required_outputs_and_blocks_adapter_build(tmp_path: Path) -> None:
    artifact_root = tmp_path / "BUILD_140_94_SAFE_CORE_ADAPTER_INPUT_MAPPING_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)
    assert summary["safe_core_rule_id_v1"] == gate.HARDENED_RECIPE_ID
    assert summary["selected_rows_v1"] == 89
    assert summary["recovered_original_140_rows_v1"] == 86
    assert summary["extra_rows_v1"] == 3
    assert summary["bad_tail_audit_only_v1"] == [86, 55]
    assert summary["safety_status_v1"] == "CLEAN"
    assert summary["simulated_adapter_dry_run_status_v1"] == "EXACT_MATCH_WITH_CURRENT_AUDIT_VETO"
    assert summary["adapter_build_can_start_next_v1"] is False
    assert summary["r6_run_v1"] is False
    assert summary["adapter_built_v1"] is False
    assert summary["package_built_v1"] is False
    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name
    go = json.loads((artifact_root / "build_140_94_safe_core_adapter_input_mapping_go_no_go_v1.json").read_text())
    assert go["status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert go["next_recommended_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert go["adapter_build_approved_next_v1"] is False
    blockers = json.loads((artifact_root / "build_140_94_safe_core_missing_input_and_blocker_audit_v1.json").read_text())
    assert blockers["summary_v1"]["primary_blocker_v1"] == "UNMAPPED_AUDIT_ONLY_HARD_SAFETY_VETO"
    contract = json.loads((artifact_root / "build_140_94_safe_core_adapter_input_contract_v1.json").read_text())
    assert any(
        row["blocker_reason_v1"] == "UNMAPPED_AUDIT_ONLY_SAFETY_VETO"
        for row in contract["rows_v1"]
    )
