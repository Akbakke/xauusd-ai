from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_hold_140_94_safe_core_adapter_until_deployable_veto_exists_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1_20260428T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_reproducibility_requires_exact_safe_core_and_blocker() -> None:
    payload = {
        "selected_rows_v1": 89,
        "recovered_original_140_rows_v1": 86,
        "extra_rows_v1": 3,
        "bad_count_audit_only_v1": 86,
        "tail_count_audit_only_v1": 55,
        "precision_audit_only_v1": 0.9662921348314607,
        "safety_status_v1": "CLEAN",
        "hard_safety_veto_status_v1": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
        "unsafe_extra_without_hard_veto_rows_v1": 1,
    }
    assert gate.validate_reproducibility(payload)
    payload["unsafe_extra_without_hard_veto_rows_v1"] = 0
    with pytest.raises(RuntimeError, match="SAFE_CORE_HOLD_REPRODUCTION_FAILED"):
        gate.validate_reproducibility(payload)


def test_blocker_contract_keeps_adapter_closed() -> None:
    payload = {
        "adapter_may_resume_now_v1": False,
        "required_conditions_v1": [
            {
                "condition_id_v1": "DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_EXISTS",
                "current_value_v1": False,
            },
            {"condition_id_v1": "VETO_STOPS_UNSAFE_EXTRA_ROW", "current_value_v1": True},
            {"condition_id_v1": "NO_ROW_IDENTITY_SHORTCUT", "current_value_v1": True},
            {"condition_id_v1": "NO_AUDIT_ONLY_LABELS_OR_HINDSIGHT", "current_value_v1": False},
            {"condition_id_v1": "CLEAN_SIMULATED_ADAPTER_DRY_RUN", "current_value_v1": False},
        ],
    }
    assert gate.validate_blocker_contract(payload)
    payload["adapter_may_resume_now_v1"] = True
    with pytest.raises(RuntimeError, match="ADAPTER_MUST_NOT_RESUME"):
        gate.validate_blocker_contract(payload)


def test_decision_record_blocks_adapter_r6_iql_and_preserves_diagnostic_tracks() -> None:
    payload = {
        "safe_core_preserved_as_best_current_adapter_candidate_v1": True,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "iql_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "missing_54_expansion_active_v1": False,
        "best_lane_185_139_comparator_only_v1": True,
        "plus45_diagnostic_only_v1": True,
    }
    assert gate.validate_decision_record(payload)
    payload["r6_allowed_v1"] = True
    with pytest.raises(RuntimeError, match="SAFE_CORE_HOLD_DECISION_RECORD_INVALID"):
        gate.validate_decision_record(payload)


def test_no_forbidden_actions_guard_includes_iql() -> None:
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
    )
    assert blocked["status_v1"] == "FAIL"
    assert "R6_FORBIDDEN" in blocked["failures_v1"]
    assert "ADAPTER_BUILD_FORBIDDEN" in blocked["failures_v1"]
    assert "IQL_FORBIDDEN" in blocked["failures_v1"]


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(
        "140_94_SAFE_CORE_ADAPTER_HELD_UNTIL_DEPLOYABLE_VETO",
        "DISCOVER_DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_FOR_140_94_SAFE_CORE_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("RUN_ADAPTER_NOW", gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(gate.FINAL_STATUS, "RUN_R6_NOW")


def test_materializer_writes_required_outputs_and_holds_all_downstream_work(tmp_path: Path) -> None:
    artifact_root = tmp_path / "HOLD_140_94_SAFE_CORE_ADAPTER_UNTIL_DEPLOYABLE_VETO_EXISTS_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)
    assert summary["safe_core_rule_id_v1"] == gate.SAFE_CORE_RULE_ID
    assert summary["selected_rows_v1"] == 89
    assert summary["recovered_original_140_rows_v1"] == 86
    assert summary["extra_rows_v1"] == 3
    assert summary["bad_tail_audit_only_v1"] == [86, 55]
    assert summary["safety_status_v1"] == "CLEAN"
    assert summary["hard_safety_veto_status_v1"] == "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE"
    assert summary["unsafe_extra_without_hard_veto_rows_v1"] == 1
    assert summary["adapter_held_v1"] is True
    assert summary["r6_held_v1"] is True
    assert summary["iql_held_v1"] is True
    assert summary["final_status_v1"] == gate.FINAL_STATUS
    assert summary["next_recommended_action_v1"] == gate.NEXT_ACTION
    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name

    go = json.loads(
        (artifact_root / "hold_140_94_safe_core_adapter_until_deployable_veto_exists_go_no_go_v1.json").read_text()
    )
    assert go["status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert go["next_recommended_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert go["adapter_build_allowed_v1"] is False
    assert go["r6_allowed_v1"] is False
    assert go["iql_allowed_v1"] is False

    decision = json.loads((artifact_root / "hold_140_94_safe_core_decision_record_v1.json").read_text())
    assert decision["best_lane_185_139_comparator_only_v1"] is True
    assert decision["plus45_diagnostic_only_v1"] is True
    assert decision["missing_54_expansion_active_v1"] is False
