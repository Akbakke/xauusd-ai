from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_exit_hold_exit_now_mdp_reward_contract_v1 as gate


def test_explicit_artifact_roots_reject_latest() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_V1_20260429T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])


def test_no_forbidden_actions_default_pass() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"
    blocked = gate.validate_no_forbidden_actions(adapter=True, r6=True)
    assert blocked["status_v1"] == "FAIL"


def test_validate_final_status_only_allowed_pairs() -> None:
    assert gate.validate_final_status(
        "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_LOCKED_PRE_TRAIN_DEPENDENCIES_ENUMERATED",
        "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status(
            "ARBITRARY",
            "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1",
        )
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_LOCKED_PRE_TRAIN_DEPENDENCIES_ENUMERATED",
            "TRAIN_EXIT_BANDIT_NOW_V1",
        )


def test_validate_no_deprecated_revival(tmp_path: Path) -> None:
    bad = tmp_path / "imports_quarantine.py"
    bad.write_text(
        "from gx1.quarantine._DEPRECATED_SCRIPTS_20260219 import x\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN"):
        gate.validate_no_deprecated_revival(bad)
    good = tmp_path / "clean.py"
    good.write_text("import pandas\n", encoding="utf-8")
    assert gate.validate_no_deprecated_revival(good)
    assert gate.validate_no_deprecated_revival(Path(gate.__file__))


def test_action_set_locked_to_binary_hold_exit_now() -> None:
    assert gate.validate_action_set(gate.ACTION_SET_V1)
    with pytest.raises(RuntimeError, match="ACTION_SET_MUST_BE_BINARY"):
        gate.validate_action_set({"HOLD": {"action_id_v1": 0}, "PARTIAL_EXIT": {"action_id_v1": 1}})
    with pytest.raises(RuntimeError, match="ACTION_IDS_MUST_BE_FIXED"):
        gate.validate_action_set(
            {
                "HOLD": {"action_id_v1": 1, "meaning_v1": "x", "transition_v1": "x"},
                "EXIT_NOW": {"action_id_v1": 0, "meaning_v1": "x", "transition_v1": "x"},
            }
        )


def test_hold_reward_lock_zero_immediate() -> None:
    assert gate.validate_hold_reward_lock(gate.HOLD_REWARD_LOCK_V1)
    with pytest.raises(RuntimeError, match="HOLD_IMMEDIATE_REWARD_MUST_BE_ZERO"):
        bad = dict(gate.HOLD_REWARD_LOCK_V1)
        bad["hold_immediate_reward_v1"] = 0.5
        gate.validate_hold_reward_lock(bad)
    with pytest.raises(RuntimeError, match="HOLD_REWARD_SCHEME_NOT_LOCKED"):
        bad2 = dict(gate.HOLD_REWARD_LOCK_V1)
        bad2["scheme_v1"] = "PER_BAR_MARK_TO_MARKET_DELTA"
        gate.validate_hold_reward_lock(bad2)


def test_terminal_reward_variants_complete_and_runner_audit_only() -> None:
    assert gate.validate_terminal_reward_variants(gate.TERMINAL_REWARD_VARIANTS_V1)
    bad = [
        v for v in gate.TERMINAL_REWARD_VARIANTS_V1 if v["reward_id_v1"] != "REALIZED_PNL_REWARD"
    ]
    with pytest.raises(RuntimeError, match="TERMINAL_REWARD_VARIANT_SET_MISMATCH"):
        gate.validate_terminal_reward_variants(bad)
    bad2 = [dict(v) for v in gate.TERMINAL_REWARD_VARIANTS_V1]
    for v in bad2:
        if v["reward_id_v1"] == "RUNNER_DAMAGE_PENALTY":
            v["applies_to_action_v1"] = "EXIT_NOW_OR_FORCED_TERMINAL"
    with pytest.raises(RuntimeError, match="RUNNER_DAMAGE_PENALTY_MUST_BE_AUDIT_ONLY"):
        gate.validate_terminal_reward_variants(bad2)


def test_discount_lock_in_valid_range() -> None:
    assert gate.validate_discount_lock(gate.DISCOUNT_LOCK_V1)
    with pytest.raises(RuntimeError, match="DEFAULT_GAMMA_OUTSIDE_VALID_RANGE"):
        bad = dict(gate.DISCOUNT_LOCK_V1)
        bad["default_gamma_v1"] = 1.5
        gate.validate_discount_lock(bad)
    with pytest.raises(RuntimeError, match="SENSITIVITY_GAMMA_VALUE_OUTSIDE_VALID_RANGE"):
        bad2 = dict(gate.DISCOUNT_LOCK_V1)
        bad2["sensitivity_range_v1"] = [0.99, 1.01]
        gate.validate_discount_lock(bad2)


def test_forbidden_state_fields_include_outcome_and_identity() -> None:
    assert gate.validate_forbidden_state_fields(gate.FORBIDDEN_STATE_FIELDS_V1)
    with pytest.raises(RuntimeError, match="FORBIDDEN_STATE_FIELDS_MISSING_REQUIRED"):
        gate.validate_forbidden_state_fields(["bar_count_v1"])


def test_action_support_requirement_blocks_training_until_augmentation() -> None:
    assert gate.validate_action_support_requirement(gate.ACTION_SUPPORT_REQUIREMENT_V1)
    with pytest.raises(RuntimeError, match="ACTION_SUPPORT_REQUIREMENT_MUST_BLOCK_TRAINING"):
        bad = dict(gate.ACTION_SUPPORT_REQUIREMENT_V1)
        bad["before_augmentation_iql_training_forbidden_v1"] = False
        gate.validate_action_support_requirement(bad)
    with pytest.raises(RuntimeError, match="ACTION_SUPPORT_REQUIREMENT_MUST_REQUIRE_AUGMENTATION"):
        bad2 = dict(gate.ACTION_SUPPORT_REQUIREMENT_V1)
        bad2["training_blocked_until_v1"] = [
            {"blocker_v1": "SOMETHING_ELSE", "must_be_resolved_in_v1": "X"}
        ]
        gate.validate_action_support_requirement(bad2)


def test_dependency_graph_well_ordered() -> None:
    assert gate.validate_dependency_graph(gate.PRE_TRAIN_DEPENDENCY_GRAPH_V1)
    with pytest.raises(RuntimeError, match="DEPENDENCY_GRAPH_MUST_START_WITH_THIS_GATE"):
        gate.validate_dependency_graph(gate.PRE_TRAIN_DEPENDENCY_GRAPH_V1[1:])
    with pytest.raises(RuntimeError, match="DEPENDENCY_GRAPH_MUST_END_WITH_FIRST_TRAINING"):
        bad = list(gate.PRE_TRAIN_DEPENDENCY_GRAPH_V1[:-1])
        gate.validate_dependency_graph(bad)


def test_self_consistency_audit_all_pass() -> None:
    audit = gate._self_consistency_audit()
    assert audit["status_v1"] == "PASS"
    for k, v in audit["checks_v1"].items():
        assert v is True, f"{k} failed"
