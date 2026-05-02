from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import materialize_run_contextual_iql_with_v2_state_and_reward_variants_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [
            Path(
                "/tmp/RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1_20260429T000000Z_LOCK"
            )
        ]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])


def test_no_forbidden_actions_default_pass() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"
    blocked = gate.validate_no_forbidden_actions(adapter=True, r6=True)
    assert blocked["status_v1"] == "FAIL"


def test_validate_final_status_only_allowed_pairs() -> None:
    assert gate.validate_final_status(
        "RUN_CONTEXTUAL_IQL_V2_PASS_REWARD_VARIANTS_NEUTRAL",
        "DEEPEN_IQL_ACTION_SUPPORT_AND_BEHAVIOR_POLICY_AUDIT_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status(
            "ARBITRARY",
            "DEEPEN_IQL_ACTION_SUPPORT_AND_BEHAVIOR_POLICY_AUDIT_V1",
        )
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "RUN_CONTEXTUAL_IQL_V2_PASS_REWARD_VARIANTS_NEUTRAL",
            "OPEN_R6_NOW",
        )


def test_validate_no_deprecated_revival_blocks_quarantine_imports(tmp_path: Path) -> None:
    bad = tmp_path / "imports_quarantine.py"
    bad.write_text(
        "from gx1.quarantine._DEPRECATED_SCRIPTS_20260219 import run_shadow_counterfactual\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN"):
        gate.validate_no_deprecated_revival(bad)
    good = tmp_path / "clean.py"
    good.write_text("import pandas as pd\n", encoding="utf-8")
    assert gate.validate_no_deprecated_revival(good)
    assert gate.validate_no_deprecated_revival(Path(gate.__file__))


def test_compute_quality_metrics_handles_empty_and_nonempty_masks() -> None:
    join_aligned = pd.DataFrame(
        {
            "candidate_uid_v1": ["a", "b", "c"],
            "pnl_bps": [50.0, -20.0, 80.0],
            "mfe_bps": [60.0, 10.0, 100.0],
            "mae_bps": [-10.0, -50.0, -5.0],
            "exit_reason": ["THRESHOLD", "CATASTROPHIC_GUARD", "BE_PLUS_FLOOR"],
        }
    )
    empty = gate._compute_quality_metrics(np.zeros(3, dtype=bool), join_aligned)
    assert empty["mean_pnl_bps_v1"] == 0.0
    assert empty["cata_exit_count_v1"] == 0

    full = gate._compute_quality_metrics(np.array([True, True, True]), join_aligned)
    assert full["mean_pnl_bps_v1"] == pytest.approx((50 - 20 + 80) / 3)
    assert full["cata_exit_count_v1"] == 1
    assert full["mae_dominated_count_v1"] == 1
    # Row 1 triggers peak_giveback: mfe=10, pnl=-20, mfe-pnl=30 > 0.5*mfe=5
    assert full["peak_giveback_count_v1"] == 1


def test_reward_class_audit_passes_when_state_columns_clean() -> None:
    audit = gate._reward_class_audit()
    assert audit["leakage_status_v1"] == "PASS"
    assert audit["leaked_input_fields_v1"] == []


def test_compute_reward_arrays_zero_outside_shield_and_correct_inside() -> None:
    n = 10
    frame = pd.DataFrame(
        {
            "is_140_94_baseline_v1": [False] * n,
            "bad_label_v1": [True, False, True, False, True, False, True, False, True, False],
            "tail_label_v1": [True, False, False, True, True, False, False, True, True, False],
            "unsafe_audit_v1": [False] * n,
        }
    )
    shield = pd.Series([True, False, True, False, True, False, True, False, True, False])
    masks = {
        "hardened": shield,
        "source_confluence_repairable_v1": pd.Series([False] * n),
    }
    join_aligned = pd.DataFrame(
        {
            "candidate_uid_v1": [f"u{i}" for i in range(n)],
            "pnl_bps": [10.0, 20.0, -5.0, 50.0, 100.0, -10.0, 5.0, -2.0, 30.0, 8.0],
            "mfe_bps": [50.0, 60.0, 5.0, 80.0, 200.0, 1.0, 8.0, 0.0, 40.0, 12.0],
            "mae_bps": [-5.0, -10.0, -10.0, -3.0, -40.0, -20.0, -2.0, -5.0, -8.0, -1.0],
        }
    )
    rewards = gate._compute_reward_arrays(frame, masks, join_aligned)
    assert set(rewards.keys()) == set(gate.ALL_REWARD_FAMILIES)
    pnl_arr = rewards["ENTRY_REALIZED_PNL_REWARD_V2"]
    expected_pnl = np.array(
        [10.0, 0.0, -5.0, 0.0, 100.0, 0.0, 5.0, 0.0, 30.0, 0.0]
    )
    assert np.allclose(pnl_arr, expected_pnl)
    mfe_arr = rewards["ENTRY_MFE_CAPTURE_REWARD_V2"]
    inside_indices = np.array([0, 2, 4, 6, 8])
    expected_inside = np.clip(
        np.array([10.0 / 50.0, -5.0 / 5.0, 100.0 / 200.0, 5.0 / 8.0, 30.0 / 40.0]),
        -2.0,
        2.0,
    )
    assert np.allclose(mfe_arr[inside_indices], expected_inside)
    outside_indices = np.array([1, 3, 5, 7, 9])
    assert np.allclose(mfe_arr[outside_indices], 0.0)


def test_go_no_go_emits_lift_when_at_least_one_v2_diverges() -> None:
    policy_rows = [
        {"reward_id_v1": "SAFETY_WEIGHTED_REWARD_V1", "selected_rows_v1": 76, "safety_violations_v1": 0},
        {"reward_id_v1": "ENTRY_REALIZED_PNL_REWARD_V2", "selected_rows_v1": 70, "safety_violations_v1": 0},
        {"reward_id_v1": "ENTRY_MFE_CAPTURE_REWARD_V2", "selected_rows_v1": 76, "safety_violations_v1": 0},
        {"reward_id_v1": "ENTRY_MAE_BURDEN_REWARD_V2", "selected_rows_v1": 76, "safety_violations_v1": 0},
        {"reward_id_v1": "ENTRY_TRANSPARENT_COMBINED_REWARD_V2", "selected_rows_v1": 76, "safety_violations_v1": 0},
    ]
    baseline_rows = [
        {"policy_name_v1": "ALWAYS_SKIP", "selected_rows_v1": 0},
        {"policy_name_v1": "ALWAYS_TAKE_WITHIN_78_SHIELD", "selected_rows_v1": 78},
    ]
    status, action, _ = gate._go_no_go(policy_rows, baseline_rows)
    assert status == "RUN_CONTEXTUAL_IQL_V2_PASS_REWARD_VARIANTS_LIFT_OBSERVED"
    assert action == "RUN_IQL_REWARD_VARIANT_SENSITIVITY_V1"


def test_go_no_go_emits_neutral_when_all_variants_match_v1() -> None:
    policy_rows = [
        {"reward_id_v1": rid, "selected_rows_v1": 76, "safety_violations_v1": 0}
        for rid in gate.ALL_REWARD_FAMILIES
    ]
    baseline_rows = [
        {"policy_name_v1": "ALWAYS_SKIP", "selected_rows_v1": 0},
        {"policy_name_v1": "ALWAYS_TAKE_WITHIN_78_SHIELD", "selected_rows_v1": 78},
    ]
    status, action, _ = gate._go_no_go(policy_rows, baseline_rows)
    assert status == "RUN_CONTEXTUAL_IQL_V2_PASS_REWARD_VARIANTS_NEUTRAL"
    assert action == "DEEPEN_IQL_ACTION_SUPPORT_AND_BEHAVIOR_POLICY_AUDIT_V1"


def test_go_no_go_blocks_on_safety_violation() -> None:
    policy_rows = [
        {"reward_id_v1": rid, "selected_rows_v1": 76, "safety_violations_v1": 0}
        for rid in gate.ALL_REWARD_FAMILIES
    ]
    policy_rows[2]["safety_violations_v1"] = 1
    baseline_rows = [
        {"policy_name_v1": "ALWAYS_SKIP", "selected_rows_v1": 0},
        {"policy_name_v1": "ALWAYS_TAKE_WITHIN_78_SHIELD", "selected_rows_v1": 78},
    ]
    status, action, _ = gate._go_no_go(policy_rows, baseline_rows)
    assert status == "RUN_CONTEXTUAL_IQL_V2_BLOCKED_BY_SAFETY_VIOLATION"
    assert action == "HOLD_UNTIL_NEW_AS_OF_FAMILIES_LANDED_V1"


def test_write_artifacts_produces_required_outputs(tmp_path: Path) -> None:
    out_root = tmp_path / "RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1_20260429T999999Z_LOCK"
    result = gate.write_artifacts(out_root=out_root, built_at_utc="20260429T999999Z")
    artifact_root = Path(result["artifact_root"])
    assert artifact_root == out_root
    for required in [
        "manifest_v1.json",
        "summary_v1.json",
        "status_v1.json",
        "report_v1.md",
        "run_contextual_iql_with_v2_state_and_reward_variants_go_no_go_v1.json",
        "input_manifest_v1.json",
        "reproducibility_audit_v1.json",
        "no_shortcut_audit_v1.json",
        "reward_class_audit_v1.json",
        "iql_policy_per_reward_comparator_v1.json",
        "iql_baseline_policy_comparator_v1.json",
        "iql_per_reward_training_configs_v1.json",
        "reward_arrays_summary_v1.csv",
    ]:
        assert (artifact_root / required).exists(), f"missing {required}"

    summary = json.loads((artifact_root / "summary_v1.json").read_text())
    assert summary["row_count_invariant_v1"] is True
    assert summary["seventy_eight_shield_invariant_v1"] is True
    assert summary["reward_family_count_v1"] == 5
    assert summary["reward_v2_variant_count_v1"] == 4
    assert summary["policy_count_v1"] == 5
    assert summary["final_status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert summary["next_action_v1"] in gate.ALLOWED_NEXT_ACTIONS

    no_shortcut = json.loads((artifact_root / "no_shortcut_audit_v1.json").read_text())
    assert no_shortcut["status_v1"] == "PASS"
    reward_class = json.loads(
        (artifact_root / "reward_class_audit_v1.json").read_text()
    )
    assert reward_class["leakage_status_v1"] == "PASS"

    policy_data = json.loads(
        (artifact_root / "iql_policy_per_reward_comparator_v1.json").read_text()
    )
    reward_ids = {row["reward_id_v1"] for row in policy_data["rows_v1"]}
    assert reward_ids == set(gate.ALL_REWARD_FAMILIES)
    for row in policy_data["rows_v1"]:
        assert row["safety_status_v1"] == "CLEAN"
