from __future__ import annotations

import copy

import pytest

from gx1.scripts import materialize_parallel_tail_r6_r5_2_repair_lane_pack_v1 as lanes


def test_exactly_10_lanes_are_pre_registered_before_execution() -> None:
    registered = lanes.pre_registered_lanes()
    assert [row["lane_id_v1"] for row in registered] == lanes.LANE_IDS
    assert len(registered) == 10


def test_lane_configs_cannot_be_mutated_after_execution_starts() -> None:
    registered = lanes.pre_registered_lanes()
    before = lanes.lane_config_hash(registered)
    mutated = copy.deepcopy(registered)
    mutated[0]["deterministic_config_v1"]["tail_threshold_v1"] = 0.01
    with pytest.raises(RuntimeError, match="LANE_CONFIG_MUTATED"):
        lanes.validate_lane_configs_unchanged(before, mutated)


def test_lane_10_must_reproduce_140_94_or_pack_is_invalid() -> None:
    valid = {
        "lane_id_v1": "LANE_10_NULL_REPLAY_REPRODUCIBILITY_CONTROL",
        "bad_count_v1": 140,
        "tail_count_v1": 94,
        "rows_added_vs_140_94_v1": 0,
        "rows_lost_vs_140_94_v1": 0,
        "safety_clean_v1": True,
    }
    assert lanes.validate_lane10_reproduces(valid) is True
    invalid = {**valid, "tail_count_v1": 95}
    with pytest.raises(RuntimeError, match="REPRODUCIBILITY_FAILURE"):
        lanes.validate_lane10_reproduces(invalid)


def test_unsafe_lanes_cannot_rank_above_safety_clean_lane() -> None:
    unsafe = {
        "lane_id_v1": "UNSAFE_HIGH_RAW",
        "safety_clean_v1": False,
        "oof_provenance_status_v1": "PASS",
        "in_sample_decisioning_used_v1": False,
        "bad_count_v1": 200,
        "tail_count_v1": 160,
        "precision_v1": 1.0,
    }
    safe = {
        "lane_id_v1": "SAFE_BASE",
        "safety_clean_v1": True,
        "oof_provenance_status_v1": "PASS",
        "in_sample_decisioning_used_v1": False,
        "bad_count_v1": 140,
        "tail_count_v1": 94,
        "precision_v1": 1.0,
    }
    ranked = lanes.rank_lanes([unsafe, safe])
    assert ranked[0]["lane_id_v1"] == "SAFE_BASE"


def test_true_safety_failure_cannot_be_selected_when_safe_lane_exists() -> None:
    ranked = lanes.rank_lanes(
        [
            {
                "lane_id_v1": "UNSAFE",
                "safety_clean_v1": False,
                "oof_provenance_status_v1": "PASS",
                "in_sample_decisioning_used_v1": False,
                "bad_count_v1": 190,
                "tail_count_v1": 140,
                "precision_v1": 1.0,
                "rows_added_vs_140_94_v1": 50,
            },
            {
                "lane_id_v1": "SAFE",
                "safety_clean_v1": True,
                "oof_provenance_status_v1": "PASS",
                "in_sample_decisioning_used_v1": False,
                "bad_count_v1": 141,
                "tail_count_v1": 95,
                "precision_v1": 1.0,
                "rows_added_vs_140_94_v1": 1,
            },
        ]
    )
    assert ranked[0]["lane_id_v1"] == "SAFE"


def test_no_lane_may_hide_strict_loso_or_low_support_by_contract() -> None:
    contract = lanes._contract(lanes.INPUT_TAIL_REPAIRED_PACKAGE_ROOT, lanes.INPUT_R6_TAIL_REPAIRED_ROOT)
    assert contract["common_low_support_reporting_required_v1"] is True
    assert contract["common_strict_loso_reporting_required_v1"] is True


def test_filter_only_lanes_are_marked_no_training() -> None:
    registered = lanes.pre_registered_lanes()
    assert all(row["training_allowed_v1"] is False for row in registered)


def test_training_lane_would_require_oof_provenance_in_contract() -> None:
    contract = lanes._contract(lanes.INPUT_TAIL_REPAIRED_PACKAGE_ROOT, lanes.INPUT_R6_TAIL_REPAIRED_ROOT)
    assert "OOF" in contract["common_oof_provenance_requirements_v1"]


def test_no_optuna_broad_sweep_freeze_promo_live() -> None:
    clean = lanes.validate_no_forbidden_actions(optuna=False, broad_sweep=False, freeze=False, promo=False, live=False)
    blocked = lanes.validate_no_forbidden_actions(optuna=True, broad_sweep=True, freeze=True, promo=True, live=True)
    assert clean["status_v1"] == "PASS"
    assert blocked["status_v1"] == "FAIL"


def test_no_dummy_synthetic_fallback_is_lane_output_rule() -> None:
    registered = lanes.pre_registered_lanes()
    assert all("fallback" not in row["allowed_inputs_v1"].lower() for row in registered)


def test_no_implicit_latest_glob_artifact_selection() -> None:
    assert lanes.validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB") is True
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        lanes.validate_explicit_artifact_selection("LATEST")


def test_fixed_controls_include_140_94_and_wednesday_180_149() -> None:
    assert lanes.validate_fixed_controls(lanes.FIXED_CONTROLS) is True
    by_id = {row["control_v1"]: row for row in lanes.FIXED_CONTROLS}
    assert by_id["tail_repaired_r5_2"]["bad_v1"] == 140
    assert by_id["tail_repaired_r5_2"]["tail_v1"] == 94
    assert by_id["wednesday"]["bad_v1"] == 180
    assert by_id["wednesday"]["tail_v1"] == 149
    assert by_id["wednesday"]["role_v1"] == "COMPARATOR_ONLY_NOT_ROW_TARGET"


def test_failed_r6_expansion_safe_subset_cannot_alter_baseline_without_lane_proof() -> None:
    lane = [row for row in lanes.pre_registered_lanes() if row["lane_id_v1"] == "LANE_02_R6_FAILED_EXPANSION_SAFE_SUBSET_ONLY"][0]
    assert lane["training_allowed_v1"] is False
    assert lane["deterministic_config_v1"]["mode_v1"] == "BASE_PLUS_UNION_SAFE_EXTRAS_FROM_FAILED_R6_EXPANSIONS"
    assert lane["final_promotion_allowed_v1"] is False


def test_recommendation_does_not_suggest_blind_sweep_when_baseline_remains_best() -> None:
    ranked = [
        {
            "lane_id_v1": "LANE_10_NULL_REPLAY_REPRODUCIBILITY_CONTROL",
            "safety_clean_v1": True,
            "bad_count_v1": 140,
            "tail_count_v1": 94,
            "rows_added_vs_140_94_v1": 0,
        }
    ]
    status, next_action = lanes._pack_status(ranked, lane10_ok=True)
    assert "SWEEP" not in next_action
    assert status == "R5_2_TAIL_REPAIR_TRACK_REMAINS_BEST"


def test_safe_improvement_status_uses_explicit_package_gate_not_promotion() -> None:
    ranked = [
        {
            "lane_id_v1": "SAFE_STRONG",
            "safety_clean_v1": True,
            "bad_count_v1": 150,
            "tail_count_v1": 100,
            "precision_v1": 1.0,
            "rows_added_vs_140_94_v1": 10,
        }
    ]
    status, next_action = lanes._pack_status(ranked, lane10_ok=True)
    assert status == "LANE_FOUND_SAFE_IMPROVEMENT_BEYOND_140_94"
    assert next_action == "MATERIALIZE_BEST_LANE_CANDIDATE_PACKAGE_REQUIRES_EXPLICIT_GATE_V1"


def test_structural_low_support_is_not_final_promotion_pass() -> None:
    lane = lanes.pre_registered_lanes()[0]
    assert lane["final_promotion_allowed_v1"] is False


def test_lane_pack_contract_is_not_canonical_r6() -> None:
    contract = lanes._contract(lanes.INPUT_TAIL_REPAIRED_PACKAGE_ROOT, lanes.INPUT_R6_TAIL_REPAIRED_ROOT)
    assert contract["final_promotion_allowed_v1"] is False
    assert contract["freeze_live_allowed_v1"] is False
