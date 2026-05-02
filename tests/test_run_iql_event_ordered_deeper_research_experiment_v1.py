from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_run_iql_event_ordered_deeper_research_experiment_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/RUN_IQL_EVENT_ORDERED_DEEPER_RESEARCH_EXPERIMENT_V1_20260428T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(gate.FINAL_STATUS, gate.NEXT_ACTION)
    assert gate.validate_final_status(gate.STABLE_STATUS, gate.STABLE_NEXT_ACTION)
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("PROMOTE_POLICY", gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(gate.FINAL_STATUS, "RUN_R6_NOW")


def test_state_column_guard_blocks_denied_tokens() -> None:
    assert gate.validate_state_columns(["candidate_score_z_train_only_v1", "signal_tail_repair_v1"])
    with pytest.raises(RuntimeError, match="DENIED_STATE_TOKEN_IN_MATRIX"):
        gate.validate_state_columns(["reward_t_v1"])
    with pytest.raises(RuntimeError, match="DENIED_STATE_TOKEN_IN_MATRIX"):
        gate.validate_state_columns(["historical_v2_blueprint_v1"])
    with pytest.raises(RuntimeError, match="DENIED_STATE_TOKEN_IN_MATRIX"):
        gate.validate_state_columns(["row_id_v1"])


def test_go_no_go_blocks_production_adapter_r6_full_lifecycle_and_promotion() -> None:
    payload = {
        "status_v1": gate.FINAL_STATUS,
        "next_recommended_action_v1": gate.NEXT_ACTION,
        "next_research_stage_allowed_v1": False,
        "full_lifecycle_sequential_iql_ready_v1": False,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
    }
    assert gate.validate_go_no_go(payload)
    with pytest.raises(RuntimeError, match="FORBIDDEN_PATH_OPENED"):
        gate.validate_go_no_go(dict(payload, iql_production_allowed_v1=True))
    with pytest.raises(RuntimeError, match="FORBIDDEN_PATH_OPENED"):
        gate.validate_go_no_go(dict(payload, policy_promotion_allowed_v1=True))
    with pytest.raises(RuntimeError, match="FULL_LIFECYCLE_IQL_OPENED"):
        gate.validate_go_no_go(dict(payload, full_lifecycle_sequential_iql_ready_v1=True))


def test_reproducibility_guard_locks_prior_training_and_dataset_counts() -> None:
    payload = {
        "input_prior_training_status_v1": "IQL_EVENT_ORDERED_RESEARCH_TRAINING_PASS_READY_FOR_DEEPER_EXPERIMENT",
        "input_event_dataset_status_v1": "IQL_EVENT_ORDERED_TRANSITION_DATASET_READY_FOR_RESEARCH_TRAINING",
        "input_no_fake_transition_status_v1": "PASS",
        "rows_v1": 1914,
        "episodes_v1": 58,
        "nonterminal_transitions_v1": 1856,
        "terminal_rows_v1": 58,
        "cross_run_transitions_v1": 0,
        "state_feature_count_v1": 11,
        "take_trade_count_v1": 78,
        "skip_count_v1": 1836,
        "reward_sum_v1": 89.0,
        "prior_policy_selected_rows_v1": 71,
        "prior_policy_reward_sum_v1": 91.75,
        "prior_policy_bad_tail_audit_only_v1": [70, 55],
        "prior_policy_precision_audit_only_v1": 0.9859154929577465,
        "prior_no_shortcut_audit_status_v1": "PASS",
    }
    assert gate.validate_reproducibility(payload)
    with pytest.raises(RuntimeError, match="IQL_EVENT_ORDERED_DEEPER_REPRODUCTION_FAILED"):
        gate.validate_reproducibility(dict(payload, prior_policy_reward_sum_v1=90.5))


def test_no_shortcut_guard_blocks_critical_failures() -> None:
    assert gate.validate_no_shortcut({"status_v1": "PASS", "critical_failures_v1": []})
    with pytest.raises(RuntimeError, match="IQL_EVENT_ORDERED_DEEPER_NO_SHORTCUT_FAILED"):
        gate.validate_no_shortcut({"status_v1": "FAIL", "critical_failures_v1": ["label leaked"]})


def test_materializer_writes_deeper_outputs_and_contextual_preferred_verdict(tmp_path: Path) -> None:
    artifact_root = tmp_path / "RUN_IQL_EVENT_ORDERED_DEEPER_RESEARCH_EXPERIMENT_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)

    assert summary["final_status_v1"] == gate.FINAL_STATUS
    assert summary["next_recommended_action_v1"] == gate.NEXT_ACTION
    assert summary["dataset_kind_v1"] == gate.DATASET_KIND
    assert summary["best_policy_id_v1"] == "EVENT_ORDER_ABLATION_CONTEXTUAL_EQUIVALENT_V1"
    assert summary["best_policy_selected_rows_v1"] == 70
    assert summary["best_policy_reward_v1"] == pytest.approx(92.0)
    assert summary["best_policy_bad_tail_audit_only_v1"] == [69, 55]
    assert summary["best_policy_precision_audit_only_v1"] == pytest.approx(0.9857142857142858)
    assert summary["best_policy_safety_status_v1"] == "CLEAN"
    assert summary["seed_reward_std_v1"] == pytest.approx(0.0)
    assert summary["seed_selected_std_v1"] == pytest.approx(0.0)
    assert summary["reward_delta_vs_contextual_v1"] == pytest.approx(1.5)
    assert summary["reward_delta_vs_event_order_ablation_v1"] == pytest.approx(0.0)
    assert summary["no_shortcut_audit_status_v1"] == "PASS"
    assert summary["research_only_event_ordered_not_full_lifecycle_iql_v1"] is True
    assert summary["adapter_r6_iql_production_live_remain_blocked_v1"] is True

    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name

    repro = json.loads((artifact_root / "iql_event_ordered_deeper_reproducibility_audit_v1.json").read_text())
    assert repro["rows_v1"] == 1914
    assert repro["episodes_v1"] == 58
    assert repro["nonterminal_transitions_v1"] == 1856
    assert repro["terminal_rows_v1"] == 58
    assert repro["cross_run_transitions_v1"] == 0
    assert repro["take_trade_count_v1"] == 78
    assert repro["skip_count_v1"] == 1836
    assert repro["prior_policy_selected_rows_v1"] == 71
    assert repro["prior_policy_reward_sum_v1"] == pytest.approx(91.75)

    split_rows = json.loads((artifact_root / "iql_event_ordered_deeper_split_audit_v1.json").read_text())[
        "rows_v1"
    ]
    split_by_id = {row["split_id_v1"]: row for row in split_rows}
    assert split_by_id["train"]["episodes_v1"] == 28
    assert split_by_id["train"]["transitions_v1"] == 930
    assert split_by_id["train"]["take_trade_count_v1"] == 39
    assert split_by_id["validation"]["episodes_v1"] == 19
    assert split_by_id["validation"]["transitions_v1"] == 742
    assert split_by_id["validation"]["take_trade_count_v1"] == 29
    assert split_by_id["test"]["episodes_v1"] == 11
    assert split_by_id["test"]["transitions_v1"] == 242
    assert split_by_id["test"]["take_trade_count_v1"] == 10

    variant_rows = json.loads((artifact_root / "iql_event_ordered_deeper_variant_metrics_v1.json").read_text())[
        "rows_v1"
    ]
    all_by_id = {row["policy_name_v1"]: row for row in variant_rows if row["split_id_v1"] == "all"}
    assert len(all_by_id) == 12
    assert all_by_id["LINEAR_FITTED_IQL_EVENT_ORDERED_RESEARCH_FIXED_V1"]["total_reward_v1"] == pytest.approx(
        91.75
    )
    assert all_by_id["LINEAR_FITTED_IQL_EVENT_ORDERED_RESEARCH_FIXED_V1"]["selected_take_rows_v1"] == 71
    assert all_by_id["EVENT_ORDER_ABLATION_CONTEXTUAL_EQUIVALENT_V1"]["total_reward_v1"] == pytest.approx(92.0)
    assert all_by_id["EVENT_ORDER_ABLATION_CONTEXTUAL_EQUIVALENT_V1"]["selected_take_rows_v1"] == 70
    assert all(row["safety_status_v1"] == "CLEAN" for row in all_by_id.values())

    usefulness = json.loads(
        (artifact_root / "iql_event_ordered_deeper_event_order_usefulness_audit_v1.json").read_text()
    )
    assert usefulness["status_v1"] == "PASS_BUT_CONTEXTUAL_EQUIVALENT_REMAINS_PREFERRED"
    assert usefulness["fixed_event_ordered_reward_v1"] == pytest.approx(91.75)
    assert usefulness["event_order_ablation_reward_v1"] == pytest.approx(92.0)
    assert usefulness["fixed_event_ordered_reward_delta_vs_ablation_v1"] == pytest.approx(-0.25)
    assert usefulness["event_order_beats_contextual_equivalent_ablation_v1"] is False
    assert usefulness["event_order_useful_or_decorative_v1"] == "DECORATIVE_OR_WEAKER_THAN_CONTEXTUAL_EQUIVALENT"

    baselines = json.loads((artifact_root / "iql_event_ordered_deeper_baseline_comparison_v1.json").read_text())[
        "rows_v1"
    ]
    baseline_by_name = {row["policy_name_v1"]: row for row in baselines}
    assert baseline_by_name["CONTEXTUAL_IQL_SANITY_POLICY"]["total_reward_v1"] == pytest.approx(90.5)
    assert baseline_by_name["140_94_COMPARATOR_POLICY"]["total_reward_v1"] == pytest.approx(91.25)
    assert baseline_by_name["BEST_EVENT_ORDERED_DEEPER_RESEARCH_POLICY"]["total_reward_v1"] == pytest.approx(92.0)

    action_support = json.loads((artifact_root / "iql_event_ordered_deeper_action_support_audit_v1.json").read_text())
    assert action_support["take_trade_count_v1"] == 78
    assert action_support["skip_count_v1"] == 1836
    assert action_support["take_examples_sufficient_for_small_research_v1"] is True
    assert action_support["take_examples_sufficient_for_production_iql_v1"] is False

    no_shortcut = json.loads((artifact_root / "iql_event_ordered_deeper_no_shortcut_audit_v1.json").read_text())
    assert no_shortcut["status_v1"] == "PASS"
    assert no_shortcut["checks_v1"]["labels_absent_from_state_next_state_v1"] is True
    assert no_shortcut["checks_v1"]["reward_absent_from_state_next_state_v1"] is True
    assert no_shortcut["checks_v1"]["historical_v2_blueprint_absent_v1"] is True
    assert no_shortcut["checks_v1"]["no_fake_next_state_v1"] is True
    assert no_shortcut["checks_v1"]["no_policy_promotion_v1"] is True

    go = json.loads(
        (artifact_root / "run_iql_event_ordered_deeper_research_experiment_go_no_go_v1.json").read_text()
    )
    assert go["status_v1"] == gate.FINAL_STATUS
    assert go["next_research_stage_allowed_v1"] is False
    assert go["full_lifecycle_sequential_iql_ready_v1"] is False
    assert go["adapter_build_allowed_v1"] is False
    assert go["r6_allowed_v1"] is False
    assert go["iql_production_allowed_v1"] is False
    assert go["policy_promotion_allowed_v1"] is False
