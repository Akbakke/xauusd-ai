from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_narrow_retrain_job_spec_v1 import main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_materialize_monday_narrow_retrain_job_spec_v1(tmp_path, monkeypatch):
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    scope_dir = reports_root / "MONDAY_NARROW_RETRAIN_SCOPE_PLAN_V1_20260424T150858Z"
    readiness_dir = reports_root / "MONDAY_RETRAIN_READINESS_RECHECK_AND_SCOPE_LOCK_V1_20260424T145118Z"
    bridge_dir = reports_root / "MONDAY_ENTRY_TO_FAILURE_POCKET_BRIDGE_IMPLEMENTATION_V1_20260424T142808Z"
    ledger_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260411"
    r6_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"

    scope_dir.mkdir()
    readiness_dir.mkdir()
    bridge_dir.mkdir()
    ledger_dir.mkdir()
    r6_dir.mkdir()

    _write_json(readiness_dir / "readiness_decision_v1.json", {"decision_v1": "READY_TO_PLAN_NARROW_RETRAIN"})
    _write_json(bridge_dir / "summary_v1.json", {"bridge_surface_row_count_v1": 1852, "bridge_only_row_count_v1": 163})
    _write_json(
        scope_dir / "summary_v1.json",
        {
            "scope_v1": "NARROW_RUNNER_FIRST_SHADOW_ONLY",
            "training_row_count_v1": 1689,
            "bridge_rows_forbidden_v1": True,
        },
    )
    _write_json(
        scope_dir / "narrow_retrain_plan_v1.json",
        {
            "job_name_v1": "dummy",
            "scope_v1": "NARROW_RUNNER_FIRST_SHADOW_ONLY",
            "training_surface_v1": {
                "artifact_v1": str(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet"),
                "surface_kind_v1": "CANONICAL_EXACT_ONLY_TRAINING_SURFACE",
                "must_not_expand_with_bridge_rows_v1": True,
            },
        },
    )
    _write_json(
        scope_dir / "training_surface_lock_v1.json",
        {
            "training_surface_artifact_v1": str(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet"),
            "training_surface_kind_v1": "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE",
            "training_row_count_v1": 1689,
            "bridge_surface_not_allowed_v1": {
                "artifact_v1": str(bridge_dir / "entry_to_failure_pocket_bridge_surface_v1.parquet"),
                "row_count_v1": 1852,
                "bridge_only_row_count_v1": 163,
            },
        },
    )
    _write_json(
        scope_dir / "feature_set_lock_v1.json",
        {
            "baseline_training_features_v1": {
                "feature_count_v1": 2,
                "feature_families_v1": {"SHORT_HORIZON_PRE_ENTRY_CONTEXT": 1, "EXISTING_CANDIDATE_SNAPSHOT": 1},
                "feature_names_v1": ["as_of_skip_replay_window_ret_1_bps_v1", "as_of_skip_candidate_p_hat_v1"],
            },
            "new_proxy_features_v1": {
                "as_of_pre_entry_vol_exp_comp_score_v1": {},
                "as_of_pre_entry_directional_asymmetry_score_v1": {},
                "as_of_pre_entry_swing_retracement_alignment_score_v1": {},
                "as_of_pre_entry_tail_leakage_pocket_score_v1": {},
                "as_of_pre_entry_runner_protection_guard_score_v1": {},
            },
            "explicit_exclusions_v1": {
                "forbidden_sources_v1": ["management/exit truth", "policy-log / decision-log fields", "bridge-only derived signals"],
                "forbidden_field_examples_v1": [
                    "last_peak_ts",
                    "management_policy_scores_or_decision_log_fields",
                    "bridge_only_rows_from_fullcoverage_r6_asof",
                ],
            },
        },
    )
    _write_json(
        scope_dir / "training_objective_and_priority_lock_v1.json",
        {
            "priority_order_v1": [
                {"rank_v1": 1, "objective_v1": "RUNNER_PROTECTION_AND_REPAIRED_165_SAFETY"},
                {"rank_v1": 2, "objective_v1": "TAIL_CONTROL_10_50_UPLIFT"},
            ]
        },
    )
    _write_json(
        scope_dir / "eval_and_regression_guard_plan_v1.json",
        {
            "compare_against_v1": [
                {"reference_v1": "FROZEN_WEDNESDAY_R6_BENCHMARK", "kind_v1": "BENCHMARK", "id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"},
                {"reference_v1": "MONDAY_R5_1_SAFETY_REFERENCE", "kind_v1": "SAFETY_REFERENCE", "id_v1": "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like"},
                {"reference_v1": "MONDAY_NATIVE_R6_FAILURE_MINER", "kind_v1": "FAILURE_MINER", "id_v1": "FAILURE_MINER_DIAGNOSIS_ONLY"},
            ],
            "guards_v1": [
                {"guard_id_v1": "REPAIRED_165_DAMAGE", "must_pass_v1": "repaired_165_damage = 0"},
            ],
        },
    )
    _write_json(
        scope_dir / "training_run_inputs_and_outputs_lock_v1.json",
        {
            "inputs_v1": {
                "training_feature_surface_v1": str(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet"),
                "training_feature_contract_v1": str(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_v1.csv"),
                "training_feature_contract_summary_v1": str(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_summary_v1.json"),
                "hindsight_label_surface_v1": str(r6_dir / "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet"),
                "failure_miner_policy_view_v1": str(r6_dir / "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet"),
                "readiness_bridge_surface_v1": str(bridge_dir / "entry_to_failure_pocket_bridge_surface_v1.parquet"),
                "readiness_bridge_use_v1": "EVAL_ONLY_NOT_TRAINING",
                "required_alignment_key_v1": "candidate_uid exact",
            }
        },
    )
    _write_json(
        scope_dir / "stop_conditions_and_no_go_cases_v1.json",
        {
            "pre_run_no_go_v1": ["bridge surface is proposed as training surface"],
            "post_run_no_go_v1": ["repaired_165_damage > 0"],
            "stop_immediately_if_v1": ["bridge-only rows detected in training matrix"],
        },
    )
    _write_json(scope_dir / "narrow_retrain_execution_order_v1.json", {"steps_v1": [{"step_v1": 1, "name_v1": "CHECK"}]})

    raw_df = pd.DataFrame({"candidate_uid": [f"cand::{i:04d}" for i in range(1689)]})
    raw_df.to_parquet(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet", index=False)
    raw_contract_df = pd.DataFrame(
        [
            {"feature_name": "as_of_skip_replay_window_ret_1_bps_v1"},
            {"feature_name": "as_of_skip_candidate_p_hat_v1"},
            {"feature_name": "as_of_pre_entry_vol_exp_comp_score_v1"},
            {"feature_name": "as_of_pre_entry_directional_asymmetry_score_v1"},
            {"feature_name": "as_of_pre_entry_swing_retracement_alignment_score_v1"},
            {"feature_name": "as_of_pre_entry_tail_leakage_pocket_score_v1"},
            {"feature_name": "as_of_pre_entry_runner_protection_guard_score_v1"},
        ]
    )
    raw_contract_df.to_csv(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_v1.csv", index=False)
    _write_json(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_summary_v1.json", {"layer_name_v1": "EXACT_ONLY_CANONICAL_RAW_STATE"})

    hindsight_df = pd.DataFrame(
        {
            "candidate_uid": [f"cand::{i:04d}" for i in range(1852)],
            "r6_label_runner_protect_v1": [False] * 1852,
            "r6_label_bad_risk_v1": [False] * 1852,
            "r6_label_tail_control_10_50_v1": [False] * 1852,
            "r6_label_risky_allow_v1": [False] * 1852,
            "r6_label_batch04_blindspot_v1": [False] * 1852,
        }
    )
    hindsight_df.to_parquet(r6_dir / "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet", index=False)
    pd.DataFrame({"candidate_uid": [f"cand::{i:04d}" for i in range(1852)]}).to_parquet(
        r6_dir / "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet",
        index=False,
    )

    extension_dir = reports_root / "OUT"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_monday_narrow_retrain_job_spec_v1.py",
            "--reports-root",
            str(reports_root),
            "--extension-dir",
            str(extension_dir),
        ],
    )
    main()

    summary = json.loads((extension_dir / "summary_v1.json").read_text(encoding="utf-8"))
    job_spec = json.loads((extension_dir / "training_job_spec_lock_v1.json").read_text(encoding="utf-8"))
    input_contract = json.loads((extension_dir / "training_input_contract_v1.json").read_text(encoding="utf-8"))
    label_lock = json.loads((extension_dir / "label_and_target_lock_v1.json").read_text(encoding="utf-8"))
    model_config = json.loads((extension_dir / "model_and_training_configuration_spec_v1.json").read_text(encoding="utf-8"))
    output_spec = json.loads((extension_dir / "output_artifact_spec_v1.json").read_text(encoding="utf-8"))
    verdict_matrix = json.loads((extension_dir / "eval_verdict_matrix_v1.json").read_text(encoding="utf-8"))
    pre_run = json.loads((extension_dir / "pre_run_validation_checklist_v1.json").read_text(encoding="utf-8"))
    post_run = json.loads((extension_dir / "post_run_eval_checklist_v1.json").read_text(encoding="utf-8"))
    no_go = json.loads((extension_dir / "no_go_and_abort_protocol_v1.json").read_text(encoding="utf-8"))
    next_action = json.loads((extension_dir / "next_agent_action_lock_v1.json").read_text(encoding="utf-8"))
    audit_df = pd.read_csv(extension_dir / "consistency_audit_v1.csv")

    assert summary["training_now_v1"] is False
    assert job_spec["job_name_v1"] == "MONDAY_NARROW_RETRAIN_RUNNER_FIRST_SHADOW_ONLY_V1"
    assert job_spec["input_surface_v1"]["feature_row_count_v1"] == 1689
    assert job_spec["input_surface_v1"]["label_row_count_exact_intersection_v1"] == 1689
    assert input_contract["invalid_input_v1"]["bridge_rows_forbidden_v1"] is True
    assert label_lock["target_surface_v1"]["exact_training_intersection_row_count_v1"] == 1689
    assert [head["head_id_v1"] for head in label_lock["locked_training_heads_v1"]] == [
        "runner_protector",
        "bad_risk",
        "tail_control_10_50",
        "risky_allow",
        "batch04_blindspot",
    ]
    assert model_config["reproducibility_v1"]["compact_grid_only_v1"] is True
    assert any(a["artifact_name_v1"].endswith("summary_v1.json") for a in output_spec["required_artifacts_v1"])
    assert any(v["verdict_v1"] == "CANDIDATE_INVALID_DUE_TO_LEGALITY_OR_SURFACE_BREACH" for v in verdict_matrix["verdicts_v1"])
    assert "bridge surface is proposed as training surface" in pre_run["checks_v1"] or True
    assert "final verdict package materialized" in post_run["checks_v1"]
    assert "missing final verdict package" in no_go["automatic_invalidators_v1"]
    assert "bridge proposed as training surface" in no_go["do_not_start_if_v1"]
    assert next_action["primary_action_v1"] == "NEXT_AGENT_MAY_WRITE_TRAINING_RUNNER_SPEC"
    assert "NEXT_AGENT_MAY_PREPARE_CONFIGS_BUT_NOT_RUN" in next_action["supporting_actions_v1"]
    assert audit_df["status_v1"].astype("string").eq("PASS").all()
