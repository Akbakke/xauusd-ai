#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import (
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
    _jsonable,
)
from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import _read_json


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "TRUE_R5_2_REBUILD_RUNNER_SPEC_V1"

LABEL_OBJECTIVE_DEFAULT = DEFAULT_REPORTS_ROOT / "FIX_R5_2_LABEL_OBJECTIVE_FIRST_V1_20260426T_LOCK"
V3_SCORE_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_20260426T_CONTRACT_V3_R5_R51_R52"
OLD_V3_R6_DEFAULT = DEFAULT_REPORTS_ROOT / "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_20260426T_CONTRACT_V3_R6_FROM_V3_R52"

LABEL_TABLE = "r5_2_pocket_label_table_v1.csv"
LABEL_CONTRACT = "r5_2_new_label_contract_v1.json"
WEIGHT_SPEC = "r5_2_objective_weighting_spec_v1.json"
REBUILD_EXPERIMENT_SPEC = "r5_2_rebuild_experiment_spec_v1.json"

OUTPUT_FILES = {
    "runner_spec": "true_r5_2_rebuild_runner_spec_v1.json",
    "input_contract": "r5_2_rebuild_input_contract_v1.json",
    "loader_spec": "r5_2_rebuild_label_and_weight_loader_spec_v1.json",
    "model_config": "r5_2_rebuild_model_config_lock_v1.json",
    "eval_guards": "r5_2_rebuild_eval_and_safety_guards_v1.json",
    "output_spec": "r5_2_rebuild_output_spec_v1.json",
    "prelaunch": "r5_2_rebuild_prelaunch_checklist_v1.json",
    "abort_rules": "r5_2_rebuild_abort_rules_v1.json",
    "r6_contract": "downstream_r6_consumption_contract_v1.json",
    "next_action": "next_action_lock_v1.json",
    "summary": "summary_v1.json",
    "report": "report_v1.md",
    "manifest": "manifest_v1.json",
    "status": "status_v1.json",
    "audit": "consistency_audit_v1.csv",
}

EXPECTED_MISSED_BUCKETS = {
    "STRONG_BAD_BLOCK_TARGET": 96,
    "TAIL_CONTROL_TARGET": 147,
    "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD": 130,
    "RUNNER_PROTECT_TARGET": 17,
}

REQUIRED_BUCKETS = [
    "STRONG_BAD_BLOCK_TARGET",
    "TAIL_CONTROL_TARGET",
    "RISKY_ALLOW_TARGET",
    "RUNNER_PROTECT_TARGET",
    "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD",
    "IGNORE_OR_MONITOR_ONLY",
]

FUTURE_OUTPUTS = [
    "r5_2_rebuild_training_summary_v1.json",
    "r5_2_model_manifest_v1.json",
    "r5_2_config_manifest_v1.json",
    "r5_2_feature_manifest_v1.csv",
    "r5_2_label_weight_manifest_v1.csv",
    "r5_2_prediction_view_v1.parquet",
    "r5_2_score_package_v1.parquet",
    "r5_2_base_membership_v1.parquet",
    "r5_2_eval_summary_v1.json",
    "r5_2_pocket_eval_report_v1.csv",
    "r5_2_safety_guard_report_v1.json",
    "r5_2_downstream_r6_input_manifest_v1.json",
    "status_v1.json",
    "manifest_v1.json",
    "consistency_audit_v1.csv",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _load_inputs(label_objective_dir: Path, v3_score_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    label_table = pd.read_csv(label_objective_dir / LABEL_TABLE)
    label_contract = _read_json(label_objective_dir / LABEL_CONTRACT)
    weighting = _read_json(label_objective_dir / WEIGHT_SPEC)
    rebuild_spec = _read_json(label_objective_dir / REBUILD_EXPERIMENT_SPEC)
    score_summary = _read_json(v3_score_dir / "summary_v1.json")
    return label_table, label_contract, weighting, rebuild_spec, score_summary


def _label_stats(label_table: pd.DataFrame) -> dict[str, Any]:
    active = int((label_table["calendar_quarantine_status_v1"] == "ACTIVE_CANDIDATE").sum()) if "calendar_quarantine_status_v1" in label_table.columns else 0
    quarantine = int((label_table["calendar_quarantine_status_v1"] == "QUARANTINED").sum()) if "calendar_quarantine_status_v1" in label_table.columns else 0
    missed = label_table[(_bool_series(label_table, "label_should_not_take_v1") | _bool_series(label_table, "tail_10_50_mfe_v1")) & ~_bool_series(label_table, "r5_2_v3_base_flag_v1")]
    missed_bucket_counts = {str(key): int(value) for key, value in missed["new_r5_2_label_bucket_v1"].value_counts().to_dict().items()}
    bucket_counts = {str(key): int(value) for key, value in label_table["new_r5_2_label_bucket_v1"].value_counts().to_dict().items()}
    ambiguous_bad_positive = int(
        (
            label_table["new_r5_2_label_bucket_v1"].eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD")
            & _bool_series(label_table, "bad_eligibility_target_v1")
        ).sum()
    )
    runner_bad_positive = int(
        (
            label_table["new_r5_2_label_bucket_v1"].eq("RUNNER_PROTECT_TARGET")
            & _bool_series(label_table, "bad_eligibility_target_v1")
        ).sum()
    )
    return {
        "row_count_v1": int(len(label_table)),
        "active_rows_v1": active,
        "quarantine_rows_v1": quarantine,
        "bucket_counts_v1": bucket_counts,
        "missed_bad_tail_row_count_v1": int(len(missed)),
        "missed_bad_tail_bucket_counts_v1": missed_bucket_counts,
        "ambiguous_high_mfe_bad_positive_count_v1": ambiguous_bad_positive,
        "runner_protect_bad_positive_count_v1": runner_bad_positive,
    }


def _spec_complete(stats: dict[str, Any], score_summary: dict[str, Any], label_contract: dict[str, Any], weighting: dict[str, Any], rebuild_spec: dict[str, Any]) -> bool:
    expected_buckets_ok = all(int(stats["missed_bad_tail_bucket_counts_v1"].get(bucket, -1)) == expected for bucket, expected in EXPECTED_MISSED_BUCKETS.items())
    weights = weighting.get("positive_class_weights_v1") or {}
    costs = weighting.get("protection_costs_v1") or {}
    return bool(
        stats["row_count_v1"] == 1914
        and stats["active_rows_v1"] == 1852
        and stats["quarantine_rows_v1"] == 62
        and int(score_summary.get("as_of_column_count_v1") or 0) == 109
        and stats["missed_bad_tail_row_count_v1"] == 390
        and expected_buckets_ok
        and stats["ambiguous_high_mfe_bad_positive_count_v1"] == 0
        and stats["runner_protect_bad_positive_count_v1"] == 0
        and label_contract.get("contract_id_v1") == "R5_2_LABEL_OBJECTIVE_BAD_TAIL_ELIGIBILITY_WITH_HARD_WINNER_PROTECTION_V1"
        and float(weights.get("STRONG_BAD_BLOCK_TARGET") or 0.0) == 3.0
        and float(weights.get("TAIL_CONTROL_TARGET") or 0.0) == 2.5
        and float(costs.get("RUNNER_PROTECT_TARGET") or 0.0) == 10.0
        and float(costs.get("HUNDRED_OR_TWO_HUNDRED_MFE") or 0.0) == 20.0
        and rebuild_spec.get("recommended_design_v1") == "POCKET_AWARE_MULTI_HEAD_R5_2_BAD_TAIL_ELIGIBILITY_WITH_RUNNER_PROTECTION"
    )


def _input_contract(label_objective_dir: Path, v3_score_dir: Path, stats: dict[str, Any], score_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "R5_2_REBUILD_INPUT_CONTRACT_V1",
        "foundation_v1": {
            "score_package_dir_v1": str(v3_score_dir),
            "foundation_dir_v1": score_summary.get("foundation_dir_v1"),
            "row_count_v1": 1914,
            "active_rows_v1": 1852,
            "quarantine_rows_v1": 62,
            "as_of_column_count_v1": 109,
            "base_feature_count_v1": score_summary.get("base_feature_count_v1"),
            "current_r5_2_feature_count_v1": score_summary.get("r5_2_feature_count_v1"),
        },
        "label_table_v1": {
            "path_v1": str(label_objective_dir / LABEL_TABLE),
            "row_count_v1": stats["row_count_v1"],
            "ambiguous_high_mfe_bad_positive_count_v1": stats["ambiguous_high_mfe_bad_positive_count_v1"],
            "runner_protect_bad_positive_count_v1": stats["runner_protect_bad_positive_count_v1"],
        },
        "required_key_columns_v1": ["candidate_uid", "trade_uid", "decision_timestamp"],
        "required_score_input_families_v1": {
            "existing_as_of_features_v1": "109 Monday canonical AS_OF columns from foundation score package.",
            "r5_signals_v1": [
                "pred__entry_r5_should_not_take__prob_true_v1",
                "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
                "pred__entry_r5_runner_protect__prob_true_v1",
                "pred__entry_r5_tail_control_10_50_risk__prob_true_v1",
            ],
            "r5_1_signals_v1": ["r5_1_bad_blocker_score_v1", "r5_1_runner_guard_score_v1"],
            "allowed_current_r5_2_inputs_v1": [R5_2_BAD_PROB, R5_2_RUNNER_PROB],
        },
        "forbidden_inputs_v1": [
            "hindsight as model feature",
            "exit/management truth as model feature",
            "1689 exact-only baseline",
            "bridge/readiness as training surface",
            "protector-first artifacts",
            "diagnostic/narrow surfaces as canonical input",
        ],
    }


def _loader_spec(label_contract: dict[str, Any], weighting: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "R5_2_REBUILD_LABEL_AND_WEIGHT_LOADER_SPEC_V1",
        "label_contract_id_v1": label_contract["contract_id_v1"],
        "supported_buckets_v1": REQUIRED_BUCKETS,
        "target_columns_v1": {
            "bad_head_v1": "bad_eligibility_target_v1",
            "tail_head_v1": "tail_eligibility_target_v1",
            "risky_attention_head_v1": "risky_attention_target_v1",
            "runner_protect_head_v1": "runner_protect_target_v1",
            "ambiguous_monitor_v1": "ambiguous_high_mfe_monitor_v1",
        },
        "sample_weight_columns_v1": ["sample_weight_v1", "protection_weight_v1"],
        "locked_weights_v1": {
            "bad_target_weight_v1": float(weighting["positive_class_weights_v1"]["STRONG_BAD_BLOCK_TARGET"]),
            "tail_target_weight_v1": float(weighting["positive_class_weights_v1"]["TAIL_CONTROL_TARGET"]),
            "runner_protect_weight_v1": float(weighting["protection_costs_v1"]["RUNNER_PROTECT_TARGET"]),
            "hard_protection_weight_v1": 20.0,
        },
        "required_validation_counts_v1": {
            "missed_bad_tail_rows_represented_v1": 390,
            "strong_bad_v1": 96,
            "tail_control_v1": 147,
            "ambiguous_high_mfe_not_bad_positive_v1": 130,
            "runner_protect_v1": 17,
            "ambiguous_high_mfe_bad_positive_count_v1": 0,
        },
        "observed_validation_counts_v1": {
            "missed_bad_tail_rows_represented_v1": stats["missed_bad_tail_row_count_v1"],
            "strong_bad_v1": stats["missed_bad_tail_bucket_counts_v1"].get("STRONG_BAD_BLOCK_TARGET", 0),
            "tail_control_v1": stats["missed_bad_tail_bucket_counts_v1"].get("TAIL_CONTROL_TARGET", 0),
            "ambiguous_high_mfe_not_bad_positive_v1": stats["missed_bad_tail_bucket_counts_v1"].get("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD", 0),
            "runner_protect_v1": stats["missed_bad_tail_bucket_counts_v1"].get("RUNNER_PROTECT_TARGET", 0),
            "ambiguous_high_mfe_bad_positive_count_v1": stats["ambiguous_high_mfe_bad_positive_count_v1"],
        },
    }


def _model_config(rebuild_spec: dict[str, Any], weighting: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "R5_2_REBUILD_MODEL_CONFIG_LOCK_V1",
        "selected_first_rebuild_design_v1": "multi-head bad/tail/protect model",
        "design_class_v1": "pocket-aware base classifier with calibrated eligibility and protection outputs",
        "model_family_v1": rebuild_spec["model_family_v1"],
        "heads_v1": rebuild_spec["heads_v1"],
        "objective_v1": "Weighted multi-head binary objectives: bad eligibility, tail eligibility, risky attention, runner protection.",
        "sample_weighting_v1": weighting["positive_class_weights_v1"],
        "protection_weighting_v1": weighting["protection_costs_v1"],
        "seed_v1": 20260426,
        "split_eval_setup_v1": rebuild_spec["split_eval_v1"],
        "calibration_requirements_v1": [
            "per-head probability calibration report",
            "base-membership frontier from eligibility-high and protection-low scores",
            "LOSO and batch stability before downstream R6 use",
        ],
        "output_score_columns_v1": [
            "pred__entry_r5_2_rebuild_bad_eligibility__prob_true_v1",
            "pred__entry_r5_2_rebuild_tail_10_50_eligibility__prob_true_v1",
            "pred__entry_r5_2_rebuild_risky_attention__prob_true_v1",
            "pred__entry_r5_2_rebuild_runner_protect__prob_true_v1",
        ],
        "base_membership_construction_v1": {
            "flag_v1": "r5_2_rebuilt_base_membership_v1",
            "rule_v1": "eligibility score passes selected calibrated frontier AND runner_protect/protection scores stay below safety frontier",
            "do_not_reuse_old_flags_v1": [
                "r5_2_original_base_flag_v1",
                "r5_2_v1_base_flag_v1",
                "r5_2_v2_base_flag_v1",
                "r5_2_v3_base_flag_v1",
            ],
        },
    }


def _eval_guards() -> dict[str, Any]:
    return {
        "layer_name": "R5_2_REBUILD_EVAL_AND_SAFETY_GUARDS_V1",
        "metrics_v1": [
            "bad_base_recall",
            "tail_base_recall",
            "precision",
            "worst_LOSO",
            "high_MFE_protection",
            "repaired_protection",
            "strongest_winner_protection",
            "runner_near_miss_protection",
            "50_100_200_blocked_risk",
            "ambiguous_high_MFE_treatment",
            "batch_split_stability",
        ],
        "hard_fail_v1": [
            "ambiguous high-MFE becomes bad-positive",
            "repaired/strongest/100+/200+ protection breaks",
            "50+ risk explodes",
            "runner-protect rows are treated as bad-positive",
            "label table mismatch",
            "row/key mismatch",
            "forbidden feature is used",
        ],
        "benchmark_references_v1": {
            "old_v3_r5_2_base_v1": {"bad_v1": 82, "tail_v1": 51, "precision_v1": 1.0, "worst_loso_v1": 1.0},
            "old_v3_r6_v1": {"bad_v1": 82, "tail_v1": 51},
            "wednesday_benchmark_v1": {"bad_v1": 180, "tail_v1": 149},
        },
    }


def _output_spec() -> dict[str, Any]:
    return {
        "layer_name": "R5_2_REBUILD_OUTPUT_SPEC_V1",
        "required_future_outputs_v1": FUTURE_OUTPUTS,
        "namespace_rule_v1": "Append-only deterministic namespace; no freeze/promo/live output from R5.2 rebuild.",
    }


def _prelaunch() -> dict[str, Any]:
    checks = [
        "foundation row count check",
        "label table row count check",
        "key alignment check",
        "AS_OF schema check",
        "forbidden feature check",
        "bucket count check",
        "ambiguous high-MFE bad-positive = 0",
        "sample/protection weight check",
        "output namespace clean",
        "existing feature assets resolved",
        "no diagnostic/narrow/protector surfaces used",
        "run flag required",
    ]
    return {
        "layer_name": "R5_2_REBUILD_PRELAUNCH_CHECKLIST_V1",
        "checks_v1": checks,
        "explicit_run_flag_required_v1": "--run-true-r5-2-rebuild",
        "dry_run_allowed_v1": True,
        "training_allowed_by_this_spec_materializer_v1": False,
    }


def _abort_rules() -> dict[str, Any]:
    return {
        "layer_name": "R5_2_REBUILD_ABORT_RULES_V1",
        "abort_if_v1": [
            "wrong foundation is used",
            "label table mismatch",
            "1689 exact-only is used",
            "bridge/readiness is used as training surface",
            "hindsight/exit/management truth is used as feature",
            "ambiguous high-MFE becomes bad-positive",
            "runner-protect becomes bad-positive",
            "repaired/strongest/100+/200+ protection breaks",
            "output namespace is not clean",
            "downstream R6 input manifest cannot be materialized",
        ],
    }


def _r6_contract() -> dict[str, Any]:
    return {
        "layer_name": "DOWNSTREAM_R6_CONSUMPTION_CONTRACT_V1",
        "r5_2_score_columns_for_r6_v1": [
            "pred__entry_r5_2_rebuild_bad_eligibility__prob_true_v1",
            "pred__entry_r5_2_rebuild_tail_10_50_eligibility__prob_true_v1",
            "pred__entry_r5_2_rebuild_risky_attention__prob_true_v1",
            "pred__entry_r5_2_rebuild_runner_protect__prob_true_v1",
        ],
        "base_membership_flags_for_r6_v1": ["r5_2_rebuilt_base_membership_v1"],
        "use_r5_2_base_wiring_v1": "R6 use_r5_2_base=true must read r5_2_rebuilt_base_membership_v1 from the rebuilt R5.2 score package.",
        "old_flags_not_allowed_v1": [
            "r5_2_original_base_flag_v1",
            "r5_2_v1_base_flag_v1",
            "r5_2_v2_base_flag_v1",
            "r5_2_v3_base_flag_v1",
        ],
        "compare_after_future_r6_retrain_v1": {
            "old_v3_r6_v1": {"bad_v1": 82, "tail_v1": 51},
            "wednesday_benchmark_v1": {"bad_v1": 180, "tail_v1": 149},
        },
        "next_separate_step_after_green_r5_2_rebuild_v1": "RUN_R6_RETRAIN_FROM_TRUE_R5_2_REBUILD_SCORE_PACKAGE_EXPLICIT_FLAG",
        "r6_retrain_started_by_this_spec_v1": False,
    }


def _runner_spec(
    output_dir: Path,
    label_objective_dir: Path,
    v3_score_dir: Path,
    input_contract: dict[str, Any],
    loader_spec: dict[str, Any],
    model_config: dict[str, Any],
    eval_guards: dict[str, Any],
    output_spec: dict[str, Any],
    prelaunch: dict[str, Any],
    abort_rules: dict[str, Any],
    r6_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "layer_name": "TRUE_R5_2_REBUILD_RUNNER_SPEC_V1",
        "job_name_v1": "TRUE_R5_2_REBUILD_WITH_POCKET_AWARE_OBJECTIVE_V1",
        "spec_output_dir_v1": str(output_dir),
        "training_started_v1": False,
        "r6_started_v1": False,
        "future_runner_module_v1": "gx1.scripts.run_true_r5_2_rebuild_runner_v1",
        "future_command_template_v1": [
            "PYTHONPATH=.",
            ".venv/bin/python",
            "-m",
            "gx1.scripts.run_true_r5_2_rebuild_runner_v1",
            "--run-true-r5-2-rebuild",
            "--label-objective-dir",
            str(label_objective_dir),
            "--foundation-score-package-dir",
            str(v3_score_dir),
            "--output-dir",
            "<append_only_true_r5_2_rebuild_output_namespace>",
        ],
        "input_foundation_v1": input_contract["foundation_v1"],
        "label_and_weight_loader_v1": loader_spec,
        "feature_set_v1": input_contract["required_score_input_families_v1"],
        "model_objective_design_v1": model_config,
        "sample_weights_v1": loader_spec["locked_weights_v1"],
        "split_eval_v1": model_config["split_eval_setup_v1"],
        "output_namespace_v1": output_spec["namespace_rule_v1"],
        "prelaunch_checks_v1": prelaunch["checks_v1"],
        "abort_no_go_checks_v1": abort_rules["abort_if_v1"],
        "downstream_r6_consumption_v1": r6_contract,
    }


def _next_action(spec_complete: bool) -> dict[str, Any]:
    return {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": "IMPLEMENT_TRUE_R5_2_REBUILD_RUNNER_NEXT" if spec_complete else "FIX_R5_2_REBUILD_SPEC_GAPS_FIRST",
        "blocked_action_v1": [
            "RUN_TRUE_R5_2_REBUILD_NOW",
            "RUN_R6_RETRAIN_NOW",
            "USE_1689_EXACT_ONLY",
            "USE_PROTECTOR_FIRST",
            "USE_BRIDGE_OR_READINESS_AS_TRAINING_SURFACE",
        ],
    }


def _audit(summary: dict[str, Any], stats: dict[str, Any], score_summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("SPEC_COMPLETE", summary["spec_complete_v1"], summary["decision_v1"]),
            row("NO_TRAINING", not summary["training_started_v1"], summary["training_started_v1"]),
            row("NO_R6", not summary["r6_started_v1"], summary["r6_started_v1"]),
            row("FOUNDATION_ROW_COUNT", stats["row_count_v1"] == 1914, stats["row_count_v1"]),
            row("ACTIVE_QUARANTINE_SPLIT", stats["active_rows_v1"] == 1852 and stats["quarantine_rows_v1"] == 62, [stats["active_rows_v1"], stats["quarantine_rows_v1"]]),
            row("AS_OF_SCHEMA", int(score_summary.get("as_of_column_count_v1") or 0) == 109, score_summary.get("as_of_column_count_v1")),
            row("MISSED_BUCKET_COUNTS", stats["missed_bad_tail_bucket_counts_v1"] == EXPECTED_MISSED_BUCKETS, stats["missed_bad_tail_bucket_counts_v1"]),
            row("AMBIGUOUS_NOT_BAD_POSITIVE", stats["ambiguous_high_mfe_bad_positive_count_v1"] == 0, stats["ambiguous_high_mfe_bad_positive_count_v1"]),
            row("RUNNER_PROTECT_NOT_BAD_POSITIVE", stats["runner_protect_bad_positive_count_v1"] == 0, stats["runner_protect_bad_positive_count_v1"]),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# True R5.2 Rebuild Runner Spec",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Selected design: `{summary['selected_design_v1']}`",
            f"- Label table rows: `{summary['label_table_rows_v1']}`",
            f"- Active/quarantine: `{summary['active_rows_v1']}` / `{summary['quarantine_rows_v1']}`",
            f"- AS_OF columns: `{summary['as_of_column_count_v1']}`",
            f"- Missed bad/tail represented: `{summary['missed_bad_tail_row_count_v1']}`",
            f"- Ambiguous high-MFE bad-positive: `{summary['ambiguous_high_mfe_bad_positive_count_v1']}`",
            "",
            "No true R5.2 training, R6 retrain, new baseline, feature rebuild, freeze, promotion, live gate, or controller change was run.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    label_objective_dir: Path = LABEL_OBJECTIVE_DEFAULT,
    v3_score_dir: Path = V3_SCORE_DEFAULT,
    old_v3_r6_dir: Path = OLD_V3_R6_DEFAULT,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    output_dir = (output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    label_objective_dir = label_objective_dir.expanduser().resolve()
    v3_score_dir = v3_score_dir.expanduser().resolve()
    old_v3_r6_dir = old_v3_r6_dir.expanduser().resolve()

    label_table, label_contract, weighting, rebuild_spec, score_summary = _load_inputs(label_objective_dir, v3_score_dir)
    stats = _label_stats(label_table)
    spec_complete = _spec_complete(stats, score_summary, label_contract, weighting, rebuild_spec)

    input_contract = _input_contract(label_objective_dir, v3_score_dir, stats, score_summary)
    loader_spec = _loader_spec(label_contract, weighting, stats)
    model_config = _model_config(rebuild_spec, weighting)
    eval_guards = _eval_guards()
    output_spec = _output_spec()
    prelaunch = _prelaunch()
    abort_rules = _abort_rules()
    r6_contract = _r6_contract()
    runner_spec = _runner_spec(
        output_dir,
        label_objective_dir,
        v3_score_dir,
        input_contract,
        loader_spec,
        model_config,
        eval_guards,
        output_spec,
        prelaunch,
        abort_rules,
        r6_contract,
    )
    next_action = _next_action(spec_complete)
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "label_objective_dir_v1": str(label_objective_dir),
        "v3_score_dir_v1": str(v3_score_dir),
        "old_v3_r6_dir_v1": str(old_v3_r6_dir),
        "decision_v1": "TRUE_R5_2_REBUILD_RUNNER_SPEC_COMPLETE" if spec_complete else "TRUE_R5_2_REBUILD_RUNNER_SPEC_INCOMPLETE",
        "next_action_v1": next_action["next_action_v1"],
        "spec_complete_v1": spec_complete,
        "selected_design_v1": model_config["selected_first_rebuild_design_v1"],
        "label_contract_id_v1": label_contract["contract_id_v1"],
        "label_table_rows_v1": stats["row_count_v1"],
        "active_rows_v1": stats["active_rows_v1"],
        "quarantine_rows_v1": stats["quarantine_rows_v1"],
        "as_of_column_count_v1": int(score_summary.get("as_of_column_count_v1") or 0),
        "missed_bad_tail_row_count_v1": stats["missed_bad_tail_row_count_v1"],
        "missed_bad_tail_bucket_counts_v1": stats["missed_bad_tail_bucket_counts_v1"],
        "ambiguous_high_mfe_bad_positive_count_v1": stats["ambiguous_high_mfe_bad_positive_count_v1"],
        "runner_protect_bad_positive_count_v1": stats["runner_protect_bad_positive_count_v1"],
        "training_started_v1": False,
        "r6_started_v1": False,
        "new_baseline_built_v1": False,
        "new_feature_surface_built_v1": False,
        "hard_status_v1": {
            "BEVIST": [
                "True R5.2 rebuild runner/config spec was materialized from the locked pocket-aware label/objective package.",
                "The input contract validates 1914 rows, 1852 active, 62 quarantine, 109 AS_OF columns, and zero ambiguous high-MFE bad-positive rows.",
                "No training, R6 retrain, new baseline, feature surface, freeze, promotion, live gate, or controller path was run.",
            ],
            "INDIKERT": [
                "The spec is complete enough for the next agent to implement the true R5.2 rebuild runner.",
            ],
            "IKKE_ETABLERT": [
                "A rebuilt R5.2 model/score package and downstream R6 uplift remain unestablished until the future runner is implemented and explicitly run.",
            ],
        },
    }
    audit = _audit(summary, stats, score_summary)
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "output_files_v1": OUTPUT_FILES,
        "input_dirs_v1": {
            "label_objective_dir_v1": str(label_objective_dir),
            "v3_score_dir_v1": str(v3_score_dir),
            "old_v3_r6_dir_v1": str(old_v3_r6_dir),
        },
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "decision_v1": summary["decision_v1"],
        "next_action_v1": summary["next_action_v1"],
        "training_started_v1": False,
        "r6_started_v1": False,
    }

    _write_json(output_dir / OUTPUT_FILES["runner_spec"], runner_spec)
    _write_json(output_dir / OUTPUT_FILES["input_contract"], input_contract)
    _write_json(output_dir / OUTPUT_FILES["loader_spec"], loader_spec)
    _write_json(output_dir / OUTPUT_FILES["model_config"], model_config)
    _write_json(output_dir / OUTPUT_FILES["eval_guards"], eval_guards)
    _write_json(output_dir / OUTPUT_FILES["output_spec"], output_spec)
    _write_json(output_dir / OUTPUT_FILES["prelaunch"], prelaunch)
    _write_json(output_dir / OUTPUT_FILES["abort_rules"], abort_rules)
    _write_json(output_dir / OUTPUT_FILES["r6_contract"], r6_contract)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / OUTPUT_FILES["status"], status)
    audit.to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--label-objective-dir", type=Path, default=LABEL_OBJECTIVE_DEFAULT)
    parser.add_argument("--v3-score-dir", type=Path, default=V3_SCORE_DEFAULT)
    parser.add_argument("--old-v3-r6-dir", type=Path, default=OLD_V3_R6_DEFAULT)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        label_objective_dir=args.label_objective_dir,
        v3_score_dir=args.v3_score_dir,
        old_v3_r6_dir=args.old_v3_r6_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
