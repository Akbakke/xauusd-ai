#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "DESIGN_R5_2_OBJECTIVE_V2_REBUILD_NEXT_V1"

DEFAULT_INVESTIGATION_DIR = (
    DEFAULT_REPORTS_ROOT / "INVESTIGATE_R5_2_OBJECTIVE_V2_OR_R6_HEAD_RECALL_NEXT_V1_20260426T_LOCK"
)
DEFAULT_LABEL_OBJECTIVE_DIR = DEFAULT_REPORTS_ROOT / "FIX_R5_2_LABEL_OBJECTIVE_FIRST_V1_20260426T_LOCK"
DEFAULT_RESCUE_R6_DIR = DEFAULT_REPORTS_ROOT / "RUN_R6_RETRAIN_FROM_TRUE_R5_2_RESCUE_PACKAGE_V1_20260426T_EXPLICIT"

OUTPUT_FILES = {
    "design_lock": "r5_2_objective_v2_design_lock_v1.json",
    "label_contract": "r5_2_objective_v2_label_contract_v1.json",
    "weight_cost": "r5_2_objective_v2_weight_and_cost_spec_v1.json",
    "architecture": "r5_2_objective_v2_model_architecture_spec_v1.json",
    "base_contract": "r5_2_objective_v2_base_membership_contract_v1.json",
    "target_table": "r5_2_objective_v2_target_table_spec_v1.json",
    "feature_use": "r5_2_objective_v2_existing_feature_use_spec_v1.csv",
    "parallel_run": "r5_2_objective_v2_parallel_rebuild_run_spec_v1.json",
    "eval_gate": "r5_2_objective_v2_eval_and_gate_spec_v1.json",
    "runner_lock": "r5_2_objective_v2_next_runner_spec_lock_v1.json",
    "next_action": "next_action_lock_v1.json",
    "summary": "summary_v1.json",
    "report": "report_v1.md",
    "manifest": "manifest_v1.json",
    "status": "status_v1.json",
    "audit": "consistency_audit_v1.csv",
}

LABEL_TABLE = "r5_2_pocket_label_table_v1.csv"
INVESTIGATION_SUMMARY = "summary_v1.json"
GAP_MAP = "post_rescue_recall_gap_map_v1.csv"
OBJECTIVE_SCAN = "r5_2_objective_v2_opportunity_scan_v1.json"
DECISION_MATRIX = "r5_2_v2_vs_r6_outside_base_decision_matrix_v1.json"

DESIGN_ID = "TWO_STAGE_RECALL_WITH_HARD_PROTECTION_VETO"
NEXT_ACTION = "IMPLEMENT_R5_2_OBJECTIVE_V2_PARALLEL_REBUILD_RUNNER"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if pd.isna(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).to_dict().items()}


def _load_inputs(
    investigation_dir: Path,
    label_objective_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame]:
    summary = _read_json(investigation_dir / INVESTIGATION_SUMMARY)
    gap_map = pd.read_csv(investigation_dir / GAP_MAP)
    objective_scan = _read_json(investigation_dir / OBJECTIVE_SCAN)
    decision_matrix = _read_json(investigation_dir / DECISION_MATRIX)
    label_table = pd.read_csv(label_objective_dir / LABEL_TABLE)
    return summary, gap_map, objective_scan, decision_matrix, label_table


def _evidence(summary: dict[str, Any], gap_map: pd.DataFrame, label_table: pd.DataFrame) -> dict[str, Any]:
    return {
        "current_r6_rescue_bad_tail_v1": summary.get("current_r6_rescue_bad_tail_v1", [88, 57]),
        "wednesday_bad_tail_v1": summary.get("wednesday_bad_tail_v1", [180, 149]),
        "gap_to_wednesday_bad_tail_v1": summary.get("gap_to_wednesday_bad_tail_v1", [92, 92]),
        "gap_map_rows_v1": int(len(gap_map)),
        "missed_bad_label_rows_v1": int(_bool_series(gap_map, "label_should_not_take_v1").sum()),
        "missed_tail_rows_v1": int(_bool_series(gap_map, "tail_10_50_mfe_v1").sum()),
        "gap_bucket_counts_v1": _value_counts(gap_map, "post_rescue_gap_bucket_v1"),
        "r6_signal_class_counts_v1": _value_counts(gap_map, "r6_head_signal_class_v1"),
        "label_table_rows_v1": int(len(label_table)),
        "v1_label_bucket_counts_v1": _value_counts(label_table, "new_r5_2_label_bucket_v1"),
        "fifty_plus_rows_v1": int(_bool_series(label_table, "fifty_plus_mfe_v1").sum()),
        "hundred_plus_rows_v1": int(_bool_series(label_table, "hundred_plus_mfe_v1").sum()),
        "two_hundred_plus_rows_v1": int(_bool_series(label_table, "two_hundred_plus_mfe_v1").sum()),
        "strongest_winner_rows_v1": int(_bool_series(label_table, "strongest_winner_path_v1").sum()),
        "repaired_like_rows_v1": int(_bool_series(label_table, "r6_label_repaired_165_like_runner_v1").sum()),
        "runner_near_miss_rows_v1": int(_bool_series(label_table, "r6_label_runner_near_miss_v1").sum()),
        "ambiguous_high_mfe_rows_v1": int(
            label_table.get("new_r5_2_label_bucket_v1", pd.Series("", index=label_table.index))
            .eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD")
            .sum()
        ),
        "runner_protect_rows_v1": int(
            label_table.get("new_r5_2_label_bucket_v1", pd.Series("", index=label_table.index))
            .eq("RUNNER_PROTECT_TARGET")
            .sum()
        ),
        "tail_control_rows_v1": int(
            label_table.get("new_r5_2_label_bucket_v1", pd.Series("", index=label_table.index))
            .eq("TAIL_CONTROL_TARGET")
            .sum()
        ),
        "strong_bad_rows_v1": int(
            label_table.get("new_r5_2_label_bucket_v1", pd.Series("", index=label_table.index))
            .eq("STRONG_BAD_BLOCK_TARGET")
            .sum()
        ),
    }


def _design_lock(objective_scan: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    raw_findings = objective_scan.get("raw_true_rebuild_findings_v1", {})
    return {
        "layer_name": "R5_2_OBJECTIVE_V2_DESIGN_LOCK_V1",
        "design_id_v1": DESIGN_ID,
        "stage_1_recall_v1": {
            "purpose_v1": "Find bad/tail/risky candidates with higher recall than V1 true rebuild and V3/rescue.",
            "heads_v1": [
                "bad recall eligibility",
                "10-50 tail recall eligibility",
                "risky attention",
            ],
            "raw_recall_head_exposed_to_r6_v1": False,
        },
        "stage_2_hard_protection_veto_v1": {
            "purpose_v1": "Stop high-MFE, runner, repaired, strongest-winner, forensic, and ambiguous danger before base membership reaches R6.",
            "veto_is_hard_contract_v1": True,
            "protection_not_soft_score_only_v1": True,
        },
        "what_v2_changes_from_v1_true_rebuild_v1": [
            "V1 allowed a raw learned base to include safety-damage rows before post-score rescue.",
            "V2 separates recall heads from protection heads and only materializes final base after hard veto.",
            "Ambiguous/high-MFE rows become protected/negative by contract instead of monitor-only leakage risk.",
            "R6 receives only post-veto `r5_2_v2_final_base_membership`, never raw recall membership.",
        ],
        "why_raw_true_v1_failed_safety_v1": {
            "raw_true_findings_v1": raw_findings,
            "root_v1": "Protection was a learned score and post-hoc guard, not the base-construction contract itself.",
        },
        "why_rescue_only_tiny_uplift_v1": {
            "rescued_bad_tail_v1": raw_findings.get("rescued_bad_tail_v1", [88, 57]),
            "reason_v1": "The safe rescue had to discard most raw true recall signal to remove high-MFE/winner/ambiguous leakage.",
        },
        "why_r6_outside_base_is_not_next_v1": {
            "safe_outside_base_rule_count_v1": 0,
            "positive_rules_tested_v1": 122,
            "reason_v1": "Existing R6 heads outside rescued R5.2-base had no safe positive recovery rule; R6 needs a cleaner upstream R5.2 eligibility/protection base first.",
        },
        "evidence_counts_v1": evidence,
        "no_training_v1": True,
        "no_r6_run_v1": True,
        "no_new_baseline_v1": True,
        "no_new_feature_surface_v1": True,
    }


def _label_contract() -> dict[str, Any]:
    buckets = [
        {
            "bucket_v1": "BAD_RECALL_POSITIVE",
            "definition_v1": "Strong should-not-take or bad-risk rows with low high-MFE/runner/winner risk.",
            "input_fields_v1": ["label_should_not_take_v1", "r5_2_label_bad_blocker_v1", "MAE/MFE pocket flags", "winner protection flags"],
            "training_target_role_v1": "positive target for bad recall head",
            "sample_weight_v1": "profile.bad_weight_v1",
            "protection_veto_role_v1": "cannot override hard protection veto",
            "eval_role_v1": "primary bad recall and precision check",
            "risk_v1": "over-rewarding high-MFE ambiguous rows if protection target is wrong",
        },
        {
            "bucket_v1": "TAIL_RECALL_POSITIVE",
            "definition_v1": "10-50 MFE tail-control rows that are not 50+/100+/200+ or runner candidates.",
            "input_fields_v1": ["tail_10_50_mfe_v1", "fifty_plus_mfe_v1", "hundred_plus_mfe_v1", "two_hundred_plus_mfe_v1", "runner flags"],
            "training_target_role_v1": "positive target for tail recall head",
            "sample_weight_v1": "profile.tail_weight_v1",
            "protection_veto_role_v1": "vetoed if high-MFE/winner/runner protection is active",
            "eval_role_v1": "10-50 tail help without 50+/100+/200+ damage",
            "risk_v1": "tail-like runner seeds leaking into base",
        },
        {
            "bucket_v1": "RISKY_ATTENTION_POSITIVE",
            "definition_v1": "Risky rows that deserve R5.2 attention but should not create final base alone without bad/tail confirmation.",
            "input_fields_v1": ["r6_label_risky_allow_v1", "r5/r5.1 risk signals", "bad/tail confirmation scores"],
            "training_target_role_v1": "positive target for risky attention head",
            "sample_weight_v1": "profile.risky_weight_v1",
            "protection_veto_role_v1": "never overrides protection",
            "eval_role_v1": "supporting confirmation for safe base expansion",
            "risk_v1": "attention head becoming a broad blocker if used alone",
        },
        {
            "bucket_v1": "HARD_PROTECT_NEGATIVE",
            "definition_v1": "Repaired-like, strongest-winner, 100+/200+, forensic repaired trade, runner-protect, and clear high-MFE winner rows.",
            "input_fields_v1": [
                "r6_label_repaired_165_like_runner_v1",
                "strongest_winner_path_v1",
                "hundred_plus_mfe_v1",
                "two_hundred_plus_mfe_v1",
                "runner-protect flags",
                "forensic repaired candidate id",
            ],
            "training_target_role_v1": "negative for recall heads and positive for hard winner protection head",
            "sample_weight_v1": "profile.hard_protect_weight_v1",
            "protection_veto_role_v1": "mandatory veto source",
            "eval_role_v1": "hard safety no-go checks",
            "risk_v1": "if underweighted, raw recall can leak winners into base",
        },
        {
            "bucket_v1": "AMBIGUOUS_HIGH_MFE_PROTECTED",
            "definition_v1": "Rows with high-MFE or runner-like upside that may look bad on some signals but are unsafe as bad-positive labels.",
            "input_fields_v1": ["fifty_plus_mfe_v1", "peak_mfe_bps_v1", "runner near-miss flags", "ambiguous high-MFE bucket"],
            "training_target_role_v1": "not bad-positive; positive for high-MFE ambiguous protection head or monitor-only with hard exclusion",
            "sample_weight_v1": "profile.ambiguous_high_mfe_protection_weight_v1",
            "protection_veto_role_v1": "veto if risk score/flag is high",
            "eval_role_v1": "ambiguous leakage must remain zero unless explicitly safe-proven",
            "risk_v1": "main raw true V1 failure mode",
        },
        {
            "bucket_v1": "MONITOR_ONLY",
            "definition_v1": "Rows that are not reliable enough to drive recall or protection targets.",
            "input_fields_v1": ["remaining active/quarantine rows", "weak/unknown pockets"],
            "training_target_role_v1": "excluded from positive target pressure or low-weight neutral",
            "sample_weight_v1": "profile.monitor_weight_v1",
            "protection_veto_role_v1": "not a base source",
            "eval_role_v1": "coverage and drift monitoring",
            "risk_v1": "turning weak evidence into training pressure",
        },
    ]
    return {
        "layer_name": "R5_2_OBJECTIVE_V2_LABEL_CONTRACT_V1",
        "contract_id_v1": "R5_2_OBJECTIVE_V2_TWO_STAGE_RECALL_HARD_PROTECTION_LABEL_CONTRACT",
        "buckets_v1": buckets,
        "ambiguous_high_mfe_bad_positive_allowed_v1": False,
        "hard_protect_negative_can_be_final_base_positive_v1": False,
        "training_surface_v1": "canonical Monday foundation 1914 rows; no 1689 exact-only, bridge/readiness, protector, narrow, or diagnostic surface",
    }


def _weight_profiles() -> list[dict[str, Any]]:
    return [
        {
            "profile_id_v1": "V2_BALANCED_STRICT_PROTECT",
            "bad_weight_v1": 3.5,
            "tail_weight_v1": 3.0,
            "risky_weight_v1": 1.5,
            "runner_protect_weight_v1": 16.0,
            "ambiguous_high_mfe_protection_weight_v1": 24.0,
            "hard_protect_weight_v1": 32.0,
            "monitor_weight_v1": 0.25,
            "expected_effect_v1": "Balanced recall uplift with materially stronger protection than V1.",
            "risk_v1": "May still be too conservative if protection dominates all ambiguous-but-safe rows.",
        },
        {
            "profile_id_v1": "V2_STRONG_BAD_TAIL_WITH_HARD_VETO",
            "bad_weight_v1": 5.0,
            "tail_weight_v1": 4.5,
            "risky_weight_v1": 2.0,
            "runner_protect_weight_v1": 18.0,
            "ambiguous_high_mfe_protection_weight_v1": 30.0,
            "hard_protect_weight_v1": 40.0,
            "monitor_weight_v1": 0.20,
            "expected_effect_v1": "Tests whether stronger recall can recover a meaningful part of the 92/92 R6 gap while hard veto protects winners.",
            "risk_v1": "Most likely profile to overfit recall if veto calibration is weak.",
        },
        {
            "profile_id_v1": "V2_PROTECTION_HEAVY",
            "bad_weight_v1": 3.0,
            "tail_weight_v1": 2.8,
            "risky_weight_v1": 1.2,
            "runner_protect_weight_v1": 24.0,
            "ambiguous_high_mfe_protection_weight_v1": 36.0,
            "hard_protect_weight_v1": 50.0,
            "monitor_weight_v1": 0.15,
            "expected_effect_v1": "Best safety stress profile; should prevent raw true V1 style leakage.",
            "risk_v1": "May repeat tiny-uplift behavior if recall pressure is too weak.",
        },
        {
            "profile_id_v1": "V2_TAIL_RECOVERY_FOCUSED",
            "bad_weight_v1": 3.2,
            "tail_weight_v1": 5.0,
            "risky_weight_v1": 1.8,
            "runner_protect_weight_v1": 20.0,
            "ambiguous_high_mfe_protection_weight_v1": 32.0,
            "hard_protect_weight_v1": 44.0,
            "monitor_weight_v1": 0.20,
            "expected_effect_v1": "Specifically tests whether 10-50 tail recall can improve without touching 50+/100+/200+ winners.",
            "risk_v1": "Tail-like runner seeds can be tempting unless protection heads separate them.",
        },
        {
            "profile_id_v1": "V2_RECALL_LIGHT_ULTRA_SAFE",
            "bad_weight_v1": 2.5,
            "tail_weight_v1": 2.2,
            "risky_weight_v1": 1.0,
            "runner_protect_weight_v1": 30.0,
            "ambiguous_high_mfe_protection_weight_v1": 45.0,
            "hard_protect_weight_v1": 60.0,
            "monitor_weight_v1": 0.10,
            "expected_effect_v1": "Ultra-safe lower-bound check; useful to prove whether protection can be clean before adding recall pressure.",
            "risk_v1": "Likely too weak on recall but valuable as safety anchor.",
        },
        {
            "profile_id_v1": "V2_AMBIGUOUS_HARD_NEGATIVE_STRESS",
            "bad_weight_v1": 3.8,
            "tail_weight_v1": 3.2,
            "risky_weight_v1": 1.4,
            "runner_protect_weight_v1": 22.0,
            "ambiguous_high_mfe_protection_weight_v1": 55.0,
            "hard_protect_weight_v1": 45.0,
            "monitor_weight_v1": 0.15,
            "expected_effect_v1": "Isolates the ambiguous/high-MFE failure mode from raw true V1.",
            "risk_v1": "Can over-exclude genuinely safe low-MFE tail rows if ambiguity tagging is too broad.",
        },
        {
            "profile_id_v1": "V2_BAD_RECALL_FOCUSED_WITH_STRONG_VETO",
            "bad_weight_v1": 5.5,
            "tail_weight_v1": 3.0,
            "risky_weight_v1": 1.5,
            "runner_protect_weight_v1": 22.0,
            "ambiguous_high_mfe_protection_weight_v1": 34.0,
            "hard_protect_weight_v1": 48.0,
            "monitor_weight_v1": 0.20,
            "expected_effect_v1": "Tests bad-block recall without letting tail objective dominate.",
            "risk_v1": "Bad head can confuse high-MFE ambiguous rows unless hard veto is calibrated.",
        },
    ]


def _weight_cost_spec() -> dict[str, Any]:
    return {
        "layer_name": "R5_2_OBJECTIVE_V2_WEIGHT_AND_COST_SPEC_V1",
        "v1_reference_weights_v1": {
            "bad_weight_v1": 3.0,
            "tail_weight_v1": 2.5,
            "runner_protect_weight_v1": 10.0,
            "hard_protect_weight_v1": 20.0,
        },
        "v2_principle_v1": "Run a small controlled profile grid; do not pick one arbitrary config before evidence.",
        "candidate_weight_profiles_v1": _weight_profiles(),
        "selection_rule_v1": "Choose only variants that beat V3/rescue materially and pass hard protection gates.",
    }


def _architecture_spec() -> dict[str, Any]:
    return {
        "layer_name": "R5_2_OBJECTIVE_V2_MODEL_ARCHITECTURE_SPEC_V1",
        "architecture_id_v1": DESIGN_ID,
        "model_family_v1": "deterministic XGB-style multi-head stack matching current R5.2/R6 tooling",
        "training_heads_v1": {
            "recall_heads_v1": [
                "r5_2_v2_bad_recall_score",
                "r5_2_v2_tail_recall_score",
                "r5_2_v2_risky_attention_score",
            ],
            "protection_heads_v1": [
                "r5_2_v2_runner_protection_score",
                "r5_2_v2_high_mfe_ambiguous_protection_score",
                "r5_2_v2_hard_winner_protection_score",
            ],
        },
        "final_outputs_v1": [
            "r5_2_v2_bad_recall_score",
            "r5_2_v2_tail_recall_score",
            "r5_2_v2_risky_attention_score",
            "r5_2_v2_runner_protection_score",
            "r5_2_v2_high_mfe_ambiguous_protection_score",
            "r5_2_v2_hard_winner_protection_score",
            "r5_2_v2_base_membership_pre_veto",
            "r5_2_v2_hard_protection_veto",
            "r5_2_v2_final_base_membership",
        ],
        "combination_logic_v1": {
            "pre_veto_v1": "bad_recall high OR tail_recall high OR risky_attention high with bad/tail confirmation",
            "hard_veto_v1": "hard_winner OR high_mfe_ambiguous OR runner_protection OR explicit repaired/strongest/100+/200+/forensic/runner-protect flag",
            "final_base_v1": "pre_veto AND NOT hard_veto",
        },
        "conflict_resolution_v1": "Protection always wins over recall. A row with high recall and high protection is vetoed and reported as a conflict, not exposed to R6.",
        "raw_recall_base_allowed_downstream_v1": False,
    }


def _base_contract() -> dict[str, Any]:
    return {
        "layer_name": "R5_2_OBJECTIVE_V2_BASE_MEMBERSHIP_CONTRACT_V1",
        "contract_id_v1": "R5_2_V2_BASE_MEMBERSHIP_TWO_STAGE_RECALL_HARD_PROTECTION_VETO",
        "pre_veto_base_rule_v1": {
            "ADDED_BY_BAD_RECALL": "r5_2_v2_bad_recall_score >= variant.bad_recall_threshold_v1",
            "ADDED_BY_TAIL_RECALL": "r5_2_v2_tail_recall_score >= variant.tail_recall_threshold_v1",
            "ADDED_BY_RISKY_CONFIRMATION": "r5_2_v2_risky_attention_score >= variant.risky_attention_threshold_v1 AND (bad_recall or tail_recall reaches confirmation threshold)",
        },
        "veto_rule_v1": {
            "VETO_HARD_WINNER": "r5_2_v2_hard_winner_protection_score >= variant.hard_winner_veto_threshold_v1 OR 100+/200+/strongest/forensic flag active",
            "VETO_HIGH_MFE_AMBIGUOUS": "r5_2_v2_high_mfe_ambiguous_protection_score >= variant.ambiguous_veto_threshold_v1 OR ambiguous high-MFE protected bucket active",
            "VETO_RUNNER_PROTECT": "r5_2_v2_runner_protection_score >= variant.runner_veto_threshold_v1 OR runner-protect bucket active",
            "VETO_REPAIRED_OR_STRONGEST": "repaired-like, strongest-winner, 100+, 200+, or forensic repaired trade flag active",
            "MONITOR_ONLY_NOT_BASE": "monitor-only rows cannot become base unless a future explicit contract changes that",
        },
        "final_base_rule_v1": "r5_2_v2_base_membership_pre_veto AND NOT r5_2_v2_hard_protection_veto",
        "reason_codes_v1": [
            "ADDED_BY_BAD_RECALL",
            "ADDED_BY_TAIL_RECALL",
            "ADDED_BY_RISKY_CONFIRMATION",
            "VETO_HARD_WINNER",
            "VETO_HIGH_MFE_AMBIGUOUS",
            "VETO_RUNNER_PROTECT",
            "VETO_REPAIRED_OR_STRONGEST",
            "MONITOR_ONLY_NOT_BASE",
        ],
        "downstream_flag_v1": "r5_2_v2_final_base_membership",
        "forbidden_downstream_flags_v1": [
            "r5_2_rebuilt_base_membership_v1",
            "raw_true_base_membership_v1",
            "r5_2_true_rescue_base_membership_v1",
            "r5_2_v3_base_flag_v1",
            "r5_2_v2_base_flag_v1",
            "r5_2_v1_base_flag_v1",
            "r5_2_original_base_flag_v1",
        ],
    }


def _target_table_spec(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "R5_2_OBJECTIVE_V2_TARGET_TABLE_SPEC_V1",
        "future_target_table_id_v1": "r5_2_objective_v2_target_table_v1",
        "row_coverage_v1": {
            "required_rows_v1": 1914,
            "observed_v1_source_label_rows_v1": evidence["label_table_rows_v1"],
            "required_key_columns_v1": ["candidate_uid", "trade_uid", "decision_timestamp"],
            "full_row_coverage_required_v1": True,
        },
        "required_columns_per_row_v1": [
            "candidate_uid",
            "trade_uid",
            "decision_timestamp",
            "original_bucket",
            "v2_bucket",
            "bad_recall_target",
            "tail_recall_target",
            "risky_attention_target",
            "runner_protection_target",
            "high_mfe_ambiguous_protection_target",
            "hard_winner_protection_target",
            "sample_weight",
            "protection_weight",
            "monitor_only_flag",
            "reason",
        ],
        "special_population_coverage_v1": {
            "post_rescue_gap_not_in_rescued_base_v1": evidence["gap_bucket_counts_v1"].get("NOT_IN_RESCUED_R5_2_BASE", 0),
            "post_rescue_gap_dangerous_or_protected_v1": evidence["gap_bucket_counts_v1"].get("DANGEROUS_OR_PROTECTED", 0),
            "post_rescue_gap_r6_could_recover_but_base_blocks_v1": evidence["gap_bucket_counts_v1"].get("R6_COULD_RECOVER_BUT_BASE_GATE_BLOCKS", 0),
            "ambiguous_high_mfe_v1": evidence["ambiguous_high_mfe_rows_v1"],
            "runner_protect_v1": evidence["runner_protect_rows_v1"],
            "fifty_plus_v1": evidence["fifty_plus_rows_v1"],
            "hundred_plus_v1": evidence["hundred_plus_rows_v1"],
            "two_hundred_plus_v1": evidence["two_hundred_plus_rows_v1"],
            "strongest_winner_v1": evidence["strongest_winner_rows_v1"],
            "repaired_like_v1": evidence["repaired_like_rows_v1"],
            "tail_control_v1": evidence["tail_control_rows_v1"],
            "strong_bad_v1": evidence["strong_bad_rows_v1"],
        },
        "mapping_rules_v1": {
            "BAD_RECALL_POSITIVE": "low winner-risk should-not-take/bad-risk rows",
            "TAIL_RECALL_POSITIVE": "10-50 tail rows excluding high-MFE/winner/runner protection",
            "RISKY_ATTENTION_POSITIVE": "risky rows as attention/confirmation target, not standalone base",
            "HARD_PROTECT_NEGATIVE": "repaired/strongest/100+/200+/forensic/clear high-MFE winner/runner-protect",
            "AMBIGUOUS_HIGH_MFE_PROTECTED": "high-MFE ambiguous rows; not bad-positive",
            "MONITOR_ONLY": "weak or unresolved rows",
        },
        "hard_validation_v1": [
            "target table rows == 1914",
            "all required keys non-null and unique enough for candidate/trade alignment",
            "ambiguous high-MFE bad-positive count == 0",
            "runner-protect bad-positive count == 0",
            "hard protect negatives have protection target and veto eligibility",
        ],
    }


def _feature_use_spec() -> pd.DataFrame:
    rows = [
        {
            "feature_family_v1": "existing_109_as_of_features",
            "source_asset_v1": "canonical Monday foundation / score package",
            "role_v1": "REUSE_NOW",
            "legal_for_entry_v1": True,
            "reason_v1": "Canonical AS_OF feature schema; no rebuild needed.",
        },
        {
            "feature_family_v1": "r5_score_signals",
            "source_asset_v1": "existing R5 score layer",
            "role_v1": "REUSE_NOW",
            "legal_for_entry_v1": True,
            "reason_v1": "Already AS_OF-style score signals used upstream.",
        },
        {
            "feature_family_v1": "r5_1_score_signals",
            "source_asset_v1": "existing R5.1 score layer",
            "role_v1": "REUSE_NOW",
            "legal_for_entry_v1": True,
            "reason_v1": "Supports runner and bad/tail separation without new feature surface.",
        },
        {
            "feature_family_v1": "legal_r5_2_rebuild_inputs",
            "source_asset_v1": "current R5.2 rebuild input contract",
            "role_v1": "REUSE_NOW",
            "legal_for_entry_v1": True,
            "reason_v1": "Existing legal R5.2 inputs remain allowed; outputs are not training labels.",
        },
        {
            "feature_family_v1": "true_r5_2_v1_scores",
            "source_asset_v1": "TRUE_R5_2_REBUILD_RUNNER_V1_20260426T_EXPLICIT_TRAINING",
            "role_v1": "REUSE_FOR_EVAL_AND_INITIALIZATION_REFERENCE_ONLY",
            "legal_for_entry_v1": False,
            "reason_v1": "Useful to diagnose V1 leakage; not a canonical training feature for V2 unless explicitly promoted by runner spec.",
        },
        {
            "feature_family_v1": "underused_runner_high_mfe_protection_signals",
            "source_asset_v1": "existing AS_OF/R5/R5.1/path-derived legal pre-entry signals",
            "role_v1": "REUSE_NOW_IF_ALREADY_ASOF_AND_KEY_ALIGNED",
            "legal_for_entry_v1": True,
            "reason_v1": "V1 underused protection separation; V2 should wire legal protection signals before inventing new features.",
        },
        {
            "feature_family_v1": "path_dynamics_or_pre_rl_entry_proxies",
            "source_asset_v1": "existing path-dynamics/pre-RL assets",
            "role_v1": "REUSE_ONLY_IF_ASOF_AND_PRE_ENTRY",
            "legal_for_entry_v1": "CONDITIONAL",
            "reason_v1": "Allowed only when physically AS_OF and not hindsight/exit/management truth.",
        },
        {
            "feature_family_v1": "hindsight_path_labels",
            "source_asset_v1": "hindsight/backfill/eval surfaces",
            "role_v1": "EVAL_ONLY_DO_NOT_USE_AS_MODEL_FEATURE",
            "legal_for_entry_v1": False,
            "reason_v1": "Can define labels/eval guards; cannot be model features.",
        },
        {
            "feature_family_v1": "exit_management_truth",
            "source_asset_v1": "exit/path/management assets",
            "role_v1": "DO_NOT_USE_FOR_ENTRY",
            "legal_for_entry_v1": False,
            "reason_v1": "Exit/management truth is forbidden as entry model feature.",
        },
        {
            "feature_family_v1": "bridge_readiness_1689_protector_narrow_diagnostic",
            "source_asset_v1": "diagnostic-only surfaces",
            "role_v1": "FORBIDDEN",
            "legal_for_entry_v1": False,
            "reason_v1": "Not canonical R6/R5.2 baseline or training surface.",
        },
    ]
    return pd.DataFrame(rows)


def _variant_thresholds(profile: dict[str, Any], index: int) -> dict[str, float]:
    if profile["profile_id_v1"] == "V2_RECALL_LIGHT_ULTRA_SAFE":
        return {
            "bad_recall_threshold_v1": 0.78,
            "tail_recall_threshold_v1": 0.76,
            "risky_attention_threshold_v1": 0.72,
            "bad_tail_confirmation_threshold_v1": 0.55,
            "runner_veto_threshold_v1": 0.18,
            "ambiguous_veto_threshold_v1": 0.12,
            "hard_winner_veto_threshold_v1": 0.08,
        }
    if profile["profile_id_v1"] == "V2_STRONG_BAD_TAIL_WITH_HARD_VETO":
        return {
            "bad_recall_threshold_v1": 0.62,
            "tail_recall_threshold_v1": 0.60,
            "risky_attention_threshold_v1": 0.62,
            "bad_tail_confirmation_threshold_v1": 0.45,
            "runner_veto_threshold_v1": 0.24,
            "ambiguous_veto_threshold_v1": 0.16,
            "hard_winner_veto_threshold_v1": 0.10,
        }
    base = 0.70 - min(index, 4) * 0.02
    return {
        "bad_recall_threshold_v1": round(base, 2),
        "tail_recall_threshold_v1": round(base - 0.02, 2),
        "risky_attention_threshold_v1": round(base, 2),
        "bad_tail_confirmation_threshold_v1": 0.50,
        "runner_veto_threshold_v1": 0.20,
        "ambiguous_veto_threshold_v1": 0.14,
        "hard_winner_veto_threshold_v1": 0.10,
    }


def _parallel_run_spec(weight_spec: dict[str, Any]) -> dict[str, Any]:
    variants = []
    for idx, profile in enumerate(weight_spec["candidate_weight_profiles_v1"], start=1):
        variants.append(
            {
                "variant_id_v1": f"R5_2_OBJECTIVE_V2_VARIANT_{idx:02d}_{profile['profile_id_v1']}",
                "weights_v1": profile,
                "model_config_v1": {
                    "architecture_id_v1": DESIGN_ID,
                    "seed_v1": 20260426 + idx,
                    "model_family_v1": "deterministic XGB-style multi-head",
                    "heads_v1": [
                        "bad_recall",
                        "tail_recall",
                        "risky_attention",
                        "runner_protection",
                        "high_mfe_ambiguous_protection",
                        "hard_winner_protection",
                    ],
                },
                "base_membership_rule_v1": "pre_veto recall/risky rule AND NOT hard protection veto",
                "veto_strictness_v1": _variant_thresholds(profile, idx),
                "expected_outputs_v1": [
                    "r5_2_v2_prediction_view_v1.parquet",
                    "r5_2_v2_score_package_v1.parquet",
                    "r5_2_v2_base_membership_v1.parquet",
                    "r5_2_v2_eval_summary_v1.json",
                    "r5_2_v2_downstream_r6_input_manifest_v1.json",
                ],
                "no_go_conditions_v1": [
                    "forbidden feature",
                    "target/key mismatch",
                    "ambiguous leakage",
                    "runner-protect leakage",
                    "100+/200+/strongest/repaired damage",
                    "LOSO collapse",
                ],
            }
        )
    return {
        "layer_name": "R5_2_OBJECTIVE_V2_PARALLEL_REBUILD_RUN_SPEC_V1",
        "run_started_by_this_spec_v1": False,
        "parallel_variant_count_v1": len(variants),
        "variants_v1": variants,
        "shared_inputs_v1": {
            "foundation_rows_v1": 1914,
            "as_of_columns_v1": 109,
            "target_table_v1": "r5_2_objective_v2_target_table_v1",
            "no_new_baseline_v1": True,
            "no_new_feature_surface_v1": True,
        },
        "shared_eval_metrics_v1": [
            "bad recall",
            "tail recall",
            "precision",
            "worst LOSO",
            "50+/100+/200+",
            "strongest-winner",
            "runner near-miss",
            "ambiguous high-MFE leakage",
            "repaired-like leakage",
            "downstream R6 readiness",
        ],
    }


def _eval_gate_spec() -> dict[str, Any]:
    return {
        "layer_name": "R5_2_OBJECTIVE_V2_EVAL_AND_GATE_SPEC_V1",
        "hard_fail_if_v1": [
            "raw base includes repaired-like unsafe row",
            "forensic trade becomes blockable",
            "100+/200+ overlap > 0",
            "strongest-winner overlap > 0",
            "ambiguous high-MFE leakage > 0 unless explicitly safe",
            "runner-protect leakage > 0",
            "50+ overlap > allowed safety cap",
            "worst LOSO collapses",
            "forbidden feature used",
            "target table mismatch",
            "key mismatch",
        ],
        "pass_candidate_must_v1": [
            "beat V3/rescue 88/57 meaningfully",
            "hold hard safety",
            "write downstream R6-ready manifest",
            "expose only post-veto final base to R6",
            "document every vetoed recall/protection conflict",
        ],
        "benchmark_references_v1": {
            "r6_rescue_v1": {"bad_v1": 88, "tail_v1": 57, "precision_v1": 1.0, "worst_loso_v1": 1.0},
            "wednesday_r6_v1": {"bad_v1": 180, "tail_v1": 149, "precision_v1": 0.972972972972973, "worst_loso_v1": 0.9285714285714286},
        },
        "gate_decisions_v1": [
            "R5_2_OBJECTIVE_V2_REBUILD_PASS_READY_FOR_R6",
            "R5_2_OBJECTIVE_V2_SAFE_BUT_TOO_WEAK",
            "R5_2_OBJECTIVE_V2_RECALL_IMPROVES_BUT_SAFETY_FAILS",
            "R5_2_OBJECTIVE_V2_INVALID_FEATURE_OR_SURFACE",
            "R5_2_OBJECTIVE_V2_NOT_ESTABLISHED",
        ],
    }


def _runner_lock(spec_complete: bool) -> dict[str, Any]:
    return {
        "layer_name": "R5_2_OBJECTIVE_V2_NEXT_RUNNER_SPEC_LOCK_V1",
        "decision_v1": "IMPLEMENT_R5_2_OBJECTIVE_V2_PARALLEL_REBUILD_RUNNER" if spec_complete else "FIX_V2_SPEC_GAPS_FIRST",
        "next_actions_allowed_v1": [
            "IMPLEMENT_R5_2_OBJECTIVE_V2_PARALLEL_REBUILD_RUNNER",
        ]
        if spec_complete
        else [
            "FIX_V2_TARGET_TABLE_SPEC_FIRST",
            "FIX_V2_WEIGHT_PROFILE_SPEC_FIRST",
            "FIX_V2_FEATURE_USE_SPEC_FIRST",
        ],
        "blocked_actions_v1": [
            "RUN_R5_2_OBJECTIVE_V2_TRAINING_NOW",
            "RUN_R6_RETRAIN_NOW",
            "BUILD_NEW_BASELINE",
            "BUILD_NEW_FEATURE_SURFACE",
            "USE_RAW_TRUE_UNSAFE_PACKAGE_DIRECTLY",
        ],
    }


def _next_action(spec_complete: bool) -> dict[str, Any]:
    return {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": NEXT_ACTION if spec_complete else "FIX_V2_SPEC_GAPS_FIRST",
        "blocked_action_v1": [
            "RUN_TRUE_R5_2_V2_REBUILD_NOW",
            "RUN_R6_RETRAIN_NOW",
            "BUILD_NEW_BASELINE",
            "BUILD_NEW_FEATURE_SURFACE",
            "USE_1689_EXACT_ONLY",
            "USE_PROTECTOR_FIRST",
            "USE_RAW_TRUE_UNSAFE_PACKAGE_DIRECTLY",
        ],
    }


def _audit(summary: dict[str, Any], evidence: dict[str, Any], weight_spec: dict[str, Any], feature_spec: pd.DataFrame) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence_value: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence_value), sort_keys=True)}

    return pd.DataFrame(
        [
            row("NO_TRAINING", not summary["training_started_v1"], summary["training_started_v1"]),
            row("NO_R6", not summary["r6_started_v1"], summary["r6_started_v1"]),
            row("NO_BASELINE_OR_FEATURE_BUILD", not summary["new_baseline_built_v1"] and not summary["new_feature_surface_built_v1"], [summary["new_baseline_built_v1"], summary["new_feature_surface_built_v1"]]),
            row("DESIGN_LOCKED", summary["design_id_v1"] == DESIGN_ID, summary["design_id_v1"]),
            row("GAP_COUNTS_PRESENT", evidence["gap_bucket_counts_v1"].get("NOT_IN_RESCUED_R5_2_BASE", 0) == 230 and evidence["gap_bucket_counts_v1"].get("DANGEROUS_OR_PROTECTED", 0) == 149, evidence["gap_bucket_counts_v1"]),
            row("WEIGHT_PROFILE_COUNT", len(weight_spec["candidate_weight_profiles_v1"]) >= 5, len(weight_spec["candidate_weight_profiles_v1"])),
            row("FEATURE_FORBIDDEN_PRESENT", "FORBIDDEN" in set(feature_spec["role_v1"]), feature_spec["role_v1"].tolist()),
            row("NEXT_RUNNER_LOCKED", summary["next_action_v1"] == NEXT_ACTION, summary["next_action_v1"]),
        ]
    )


def _report(summary: dict[str, Any], weight_spec: dict[str, Any]) -> str:
    profiles = ", ".join(profile["profile_id_v1"] for profile in weight_spec["candidate_weight_profiles_v1"])
    return "\n".join(
        [
            "# R5.2 Objective V2 Rebuild Design",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Design: `{summary['design_id_v1']}`",
            f"- R6 rescue gap: `{summary['gap_to_wednesday_bad_tail_v1'][0]}/{summary['gap_to_wednesday_bad_tail_v1'][1]}`",
            f"- Gap buckets: `{summary['gap_bucket_counts_v1']}`",
            f"- Weight/config variants: `{summary['v2_variant_count_v1']}`",
            f"- Profiles: `{profiles}`",
            "",
            "V2 is a contract/spec only. No R5.2 training, R6 run, baseline build, feature rebuild, freeze, promo, live gate, or controller change was performed.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    investigation_dir: Path = DEFAULT_INVESTIGATION_DIR,
    label_objective_dir: Path = DEFAULT_LABEL_OBJECTIVE_DIR,
    rescue_r6_dir: Path = DEFAULT_RESCUE_R6_DIR,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    investigation_dir = investigation_dir.expanduser().resolve()
    label_objective_dir = label_objective_dir.expanduser().resolve()
    rescue_r6_dir = rescue_r6_dir.expanduser().resolve()
    output_dir = (output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    investigation_summary, gap_map, objective_scan, decision_matrix, label_table = _load_inputs(investigation_dir, label_objective_dir)
    evidence = _evidence(investigation_summary, gap_map, label_table)

    design_lock = _design_lock(objective_scan, evidence)
    label_contract = _label_contract()
    weight_spec = _weight_cost_spec()
    architecture = _architecture_spec()
    base_contract = _base_contract()
    target_table = _target_table_spec(evidence)
    feature_spec = _feature_use_spec()
    parallel_run = _parallel_run_spec(weight_spec)
    eval_gate = _eval_gate_spec()

    spec_complete = bool(
        investigation_summary.get("decision_v1") == "R5_2_OBJECTIVE_V2_REBUILD"
        and decision_matrix.get("decision_v1") == "R5_2_OBJECTIVE_V2_REBUILD"
        and evidence["gap_bucket_counts_v1"].get("NOT_IN_RESCUED_R5_2_BASE", 0) == 230
        and evidence["gap_bucket_counts_v1"].get("DANGEROUS_OR_PROTECTED", 0) == 149
        and evidence["gap_bucket_counts_v1"].get("R6_COULD_RECOVER_BUT_BASE_GATE_BLOCKS", 0) == 5
        and len(weight_spec["candidate_weight_profiles_v1"]) >= 5
        and len(label_contract["buckets_v1"]) == 6
        and "r5_2_v2_final_base_membership" in architecture["final_outputs_v1"]
    )
    runner_lock = _runner_lock(spec_complete)
    next_action = _next_action(spec_complete)

    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "investigation_dir_v1": str(investigation_dir),
        "label_objective_dir_v1": str(label_objective_dir),
        "rescue_r6_dir_v1": str(rescue_r6_dir),
        "decision_v1": "R5_2_OBJECTIVE_V2_DESIGN_READY_FOR_RUNNER_IMPLEMENTATION" if spec_complete else "R5_2_OBJECTIVE_V2_DESIGN_SPEC_INCOMPLETE",
        "next_action_v1": next_action["next_action_v1"],
        "spec_complete_v1": spec_complete,
        "design_id_v1": DESIGN_ID,
        "current_r6_rescue_bad_tail_v1": evidence["current_r6_rescue_bad_tail_v1"],
        "wednesday_bad_tail_v1": evidence["wednesday_bad_tail_v1"],
        "gap_to_wednesday_bad_tail_v1": evidence["gap_to_wednesday_bad_tail_v1"],
        "missed_bad_label_rows_v1": evidence["missed_bad_label_rows_v1"],
        "missed_tail_rows_v1": evidence["missed_tail_rows_v1"],
        "gap_bucket_counts_v1": evidence["gap_bucket_counts_v1"],
        "label_bucket_count_v1": len(label_contract["buckets_v1"]),
        "v2_variant_count_v1": len(weight_spec["candidate_weight_profiles_v1"]),
        "ambiguous_high_mfe_bad_positive_allowed_v1": False,
        "hard_protection_veto_v1": True,
        "training_started_v1": False,
        "r6_started_v1": False,
        "new_baseline_built_v1": False,
        "new_feature_surface_built_v1": False,
        "hard_status_v1": {
            "BEVIST": [
                "The V2 design is a spec-only artifact: no training, R6 run, baseline build, or feature surface build was performed.",
                "Post-rescue recall gap evidence is locked from the existing investigation package.",
                "R6 outside-base recovery was rejected because 122 positive rules found zero safe outside-base recoveries.",
            ],
            "INDIKERT": [
                "The next meaningful route is R5.2 objective V2 with two-stage recall heads and a hard protection veto.",
                "Existing legal AS_OF/R5/R5.1/R5.2 inputs should be reused before any new feature work.",
            ],
            "IKKE_ETABLERT": [
                "Actual V2 model uplift and downstream R6 improvement remain unestablished until a later explicit V2 rebuild run.",
            ],
        },
    }

    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "input_artifacts_v1": {
            "investigation_summary_v1": str(investigation_dir / INVESTIGATION_SUMMARY),
            "post_rescue_gap_map_v1": str(investigation_dir / GAP_MAP),
            "objective_scan_v1": str(investigation_dir / OBJECTIVE_SCAN),
            "decision_matrix_v1": str(investigation_dir / DECISION_MATRIX),
            "label_table_v1": str(label_objective_dir / LABEL_TABLE),
            "rescue_r6_dir_v1": str(rescue_r6_dir),
        },
        "output_files_v1": OUTPUT_FILES,
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "decision_v1": summary["decision_v1"],
        "next_action_v1": summary["next_action_v1"],
        "training_started_v1": False,
        "r6_started_v1": False,
        "new_baseline_built_v1": False,
        "new_feature_surface_built_v1": False,
    }
    audit = _audit(summary, evidence, weight_spec, feature_spec)

    _write_json(output_dir / OUTPUT_FILES["design_lock"], design_lock)
    _write_json(output_dir / OUTPUT_FILES["label_contract"], label_contract)
    _write_json(output_dir / OUTPUT_FILES["weight_cost"], weight_spec)
    _write_json(output_dir / OUTPUT_FILES["architecture"], architecture)
    _write_json(output_dir / OUTPUT_FILES["base_contract"], base_contract)
    _write_json(output_dir / OUTPUT_FILES["target_table"], target_table)
    feature_spec.to_csv(output_dir / OUTPUT_FILES["feature_use"], index=False)
    _write_json(output_dir / OUTPUT_FILES["parallel_run"], parallel_run)
    _write_json(output_dir / OUTPUT_FILES["eval_gate"], eval_gate)
    _write_json(output_dir / OUTPUT_FILES["runner_lock"], runner_lock)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / OUTPUT_FILES["status"], status)
    audit.to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary, weight_spec), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--investigation-dir", type=Path, default=DEFAULT_INVESTIGATION_DIR)
    parser.add_argument("--label-objective-dir", type=Path, default=DEFAULT_LABEL_OBJECTIVE_DIR)
    parser.add_argument("--rescue-r6-dir", type=Path, default=DEFAULT_RESCUE_R6_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        investigation_dir=args.investigation_dir,
        label_objective_dir=args.label_objective_dir,
        rescue_r6_dir=args.rescue_r6_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
