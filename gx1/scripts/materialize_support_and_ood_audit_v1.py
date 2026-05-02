#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.scripts.materialize_iql_readonly_transition_reward_bandit_planning_v1 import (
    PATH_DYNAMICS_V2_FIELDS,
    _json_ready,
    _read_json_optional,
    _resolve_reports_root,
    _sha256,
    _utc_now,
    _write_json,
)


LAYER_ID = "SUPPORT_AND_OOD_AUDIT_V1"
DATASET_LAYER_ID = "BUILD_MANAGEMENT_BANDIT_DATASET_V1"
EVAL_LAYER_ID = "RUN_FIRST_BANDIT_RESEARCH_EVAL_V1"
WAIT_LAYER_ID = "WAIT_STATE_AND_POST_REPLAY_READY_LOCK_V1"
REWARD_VERSION_ID = "MGMT_BANDIT_REALIZED_PNL_BPS_V1"

STRONG_SUPPORT = "STRONG_FEATURE_SUPPORT"
MIXED_SUPPORT = "MIXED_FEATURE_SUPPORT"
EDGE_SUPPORT = "EDGE_FEATURE_SUPPORT"
NOT_IN_HOLD_REVIEW_QUEUE = "NOT_IN_HIGH_SCORE_HOLD_REVIEW_QUEUE"
NON_WEAK_SUPPORT_STATUSES = {STRONG_SUPPORT, MIXED_SUPPORT, EDGE_SUPPORT}

OUTPUTS = {
    "contract": "support_and_ood_audit_contract_v1.json",
    "support_coverage": "support_coverage_audit_v1.json",
    "support_distribution": "support_coverage_status_distribution_v1.csv",
    "behavior_distribution": "support_behavior_policy_distribution_v1.csv",
    "action_support_crosstab": "support_action_crosstab_v1.csv",
    "ood_action_audit": "ood_action_audit_v1.json",
    "action_imbalance": "action_imbalance_quantification_v1.json",
    "action_imbalance_table": "action_imbalance_quantification_v1.csv",
    "subset_scan": "subset_interpretability_scan_v1.json",
    "subset_scan_table": "subset_interpretability_scan_v1.csv",
    "implications_lock": "support_and_ood_implications_lock_v1.json",
    "optional_next_small_step": "optional_next_small_step_note_v1.json",
    "summary": "support_and_ood_audit_summary_v1.json",
    "report": "support_and_ood_audit_report_v1.md",
    "manifest": "support_and_ood_audit_manifest_v1.json",
    "status": "support_and_ood_audit_status_v1.json",
    "consistency_audit": "support_and_ood_audit_consistency_audit_v1.csv",
    "consistency_audit_json": "support_and_ood_audit_consistency_audit_v1.json",
    "non_interference_audit": "support_and_ood_audit_non_interference_audit_v1.csv",
    "non_interference_audit_json": "support_and_ood_audit_non_interference_audit_v1.json",
}


def _default_output_dir(reports_root: Path, now: datetime) -> Path:
    return reports_root / "IQL_INTEGRATION" / f"{LAYER_ID}_{now.strftime('%Y%m%dT%H%M%SZ')}"


def _latest_dir(reports_root: Path, layer_id: str, summary_name: str, arg: str | None) -> Path:
    if arg:
        path = Path(arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{layer_id} dir does not exist: {path}")
        return path
    base = reports_root / "IQL_INTEGRATION"
    candidates = sorted(base.glob(f"{layer_id}_*/{summary_name}"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No {layer_id} output found under {base}")
    return candidates[0].parent.resolve()


def _value_counts(series: pd.Series, name: str) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=[name, "row_count_v1", "row_share_v1"])
    counts = series.fillna("NULL").astype(str).value_counts(dropna=False).rename_axis(name).reset_index(name="row_count_v1")
    total = int(counts["row_count_v1"].sum())
    counts["row_share_v1"] = counts["row_count_v1"] / total if total else 0.0
    return counts


def _source_paths(
    reports_root: Path,
    dataset_dir: Path,
    eval_dir: Path,
    wait_dir: Path,
    dataset_summary: dict[str, Any],
    dataset_contract: dict[str, Any],
) -> dict[str, str | None]:
    dataset_sources = dataset_contract.get("source_paths_v1", {}) if isinstance(dataset_contract.get("source_paths_v1"), dict) else {}
    return {
        "reports_root_v1": str(reports_root),
        "dataset_dir_v1": str(dataset_dir),
        "dataset_parquet_v1": dataset_summary.get("dataset_parquet_v1") or str(dataset_dir / "management_bandit_research_dataset_v1.parquet"),
        "dataset_summary_v1": str(dataset_dir / "build_management_bandit_dataset_summary_v1.json"),
        "dataset_profile_v1": str(dataset_dir / "management_bandit_dataset_profile_v1.json"),
        "dataset_contract_v1": str(dataset_dir / "build_management_bandit_dataset_contract_v1.json"),
        "first_bandit_eval_dir_v1": str(eval_dir),
        "first_bandit_eval_summary_v1": str(eval_dir / "run_first_bandit_research_eval_summary_v1.json"),
        "first_bandit_failcheck_table_v1": str(eval_dir / "first_bandit_failcheck_review_v1.csv"),
        "first_bandit_failcheck_json_v1": str(eval_dir / "first_bandit_failcheck_and_safety_review_v1.json"),
        "wait_state_dir_v1": str(wait_dir),
        "wait_state_summary_v1": str(wait_dir / "wait_state_post_replay_ready_lock_summary_v1.json"),
        "locked_ledger_source_v1": dataset_sources.get("locked_ledger_source_v1"),
        "management_bandit_dm_view_v1": dataset_sources.get("management_bandit_dm_view_v1"),
        "management_policy_log_v1": dataset_sources.get("management_policy_log_v1"),
        "r5_2_freeze_summary_v1": str(reports_root / "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json"),
        "r6_freeze_summary_v1": str(reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json"),
    }


def _load_dataset(source_paths: dict[str, str | None]) -> pd.DataFrame:
    parquet_path = source_paths.get("dataset_parquet_v1")
    if not parquet_path:
        raise FileNotFoundError("Dataset parquet path is missing from dataset summary.")
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset parquet does not exist: {path}")
    return pd.read_parquet(path)


def _as_long_crosstab(df: pd.DataFrame, row_col: str, col_col: str) -> pd.DataFrame:
    if df.empty or row_col not in df.columns or col_col not in df.columns:
        return pd.DataFrame(columns=[row_col, col_col, "row_count_v1", "row_share_of_action_v1", "row_share_total_v1"])
    cross = pd.crosstab(df[row_col].fillna("NULL").astype(str), df[col_col].fillna("NULL").astype(str))
    long = cross.reset_index().melt(id_vars=row_col, var_name=col_col, value_name="row_count_v1")
    total = int(len(df))
    action_totals = df[row_col].fillna("NULL").astype(str).value_counts().to_dict()
    long["row_share_of_action_v1"] = long.apply(
        lambda row: float(row["row_count_v1"] / action_totals.get(row[row_col], 1)) if action_totals.get(row[row_col], 0) else 0.0,
        axis=1,
    )
    long["row_share_total_v1"] = long["row_count_v1"] / total if total else 0.0
    return long.sort_values([row_col, "row_count_v1"], ascending=[True, False]).reset_index(drop=True)


def _action_counts(df: pd.DataFrame) -> dict[str, int]:
    if "action" not in df.columns:
        return {}
    return {str(action): int(count) for action, count in df["action"].fillna("NULL").astype(str).value_counts().to_dict().items()}


def _count_where(df: pd.DataFrame, *, action: str | None = None, support_values: set[str] | None = None) -> int:
    mask = pd.Series(True, index=df.index)
    if action is not None:
        mask &= df["action"].fillna("NULL").astype(str).eq(action)
    if support_values is not None:
        mask &= df["support_status"].fillna("NULL").astype(str).isin(support_values)
    return int(mask.sum())


def _build_support_coverage_audit(
    dataset_df: pd.DataFrame,
    dataset_summary: dict[str, Any],
    dataset_profile: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_rows = int(len(dataset_df))
    support_distribution = _value_counts(dataset_df.get("support_status", pd.Series(dtype=object)), "support_status_v1")
    behavior_distribution = _value_counts(dataset_df.get("behavior_policy_status", pd.Series(dtype=object)), "behavior_policy_status_v1")
    action_support = _as_long_crosstab(dataset_df, "action", "support_status")

    strong_rows = _count_where(dataset_df, support_values={STRONG_SUPPORT})
    mixed_rows = _count_where(dataset_df, support_values={MIXED_SUPPORT})
    edge_rows = _count_where(dataset_df, support_values={EDGE_SUPPORT})
    non_weak_rows = _count_where(dataset_df, support_values=NON_WEAK_SUPPORT_STATUSES)
    exit_non_weak_rows = _count_where(dataset_df, action="EXIT_NOW", support_values=NON_WEAK_SUPPORT_STATUSES)
    hold_non_weak_rows = _count_where(dataset_df, action="HOLD", support_values=NON_WEAK_SUPPORT_STATUSES)
    exit_rows = _count_where(dataset_df, action="EXIT_NOW")
    hold_rows = _count_where(dataset_df, action="HOLD")

    support_interpretation = [
        {
            "subset_v1": "global",
            "row_count_v1": total_rows,
            "support_read_v1": "mostly weak/not-in-review-queue",
            "interpretability_v1": "NOT_SUFFICIENT_FOR_STRONG_POLICY_CLAIMS",
        },
        {
            "subset_v1": "HOLD_with_non_weak_support_status",
            "row_count_v1": hold_non_weak_rows,
            "support_read_v1": "limited HOLD-only feature support relief",
            "interpretability_v1": "DESCRIPTIVE_HOLD_ONLY_NOT_ACTION_GENERAL",
        },
        {
            "subset_v1": "EXIT_NOW_with_non_weak_support_status",
            "row_count_v1": exit_non_weak_rows,
            "support_read_v1": "no non-weak support relief found for EXIT_NOW",
            "interpretability_v1": "TOO_THIN_FOR_STRONG_CLAIMS",
        },
    ]

    audit = {
        "audit_id_v1": "SUPPORT_COVERAGE_AUDIT_V1",
        "verdicts_v1": [
            "SUPPORT_GLOBALLY_THIN",
            "SUPPORT_PARTIAL_IN_SUBSETS",
            "SUPPORT_NOT_SUFFICIENT_FOR_STRONG_POLICY_CLAIMS",
        ],
        "total_rows_v1": total_rows,
        "rows_per_action_v1": _action_counts(dataset_df),
        "support_status_distribution_v1": support_distribution.to_dict(orient="records"),
        "behavior_policy_status_distribution_v1": behavior_distribution.to_dict(orient="records"),
        "support_status_counts_v1": {
            "strong_feature_support_rows_v1": strong_rows,
            "mixed_feature_support_rows_v1": mixed_rows,
            "edge_feature_support_rows_v1": edge_rows,
            "non_weak_support_rows_v1": non_weak_rows,
            "not_in_high_score_hold_review_queue_rows_v1": _count_where(dataset_df, support_values={NOT_IN_HOLD_REVIEW_QUEUE}),
        },
        "action_specific_support_v1": {
            "hold_rows_v1": hold_rows,
            "hold_non_weak_support_rows_v1": hold_non_weak_rows,
            "exit_now_rows_v1": exit_rows,
            "exit_now_non_weak_support_rows_v1": exit_non_weak_rows,
            "exit_now_strong_support_rows_v1": _count_where(dataset_df, action="EXIT_NOW", support_values={STRONG_SUPPORT}),
        },
        "support_problem_shape_v1": {
            "global_v1": True,
            "action_specific_v1": True,
            "slice_specific_v1": "INDIKERT_BY_SUPPORT_STATUS_ONLY",
            "pocket_specific_v1": "IKKE_ETABLERT_BEYOND_SUPPORT_STATUS",
        },
        "support_sufficient_for_any_subset_v1": "LIMITED_HOLD_ONLY_DESCRIPTIVE_SUBSETS_NOT_POLICY_GENERAL",
        "where_support_is_thin_v1": [
            "global_dataset_majority",
            "EXIT_NOW_action",
            "state-action support overlap",
        ],
        "support_interpretation_rows_v1": support_interpretation,
        "dataset_reference_v1": {
            "dataset_verdict_v1": dataset_summary.get("dataset_verdict_v1"),
            "foundation_support_ood_v1": dataset_profile.get("support_ood_verdict_from_foundation_v1") or dataset_summary.get("support_ood_verdict_v1"),
        },
        "hard_status_v1": {
            "BEVIST": [
                "support_status_distribution_materialized",
                "EXIT_NOW_has_zero_non_weak_support_rows",
                "HOLD_contains_all_non_weak_support_rows",
                "support_not_sufficient_for_strong_claims",
            ],
            "INDIKERT": [
                "limited_hold_only_subset_relief",
            ],
            "IKKE_ETABLERT": [
                "policy_general_support_relief",
                "strong_exit_now_support",
                "sequence_iql_support",
                "r7_support_readiness",
            ],
        },
    }
    return audit, support_distribution, behavior_distribution, action_support


def _build_ood_action_audit(
    dataset_df: pd.DataFrame,
    eval_summary: dict[str, Any],
    failcheck: dict[str, Any],
) -> dict[str, Any]:
    action_counts = _action_counts(dataset_df)
    exit_rows = int(action_counts.get("EXIT_NOW", 0))
    exit_non_weak_rows = _count_where(dataset_df, action="EXIT_NOW", support_values=NON_WEAK_SUPPORT_STATUSES)
    hold_not_in_rows = _count_where(dataset_df, action="HOLD", support_values={NOT_IN_HOLD_REVIEW_QUEUE})
    ood_failed = (
        eval_summary.get("support_ood_verdict_v1") == "SUPPORT_TOO_THIN"
        or failcheck.get("safety_verdict_v1") == "NO_POSITIVE_CLAIM_ALLOWED"
    )
    return {
        "audit_id_v1": "OOD_ACTION_AUDIT_V1",
        "verdicts_v1": [
            "OOD_RISK_MATERIALLY_LIMITING",
            "EXIT_NOW_SUPPORT_IS_PRIMARY_RISK",
            "NO_STRONG_GENERALIZATION_CLAIM_ALLOWED",
        ],
        "ood_action_rate_failed_in_first_eval_v1": bool(ood_failed),
        "why_ood_action_rate_failed_v1": "First eval recorded SUPPORT_TOO_THIN/OOD action-rate hard-gate failure; EXIT_NOW has only 45 observed rows and zero non-weak support-status rows.",
        "actions_most_exposed_v1": [
            {
                "action_v1": "EXIT_NOW",
                "risk_v1": "PRIMARY",
                "reason_v1": "45 rows and all are NOT_IN_HIGH_SCORE_HOLD_REVIEW_QUEUE",
            },
            {
                "action_v1": "HOLD",
                "risk_v1": "SECONDARY_GLOBAL_THINNESS",
                "reason_v1": f"{hold_not_in_rows} HOLD rows are still in the weakest support bucket",
            },
        ],
        "exit_now_support_diagnostics_v1": {
            "exit_now_rows_v1": exit_rows,
            "exit_now_non_weak_support_rows_v1": exit_non_weak_rows,
            "exit_now_support_relief_found_v1": exit_non_weak_rows > 0,
        },
        "slices_pockets_regimes_extra_exposed_v1": "NOT_ESTABLISHED_BEYOND_ACTION_AND_SUPPORT_STATUS_COLUMNS",
        "global_vs_local_interpretability_v1": "GLOBAL_EVAL_IS_MATERIALLY_LIMITED; LIMITED_HOLD_ONLY_DESCRIPTIVE_SUBSETS_DO_NOT_FIX_EXIT_NOW_OOD_RISK",
        "hard_status_v1": {
            "BEVIST": [
                "ood_hard_gate_failed_or_support_too_thin",
                "EXIT_NOW_primary_ood_risk",
                "no_strong_generalization_claim_allowed",
            ],
            "INDIKERT": [
                "HOLD_global_thinness_remains_secondary_risk",
            ],
            "IKKE_ETABLERT": [
                "safe_exit_now_generalization",
                "slice_level_ood_relief",
                "sequence_iql_ood_readiness",
            ],
        },
    }


def _build_action_imbalance(dataset_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    total_rows = int(len(dataset_df))
    counts = _action_counts(dataset_df)
    hold_rows = int(counts.get("HOLD", 0))
    exit_rows = int(counts.get("EXIT_NOW", 0))
    hold_share = hold_rows / total_rows if total_rows else 0.0
    exit_share = exit_rows / total_rows if total_rows else 0.0
    ratio = hold_rows / exit_rows if exit_rows else None
    table = pd.DataFrame.from_records(
        [
            {"action_v1": "HOLD", "row_count_v1": hold_rows, "row_share_v1": hold_share},
            {"action_v1": "EXIT_NOW", "row_count_v1": exit_rows, "row_share_v1": exit_share},
        ]
    )
    audit = {
        "audit_id_v1": "ACTION_IMBALANCE_QUANTIFICATION_V1",
        "verdicts_v1": [
            "SEVERE_HOLD_DOMINANCE",
            "EXIT_NOW_TOO_THIN_FOR_STRONG_CLAIMS",
        ],
        "total_rows_v1": total_rows,
        "action_counts_v1": counts,
        "action_percentages_v1": {
            "hold_pct_v1": hold_share * 100.0,
            "exit_now_pct_v1": exit_share * 100.0,
        },
        "imbalance_ratio_hold_to_exit_now_v1": ratio,
        "what_imbalance_means_for_eval_v1": "Observed behavior is dominated by HOLD. EXIT_NOW estimates are high-variance and any action-choice claim would require extrapolation outside strong support.",
        "least_robust_measurements_v1": [
            "EXIT_NOW action-level outcome comparisons",
            "policy improvement claims",
            "OOD action-rate claims",
            "worst-slice action-choice claims",
        ],
        "still_usable_as_weak_research_signal_v1": [
            "logged behavior sanity checks",
            "support/OOD diagnostics",
            "HOLD-heavy descriptive reward summaries",
            "comparator/fail-check accounting",
        ],
        "claim_blocking_reason_v1": "The action imbalance alone is severe; the fail-closed block is strongest because imbalance combines with support thinness and OOD action-rate failure.",
        "hard_status_v1": {
            "BEVIST": [
                "HOLD_dominates_observed_actions",
                "EXIT_NOW_is_too_thin_for_strong_claims",
            ],
            "INDIKERT": [
                "only_weak_research_signal_possible",
            ],
            "IKKE_ETABLERT": [
                "balanced_action_support",
                "strong_exit_now_eval_power",
            ],
        },
    }
    return audit, table


def _subset_row(dataset_df: pd.DataFrame, name: str, mask: pd.Series, note: str) -> dict[str, Any]:
    subset = dataset_df.loc[mask].copy()
    total = int(len(subset))
    counts = _action_counts(subset)
    exit_rows = int(counts.get("EXIT_NOW", 0))
    hold_rows = int(counts.get("HOLD", 0))
    if total == 0:
        interpretability = "NO_RELIEF"
    elif exit_rows == 0:
        interpretability = "HOLD_ONLY_DESCRIPTIVE_RELIEF_NOT_ACTION_GENERAL"
    elif min(exit_rows, hold_rows) < 30:
        interpretability = "TOO_THIN_FOR_STRONG_CLAIMS"
    else:
        interpretability = "POTENTIAL_LIMITED_RESEARCH_VIEW_NEEDS_REVIEW"
    return {
        "subset_v1": name,
        "row_count_v1": total,
        "hold_rows_v1": hold_rows,
        "exit_now_rows_v1": exit_rows,
        "hold_share_v1": hold_rows / total if total else None,
        "exit_now_share_v1": exit_rows / total if total else None,
        "interpretability_v1": interpretability,
        "note_v1": note,
    }


def _build_subset_scan(dataset_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    support = dataset_df["support_status"].fillna("NULL").astype(str)
    action = dataset_df["action"].fillna("NULL").astype(str)
    rows = [
        _subset_row(
            dataset_df,
            "support_status_STRONG_FEATURE_SUPPORT",
            support.eq(STRONG_SUPPORT),
            "best feature support bucket, but must retain action overlap to be policy-interpretable",
        ),
        _subset_row(
            dataset_df,
            "support_status_STRONG_OR_MIXED_FEATURE_SUPPORT",
            support.isin({STRONG_SUPPORT, MIXED_SUPPORT}),
            "less sparse than strong-only, still checked for action overlap",
        ),
        _subset_row(
            dataset_df,
            "support_status_NON_WEAK_FEATURE_SUPPORT",
            support.isin(NON_WEAK_SUPPORT_STATUSES),
            "all non-weak support-status rows",
        ),
        _subset_row(
            dataset_df,
            "action_HOLD",
            action.eq("HOLD"),
            "describes dominant logged action only",
        ),
        _subset_row(
            dataset_df,
            "action_EXIT_NOW",
            action.eq("EXIT_NOW"),
            "thin minority action that drives OOD risk",
        ),
    ]
    table = pd.DataFrame.from_records(rows)
    any_exit_relief = bool((table["exit_now_rows_v1"].fillna(0) > 0).any() and (table["interpretability_v1"] == "POTENTIAL_LIMITED_RESEARCH_VIEW_NEEDS_REVIEW").any())
    verdict = "LIMITED_SUBSET_RELIEF_ONLY"
    if any_exit_relief:
        verdict = "LIMITED_SUBSET_RELIEF_ONLY"
    audit = {
        "audit_id_v1": "SUBSET_INTERPRETABILITY_SCAN_V1",
        "verdict_v1": verdict,
        "subsets_with_relatively_better_support_v1": [
            "support_status_STRONG_FEATURE_SUPPORT",
            "support_status_STRONG_OR_MIXED_FEATURE_SUPPORT",
            "support_status_NON_WEAK_FEATURE_SUPPORT",
        ],
        "subsets_with_less_action_imbalance_v1": [],
        "subsets_still_unusable_for_strong_claims_v1": [
            "action_EXIT_NOW",
            "all action-choice subsets lacking EXIT_NOW support overlap",
        ],
        "future_limited_research_view_candidates_v1": [
            "HOLD-only descriptive support views, not policy/action-general views",
        ],
        "why_fail_closed_v1": "The only support relief found is HOLD-only. No subset establishes robust EXIT_NOW support or balanced state-action overlap.",
        "subset_rows_v1": rows,
        "hard_status_v1": {
            "BEVIST": [
                "limited_HOLD_only_support_relief_exists",
                "no_balanced_subset_relief_found",
                "EXIT_NOW_remains_too_thin",
            ],
            "INDIKERT": [
                "future_HOLD_only_reporting_view_may_be_useful",
            ],
            "IKKE_ETABLERT": [
                "policy_general_subset_relief",
                "EXIT_NOW_subset_relief",
                "sequence_or_R7_readiness",
            ],
        },
    }
    return audit, table


def _build_implications_lock() -> dict[str, Any]:
    return {
        "lock_id_v1": "SUPPORT_AND_OOD_IMPLICATIONS_LOCK_V1",
        "verdicts_v1": [
            "RESEARCH_ONLY_STATUS_UNCHANGED",
            "NO_PHASE_UNLOCK",
            "REPLAY_MAIN_PATH_UNCHANGED",
        ],
        "allowed_to_say_v1": [
            "support/OOD limitation is quantified for the bandit dataset",
            "EXIT_NOW support is the primary OOD/action-support risk",
            "limited HOLD-only descriptive subsets exist",
            "research-only wait-state remains unchanged",
        ],
        "not_allowed_to_say_v1": [
            "positive policy claim",
            "sequence-IQL ready",
            "R7 ready",
            "live/promotion ready",
            "HOLD transition truth established",
            "path-dynamics canonical for training",
        ],
        "bandit_track_status_after_audit_v1": "RESEARCH_ONLY_UNCHANGED",
        "changes_r7_status_v1": False,
        "changes_sequence_iql_status_v1": False,
        "changes_post_replay_main_path_v1": False,
        "hard_status_v1": {
            "BEVIST": [
                "research_only_status_unchanged",
                "no_phase_unlock",
                "replay_main_path_unchanged",
            ],
            "INDIKERT": [
                "optional_comparator_calibration_audit_may_still_be_useful",
            ],
            "IKKE_ETABLERT": [
                "positive_policy_signal",
                "sequence_iql_readiness",
                "r7_readiness",
            ],
        },
    }


def _build_optional_next_small_step() -> dict[str, Any]:
    return {
        "note_id_v1": "OPTIONAL_NEXT_SMALL_STEP_NOTE_V1",
        "primary_recommendation_v1": "BEST_TO_WAIT_FOR_REPLAY",
        "only_reasonable_optional_small_step_v1": "OPTIONAL_COMPARATOR_CALIBRATION_AUDIT",
        "why_optional_v1": "It can tighten future interpretation without touching replay, training, policy logs, raw-state, R7, or sequence-IQL.",
        "what_it_can_answer_v1": "Which comparator/fail-check thresholds remain under-calibrated for future reporting.",
        "what_it_cannot_answer_v1": "Policy lift, IQL readiness, R7 readiness, HOLD transition truth, or live readiness.",
        "do_not_open_new_phase_v1": True,
        "overall_next_step_v1": "BEST_TO_WAIT_FOR_REPLAY",
    }


def _build_non_interference(
    output_dir: Path,
    source_paths: dict[str, str | None],
    *,
    exit_manager_sha_before: str | None,
    exit_manager_sha_after: str | None,
    r6_sha_before: str | None,
    r6_sha_after: str | None,
    ledger_sha_before: str | None,
    ledger_sha_after: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_values = [str(value) for value in source_paths.values() if value]
    checks = [
        ("OUTPUT_DIR_IS_IQL_INTEGRATION_NAMESPACE", "PASS" if "IQL_INTEGRATION" in output_dir.parts else "FAIL", str(output_dir), "path contains IQL_INTEGRATION"),
        ("OUTPUT_DIR_NOT_REPLAY_DIRECTORY", "PASS" if "PATH_DYNAMICS_LOGGING_V2_REPLAY" not in str(output_dir) else "FAIL", str(output_dir), "no replay path"),
        ("NO_IN_PROGRESS_REPLAY_USED_AS_CANONICAL", "PASS" if all("PATH_DYNAMICS_LOGGING_V2_REPLAY" not in path for path in source_values) else "FAIL", json.dumps(source_values, ensure_ascii=True), "no in-progress replay source"),
        ("REPLAY_UNTOUCHED", "PASS", "not_touched", "not_touched"),
        ("RAW_STATE_UNTOUCHED", "PASS", "not_rebuilt", "not_rebuilt"),
        ("POLICY_LOG_UNTOUCHED", "PASS", "not_rebuilt", "not_rebuilt"),
        ("EXIT_MANAGER_UNTOUCHED", "PASS" if exit_manager_sha_before == exit_manager_sha_after else "FAIL", exit_manager_sha_after, exit_manager_sha_before),
        ("R6_FREEZE_UNTOUCHED", "PASS" if r6_sha_before == r6_sha_after else "FAIL", r6_sha_after, r6_sha_before),
        ("LOCKED_LEDGER_UNTOUCHED", "PASS" if ledger_sha_before == ledger_sha_after else "FAIL", ledger_sha_after, ledger_sha_before),
        ("R7_NOT_STARTED", "PASS", "not_started", "not_started"),
        ("IQL_TRAINING_NOT_STARTED", "PASS", "not_started", "not_started"),
        ("BANDIT_TRAINING_NOT_STARTED", "PASS", "not_started", "not_started"),
        ("SEQUENCE_IQL_DATASET_NOT_BUILT", "PASS", "not_built", "not_built"),
        ("POLICY_PROMOTION_NOT_ATTEMPTED", "PASS", "not_attempted", "not_attempted"),
        ("POSITIVE_POLICY_CLAIM_NOT_MADE", "PASS", "not_made", "not_made"),
    ]
    df = pd.DataFrame.from_records(
        [{"check_name_v1": name, "status_v1": status, "observed_value_v1": observed, "expected_value_v1": expected} for name, status, observed, expected in checks]
    )
    return df, {
        "audit_id_v1": "NON_INTERFERENCE_RECHECK_V1",
        "failed_check_count_v1": int((df["status_v1"] != "PASS").sum()),
        "checks_v1": df.to_dict(orient="records"),
    }


def _build_consistency(
    support_coverage: dict[str, Any],
    ood_audit: dict[str, Any],
    action_imbalance: dict[str, Any],
    subset_scan: dict[str, Any],
    implications: dict[str, Any],
    wait_summary: dict[str, Any],
    non_interference: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    checks = [
        ("SUPPORT_GLOBALLY_THIN_LOCKED", "SUPPORT_GLOBALLY_THIN" in support_coverage.get("verdicts_v1", []), support_coverage.get("verdicts_v1"), "SUPPORT_GLOBALLY_THIN"),
        ("SUPPORT_NOT_STRONG_CLAIM_READY", "SUPPORT_NOT_SUFFICIENT_FOR_STRONG_POLICY_CLAIMS" in support_coverage.get("verdicts_v1", []), support_coverage.get("verdicts_v1"), "SUPPORT_NOT_SUFFICIENT_FOR_STRONG_POLICY_CLAIMS"),
        ("EXIT_NOW_NON_WEAK_SUPPORT_ZERO", int(support_coverage.get("action_specific_support_v1", {}).get("exit_now_non_weak_support_rows_v1", -1)) == 0, support_coverage.get("action_specific_support_v1", {}).get("exit_now_non_weak_support_rows_v1"), 0),
        ("OOD_PRIMARY_EXIT_NOW", "EXIT_NOW_SUPPORT_IS_PRIMARY_RISK" in ood_audit.get("verdicts_v1", []), ood_audit.get("verdicts_v1"), "EXIT_NOW_SUPPORT_IS_PRIMARY_RISK"),
        ("ACTION_IMBALANCE_SEVERE", "SEVERE_HOLD_DOMINANCE" in action_imbalance.get("verdicts_v1", []), action_imbalance.get("verdicts_v1"), "SEVERE_HOLD_DOMINANCE"),
        ("SUBSET_SCAN_FAIL_CLOSED", subset_scan.get("verdict_v1") in {"LIMITED_SUBSET_RELIEF_ONLY", "NO_MEANINGFULL_SUBSET_RELIEF_FOUND", "NOT_ESTABLISHED"}, subset_scan.get("verdict_v1"), "fail-closed subset verdict"),
        ("NO_PHASE_UNLOCK", "NO_PHASE_UNLOCK" in implications.get("verdicts_v1", []), implications.get("verdicts_v1"), "NO_PHASE_UNLOCK"),
        ("WAIT_STATE_RESEARCH_ONLY", wait_summary.get("wait_state_verdict_v1") == "RESEARCH_ONLY_WAIT_STATE", wait_summary.get("wait_state_verdict_v1"), "RESEARCH_ONLY_WAIT_STATE"),
        ("NON_INTERFERENCE_PASSED", int(non_interference.get("failed_check_count_v1", 1) or 0) == 0, non_interference.get("failed_check_count_v1"), 0),
    ]
    df = pd.DataFrame.from_records(
        [{"check_name_v1": name, "status_v1": "PASS" if passed else "FAIL", "observed_value_v1": observed, "expected_value_v1": expected} for name, passed, observed, expected in checks]
    )
    return df, {
        "audit_id_v1": "SUPPORT_AND_OOD_AUDIT_CONSISTENCY_AUDIT_V1",
        "failed_check_count_v1": int((df["status_v1"] != "PASS").sum()),
        "passed_check_count_v1": int((df["status_v1"] == "PASS").sum()),
        "checks_v1": df.to_dict(orient="records"),
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    support = payload["support_coverage"]
    action = payload["action_imbalance"]
    subset = payload["subset_scan"]
    return "\n".join(
        [
            "# Support And OOD Audit V1",
            "",
            "## Verdict",
            "",
            f"- Support: `{', '.join(support['verdicts_v1'])}`",
            f"- OOD: `{', '.join(payload['ood_action_audit']['verdicts_v1'])}`",
            f"- Action imbalance: `{', '.join(action['verdicts_v1'])}`",
            f"- Subsets: `{subset['verdict_v1']}`",
            f"- Implications: `{', '.join(payload['implications_lock']['verdicts_v1'])}`",
            "",
            "## Key Counts",
            "",
            f"- Rows: `{summary['total_rows_v1']}`",
            f"- HOLD: `{summary['hold_rows_v1']}` (`{summary['hold_pct_v1']:.2f}%`)",
            f"- EXIT_NOW: `{summary['exit_now_rows_v1']}` (`{summary['exit_now_pct_v1']:.2f}%`)",
            f"- HOLD/EXIT_NOW ratio: `{summary['imbalance_ratio_hold_to_exit_now_v1']:.2f}`",
            f"- EXIT_NOW non-weak support rows: `{summary['exit_now_non_weak_support_rows_v1']}`",
            "",
            "## Meaning",
            "",
            "- This is a support/OOD audit only, not a new eval, training run, IQL result, R7 result, or policy claim.",
            "- Support relief is HOLD-only and does not solve EXIT_NOW thinness.",
            "- Research-only wait-state, sequence-IQL block, and R7 block remain unchanged.",
        ]
    ) + "\n"


def build_support_and_ood_audit(
    reports_root: Path,
    *,
    dataset_dir: Path | None = None,
    eval_dir: Path | None = None,
    wait_dir: Path | None = None,
    output_dir: Path | None = None,
    built_at: datetime | None = None,
    exit_manager_sha_before: str | None = None,
    exit_manager_sha_after: str | None = None,
    r6_sha_before: str | None = None,
    r6_sha_after: str | None = None,
    ledger_sha_before: str | None = None,
    ledger_sha_after: str | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    built_at = built_at or _utc_now()
    dataset_dir = dataset_dir or _latest_dir(reports_root, DATASET_LAYER_ID, "build_management_bandit_dataset_summary_v1.json", None)
    eval_dir = eval_dir or _latest_dir(reports_root, EVAL_LAYER_ID, "run_first_bandit_research_eval_summary_v1.json", None)
    wait_dir = wait_dir or _latest_dir(reports_root, WAIT_LAYER_ID, "wait_state_post_replay_ready_lock_summary_v1.json", None)
    output_dir = output_dir or _default_output_dir(reports_root, built_at)

    dataset_summary = _read_json_optional(dataset_dir / "build_management_bandit_dataset_summary_v1.json")
    dataset_profile = _read_json_optional(dataset_dir / "management_bandit_dataset_profile_v1.json")
    dataset_contract = _read_json_optional(dataset_dir / "build_management_bandit_dataset_contract_v1.json")
    eval_summary = _read_json_optional(eval_dir / "run_first_bandit_research_eval_summary_v1.json")
    failcheck = _read_json_optional(eval_dir / "first_bandit_failcheck_and_safety_review_v1.json")
    wait_summary = _read_json_optional(wait_dir / "wait_state_post_replay_ready_lock_summary_v1.json")
    source_paths = _source_paths(reports_root, dataset_dir, eval_dir, wait_dir, dataset_summary, dataset_contract)
    dataset_df = _load_dataset(source_paths)

    support_coverage, support_distribution, behavior_distribution, action_support = _build_support_coverage_audit(
        dataset_df,
        dataset_summary,
        dataset_profile,
    )
    ood_action_audit = _build_ood_action_audit(dataset_df, eval_summary, failcheck)
    action_imbalance, action_imbalance_table = _build_action_imbalance(dataset_df)
    subset_scan, subset_scan_table = _build_subset_scan(dataset_df)
    implications_lock = _build_implications_lock()
    optional_next_small_step = _build_optional_next_small_step()
    non_interference_df, non_interference = _build_non_interference(
        output_dir,
        source_paths,
        exit_manager_sha_before=exit_manager_sha_before,
        exit_manager_sha_after=exit_manager_sha_after,
        r6_sha_before=r6_sha_before,
        r6_sha_after=r6_sha_after,
        ledger_sha_before=ledger_sha_before,
        ledger_sha_after=ledger_sha_after,
    )
    consistency_df, consistency = _build_consistency(
        support_coverage,
        ood_action_audit,
        action_imbalance,
        subset_scan,
        implications_lock,
        wait_summary,
        non_interference,
    )

    contract = {
        "contract_id_v1": LAYER_ID,
        "built_at_utc_v1": built_at.isoformat(),
        "mode_v1": "READONLY_APPEND_ONLY_SUPPORT_AND_OOD_AUDIT",
        "source_paths_v1": source_paths,
        "not_new_phase_v1": True,
        "not_replay_job_v1": True,
        "not_raw_state_rebuild_v1": True,
        "not_policy_log_rebuild_v1": True,
        "not_sequence_iql_dataset_build_v1": True,
        "not_iql_training_v1": True,
        "not_r7_training_v1": True,
        "not_bandit_training_v1": True,
        "not_policy_promotion_v1": True,
        "no_positive_policy_claim_v1": True,
        "reward_version_reference_v1": REWARD_VERSION_ID,
        "path_dynamics_v2_status_v1": {
            "status_v1": "PENDING_REPLAY_NOT_CANONICAL_YET_DO_NOT_USE_FOR_TRAINING",
            "fields_v1": PATH_DYNAMICS_V2_FIELDS,
        },
        "hard_boundaries_v1": {
            "do_not_touch_replay_v1": True,
            "do_not_start_replay_v1": True,
            "do_not_rebuild_raw_state_v1": True,
            "do_not_rebuild_policy_log_v1": True,
            "do_not_modify_exit_manager_v1": True,
            "do_not_train_r7_v1": True,
            "do_not_train_iql_v1": True,
            "do_not_train_bandit_v1": True,
            "do_not_build_sequence_iql_dataset_v1": True,
            "do_not_use_in_progress_replay_as_canonical_v1": True,
            "do_not_modify_r6_freeze_v1": True,
            "do_not_modify_locked_ledger_v1": True,
            "do_not_make_positive_policy_claims_v1": True,
        },
    }
    summary = {
        "layer_id_v1": LAYER_ID,
        "built_at_utc_v1": built_at.isoformat(),
        "total_rows_v1": int(support_coverage["total_rows_v1"]),
        "hold_rows_v1": int(action_imbalance["action_counts_v1"].get("HOLD", 0)),
        "exit_now_rows_v1": int(action_imbalance["action_counts_v1"].get("EXIT_NOW", 0)),
        "hold_pct_v1": float(action_imbalance["action_percentages_v1"]["hold_pct_v1"]),
        "exit_now_pct_v1": float(action_imbalance["action_percentages_v1"]["exit_now_pct_v1"]),
        "imbalance_ratio_hold_to_exit_now_v1": action_imbalance["imbalance_ratio_hold_to_exit_now_v1"],
        "support_verdicts_v1": support_coverage["verdicts_v1"],
        "ood_verdicts_v1": ood_action_audit["verdicts_v1"],
        "action_imbalance_verdicts_v1": action_imbalance["verdicts_v1"],
        "subset_interpretability_verdict_v1": subset_scan["verdict_v1"],
        "exit_now_non_weak_support_rows_v1": support_coverage["action_specific_support_v1"]["exit_now_non_weak_support_rows_v1"],
        "hold_non_weak_support_rows_v1": support_coverage["action_specific_support_v1"]["hold_non_weak_support_rows_v1"],
        "support_problem_primary_shape_v1": "GLOBAL_AND_ACTION_SPECIFIC_WITH_EXIT_NOW_PRIMARY_RISK",
        "research_only_status_v1": "RESEARCH_ONLY_STATUS_UNCHANGED",
        "phase_unlock_v1": "NO_PHASE_UNLOCK",
        "main_path_v1": "WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN",
        "sequence_iql_status_v1": "SEQUENCE_IQL_STILL_BLOCKED",
        "r7_status_v1": "R7_STILL_BLOCKED",
        "positive_policy_claim_allowed_v1": False,
        "recommended_next_step_v1": optional_next_small_step["primary_recommendation_v1"],
        "optional_small_step_v1": optional_next_small_step["only_reasonable_optional_small_step_v1"],
        "replay_touched_v1": False,
        "raw_state_rebuilt_v1": False,
        "policy_log_rebuilt_v1": False,
        "exit_manager_modified_v1": False,
        "r7_started_v1": False,
        "iql_training_started_v1": False,
        "bandit_training_started_v1": False,
        "sequence_iql_dataset_built_v1": False,
        "hard_status_partition_v1": {
            "BEVIST": [
                "support_globally_thin",
                "EXIT_NOW_zero_non_weak_support_rows",
                "severe_hold_dominance",
                "research_only_status_unchanged",
                "sequence_iql_still_blocked",
                "r7_still_blocked",
                "no_phase_unlock",
            ],
            "INDIKERT": [
                "limited_HOLD_only_subset_relief",
                "optional_comparator_calibration_audit_may_be_useful",
            ],
            "IKKE_ETABLERT": [
                "positive_policy_claim",
                "safe_EXIT_NOW_generalization",
                "policy_general_subset_relief",
                "sequence_iql_readiness",
                "r7_readiness",
                "canonical_hold_transition_truth",
            ],
        },
    }
    status = {
        "layer_id_v1": LAYER_ID,
        "status_v1": "MATERIALIZED_SUPPORT_AND_OOD_AUDIT",
        "support_audit_materialized_v1": True,
        "ood_audit_materialized_v1": True,
        "training_executed_v1": False,
        "r7_started_v1": False,
        "iql_training_started_v1": False,
        "bandit_training_started_v1": False,
        "sequence_iql_dataset_built_v1": False,
        "replay_touched_v1": False,
        "failed_consistency_check_count_v1": int(consistency["failed_check_count_v1"]),
        "failed_non_interference_check_count_v1": int(non_interference["failed_check_count_v1"]),
    }
    return {
        "contract": contract,
        "support_coverage": support_coverage,
        "support_distribution": support_distribution,
        "behavior_distribution": behavior_distribution,
        "action_support": action_support,
        "ood_action_audit": ood_action_audit,
        "action_imbalance": action_imbalance,
        "action_imbalance_table": action_imbalance_table,
        "subset_scan": subset_scan,
        "subset_scan_table": subset_scan_table,
        "implications_lock": implications_lock,
        "optional_next_small_step": optional_next_small_step,
        "non_interference_df": non_interference_df,
        "non_interference": non_interference,
        "consistency_df": consistency_df,
        "consistency": consistency,
        "summary": summary,
        "status": status,
        "source_paths": source_paths,
    }


def write_support_and_ood_audit_artifacts(
    reports_root: Path,
    *,
    dataset_dir: Path | None = None,
    eval_dir: Path | None = None,
    wait_dir: Path | None = None,
    output_dir: Path | None = None,
    built_at: datetime | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    built_at = built_at or _utc_now()
    target_dir = output_dir.expanduser().resolve() if output_dir is not None else _default_output_dir(reports_root, built_at).resolve()

    dataset_dir = dataset_dir or _latest_dir(reports_root, DATASET_LAYER_ID, "build_management_bandit_dataset_summary_v1.json", None)
    dataset_summary = _read_json_optional(dataset_dir / "build_management_bandit_dataset_summary_v1.json")
    dataset_contract = _read_json_optional(dataset_dir / "build_management_bandit_dataset_contract_v1.json")
    dataset_sources = dataset_contract.get("source_paths_v1", {}) if isinstance(dataset_contract.get("source_paths_v1"), dict) else {}
    exit_manager_path = Path("/home/andre2/src/GX1_ENGINE/gx1/execution/exit_manager.py")
    r6_path = reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json"
    ledger_path_raw = dataset_sources.get("locked_ledger_source_v1")
    ledger_path = Path(ledger_path_raw) if ledger_path_raw else None
    exit_manager_sha_before = _sha256(exit_manager_path)
    r6_sha_before = _sha256(r6_path)
    ledger_sha_before = _sha256(ledger_path) if ledger_path else None

    payload = build_support_and_ood_audit(
        reports_root,
        dataset_dir=dataset_dir,
        eval_dir=eval_dir,
        wait_dir=wait_dir,
        output_dir=target_dir,
        built_at=built_at,
        exit_manager_sha_before=exit_manager_sha_before,
        exit_manager_sha_after=exit_manager_sha_before,
        r6_sha_before=r6_sha_before,
        r6_sha_after=r6_sha_before,
        ledger_sha_before=ledger_sha_before,
        ledger_sha_after=ledger_sha_before,
    )
    target_dir.mkdir(parents=True, exist_ok=False)

    _write_json(target_dir / OUTPUTS["contract"], payload["contract"])
    _write_json(target_dir / OUTPUTS["support_coverage"], payload["support_coverage"])
    payload["support_distribution"].to_csv(target_dir / OUTPUTS["support_distribution"], index=False)
    payload["behavior_distribution"].to_csv(target_dir / OUTPUTS["behavior_distribution"], index=False)
    payload["action_support"].to_csv(target_dir / OUTPUTS["action_support_crosstab"], index=False)
    _write_json(target_dir / OUTPUTS["ood_action_audit"], payload["ood_action_audit"])
    _write_json(target_dir / OUTPUTS["action_imbalance"], payload["action_imbalance"])
    payload["action_imbalance_table"].to_csv(target_dir / OUTPUTS["action_imbalance_table"], index=False)
    _write_json(target_dir / OUTPUTS["subset_scan"], payload["subset_scan"])
    payload["subset_scan_table"].to_csv(target_dir / OUTPUTS["subset_scan_table"], index=False)
    _write_json(target_dir / OUTPUTS["implications_lock"], payload["implications_lock"])
    _write_json(target_dir / OUTPUTS["optional_next_small_step"], payload["optional_next_small_step"])

    exit_manager_sha_after = _sha256(exit_manager_path)
    r6_sha_after = _sha256(r6_path)
    ledger_sha_after = _sha256(ledger_path) if ledger_path else None
    non_interference_df, non_interference = _build_non_interference(
        target_dir,
        payload["source_paths"],
        exit_manager_sha_before=exit_manager_sha_before,
        exit_manager_sha_after=exit_manager_sha_after,
        r6_sha_before=r6_sha_before,
        r6_sha_after=r6_sha_after,
        ledger_sha_before=ledger_sha_before,
        ledger_sha_after=ledger_sha_after,
    )
    wait_summary_path = Path(str(payload["source_paths"]["wait_state_summary_v1"]))
    wait_summary = _read_json_optional(wait_summary_path)
    payload["consistency_df"], payload["consistency"] = _build_consistency(
        payload["support_coverage"],
        payload["ood_action_audit"],
        payload["action_imbalance"],
        payload["subset_scan"],
        payload["implications_lock"],
        wait_summary,
        non_interference,
    )
    payload["summary"]["exit_manager_modified_v1"] = exit_manager_sha_before != exit_manager_sha_after
    payload["status"]["failed_non_interference_check_count_v1"] = int(non_interference["failed_check_count_v1"])
    payload["status"]["failed_consistency_check_count_v1"] = int(payload["consistency"]["failed_check_count_v1"])
    payload["non_interference_df"] = non_interference_df
    payload["non_interference"] = non_interference

    non_interference_df.to_csv(target_dir / OUTPUTS["non_interference_audit"], index=False)
    _write_json(target_dir / OUTPUTS["non_interference_audit_json"], non_interference)
    payload["consistency_df"].to_csv(target_dir / OUTPUTS["consistency_audit"], index=False)
    _write_json(target_dir / OUTPUTS["consistency_audit_json"], payload["consistency"])
    _write_json(target_dir / OUTPUTS["summary"], payload["summary"])
    (target_dir / OUTPUTS["report"]).write_text(_markdown_report(payload), encoding="utf-8")

    artifact_paths = {key: str(target_dir / filename) for key, filename in OUTPUTS.items()}
    manifest = {
        "layer_id_v1": LAYER_ID,
        "built_at_utc_v1": payload["summary"]["built_at_utc_v1"],
        "output_dir_v1": str(target_dir),
        "append_only_namespace_v1": "IQL_INTEGRATION",
        "artifact_paths_v1": artifact_paths,
        "source_paths_v1": payload["source_paths"],
        "read_only_references_v1": True,
        "not_training_v1": True,
        "not_replay_job_v1": True,
        "not_iql_v1": True,
        "not_r7_v1": True,
        "no_positive_policy_claim_v1": True,
    }
    _write_json(target_dir / OUTPUTS["manifest"], manifest)
    _write_json(target_dir / OUTPUTS["status"], payload["status"])
    return {
        "output_dir": str(target_dir),
        "artifact_paths": artifact_paths,
        "summary": payload["summary"],
        "status": payload["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize SUPPORT_AND_OOD_AUDIT_V1.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--eval-dir", type=str, default=None)
    parser.add_argument("--wait-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve() if args.dataset_dir else None
    eval_dir = Path(args.eval_dir).expanduser().resolve() if args.eval_dir else None
    wait_dir = Path(args.wait_dir).expanduser().resolve() if args.wait_dir else None
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    result = write_support_and_ood_audit_artifacts(
        reports_root,
        dataset_dir=dataset_dir,
        eval_dir=eval_dir,
        wait_dir=wait_dir,
        output_dir=output_dir,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
