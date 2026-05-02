#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate
from gx1.scripts import materialize_refine_clean_as_of_safety_layer_to_retain_safe_core_v1 as refine_gate


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_V1"

INPUT_CONTRACT_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "BUILD_IQL_OFFLINE_DATA_CONTRACT_RESEARCH_ONLY_V1_20260428T190901Z_LOCK"
)
INPUT_LANE_PACK_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "PARALLEL_CONTEXTUAL_IQL_STATE_ACTION_RESEARCH_LANE_PACK_V1_20260429T062019Z_LOCK"
)
INPUT_REFINE_CLEAN_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "REFINE_CLEAN_AS_OF_SAFETY_LAYER_TO_RETAIN_SAFE_CORE_V1_20260428T185018Z_LOCK"
)

TRADE_OUTCOMES_ROOT_GLOB_PATTERN = "TRUTH_MONFRI_WEEK_*"
TRADE_OUTCOMES_FILE_PATTERN = "trade_outcomes_*_MERGED.parquet"

EXPECTED_FRAME_ROWS = 1914
EXPECTED_HARDENED_ROWS = 89
EXPECTED_SHIELD_ROWS = 78

V1_STATE_FIELD_NAMES_FROZEN = [
    "candidate_score_v1",
    "signal_r5_1_bad_score_v1",
    "signal_r5_bad_score_v1",
    "signal_r5_tail_score_v1",
    "signal_v2_like_bad_tail_v1",
    "signal_tail_repair_v1",
    "run_id_policy_class_v1",
    "structural_low_support_v1",
    "zero_denominator_group_v1",
]
V1_STATE_DENIED_FIELDS_FROZEN = [
    "decision_timestamp_v1",
    "run_id_v1",
    "fold_id_v1",
    "student_oof_score_v1",
    "student_core_selected_v1",
    "bad_label_v1",
    "tail_label_v1",
    "unsafe_audit_v1",
    "safety_clear_audit_v1",
    "hard_veto_clear_shadow_v1",
    "source_evidence_v1",
    "HISTORICAL_V2_BLUEPRINT",
    "candidate_uid_v1",
    "trade_uid_v1",
    "selected_original_140_v1",
    "is_185_139_teacher_v1",
    "is_plus45_diagnostic_v1",
    "rows_added_vs_140_94_v1",
    "protected_winner_status_v1",
    "runner_protect_status_v1",
    "ambiguous_high_mfe_status_v1",
    "fifty_plus_mfe_risk_v1",
]

NEW_FIELD_NAME_BLOCKLIST_PATTERN = re.compile(
    r"(mfe|mae|hindsight|peak|giveback|cata|teacher|student_oof|student_core"
    r"|140_baseline|185_139|plus45|protected_winner|runner_protect"
    r"|bad_label|tail_label|unsafe_audit|safety_clear|hard_veto|safe_recoverable"
    r"|HISTORICAL_V2_BLUEPRINT|candidate_uid|trade_uid|trade_id|fold_id"
    r"|decision_timestamp|membership|selected_by"
    r"|lane_id|lane_selected|rows_added|rows_lost|active_quarantine"
    r"|r5_2_package_selected|hundred_plus|two_hundred_plus|fifty_plus"
    r"|candidate_id|candidate_selected)",
    re.IGNORECASE,
)

REGIME_PATTERN = re.compile(
    r"(session|regime|phase|weekday|hour_bucket|day_of_week|asia|eu|us|overlap)",
    re.IGNORECASE,
)
UNCERTAINTY_PATTERN = re.compile(
    r"(disagree|dispersion|active_count|score_pctile|score_std|cv_|score_rank|score_quantile)",
    re.IGNORECASE,
)
MARGIN_PATTERN = re.compile(
    r"(margin|distance|threshold)",
    re.IGNORECASE,
)
SOURCE_QUALITY_PATTERN = re.compile(
    r"(evidence|lineage|repair_path|present|count|policy_class|model_family|interpretability)",
    re.IGNORECASE,
)

FAMILY_PATTERNS = {
    "REGIME": REGIME_PATTERN,
    "UNCERTAINTY": UNCERTAINTY_PATTERN,
    "MARGIN": MARGIN_PATTERN,
    "SOURCE_QUALITY": SOURCE_QUALITY_PATTERN,
}

REWARD_VARIANT_SPECS = [
    {
        "reward_id_v1": "ENTRY_REALIZED_PNL_REWARD_V2",
        "formula_v1": "pnl_bps",
        "input_fields_v1": ["pnl_bps"],
        "input_class_v1": "HINDSIGHT_TERMINAL_OUTCOME_REWARD_ONLY",
        "sign_v1": "MAXIMIZE",
        "clip_v1": None,
    },
    {
        "reward_id_v1": "ENTRY_MFE_CAPTURE_REWARD_V2",
        "formula_v1": "pnl_bps / max(mfe_bps, eps), clipped [-2, 2]",
        "input_fields_v1": ["pnl_bps", "mfe_bps"],
        "input_class_v1": "HINDSIGHT_PATH_OUTCOME_REWARD_ONLY",
        "sign_v1": "MAXIMIZE",
        "clip_v1": [-2.0, 2.0],
    },
    {
        "reward_id_v1": "ENTRY_MAE_BURDEN_REWARD_V2",
        "formula_v1": "pnl_bps - 0.5 * abs(mae_bps)",
        "input_fields_v1": ["pnl_bps", "mae_bps"],
        "input_class_v1": "HINDSIGHT_PATH_OUTCOME_REWARD_ONLY",
        "sign_v1": "MAXIMIZE",
        "clip_v1": None,
    },
    {
        "reward_id_v1": "ENTRY_TRANSPARENT_COMBINED_REWARD_V2",
        "formula_v1": "pnl_bps - 0.25*abs(mae_bps) - 0.25*max(mfe_bps - pnl_bps, 0)",
        "input_fields_v1": ["pnl_bps", "mae_bps", "mfe_bps"],
        "input_class_v1": "MIXED_HINDSIGHT_COMPOSITE_REWARD_ONLY",
        "sign_v1": "MAXIMIZE",
        "clip_v1": None,
    },
]

REWARD_INPUT_FIELDS_FORBIDDEN_AS_STATE = {
    "mfe_bps",
    "mae_bps",
    "pnl_bps",
    "post_exit_mfe_bps",
    "early_exit_regret",
    "duration_bars",
    "exit_reason",
}

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")

ALLOWED_FINAL_STATUSES = {
    "REBUILD_STATE_CONTRACT_PASS_V2_READY_REWARD_VARIANTS_LOCKED_TIMING_AUDIT_AVAILABLE",
    "REBUILD_STATE_PARTIAL_REWARD_VARIANTS_LOCKED_STATE_INSUFFICIENT",
    "REBUILD_STATE_PARTIAL_STATE_OK_REWARD_JOIN_NOT_ESTABLISHED",
    "REBUILD_STATE_PARTIAL_TIMING_NOT_ESTABLISHED",
    "REBUILD_STATE_BLOCKED_NO_NEW_AS_OF_FIELDS_AND_REWARD_JOIN_FAILED",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1",
    "DEEPEN_IQL_STATE_FAMILY_DISCOVERY_V1",
    "REPAIR_REWARD_JOIN_LINEAGE_V1",
    "DEEPEN_TIMING_AUDIT_ALT_PATH_V1",
    "HOLD_UNTIL_NEW_AS_OF_FAMILIES_LANDED_V1",
}

REQUIRED_OUTPUTS_TOPLEVEL = [
    "manifest_v1.json",
    "summary_v1.json",
    "status_v1.json",
    "report_v1.md",
    "rebuild_iql_state_contract_with_more_as_of_features_go_no_go_v1.json",
    "input_manifest_v1.json",
    "reproducibility_audit_v1.json",
]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    return contract_gate._jsonable(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    contract_gate._write_json(path, payload)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    contract_gate._write_rows(path, rows)


def _write_report(path: Path, lines: Sequence[str]) -> None:
    contract_gate._write_report(path, lines)


def _read_json(path: Path) -> dict[str, Any]:
    return contract_gate._read_json(path)


def _file_hash(path: Path) -> str:
    return contract_gate._file_hash(path)


def _python_manifest() -> dict[str, Any]:
    return contract_gate._python_manifest()


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    return refine_gate._bool(frame, column)


def validate_explicit_artifact_roots(paths: Iterable[Path]) -> bool:
    return contract_gate.validate_explicit_artifact_roots(paths)


def validate_no_forbidden_actions(**kwargs: Any) -> dict[str, Any]:
    return contract_gate.validate_no_forbidden_actions(**kwargs)


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_no_deprecated_revival(script_path: Path) -> bool:
    text = script_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.lstrip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        for fragment in QUARANTINE_FORBIDDEN_PATH_FRAGMENTS:
            if fragment in stripped:
                raise RuntimeError("DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN")
    return True


def _load_inputs() -> dict[str, Any]:
    roots = [INPUT_CONTRACT_ROOT, INPUT_LANE_PACK_ROOT, INPUT_REFINE_CLEAN_ROOT]
    validate_explicit_artifact_roots(roots)
    required = {
        "v1_state_contract": INPUT_CONTRACT_ROOT / "iql_offline_state_contract_v1.json",
        "v1_reward_contract": INPUT_CONTRACT_ROOT / "iql_offline_reward_contract_v1.json",
        "v1_action_contract": INPUT_CONTRACT_ROOT / "iql_offline_action_contract_v1.json",
        "v1_safety_shield_contract": INPUT_CONTRACT_ROOT / "iql_offline_safety_shield_contract_v1.json",
        "v1_data_contract_go_no_go": INPUT_CONTRACT_ROOT
        / "build_iql_offline_data_contract_research_only_go_no_go_v1.json",
        "lane_pack_summary": INPUT_LANE_PACK_ROOT
        / "contextual_iql_parallel_lane_pack_summary_v1.json",
        "lane_pack_fan_in": INPUT_LANE_PACK_ROOT
        / "contextual_iql_parallel_fan_in_recommendation_v1.json",
        "lane_02_result": INPUT_LANE_PACK_ROOT
        / "LANE_02_AS_OF_SOURCE_STATE_FEATURE_EXPANSION"
        / "lane_result_v1.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_INPUT_ARTIFACTS: {missing}")
    return {
        "required_paths": required,
        "v1_state_contract": _read_json(required["v1_state_contract"]),
        "v1_reward_contract": _read_json(required["v1_reward_contract"]),
        "v1_action_contract": _read_json(required["v1_action_contract"]),
        "v1_safety_shield_contract": _read_json(required["v1_safety_shield_contract"]),
        "v1_data_contract_go_no_go": _read_json(required["v1_data_contract_go_no_go"]),
        "lane_pack_summary": _read_json(required["lane_pack_summary"]),
        "lane_pack_fan_in": _read_json(required["lane_pack_fan_in"]),
        "lane_02_result": _read_json(required["lane_02_result"]),
    }


def _frame_and_masks() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    inputs = refine_gate._load_inputs()
    frame, masks = refine_gate._build_frame_and_masks(inputs)
    if frame.shape[0] != EXPECTED_FRAME_ROWS:
        raise RuntimeError(
            f"FRAME_ROW_COUNT_MISMATCH: got {frame.shape[0]}, expected {EXPECTED_FRAME_ROWS}"
        )
    if int(masks["hardened"].sum()) != EXPECTED_HARDENED_ROWS:
        raise RuntimeError("HARDENED_MASK_COUNT_MISMATCH")
    shielded = masks["hardened"] & ~masks["source_confluence_repairable_v1"]
    if int(shielded.sum()) != EXPECTED_SHIELD_ROWS:
        raise RuntimeError("SHIELD_MASK_COUNT_MISMATCH")
    return frame, masks


def _build_input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "REBUILD_IQL_STATE_CONTRACT_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "v1_data_contract_root_v1": str(INPUT_CONTRACT_ROOT),
            "lane_pack_root_v1": str(INPUT_LANE_PACK_ROOT),
            "refine_clean_root_v1": str(INPUT_REFINE_CLEAN_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": False,
        "iql_production_allowed_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


# ---------------------------------------------------------------------------
# STATE_EXPANSION_V2
# ---------------------------------------------------------------------------


def _classify_family(field: str) -> str | None:
    for family, pattern in FAMILY_PATTERNS.items():
        if pattern.search(field):
            return family
    return None


def _state_v2_discovery(frame: pd.DataFrame) -> dict[str, Any]:
    v1_known = set(V1_STATE_FIELD_NAMES_FROZEN) | set(V1_STATE_DENIED_FIELDS_FROZEN)
    candidates = [c for c in frame.columns if c not in v1_known]
    family_records: dict[str, list[dict[str, Any]]] = {f: [] for f in FAMILY_PATTERNS}
    family_records["UNCLASSIFIED"] = []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_series_by_field: dict[str, pd.Series] = {}
    accepted_for_v1 = {
        name: frame[name] for name in V1_STATE_FIELD_NAMES_FROZEN if name in frame.columns
    }

    for field in candidates:
        family = _classify_family(field)
        record_family = family or "UNCLASSIFIED"
        record: dict[str, Any] = {
            "field_name_v1": field,
            "family_v1": record_family,
            "present_in_source_frame_v1": True,
            "decision_v1": "PENDING",
            "reason_v1": "",
        }
        if family is None:
            record["decision_v1"] = "REJECT_NO_FAMILY_PATTERN_MATCH"
            record["reason_v1"] = "name does not match any family pattern"
            family_records["UNCLASSIFIED"].append(record)
            rejected.append(record)
            continue
        if NEW_FIELD_NAME_BLOCKLIST_PATTERN.search(field):
            record["decision_v1"] = "REJECT_NAME_BLOCKLIST_MATCH"
            record["reason_v1"] = "field name matches forbidden token pattern"
            family_records[family].append(record)
            rejected.append(record)
            continue
        series = frame[field]
        non_null = int(series.notna().sum())
        record["non_null_count_v1"] = non_null
        record["coverage_v1"] = float(non_null / max(len(frame), 1))
        if non_null < 1500:
            record["decision_v1"] = "REJECT_LOW_COVERAGE"
            record["reason_v1"] = f"non-null {non_null} < 1500 floor"
            family_records[family].append(record)
            rejected.append(record)
            continue
        coverage_status = (
            "FULL_COVERAGE" if non_null >= 1900 else "NEEDS_REJECT_ROW_HANDLING"
        )
        record["coverage_status_v1"] = coverage_status
        try:
            numeric = pd.to_numeric(series, errors="coerce")
        except (TypeError, ValueError):
            numeric = pd.Series([], dtype="float64")
        if numeric.notna().sum() >= max(int(0.9 * non_null), 1):
            variance = float(numeric.var(ddof=0)) if numeric.notna().sum() > 1 else 0.0
            record["variance_v1"] = variance
            if variance < 1e-9:
                record["decision_v1"] = "REJECT_DEGENERATE"
                record["reason_v1"] = f"variance {variance:.3e} < 1e-9"
                family_records[family].append(record)
                rejected.append(record)
                continue
            corr_check_series = numeric.fillna(0.0)
            kind = "NUMERIC"
        elif series.dtype == bool or series.dtype == "boolean":
            corr_check_series = series.astype("Int64").fillna(0).astype("float64")
            unique_vals = int(corr_check_series.nunique(dropna=True))
            record["unique_value_count_v1"] = unique_vals
            if unique_vals < 2:
                record["decision_v1"] = "REJECT_DEGENERATE"
                record["reason_v1"] = "boolean field has < 2 unique values"
                family_records[family].append(record)
                rejected.append(record)
                continue
            kind = "BOOL"
        else:
            stringified = series.astype("object").where(series.notna(), "__NULL__")
            unique_vals = int(stringified.nunique(dropna=False))
            record["unique_value_count_v1"] = unique_vals
            if unique_vals < 2:
                record["decision_v1"] = "REJECT_DEGENERATE"
                record["reason_v1"] = "categorical field has < 2 unique values"
                family_records[family].append(record)
                rejected.append(record)
                continue
            corr_check_series = pd.Series(
                pd.Categorical(stringified).codes,
                index=series.index,
            ).astype("float64")
            kind = "CATEGORICAL"
        max_corr = 0.0
        max_corr_against = ""
        existing_pool: dict[str, pd.Series] = {**accepted_for_v1}
        for accepted_record in accepted:
            existing_pool[accepted_record["field_name_v1"]] = accepted_series_by_field[
                accepted_record["field_name_v1"]
            ]
        for other_name, other_series in existing_pool.items():
            try:
                if other_series.dtype == bool or other_series.dtype == "boolean":
                    other_corr = other_series.astype("Int64").fillna(0).astype("float64")
                else:
                    other_numeric = pd.to_numeric(other_series, errors="coerce")
                    if other_numeric.notna().sum() < 0.5 * len(other_series):
                        other_corr = pd.Series(
                            pd.Categorical(other_series.astype(str).fillna("__NULL__")).codes,
                            index=other_series.index,
                        ).astype("float64")
                    else:
                        other_corr = other_numeric.fillna(0.0)
                if other_corr.std(ddof=0) < 1e-12 or corr_check_series.std(ddof=0) < 1e-12:
                    continue
                corr_value = float(corr_check_series.corr(other_corr))
                if not math.isnan(corr_value) and abs(corr_value) > max_corr:
                    max_corr = abs(corr_value)
                    max_corr_against = other_name
            except (TypeError, ValueError):
                continue
        record["max_abs_corr_v1"] = float(max_corr)
        record["max_abs_corr_against_v1"] = max_corr_against
        if max_corr > 0.99:
            record["decision_v1"] = "REJECT_NEAR_DUPLICATE"
            record["reason_v1"] = (
                f"|corr| {max_corr:.4f} > 0.99 vs {max_corr_against}"
            )
            family_records[family].append(record)
            rejected.append(record)
            continue
        record["kind_v1"] = kind
        record["decision_v1"] = "ACCEPT"
        record["reason_v1"] = "passed family pattern, blocklist, coverage, variance, correlation"
        record["normalization_needed_v1"] = kind in {"NUMERIC", "CATEGORICAL"}
        record["leakage_risk_v1"] = "LOW" if kind == "BOOL" else "MEDIUM_CONTROLLED"
        family_records[family].append(record)
        accepted.append(record)
        accepted_series_by_field[field] = series

    family_status: dict[str, str] = {}
    for family in FAMILY_PATTERNS:
        accepted_in_family = [
            r for r in family_records[family] if r["decision_v1"] == "ACCEPT"
        ]
        family_status[family] = (
            "QUALIFYING_AS_OF_CANDIDATE_ACCEPTED"
            if accepted_in_family
            else "NOT_ESTABLISHED_NO_QUALIFYING_AS_OF_CANDIDATE"
        )
    summary = {
        "total_candidate_columns_v1": len(candidates),
        "rejected_count_v1": len(rejected),
        "accepted_count_v1": len(accepted),
        "family_status_v1": family_status,
        "accepted_field_names_v1": [r["field_name_v1"] for r in accepted],
    }
    return {
        "summary_v1": summary,
        "family_records_v1": family_records,
        "accepted_v1": accepted,
        "rejected_v1": rejected,
    }


def _v1_state_rows() -> list[dict[str, Any]]:
    contract = _read_json(
        INPUT_CONTRACT_ROOT / "iql_offline_state_contract_v1.json"
    )
    rows = list(contract.get("rows_v1", []))
    if not rows:
        raise RuntimeError("V1_STATE_CONTRACT_EMPTY")
    return rows


def _build_state_v2_contract(
    discovery: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    v1_rows = _v1_state_rows()
    v1_by_name = {row["field_name_v1"]: row for row in v1_rows}
    v2_rows: list[dict[str, Any]] = []
    diff: list[dict[str, Any]] = []
    for row in v1_rows:
        out = dict(row)
        if row["field_name_v1"] in V1_STATE_FIELD_NAMES_FROZEN:
            family = "EXISTING_V1_ALLOWED"
        else:
            family = "EXISTING_V1_DENIED"
        out["family_v1"] = family
        v2_rows.append(out)
        diff.append(
            {
                "field_name_v1": row["field_name_v1"],
                "change_v1": "UNCHANGED_FROM_V1",
                "family_v1": family,
                "allowed_as_state_v1": row["allowed_as_state_v1"],
            }
        )
    for accepted in discovery["accepted_v1"]:
        family = accepted["family_v1"]
        kind = accepted.get("kind_v1", "BOOL")
        datatype = (
            "float"
            if kind == "NUMERIC"
            else ("category" if kind == "CATEGORICAL" else "bool")
        )
        v2_row = {
            "field_name_v1": accepted["field_name_v1"],
            "source_artifact_path_v1": "tail-repaired R5.2/source candidate frame",
            "datatype_v1": datatype,
            "as_of_lineage_v1": (
                "AS_OF_SAFE_NEEDS_NORMALIZATION"
                if accepted["normalization_needed_v1"]
                else "AS_OF_SAFE_ALLOWED"
            ),
            "allowed_as_state_v1": True,
            "blocked_reason_v1": "",
            "normalization_needed_v1": accepted["normalization_needed_v1"],
            "missing_handling_v1": "REJECT_ROW_IF_REQUIRED_MISSING",
            "leakage_risk_v1": accepted["leakage_risk_v1"],
            "adapter_r6_future_feasibility_v1": "POSSIBLE",
            "present_in_source_frame_v1": True,
            "family_v1": family,
        }
        v2_rows.append(v2_row)
        diff.append(
            {
                "field_name_v1": accepted["field_name_v1"],
                "change_v1": "ADDED_IN_V2",
                "family_v1": family,
                "allowed_as_state_v1": True,
            }
        )
    return v2_rows, diff


def _state_no_shortcut_audit_v2(v2_rows: list[dict[str, Any]]) -> dict[str, Any]:
    allowed_in_v2 = {
        r["field_name_v1"] for r in v2_rows if r.get("allowed_as_state_v1")
    }
    intersection_with_denied = sorted(
        allowed_in_v2 & set(V1_STATE_DENIED_FIELDS_FROZEN)
    )
    intersection_with_reward_inputs = sorted(
        allowed_in_v2 & REWARD_INPUT_FIELDS_FORBIDDEN_AS_STATE
    )
    blocklist_hits = sorted(
        f for f in allowed_in_v2 if NEW_FIELD_NAME_BLOCKLIST_PATTERN.search(f)
    )
    payload = {
        "layer_name": "REBUILD_IQL_STATE_NO_SHORTCUT_AUDIT_V2",
        "v2_allowed_field_names_v1": sorted(allowed_in_v2),
        "intersection_with_v1_denied_v1": intersection_with_denied,
        "intersection_with_reward_inputs_v1": intersection_with_reward_inputs,
        "blocklist_pattern_hits_v1": blocklist_hits,
        "no_shortcut_status_v1": "PASS",
    }
    if (
        intersection_with_denied
        or intersection_with_reward_inputs
        or blocklist_hits
    ):
        payload["no_shortcut_status_v1"] = "FAIL"
        raise RuntimeError(f"STATE_V2_NO_SHORTCUT_AUDIT_FAILED: {payload}")
    return payload


def _state_normalization_plan(v2_rows: list[dict[str, Any]]) -> dict[str, Any]:
    plan = []
    for row in v2_rows:
        if not row.get("allowed_as_state_v1"):
            continue
        if not row.get("normalization_needed_v1"):
            continue
        kind = row.get("datatype_v1", "")
        if kind == "float":
            method = "Z_SCORE_TRAIN_ONLY_SUFFIX_z_train_only_v1"
        elif kind == "category":
            method = "ONE_HOT_TRAIN_ONLY_LIMIT_TOP_K_BY_TRAIN_FREQUENCY"
        else:
            method = "PASSTHROUGH_BOOLEAN"
        plan.append(
            {
                "field_name_v1": row["field_name_v1"],
                "datatype_v1": kind,
                "normalization_method_v1": method,
                "research_only_v1": True,
            }
        )
    return {
        "layer_name": "REBUILD_IQL_STATE_NORMALIZATION_PLAN_V1",
        "plan_v1": plan,
        "research_only_v1": True,
    }


# ---------------------------------------------------------------------------
# REWARD_VARIANTS_V2
# ---------------------------------------------------------------------------


def _trade_outcomes_paths() -> list[Path]:
    weeks = sorted(DEFAULT_REPORTS_ROOT.glob(TRADE_OUTCOMES_ROOT_GLOB_PATTERN))
    paths: list[Path] = []
    for week in weeks:
        if not week.is_dir():
            continue
        matching = sorted(week.glob(TRADE_OUTCOMES_FILE_PATTERN))
        for parquet in matching:
            paths.append(parquet)
    paths = sorted(paths, key=lambda p: p.parent.name + "/" + p.name)
    if not paths:
        raise RuntimeError("TRADE_OUTCOMES_NO_FILES_FOUND")
    return paths


def _load_trade_outcomes() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    paths = _trade_outcomes_paths()
    used: list[dict[str, Any]] = []
    frames = []
    for path in paths:
        used.append(
            {
                "path_v1": str(path),
                "sha256_v1": _file_hash(path),
            }
        )
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        raise RuntimeError("TRADE_OUTCOMES_ALL_EMPTY")
    common_cols = list(frames[0].columns)
    for f in frames[1:]:
        common_cols = [c for c in common_cols if c in f.columns]
    aligned = [f.loc[:, common_cols].copy() for f in frames]
    concat = pd.concat(aligned, ignore_index=True)
    return concat, used


def _build_reward_join(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    trade_outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "candidate_uid" not in trade_outcomes.columns:
        raise RuntimeError("TRADE_OUTCOMES_MISSING_CANDIDATE_UID")
    if "candidate_uid_v1" not in frame.columns:
        raise RuntimeError("FRAME_MISSING_CANDIDATE_UID_V1")
    iql_view = pd.DataFrame(
        {
            "candidate_uid_v1": frame["candidate_uid_v1"].astype(str),
            "trade_uid_v1": frame.get(
                "trade_uid_v1", pd.Series([None] * len(frame))
            ).astype(str),
            "is_inside_78_shield_v1": (
                masks["hardened"] & ~masks["source_confluence_repairable_v1"]
            ).reset_index(drop=True),
            "is_safe_core_89_v1": masks["hardened"].reset_index(drop=True),
        }
    )
    iql_view["take_trade_action_v1"] = iql_view["is_inside_78_shield_v1"]
    outcome_view = trade_outcomes.copy()
    outcome_view["candidate_uid"] = outcome_view["candidate_uid"].astype(str)
    duplicates = outcome_view["candidate_uid"].duplicated().sum()
    join = iql_view.merge(
        outcome_view,
        left_on="candidate_uid_v1",
        right_on="candidate_uid",
        how="left",
        suffixes=("_iql", "_outcome"),
    )
    matched_mask = join["pnl_bps"].notna()
    take_match_count = int((iql_view["take_trade_action_v1"] & matched_mask).sum())
    take_count = int(iql_view["take_trade_action_v1"].sum())
    overall_match_count = int(matched_mask.sum())
    overall_match_rate = float(overall_match_count / max(len(iql_view), 1))
    take_match_rate = float(take_match_count / max(take_count, 1))
    audit = {
        "layer_name": "ENTRY_IQL_POST_TRADE_OUTCOME_JOIN_AUDIT_V1",
        "iql_dataset_row_count_v1": int(len(iql_view)),
        "trade_outcomes_row_count_v1": int(len(outcome_view)),
        "trade_outcomes_duplicate_uid_count_v1": int(duplicates),
        "overall_match_count_v1": overall_match_count,
        "overall_match_rate_v1": overall_match_rate,
        "take_trade_count_v1": take_count,
        "take_trade_match_count_v1": take_match_count,
        "take_trade_match_rate_v1": take_match_rate,
        "join_status_v1": (
            "REWARD_JOIN_LOCKED"
            if take_match_rate >= 0.95
            else "REWARD_JOIN_NOT_ESTABLISHED"
        ),
    }
    return join, audit


def _compute_reward_variants(join: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    matched = join["pnl_bps"].notna()
    out = pd.DataFrame(
        {
            "candidate_uid_v1": join["candidate_uid_v1"].values,
            "is_inside_78_shield_v1": join["is_inside_78_shield_v1"].values,
            "is_safe_core_89_v1": join["is_safe_core_89_v1"].values,
            "take_trade_action_v1": join["take_trade_action_v1"].values,
            "matched_v1": matched.values,
        }
    )
    distributions: dict[str, Any] = {}
    eps = 1e-6
    for spec in REWARD_VARIANT_SPECS:
        rid = spec["reward_id_v1"]
        col = f"{rid}_value_v1"
        out[col] = np.nan
        if rid == "ENTRY_REALIZED_PNL_REWARD_V2":
            out.loc[matched.values, col] = join.loc[matched, "pnl_bps"].astype(float).values
        elif rid == "ENTRY_MFE_CAPTURE_REWARD_V2":
            denom = np.maximum(
                join.loc[matched, "mfe_bps"].astype(float).values, eps
            )
            ratio = join.loc[matched, "pnl_bps"].astype(float).values / denom
            ratio = np.clip(ratio, -2.0, 2.0)
            out.loc[matched.values, col] = ratio
        elif rid == "ENTRY_MAE_BURDEN_REWARD_V2":
            value = (
                join.loc[matched, "pnl_bps"].astype(float).values
                - 0.5 * np.abs(join.loc[matched, "mae_bps"].astype(float).values)
            )
            out.loc[matched.values, col] = value
        elif rid == "ENTRY_TRANSPARENT_COMBINED_REWARD_V2":
            pnl = join.loc[matched, "pnl_bps"].astype(float).values
            mae = np.abs(join.loc[matched, "mae_bps"].astype(float).values)
            giveback = np.maximum(
                join.loc[matched, "mfe_bps"].astype(float).values - pnl, 0.0
            )
            value = pnl - 0.25 * mae - 0.25 * giveback
            out.loc[matched.values, col] = value
        else:
            raise RuntimeError(f"UNKNOWN_REWARD_VARIANT: {rid}")
        series = out[col].dropna()
        if series.empty:
            distributions[rid] = {
                "count_v1": 0,
                "mean_v1": None,
                "std_v1": None,
                "p5_v1": None,
                "p25_v1": None,
                "p50_v1": None,
                "p75_v1": None,
                "p95_v1": None,
                "clip_low_count_v1": 0,
                "clip_high_count_v1": 0,
            }
            continue
        clip_low = 0
        clip_high = 0
        if spec["clip_v1"] is not None:
            low, high = spec["clip_v1"]
            clip_low = int((series <= low + 1e-12).sum())
            clip_high = int((series >= high - 1e-12).sum())
        distributions[rid] = {
            "count_v1": int(len(series)),
            "mean_v1": float(series.mean()),
            "std_v1": float(series.std(ddof=0)),
            "p5_v1": float(series.quantile(0.05)),
            "p25_v1": float(series.quantile(0.25)),
            "p50_v1": float(series.quantile(0.50)),
            "p75_v1": float(series.quantile(0.75)),
            "p95_v1": float(series.quantile(0.95)),
            "clip_low_count_v1": clip_low,
            "clip_high_count_v1": clip_high,
        }
    return out, distributions


def _reward_class_audit(v2_state_rows: list[dict[str, Any]]) -> dict[str, Any]:
    allowed_state_fields = {
        r["field_name_v1"] for r in v2_state_rows if r.get("allowed_as_state_v1")
    }
    leak = sorted(allowed_state_fields & REWARD_INPUT_FIELDS_FORBIDDEN_AS_STATE)
    classification = []
    for spec in REWARD_VARIANT_SPECS:
        for field in spec["input_fields_v1"]:
            classification.append(
                {
                    "reward_id_v1": spec["reward_id_v1"],
                    "input_field_v1": field,
                    "input_class_v1": spec["input_class_v1"],
                    "is_in_v2_allowed_state_v1": field in allowed_state_fields,
                    "is_blocked_as_state_v1": field
                    in REWARD_INPUT_FIELDS_FORBIDDEN_AS_STATE,
                }
            )
    payload = {
        "layer_name": "REWARD_VARIANT_CLASS_AUDIT_V1",
        "leakage_status_v1": "PASS" if not leak else "FAIL",
        "leaked_input_fields_v1": leak,
        "variant_input_classifications_v1": classification,
        "research_only_v1": True,
    }
    if leak:
        raise RuntimeError(f"REWARD_INPUT_LEAK_INTO_STATE: {leak}")
    return payload


def _build_reward_variants_contract(
    distributions: dict[str, Any],
    join_audit: dict[str, Any],
) -> dict[str, Any]:
    variants_locked = join_audit["join_status_v1"] == "REWARD_JOIN_LOCKED"
    variants = []
    for spec in REWARD_VARIANT_SPECS:
        rid = spec["reward_id_v1"]
        variants.append(
            {
                **spec,
                "distribution_v1": distributions.get(rid, {}),
                "lock_status_v1": (
                    "LOCKED_FOR_NEXT_GATE"
                    if variants_locked and distributions.get(rid, {}).get("count_v1", 0) > 0
                    else "NOT_ESTABLISHED"
                ),
                "use_as_state_v1": False,
                "use_as_selector_v1": False,
            }
        )
    return {
        "layer_name": "IQL_ENTRY_IQL_REWARD_VARIANTS_CONTRACT_V2",
        "variants_v1": variants,
        "join_audit_summary_v1": {
            "iql_dataset_row_count_v1": join_audit["iql_dataset_row_count_v1"],
            "overall_match_rate_v1": join_audit["overall_match_rate_v1"],
            "take_trade_match_rate_v1": join_audit["take_trade_match_rate_v1"],
            "join_status_v1": join_audit["join_status_v1"],
        },
        "research_only_v1": True,
        "next_gate_hook_v1": "RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1",
    }


# ---------------------------------------------------------------------------
# TIMING_AUDIT_V1
# ---------------------------------------------------------------------------


def _build_timing_audit(
    join: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    matched = join["pnl_bps"].notna()
    audit_table = pd.DataFrame(
        {
            "candidate_uid_v1": join["candidate_uid_v1"].values,
            "is_inside_78_shield_v1": join["is_inside_78_shield_v1"].values,
            "matched_v1": matched.values,
            "pnl_bps_v1": join["pnl_bps"].values,
            "mfe_bps_v1": join["mfe_bps"].values,
            "mae_bps_v1": join["mae_bps"].values,
            "early_exit_regret_v1": (
                join["early_exit_regret"].values
                if "early_exit_regret" in join.columns
                else np.nan
            ),
            "exit_reason_v1": (
                join["exit_reason"].astype(str).values
                if "exit_reason" in join.columns
                else ""
            ),
        }
    )
    eps = 1e-6
    audit_table["mae_dominated_v1"] = (
        matched.values
        & (np.abs(audit_table["mae_bps_v1"].fillna(0).values) > audit_table["mfe_bps_v1"].fillna(0).values)
        & (audit_table["pnl_bps_v1"].fillna(0).values < 0)
    )
    audit_table["peak_giveback_v1"] = (
        matched.values
        & (audit_table["mfe_bps_v1"].fillna(0).values > eps)
        & (
            (audit_table["mfe_bps_v1"].fillna(0).values - audit_table["pnl_bps_v1"].fillna(0).values)
            > 0.5 * audit_table["mfe_bps_v1"].fillna(0).values
        )
    )
    audit_table["cata_exit_v1"] = (
        matched.values
        & (audit_table["exit_reason_v1"] == "CATASTROPHIC_GUARD")
    )
    audit_table["peak_timing_label_v1"] = "NOT_ESTABLISHED_REQUIRES_INTRABAR_TRACE"

    shield_mask = audit_table["is_inside_78_shield_v1"].astype(bool)
    shielded_table = audit_table[shield_mask]
    shielded_matched = shielded_table[shielded_table["matched_v1"]]
    summary = {
        "layer_name": "ENTRY_TIMING_AUDIT_RECOMMENDATION_V1",
        "approach_v1": "ALT_A_TRADE_OUTCOMES_POST_HOC",
        "deprecated_revival_used_v1": False,
        "shadow_counterfactual_v2_used_v1": False,
        "iql_dataset_row_count_v1": int(len(audit_table)),
        "shielded_row_count_v1": int(shield_mask.sum()),
        "shielded_matched_row_count_v1": int(len(shielded_matched)),
        "shielded_mae_dominated_count_v1": int(shielded_matched["mae_dominated_v1"].sum()),
        "shielded_peak_giveback_count_v1": int(shielded_matched["peak_giveback_v1"].sum()),
        "shielded_cata_exit_count_v1": int(shielded_matched["cata_exit_v1"].sum()),
        "peak_timing_label_status_v1": "NOT_ESTABLISHED_REQUIRES_INTRABAR_TRACE",
        "eligibility_v1": "AUDIT_TABLE_ONLY_NEVER_STATE_NEVER_SELECTOR_NEVER_REWARD",
        "research_only_v1": True,
    }
    timing_status = (
        "TIMING_AUDIT_AVAILABLE"
        if summary["shielded_matched_row_count_v1"] > 0
        else "TIMING_AUDIT_NOT_ESTABLISHED"
    )
    summary["timing_status_v1"] = timing_status
    return audit_table, summary, summary


# ---------------------------------------------------------------------------
# Reproducibility / final assembly
# ---------------------------------------------------------------------------


def _reproducibility_audit(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    join_audit: dict[str, Any],
    state_v2_summary: dict[str, Any],
) -> dict[str, Any]:
    shielded_mask = masks["hardened"] & ~masks["source_confluence_repairable_v1"]
    payload = {
        "layer_name": "REBUILD_IQL_STATE_REPRODUCIBILITY_AUDIT_V1",
        "frame_row_count_v1": int(len(frame)),
        "expected_frame_rows_v1": EXPECTED_FRAME_ROWS,
        "row_count_invariant_v1": int(len(frame)) == EXPECTED_FRAME_ROWS,
        "hardened_count_v1": int(masks["hardened"].sum()),
        "expected_hardened_v1": EXPECTED_HARDENED_ROWS,
        "shielded_count_v1": int(shielded_mask.sum()),
        "expected_shielded_v1": EXPECTED_SHIELD_ROWS,
        "seventy_eight_shield_invariant_v1": int(shielded_mask.sum()) == EXPECTED_SHIELD_ROWS,
        "iql_join_dataset_size_v1": join_audit["iql_dataset_row_count_v1"],
        "state_v2_added_count_v1": state_v2_summary["accepted_count_v1"],
        "state_v2_rejected_count_v1": state_v2_summary["rejected_count_v1"],
        "research_only_v1": True,
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
    }
    if not payload["row_count_invariant_v1"]:
        raise RuntimeError("ROW_COUNT_INVARIANT_FAILED")
    if not payload["seventy_eight_shield_invariant_v1"]:
        raise RuntimeError("SEVENTY_EIGHT_SHIELD_INVARIANT_FAILED")
    return payload


def _go_no_go(
    state_v2_summary: dict[str, Any],
    join_audit: dict[str, Any],
    timing_summary: dict[str, Any],
) -> tuple[str, str, str]:
    state_pass = state_v2_summary["accepted_count_v1"] >= 2
    reward_pass = join_audit["join_status_v1"] == "REWARD_JOIN_LOCKED"
    timing_pass = timing_summary["timing_status_v1"] == "TIMING_AUDIT_AVAILABLE"
    if state_pass and reward_pass and timing_pass:
        status = (
            "REBUILD_STATE_CONTRACT_PASS_V2_READY_REWARD_VARIANTS_LOCKED_TIMING_AUDIT_AVAILABLE"
        )
        next_action = "RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1"
        recommendation = (
            "All three sub-tracks passed: V2 state contract has new AS_OF families, reward variants "
            "joined cleanly to trade outcomes, and Alt A timing audit is available. Next gate trains "
            "contextual IQL with V2 state and the four reward variants as parallel research-only "
            "comparators alongside SAFETY_WEIGHTED_REWARD."
        )
    elif (not state_pass) and reward_pass:
        status = "REBUILD_STATE_PARTIAL_REWARD_VARIANTS_LOCKED_STATE_INSUFFICIENT"
        next_action = "DEEPEN_IQL_STATE_FAMILY_DISCOVERY_V1"
        recommendation = (
            "Reward variants locked but fewer than two new AS_OF state fields qualified. The source "
            "frame is likely too thin in regime/uncertainty/margin/source-quality families. Next gate "
            "deepens family discovery via additional AS_OF source signals before running IQL with V2."
        )
    elif state_pass and (not reward_pass):
        status = "REBUILD_STATE_PARTIAL_STATE_OK_REWARD_JOIN_NOT_ESTABLISHED"
        next_action = "REPAIR_REWARD_JOIN_LINEAGE_V1"
        recommendation = (
            "V2 state contract passed but trade-outcomes join did not reach 0.95 match rate on the 78 "
            "shielded TAKE cohort. Repair candidate_uid lineage between IQL frame and trade outcomes "
            "before locking reward variants."
        )
    elif state_pass and reward_pass and (not timing_pass):
        status = "REBUILD_STATE_PARTIAL_TIMING_NOT_ESTABLISHED"
        next_action = "DEEPEN_TIMING_AUDIT_ALT_PATH_V1"
        recommendation = (
            "State and reward sub-tracks passed but Alt A timing audit returned no shielded matched "
            "rows. Deepen timing audit via an alternative path (e.g., intra-bar trace reconstruction) "
            "before timing diagnostics can support next-gate analysis."
        )
    else:
        status = (
            "REBUILD_STATE_BLOCKED_NO_NEW_AS_OF_FIELDS_AND_REWARD_JOIN_FAILED"
        )
        next_action = "HOLD_UNTIL_NEW_AS_OF_FAMILIES_LANDED_V1"
        recommendation = (
            "Neither state expansion nor reward join produced research-ready output. Hold further IQL "
            "research until new AS_OF families and verifiable reward lineage are landed."
        )
    validate_final_status(status, next_action)
    return status, next_action, recommendation


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    inputs = _load_inputs()
    frame, masks = _frame_and_masks()
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (
        DEFAULT_REPORTS_ROOT / f"REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_V1_{timestamp}_LOCK"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    state_dir = artifact_root / "STATE_EXPANSION_V2"
    reward_dir = artifact_root / "REWARD_VARIANTS_V2"
    timing_dir = artifact_root / "TIMING_AUDIT_V1"
    for sub in (state_dir, reward_dir, timing_dir):
        sub.mkdir(parents=True, exist_ok=True)

    # Self no-deprecated-revival check
    validate_no_deprecated_revival(Path(__file__))

    # Forbidden-actions guard
    forbidden_audit = validate_no_forbidden_actions(
        r6=False,
        adapter=False,
        iql_production=False,
        iql_training_now=False,
        package=False,
        freeze=False,
        promo=False,
        live=False,
        optuna=False,
        broad_sweep=False,
    )

    # Input manifest
    input_manifest = _build_input_manifest(inputs, artifact_root)
    _write_json(artifact_root / "input_manifest_v1.json", input_manifest)

    # State expansion
    discovery = _state_v2_discovery(frame)
    v2_rows, diff_rows = _build_state_v2_contract(discovery)
    no_shortcut_audit = _state_no_shortcut_audit_v2(v2_rows)
    normalization_plan = _state_normalization_plan(v2_rows)

    state_contract_v2 = {
        "row_count_v1": len(v2_rows),
        "rows_v1": v2_rows,
        "v1_baseline_field_count_v1": len(_v1_state_rows()),
        "v2_added_count_v1": discovery["summary_v1"]["accepted_count_v1"],
        "family_status_v1": discovery["summary_v1"]["family_status_v1"],
    }
    _write_json(state_dir / "iql_offline_state_contract_v2.json", state_contract_v2)
    _write_rows(state_dir / "iql_offline_state_contract_v2_diff_vs_v1.csv", diff_rows)
    family_inventory = {
        "summary_v1": discovery["summary_v1"],
        "family_records_v1": discovery["family_records_v1"],
    }
    _write_json(state_dir / "state_family_inventory_v1.json", family_inventory)
    metric_rows: list[dict[str, Any]] = []
    for family, records in discovery["family_records_v1"].items():
        for record in records:
            metric_rows.append({"family_v1": family, **record})
    _write_rows(state_dir / "state_family_inventory_dry_run_metrics_v1.csv", metric_rows)
    _write_json(state_dir / "state_no_shortcut_audit_v2.json", no_shortcut_audit)
    _write_json(state_dir / "state_normalization_plan_v1.json", normalization_plan)

    # Trade outcomes load
    trade_outcomes, trade_outcomes_files = _load_trade_outcomes()
    _write_json(
        artifact_root / "trade_outcomes_input_files_v1.json",
        {
            "files_v1": trade_outcomes_files,
            "non_empty_concat_row_count_v1": int(len(trade_outcomes)),
        },
    )

    # Reward join + variants
    join, join_audit = _build_reward_join(frame, masks, trade_outcomes)
    _write_json(
        reward_dir / "entry_iql_post_trade_outcome_join_audit_v1.json", join_audit
    )
    join_table_path = reward_dir / "entry_iql_post_trade_outcome_join_table_v1.csv"
    join_export_cols = [
        "candidate_uid_v1",
        "trade_uid_v1",
        "is_inside_78_shield_v1",
        "is_safe_core_89_v1",
        "take_trade_action_v1",
        "pnl_bps",
        "mae_bps",
        "mfe_bps",
        "exit_reason",
        "early_exit_regret",
    ]
    join_export_present = [c for c in join_export_cols if c in join.columns]
    join.loc[:, join_export_present].to_csv(join_table_path, index=False)

    reward_table, distributions = _compute_reward_variants(join)
    reward_class_audit = _reward_class_audit(v2_rows)
    reward_contract_v2 = _build_reward_variants_contract(distributions, join_audit)
    _write_json(
        reward_dir / "iql_entry_iql_reward_variants_contract_v2.json",
        reward_contract_v2,
    )
    _write_json(reward_dir / "reward_variant_class_audit_v1.json", reward_class_audit)
    distribution_rows = [
        {"reward_id_v1": rid, **dist} for rid, dist in distributions.items()
    ]
    _write_rows(
        reward_dir / "reward_variants_dry_run_distribution_v1.csv", distribution_rows
    )

    # Timing audit
    timing_table, timing_summary, timing_recommendation = _build_timing_audit(join)
    _write_json(
        timing_dir / "entry_timing_audit_recommendation_v1.json",
        timing_recommendation,
    )
    if timing_summary["timing_status_v1"] == "TIMING_AUDIT_AVAILABLE":
        timing_table.to_csv(timing_dir / "entry_timing_audit_dataset_v1.csv", index=False)
    else:
        _write_json(
            timing_dir / "entry_timing_audit_NOT_ESTABLISHED_v1.json",
            {
                "layer_name": "ENTRY_TIMING_AUDIT_NOT_ESTABLISHED_V1",
                "reason_v1": "no shielded matched rows available for Alt A audit",
                "research_only_v1": True,
            },
        )

    # Reproducibility + go/no-go
    reproducibility = _reproducibility_audit(
        frame, masks, join_audit, discovery["summary_v1"]
    )
    _write_json(artifact_root / "reproducibility_audit_v1.json", reproducibility)
    status, next_action, recommendation = _go_no_go(
        discovery["summary_v1"], join_audit, timing_summary
    )

    summary = {
        "layer_name": "REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "state_v2_field_count_v1": len(v2_rows),
        "state_v2_added_count_v1": discovery["summary_v1"]["accepted_count_v1"],
        "state_v1_field_count_v1": len(_v1_state_rows()),
        "state_family_status_v1": discovery["summary_v1"]["family_status_v1"],
        "reward_variant_count_v1": len(REWARD_VARIANT_SPECS),
        "reward_join_status_v1": join_audit["join_status_v1"],
        "reward_take_match_rate_v1": join_audit["take_trade_match_rate_v1"],
        "timing_status_v1": timing_summary["timing_status_v1"],
        "row_count_invariant_v1": reproducibility["row_count_invariant_v1"],
        "seventy_eight_shield_invariant_v1": reproducibility[
            "seventy_eight_shield_invariant_v1"
        ],
        "research_only_v1": True,
        "iql_training_run_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "forbidden_actions_audit_v1": forbidden_audit,
        "next_gate_hook_v1": "RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1",
    }
    _write_json(artifact_root / "summary_v1.json", summary)

    status_payload = {
        "layer_name": "REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": False,
        "v2_state_field_count_v1": len(v2_rows),
        "reward_variant_count_v1": len(REWARD_VARIANT_SPECS),
        "timing_status_v1": timing_summary["timing_status_v1"],
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_GO_NO_GO_V1",
        "status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "research_only_v1": True,
        "downstream_block_v1": (
            "This gate does not open adapter, R6, IQL production/live, full lifecycle sequential "
            "IQL, policy promotion, package, freeze, promo, or live. The next allowed work is "
            "research-only contextual IQL training with the V2 state contract and reward variants."
        ),
    }
    _write_json(
        artifact_root / "rebuild_iql_state_contract_with_more_as_of_features_go_no_go_v1.json",
        go_no_go,
    )

    manifest = {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "append_only_namespace_v1": "truth_e2e_sanity",
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "go_no_go": str(
                artifact_root
                / "rebuild_iql_state_contract_with_more_as_of_features_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "reproducibility_audit": str(artifact_root / "reproducibility_audit_v1.json"),
            "state_contract_v2": str(state_dir / "iql_offline_state_contract_v2.json"),
            "state_diff_csv": str(
                state_dir / "iql_offline_state_contract_v2_diff_vs_v1.csv"
            ),
            "state_family_inventory": str(state_dir / "state_family_inventory_v1.json"),
            "state_family_metrics_csv": str(
                state_dir / "state_family_inventory_dry_run_metrics_v1.csv"
            ),
            "state_no_shortcut_audit": str(state_dir / "state_no_shortcut_audit_v2.json"),
            "state_normalization_plan": str(
                state_dir / "state_normalization_plan_v1.json"
            ),
            "reward_variants_contract": str(
                reward_dir / "iql_entry_iql_reward_variants_contract_v2.json"
            ),
            "reward_variants_distribution": str(
                reward_dir / "reward_variants_dry_run_distribution_v1.csv"
            ),
            "reward_join_audit": str(
                reward_dir / "entry_iql_post_trade_outcome_join_audit_v1.json"
            ),
            "reward_join_table": str(
                reward_dir / "entry_iql_post_trade_outcome_join_table_v1.csv"
            ),
            "reward_class_audit": str(reward_dir / "reward_variant_class_audit_v1.json"),
            "timing_recommendation": str(
                timing_dir / "entry_timing_audit_recommendation_v1.json"
            ),
        },
        "read_only_references_v1": True,
        "not_trainer_v1": True,
        "not_controller_v1": True,
        "not_live_gate_v1": True,
    }
    _write_json(artifact_root / "manifest_v1.json", manifest)

    report_lines = [
        "# Rebuild IQL State Contract With More AS_OF Features V1",
        "",
        "## Final status",
        "",
        f"- `{status}`",
        f"- Next action: `{next_action}`",
        "",
        "## State expansion V2",
        "",
        f"- V1 baseline field rows: {len(_v1_state_rows())}",
        f"- V2 total field rows: {len(v2_rows)}",
        f"- New AS_OF allowed fields: {discovery['summary_v1']['accepted_count_v1']}",
        f"- Family status: `{discovery['summary_v1']['family_status_v1']}`",
        f"- Accepted: `{discovery['summary_v1']['accepted_field_names_v1']}`",
        "",
        "## Reward variants V2",
        "",
        f"- Variant count: {len(REWARD_VARIANT_SPECS)} (research-only)",
        f"- Trade outcomes overall match rate: {join_audit['overall_match_rate_v1']:.4f}",
        f"- 78-shield TAKE match rate: {join_audit['take_trade_match_rate_v1']:.4f}",
        f"- Reward join status: `{join_audit['join_status_v1']}`",
        "",
        "## Timing audit V1",
        "",
        f"- Approach: `{timing_summary['approach_v1']}`",
        f"- Shielded matched rows: {timing_summary['shielded_matched_row_count_v1']}",
        f"- MAE-dominated: {timing_summary['shielded_mae_dominated_count_v1']}",
        f"- Peak giveback: {timing_summary['shielded_peak_giveback_count_v1']}",
        f"- Catastrophic exit: {timing_summary['shielded_cata_exit_count_v1']}",
        f"- Status: `{timing_summary['timing_status_v1']}`",
        "",
        "## Recommendation",
        "",
        recommendation,
    ]
    _write_report(artifact_root / "report_v1.md", report_lines)

    return {
        "artifact_root": str(artifact_root),
        "summary": summary,
        "status": status_payload,
        "go_no_go": go_no_go,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_V1 gate."
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
