from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "MONDAY_R6_CANONICAL_SCORE_AND_LABEL_RESTORE_OR_REBUILD_V1"

WEDNESDAY_SNAPSHOT_DIR = "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_20260424T0900Z"
WEDNESDAY_FREEZE_DIR = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1"
WEDNESDAY_SUMMARY = "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_summary_v1.json"
WEDNESDAY_MANIFEST = "shadow_meta_all_trade_review_r6_shadow_freeze_manifest_v1.json"

MONDAY_TRUTH_GLOB = "MONDAY_R6_CANONICAL_TRUTH_V1_*"
REHYDRATED_GLOB = "MONDAY_R6_REHYDRATED_WEDNESDAY_CONTRACT_V1_*"
LOCAL_R6_DIR_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"
LOCAL_R5_2_FREEZE_DIR_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_2_SHADOW_FREEZE_AND_R6_FAILURE_BACKLOG_V1"

R6_AS_OF_TABLE = "shadow_meta_all_trade_review_r6_entry_runner_first_as_of_feature_table_v1.parquet"
R6_HINDSIGHT_TABLE = "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet"
R6_POLICY_VIEW = "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet"
R6_CONTRACT = "shadow_meta_all_trade_review_r6_entry_runner_first_contract_v1.json"
R6_SUMMARY = "shadow_meta_all_trade_review_r6_summary_v1.json"
R6_MODELS_DIR = "models"

R5_2_FREEZE_SUMMARY = "shadow_meta_all_trade_review_r5_2_shadow_freeze_and_r6_failure_backlog_summary_v1.json"
R5_2_FREEZE_MANIFEST = "shadow_meta_all_trade_review_r5_2_shadow_freeze_manifest_v1.json"

REHYDRATED_SUMMARY = "summary_v1.json"
REHYDRATED_BLOCKED_FIELDS = "monday_r6_rehydration_blocked_fields_v1.csv"

OUTPUT_FILES = {
    "summary": "summary_v1.json",
    "canonical_source_artifact_availability": "canonical_source_artifact_availability_v1.csv",
    "canonical_hash_scan": "canonical_hash_scan_v1.csv",
    "local_noncanonical_source_rejection": "local_noncanonical_source_rejection_v1.csv",
    "score_rebuild_plan": "score_rebuild_plan_v1.json",
    "exact_label_rebuild_plan": "exact_label_rebuild_plan_v1.json",
    "monday_rehydrated_patch_status": "monday_rehydrated_patch_status_v1.json",
    "manifest": "manifest_v1.json",
    "audit": "consistency_audit_v1.csv",
    "report": "report_v1.md",
}

REQUIRED_CANONICAL_SOURCE_ARTIFACTS = [
    R6_CONTRACT,
    R6_AS_OF_TABLE,
    R6_HINDSIGHT_TABLE,
    R6_POLICY_VIEW,
    "shadow_meta_all_trade_review_r6_model_family_bakeoff_v1.csv",
    "shadow_meta_all_trade_review_r6_loso_metrics_v1.csv",
    "shadow_meta_all_trade_review_r6_head_to_head_vs_r2_r4_r5_r5_1_r5_2_v1.csv",
    R6_MODELS_DIR,
]

SCORE_SOURCE_COLUMNS = {
    "pred__entry_r6_bad_risk__prob_true_v1",
    "pred__entry_r6_runner_protector__prob_true_v1",
    "pred__entry_r6_tail_control_10_50__prob_true_v1",
    "pred__entry_r6_risky_allow__prob_true_v1",
    "pred__entry_r6_batch04_blindspot__prob_true_v1",
    "pred__entry_r5_2_bad_blocker__prob_true_v1",
    "pred__entry_r5_2_runner_protector__prob_true_v1",
    "blocker_score_v1",
    "runner_protector_score_v1",
}

EXACT_LABEL_COLUMNS = {
    "r6_label_runner_near_miss_v1",
    "hindsight_entry_decision_review_v1",
    "hindsight_management_review_v1",
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _latest_dir(reports_root: Path, pattern: str) -> Path | None:
    matches = sorted(path for path in reports_root.glob(pattern) if path.is_dir())
    return matches[-1] if matches else None


def _safe_len_parquet(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(len(pd.read_parquet(path)))
    except Exception:
        return None


def _safe_columns_parquet(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return [str(column) for column in pd.read_parquet(path).columns]
    except Exception:
        return []


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hashes = manifest.get("hashes_v1") or {}
    for group_name, group_rows in hashes.items():
        if not isinstance(group_rows, list):
            continue
        for row in group_rows:
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "source_kind_v1": "HASHED_CANONICAL_MODEL_ARTIFACT",
                    "hash_group_v1": str(group_name),
                    "artifact_name_v1": str(row.get("relative_path_v1") or row.get("absolute_path_v1") or ""),
                    "relative_path_v1": str(row.get("relative_path_v1") or ""),
                    "expected_absolute_path_v1": str(row.get("absolute_path_v1") or ""),
                    "expected_sha256_v1": str(row.get("sha256_v1") or ""),
                    "hash_kind_v1": str(row.get("hash_kind_v1") or ""),
                    "expected_byte_size_v1": row.get("byte_size_v1"),
                }
            )
    return rows


def _default_hash_scan_root(reports_root: Path) -> Path:
    if reports_root.name == "truth_e2e_sanity" and reports_root.parent.name == "reports":
        return reports_root.parent.parent
    return reports_root


def _canonical_hash_scan_rows(manifest: dict[str, Any], scan_root: Path) -> pd.DataFrame:
    expected_rows = _hash_rows(manifest)
    if not expected_rows:
        return pd.DataFrame(
            columns=[
                "expected_sha256_v1",
                "relative_path_v1",
                "expected_absolute_path_v1",
                "hash_kind_v1",
                "expected_byte_size_v1",
                "scan_root_v1",
                "candidate_file_count_v1",
                "matched_path_v1",
                "match_found_v1",
                "status_v1",
            ]
        )
    expected_by_sha = {str(row["expected_sha256_v1"]): row for row in expected_rows if row.get("expected_sha256_v1")}
    target_names = {Path(str(row["relative_path_v1"] or row["artifact_name_v1"])).name for row in expected_rows}
    candidate_paths = sorted(path for path in scan_root.rglob("*") if path.is_file() and path.name in target_names)
    matches: dict[str, list[str]] = {sha: [] for sha in expected_by_sha}
    for path in candidate_paths:
        observed_sha = _sha256(path)
        if observed_sha in matches:
            matches[observed_sha].append(str(path))

    rows: list[dict[str, Any]] = []
    for expected in expected_rows:
        sha = str(expected.get("expected_sha256_v1") or "")
        paths = matches.get(sha, [])
        rows.append(
            {
                "expected_sha256_v1": sha,
                "relative_path_v1": expected.get("relative_path_v1"),
                "expected_absolute_path_v1": expected.get("expected_absolute_path_v1"),
                "hash_kind_v1": expected.get("hash_kind_v1"),
                "expected_byte_size_v1": expected.get("expected_byte_size_v1"),
                "scan_root_v1": str(scan_root),
                "candidate_file_count_v1": int(len(candidate_paths)),
                "matched_path_v1": paths[0] if paths else "",
                "match_found_v1": bool(paths),
                "status_v1": "FOUND_BY_HASH_SCAN" if paths else "NOT_FOUND_BY_HASH_SCAN",
            }
        )
    return pd.DataFrame(rows)


def _canonical_artifact_rows(manifest: dict[str, Any]) -> pd.DataFrame:
    source_dir = Path(str(manifest.get("r6_source_dir_v1") or "")).expanduser()
    source_dir_present = bool(source_dir.exists() and source_dir.is_dir())
    rows: list[dict[str, Any]] = []
    for artifact in REQUIRED_CANONICAL_SOURCE_ARTIFACTS:
        path = source_dir / artifact
        exists = bool(path.exists())
        rows.append(
            {
                "source_kind_v1": "REQUIRED_CANONICAL_R6_SOURCE_ARTIFACT",
                "artifact_name_v1": artifact,
                "relative_path_v1": artifact,
                "expected_absolute_path_v1": str(path),
                "observed_absolute_path_v1": str(path) if exists else "",
                "exists_v1": exists,
                "expected_sha256_v1": "",
                "observed_sha256_v1": "",
                "hash_match_v1": pd.NA,
                "status_v1": "PRESENT" if exists else ("SOURCE_TREE_MISSING" if not source_dir_present else "MISSING"),
                "action_v1": "USE_AS_CANONICAL_SOURCE" if exists else "RESTORE_WEDNESDAY_SOURCE_ARTIFACT",
            }
        )
    for row in _hash_rows(manifest):
        expected_abs = Path(str(row["expected_absolute_path_v1"])).expanduser()
        observed_sha = _sha256(expected_abs)
        exists = observed_sha is not None
        expected_sha = str(row["expected_sha256_v1"])
        hash_match = bool(exists and expected_sha and observed_sha == expected_sha)
        rows.append(
            {
                **row,
                "observed_absolute_path_v1": str(expected_abs) if exists else "",
                "exists_v1": exists,
                "observed_sha256_v1": observed_sha or "",
                "hash_match_v1": hash_match if exists else pd.NA,
                "status_v1": "PRESENT_HASH_MATCH"
                if hash_match
                else ("PRESENT_HASH_MISMATCH" if exists else ("SOURCE_TREE_MISSING" if not source_dir_present else "MISSING")),
                "action_v1": "USE_AS_CANONICAL_SOURCE" if hash_match else "RESTORE_EXACT_HASHED_CANONICAL_ARTIFACT",
            }
        )
    return pd.DataFrame(rows)


def _selected_candidate(summary: dict[str, Any]) -> str | None:
    selected = summary.get("selected_candidate_v1") or {}
    for key in ["selected_policy_name_v1", "policy_name_v1"]:
        if selected.get(key):
            return str(selected[key])
    decision = summary.get("selected_candidate_id_v1") or summary.get("candidate_id_v1")
    return str(decision) if decision else None


def _local_r6_rejection(reports_root: Path, expected_candidate: str | None, expected_rows: int | None, expected_asof_cols: int | None) -> dict[str, Any]:
    local_dir = reports_root / LOCAL_R6_DIR_NAME
    summary = _read_json_if_exists(local_dir / R6_SUMMARY)
    contract = _read_json_if_exists(local_dir / R6_CONTRACT)
    asof_path = local_dir / R6_AS_OF_TABLE
    hindsight_path = local_dir / R6_HINDSIGHT_TABLE
    policy_path = local_dir / R6_POLICY_VIEW
    asof_cols = _safe_columns_parquet(asof_path)
    observed_candidate = _selected_candidate(summary)
    freeze_benchmark = contract.get("freeze_benchmark_v1") or {}
    reasons: list[str] = []
    if not local_dir.exists():
        reasons.append("LOCAL_R6_DIR_MISSING")
    if observed_candidate and expected_candidate and observed_candidate != expected_candidate:
        reasons.append("CANDIDATE_ID_MISMATCH")
    if expected_rows is not None:
        rows = _safe_len_parquet(asof_path)
        if rows is not None and rows != expected_rows:
            reasons.append("ROW_COUNT_MISMATCH")
    if expected_asof_cols is not None and asof_cols and len(asof_cols) != expected_asof_cols:
        reasons.append("AS_OF_SCHEMA_MISMATCH")
    local_freeze = freeze_benchmark.get("freeze_id_v1")
    if local_freeze:
        reasons.append("LOCAL_R5_2_FREEZE_LINEAGE_NOT_CANONICAL_WEDNESDAY")
    assessment = "MISSING" if not local_dir.exists() else ("REJECTED_NONCANONICAL" if reasons else "NO_REJECTION_REASON_FOUND")
    return {
        "source_name_v1": "local_1852_r6_entry_runner_first_retrain",
        "source_dir_v1": str(local_dir),
        "exists_v1": bool(local_dir.exists()),
        "assessment_v1": assessment,
        "rejection_reasons_v1": reasons,
        "observed_candidate_v1": observed_candidate,
        "expected_candidate_v1": expected_candidate,
        "observed_rows_v1": _safe_len_parquet(asof_path),
        "expected_rows_v1": expected_rows,
        "observed_asof_columns_v1": len(asof_cols) if asof_cols else None,
        "expected_asof_columns_v1": expected_asof_cols,
        "observed_hindsight_rows_v1": _safe_len_parquet(hindsight_path),
        "observed_policy_rows_v1": _safe_len_parquet(policy_path),
        "observed_freeze_benchmark_id_v1": local_freeze,
        "must_not_use_as_score_source_v1": True,
    }


def _local_r5_2_rejection(reports_root: Path, expected_freeze_id: str | None) -> dict[str, Any]:
    local_dir = reports_root / LOCAL_R5_2_FREEZE_DIR_NAME
    summary = _read_json_if_exists(local_dir / R5_2_FREEZE_SUMMARY)
    manifest = _read_json_if_exists(local_dir / R5_2_FREEZE_MANIFEST)
    observed_freeze = manifest.get("freeze_id_v1") or summary.get("freeze_id_v1")
    reasons: list[str] = []
    if not local_dir.exists():
        reasons.append("LOCAL_R5_2_FREEZE_DIR_MISSING")
    if observed_freeze and expected_freeze_id and observed_freeze != expected_freeze_id:
        reasons.append("FREEZE_ID_MISMATCH")
    assessment = "MISSING" if not local_dir.exists() else ("REJECTED_NONCANONICAL" if reasons else "NO_REJECTION_REASON_FOUND")
    return {
        "source_name_v1": "local_r5_2_shadow_freeze_and_r6_failure_backlog",
        "source_dir_v1": str(local_dir),
        "exists_v1": bool(local_dir.exists()),
        "assessment_v1": assessment,
        "rejection_reasons_v1": reasons,
        "observed_freeze_id_v1": observed_freeze,
        "expected_freeze_id_v1": expected_freeze_id,
        "observed_policy_logging_rows_v1": (summary.get("artifact_counts_v1") or {}).get("policy_logging_lock_rows_v1"),
        "must_not_use_as_score_source_v1": True,
    }


def _rejection_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "source_name_v1",
                "source_dir_v1",
                "exists_v1",
                "assessment_v1",
                "rejection_reasons_v1",
                "must_not_use_as_score_source_v1",
            ]
        )
    out["rejection_reasons_json_v1"] = out["rejection_reasons_v1"].map(lambda value: json.dumps(_jsonable(value), sort_keys=True))
    return out.drop(columns=["rejection_reasons_v1"])


def _blocked_counts(rehydrated_dir: Path | None) -> dict[str, Any]:
    if rehydrated_dir is None:
        return {
            "rehydrated_dir_present_v1": False,
            "blocked_score_column_count_v1": None,
            "hindsight_proxy_column_count_v1": None,
            "blocked_fields_rows_v1": None,
            "score_fields_still_blocked_v1": [],
            "exact_label_fields_still_proxy_or_blocked_v1": [],
        }
    summary = _read_json_if_exists(rehydrated_dir / REHYDRATED_SUMMARY)
    blocked_path = rehydrated_dir / REHYDRATED_BLOCKED_FIELDS
    score_blocked: list[str] = []
    label_blocked: list[str] = []
    blocked_rows = None
    if blocked_path.exists():
        blocked = pd.read_csv(blocked_path)
        blocked_rows = int(len(blocked))
        if "field_v1" in blocked.columns:
            fields = blocked["field_v1"].astype("string")
            score_mask = (
                fields.isin(SCORE_SOURCE_COLUMNS)
                | fields.str.startswith("pred__entry_r5_", na=False)
                | fields.str.startswith("pred__entry_r6_", na=False)
                | fields.isin(["blocker_score_v1", "runner_protector_score_v1"])
            )
            score_blocked = sorted(set(fields[score_mask].dropna().astype(str).tolist()))
            label_blocked = sorted(set(fields[fields.isin(EXACT_LABEL_COLUMNS)].dropna().astype(str).tolist()))
    return {
        "rehydrated_dir_present_v1": True,
        "blocked_score_column_count_v1": summary.get("blocked_score_column_count_v1"),
        "hindsight_proxy_column_count_v1": summary.get("hindsight_proxy_column_count_v1"),
        "blocked_fields_rows_v1": blocked_rows,
        "score_fields_still_blocked_v1": score_blocked,
        "exact_label_fields_still_proxy_or_blocked_v1": label_blocked,
    }


def _score_rebuild_plan(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "MONDAY_R6_CANONICAL_SCORE_RESTORE_OR_REBUILD_PLAN_V1",
        "training_started_v1": False,
        "restore_path_v1": [
            "Restore the missing Wednesday canonical R6 source tree at r6_source_dir_v1.",
            "Verify all model/preprocessor/metadata SHA256 hashes from the Wednesday R6 freeze manifest.",
            "Run the Monday rehydrated 109-column AS_OF table through the canonical R6 five-head family.",
            "Only then patch score columns into a new Monday scored contract package.",
        ],
        "rebuild_path_v1": [
            "Use the Monday rehydrated Wednesday contract only after exact labels and canonical R5.2 reference scores are restored.",
            "Run R6 model training with an explicit training flag and deterministic namespace.",
            "Compare against the frozen Wednesday R6 metrics before any candidate is considered usable.",
        ],
        "blocked_until_v1": [
            "canonical R6 score model artifacts are hash-verified or explicitly rebuilt",
            "canonical R5.2 benchmark freeze/source is restored or explicitly rebuilt",
            "exact R6 hindsight label source is no longer proxy/null for required fields",
        ],
        "hash_scan_result_v1": {
            "scan_root_v1": summary.get("canonical_hash_scan_root_v1"),
            "candidate_file_count_v1": summary.get("canonical_hash_scan_candidate_file_count_v1"),
            "matched_hash_count_v1": summary.get("canonical_hash_scan_match_count_v1"),
            "missing_hash_count_v1": summary.get("canonical_hash_scan_missing_count_v1"),
        },
        "must_not_use_v1": [
            "local 1852-row R6 candidate 04789 as canonical score source",
            "local R5.2 freeze ADBB99533B5FC91B as canonical Wednesday R5.2 10176 source",
            "1689 exact-only/protector-first surfaces as R6 baseline",
        ],
        "next_command_when_sources_restored_v1": (
            "python3 -m gx1.scripts.materialize_monday_r6_restore_or_rebuild_canonical_score_and_label_sources_v1 "
            "--reports-root /home/andre2/GX1_DATA/reports/truth_e2e_sanity"
        ),
        "current_decision_v1": summary["decision_v1"],
    }


def _exact_label_rebuild_plan(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "MONDAY_R6_EXACT_LABEL_RESTORE_OR_REBUILD_PLAN_V1",
        "training_started_v1": False,
        "exact_label_contract_source_v1": "train_r6_entry_runner_first_retrain_v1::_prepare_frame label construction",
        "required_inputs_v1": [
            "Wednesday-contract AS_OF table",
            "Wednesday-contract HINDSIGHT table",
            "canonical R5.2 policy logging lock",
            "canonical R5.2 policy prediction view",
            "batch/week lookup for LOSO and batch04/batch05 scope",
        ],
        "currently_proxy_or_unestablished_v1": summary.get("exact_label_fields_still_proxy_or_blocked_v1", []),
        "restore_path_v1": [
            "Restore the canonical Wednesday R6 source hindsight table to confirm exact label semantics.",
            "Restore or rebuild canonical R5.2 prediction/policy-lock sources before deriving runner-near-miss and risky-allow labels.",
            "Re-materialize Monday labels from the same R6 label function, not from the 1689 narrow diagnostic surface.",
        ],
        "blocked_until_v1": [
            "exact R5.2 score/reference inputs exist for Monday rows",
            "hindsight decision review fields are exact, not placeholder/proxy",
        ],
        "current_decision_v1": summary["decision_v1"],
    }


def _patch_status(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "MONDAY_R6_REHYDRATED_PATCH_STATUS_V1",
        "patched_v1": False,
        "training_started_v1": False,
        "scores_written_v1": False,
        "labels_promoted_to_exact_v1": False,
        "reason_v1": "Canonical score and exact-label sources did not pass restore/rebuild gate.",
        "rehydrated_dir_v1": summary.get("rehydrated_dir_v1"),
        "blocked_action_v1": summary.get("blocked_action_v1"),
    }


def _audit(summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    canonical_tree_ok = bool(summary["canonical_source_tree_present_v1"])
    hashes_ok = bool(summary["canonical_hash_rows_v1"] > 0 and summary["canonical_hash_mismatch_or_missing_count_v1"] == 0)
    hash_scan_ok = bool(summary["canonical_hash_scan_match_count_v1"] == summary["canonical_hash_rows_v1"] and summary["canonical_hash_rows_v1"] > 0)
    exact_labels_ok = bool(summary["exact_label_sources_restored_v1"])
    scores_ok = bool(summary["score_sources_restored_v1"])
    local_rejected = bool(summary["local_noncanonical_rejection_count_v1"] >= 1)
    return pd.DataFrame(
        [
            row("CANONICAL_R6_SOURCE_TREE_PRESENT", "PASS" if canonical_tree_ok else "FAIL", summary["canonical_r6_source_dir_v1"]),
            row(
                "CANONICAL_R6_HASHES_MATCH",
                "PASS" if hashes_ok else "FAIL",
                {
                    "hash_rows": summary["canonical_hash_rows_v1"],
                    "match": summary["canonical_hash_match_count_v1"],
                    "mismatch_or_missing": summary["canonical_hash_mismatch_or_missing_count_v1"],
                },
            ),
            row(
                "CANONICAL_R6_HASH_SCAN_FOUND_ALL",
                "PASS" if hash_scan_ok else "FAIL",
                {
                    "scan_root": summary["canonical_hash_scan_root_v1"],
                    "candidate_files": summary["canonical_hash_scan_candidate_file_count_v1"],
                    "matched": summary["canonical_hash_scan_match_count_v1"],
                    "missing": summary["canonical_hash_scan_missing_count_v1"],
                },
            ),
            row("EXPECTED_R5_2_FREEZE_PRESENT", "PASS" if summary["expected_r5_2_freeze_found_v1"] else "FAIL", summary["expected_r5_2_freeze_id_v1"]),
            row("LOCAL_NONCANONICAL_SOURCES_REJECTED", "PASS" if local_rejected else "FAIL", summary["local_noncanonical_rejection_count_v1"]),
            row("REHYDRATED_CONTRACT_PRESENT", "PASS" if summary["rehydrated_dir_present_v1"] else "FAIL", summary.get("rehydrated_dir_v1")),
            row("CANONICAL_SCORE_SOURCES_RESTORED", "PASS" if scores_ok else "FAIL", summary["score_fields_still_blocked_v1"]),
            row("EXACT_LABEL_SOURCES_RESTORED", "PASS" if exact_labels_ok else "FAIL", summary["exact_label_fields_still_proxy_or_blocked_v1"]),
            row("NO_TRAINING_STARTED", "PASS", summary["training_started_v1"]),
            row("NO_NONCANONICAL_SCORE_FILL", "PASS", summary["noncanonical_scores_used_v1"]),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Monday R6 Canonical Score And Label Restore/Rebuild V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Canonical R6 source tree present: `{summary['canonical_source_tree_present_v1']}`",
            f"- Canonical hash rows: `{summary['canonical_hash_rows_v1']}`",
            f"- Canonical hash mismatches/missing: `{summary['canonical_hash_mismatch_or_missing_count_v1']}`",
            f"- Canonical hash scan root: `{summary['canonical_hash_scan_root_v1']}`",
            f"- Canonical hash scan matches: `{summary['canonical_hash_scan_match_count_v1']}/{summary['canonical_hash_rows_v1']}`",
            f"- Expected R5.2 freeze found: `{summary['expected_r5_2_freeze_found_v1']}`",
            f"- Rehydrated contract present: `{summary['rehydrated_dir_present_v1']}`",
            f"- Score sources restored: `{summary['score_sources_restored_v1']}`",
            f"- Exact label sources restored: `{summary['exact_label_sources_restored_v1']}`",
            f"- Local noncanonical rejection count: `{summary['local_noncanonical_rejection_count_v1']}`",
            f"- Training started: `{summary['training_started_v1']}`",
            "",
            "The local 1852-row R6/R5.2 line is diagnostic only and is not used to fill Monday R6 scores or labels.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    monday_truth_dir: Path | None = None,
    rehydrated_dir: Path | None = None,
    hash_scan_root: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    monday_truth_dir = monday_truth_dir.expanduser().resolve() if monday_truth_dir else _latest_dir(reports_root, MONDAY_TRUTH_GLOB)
    rehydrated_dir = rehydrated_dir.expanduser().resolve() if rehydrated_dir else _latest_dir(reports_root, REHYDRATED_GLOB)
    hash_scan_root = hash_scan_root.expanduser().resolve() if hash_scan_root else _default_hash_scan_root(reports_root)
    output_dir = output_dir.expanduser().resolve() if output_dir else reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    freeze_dir = reports_root / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    wednesday_summary = _read_json(freeze_dir / WEDNESDAY_SUMMARY)
    wednesday_manifest = _read_json(freeze_dir / WEDNESDAY_MANIFEST)
    expected_rows = (wednesday_summary.get("policy_logging_v1") or {}).get("row_count_v1")
    expected_asof_cols = (wednesday_manifest.get("as_of_schema_v1") or {}).get("column_count_v1")
    expected_candidate = wednesday_summary.get("selected_candidate_id_v1")
    expected_r5_2_freeze = wednesday_manifest.get("r5_2_benchmark_freeze_id_v1")

    artifact_df = _canonical_artifact_rows(wednesday_manifest)
    hash_scan_df = _canonical_hash_scan_rows(wednesday_manifest, hash_scan_root)
    local_rejection_rows = [
        _local_r6_rejection(reports_root, str(expected_candidate) if expected_candidate else None, expected_rows, expected_asof_cols),
        _local_r5_2_rejection(reports_root, str(expected_r5_2_freeze) if expected_r5_2_freeze else None),
    ]
    rejection_df = _rejection_df(local_rejection_rows)
    blocked = _blocked_counts(rehydrated_dir)

    required_artifacts = artifact_df[artifact_df["source_kind_v1"].eq("REQUIRED_CANONICAL_R6_SOURCE_ARTIFACT")]
    hash_artifacts = artifact_df[artifact_df["source_kind_v1"].eq("HASHED_CANONICAL_MODEL_ARTIFACT")]
    source_tree_present = bool(Path(str(wednesday_manifest.get("r6_source_dir_v1") or "")).expanduser().is_dir())
    hash_match_count = int(hash_artifacts["status_v1"].eq("PRESENT_HASH_MATCH").sum()) if not hash_artifacts.empty else 0
    hash_bad_count = int(len(hash_artifacts) - hash_match_count)
    hash_scan_match_count = int(hash_scan_df["match_found_v1"].fillna(False).astype(bool).sum()) if not hash_scan_df.empty else 0
    hash_scan_missing_count = int(len(hash_scan_df) - hash_scan_match_count)
    hash_scan_candidate_file_count = int(hash_scan_df["candidate_file_count_v1"].max()) if not hash_scan_df.empty else 0
    required_missing_count = int((~required_artifacts["exists_v1"].fillna(False).astype(bool)).sum()) if not required_artifacts.empty else 0
    expected_r5_found = bool(
        rejection_df["source_name_v1"].eq("local_r5_2_shadow_freeze_and_r6_failure_backlog").any()
        and rejection_df.set_index("source_name_v1").loc["local_r5_2_shadow_freeze_and_r6_failure_backlog", "assessment_v1"] == "NO_REJECTION_REASON_FOUND"
    )
    score_sources_restored = bool(source_tree_present and hash_bad_count == 0 and len(hash_artifacts) > 0 and not blocked["score_fields_still_blocked_v1"])
    exact_label_sources_restored = bool(source_tree_present and required_missing_count == 0 and not blocked["exact_label_fields_still_proxy_or_blocked_v1"])

    if score_sources_restored and exact_label_sources_restored:
        decision = "CANONICAL_SCORE_AND_EXACT_LABEL_SOURCES_RESTORED"
        next_action = "SCORE_MONDAY_REHYDRATED_CONTRACT_WITH_CANONICAL_R6_MODELS"
    else:
        decision = "CANONICAL_SCORE_AND_EXACT_LABEL_SOURCES_NOT_RESTORED"
        next_action = "RESTORE_WEDNESDAY_SOURCE_ARTIFACTS_FIRST"

    rejected_count = int(rejection_df["assessment_v1"].astype("string").eq("REJECTED_NONCANONICAL").sum()) if not rejection_df.empty else 0
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "reports_root_v1": str(reports_root),
        "monday_truth_dir_v1": str(monday_truth_dir) if monday_truth_dir else None,
        "rehydrated_dir_v1": str(rehydrated_dir) if rehydrated_dir else None,
        "rehydrated_dir_present_v1": bool(blocked["rehydrated_dir_present_v1"]),
        "wednesday_freeze_id_v1": wednesday_summary.get("freeze_id_v1"),
        "wednesday_candidate_id_v1": expected_candidate,
        "wednesday_model_version_id_v1": wednesday_manifest.get("model_version_id_v1"),
        "expected_r5_2_freeze_id_v1": expected_r5_2_freeze,
        "canonical_r6_source_dir_v1": wednesday_manifest.get("r6_source_dir_v1"),
        "canonical_source_tree_present_v1": source_tree_present,
        "canonical_required_artifact_missing_count_v1": required_missing_count,
        "canonical_hash_rows_v1": int(len(hash_artifacts)),
        "canonical_hash_match_count_v1": hash_match_count,
        "canonical_hash_mismatch_or_missing_count_v1": hash_bad_count,
        "canonical_hash_scan_root_v1": str(hash_scan_root),
        "canonical_hash_scan_candidate_file_count_v1": hash_scan_candidate_file_count,
        "canonical_hash_scan_match_count_v1": hash_scan_match_count,
        "canonical_hash_scan_missing_count_v1": hash_scan_missing_count,
        "expected_r5_2_freeze_found_v1": expected_r5_found,
        "score_sources_restored_v1": score_sources_restored,
        "exact_label_sources_restored_v1": exact_label_sources_restored,
        "blocked_score_column_count_v1": blocked["blocked_score_column_count_v1"],
        "hindsight_proxy_column_count_v1": blocked["hindsight_proxy_column_count_v1"],
        "score_fields_still_blocked_v1": blocked["score_fields_still_blocked_v1"],
        "exact_label_fields_still_proxy_or_blocked_v1": blocked["exact_label_fields_still_proxy_or_blocked_v1"],
        "local_noncanonical_rejection_count_v1": rejected_count,
        "local_noncanonical_sources_considered_v1": int(len(rejection_df)),
        "noncanonical_scores_used_v1": False,
        "training_started_v1": False,
        "decision_v1": decision,
        "next_action_v1": next_action,
        "blocked_action_v1": [
            "DO_NOT_USE_LOCAL_1852_04789_R6_AS_CANONICAL_SCORE_SOURCE",
            "DO_NOT_USE_LOCAL_ADBB_R5_2_FREEZE_AS_CANONICAL_WEDNESDAY_R5_2_SOURCE",
            "DO_NOT_TRAIN_MONDAY_R6_UNTIL_SCORE_AND_EXACT_LABEL_SOURCE_DECISION_IS_GREEN",
            "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE",
        ],
        "hard_status_v1": {
            "BEVIST": [
                "The action did not train, freeze, promote, or patch score columns.",
                "The local 1852-row R6/R5.2 artifacts were assessed separately from the canonical Wednesday R6 lock.",
                "Noncanonical local score sources were not used.",
            ],
            "INDIKERT": [
                "Monday has a rehydrated Wednesday-contract shape package, but score and exact-label fields remain blocked.",
            ],
            "IKKE_ETABLERT": [
                "Canonical Wednesday R6 model/preprocessor source tree is not hash-verified locally.",
                "Canonical Wednesday R6 hashed artifacts were not found elsewhere under the configured hash scan root.",
                "Canonical R5.2 freeze/source id required by frozen Wednesday R6 is not restored locally.",
                "Monday exact labels are not established while required fields remain proxy/null.",
            ],
        },
    }
    audit = _audit(summary)
    score_plan = _score_rebuild_plan(summary)
    label_plan = _exact_label_rebuild_plan(summary)
    patch_status = _patch_status(summary)
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "artifacts_v1": OUTPUT_FILES,
        "not_live_gate_v1": True,
        "not_controller_v1": True,
        "training_started_v1": False,
        "noncanonical_scores_used_v1": False,
    }

    artifact_df.to_csv(output_dir / OUTPUT_FILES["canonical_source_artifact_availability"], index=False)
    hash_scan_df.to_csv(output_dir / OUTPUT_FILES["canonical_hash_scan"], index=False)
    rejection_df.to_csv(output_dir / OUTPUT_FILES["local_noncanonical_source_rejection"], index=False)
    audit.to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["score_rebuild_plan"], score_plan)
    _write_json(output_dir / OUTPUT_FILES["exact_label_rebuild_plan"], label_plan)
    _write_json(output_dir / OUTPUT_FILES["monday_rehydrated_patch_status"], patch_status)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--monday-truth-dir", type=Path, default=None)
    parser.add_argument("--rehydrated-dir", type=Path, default=None)
    parser.add_argument("--hash-scan-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        monday_truth_dir=args.monday_truth_dir,
        rehydrated_dir=args.rehydrated_dir,
        hash_scan_root=args.hash_scan_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
