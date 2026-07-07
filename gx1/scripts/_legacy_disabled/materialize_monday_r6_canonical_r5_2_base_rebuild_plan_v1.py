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
LAYER_NAME = "MONDAY_R6_CANONICAL_R5_2_BASE_REBUILD_PLAN_V1"

WEDNESDAY_SNAPSHOT_DIR = "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_20260424T0900Z"
WEDNESDAY_FREEZE_DIR = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1"
WEDNESDAY_SUMMARY = "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_summary_v1.json"
WEDNESDAY_MANIFEST = "shadow_meta_all_trade_review_r6_shadow_freeze_manifest_v1.json"

RESTORE_GLOB = "MONDAY_R6_CANONICAL_SCORE_AND_LABEL_RESTORE_OR_REBUILD_V1_*"
REHYDRATED_GLOB = "MONDAY_R6_REHYDRATED_WEDNESDAY_CONTRACT_V1_*"

R5_DIR_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_ENTRY_RETRAIN_WITH_REPAIRED_COVERAGE_AND_SLICE_ROBUSTNESS_V1"
R5_1_DIR_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_LOSO_BATCH04_ROBUSTNESS_RETRAIN_V1"
R5_2_DIR_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_2_ENTRY_RUNNER_AWARE_RETRAIN_AND_LOSO_SELECTION_V1"
R5_2_FREEZE_DIR_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_2_SHADOW_FREEZE_AND_R6_FAILURE_BACKLOG_V1"
R6_DIR_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"

R5_SUMMARY = "shadow_meta_all_trade_review_r5_entry_summary_v1.json"
R5_AS_OF = "shadow_meta_all_trade_review_r5_entry_as_of_feature_table_v1.parquet"
R5_1_SUMMARY = "shadow_meta_all_trade_review_r5_1_summary_v1.json"
R5_1_AS_OF = "shadow_meta_all_trade_review_r5_1_as_of_feature_table_v1.parquet"
R5_2_SUMMARY = "shadow_meta_all_trade_review_r5_2_summary_v1.json"
R5_2_AS_OF = "shadow_meta_all_trade_review_r5_2_as_of_feature_table_v1.parquet"
R5_2_POLICY = "shadow_meta_all_trade_review_r5_2_policy_prediction_view_v1.parquet"
R5_2_FREEZE_MANIFEST = "shadow_meta_all_trade_review_r5_2_shadow_freeze_manifest_v1.json"
R6_SUMMARY = "shadow_meta_all_trade_review_r6_summary_v1.json"
R6_AS_OF = "shadow_meta_all_trade_review_r6_entry_runner_first_as_of_feature_table_v1.parquet"

OUTPUT_FILES = {
    "summary": "summary_v1.json",
    "input_source_status": "input_source_status_v1.csv",
    "command_plan": "command_plan_v1.csv",
    "blocked_fields_to_resolve": "blocked_fields_to_resolve_v1.csv",
    "manifest": "manifest_v1.json",
    "audit": "consistency_audit_v1.csv",
    "report": "report_v1.md",
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
    if isinstance(value, float):
        return None if np.isnan(value) else value
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


def _latest_dir(reports_root: Path, pattern: str, required_file: str | None = None) -> Path | None:
    dirs = sorted(path for path in reports_root.glob(pattern) if path.is_dir())
    if required_file:
        dirs = [path for path in dirs if (path / required_file).exists()]
    return dirs[-1] if dirs else None


def _safe_parquet_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(len(pd.read_parquet(path)))
    except Exception:
        return None


def _coverage_rows(summary: dict[str, Any]) -> int | None:
    coverage = summary.get("coverage_v1")
    if isinstance(coverage, dict):
        for key in ["entry_coverage_v1", "ledger_trade_count_v1"]:
            if coverage.get(key) is not None:
                return int(coverage[key])
    return None


def _selected_policy(summary: dict[str, Any]) -> str | None:
    selected = summary.get("selected_candidate_v1") or {}
    if isinstance(selected, dict):
        return selected.get("selected_policy_name_v1") or selected.get("policy_name_v1")
    decision = summary.get("decision_v1")
    if isinstance(decision, dict):
        return decision.get("selected_policy_name_v1")
    return None


def _status_rows(
    *,
    reports_root: Path,
    wednesday_summary: dict[str, Any],
    wednesday_manifest: dict[str, Any],
    restore_summary: dict[str, Any],
    rehydrated_summary: dict[str, Any],
) -> pd.DataFrame:
    expected_rows = int((wednesday_summary.get("policy_logging_v1") or {}).get("row_count_v1") or 0)
    expected_asof = int((wednesday_manifest.get("as_of_schema_v1") or {}).get("column_count_v1") or 0)
    expected_r5_2_freeze = str(wednesday_manifest.get("r5_2_benchmark_freeze_id_v1") or "")
    expected_r6_candidate = str(wednesday_summary.get("selected_candidate_id_v1") or "")

    r5_dir = reports_root / R5_DIR_NAME
    r5_1_dir = reports_root / R5_1_DIR_NAME
    r5_2_dir = reports_root / R5_2_DIR_NAME
    r5_2_freeze_dir = reports_root / R5_2_FREEZE_DIR_NAME
    r6_dir = reports_root / R6_DIR_NAME

    r5_summary = _read_json_if_exists(r5_dir / R5_SUMMARY)
    r5_1_summary = _read_json_if_exists(r5_1_dir / R5_1_SUMMARY)
    r5_2_summary = _read_json_if_exists(r5_2_dir / R5_2_SUMMARY)
    r5_2_freeze_manifest = _read_json_if_exists(r5_2_freeze_dir / R5_2_FREEZE_MANIFEST)
    r6_summary = _read_json_if_exists(r6_dir / R6_SUMMARY)

    rows = [
        {
            "check_v1": "WEDNESDAY_R6_SNAPSHOT_PRESENT",
            "status_v1": "PASS" if wednesday_summary and wednesday_manifest else "FAIL",
            "expected_v1": expected_r6_candidate,
            "observed_v1": wednesday_summary.get("freeze_id_v1"),
            "action_v1": "USE_AS_CONTRACT_LOCK",
        },
        {
            "check_v1": "WEDNESDAY_R6_HASH_SCAN_FOUND_ALL",
            "status_v1": "PASS"
            if int(restore_summary.get("canonical_hash_scan_match_count_v1") or 0) == int(restore_summary.get("canonical_hash_rows_v1") or -1)
            else "FAIL",
            "expected_v1": restore_summary.get("canonical_hash_rows_v1"),
            "observed_v1": restore_summary.get("canonical_hash_scan_match_count_v1"),
            "action_v1": "RESTORE_WEDNESDAY_SOURCE_ARTIFACTS_FIRST",
        },
        {
            "check_v1": "MONDAY_REHYDRATED_109_AS_OF_PRESENT",
            "status_v1": "PASS" if int(rehydrated_summary.get("as_of_column_count_v1") or 0) == expected_asof else "FAIL",
            "expected_v1": expected_asof,
            "observed_v1": rehydrated_summary.get("as_of_column_count_v1"),
            "action_v1": "KEEP_AS_SHAPE_INPUT_NOT_SCORE_TRUTH",
        },
        {
            "check_v1": "R5_FOUNDATION_FULL_1971",
            "status_v1": "PASS" if _coverage_rows(r5_summary) == expected_rows and _safe_parquet_rows(r5_dir / R5_AS_OF) == expected_rows else "FAIL",
            "expected_v1": expected_rows,
            "observed_v1": _coverage_rows(r5_summary),
            "action_v1": "REBUILD_R5_FOUNDATION_WITH_WEDNESDAY_FULLCOVERAGE_CONTRACT",
        },
        {
            "check_v1": "R5_1_FOUNDATION_FULL_1971",
            "status_v1": "PASS" if _coverage_rows(r5_1_summary) == expected_rows and _safe_parquet_rows(r5_1_dir / R5_1_AS_OF) == expected_rows else "FAIL",
            "expected_v1": expected_rows,
            "observed_v1": _coverage_rows(r5_1_summary),
            "action_v1": "REBUILD_R5_1_FROM_FULL_R5_FOUNDATION",
        },
        {
            "check_v1": "R5_2_EXPECTED_FREEZE_10176_PRESENT",
            "status_v1": "PASS" if r5_2_freeze_manifest.get("freeze_id_v1") == expected_r5_2_freeze else "FAIL",
            "expected_v1": expected_r5_2_freeze,
            "observed_v1": r5_2_freeze_manifest.get("freeze_id_v1"),
            "action_v1": "REBUILD_R5_2_BASE_FROM_FULL_FOUNDATION_OR_RESTORE_FREEZE",
        },
        {
            "check_v1": "R5_2_POLICY_SURFACE_FULL_1971",
            "status_v1": "PASS" if _coverage_rows(r5_2_summary) == expected_rows and _safe_parquet_rows(r5_2_dir / R5_2_POLICY) == expected_rows else "FAIL",
            "expected_v1": expected_rows,
            "observed_v1": _coverage_rows(r5_2_summary),
            "action_v1": "DO_NOT_USE_ZERO_BLOCK_ADBB_BASE_AS_R6_REFERENCE",
        },
        {
            "check_v1": "R6_LOCAL_IS_CANONICAL_04761_1971_109",
            "status_v1": "PASS"
            if _coverage_rows(r6_summary) == expected_rows
            and _safe_parquet_rows(r6_dir / R6_AS_OF) == expected_rows
            and _selected_policy(r6_summary) == expected_r6_candidate
            else "FAIL",
            "expected_v1": f"{expected_r6_candidate}|rows={expected_rows}|asof={expected_asof}",
            "observed_v1": f"{_selected_policy(r6_summary)}|rows={_coverage_rows(r6_summary)}|asof_rows={_safe_parquet_rows(r6_dir / R6_AS_OF)}",
            "action_v1": "REBUILD_R6_ONLY_AFTER_R5_2_BASE_IS_CANONICAL",
        },
        {
            "check_v1": "NARROW_1689_NOT_USED",
            "status_v1": "PASS",
            "expected_v1": "NOT_USED_AS_R6_BASELINE",
            "observed_v1": "DIAGNOSTIC_ONLY",
            "action_v1": "KEEP_QUARANTINED",
        },
    ]
    return pd.DataFrame(rows)


def _command_plan(summary: dict[str, Any]) -> pd.DataFrame:
    commands = [
        {
            "step_v1": 1,
            "action_v1": "HASH_VERIFY_OR_RESTORE_WEDNESDAY_R6_SOURCES",
            "run_now_v1": False,
            "command_v1": (
                "python3 -m gx1.scripts.materialize_monday_r6_restore_or_rebuild_canonical_score_and_label_sources_v1 "
                "--reports-root /home/andre2/GX1_DATA/reports/truth_e2e_sanity"
            ),
            "expected_output_v1": "canonical_hash_scan_v1.csv with all frozen Wednesday R6 hashes found, or explicit source-absent proof",
        },
        {
            "step_v1": 2,
            "action_v1": "REBUILD_R5_FOUNDATION_TO_1971",
            "run_now_v1": False,
            "command_v1": "REBUILD using Wednesday fullcoverage/repaired-entry contract; current local R5 is not enough if it stays 1852",
            "expected_output_v1": "R5 AS_OF/HINDSIGHT/POLICY surfaces with 1971 rows and full repaired coverage",
        },
        {
            "step_v1": 3,
            "action_v1": "REBUILD_R5_1_FROM_FULL_R5_FOUNDATION",
            "run_now_v1": False,
            "command_v1": "REBUILD R5.1 LOSO/batch04 robustness on the same 1971 fullcoverage foundation",
            "expected_output_v1": "R5.1 policy prediction and failure attribution with 1971-row lineage",
        },
        {
            "step_v1": 4,
            "action_v1": "RUN_R5_2_CANONICAL_BASE_REBUILD",
            "run_now_v1": False,
            "command_v1": (
                "python3 -m gx1.scripts.train_r5_2_entry_runner_aware_retrain_and_loso_selection_v1 "
                "--reports-root /home/andre2/GX1_DATA/reports/truth_e2e_sanity "
                "--expected-ledger-count 1971 "
                "--extension-dir /home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
                "MONDAY_R6_CANONICAL_R5_2_BASE_REBUILD_FROM_WEDNESDAY_CONTRACT_V1"
            ),
            "expected_output_v1": "R5.2 policy surface with nonzero canonical base behavior matching the Wednesday R6 contract expectations",
        },
        {
            "step_v1": 5,
            "action_v1": "RUN_MONDAY_R6_REBUILD_ON_CANONICAL_R5_2_BASE",
            "run_now_v1": False,
            "command_v1": (
                "python3 -m gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 "
                "--reports-root /home/andre2/GX1_DATA/reports/truth_e2e_sanity --run-training"
            ),
            "expected_output_v1": "Monday R6 compare/verdict passes Wednesday safety gates before any freeze/promo consideration",
        },
    ]
    if summary["decision_v1"] == "MONDAY_R5_2_BASE_REBUILD_READY_TO_RUN":
        commands[3]["run_now_v1"] = True
    return pd.DataFrame(commands)


def _blocked_fields(rehydrated_dir: Path | None) -> pd.DataFrame:
    if not rehydrated_dir:
        return pd.DataFrame(columns=["field_v1", "surface_v1", "status_v1", "required_action_v1"])
    path = rehydrated_dir / "monday_r6_rehydration_blocked_fields_v1.csv"
    if not path.exists():
        return pd.DataFrame(columns=["field_v1", "surface_v1", "status_v1", "required_action_v1"])
    frame = pd.read_csv(path)
    frame["required_action_v1"] = frame["field_v1"].astype("string").map(
        lambda field: "RESTORE_CANONICAL_SCORE_SOURCE"
        if str(field).startswith("pred__") or field in {"blocker_score_v1", "runner_protector_score_v1"}
        else "RESTORE_CANONICAL_EXACT_LABEL_SOURCE"
    )
    return frame


def _audit(summary: dict[str, Any], status_df: pd.DataFrame) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    failures = status_df[status_df["status_v1"].eq("FAIL")]
    return pd.DataFrame(
        [
            row("NO_TRAINING_STARTED", "PASS", summary["training_started_v1"]),
            row("NO_FREEZE_OR_PROMOTION", "PASS", True),
            row("R6_REMAINS_GOVERNING_LINE", "PASS", summary["governing_line_v1"]),
            row("R5_2_BASE_READY", "PASS" if summary["decision_v1"] == "MONDAY_R5_2_BASE_REBUILD_READY_TO_RUN" else "FAIL", failures["check_v1"].tolist()),
            row("NARROW_1689_QUARANTINED", "PASS", summary["blocked_action_v1"]),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Monday R6 Canonical R5.2 Base Rebuild Plan V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Wednesday expected rows: `{summary['wednesday_expected_rows_v1']}`",
            f"- Expected R5.2 freeze: `{summary['expected_r5_2_freeze_id_v1']}`",
            f"- Hash scan matches: `{summary['canonical_hash_scan_match_count_v1']}/{summary['canonical_hash_rows_v1']}`",
            f"- Failed prerequisites: `{summary['failed_prerequisite_count_v1']}`",
            f"- Training started: `{summary['training_started_v1']}`",
            "",
            "R6 is the governing line. R5/R5.1/R5.2 are upstream score/base dependencies inside that R6 line, not a competing direction.",
            "The 1689 exact-only/protector-first surface remains diagnostic only.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    restore_dir: Path | None = None,
    rehydrated_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    restore_dir = restore_dir.expanduser().resolve() if restore_dir else _latest_dir(reports_root, RESTORE_GLOB, "summary_v1.json")
    rehydrated_dir = rehydrated_dir.expanduser().resolve() if rehydrated_dir else _latest_dir(reports_root, REHYDRATED_GLOB, "summary_v1.json")
    output_dir = output_dir.expanduser().resolve() if output_dir else reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    freeze_dir = reports_root / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    wednesday_summary = _read_json(freeze_dir / WEDNESDAY_SUMMARY)
    wednesday_manifest = _read_json(freeze_dir / WEDNESDAY_MANIFEST)
    restore_summary = _read_json_if_exists((restore_dir / "summary_v1.json") if restore_dir else Path(""))
    rehydrated_summary = _read_json_if_exists((rehydrated_dir / "summary_v1.json") if rehydrated_dir else Path(""))

    status_df = _status_rows(
        reports_root=reports_root,
        wednesday_summary=wednesday_summary,
        wednesday_manifest=wednesday_manifest,
        restore_summary=restore_summary,
        rehydrated_summary=rehydrated_summary,
    )
    failed = status_df[status_df["status_v1"].eq("FAIL")]
    blocking_checks = set(failed["check_v1"].astype(str).tolist())
    expected_rows = int((wednesday_summary.get("policy_logging_v1") or {}).get("row_count_v1") or 0)

    if not blocking_checks:
        decision = "MONDAY_R5_2_BASE_REBUILD_READY_TO_RUN"
        next_action = "RUN_R5_2_CANONICAL_BASE_REBUILD_WITH_EXPLICIT_COMMAND"
    elif "WEDNESDAY_R6_HASH_SCAN_FOUND_ALL" in blocking_checks:
        decision = "MONDAY_R5_2_BASE_REBUILD_BLOCKED_BY_MISSING_WEDNESDAY_SOURCE_HASHES"
        next_action = "RESTORE_WEDNESDAY_SOURCE_ARTIFACTS_FIRST"
    elif {"R5_FOUNDATION_FULL_1971", "R5_1_FOUNDATION_FULL_1971"} & blocking_checks:
        decision = "MONDAY_R5_2_BASE_REBUILD_BLOCKED_BY_MISSING_1971_R5_FOUNDATION"
        next_action = "REBUILD_R5_AND_R5_1_FOUNDATION_TO_1971_WITH_WEDNESDAY_FULLCOVERAGE_CONTRACT_FIRST"
    else:
        decision = "MONDAY_R5_2_BASE_REBUILD_BLOCKED_BY_LINEAGE_OR_FREEZE_GAP"
        next_action = "FIX_R5_2_LINEAGE_OR_RESTORE_EXPECTED_10176_FREEZE_FIRST"

    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "reports_root_v1": str(reports_root),
        "restore_dir_v1": str(restore_dir) if restore_dir else None,
        "rehydrated_dir_v1": str(rehydrated_dir) if rehydrated_dir else None,
        "governing_line_v1": "MONDAY_R6_USING_CANONICAL_WEDNESDAY_R6_CONTRACT",
        "wednesday_freeze_id_v1": wednesday_summary.get("freeze_id_v1"),
        "wednesday_candidate_id_v1": wednesday_summary.get("selected_candidate_id_v1"),
        "wednesday_expected_rows_v1": expected_rows,
        "wednesday_as_of_columns_v1": (wednesday_manifest.get("as_of_schema_v1") or {}).get("column_count_v1"),
        "expected_r5_2_freeze_id_v1": wednesday_manifest.get("r5_2_benchmark_freeze_id_v1"),
        "canonical_hash_rows_v1": restore_summary.get("canonical_hash_rows_v1"),
        "canonical_hash_scan_match_count_v1": restore_summary.get("canonical_hash_scan_match_count_v1"),
        "canonical_hash_scan_missing_count_v1": restore_summary.get("canonical_hash_scan_missing_count_v1"),
        "failed_prerequisite_count_v1": int(len(failed)),
        "failed_prerequisites_v1": failed["check_v1"].astype(str).tolist(),
        "training_started_v1": False,
        "not_live_gate_v1": True,
        "not_freeze_or_promo_v1": True,
        "decision_v1": decision,
        "next_action_v1": next_action,
        "blocked_action_v1": [
            "DO_NOT_RUN_MONDAY_R6_RETRAIN_UNTIL_R5_2_BASE_IS_CANONICAL",
            "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE",
            "DO_NOT_USE_LOCAL_ADBB_R5_2_ZERO_BLOCK_BASE_AS_CANONICAL_R6_REFERENCE",
            "DO_NOT_FREEZE_OR_PROMOTE_FROM_THIS_PLAN",
        ],
        "hard_status_v1": {
            "BEVIST": [
                "This action did not train, freeze, promote, or write live/controller artifacts.",
                "R6 remains the governing line; R5.2 is only the upstream base dependency required by Wednesday R6.",
                "The local 1689 exact-only surface is not used.",
            ],
            "INDIKERT": [
                "Monday has a 109-column rehydrated shape package, but the score/label/base lineage is not green.",
            ],
            "IKKE_ETABLERT": [
                "Canonical R5.2 freeze 10176 is not locally restored.",
                "Current local R5/R5.1/R5.2 foundation is not proven as the 1971-row Wednesday fullcoverage line.",
                "Monday R6 cannot be declared canonical until this base is rebuilt or restored.",
            ],
        },
    }
    command_df = _command_plan(summary)
    blocked_df = _blocked_fields(rehydrated_dir)
    audit_df = _audit(summary, status_df)
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "artifacts_v1": OUTPUT_FILES,
        "training_started_v1": False,
        "not_live_gate_v1": True,
        "not_freeze_or_promo_v1": True,
    }

    status_df.to_csv(output_dir / OUTPUT_FILES["input_source_status"], index=False)
    command_df.to_csv(output_dir / OUTPUT_FILES["command_plan"], index=False)
    blocked_df.to_csv(output_dir / OUTPUT_FILES["blocked_fields_to_resolve"], index=False)
    audit_df.to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["manifest"], manifest)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--restore-dir", type=Path, default=None)
    parser.add_argument("--rehydrated-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        restore_dir=args.restore_dir,
        rehydrated_dir=args.rehydrated_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
