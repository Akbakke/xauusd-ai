#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "MONDAY_R6_WEDNESDAY_SOURCE_RESTORE_ATTEMPT_V1"

WEDNESDAY_SNAPSHOT_DIR = "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_20260424T0900Z"
WEDNESDAY_FREEZE_DIR = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1"
WEDNESDAY_MANIFEST = "shadow_meta_all_trade_review_r6_shadow_freeze_manifest_v1.json"
WEDNESDAY_SUMMARY = "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_summary_v1.json"

REQUIRED_CANONICAL_SOURCE_ARTIFACTS = [
    "shadow_meta_all_trade_review_r6_entry_runner_first_contract_v1.json",
    "shadow_meta_all_trade_review_r6_entry_runner_first_as_of_feature_table_v1.parquet",
    "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet",
    "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet",
    "shadow_meta_all_trade_review_r6_model_family_bakeoff_v1.csv",
    "shadow_meta_all_trade_review_r6_loso_metrics_v1.csv",
    "shadow_meta_all_trade_review_r6_head_to_head_vs_r2_r4_r5_r5_1_r5_2_v1.csv",
    "models",
]

OUTPUT_FILES = {
    "summary": "summary_v1.json",
    "expected_hashed_artifacts": "expected_hashed_artifacts_v1.csv",
    "filesystem_hash_scan": "filesystem_hash_scan_v1.csv",
    "archive_member_scan": "archive_member_scan_v1.csv",
    "required_source_artifact_status": "required_source_artifact_status_v1.csv",
    "restore_action_lock": "restore_action_lock_v1.json",
    "consistency_audit": "consistency_audit_v1.csv",
    "manifest": "manifest_v1.json",
    "report": "report_v1.md",
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_hash_rows(manifest: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, group_rows in (manifest.get("hashes_v1") or {}).items():
        if not isinstance(group_rows, list):
            continue
        for row in group_rows:
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "hash_group_v1": str(group_name),
                    "hash_kind_v1": str(row.get("hash_kind_v1") or ""),
                    "relative_path_v1": str(row.get("relative_path_v1") or ""),
                    "artifact_name_v1": Path(str(row.get("relative_path_v1") or "")).name,
                    "expected_absolute_path_v1": str(row.get("absolute_path_v1") or ""),
                    "expected_byte_size_v1": row.get("byte_size_v1"),
                    "expected_sha256_v1": str(row.get("sha256_v1") or ""),
                }
            )
    return pd.DataFrame(rows)


def _filesystem_hash_scan(expected: pd.DataFrame, scan_roots: list[Path]) -> pd.DataFrame:
    columns = [
        "expected_sha256_v1",
        "relative_path_v1",
        "scan_root_v1",
        "candidate_file_count_v1",
        "matched_path_v1",
        "match_found_v1",
        "status_v1",
    ]
    if expected.empty:
        return pd.DataFrame(columns=columns)

    target_names = sorted(set(expected["artifact_name_v1"].dropna().astype(str).tolist()))
    expected_by_sha = {
        str(row.expected_sha256_v1): row
        for row in expected.itertuples(index=False)
        if str(row.expected_sha256_v1)
    }
    matches: dict[str, list[str]] = {sha: [] for sha in expected_by_sha}
    candidate_count = 0
    root_list = [root.expanduser().resolve() for root in scan_roots]
    for root in root_list:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.name not in target_names:
                continue
            candidate_count += 1
            observed = _sha256(path)
            if observed in matches:
                matches[observed].append(str(path))

    rows: list[dict[str, Any]] = []
    scan_root_text = "|".join(str(root) for root in root_list)
    for row in expected.itertuples(index=False):
        sha = str(row.expected_sha256_v1)
        paths = matches.get(sha, [])
        rows.append(
            {
                "expected_sha256_v1": sha,
                "relative_path_v1": row.relative_path_v1,
                "scan_root_v1": scan_root_text,
                "candidate_file_count_v1": int(candidate_count),
                "matched_path_v1": paths[0] if paths else "",
                "match_found_v1": bool(paths),
                "status_v1": "FOUND_BY_HASH" if paths else "NOT_FOUND_BY_HASH",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _archive_members(path: Path) -> tuple[int | None, list[str], str]:
    try:
        if tarfile.is_tarfile(path):
            with tarfile.open(path, mode="r:*") as archive:
                names = archive.getnames()
            return len(names), names, "TAR_READ_OK"
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
            return len(names), names, "ZIP_READ_OK"
        return None, [], "UNSUPPORTED_ARCHIVE_TYPE"
    except Exception as exc:  # pragma: no cover - defensive archive diagnostics
        return None, [], f"ARCHIVE_READ_FAILED:{type(exc).__name__}:{exc}"


def _archive_scan(expected: pd.DataFrame, archive_paths: list[Path], source_dir_name: str) -> pd.DataFrame:
    expected_rel = sorted(set(expected["relative_path_v1"].dropna().astype(str).tolist())) if not expected.empty else []
    expected_names = sorted(set(expected["artifact_name_v1"].dropna().astype(str).tolist())) if not expected.empty else []
    rows: list[dict[str, Any]] = []
    for raw_path in archive_paths:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            rows.append(
                {
                    "archive_path_v1": str(path),
                    "archive_exists_v1": False,
                    "archive_status_v1": "MISSING",
                    "member_count_v1": None,
                    "source_dir_name_hit_count_v1": 0,
                    "expected_relative_path_hit_count_v1": 0,
                    "expected_artifact_name_hit_count_v1": 0,
                    "restorable_from_archive_v1": False,
                }
            )
            continue
        member_count, names, status = _archive_members(path)
        source_hits = [name for name in names if source_dir_name and source_dir_name in name]
        relative_hits = [rel for rel in expected_rel if any(name.endswith(rel) for name in names)]
        name_hits = [name for name in names if Path(name).name in expected_names]
        rows.append(
            {
                "archive_path_v1": str(path),
                "archive_exists_v1": True,
                "archive_status_v1": status,
                "member_count_v1": member_count,
                "source_dir_name_hit_count_v1": int(len(source_hits)),
                "expected_relative_path_hit_count_v1": int(len(relative_hits)),
                "expected_artifact_name_hit_count_v1": int(len(name_hits)),
                "restorable_from_archive_v1": bool(expected_rel and len(relative_hits) == len(expected_rel)),
            }
        )
    return pd.DataFrame(rows)


def _required_status(source_dir: Path) -> pd.DataFrame:
    rows = []
    for artifact in REQUIRED_CANONICAL_SOURCE_ARTIFACTS:
        path = source_dir / artifact
        rows.append(
            {
                "artifact_v1": artifact,
                "expected_path_v1": str(path),
                "exists_v1": bool(path.exists()),
                "status_v1": "PRESENT" if path.exists() else "MISSING",
            }
        )
    return pd.DataFrame(rows)


def _audit(summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("NO_TRAINING_STARTED", "PASS", summary["training_started_v1"]),
            row("NO_FREEZE_OR_PROMOTION", "PASS", summary["not_freeze_or_promo_v1"]),
            row("EXPECTED_HASH_ROWS_LOADED", "PASS" if summary["expected_hash_rows_v1"] > 0 else "FAIL", summary["expected_hash_rows_v1"]),
            row("CANONICAL_SOURCE_TREE_PRESENT", "PASS" if summary["canonical_source_tree_present_v1"] else "FAIL", summary["canonical_r6_source_dir_v1"]),
            row("ALL_HASHES_FOUND", "PASS" if summary["missing_hash_count_v1"] == 0 else "FAIL", summary["missing_hash_count_v1"]),
            row("ARCHIVES_SCANNED", "PASS" if summary["archive_scan_count_v1"] > 0 else "WARN", summary["archive_scan_count_v1"]),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Monday R6 Wednesday Source Restore Attempt V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Canonical source dir: `{summary['canonical_r6_source_dir_v1']}`",
            f"- Source tree present: `{summary['canonical_source_tree_present_v1']}`",
            f"- Expected hash rows: `{summary['expected_hash_rows_v1']}`",
            f"- Filesystem hash matches: `{summary['filesystem_hash_match_count_v1']}/{summary['expected_hash_rows_v1']}`",
            f"- Missing hashes: `{summary['missing_hash_count_v1']}`",
            f"- Archives scanned: `{summary['archive_scan_count_v1']}`",
            f"- Archive restorable candidates: `{summary['archive_restorable_candidate_count_v1']}`",
            f"- Training started: `{summary['training_started_v1']}`",
            "",
            "Only exact hash matches are allowed to restore the frozen Wednesday R6 source. Local noncanonical R6/R5.2 outputs are not accepted as substitutes.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    scan_roots: list[Path] | None = None,
    archive_paths: list[Path] | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve() if output_dir else reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_roots = scan_roots or [reports_root]
    archive_paths = archive_paths or []

    freeze_dir = reports_root / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    summary_lock = _read_json(freeze_dir / WEDNESDAY_SUMMARY)
    manifest_lock = _read_json(freeze_dir / WEDNESDAY_MANIFEST)
    source_dir = Path(str(manifest_lock.get("r6_source_dir_v1") or "")).expanduser()
    source_dir_name = source_dir.name

    expected = _expected_hash_rows(manifest_lock)
    fs_scan = _filesystem_hash_scan(expected, scan_roots)
    archive_scan = _archive_scan(expected, archive_paths, source_dir_name)
    required = _required_status(source_dir)

    expected_count = int(len(expected))
    fs_match_count = int(fs_scan["match_found_v1"].fillna(False).astype(bool).sum()) if not fs_scan.empty else 0
    missing_hash_count = int(expected_count - fs_match_count)
    source_tree_present = bool(source_dir.exists() and source_dir.is_dir())
    required_missing_count = int((~required["exists_v1"].fillna(False).astype(bool)).sum()) if not required.empty else 0
    archive_restorable = int(archive_scan["restorable_from_archive_v1"].fillna(False).astype(bool).sum()) if not archive_scan.empty else 0

    if source_tree_present and missing_hash_count == 0 and required_missing_count == 0:
        decision = "WEDNESDAY_SOURCE_ARTIFACTS_RESTORED_AND_HASH_VERIFIED"
        next_action = "RUN_MONDAY_FULLCOVERAGE_REBUILD_USING_WEDNESDAY_CONTRACT"
    elif missing_hash_count == 0:
        decision = "WEDNESDAY_SOURCE_HASHES_FOUND_BUT_SOURCE_TREE_INCOMPLETE"
        next_action = "RESTORE_REQUIRED_NONHASHED_WEDNESDAY_SOURCE_TABLES_FIRST"
    else:
        decision = "WEDNESDAY_SOURCE_ARTIFACTS_NOT_FOUND_LOCALLY"
        next_action = "RESTORE_WEDNESDAY_SOURCE_ARTIFACTS_FIRST"

    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "reports_root_v1": str(reports_root),
        "wednesday_freeze_id_v1": summary_lock.get("freeze_id_v1"),
        "wednesday_candidate_id_v1": summary_lock.get("selected_candidate_id_v1"),
        "canonical_r6_source_dir_v1": str(source_dir),
        "canonical_source_tree_present_v1": source_tree_present,
        "expected_hash_rows_v1": expected_count,
        "filesystem_scan_roots_v1": [str(path.expanduser().resolve()) for path in scan_roots],
        "filesystem_hash_candidate_file_count_v1": int(fs_scan["candidate_file_count_v1"].max()) if not fs_scan.empty else 0,
        "filesystem_hash_match_count_v1": fs_match_count,
        "missing_hash_count_v1": missing_hash_count,
        "required_source_artifact_missing_count_v1": required_missing_count,
        "archive_scan_count_v1": int(len(archive_scan)),
        "archive_restorable_candidate_count_v1": archive_restorable,
        "training_started_v1": False,
        "not_live_gate_v1": True,
        "not_freeze_or_promo_v1": True,
        "decision_v1": decision,
        "next_action_v1": next_action,
        "blocked_action_v1": [
            "DO_NOT_USE_LOCAL_1852_04789_R6_AS_CANONICAL_SOURCE",
            "DO_NOT_USE_LOCAL_ADBB_R5_2_FREEZE_AS_CANONICAL_SOURCE",
            "DO_NOT_RUN_MONDAY_R6_RETRAIN_UNTIL_WEDNESDAY_SOURCE_ARTIFACTS_ARE_RESTORED",
            "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE",
        ],
        "hard_status_v1": {
            "BEVIST": [
                "No training, freeze, promotion, live gate, or controller change was run.",
                "The frozen Wednesday R6 manifest was loaded and exact model/preprocessor hashes were checked.",
            ],
            "INDIKERT": [
                "Local noncanonical R6/R5.2 outputs can diagnose the gap, but are not canonical restore sources.",
            ],
            "IKKE_ETABLERT": [
                "The frozen Wednesday R6 source tree is not restored unless canonical_source_tree_present_v1 is true and missing_hash_count_v1 is zero.",
            ],
        },
    }
    restore_lock = {
        "decision_v1": decision,
        "next_action_v1": next_action,
        "required_external_restore_source_v1": str(source_dir),
        "expected_hash_rows_v1": expected_count,
        "missing_hash_count_v1": missing_hash_count,
        "acceptable_restore_condition_v1": "all 15 frozen Wednesday R6 model/preprocessor/metadata hashes match and required source tables exist",
        "forbidden_substitutes_v1": summary["blocked_action_v1"],
    }
    audit = _audit(summary)
    artifact_manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "artifacts_v1": OUTPUT_FILES,
        "training_started_v1": False,
        "not_freeze_or_promo_v1": True,
    }

    expected.to_csv(output_dir / OUTPUT_FILES["expected_hashed_artifacts"], index=False)
    fs_scan.to_csv(output_dir / OUTPUT_FILES["filesystem_hash_scan"], index=False)
    archive_scan.to_csv(output_dir / OUTPUT_FILES["archive_member_scan"], index=False)
    required.to_csv(output_dir / OUTPUT_FILES["required_source_artifact_status"], index=False)
    audit.to_csv(output_dir / OUTPUT_FILES["consistency_audit"], index=False)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / OUTPUT_FILES["restore_action_lock"], restore_lock)
    _write_json(output_dir / OUTPUT_FILES["manifest"], artifact_manifest)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scan-root", type=Path, action="append", default=None)
    parser.add_argument("--archive-path", type=Path, action="append", default=None)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        scan_roots=args.scan_root,
        archive_paths=args.archive_path,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
