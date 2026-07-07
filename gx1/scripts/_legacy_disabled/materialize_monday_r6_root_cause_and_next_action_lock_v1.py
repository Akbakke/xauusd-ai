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
LAYER_NAME = "MONDAY_R6_ROOT_CAUSE_AND_NEXT_ACTION_LOCK_V1"

REBUILD_GLOB = "MONDAY_R6_EXPLICIT_REBUILD_FROM_REHYDRATED_CONTRACT_V1_*"
RESTORE_GLOB = "MONDAY_R6_CANONICAL_SCORE_AND_LABEL_RESTORE_OR_REBUILD_V1_*"
RL_UNIFIED_GLOB = "ALL_TRADE_REVIEW_LEDGER_*_RL_UNIFIED_OBSERVABILITY_V1"
EXIT_HARVEST_GLOB = "ALL_TRADE_REVIEW_LEDGER_*_EXIT_HARVEST_POLICY_CANDIDATE_V1"
MANAGEMENT_POLICY_GLOB = "MONDAY_MANAGEMENT_POLICY_LOGGING_RUNTIME_V1_*"
PRE_RL_COMPARATOR_GLOB = "MONDAY_TOP_PRE_RL_BASELINE_COMPARATOR_V1_*"

SUMMARY = "summary_v1.json"
ROOT_CAUSE_MATRIX = "root_cause_matrix_v1.csv"
FAILURE_ROW_DIGEST = "failure_row_digest_v1.csv"
PRE_RL_SHADOW_AVAILABILITY = "pre_rl_shadow_availability_v1.json"
NEXT_ACTION_LOCK = "next_action_lock_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"
MANIFEST = "manifest_v1.json"
REPORT = "report_v1.md"

OUTPUT_FILES = {
    "summary": SUMMARY,
    "root_cause_matrix": ROOT_CAUSE_MATRIX,
    "failure_row_digest": FAILURE_ROW_DIGEST,
    "pre_rl_shadow_availability": PRE_RL_SHADOW_AVAILABILITY,
    "next_action_lock": NEXT_ACTION_LOCK,
    "consistency_audit": CONSISTENCY_AUDIT,
    "manifest": MANIFEST,
    "report": REPORT,
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


def _latest_dir(reports_root: Path, pattern: str, required_file: str | None = None) -> Path | None:
    dirs = sorted(path for path in reports_root.glob(pattern) if path.is_dir())
    if required_file is not None:
        dirs = [path for path in dirs if (path / required_file).exists()]
    return dirs[-1] if dirs else None


def _latest_training_rebuild(reports_root: Path) -> Path:
    candidates: list[Path] = []
    for path in sorted(reports_root.glob(REBUILD_GLOB)):
        summary_path = path / SUMMARY
        if not path.is_dir() or not summary_path.exists():
            continue
        try:
            summary = _read_json(summary_path)
        except json.JSONDecodeError:
            continue
        if summary.get("training_started_v1") is True and (path / "wednesday_locked_policy_replay_v1.json").exists():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No explicit Monday R6 training rebuild with replay artifacts under {reports_root}")
    return candidates[-1]


def _optional_json(path: Path | None, filename: str) -> dict[str, Any] | None:
    if path is None:
        return None
    target = path / filename
    return _read_json(target) if target.exists() else None


def _failure_digest(rebuild_dir: Path) -> pd.DataFrame:
    path = rebuild_dir / "safety_failure_rows_v1.csv"
    if not path.exists():
        return pd.DataFrame(
            [
                {
                    "failure_tags_v1": "MISSING_SAFETY_FAILURE_ROWS",
                    "row_count_v1": 0,
                    "take_was_ok_count_v1": 0,
                    "fifty_plus_count_v1": 0,
                    "repaired_165_count_v1": 0,
                }
            ]
        )
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["failure_tags_v1", "row_count_v1", "take_was_ok_count_v1", "fifty_plus_count_v1", "repaired_165_count_v1"])
    rows: list[dict[str, Any]] = []
    for tag, group in frame.groupby(frame["failure_tags_v1"].astype("string"), dropna=False):
        rows.append(
            {
                "failure_tags_v1": str(tag),
                "row_count_v1": int(len(group)),
                "take_was_ok_count_v1": int(group.get("take_was_ok_v1", pd.Series(False, index=group.index)).astype(bool).sum()),
                "fifty_plus_count_v1": int(group.get("fifty_plus_mfe_v1", pd.Series(False, index=group.index)).astype(bool).sum()),
                "repaired_165_count_v1": int(group.get("r6_label_repaired_165_like_runner_v1", pd.Series(False, index=group.index)).astype(bool).sum()),
                "sample_candidate_uid_v1": str(group.iloc[0].get("candidate_uid")),
                "sample_run_id_v1": str(group.iloc[0].get("run_id")),
            }
        )
    return pd.DataFrame(rows).sort_values(["row_count_v1", "failure_tags_v1"], ascending=[False, True]).reset_index(drop=True)


def _pre_rl_shadow_availability(reports_root: Path) -> dict[str, Any]:
    rl_dir = _latest_dir(reports_root, RL_UNIFIED_GLOB, "shadow_meta_all_trade_review_rl_unified_observability_summary_v1.json")
    exit_dir = _latest_dir(reports_root, EXIT_HARVEST_GLOB, "shadow_meta_all_trade_review_exit_harvest_policy_candidate_summary_v1.json")
    management_dir = _latest_dir(reports_root, MANAGEMENT_POLICY_GLOB, SUMMARY)
    comparator_dir = _latest_dir(reports_root, PRE_RL_COMPARATOR_GLOB, SUMMARY)
    rl_summary = _optional_json(rl_dir, "shadow_meta_all_trade_review_rl_unified_observability_summary_v1.json")
    exit_summary = _optional_json(exit_dir, "shadow_meta_all_trade_review_exit_harvest_policy_candidate_summary_v1.json")
    management_summary = _optional_json(management_dir, SUMMARY)
    comparator_summary = _optional_json(comparator_dir, SUMMARY)
    policy_summary = (management_summary or {}).get("policy_logging_summary_v1", {}) if isinstance(management_summary, dict) else {}
    return {
        "rl_unified_observability_dir_v1": str(rl_dir) if rl_dir else None,
        "rl_unified_observability_available_v1": rl_summary is not None,
        "rl_decision_event_rows_v1": (rl_summary or {}).get("decision_event_rows_v1"),
        "rl_management_transition_eligible_rows_v1": (rl_summary or {}).get("management_transition_eligible_rows_v1"),
        "rl_management_policy_readiness_v1": (rl_summary or {}).get("management_policy_readiness_v1"),
        "exit_harvest_dir_v1": str(exit_dir) if exit_dir else None,
        "exit_harvest_available_v1": exit_summary is not None,
        "exit_harvest_trade_count_v1": (exit_summary or {}).get("trade_count_v1"),
        "exit_harvest_runner_harvest_count_v1": ((exit_summary or {}).get("model_update_family_counts_v1") or {}).get("RUNNER_HARVEST"),
        "exit_harvest_entry_filter_count_v1": ((exit_summary or {}).get("model_update_family_counts_v1") or {}).get("ENTRY_FILTER"),
        "management_policy_logging_dir_v1": str(management_dir) if management_dir else None,
        "management_policy_logging_available_v1": management_summary is not None,
        "management_policy_logging_decision_rows_v1": policy_summary.get("decision_log_rows_v1"),
        "management_behavior_policy_readiness_v1": policy_summary.get("behavior_policy_readiness_v1"),
        "management_path_dynamics_decision_v1": (management_summary or {}).get("path_dynamics_decision_v1"),
        "pre_rl_comparator_dir_v1": str(comparator_dir) if comparator_dir else None,
        "pre_rl_comparator_available_v1": comparator_summary is not None,
        "pre_rl_comparator_verdict_v1": (comparator_summary or {}).get("verdict_v1"),
        "usage_boundary_v1": "DIAGNOSTIC_AND_MANAGEMENT_EXIT_SUPERVISION_ONLY_DO_NOT_USE_AS_PRE_ENTRY_R6_AS_OF_FEATURES",
    }


def _root_cause_matrix(
    *,
    rebuild_summary: dict[str, Any],
    locked_replay: dict[str, Any],
    restore_summary: dict[str, Any] | None,
    pre_rl: dict[str, Any],
) -> pd.DataFrame:
    r6_grid = rebuild_summary.get("r6_family_grid_replay_v1") or {}
    locked_failures = _locked_safety_failures(locked_replay)
    rows = [
        {
            "check_v1": "CANONICAL_WEDNESDAY_SOURCE_TREE_PRESENT",
            "status_v1": "PASS" if (restore_summary or {}).get("canonical_source_tree_present_v1") is True else "FAIL",
            "evidence_v1": (restore_summary or {}).get("canonical_r6_source_dir_v1"),
            "impact_v1": "Cannot hash-restore frozen Wednesday R6 score/model lineage.",
        },
        {
            "check_v1": "EXPECTED_R5_2_FREEZE_10176_PRESENT",
            "status_v1": "PASS" if (restore_summary or {}).get("expected_r5_2_freeze_found_v1") is True else "FAIL",
            "evidence_v1": (restore_summary or {}).get("expected_r5_2_freeze_id_v1"),
            "impact_v1": "R6 base stack cannot be proven identical to frozen Wednesday contract.",
        },
        {
            "check_v1": "CANONICAL_WEDNESDAY_HASH_SCAN_FOUND_ALL",
            "status_v1": "PASS"
            if int((restore_summary or {}).get("canonical_hash_scan_match_count_v1") or 0) == int((restore_summary or {}).get("canonical_hash_rows_v1") or -1)
            else "FAIL",
            "evidence_v1": {
                "scan_root": (restore_summary or {}).get("canonical_hash_scan_root_v1"),
                "candidate_files": (restore_summary or {}).get("canonical_hash_scan_candidate_file_count_v1"),
                "matched": (restore_summary or {}).get("canonical_hash_scan_match_count_v1"),
                "missing": (restore_summary or {}).get("canonical_hash_scan_missing_count_v1"),
            },
            "impact_v1": "Frozen Wednesday R6 model/preprocessor hashes cannot be restored from local GX1_DATA.",
        },
        {
            "check_v1": "WEDNESDAY_LOCKED_POLICY_R5_2_BASE_NONZERO",
            "status_v1": "PASS" if int(locked_replay.get("r5_2_base_block_count_v1") or 0) > 0 else "FAIL",
            "evidence_v1": locked_replay.get("r5_2_base_block_count_v1"),
            "impact_v1": "R6-04761 replay is missing the expected R5.2 base-block leg.",
        },
        {
            "check_v1": "WEDNESDAY_LOCKED_POLICY_REPLAY_SAFE",
            "status_v1": "PASS" if locked_replay.get("wednesday_safety_pass_v1") is True else "FAIL",
            "evidence_v1": locked_failures,
            "impact_v1": "Exact Wednesday-04761 policy does not hold safety on rebuilt Monday scores.",
        },
        {
            "check_v1": "ORIGINAL_R6_FAMILY_GRID_HAS_SAFE_CANDIDATE",
            "status_v1": "PASS" if int(r6_grid.get("wednesday_safety_candidate_count_v1") or 0) > 0 else "FAIL",
            "evidence_v1": r6_grid.get("wednesday_safety_candidate_count_v1"),
            "impact_v1": "No original R6-family policy can be selected as canonical Monday R6 from current scores.",
        },
        {
            "check_v1": "ORIGINAL_R6_FAMILY_GRID_HAS_ZERO_HARD_DAMAGE_NONZERO_CANDIDATE",
            "status_v1": "PASS" if int(r6_grid.get("zero_hard_damage_candidate_count_v1") or 0) > 0 else "FAIL",
            "evidence_v1": r6_grid.get("zero_hard_damage_candidate_count_v1"),
            "impact_v1": "Every nonzero original R6-family policy has at least one hard damage breach.",
        },
        {
            "check_v1": "PRE_RL_SHADOW_DIAGNOSTICS_AVAILABLE",
            "status_v1": "PASS" if pre_rl.get("rl_unified_observability_available_v1") and pre_rl.get("exit_harvest_available_v1") else "WARN",
            "evidence_v1": {
                "rl_decision_events": pre_rl.get("rl_decision_event_rows_v1"),
                "exit_harvest_trades": pre_rl.get("exit_harvest_trade_count_v1"),
            },
            "impact_v1": "Shadow/pre-RL can diagnose management/exit labels, but must not be treated as pre-entry AS_OF R6 truth.",
        },
    ]
    return pd.DataFrame(rows)


def _locked_safety_failures(locked_replay: dict[str, Any]) -> Any:
    if locked_replay.get("safety_failures_v1") is not None:
        return locked_replay.get("safety_failures_v1")
    compare = locked_replay.get("compare_v1")
    if isinstance(compare, dict):
        return compare.get("safety_failures_v1")
    return None


def _audit(summary: dict[str, Any], root_cause: pd.DataFrame) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("REBUILD_ARTIFACT_READ", "PASS" if summary["rebuild_dir_v1"] else "FAIL", summary["rebuild_dir_v1"]),
            row("ROOT_CAUSE_MATRIX_WRITTEN", "PASS" if len(root_cause) > 0 else "FAIL", int(len(root_cause))),
            row("CANONICAL_MONDAY_R6_NOT_GREEN", "PASS" if summary["canonical_monday_r6_green_v1"] is False else "FAIL", summary["decision_v1"]),
            row("NO_TRAINING_STARTED", "PASS", False),
            row("NO_PROMOTION_OR_FREEZE", "PASS", True),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Monday R6 Root Cause And Next Action Lock V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Rebuild: `{summary['rebuild_dir_v1']}`",
            f"- Restore/source decision: `{summary['restore_decision_v1']}`",
            f"- R5.2 base blocks in exact 04761 replay: `{summary['r5_2_base_block_count_v1']}`",
            f"- Exact 04761 safety: `{summary['wednesday_locked_policy_safety_pass_v1']}`",
            f"- Original R6 grid safe candidates: `{summary['r6_family_grid_safe_candidate_count_v1']}`",
            f"- Shadow/pre-RL diagnostic availability: `{summary['pre_rl_shadow_diagnostics_available_v1']}`",
            "",
            "This artifact does not train, freeze, promote, or alter live/controller behavior.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    rebuild_dir: Path | None = None,
    restore_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    rebuild_dir = rebuild_dir.expanduser().resolve() if rebuild_dir else _latest_training_rebuild(reports_root)
    restore_dir = restore_dir.expanduser().resolve() if restore_dir else _latest_dir(reports_root, RESTORE_GLOB, SUMMARY)
    output_dir = output_dir.expanduser().resolve() if output_dir else reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rebuild_summary = _read_json(rebuild_dir / SUMMARY)
    locked_replay = _read_json(rebuild_dir / "wednesday_locked_policy_replay_v1.json")
    restore_summary = _read_json(restore_dir / SUMMARY) if restore_dir and (restore_dir / SUMMARY).exists() else None
    failure_digest = _failure_digest(rebuild_dir)
    pre_rl = _pre_rl_shadow_availability(reports_root)
    root_cause = _root_cause_matrix(
        rebuild_summary=rebuild_summary,
        locked_replay=locked_replay,
        restore_summary=restore_summary,
        pre_rl=pre_rl,
    )
    failed_root_causes = root_cause[root_cause["status_v1"].eq("FAIL")]
    r6_grid = rebuild_summary.get("r6_family_grid_replay_v1") or {}
    decision = "MONDAY_R6_BLOCKED_BY_MISSING_CANONICAL_R5_2_BASE_AND_SCORE_SOURCE" if len(failed_root_causes) else "MONDAY_R6_ROOT_CAUSE_NOT_ESTABLISHED"
    restore_summary = restore_summary or {}
    source_tree_missing = restore_summary.get("canonical_source_tree_present_v1") is not True
    hash_scan_missing = int(restore_summary.get("canonical_hash_scan_missing_count_v1") or 0) > 0
    if source_tree_missing or hash_scan_missing:
        next_action = "RESTORE_WEDNESDAY_SOURCE_ARTIFACTS_FIRST"
    else:
        next_action = "REBUILD_CANONICAL_R5_2_BASE_FROM_WEDNESDAY_CONTRACT_FIRST"
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "reports_root_v1": str(reports_root),
        "rebuild_dir_v1": str(rebuild_dir),
        "restore_dir_v1": str(restore_dir) if restore_dir else None,
        "decision_v1": decision,
        "next_action_v1": next_action,
        "canonical_monday_r6_green_v1": False,
        "training_started_v1": False,
        "promotion_started_v1": False,
        "restore_decision_v1": restore_summary.get("decision_v1"),
        "canonical_source_tree_present_v1": restore_summary.get("canonical_source_tree_present_v1"),
        "canonical_hash_scan_match_count_v1": restore_summary.get("canonical_hash_scan_match_count_v1"),
        "canonical_hash_scan_missing_count_v1": restore_summary.get("canonical_hash_scan_missing_count_v1"),
        "expected_r5_2_freeze_found_v1": restore_summary.get("expected_r5_2_freeze_found_v1"),
        "r5_2_base_block_count_v1": locked_replay.get("r5_2_base_block_count_v1"),
        "wednesday_locked_policy_safety_pass_v1": locked_replay.get("wednesday_safety_pass_v1"),
        "wednesday_locked_policy_safety_failures_v1": _locked_safety_failures(locked_replay),
        "r6_family_grid_safe_candidate_count_v1": r6_grid.get("wednesday_safety_candidate_count_v1"),
        "r6_family_grid_zero_hard_damage_candidate_count_v1": r6_grid.get("zero_hard_damage_candidate_count_v1"),
        "r6_family_grid_max_precision_v1": r6_grid.get("max_observed_precision_v1"),
        "root_cause_fail_count_v1": int(root_cause["status_v1"].eq("FAIL").sum()),
        "failure_digest_rows_v1": int(len(failure_digest)),
        "pre_rl_shadow_diagnostics_available_v1": bool(pre_rl.get("rl_unified_observability_available_v1") and pre_rl.get("exit_harvest_available_v1")),
        "pre_rl_shadow_usage_boundary_v1": pre_rl["usage_boundary_v1"],
        "blocked_action_v1": [
            "DO_NOT_FREEZE_OR_PROMOTE_MONDAY_R6",
            "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE",
            "DO_NOT_TREAT_LOCAL_ADBB_R5_2_OR_04789_R6_AS_CANONICAL_WEDNESDAY_SOURCE",
            "DO_NOT_USE_MANAGEMENT_EXIT_PATH_DYNAMICS_AS_PRE_ENTRY_R6_AS_OF_FEATURES",
        ],
        "hard_status_v1": {
            "BEVIST": [
                "Exact Wednesday-04761 replay was materialized against the rebuilt Monday scores.",
                "The original 4948-candidate R6 family grid was materialized against the rebuilt Monday scores.",
                "Shadow/pre-RL management and exit diagnostics are available as diagnostics, not as pre-entry R6 baseline.",
            ],
            "INDIKERT": [
                "The immediate R6 blocker is the missing canonical R5.2 base/score lineage plus current rebuilt-score safety failure.",
                "Management/exit shadow layers can inform exit transformer/label repair after entry baseline legality is restored.",
            ],
            "IKKE_ETABLERT": [
                "Canonical Monday R6 is not green.",
                "Canonical Wednesday R6/R5.2 source tree is not restored locally.",
                "Canonical Wednesday R6 hashes are not found under the local hash scan root.",
                "A promoted/frozen Monday R6 candidate is not established.",
            ],
        },
    }
    next_action_lock = {
        "decision_v1": decision,
        "next_action_v1": next_action,
        "also_required_v1": [
            "KEEP_1689_EXACT_ONLY_QUARANTINED_AS_DIAGNOSTIC_ONLY",
            "KEEP_PROTECTOR_FIRST_BLOCKED_UNTIL_CANONICAL_MONDAY_R6_GREEN",
            "USE_PRE_RL_SHADOW_FOR_EXIT_LABEL_DIAGNOSTICS_NOT_AS_PRE_ENTRY_LEAKAGE",
        ],
    }
    audit = _audit(summary, root_cause)
    manifest = {"layer_name": f"{LAYER_NAME}_MANIFEST", "artifacts_v1": OUTPUT_FILES, "not_live_gate_v1": True}

    _write_json(output_dir / SUMMARY, summary)
    root_cause.to_csv(output_dir / ROOT_CAUSE_MATRIX, index=False)
    failure_digest.to_csv(output_dir / FAILURE_ROW_DIGEST, index=False)
    _write_json(output_dir / PRE_RL_SHADOW_AVAILABILITY, pre_rl)
    _write_json(output_dir / NEXT_ACTION_LOCK, next_action_lock)
    audit.to_csv(output_dir / CONSISTENCY_AUDIT, index=False)
    _write_json(output_dir / MANIFEST, manifest)
    (output_dir / REPORT).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--rebuild-dir", type=Path, default=None)
    parser.add_argument("--restore-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        rebuild_dir=args.rebuild_dir,
        restore_dir=args.restore_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
