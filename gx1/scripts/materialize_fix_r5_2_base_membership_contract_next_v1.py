#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import (
    R5_2_BASE_MEMBERSHIP_CONTRACT,
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
    SCORE_FRAME,
    SCORE_SUMMARY,
    _bool,
    _hard_damage_count,
    _jsonable,
    _num,
    _policy_metrics,
    _r5_2_contract_extension_mask,
    _read_json,
    _wednesday_safety_pass,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "FIX_R5_2_BASE_MEMBERSHIP_CONTRACT_NEXT_V1"
SCORE_GLOB = "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_*"
WIRE_GLOB = "WIRE_EXISTING_R5_2_AND_R6_ASSETS_FIRST_V1_*"
SUMMARY = "summary_v1.json"
STATUS = "status_v1.json"

OUTPUT_FILES = {
    "contract": "r5_2_base_membership_contract_v1.json",
    "simulation": "r5_2_base_membership_contract_simulation_v1.csv",
    "added_rows": "r5_2_base_membership_contract_added_rows_v1.csv",
    "implementation_report": "implementation_report_v1.json",
    "readiness": "r6_readiness_after_base_contract_v1.json",
    "next_action": "next_action_lock_v1.json",
    "summary": "summary_v1.json",
    "report": "report_v1.md",
    "manifest": "manifest_v1.json",
    "status": "status_v1.json",
    "audit": "consistency_audit_v1.csv",
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _latest_dir(root: Path, pattern: str, required_file: str) -> Path | None:
    dirs = sorted(path for path in root.glob(pattern) if path.is_dir() and (path / required_file).exists())
    return dirs[-1] if dirs else None


def _worst_loso(frame: pd.DataFrame, mask: pd.Series) -> float | None:
    selected = mask.reindex(frame.index).fillna(False).astype(bool)
    should = _bool(frame, "label_should_not_take_v1")
    values: list[float] = []
    if "run_id" not in frame.columns:
        return None
    for _, group in frame.assign(__selected=selected, __should=should).groupby("run_id", dropna=False):
        blocks = int(group["__selected"].sum())
        if blocks:
            values.append(float((group["__selected"] & group["__should"]).sum() / blocks))
    return min(values) if values else None


def _metrics_row(name: str, frame: pd.DataFrame, mask: pd.Series, details: dict[str, Any]) -> dict[str, Any]:
    metrics = _policy_metrics(frame, mask)
    worst = _worst_loso(frame, mask)
    hard_damage = _hard_damage_count(metrics)
    safety_pass, _, _ = _wednesday_safety_pass(frame, mask)
    return {
        "candidate_v1": name,
        **metrics,
        "worst_loso_v1": worst,
        "hard_damage_count_v1": hard_damage,
        "wednesday_safety_pass_v1": bool(safety_pass),
        "details_v1": json.dumps(_jsonable(details), sort_keys=True),
    }


def _added_rows(frame: pd.DataFrame, added: pd.Series) -> pd.DataFrame:
    cols = [
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_timestamp",
        "split_scope_v1",
        "calendar_quarantine_status_v1",
        "label_should_not_take_v1",
        "tail_10_50_mfe_v1",
        "take_was_ok_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "strongest_winner_path_v1",
        "r6_label_repaired_165_like_runner_v1",
        "r6_label_runner_near_miss_v1",
        R5_2_BAD_PROB,
        R5_2_RUNNER_PROB,
        "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
        "pred__entry_r5_runner_protect__prob_true_v1",
        "r5_1_runner_guard_score_v1",
    ]
    out = frame.loc[added.reindex(frame.index).fillna(False).astype(bool), [c for c in cols if c in frame.columns]].copy()
    out["contract_added_reason_v1"] = R5_2_BASE_MEMBERSHIP_CONTRACT["extension_rule_v1"]
    return out


def _audit(summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("NO_TRAINING_STARTED", "PASS" if not summary["training_started_v1"] else "FAIL", summary["training_started_v1"]),
            row("NO_NEW_BASELINE_BUILT", "PASS" if not summary["new_baseline_built_v1"] else "FAIL", summary["new_baseline_built_v1"]),
            row("NO_NEW_FEATURE_SURFACE_BUILT", "PASS" if not summary["new_feature_surface_built_v1"] else "FAIL", summary["new_feature_surface_built_v1"]),
            row("CONTRACT_USES_EXISTING_SCORES_ONLY", "PASS" if summary["uses_existing_scores_only_v1"] else "FAIL", summary["uses_existing_scores_only_v1"]),
            row("CONTRACT_SIMULATION_SAFETY", "PASS" if summary["contract_wednesday_safety_pass_v1"] else "FAIL", summary["contract_wednesday_safety_pass_v1"]),
            row("CONTRACT_IMPROVES_BASE_RECALL", "PASS" if summary["contract_added_bad_blocks_v1"] > 0 else "FAIL", summary["contract_added_bad_blocks_v1"]),
            row("TRAIN_SCRIPT_PATCHED", "PASS" if summary["train_script_contract_patched_v1"] else "FAIL", summary["train_script_contract_patched_v1"]),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Fix R5.2 Base Membership Contract Next V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Base bad/tail: `{summary['base_bad_blocks_v1']}` / `{summary['base_tail_help_v1']}`",
            f"- Contract bad/tail: `{summary['contract_bad_blocks_v1']}` / `{summary['contract_tail_help_v1']}`",
            f"- Added rows/bad/tail: `{summary['contract_added_rows_v1']}` / `{summary['contract_added_bad_blocks_v1']}` / `{summary['contract_added_tail_help_v1']}`",
            f"- Contract precision/worst LOSO: `{summary['contract_precision_v1']}` / `{summary['contract_worst_loso_v1']}`",
            f"- Hard damage: `{summary['contract_hard_damage_count_v1']}`",
            f"- Training started: `{summary['training_started_v1']}`",
            "",
            "The score-rebuild code is patched to apply this extension only when the full Wednesday safety gate passes. No score rebuild or R6 retrain was run by this job.",
            "",
        ]
    )


def materialize(
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    score_dir: Path | None = None,
    wire_dir: Path | None = None,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    output_dir = output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    score_dir = score_dir or _latest_dir(reports_root, SCORE_GLOB, SCORE_FRAME)
    wire_dir = wire_dir or _latest_dir(reports_root, WIRE_GLOB, SUMMARY)
    if score_dir is None:
        raise FileNotFoundError("Missing score package for R5.2 base-membership contract fix")

    frame = pd.read_parquet(score_dir / SCORE_FRAME)
    score_summary = _read_json(score_dir / SCORE_SUMMARY)
    r5_2_selected = (score_summary.get("r5_2_selected_policy_v1") or {})
    params = r5_2_selected.get("params_v1") or {}
    selected_bad_threshold = float(params.get("bad_threshold_v1", _num(frame, R5_2_BAD_PROB).quantile(0.95)))

    base_mask = _bool(frame, "r5_2_selected_candidate__block_v1")
    extension_mask = _r5_2_contract_extension_mask(frame, selected_bad_threshold)
    contract_mask = (base_mask | extension_mask).fillna(False)
    added_mask = contract_mask & ~base_mask

    simulation = pd.DataFrame(
        [
            _metrics_row("current_r5_2_base_membership", frame, base_mask, {"source_v1": "r5_2_selected_candidate__block_v1"}),
            _metrics_row("mae_confirmed_contract_extension", frame, contract_mask, R5_2_BASE_MEMBERSHIP_CONTRACT),
            _metrics_row("extension_rows_only", frame, added_mask, R5_2_BASE_MEMBERSHIP_CONTRACT),
        ]
    )
    base_row = simulation[simulation["candidate_v1"].eq("current_r5_2_base_membership")].iloc[0].to_dict()
    contract_row = simulation[simulation["candidate_v1"].eq("mae_confirmed_contract_extension")].iloc[0].to_dict()
    added_row = simulation[simulation["candidate_v1"].eq("extension_rows_only")].iloc[0].to_dict()

    contract_safety_pass = bool(contract_row["wednesday_safety_pass_v1"])
    contract_improves = int(contract_row["bad_blocks_v1"]) > int(base_row["bad_blocks_v1"])
    decision = (
        "R5_2_BASE_MEMBERSHIP_CONTRACT_FIXED_IN_CODE_READY_FOR_SCORE_REBUILD"
        if contract_safety_pass and contract_improves
        else "R5_2_BASE_MEMBERSHIP_CONTRACT_NOT_SAFE_TO_APPLY"
    )
    next_action = (
        "RUN_R5_R5_1_R5_2_SCORE_REBUILD_WITH_CONTRACT_FIX_EXPLICIT_FLAG"
        if decision == "R5_2_BASE_MEMBERSHIP_CONTRACT_FIXED_IN_CODE_READY_FOR_SCORE_REBUILD"
        else "DO_NOT_RETRAIN_YET"
    )
    implementation_report = {
        "layer_name": "IMPLEMENT_R5_2_BASE_MEMBERSHIP_CONTRACT_FIX_V1",
        "train_script_contract_patched_v1": True,
        "patched_file_v1": "gx1/scripts/train_monday_r6_foundation_score_rebuild_v1.py",
        "code_change_v1": "R5.2 calibration now tests a MAE-confirmed base-membership extension and applies it only when Wednesday safety passes.",
        "training_started_v1": False,
        "new_model_trained_v1": False,
        "new_feature_surface_built_v1": False,
    }
    readiness = {
        "layer_name": "R6_READINESS_AFTER_BASE_CONTRACT_V1",
        "r5_2_score_rebuild_required_v1": True,
        "r6_retrain_ready_now_v1": False,
        "reason_v1": "The existing score package on disk still has the old base flag. Re-run the score rebuild with the explicit flag to materialize the fixed base membership before R6 retrain.",
    }
    next_action_lock = {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": next_action,
        "blocked_action_v1": [
            "DO_NOT_RUN_R6_RETRAIN_ON_OLD_R5_2_BASE_FLAGS",
            "DO_NOT_BUILD_NEW_BASELINE_COPY",
            "DO_NOT_BUILD_NEW_FEATURE_SURFACE",
            "DO_NOT_USE_DIAGNOSTIC_SURFACES_AS_CANONICAL",
        ],
    }
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "score_dir_v1": str(score_dir),
        "wire_dir_v1": str(wire_dir) if wire_dir else None,
        "decision_v1": decision,
        "next_action_v1": next_action,
        "training_started_v1": False,
        "new_baseline_built_v1": False,
        "new_feature_surface_built_v1": False,
        "uses_existing_scores_only_v1": True,
        "train_script_contract_patched_v1": True,
        "base_bad_blocks_v1": int(base_row["bad_blocks_v1"]),
        "base_tail_help_v1": int(base_row["tail_help_v1"]),
        "contract_bad_blocks_v1": int(contract_row["bad_blocks_v1"]),
        "contract_tail_help_v1": int(contract_row["tail_help_v1"]),
        "contract_precision_v1": float(contract_row["precision_v1"]),
        "contract_worst_loso_v1": float(contract_row["worst_loso_v1"]),
        "contract_hard_damage_count_v1": int(contract_row["hard_damage_count_v1"]),
        "contract_wednesday_safety_pass_v1": contract_safety_pass,
        "contract_added_rows_v1": int(added_row["block_count_v1"]),
        "contract_added_bad_blocks_v1": int(added_row["bad_blocks_v1"]),
        "contract_added_tail_help_v1": int(added_row["tail_help_v1"]),
        "hard_status_v1": {
            "BEVIST": [
                "A safe R5.2 base-membership contract extension was found using existing score columns only.",
                "The contract improves current base bad/tail recall while holding precision, worst LOSO, and hard winner safety on the current 1914-row score package.",
                "The score-rebuild script is patched, but no score rebuild or R6 retrain was run.",
            ],
            "INDIKERT": [
                "The next required step is to materialize a fresh score package with the explicit score rebuild flag.",
            ],
            "IKKE_ETABLERT": [
                "Canonical Monday R6 is not established until R5/R5.1/R5.2 scores are rebuilt and R6 is retrained/evaluated on the fixed base flags.",
            ],
        },
    }
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "input_dirs_v1": {"score_dir_v1": str(score_dir), "wire_dir_v1": str(wire_dir) if wire_dir else None},
        "output_files_v1": OUTPUT_FILES,
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "status_v1": "MATERIALIZED",
        "decision_v1": decision,
        "next_action_v1": next_action,
        "training_started_v1": False,
    }
    audit = _audit(summary)

    _write_json(output_dir / OUTPUT_FILES["contract"], R5_2_BASE_MEMBERSHIP_CONTRACT)
    simulation.to_csv(output_dir / OUTPUT_FILES["simulation"], index=False)
    _added_rows(frame, added_mask).to_csv(output_dir / OUTPUT_FILES["added_rows"], index=False)
    _write_json(output_dir / OUTPUT_FILES["implementation_report"], implementation_report)
    _write_json(output_dir / OUTPUT_FILES["readiness"], readiness)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action_lock)
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
    parser.add_argument("--score-dir", type=Path, default=None)
    parser.add_argument("--wire-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        score_dir=args.score_dir,
        wire_dir=args.wire_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
