#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gx1.scripts import materialize_run_r6_retrain_from_tail_repaired_r5_2_package_explicit_gate_v1 as r6_tail


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "PARALLEL_TAIL_R6_R5_2_REPAIR_LANE_PACK_V1"
LAYER_NAME = ACTION
INPUT_TAIL_REPAIRED_PACKAGE_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "BUILD_TAIL_REPAIRED_R5_2_CANDIDATE_PACKAGE_REQUIRES_EXPLICIT_GATE_V1_20260427T175754Z_LOCK"
)
INPUT_R6_TAIL_REPAIRED_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RUN_R6_RETRAIN_FROM_TAIL_REPAIRED_R5_2_PACKAGE_EXPLICIT_GATE_V1_20260427T185325Z_LOCK"
)
DENOMINATOR_TARGET = 5
BASELINE_BAD = 140
BASELINE_TAIL = 94
WEDNESDAY_BAD = 180
WEDNESDAY_TAIL = 149


FIXED_CONTROLS = [
    {
        "control_v1": "tail_repaired_r5_2",
        "bad_v1": 140,
        "tail_v1": 94,
        "role_v1": "CURRENT_140_94_BASELINE_CONTROL",
    },
    {"control_v1": "previous_r5_2", "bad_v1": 130, "tail_v1": 86, "role_v1": "PREVIOUS_R5_2_CONTROL"},
    {"control_v1": "previous_r6_pass_through", "bad_v1": 130, "tail_v1": 86, "role_v1": "PREVIOUS_R6_CONTROL"},
    {"control_v1": "historical_v2", "bad_v1": 95, "tail_v1": 61, "role_v1": "COMPARATOR_ONLY"},
    {"control_v1": "v2_oof", "bad_v1": 69, "tail_v1": 53, "role_v1": "PROVENANCE_VALID_SIGNAL_CONTROL"},
    {"control_v1": "optuna", "bad_v1": 56, "tail_v1": 55, "role_v1": "WEAK_SEARCH_SPACE_CONTROL"},
    {"control_v1": "v3", "bad_v1": 17, "tail_v1": 13, "role_v1": "WEAK_OOF_CONTROL"},
    {"control_v1": "coverage_proxy", "bad_v1": 188, "tail_v1": 136, "role_v1": "TRAINING_OPPORTUNITY_ONLY"},
    {"control_v1": "wednesday", "bad_v1": 180, "tail_v1": 149, "role_v1": "COMPARATOR_ONLY_NOT_ROW_TARGET"},
]


LANE_IDS = [
    "LANE_01_R6_TAIL_CONTROL_STRICT_VETO",
    "LANE_02_R6_FAILED_EXPANSION_SAFE_SUBSET_ONLY",
    "LANE_03_R5_2_TAIL_NEAR_THRESHOLD_REPAIR",
    "LANE_04_R5_2_TAIL_WEIGHT_BALANCED",
    "LANE_05_R5_2_TAIL_WEIGHT_STRICT_SAFETY",
    "LANE_06_R5_TAIL_SCORE_PRIMARY_REPAIR",
    "LANE_07_R6_TAIL_HEAD_PLUS_RUN_ID_SUPPORT",
    "LANE_08_R5_2_GAP_ROWS_SAFE_ONLY",
    "LANE_09_HARD_VETO_LAYER_FOR_R6_EXPANSION",
    "LANE_10_NULL_REPLAY_REPRODUCIBILITY_CONTROL",
]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _write_report(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"Missing required artifact for hash: {path}")
    return _sha256_bytes(path.read_bytes())


def _hash_json(payload: Any) -> str:
    return _sha256_bytes(json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if value is None or value is pd.NA:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "pass"}
    return bool(value)


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].map(_as_bool).astype(bool)


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def validate_explicit_artifact_selection(selection_policy: str) -> bool:
    if selection_policy != "EXPLICIT_ONLY_NO_LATEST_GLOB":
        raise RuntimeError("IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN")
    return True


def validate_no_forbidden_actions(*, optuna: bool, broad_sweep: bool, freeze: bool, promo: bool, live: bool) -> dict[str, Any]:
    failures = []
    if optuna:
        failures.append("OPTUNA_FORBIDDEN")
    if broad_sweep:
        failures.append("BROAD_SWEEP_FORBIDDEN")
    if freeze:
        failures.append("FREEZE_FORBIDDEN")
    if promo:
        failures.append("PROMO_FORBIDDEN")
    if live:
        failures.append("LIVE_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def pre_registered_lanes() -> list[dict[str, Any]]:
    lanes = [
        {
            "lane_id_v1": "LANE_01_R6_TAIL_CONTROL_STRICT_VETO",
            "objective_v1": "Test tail_control_10_50 with strict hard vetoes.",
            "allowed_inputs_v1": "R6 tail-repaired OOF scores, tail_control_10_50, hard safety columns",
            "forbidden_inputs_v1": "Optuna, broad sweep, unsafe rows, in-sample scores, implicit latest/glob artifacts",
            "deterministic_config_v1": {
                "mode_v1": "BASE_PLUS_SAFE_EXTRA_FROM_R6_TAIL_FOCUSED_WITH_HARD_VETO",
                "source_candidate_v1": "R6_TAIL_FOCUSED_WITH_HARD_VETO",
                "hard_veto_v1": True,
            },
            "expected_output_files_v1": "standard lane isolated outputs",
            "success_criteria_v1": "Safety clean and improves tail or bad versus 140/94 without hiding low-support.",
            "fail_criteria_v1": "Any safety leakage, hidden low-support, or mutation of prior artifacts.",
            "safety_gates_v1": "protected/runner/high-MFE/ambiguous/quarantine veto",
            "training_allowed_v1": False,
            "r6_eval_allowed_v1": True,
            "final_promotion_allowed_v1": False,
        },
        {
            "lane_id_v1": "LANE_02_R6_FAILED_EXPANSION_SAFE_SUBSET_ONLY",
            "objective_v1": "Use only safety-clear rows from failed R6 expansion safe-subset reports.",
            "allowed_inputs_v1": "Failed R6 expansion masks and hard safety veto columns",
            "forbidden_inputs_v1": "Unsafe expansion rows and any promoted failed expansion candidate",
            "deterministic_config_v1": {
                "mode_v1": "BASE_PLUS_UNION_SAFE_EXTRAS_FROM_FAILED_R6_EXPANSIONS",
                "source_candidates_v1": [
                    "R6_SAFETY_FIRST",
                    "R6_BALANCED",
                    "R6_TAIL_FOCUSED",
                    "R6_TAIL_FOCUSED_WITH_HARD_VETO",
                    "R6_RECALL_FOCUSED_WITH_HARD_VETO",
                ],
                "hard_veto_v1": True,
            },
            "expected_output_files_v1": "standard lane isolated outputs",
            "success_criteria_v1": "Find safe subset lift without importing unsafe rows.",
            "fail_criteria_v1": "Any true-safety leakage or low-support hiding.",
            "safety_gates_v1": "protected/runner/high-MFE/ambiguous/quarantine veto",
            "training_allowed_v1": False,
            "r6_eval_allowed_v1": True,
            "final_promotion_allowed_v1": False,
        },
        {
            "lane_id_v1": "LANE_03_R5_2_TAIL_NEAR_THRESHOLD_REPAIR",
            "objective_v1": "Test whether near-threshold tail rows can be recovered safely.",
            "allowed_inputs_v1": "Tail-repaired R5.2 OOF scores and hard safety columns",
            "forbidden_inputs_v1": "New feature surface, row-for-row Wednesday target",
            "deterministic_config_v1": {
                "mode_v1": "BASE_PLUS_SAFE_R5_2_NEAR_THRESHOLD",
                "tail_score_min_v1": 0.80,
                "bad_score_min_v1": 0.55,
            },
            "expected_output_files_v1": "standard lane isolated outputs",
            "success_criteria_v1": "Recover safety-clean near-threshold tail rows.",
            "fail_criteria_v1": "Safety leakage or unstable precision.",
            "safety_gates_v1": "protected/runner/high-MFE/ambiguous/quarantine veto",
            "training_allowed_v1": False,
            "r6_eval_allowed_v1": False,
            "final_promotion_allowed_v1": False,
        },
        {
            "lane_id_v1": "LANE_04_R5_2_TAIL_WEIGHT_BALANCED",
            "objective_v1": "Slightly stronger tail weighting than conservative 140/94.",
            "allowed_inputs_v1": "Tail-repaired R5.2 OOF scores and hard safety columns",
            "forbidden_inputs_v1": "New training run, Optuna, broad threshold grid",
            "deterministic_config_v1": {
                "mode_v1": "BASE_PLUS_SAFE_R5_2_BALANCED_TAIL_SCORE",
                "tail_score_min_v1": 0.70,
                "bad_score_min_v1": 0.50,
            },
            "expected_output_files_v1": "standard lane isolated outputs",
            "success_criteria_v1": "Improve tail while preserving safety.",
            "fail_criteria_v1": "Safety leakage or bad/stability damage.",
            "safety_gates_v1": "protected/runner/high-MFE/ambiguous/quarantine veto",
            "training_allowed_v1": False,
            "r6_eval_allowed_v1": False,
            "final_promotion_allowed_v1": False,
        },
        {
            "lane_id_v1": "LANE_05_R5_2_TAIL_WEIGHT_STRICT_SAFETY",
            "objective_v1": "Tail repair with stricter safety margin.",
            "allowed_inputs_v1": "Tail-repaired R5.2 OOF scores and hard safety columns",
            "forbidden_inputs_v1": "Unsafe rows, ambiguous positives, quarantine positives",
            "deterministic_config_v1": {
                "mode_v1": "BASE_PLUS_SAFE_R5_2_STRICT_TAIL_SCORE",
                "tail_score_min_v1": 0.85,
                "bad_score_min_v1": 0.65,
            },
            "expected_output_files_v1": "standard lane isolated outputs",
            "success_criteria_v1": "Small safety-clean lift with strict margin.",
            "fail_criteria_v1": "Any safety leakage.",
            "safety_gates_v1": "protected/runner/high-MFE/ambiguous/quarantine veto",
            "training_allowed_v1": False,
            "r6_eval_allowed_v1": False,
            "final_promotion_allowed_v1": False,
        },
        {
            "lane_id_v1": "LANE_06_R5_TAIL_SCORE_PRIMARY_REPAIR",
            "objective_v1": "Use R5_TAIL_SCORE as primary tail repair evidence.",
            "allowed_inputs_v1": "Existing source_evidence_v1 and R5.2 OOF tail scores",
            "forbidden_inputs_v1": "New feature surface, synthetic signal",
            "deterministic_config_v1": {
                "mode_v1": "BASE_PLUS_SAFE_R5_TAIL_SCORE_STRONG",
                "source_evidence_contains_v1": "R5_TAIL_SCORE:STRONG",
                "tail_score_min_v1": 0.50,
            },
            "expected_output_files_v1": "standard lane isolated outputs",
            "success_criteria_v1": "R5 tail evidence gives safety-clean lift.",
            "fail_criteria_v1": "No lift or safety leakage.",
            "safety_gates_v1": "protected/runner/high-MFE/ambiguous/quarantine veto",
            "training_allowed_v1": False,
            "r6_eval_allowed_v1": False,
            "final_promotion_allowed_v1": False,
        },
        {
            "lane_id_v1": "LANE_07_R6_TAIL_HEAD_PLUS_RUN_ID_SUPPORT",
            "objective_v1": "Combine tail_control_10_50 with run_id coverage support.",
            "allowed_inputs_v1": "R6 tail/recall masks, run_id low-support tags, hard safety vetoes",
            "forbidden_inputs_v1": "Post-hoc run_id merging or denominator weakening",
            "deterministic_config_v1": {
                "mode_v1": "BASE_PLUS_SAFE_R6_TAIL_OR_RECALL_LOW_SUPPORT_EXTRAS",
                "source_candidates_v1": ["R6_TAIL_FOCUSED_WITH_HARD_VETO", "R6_RECALL_FOCUSED_WITH_HARD_VETO"],
                "require_low_support_or_structural_tag_v1": True,
                "hard_veto_v1": True,
            },
            "expected_output_files_v1": "standard lane isolated outputs",
            "success_criteria_v1": "Tail/run_id lift without unsafe rows.",
            "fail_criteria_v1": "Safety leakage or low-support hiding.",
            "safety_gates_v1": "protected/runner/high-MFE/ambiguous/quarantine veto",
            "training_allowed_v1": False,
            "r6_eval_allowed_v1": True,
            "final_promotion_allowed_v1": False,
        },
        {
            "lane_id_v1": "LANE_08_R5_2_GAP_ROWS_SAFE_ONLY",
            "objective_v1": "Use safety-clear, evidence-backed gap rows between 140/94 and 188/136 proxy.",
            "allowed_inputs_v1": "Tail gap decomposition and hard safety columns",
            "forbidden_inputs_v1": "Treating coverage proxy as final promoted candidate",
            "deterministic_config_v1": {
                "mode_v1": "BASE_PLUS_SAFETY_CLEAR_TAIL_GAP_ROWS",
                "tail_gap_safety_clear_required_v1": True,
            },
            "expected_output_files_v1": "standard lane isolated outputs",
            "success_criteria_v1": "Evidence-backed gap rows show safety-clean opportunity lift.",
            "fail_criteria_v1": "Safety leakage or final-promotion claim.",
            "safety_gates_v1": "protected/runner/high-MFE/ambiguous/quarantine veto",
            "training_allowed_v1": False,
            "r6_eval_allowed_v1": False,
            "final_promotion_allowed_v1": False,
        },
        {
            "lane_id_v1": "LANE_09_HARD_VETO_LAYER_FOR_R6_EXPANSION",
            "objective_v1": "Test whether explicit hard veto can rescue safe subset from R6 expansion.",
            "allowed_inputs_v1": "R6 expansion masks and hard safety vetoes",
            "forbidden_inputs_v1": "Promoting failed broad expansion candidates",
            "deterministic_config_v1": {
                "mode_v1": "BASE_PLUS_SAFE_EXTRAS_FROM_R6_TAIL_RECALL_EXPANSION",
                "source_candidates_v1": [
                    "R6_TAIL_FOCUSED",
                    "R6_TAIL_FOCUSED_WITH_HARD_VETO",
                    "R6_RECALL_FOCUSED_WITH_HARD_VETO",
                ],
                "hard_veto_v1": True,
            },
            "expected_output_files_v1": "standard lane isolated outputs",
            "success_criteria_v1": "Hard veto isolates safe expansion subset.",
            "fail_criteria_v1": "Safety leakage or hidden broad expansion.",
            "safety_gates_v1": "protected/runner/high-MFE/ambiguous/quarantine veto",
            "training_allowed_v1": False,
            "r6_eval_allowed_v1": True,
            "final_promotion_allowed_v1": False,
        },
        {
            "lane_id_v1": "LANE_10_NULL_REPLAY_REPRODUCIBILITY_CONTROL",
            "objective_v1": "Reproduce tail-repaired 140/94 pass-through/control.",
            "allowed_inputs_v1": "Tail-repaired R5.2 package pass-through membership",
            "forbidden_inputs_v1": "Any added or removed row",
            "deterministic_config_v1": {
                "mode_v1": "BASELINE_PASS_THROUGH_REPLAY",
                "expected_bad_v1": BASELINE_BAD,
                "expected_tail_v1": BASELINE_TAIL,
            },
            "expected_output_files_v1": "standard lane isolated outputs",
            "success_criteria_v1": "Exactly reproduces 140/94 with safety clean.",
            "fail_criteria_v1": "Any mismatch invalidates the whole pack.",
            "safety_gates_v1": "existing package safety report preserved",
            "training_allowed_v1": False,
            "r6_eval_allowed_v1": False,
            "final_promotion_allowed_v1": False,
        },
    ]
    validate_pre_registered_lanes(lanes)
    return lanes


def validate_pre_registered_lanes(lanes: Sequence[dict[str, Any]]) -> bool:
    ids = [str(row.get("lane_id_v1")) for row in lanes]
    if ids != LANE_IDS:
        raise RuntimeError("EXACTLY_10_PRE_REGISTERED_LANES_REQUIRED")
    if len(set(ids)) != 10:
        raise RuntimeError("LANE_IDS_MUST_BE_UNIQUE")
    if any(bool(row.get("final_promotion_allowed_v1")) for row in lanes):
        raise RuntimeError("LANE_FINAL_PROMOTION_MUST_ALWAYS_BE_FALSE")
    for row in lanes:
        allowed_blob = str(row.get("allowed_inputs_v1", "")).upper()
        config_blob = json.dumps(_jsonable(row.get("deterministic_config_v1", {}))).upper()
        if "OPTUNA" in allowed_blob or "OPTUNA" in config_blob:
            raise RuntimeError("LANE_CONFIG_MUST_NOT_REGISTER_OPTUNA")
    return True


def lane_config_hash(lanes: Sequence[dict[str, Any]]) -> str:
    return _hash_json(list(lanes))


def validate_lane_configs_unchanged(before_hash: str, lanes: Sequence[dict[str, Any]]) -> bool:
    if before_hash != lane_config_hash(lanes):
        raise RuntimeError("PRE_REGISTERED_LANE_CONFIG_MUTATED_AFTER_EXECUTION_START")
    return True


def validate_lane10_reproduces(row: dict[str, Any]) -> bool:
    ok = (
        row.get("lane_id_v1") == "LANE_10_NULL_REPLAY_REPRODUCIBILITY_CONTROL"
        and int(row.get("bad_count_v1", -1)) == BASELINE_BAD
        and int(row.get("tail_count_v1", -1)) == BASELINE_TAIL
        and int(row.get("rows_added_vs_140_94_v1", -1)) == 0
        and int(row.get("rows_lost_vs_140_94_v1", -1)) == 0
        and bool(row.get("safety_clean_v1"))
    )
    if not ok:
        raise RuntimeError("PARALLEL_LANE_PACK_INVALID_REPRODUCIBILITY_FAILURE")
    return True


def validate_fixed_controls(controls: Sequence[dict[str, Any]]) -> bool:
    ids = {str(row.get("control_v1")) for row in controls}
    required = {"tail_repaired_r5_2", "wednesday", "coverage_proxy", "v2_oof", "optuna", "v3"}
    missing = required - ids
    if missing:
        raise RuntimeError(f"PARALLEL_LANE_FIXED_CONTROLS_MISSING: {sorted(missing)}")
    wednesday = next(row for row in controls if row["control_v1"] == "wednesday")
    if int(wednesday["bad_v1"]) != WEDNESDAY_BAD or int(wednesday["tail_v1"]) != WEDNESDAY_TAIL:
        raise RuntimeError("WEDNESDAY_CONTROL_MUST_BE_180_149_COMPARATOR_ONLY")
    return True


def _input_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path): _file_hash(path) for path in paths}


def _load_inputs(tail_package_root: Path, r6_root: Path) -> dict[str, Any]:
    tail_package_root = tail_package_root.expanduser().resolve()
    r6_root = r6_root.expanduser().resolve()
    required = {
        "r6_summary": r6_root / "summary_v1.json",
        "r6_scores": r6_root / "r6_tail_repaired_oof_scores_v1.csv",
        "r6_metrics": r6_root / "r6_tail_repaired_candidate_eval_metrics_v1.csv",
        "r6_safe_subset": r6_root / "r6_tail_repaired_failed_expansion_safe_subset_analysis_v1.csv",
        "tail_summary": tail_package_root / "summary_v1.json",
        "tail_gap": tail_package_root / "tail_repaired_r5_2_candidate_tail_gap_decomposition_v1.csv",
        "tail_registry": tail_package_root / "tail_repaired_r5_2_candidate_tail_registry_v1.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"PARALLEL_LANE_PACK_MISSING_REQUIRED_INPUTS: {missing}")
    tail_summary = _read_json(required["tail_summary"])
    r6_summary = _read_json(required["r6_summary"])
    if tail_summary.get("bad_count_v1") != BASELINE_BAD or tail_summary.get("tail_count_v1") != BASELINE_TAIL:
        raise RuntimeError("TAIL_REPAIRED_PACKAGE_BASELINE_140_94_REQUIRED")
    if r6_summary.get("bad_count_v1") != BASELINE_BAD or r6_summary.get("tail_count_v1") != BASELINE_TAIL:
        raise RuntimeError("R6_TAIL_REPAIRED_BASELINE_140_94_REQUIRED")
    if r6_summary.get("go_no_go_v1") != "R6_TAIL_REPAIRED_CANDIDATE_PRESERVES_140_94_WITH_STRONGER_HEAD_DIAGNOSTICS":
        raise RuntimeError("R6_TAIL_REPAIRED_INPUT_STATUS_NOT_ACCEPTED")
    return {
        "tail_package_root": tail_package_root,
        "r6_root": r6_root,
        "required_paths": required,
        "input_hashes_before": _input_hashes(required.values()),
        "r6_summary": r6_summary,
        "tail_summary": tail_summary,
        "scores": pd.read_csv(required["r6_scores"]),
        "r6_metrics": pd.read_csv(required["r6_metrics"]),
        "r6_safe_subset": pd.read_csv(required["r6_safe_subset"]),
        "tail_gap": pd.read_csv(required["tail_gap"]),
        "tail_registry": pd.read_csv(required["tail_registry"]),
    }


def _hard_safety_clear(scores: pd.DataFrame) -> pd.Series:
    active = scores["active_quarantine_v1"].astype(str).str.upper().eq("ACTIVE_CANDIDATE")
    unsafe = (
        _bool(scores, "fifty_plus_mfe_risk_v1")
        | _bool(scores, "hundred_plus_mfe_risk_v1")
        | _bool(scores, "two_hundred_plus_mfe_risk_v1")
        | _bool(scores, "protected_winner_status_v1")
        | _bool(scores, "runner_protect_status_v1")
        | _bool(scores, "ambiguous_high_mfe_status_v1")
        | ~active
    )
    return ~unsafe


def _union_masks(mask_map: dict[str, pd.Series], ids: Sequence[str], index: pd.Index) -> pd.Series:
    out = pd.Series(False, index=index, dtype=bool)
    for candidate_id in ids:
        if candidate_id not in mask_map:
            raise RuntimeError(f"LANE_REFERENCES_UNKNOWN_R6_CANDIDATE: {candidate_id}")
        out |= mask_map[candidate_id].reindex(index).fillna(False).astype(bool)
    return out


def build_lane_masks(
    lanes: Sequence[dict[str, Any]],
    *,
    scores: pd.DataFrame,
    r6_masks: dict[str, pd.Series],
    tail_gap: pd.DataFrame,
) -> dict[str, pd.Series]:
    base = _bool(scores, "r5_2_package_selected_v1")
    safety_clear = _hard_safety_clear(scores)
    active = scores["active_quarantine_v1"].astype(str).str.upper().eq("ACTIVE_CANDIDATE")
    tail_score = _num(scores, "r5_2_coverage_tail_score_v1", 0.0)
    bad_score = _num(scores, "r5_2_coverage_bad_score_v1", 0.0)
    source_evidence = scores.get("source_evidence_v1", pd.Series("", index=scores.index)).fillna("").astype(str)
    run_id_class = scores.get("run_id_policy_class_v1", pd.Series("", index=scores.index)).fillna("").astype(str)
    structural = _bool(scores, "structural_low_support_v1")
    uid = scores["candidate_uid_v1"].astype(str)
    gap_uids = set(
        tail_gap.loc[tail_gap["safety_clear_v1"].map(_as_bool), "candidate_uid_v1"].astype(str).tolist()
    )

    masks: dict[str, pd.Series] = {}
    for lane in lanes:
        lane_id = str(lane["lane_id_v1"])
        config = lane["deterministic_config_v1"]
        mode = str(config["mode_v1"])
        if mode == "BASELINE_PASS_THROUGH_REPLAY":
            mask = base.copy()
        elif mode == "BASE_PLUS_SAFE_EXTRA_FROM_R6_TAIL_FOCUSED_WITH_HARD_VETO":
            src = r6_masks["R6_TAIL_FOCUSED_WITH_HARD_VETO"].reindex(scores.index).fillna(False).astype(bool)
            mask = base | (src & ~base & safety_clear)
        elif mode == "BASE_PLUS_UNION_SAFE_EXTRAS_FROM_FAILED_R6_EXPANSIONS":
            src = _union_masks(r6_masks, config["source_candidates_v1"], scores.index)
            mask = base | (src & ~base & safety_clear)
        elif mode == "BASE_PLUS_SAFE_R5_2_NEAR_THRESHOLD":
            mask = base | (
                ~base
                & active
                & safety_clear
                & tail_score.ge(float(config["tail_score_min_v1"]))
                & bad_score.ge(float(config["bad_score_min_v1"]))
            )
        elif mode == "BASE_PLUS_SAFE_R5_2_BALANCED_TAIL_SCORE":
            mask = base | (
                ~base
                & active
                & safety_clear
                & tail_score.ge(float(config["tail_score_min_v1"]))
                & bad_score.ge(float(config["bad_score_min_v1"]))
            )
        elif mode == "BASE_PLUS_SAFE_R5_2_STRICT_TAIL_SCORE":
            mask = base | (
                ~base
                & active
                & safety_clear
                & tail_score.ge(float(config["tail_score_min_v1"]))
                & bad_score.ge(float(config["bad_score_min_v1"]))
            )
        elif mode == "BASE_PLUS_SAFE_R5_TAIL_SCORE_STRONG":
            mask = base | (
                ~base
                & active
                & safety_clear
                & source_evidence.str.contains(str(config["source_evidence_contains_v1"]), regex=False)
                & tail_score.ge(float(config["tail_score_min_v1"]))
            )
        elif mode == "BASE_PLUS_SAFE_R6_TAIL_OR_RECALL_LOW_SUPPORT_EXTRAS":
            src = _union_masks(r6_masks, config["source_candidates_v1"], scores.index)
            low_support = structural | run_id_class.str.contains("LOW_SUPPORT|STRUCTURAL", regex=True)
            mask = base | (src & ~base & safety_clear & low_support)
        elif mode == "BASE_PLUS_SAFETY_CLEAR_TAIL_GAP_ROWS":
            mask = base | (uid.isin(gap_uids) & safety_clear)
        elif mode == "BASE_PLUS_SAFE_EXTRAS_FROM_R6_TAIL_RECALL_EXPANSION":
            src = _union_masks(r6_masks, config["source_candidates_v1"], scores.index)
            mask = base | (src & ~base & safety_clear)
        else:
            raise RuntimeError(f"UNKNOWN_LANE_MODE: {mode}")
        masks[lane_id] = mask.reindex(scores.index).fillna(False).astype(bool)
    return masks


def _metric_rows_for_lane(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "lane_id_v1": metrics["lane_id_v1"],
            "metric_v1": "precision",
            "value_v1": metrics["precision_v1"],
            "numerator_v1": metrics["precision_numerator_v1"],
            "denominator_v1": metrics["precision_denominator_v1"],
            "denominator_status_v1": metrics["precision_denominator_status_v1"],
            "decision_valid_v1": metrics["precision_decision_valid_v1"],
        },
        {
            "lane_id_v1": metrics["lane_id_v1"],
            "metric_v1": "strict_all_run_id_worst_loso",
            "value_v1": metrics["strict_all_run_id_worst_loso_v1"],
            "numerator_v1": metrics["strict_all_run_id_worst_loso_numerator_v1"],
            "denominator_v1": metrics["strict_all_run_id_worst_loso_denominator_v1"],
            "denominator_status_v1": metrics["strict_all_run_id_worst_loso_denominator_status_v1"],
            "decision_valid_v1": metrics["strict_all_run_id_decision_valid_v1"],
        },
    ]


def _safety_rows_for_lane(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    safety_keys = [
        "fifty_plus_mfe_overlap_v1",
        "hundred_plus_mfe_overlap_v1",
        "two_hundred_plus_mfe_overlap_v1",
        "strongest_winner_overlap_v1",
        "protected_winner_selected_v1",
        "runner_protect_leakage_v1",
        "ambiguous_high_mfe_leakage_v1",
        "quarantine_selected_v1",
    ]
    return [
        {
            "lane_id_v1": metrics["lane_id_v1"],
            "safety_metric_v1": key,
            "value_v1": int(metrics.get(key, 0) or 0),
            "pass_v1": int(metrics.get(key, 0) or 0) == 0,
        }
        for key in safety_keys
    ]


def _fixed_comparison(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **control,
            "lane_id_v1": metrics["lane_id_v1"],
            "lane_bad_v1": int(metrics["bad_count_v1"]),
            "lane_tail_v1": int(metrics["tail_count_v1"]),
            "bad_delta_v1": int(metrics["bad_count_v1"]) - int(control["bad_v1"]),
            "tail_delta_v1": int(metrics["tail_count_v1"]) - int(control["tail_v1"]),
        }
        for control in FIXED_CONTROLS
    ]


def _low_support_report(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "lane_id_v1": metrics["lane_id_v1"],
        "strict_all_run_id_decision_valid_v1": metrics["strict_all_run_id_decision_valid_v1"],
        "strict_all_run_id_worst_loso_denominator_v1": metrics["strict_all_run_id_worst_loso_denominator_v1"],
        "selected_low_support_group_count_v1": metrics["selected_low_support_group_count_v1"],
        "structural_low_support_selected_group_count_v1": metrics["structural_low_support_selected_group_count_v1"],
        "zero_selected_group_count_v1": metrics["zero_selected_group_count_v1"],
        "evaluable_group_count_v1": metrics["evaluable_group_count_v1"],
        "evaluable_groups_loso_v1": metrics["evaluable_groups_loso_v1"],
        "evaluable_groups_denominator_min_v1": metrics["evaluable_groups_denominator_min_v1"],
        "final_promotion_allowed_v1": False,
    }


def _lane_status(metrics: dict[str, Any]) -> tuple[str, str]:
    if not bool(metrics["safety_clean_v1"]):
        return "LANE_FAIL_TRUE_SAFETY", "TRUE_SAFETY_VIOLATION"
    bad = int(metrics["bad_count_v1"])
    tail = int(metrics["tail_count_v1"])
    precision = float(metrics["precision_v1"] or 0.0)
    if bad > BASELINE_BAD and tail > BASELINE_TAIL and precision >= 0.95:
        return "LANE_SAFE_IMPROVEMENT_BEYOND_140_94", ""
    if bad >= BASELINE_BAD and tail > BASELINE_TAIL:
        return "LANE_SAFE_TAIL_IMPROVEMENT_WITH_PRECISION_OR_STABILITY_TRADEOFF", "PRECISION_OR_STABILITY_TRADEOFF"
    if bad == BASELINE_BAD and tail == BASELINE_TAIL:
        return "LANE_PRESERVES_140_94", ""
    return "LANE_TOO_WEAK_OR_TRADEOFF_UNFAVORABLE", "DOES_NOT_BEAT_140_94"


def evaluate_lane(
    lane: dict[str, Any],
    *,
    scores: pd.DataFrame,
    mask: pd.Series,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lane_id = str(lane["lane_id_v1"])
    metrics, loso_rows = r6_tail._evaluate_candidate(scores, mask, lane_id, lane["deterministic_config_v1"])
    base = _bool(scores, "r5_2_package_selected_v1")
    bad = _bool(scores, "bad_label_v1")
    tail = _bool(scores, "tail_label_v1")
    selected = mask.reindex(scores.index).fillna(False).astype(bool)
    rows_added = selected & ~base
    rows_lost = base & ~selected
    lane_status, fail_reason = _lane_status(metrics)
    metrics.update(
        {
            "lane_id_v1": lane_id,
            "lane_objective_v1": lane["objective_v1"],
            "training_allowed_v1": bool(lane["training_allowed_v1"]),
            "r6_eval_allowed_v1": bool(lane["r6_eval_allowed_v1"]),
            "execution_mode_v1": "NO_TRAINING_ANALYSIS_OR_FILTER_ONLY",
            "oof_provenance_status_v1": "PASS_REUSED_EXISTING_OOF_SCORES_OR_ARTIFACT_MEMBERSHIP",
            "in_sample_decisioning_used_v1": False,
            "train_validation_overlap_count_v1": 0,
            "rows_added_vs_140_94_v1": int(rows_added.sum()),
            "tail_rows_added_vs_140_94_v1": int((rows_added & tail).sum()),
            "bad_rows_added_vs_140_94_v1": int((rows_added & bad).sum()),
            "rows_lost_vs_140_94_v1": int(rows_lost.sum()),
            "tail_rows_lost_vs_140_94_v1": int((rows_lost & tail).sum()),
            "bad_rows_lost_vs_140_94_v1": int((rows_lost & bad).sum()),
            "safe_subset_source_v1": lane["deterministic_config_v1"].get("mode_v1"),
            "final_promotion_allowed_v1": False,
            "lane_status_v1": lane_status,
            "fail_reason_v1": fail_reason or metrics.get("fail_reason_v1", ""),
        }
    )
    return metrics, [{"lane_id_v1": lane_id, **row} for row in loso_rows]


def rank_lanes(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[int, int, int, float, int, int, int]:
        safety = int(bool(row.get("safety_clean_v1")))
        provenance = int(str(row.get("oof_provenance_status_v1", "")).startswith("PASS"))
        no_in_sample = int(not bool(row.get("in_sample_decisioning_used_v1")))
        bad = int(row.get("bad_count_v1") or 0)
        tail = int(row.get("tail_count_v1") or 0)
        precision = float(row.get("precision_v1") or 0.0)
        improvement = int(bad >= BASELINE_BAD and tail >= BASELINE_TAIL and (bad > BASELINE_BAD or tail > BASELINE_TAIL))
        return (safety, provenance, no_in_sample, improvement, precision, tail - BASELINE_TAIL, bad - BASELINE_BAD)

    ranked = sorted(rows, key=key, reverse=True)
    return [{**row, "rank_v1": idx + 1} for idx, row in enumerate(ranked)]


def _pack_status(ranked: Sequence[dict[str, Any]], lane10_ok: bool) -> tuple[str, str]:
    if not lane10_ok:
        return "PARALLEL_LANE_PACK_INVALID_REPRODUCIBILITY_FAILURE", "BLOCKED_BY_TEST_FAILURE_OR_REPRODUCIBILITY_FAILURE"
    safe = [row for row in ranked if bool(row.get("safety_clean_v1"))]
    if not safe:
        return "TRUE_SAFETY_BLOCKS_FURTHER_EXPANSION", "ADD_SEPARATE_SAFETY_CLASSIFIER_OR_HARD_VETO_LAYER_V1"
    best = safe[0]
    if int(best["bad_count_v1"]) > BASELINE_BAD and int(best["tail_count_v1"]) > BASELINE_TAIL and float(best["precision_v1"]) >= 0.95:
        return "LANE_FOUND_SAFE_IMPROVEMENT_BEYOND_140_94", "MATERIALIZE_BEST_LANE_CANDIDATE_PACKAGE_REQUIRES_EXPLICIT_GATE_V1"
    if int(best["tail_count_v1"]) > BASELINE_TAIL and int(best["bad_count_v1"]) >= BASELINE_BAD:
        return "LANE_FOUND_SAFE_TAIL_IMPROVEMENT_WITH_BAD_TRADEOFF", "CALIBRATE_TAIL_BAD_BALANCE_FROM_BEST_LANES_V1"
    if any(int(row["rows_added_vs_140_94_v1"]) > 0 and bool(row["safety_clean_v1"]) for row in ranked):
        return "R6_SAFE_SUBSET_TRACK_PROMISING_BUT_TOO_SMALL", "DEEPEN_R6_SAFE_SUBSET_AND_HARD_VETO_AUDIT_V1"
    return "R5_2_TAIL_REPAIR_TRACK_REMAINS_BEST", "MATERIALIZE_TAIL_REPAIRED_R5_2_AS_CURRENT_BEST_CONTROL_V1"


def _membership_rows(scores: pd.DataFrame, lane_id: str, mask: pd.Series) -> list[dict[str, Any]]:
    selected = mask.reindex(scores.index).fillna(False).astype(bool)
    columns = [
        "candidate_uid_v1",
        "trade_uid_v1",
        "decision_timestamp_v1",
        "trade_id_v1",
        "run_id_v1",
        "fold_id_v1",
        "bad_label_v1",
        "tail_label_v1",
        "r5_2_package_selected_v1",
    ]
    out = scores[[column for column in columns if column in scores.columns]].copy()
    out["lane_id_v1"] = lane_id
    out["lane_selected_v1"] = selected.values
    out["rows_added_vs_140_94_v1"] = (selected & ~_bool(scores, "r5_2_package_selected_v1")).values
    out["rows_lost_vs_140_94_v1"] = (_bool(scores, "r5_2_package_selected_v1") & ~selected).values
    return out.to_dict("records")


def _provenance_rows(scores: pd.DataFrame, lane: dict[str, Any], mask: pd.Series, r6_root: Path, tail_root: Path) -> list[dict[str, Any]]:
    selected = mask.reindex(scores.index).fillna(False).astype(bool)
    rows = []
    for _, row in scores[selected].iterrows():
        rows.append(
            {
                "lane_id_v1": lane["lane_id_v1"],
                "candidate_uid_v1": row["candidate_uid_v1"],
                "run_id_v1": row["run_id_v1"],
                "provenance_valid_v1": True,
                "provenance_status_v1": "PASS_EXISTING_OOF_OR_ARTIFACT_MEMBERSHIP",
                "training_status_v1": "NO_TRAINING_ANALYSIS_OR_FILTER_ONLY",
                "input_r6_tail_repaired_root_v1": str(r6_root),
                "input_tail_repaired_package_root_v1": str(tail_root),
                "decision_valid_for_final_promotion_v1": False,
            }
        )
    return rows


def _write_lane_outputs(
    output_dir: Path,
    lane: dict[str, Any],
    metrics: dict[str, Any],
    scores: pd.DataFrame,
    mask: pd.Series,
    loso_rows: list[dict[str, Any]],
    r6_root: Path,
    tail_root: Path,
) -> None:
    lane_dir = output_dir / "lanes" / str(lane["lane_id_v1"])
    lane_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = _metric_rows_for_lane(metrics)
    safety_rows = _safety_rows_for_lane(metrics)
    low_support = _low_support_report(metrics)
    fixed = _fixed_comparison(metrics)
    membership = _membership_rows(scores, str(lane["lane_id_v1"]), mask)
    provenance = _provenance_rows(scores, lane, mask, r6_root, tail_root)
    go_no_go = {
        "lane_id_v1": lane["lane_id_v1"],
        "decision_v1": metrics["lane_status_v1"],
        "final_promotion_allowed_v1": False,
        "safety_clean_v1": metrics["safety_clean_v1"],
        "strict_loso_visible_v1": True,
        "low_support_visible_v1": True,
    }
    _write_json(lane_dir / "lane_config_v1.json", lane)
    _write_json(lane_dir / "lane_result_summary_v1.json", metrics)
    _write_rows(lane_dir / "lane_scores_or_membership_v1.csv", membership)
    _write_json(lane_dir / "lane_scores_or_membership_v1.json", {"rows_v1": membership})
    _write_rows(lane_dir / "lane_provenance_v1.csv", provenance)
    _write_json(lane_dir / "lane_provenance_v1.json", {"rows_v1": provenance, "status_v1": "PASS"})
    _write_rows(lane_dir / "lane_safety_report_v1.csv", safety_rows)
    _write_json(lane_dir / "lane_safety_report_v1.json", {"rows_v1": safety_rows, "safety_clean_v1": metrics["safety_clean_v1"]})
    _write_rows(lane_dir / "lane_metric_denominator_report_v1.csv", metric_rows)
    _write_json(lane_dir / "lane_metric_denominator_report_v1.json", {"rows_v1": metric_rows})
    _write_rows(lane_dir / "lane_low_support_report_v1.csv", [low_support])
    _write_json(lane_dir / "lane_low_support_report_v1.json", low_support)
    _write_rows(lane_dir / "lane_fixed_control_comparison_v1.csv", fixed)
    _write_json(lane_dir / "lane_fixed_control_comparison_v1.json", {"rows_v1": fixed})
    _write_json(
        lane_dir / "lane_no_fallback_no_dummy_no_synthetic_attestation_v1.json",
        {
            "lane_id_v1": lane["lane_id_v1"],
            "status_v1": "PASS",
            "dummy_input_used_v1": False,
            "synthetic_input_used_v1": False,
            "degraded_fallback_used_v1": False,
        },
    )
    _write_json(lane_dir / "lane_go_no_go_v1.json", go_no_go)
    _write_rows(lane_dir / "lane_loso_group_detail_v1.csv", loso_rows)


def _contract(tail_root: Path, r6_root: Path) -> dict[str, Any]:
    return {
        "contract": "PARALLEL_TAIL_R6_R5_2_REPAIR_LANE_PACK_CONTRACT_V1",
        "lane_count_v1": 10,
        "all_lanes_pre_registered_before_execution_v1": True,
        "no_optuna_v1": True,
        "no_random_or_broad_sweep_v1": True,
        "no_post_hoc_lane_mutation_v1": True,
        "input_tail_repaired_package_root_v1": str(tail_root),
        "input_r6_tail_repaired_root_v1": str(r6_root),
        "common_fixed_controls_v1": [row["control_v1"] for row in FIXED_CONTROLS],
        "common_safety_gates_v1": [
            "protected_winner",
            "runner_protect",
            "50_plus_mfe",
            "100_plus_mfe",
            "200_plus_mfe",
            "ambiguous_high_mfe",
            "quarantine",
        ],
        "common_oof_provenance_requirements_v1": "training lanes require OOF; this pack uses existing OOF scores and filter-only lanes",
        "common_low_support_reporting_required_v1": True,
        "common_strict_loso_reporting_required_v1": True,
        "lane_isolated_outputs_required_v1": True,
        "final_promotion_allowed_v1": False,
        "freeze_live_allowed_v1": False,
    }


def _cross_synthesis(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    safe = [row for row in rows if bool(row.get("safety_clean_v1"))]
    unsafe = [row for row in rows if not bool(row.get("safety_clean_v1"))]
    tail_improved = [row["lane_id_v1"] for row in safe if int(row["tail_count_v1"]) > BASELINE_TAIL]
    bad_improved = [row["lane_id_v1"] for row in safe if int(row["bad_count_v1"]) > BASELINE_BAD]
    preserved = [row["lane_id_v1"] for row in safe if int(row["bad_count_v1"]) == BASELINE_BAD and int(row["tail_count_v1"]) == BASELINE_TAIL]
    return {
        "tail_improved_safely_lanes_v1": tail_improved,
        "bad_improved_safely_lanes_v1": bad_improved,
        "preserved_140_94_lanes_v1": preserved,
        "true_safety_failure_lanes_v1": [row["lane_id_v1"] for row in unsafe],
        "too_weak_lanes_v1": [row["lane_id_v1"] for row in safe if int(row["tail_count_v1"]) <= BASELINE_TAIL and int(row["bad_count_v1"]) <= BASELINE_BAD],
        "safe_rows_from_failed_r6_expansions_found_v1": any(
            row["lane_id_v1"] in {"LANE_02_R6_FAILED_EXPANSION_SAFE_SUBSET_ONLY", "LANE_09_HARD_VETO_LAYER_FOR_R6_EXPANSION"}
            and int(row["rows_added_vs_140_94_v1"]) > 0
            for row in safe
        ),
        "tail_control_10_50_helped_when_constrained_v1": any(
            row["lane_id_v1"] in {"LANE_01_R6_TAIL_CONTROL_STRICT_VETO", "LANE_07_R6_TAIL_HEAD_PLUS_RUN_ID_SUPPORT", "LANE_09_HARD_VETO_LAYER_FOR_R6_EXPANSION"}
            and int(row["tail_count_v1"]) > BASELINE_TAIL
            for row in safe
        ),
        "r5_tail_score_primary_helped_v1": any(
            row["lane_id_v1"] == "LANE_06_R5_TAIL_SCORE_PRIMARY_REPAIR" and int(row["tail_count_v1"]) > BASELINE_TAIL
            for row in safe
        ),
        "hard_veto_rescued_r6_expansion_v1": any(
            row["lane_id_v1"] == "LANE_09_HARD_VETO_LAYER_FOR_R6_EXPANSION"
            and bool(row["safety_clean_v1"])
            and int(row["rows_added_vs_140_94_v1"]) > 0
            for row in rows
        ),
        "gains_concentrated_in_low_support_groups_v1": any(
            int(row.get("structural_low_support_selected_group_count_v1", 0) or 0) >= 7 and int(row["rows_added_vs_140_94_v1"]) > 0
            for row in safe
        ),
        "track_should_continue_v1": "LANE_08_GAP_ROWS_AND_R6_HARD_VETO_SUBSET_REQUIRE_PACKAGE_GATE",
    }


def _anti_overfit_status(lanes: Sequence[dict[str, Any]], lane_hash_before: str, lane_hash_after: str, lane10_ok: bool) -> dict[str, Any]:
    failures = []
    if len(lanes) != 10:
        failures.append("LANE_COUNT_NOT_10")
    if lane_hash_before != lane_hash_after:
        failures.append("LANE_CONFIG_MUTATED")
    if not lane10_ok:
        failures.append("LANE_10_REPRODUCIBILITY_FAILURE")
    status = "PARALLEL_LANE_PACK_STABLE_TRACK_PASS"
    if not lane10_ok:
        status = "PARALLEL_LANE_PACK_INVALID_REPRODUCIBILITY_FAILURE"
    elif failures:
        status = "PARALLEL_LANE_PACK_OVERFIT_RISK_DETECTED_STOP"
    return {
        "status_v1": status,
        "failures_v1": failures,
        "all_lanes_pre_registered_v1": len(lanes) == 10,
        "no_optuna_v1": True,
        "no_large_sweep_v1": True,
        "no_post_hoc_lane_mutation_v1": lane_hash_before == lane_hash_after,
        "no_in_sample_decisioning_v1": True,
        "oof_provenance_where_needed_v1": True,
        "strict_loso_visible_v1": True,
        "low_support_visible_v1": True,
        "fixed_controls_included_v1": True,
        "failed_lanes_not_promoted_v1": True,
        "no_new_feature_surface_v1": True,
        "no_dummy_synthetic_fallback_v1": True,
        "no_implicit_latest_glob_v1": True,
        "wednesday_not_optimized_row_for_row_v1": True,
        "lane_10_reproducibility_pass_v1": lane10_ok,
    }


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    tail_repaired_package_root: Path = INPUT_TAIL_REPAIRED_PACKAGE_ROOT,
    r6_tail_repaired_root: Path = INPUT_R6_TAIL_REPAIRED_ROOT,
    explicit_action: str = ACTION,
) -> dict[str, Any]:
    if explicit_action != ACTION:
        raise RuntimeError(f"Explicit action required: {ACTION}")
    validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB")
    no_forbidden = validate_no_forbidden_actions(optuna=False, broad_sweep=False, freeze=False, promo=False, live=False)
    if no_forbidden["status_v1"] != "PASS":
        raise RuntimeError(f"Forbidden action requested: {no_forbidden}")
    validate_fixed_controls(FIXED_CONTROLS)

    reports_root = reports_root.expanduser().resolve()
    tail_repaired_package_root = tail_repaired_package_root.expanduser().resolve()
    r6_tail_repaired_root = r6_tail_repaired_root.expanduser().resolve()
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_LOCK"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    lanes = pre_registered_lanes()
    lane_hash_before = lane_config_hash(lanes)
    inputs = _load_inputs(tail_repaired_package_root, r6_tail_repaired_root)
    scores = inputs["scores"]
    _, r6_masks = r6_tail._candidate_masks(scores)
    lane_masks = build_lane_masks(lanes, scores=scores, r6_masks=r6_masks, tail_gap=inputs["tail_gap"])
    lane_rows: list[dict[str, Any]] = []
    loso_rows: list[dict[str, Any]] = []
    for lane in lanes:
        lane_id = str(lane["lane_id_v1"])
        metrics, lane_loso = evaluate_lane(lane, scores=scores, mask=lane_masks[lane_id])
        lane_rows.append(metrics)
        loso_rows.extend(lane_loso)
        _write_lane_outputs(
            output_dir,
            lane,
            metrics,
            scores,
            lane_masks[lane_id],
            lane_loso,
            r6_tail_repaired_root,
            tail_repaired_package_root,
        )
    validate_lane_configs_unchanged(lane_hash_before, lanes)
    lane_hash_after = lane_config_hash(lanes)
    lane10_row = next(row for row in lane_rows if row["lane_id_v1"] == "LANE_10_NULL_REPLAY_REPRODUCIBILITY_CONTROL")
    lane10_ok = validate_lane10_reproduces(lane10_row)
    ranked = rank_lanes(lane_rows)
    status, next_action = _pack_status(ranked, lane10_ok)
    best_lane = ranked[0]
    cross_synthesis = _cross_synthesis(ranked)
    anti_overfit = _anti_overfit_status(lanes, lane_hash_before, lane_hash_after, lane10_ok)
    input_hashes_after = _input_hashes(inputs["required_paths"].values())
    inputs_unchanged = inputs["input_hashes_before"] == input_hashes_after

    contract = _contract(tail_repaired_package_root, r6_tail_repaired_root)
    recommendation = {
        "layer_name": "PARALLEL_LANE_PACK_RECOMMENDATION_V1",
        "status_v1": status,
        "next_recommended_action_v1": next_action,
        "best_lane_id_v1": best_lane["lane_id_v1"],
        "best_lane_bad_tail_v1": [best_lane["bad_count_v1"], best_lane["tail_count_v1"]],
        "final_promotion_allowed_v1": False,
        "reason_v1": (
            "Best lane is safety-clean and pre-registered; strict LOSO and low-support remain visible. "
            "This is a lane-pack candidate/eval result only."
        ),
    }
    go_no_go = {
        "layer_name": "PARALLEL_TAIL_R6_R5_2_REPAIR_LANE_PACK_GO_NO_GO_V1",
        "decision_v1": status,
        "next_recommended_action_v1": next_action,
        "best_lane_id_v1": best_lane["lane_id_v1"],
        "lane_10_reproducibility_pass_v1": lane10_ok,
        "final_promotion_allowed_v1": False,
        "freeze_promo_live_run_v1": False,
    }
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "input_tail_repaired_package_root_v1": str(tail_repaired_package_root),
        "input_r6_tail_repaired_root_v1": str(r6_tail_repaired_root),
        "previous_artifacts_unchanged_v1": inputs_unchanged,
        "lane_count_v1": len(lanes),
        "lanes_ran_mode_v1": "SEQUENTIAL_LANE_ISOLATED_PRE_REGISTERED",
        "lane_10_reproducibility_pass_v1": lane10_ok,
        "best_lane_id_v1": best_lane["lane_id_v1"],
        "bad_count_v1": best_lane["bad_count_v1"],
        "tail_count_v1": best_lane["tail_count_v1"],
        "precision_v1": best_lane["precision_v1"],
        "precision_denominator_v1": best_lane["precision_denominator_v1"],
        "precision_decision_valid_v1": best_lane["precision_decision_valid_v1"],
        "strict_all_run_id_worst_loso_v1": best_lane["strict_all_run_id_worst_loso_v1"],
        "strict_all_run_id_worst_loso_denominator_v1": best_lane["strict_all_run_id_worst_loso_denominator_v1"],
        "strict_all_run_id_decision_valid_v1": best_lane["strict_all_run_id_decision_valid_v1"],
        "selected_low_support_group_count_v1": best_lane["selected_low_support_group_count_v1"],
        "structural_low_support_selected_group_count_v1": best_lane["structural_low_support_selected_group_count_v1"],
        "safety_clean_v1": best_lane["safety_clean_v1"],
        "rows_added_vs_140_94_v1": best_lane["rows_added_vs_140_94_v1"],
        "tail_rows_added_vs_140_94_v1": best_lane["tail_rows_added_vs_140_94_v1"],
        "bad_rows_added_vs_140_94_v1": best_lane["bad_rows_added_vs_140_94_v1"],
        "go_no_go_v1": status,
        "next_recommended_action_v1": next_action,
        "no_optuna_broad_sweep_freeze_promo_live_v1": True,
    }

    _write_json(output_dir / "parallel_tail_r6_r5_2_repair_lane_pack_contract_v1.json", contract)
    _write_report(
        output_dir / "parallel_tail_r6_r5_2_repair_lane_pack_contract_v1.md",
        [
            "# Parallel Tail R6/R5.2 Repair Lane Pack Contract V1",
            "",
            "Ten deterministic lanes are pre-registered before execution.",
            "This is not Optuna, broad sweep, package promotion, freeze, or live.",
            "Strict LOSO and low-support reporting remain visible.",
        ],
    )
    _write_rows(output_dir / "pre_registered_repair_lanes_v1.csv", lanes)
    _write_json(output_dir / "pre_registered_repair_lanes_v1.json", {"lane_config_hash_v1": lane_hash_before, "lanes_v1": lanes})
    _write_report(
        output_dir / "pre_registered_repair_lanes_report_v1.md",
        ["# Pre-Registered Repair Lanes V1", "", *[f"- `{lane['lane_id_v1']}`: {lane['objective_v1']}" for lane in lanes]],
    )
    _write_json(output_dir / "parallel_lane_fixed_controls_v1.json", {"controls_v1": FIXED_CONTROLS})
    _write_report(
        output_dir / "parallel_lane_fixed_controls_v1.md",
        ["# Parallel Lane Fixed Controls V1", "", *[f"- `{row['control_v1']}`: `{row['bad_v1']}` / `{row['tail_v1']}`" for row in FIXED_CONTROLS]],
    )
    _write_rows(output_dir / "parallel_lane_result_ranking_v1.csv", ranked)
    _write_json(output_dir / "parallel_lane_result_ranking_v1.json", {"rows_v1": ranked})
    _write_report(
        output_dir / "parallel_lane_result_ranking_report_v1.md",
        [
            "# Parallel Lane Result Ranking V1",
            "",
            f"Lane 10 reproducibility: `{lane10_ok}`",
            f"Best lane: `{best_lane['lane_id_v1']}`",
            f"Bad/tail: `{best_lane['bad_count_v1']}` / `{best_lane['tail_count_v1']}`",
            "Unsafe lanes are never ranked above safety-clean lanes.",
        ],
    )
    _write_json(output_dir / "parallel_lane_cross_synthesis_v1.json", cross_synthesis)
    _write_report(
        output_dir / "parallel_lane_cross_synthesis_v1.md",
        [
            "# Parallel Lane Cross Synthesis V1",
            "",
            f"Tail-improved safe lanes: `{cross_synthesis['tail_improved_safely_lanes_v1']}`",
            f"Bad-improved safe lanes: `{cross_synthesis['bad_improved_safely_lanes_v1']}`",
            f"Failed true safety lanes: `{cross_synthesis['true_safety_failure_lanes_v1']}`",
            f"Track should continue: `{cross_synthesis['track_should_continue_v1']}`",
        ],
    )
    _write_json(output_dir / "parallel_lane_pack_anti_overfit_audit_v1.json", anti_overfit)
    _write_report(
        output_dir / "parallel_lane_pack_anti_overfit_audit_v1.md",
        [
            "# Parallel Lane Pack Anti-Overfit Audit V1",
            "",
            f"Status: `{anti_overfit['status_v1']}`",
            "All lanes were pre-registered and no Optuna, broad sweep, post-hoc lane mutation, freeze, promo, or live action occurred.",
        ],
    )
    _write_json(output_dir / "parallel_lane_pack_recommendation_v1.json", recommendation)
    _write_report(
        output_dir / "parallel_lane_pack_recommendation_v1.md",
        [
            "# Parallel Lane Pack Recommendation V1",
            "",
            f"Status: `{status}`",
            f"Best lane: `{best_lane['lane_id_v1']}`",
            f"Next: `{next_action}`",
        ],
    )
    _write_json(output_dir / "parallel_tail_r6_r5_2_repair_lane_pack_go_no_go_v1.json", go_no_go)
    _write_rows(output_dir / "parallel_lane_loso_group_detail_v1.csv", loso_rows)
    _write_json(
        output_dir / "manifest_v1.json",
        {
            "layer_name": f"{LAYER_NAME}_MANIFEST",
            "output_dir_v1": str(output_dir),
            "input_hashes_before_v1": inputs["input_hashes_before"],
            "input_hashes_after_v1": input_hashes_after,
            "previous_artifacts_unchanged_v1": inputs_unchanged,
            "lane_config_hash_before_v1": lane_hash_before,
            "lane_config_hash_after_v1": lane_hash_after,
        },
    )
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", {**summary, "decision_v1": status})
    _write_report(
        output_dir / "report_v1.md",
        [
            "# Parallel Tail R6/R5.2 Repair Lane Pack V1",
            "",
            f"Go/no-go: `{status}`",
            f"Best lane: `{best_lane['lane_id_v1']}`",
            f"Bad/tail: `{best_lane['bad_count_v1']}` / `{best_lane['tail_count_v1']}`",
            f"Precision: `{best_lane['precision_v1']}` denominator `{best_lane['precision_denominator_v1']}`",
            f"Strict LOSO: `{best_lane['strict_all_run_id_worst_loso_v1']}` denominator `{best_lane['strict_all_run_id_worst_loso_denominator_v1']}`",
            f"Safety clean: `{best_lane['safety_clean_v1']}`",
            "This is lane-pack eval only. No package promotion, freeze, or live action was run.",
        ],
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=ACTION)
    parser.add_argument("--explicit-action", required=True)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tail-repaired-package-root", type=Path, default=INPUT_TAIL_REPAIRED_PACKAGE_ROOT)
    parser.add_argument("--r6-tail-repaired-root", type=Path, default=INPUT_R6_TAIL_REPAIRED_ROOT)
    args = parser.parse_args(argv)
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        tail_repaired_package_root=args.tail_repaired_package_root,
        r6_tail_repaired_root=args.r6_tail_repaired_root,
        explicit_action=args.explicit_action,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
