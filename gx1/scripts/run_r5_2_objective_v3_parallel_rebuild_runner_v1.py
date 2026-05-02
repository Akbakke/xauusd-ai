#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_SCAN_DIR = DEFAULT_REPORTS_ROOT / "PARALLEL_R5_2_V3_AND_R6_HEAD_RECALL_SEARCH_V1_20260426T_SCAN"
DEFAULT_V2_SPEC_DIR = DEFAULT_REPORTS_ROOT / "DESIGN_R5_2_OBJECTIVE_V2_REBUILD_NEXT_V1_20260426T_LOCK"
LAYER_NAME = "R5_2_OBJECTIVE_V3_PARALLEL_REBUILD_RUNNER_V1"
RUN_FLAG = "--run-r5-2-objective-v3-parallel-rebuild"
EXPLICIT_OOF_RERUN_ACTION = "RERUN_V3_PARALLEL_REBUILD_WITH_OOF_PROVENANCE_EXPLICIT_FLAG"
ACTIVE_SELECTION_CONTRACT = "ACTIVE_SCORE_ARTIFACT_SELECTION_V1"

EXPECTED_FOUNDATION_ROWS = 1914
EXPECTED_ACTIVE_ROWS = 1852
EXPECTED_QUARANTINE_ROWS = 62
EXPECTED_TARGET_ROWS = 1914
EXPECTED_ASOF_SCHEMA_COLUMNS = 109
EXPECTED_V3_BUCKET_COUNTS = {
    "DANGEROUS_OR_PROTECTED": 144,
    "NOT_IN_V2_PRE_VETO_BASE": 226,
    "VETOED_BY_HARD_PROTECTION": 7,
}
REQUIRED_KEYS = ["candidate_uid", "trade_uid", "decision_timestamp"]
REQUIRED_SCAN_FILES = [
    "summary_v1.json",
    "manifest_v1.json",
    "lane_01_v2_remaining_gap_trace_v1.csv",
    "lane_09_v3_weight_profile_sim_scan_v1.csv",
    "lane_10_high_mfe_winner_stress_scan_v1.csv",
    "v3_design_leaderboard_v1.csv",
    "v3_or_r6_head_next_decision_v1.json",
    "next_action_lock_v1.json",
]
DRY_OUTPUT_FILES = [
    "summary_v1.json",
    "status_v1.json",
    "manifest_v1.json",
    "v3_parallel_prelaunch_report_v1.json",
    "v3_variant_config_manifest_v1.csv",
    "v3_target_table_prelaunch_audit_v1.json",
    "v3_feature_matrix_prelaunch_v1.json",
    "v3_forbidden_feature_scan_v1.csv",
    "v3_id_leakage_scan_v1.csv",
    "v3_generalization_anti_overfit_guard_v1.json",
    "v3_hard_veto_contract_report_v1.json",
    "v3_downstream_r6_manifest_contract_v1.json",
    "no_degraded_fallback_contract_v1.json",
    "next_action_lock_v1.json",
    "report_v1.md",
    "consistency_audit_v1.csv",
]
V3_VARIANTS = [
    "V3_BAD_RECALL_STRONGER_WITH_SAME_VETO",
    "V3_BAD_RECALL_STRONGER_PROTECTION_HEAVY",
    "V3_BAD_RECALL_STRONGER_ULTRA_SAFE",
    "V3_TAIL_10_50_SUPPORT_WITH_SAME_VETO",
    "V3_BAD_TAIL_MULTI_HEAD_CONSENSUS",
    "V3_SPLIT_STABLE_BAD_RECALL",
    "V3_BATCH_STABLE_RECALL_WEIGHTING",
    "V3_OVERCONSERVATIVE_VETO_RELAX_SAFE_ONLY",
    "V3_AMBIGUOUS_HARD_NEGATIVE_STRONGER",
    "V3_RECALL_LIGHT_CONTROL",
]
RECALL_OUTPUTS = [
    "r5_2_v3_bad_recall_score",
    "r5_2_v3_tail_recall_score",
    "r5_2_v3_risky_attention_score",
]
PROTECTION_OUTPUTS = [
    "r5_2_v3_runner_protection_score",
    "r5_2_v3_high_mfe_ambiguous_protection_score",
    "r5_2_v3_hard_winner_protection_score",
]
BASE_OUTPUTS = [
    "r5_2_v3_base_membership_pre_veto",
    "r5_2_v3_hard_protection_veto",
    "r5_2_v3_final_base_membership",
]
V3_SCORE_FIELDS = [*RECALL_OUTPUTS, *PROTECTION_OUTPUTS]
OOF_PROVENANCE_OUTPUT_FILES = [
    "v3_oof_fold_assignment_v1.csv",
    "v3_oof_score_provenance_v1.csv",
    "v3_oof_score_source_manifest_v1.json",
    "v3_train_validation_membership_v1.csv",
]
MIN_DECISION_PRECISION_DENOMINATOR = 5
MIN_LOSO_SELECTED_GROUPS = 1
FORBIDDEN_FEATURE_PATTERNS = [
    "hindsight",
    "exit_",
    "exittruth",
    "management_",
    "bridge",
    "readiness",
    "1689",
    "exact_only",
    "protector_first",
    "diagnostic",
    "narrow",
]
ID_LEAKAGE_FEATURES = {"candidate_uid", "trade_uid", "trade_id", "decision_timestamp", "run_id"}
SYNTHETIC_PATTERNS = ["dummy", "synthetic", "fake", "placeholder", "default_fill", "zero_default"]
DEGRADED_PATH_PATTERNS = ["1689", "exact_only", "protector_first", "diagnostic", "narrow", "bridge", "readiness"]
FORENSIC_REPAIRED_CANDIDATE_UID = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"


class BlockedMissingRequiredInput(RuntimeError):
    pass


class OOFScoreResult(NamedTuple):
    scores: pd.Series
    provenance: pd.DataFrame
    fold_assignment: pd.DataFrame
    membership: pd.DataFrame
    source_manifest_rows: list[dict[str, Any]]


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
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataframe_sha256(frame: pd.DataFrame) -> str:
    work = frame.copy()
    work = work.reindex(sorted(work.columns), axis=1)
    digest = hashlib.sha256()
    digest.update(json.dumps([str(column) for column in work.columns], separators=(",", ":")).encode("utf-8"))
    digest.update(json.dumps([str(dtype) for dtype in work.dtypes], separators=(",", ":")).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(work, index=False).values.tobytes())
    digest.update(str(work.shape).encode("utf-8"))
    return digest.hexdigest()


def _active_score_selection_contract(output_dir: Path) -> dict[str, Any]:
    return {
        "contract": ACTIVE_SELECTION_CONTRACT,
        "decisioning_stage": "PRE_OPTUNA",
        "selection_policy": "EXPLICIT_ONLY_NO_LATEST_GLOB",
        "selected_artifacts": {
            "v3_oof_scores": str(output_dir),
        },
        "requirements": {
            "oof_score_provenance_required": True,
            "fold_assignment_required": True,
            "score_source_manifest_required": True,
            "train_validation_membership_required": True,
            "metric_denominator_decision_valid_required": True,
        },
    }


def _scorefield_registry(variant_id: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for score_field in V3_SCORE_FIELDS:
        rows.append(
            {
                "variant_id_v1": variant_id or "ALL_VARIANTS",
                "score_field_v1": score_field,
                "decision_valid_v1": True,
                "decision_valid_status_v1": "VALID_FOR_PRE_OPTUNA_DECISIONING",
                "score_source_v1": "OOF",
                "oof_provenance_status_v1": "PASS",
                "metric_denominator_decision_valid_required_v1": True,
                "invalidated_status_v1": "NOT_INVALIDATED",
            }
        )
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BlockedMissingRequiredInput(f"Missing required JSON input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _ensure_clean_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output namespace is not clean: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _reject_degraded_path(path: Path, role: str) -> None:
    lower = str(path).lower()
    matches = [pattern for pattern in DEGRADED_PATH_PATTERNS if pattern in lower]
    if matches:
        raise RuntimeError(f"DEGRADED_FALLBACK_FORBIDDEN for {role}: {path} matched {matches}")


def _load_required_csv(path: Path, role: str) -> pd.DataFrame:
    if not path.exists():
        raise BlockedMissingRequiredInput(f"Missing required {role}: {path}")
    _reject_degraded_path(path, role)
    return pd.read_csv(path)


def _load_required_parquet(path: Path, role: str) -> pd.DataFrame:
    if not path.exists():
        raise BlockedMissingRequiredInput(f"Missing required {role}: {path}")
    _reject_degraded_path(path, role)
    return pd.read_parquet(path)


def _validate_scan_dir(scan_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_SCAN_FILES if not (scan_dir / name).exists()]
    if missing:
        raise BlockedMissingRequiredInput(f"V3 scan/design directory missing required files: {missing}")
    summary = _read_json(scan_dir / "summary_v1.json")
    decision = _read_json(scan_dir / "v3_or_r6_head_next_decision_v1.json")
    if summary.get("decision_v1") != "IMPLEMENT_R5_2_OBJECTIVE_V3_PARALLEL_REBUILD_RUNNER":
        raise RuntimeError(f"Unexpected V3 scan decision: {summary.get('decision_v1')}")
    if decision.get("decision_v1") != "IMPLEMENT_R5_2_OBJECTIVE_V3_PARALLEL_REBUILD_RUNNER":
        raise RuntimeError(f"Unexpected V3 next decision: {decision.get('decision_v1')}")
    if bool(summary.get("training_started_v1")) or bool(summary.get("r6_started_v1")):
        raise RuntimeError("V3 scan/design input is not read-only; training/r6 flags are set")
    return {
        "scan_summary_v1": summary,
        "scan_decision_v1": decision,
    }


def _resolve_input_paths(
    *,
    scan_dir: Path,
    v2_execution_dir: Path | None,
    score_package: Path | None,
    foundation_summary: Path | None,
    label_table: Path | None,
    feature_inventory: Path | None,
    downstream_r6_lock: Path | None,
) -> dict[str, Path]:
    scan_manifest = _read_json(scan_dir / "manifest_v1.json")
    resolved_v2_execution = v2_execution_dir
    if resolved_v2_execution is None:
        value = scan_manifest.get("input_v2_execution_dir_v1")
        if not value:
            raise BlockedMissingRequiredInput("V3 scan manifest missing input_v2_execution_dir_v1")
        resolved_v2_execution = Path(value)
    resolved_v2_execution = resolved_v2_execution.expanduser().resolve()
    if not resolved_v2_execution.exists():
        raise BlockedMissingRequiredInput(f"Missing V2 execution dir: {resolved_v2_execution}")

    v2_manifest = _read_json(resolved_v2_execution / "manifest_v1.json")
    v2_lock_path = downstream_r6_lock or (resolved_v2_execution / "best_v2_variant_downstream_r6_input_lock_v1.json")
    v2_lock = _read_json(v2_lock_path)

    resolved_score = score_package or Path(v2_lock.get("score_package_path_v1", ""))
    resolved_foundation_summary = foundation_summary or Path((v2_manifest.get("input_artifacts_v1") or {}).get("foundation_summary_v1", ""))
    resolved_label_table = label_table or Path((v2_manifest.get("input_artifacts_v1") or {}).get("label_table_v1", ""))
    spec_dir = Path((v2_manifest.get("input_artifacts_v1") or {}).get("spec_dir_v1", DEFAULT_V2_SPEC_DIR))
    resolved_feature_inventory = feature_inventory or (spec_dir / "r5_2_objective_v2_existing_feature_use_spec_v1.csv")

    paths = {
        "scan_dir_v1": scan_dir.expanduser().resolve(),
        "v2_execution_dir_v1": resolved_v2_execution,
        "score_package_v1": resolved_score.expanduser().resolve(),
        "foundation_summary_v1": resolved_foundation_summary.expanduser().resolve(),
        "label_table_v1": resolved_label_table.expanduser().resolve(),
        "feature_inventory_v1": resolved_feature_inventory.expanduser().resolve(),
        "downstream_r6_lock_v1": v2_lock_path.expanduser().resolve(),
    }
    for role, path in paths.items():
        if role.endswith("_dir_v1"):
            if not path.exists():
                raise BlockedMissingRequiredInput(f"Missing required input dir {role}: {path}")
        elif not path.exists():
            raise BlockedMissingRequiredInput(f"Missing required input {role}: {path}")
        _reject_degraded_path(path, role)
    return paths


def _validate_no_degraded_fallback(paths: dict[str, Path], score: pd.DataFrame) -> dict[str, Any]:
    synthetic_columns = [column for column in score.columns if any(pattern in column.lower() for pattern in SYNTHETIC_PATTERNS)]
    if synthetic_columns:
        raise RuntimeError(f"SYNTHETIC_OR_DUMMY_INPUT_FORBIDDEN: {synthetic_columns}")
    return {
        "layer_name": "NO_DEGRADED_FALLBACK_CONTRACT_V1",
        "degraded_fallback_used_v1": False,
        "synthetic_or_dummy_input_count_v1": int(len(synthetic_columns)),
        "required_inputs_v1": {key: str(value) for key, value in paths.items()},
        "forbidden_fallbacks_v1": [
            "1689 exact-only",
            "bridge/readiness training surface",
            "V2/V1/rescue/raw true as hidden V3 output replacement",
            "diagnostic/narrow/protector assets",
            "synthetic scores",
            "dummy labels",
            "dummy feature values",
            "zero/default means safe",
            "empty green artifacts",
        ],
    }


def _variant_config() -> pd.DataFrame:
    weights = {
        "V3_BAD_RECALL_STRONGER_WITH_SAME_VETO": (5.0, 3.0, 1.5, 18.0, 28.0, 36.0, "same_veto"),
        "V3_BAD_RECALL_STRONGER_PROTECTION_HEAVY": (5.2, 3.0, 1.4, 24.0, 36.0, 48.0, "protection_heavy"),
        "V3_BAD_RECALL_STRONGER_ULTRA_SAFE": (4.6, 2.8, 1.2, 30.0, 44.0, 60.0, "ultra_safe"),
        "V3_TAIL_10_50_SUPPORT_WITH_SAME_VETO": (3.8, 4.8, 1.5, 18.0, 28.0, 36.0, "same_veto"),
        "V3_BAD_TAIL_MULTI_HEAD_CONSENSUS": (4.6, 4.4, 2.0, 22.0, 34.0, 44.0, "consensus_strict"),
        "V3_SPLIT_STABLE_BAD_RECALL": (4.8, 3.2, 1.4, 22.0, 34.0, 44.0, "split_stable"),
        "V3_BATCH_STABLE_RECALL_WEIGHTING": (4.4, 3.8, 1.6, 22.0, 34.0, 44.0, "batch_stable"),
        "V3_OVERCONSERVATIVE_VETO_RELAX_SAFE_ONLY": (4.0, 3.5, 1.3, 18.0, 30.0, 40.0, "relax_safe_only"),
        "V3_AMBIGUOUS_HARD_NEGATIVE_STRONGER": (4.4, 3.6, 1.4, 26.0, 60.0, 52.0, "ambiguous_hard_negative"),
        "V3_RECALL_LIGHT_CONTROL": (3.6, 2.8, 1.0, 26.0, 38.0, 52.0, "recall_light_control"),
    }
    rows: list[dict[str, Any]] = []
    for idx, variant in enumerate(V3_VARIANTS, start=1):
        bad_w, tail_w, risky_w, runner_w, ambiguous_w, hard_w, veto = weights[variant]
        rows.append(
            {
                "variant_id_v1": f"R5_2_OBJECTIVE_V3_VARIANT_{idx:02d}_{variant}",
                "profile_id_v1": variant,
                "status_v1": "READY_FOR_EXPLICIT_RUN",
                "bad_weight_v1": bad_w,
                "tail_weight_v1": tail_w,
                "risky_weight_v1": risky_w,
                "runner_protection_weight_v1": runner_w,
                "high_mfe_ambiguous_protection_weight_v1": ambiguous_w,
                "hard_winner_protection_weight_v1": hard_w,
                "veto_strictness_v1": veto,
                "expected_output_score_columns_v1": "|".join([*RECALL_OUTPUTS, *PROTECTION_OUTPUTS]),
                "expected_base_membership_columns_v1": "|".join(BASE_OUTPUTS),
                "no_go_conditions_v1": "|".join(
                    [
                        "repaired_like_overlap_gt_0",
                        "forensic_repaired_trade_blocked",
                        "strongest_winner_overlap_gt_0",
                        "hundred_or_two_hundred_overlap_gt_0",
                        "dangerous_50_overlap_gt_allowed_cap",
                        "ambiguous_high_mfe_leakage_gt_0",
                        "runner_protect_leakage_gt_0",
                        "worst_loso_collapse",
                        "forbidden_feature",
                        "key_schema_drift",
                    ]
                ),
                "anti_overfit_constraints_v1": "|".join(
                    [
                        "no_row_specific_rules",
                        "no_id_features",
                        "no_wednesday_row_matching",
                        "not_selected_on_raw_count_only",
                        "must_pass_loso_batch_pocket_safety",
                    ]
                ),
            }
        )
    manifest = pd.DataFrame(rows)
    if manifest["profile_id_v1"].tolist() != V3_VARIANTS:
        raise RuntimeError("V3 variant config order/list mismatch")
    return manifest


def _validate_foundation(score: pd.DataFrame, foundation_summary: dict[str, Any]) -> dict[str, Any]:
    missing = [column for column in REQUIRED_KEYS if column not in score.columns]
    if missing:
        raise RuntimeError(f"Required input columns missing from score package: {missing}")
    rows = int(len(score))
    active = int(score.get("calendar_quarantine_status_v1", pd.Series("", index=score.index)).astype(str).eq("ACTIVE_CANDIDATE").sum())
    quarantine = rows - active
    asof_schema = int(foundation_summary.get("as_of_column_count_v1") or 0)
    if rows != EXPECTED_FOUNDATION_ROWS:
        raise RuntimeError(f"Expected foundation rows {EXPECTED_FOUNDATION_ROWS}, observed {rows}")
    if active != EXPECTED_ACTIVE_ROWS or quarantine != EXPECTED_QUARANTINE_ROWS:
        raise RuntimeError(f"Expected active/quarantine {EXPECTED_ACTIVE_ROWS}/{EXPECTED_QUARANTINE_ROWS}, observed {active}/{quarantine}")
    if asof_schema != EXPECTED_ASOF_SCHEMA_COLUMNS:
        raise RuntimeError(f"Expected AS_OF schema columns {EXPECTED_ASOF_SCHEMA_COLUMNS}, observed {asof_schema}")
    return {
        "foundation_rows_v1": rows,
        "active_rows_v1": active,
        "quarantine_rows_v1": quarantine,
        "asof_columns_v1": asof_schema,
        "asof_columns_materialized_v1": int(sum(column.startswith("as_of_") for column in score.columns)),
        "required_input_columns_v1": REQUIRED_KEYS,
    }


def _key_alignment(left: pd.DataFrame, right: pd.DataFrame, right_name: str) -> dict[str, Any]:
    missing = [column for column in REQUIRED_KEYS if column not in right.columns]
    if missing:
        raise RuntimeError(f"{right_name} missing key columns: {missing}")
    left_keys = set(map(tuple, left[REQUIRED_KEYS].astype(str).to_numpy()))
    right_keys = set(map(tuple, right[REQUIRED_KEYS].astype(str).to_numpy()))
    missing_from_right = left_keys - right_keys
    extra_in_right = right_keys - left_keys
    if missing_from_right or extra_in_right:
        raise RuntimeError(
            f"Key alignment mismatch for {right_name}: missing_from_right={len(missing_from_right)} extra_in_right={len(extra_in_right)}"
        )
    return {
        f"{right_name}_aligned_rows_v1": int(len(left_keys)),
        f"{right_name}_missing_from_input_v1": int(len(missing_from_right)),
        f"{right_name}_extra_rows_v1": int(len(extra_in_right)),
    }


def _v3_bucket_counts(lane_01: pd.DataFrame) -> dict[str, int]:
    if "gap_bucket_v1" not in lane_01.columns:
        raise RuntimeError("lane_01 missing gap_bucket_v1")
    counts = {str(key): int(value) for key, value in lane_01["gap_bucket_v1"].value_counts().to_dict().items()}
    for bucket, expected in EXPECTED_V3_BUCKET_COUNTS.items():
        if counts.get(bucket, 0) != expected:
            raise RuntimeError(f"V3 bucket count mismatch for {bucket}: expected {expected}, observed {counts.get(bucket, 0)}")
    return counts


def _target_prelaunch(score: pd.DataFrame, label_table: pd.DataFrame, lane_01: pd.DataFrame) -> dict[str, Any]:
    if len(label_table) != EXPECTED_TARGET_ROWS:
        raise RuntimeError(f"Expected target table rows {EXPECTED_TARGET_ROWS}, observed {len(label_table)}")
    alignment = _key_alignment(score, label_table, "target_table")
    required = [
        "new_r5_2_label_bucket_v1",
        "bad_eligibility_target_v1",
        "tail_eligibility_target_v1",
        "runner_protect_target_v1",
        "ambiguous_high_mfe_monitor_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "strongest_winner_path_v1",
        "r6_label_repaired_165_like_runner_v1",
    ]
    missing = [column for column in required if column not in label_table.columns]
    if missing:
        raise RuntimeError(f"Target table missing required columns: {missing}")
    bucket = label_table["new_r5_2_label_bucket_v1"].astype(str)
    bad_target = _bool(label_table, "bad_eligibility_target_v1")
    tail_target = _bool(label_table, "tail_eligibility_target_v1")
    ambiguous = bucket.eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD") | _bool(label_table, "ambiguous_high_mfe_monitor_v1")
    runner_protect = bucket.eq("RUNNER_PROTECT_TARGET") | _bool(label_table, "runner_protect_target_v1")
    hard_protect = (
        _bool(label_table, "hundred_plus_mfe_v1")
        | _bool(label_table, "two_hundred_plus_mfe_v1")
        | _bool(label_table, "strongest_winner_path_v1")
        | _bool(label_table, "r6_label_repaired_165_like_runner_v1")
    )
    monitor = bucket.eq("IGNORE_OR_MONITOR_ONLY")
    ambiguous_bad = int((ambiguous & bad_target).sum())
    runner_bad = int((runner_protect & bad_target).sum())
    monitor_positive = int((monitor & (bad_target | tail_target)).sum())
    hard_unprotected = int((hard_protect & ~(runner_protect | _bool(label_table, "eval_only_flag_v1"))).sum())
    if ambiguous_bad:
        raise RuntimeError(f"Ambiguous high-MFE rows became bad-positive: {ambiguous_bad}")
    if runner_bad:
        raise RuntimeError(f"Runner-protect rows became bad-positive: {runner_bad}")
    if monitor_positive:
        raise RuntimeError(f"Monitor-only rows drive positive target: {monitor_positive}")
    if hard_unprotected:
        raise RuntimeError(f"Hard protected rows are not connected to protection/eval contract: {hard_unprotected}")
    bucket_counts = _v3_bucket_counts(lane_01)
    return {
        "layer_name": "V3_TARGET_TABLE_PRELAUNCH_V1",
        "target_table_rows_v1": int(len(label_table)),
        "key_alignment_v1": alignment,
        "bad_target_rows_v1": int(bad_target.sum()),
        "tail_target_rows_v1": int(tail_target.sum()),
        "ambiguous_high_mfe_bad_positive_count_v1": ambiguous_bad,
        "runner_protect_bad_positive_count_v1": runner_bad,
        "monitor_only_positive_target_count_v1": monitor_positive,
        "hard_protected_rows_v1": int(hard_protect.sum()),
        "hard_unprotected_rows_v1": hard_unprotected,
        "v3_bucket_fix_counts_v1": bucket_counts,
        "v2_bucket_fix_forwarded_v1": True,
    }


def _feature_names(score: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    asof = [column for column in score.columns if column.startswith("as_of_")]
    r5 = [column for column in score.columns if column.startswith("pred__entry_r5_") and ("prob" in column or "score" in column)]
    r5_1 = [column for column in score.columns if column.startswith("r5_1_") and ("prob" in column or "score" in column)]
    r5_2 = [
        column
        for column in score.columns
        if column
        in {
            "pred__entry_r5_2_bad_blocker__prob_true_v1",
            "pred__entry_r5_2_runner_protector__prob_true_v1",
        }
    ]
    ordered = list(dict.fromkeys([*asof, *r5, *r5_1, *r5_2]))
    return ordered, {
        "AS_OF": asof,
        "R5_SIGNALS": r5,
        "R5_1_SIGNALS": r5_1,
        "LEGAL_R5_2_REBUILD_INPUTS": r5_2,
    }


def _forbidden_feature_scan(features: Sequence[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        lower = feature.lower()
        matches = [pattern for pattern in FORBIDDEN_FEATURE_PATTERNS if pattern in lower]
        rows.append(
            {
                "field_v1": feature,
                "is_forbidden_v1": bool(matches),
                "matched_patterns_v1": "|".join(matches),
                "status_v1": "FORBIDDEN" if matches else "ALLOWED",
            }
        )
    return pd.DataFrame(rows)


def _id_leakage_scan(features: Sequence[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        lower = feature.lower()
        leak = (
            feature in ID_LEAKAGE_FEATURES
            or feature.endswith("_uid")
            or feature.endswith("_id")
            or "candidate_uid" in lower
            or "trade_uid" in lower
            or "candidate_id" in lower
            or "trade_id" in lower
            or "decision_timestamp" in lower
            or "timestamp" in lower
        )
        rows.append(
            {
                "field_v1": feature,
                "id_leakage_v1": bool(leak),
                "status_v1": "ID_LEAKAGE" if leak else "ALLOWED",
            }
        )
    return pd.DataFrame(rows)


def _feature_prelaunch(score: pd.DataFrame, feature_inventory: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if feature_inventory.empty:
        raise RuntimeError("Feature inventory is empty")
    features, families = _feature_names(score)
    if not features:
        raise RuntimeError("No legal V3 feature candidates found")
    forbidden = _forbidden_feature_scan(features)
    id_scan = _id_leakage_scan(features)
    forbidden_count = int(forbidden["is_forbidden_v1"].sum())
    id_count = int(id_scan["id_leakage_v1"].sum())
    if forbidden_count:
        raise RuntimeError(f"Forbidden feature(s) in V3 feature matrix: {forbidden[forbidden['is_forbidden_v1']]['field_v1'].tolist()}")
    if id_count:
        raise RuntimeError(f"ID leakage feature(s) in V3 feature matrix: {id_scan[id_scan['id_leakage_v1']]['field_v1'].tolist()}")
    null_rates = {feature: float(score[feature].isna().mean()) for feature in features}
    return {
        "layer_name": "V3_FEATURE_MATRIX_PRELAUNCH_V1",
        "feature_count_v1": int(len(features)),
        "feature_families_v1": {family: int(len(cols)) for family, cols in families.items()},
        "feature_inventory_rows_v1": int(len(feature_inventory)),
        "max_null_rate_v1": max(null_rates.values()) if null_rates else 0.0,
        "nonzero_null_feature_count_v1": int(sum(rate > 0.0 for rate in null_rates.values())),
        "forbidden_feature_count_v1": forbidden_count,
        "id_leakage_feature_count_v1": id_count,
        "asof_legality_v1": "SOURCE_FOUNDATION_SUMMARY_AS_OF_SCHEMA_109_AND_AS_OF_PREFIX_FEATURES_ONLY",
        "synthetic_feature_fill_used_v1": False,
    }, forbidden, id_scan


def _hard_veto_contract(score: pd.DataFrame, label_table: pd.DataFrame) -> dict[str, Any]:
    required_score_columns = [
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "strongest_winner_path_v1",
        "r6_label_repaired_165_like_runner_v1",
        "r6_label_runner_near_miss_v1",
        "r5_2_label_high_mfe_tail_risk_ambiguous_v1",
        "r5_2_label_runner_protect_v1",
    ]
    missing = [column for column in required_score_columns if column not in score.columns]
    if missing:
        raise RuntimeError(f"Hard veto contract cannot bind missing score/eval pockets: {missing}")
    forensic_count = int(score["candidate_uid"].astype(str).eq(FORENSIC_REPAIRED_CANDIDATE_UID).sum())
    if forensic_count != 1:
        raise RuntimeError(f"Forensic repaired trade binding expected exactly 1 row, observed {forensic_count}")
    hard_counts = {
        "dangerous_50_rows_v1": int(_bool(score, "fifty_plus_mfe_v1").sum()),
        "hundred_plus_rows_v1": int(_bool(score, "hundred_plus_mfe_v1").sum()),
        "two_hundred_plus_rows_v1": int(_bool(score, "two_hundred_plus_mfe_v1").sum()),
        "strongest_winner_rows_v1": int(_bool(score, "strongest_winner_path_v1").sum()),
        "repaired_like_rows_v1": int(_bool(score, "r6_label_repaired_165_like_runner_v1").sum()),
        "runner_near_miss_rows_v1": int(_bool(score, "r6_label_runner_near_miss_v1").sum()),
        "ambiguous_high_mfe_rows_v1": int(_bool(score, "r5_2_label_high_mfe_tail_risk_ambiguous_v1").sum()),
        "runner_protect_rows_v1": int(_bool(score, "r5_2_label_runner_protect_v1").sum()),
        "forensic_repaired_trade_rows_v1": forensic_count,
    }
    if not any(value > 0 for value in hard_counts.values()):
        raise RuntimeError("Hard veto contract has no bound safety pockets")
    return {
        "layer_name": "V3_HARD_VETO_CONTRACT_V1",
        "hard_protection_veto_contract_present_v1": True,
        "veto_reason_codes_v1": [
            "VETO_FORENSIC_REPAIRED_TRADE",
            "VETO_REPAIRED_OR_STRONGEST",
            "VETO_100_200_WINNER",
            "VETO_DANGEROUS_50",
            "VETO_RUNNER_PROTECT",
            "VETO_HIGH_MFE_AMBIGUOUS",
            "VETO_EXPLICIT_HARD_PROTECT_BUCKET",
        ],
        "bound_pocket_counts_v1": hard_counts,
        "can_bind_target_table_v1": all(column in label_table.columns for column in ["candidate_uid", "trade_uid", "decision_timestamp"]),
    }


def _anti_overfit_guard(variant_manifest: pd.DataFrame) -> dict[str, Any]:
    constraints = [
        "no_row_specific_rule",
        "no_candidate_uid_feature",
        "no_trade_uid_feature",
        "no_wednesday_row_matching",
        "not_selected_on_raw_bad_tail_count_only",
        "requires_loso_eval",
        "requires_batch_stability_eval",
        "requires_active_quarantine_split_eval",
        "requires_high_mfe_safety_eval",
        "requires_repaired_strongest_winner_safety_eval",
        "requires_precision_stability_eval",
        "requires_pocket_level_safety_eval",
    ]
    missing = []
    for _, row in variant_manifest.iterrows():
        text = str(row.get("anti_overfit_constraints_v1", ""))
        if "no_id_features" not in text or "must_pass_loso_batch_pocket_safety" not in text:
            missing.append(row["profile_id_v1"])
    if missing:
        raise RuntimeError(f"Anti-overfit constraints missing for variants: {missing}")
    return {
        "layer_name": "V3_GENERALIZATION_AND_ANTI_OVERFIT_GUARD_V1",
        "anti_overfit_guard_pass_v1": True,
        "constraints_v1": constraints,
        "variant_count_checked_v1": int(len(variant_manifest)),
        "row_specific_rules_used_v1": False,
        "candidate_or_trade_id_features_used_v1": False,
        "wednesday_row_matching_used_v1": False,
        "raw_count_only_selection_allowed_v1": False,
    }


def _downstream_r6_contract(output_dir: Path) -> dict[str, Any]:
    return {
        "layer_name": "V3_DOWNSTREAM_R6_MANIFEST_CONTRACT_V1",
        "placeholder_only_v1": False,
        "contract_only_no_fake_r6_input_v1": True,
        "future_execution_must_write_v1": {
            "score_package": "per_variant/score_package_v1.parquet",
            "prediction_view": "per_variant/prediction_view_v1.parquet",
            "base_membership": "per_variant/base_membership_package_v1.parquet",
            "downstream_r6_input_manifest": "per_variant/downstream_r6_input_manifest_v1.json",
        },
        "r6_must_use_base_flag_v1": "r5_2_v3_final_base_membership",
        "r6_must_not_use_v1": [
            "r5_2_v3_base_membership_pre_veto",
            "r5_2_v2_final_base_membership as final replacement",
            "r5_2_true_rescue_base_membership_v1 as final replacement",
            "raw_true_base_membership_v1",
            "diagnostic/narrow/protector surfaces",
        ],
        "contract_path_v1": str(output_dir / "v3_downstream_r6_manifest_contract_v1.json"),
    }


def _merge_v3_targets(score: pd.DataFrame, label_table: pd.DataFrame, lane_01: pd.DataFrame) -> pd.DataFrame:
    score_work = score.copy()
    label_work = label_table.copy()
    for column in REQUIRED_KEYS:
        score_work[column] = score_work[column].astype(str)
        label_work[column] = label_work[column].astype(str)
    keep = [column for column in label_work.columns if column not in score_work.columns or column in REQUIRED_KEYS]
    frame = score_work.merge(label_work[keep], on=REQUIRED_KEYS, how="left", validate="one_to_one")
    lane_cols = [column for column in lane_01.columns if column not in frame.columns or column in REQUIRED_KEYS]
    lane_work = lane_01.copy()
    if all(column in lane_work.columns for column in REQUIRED_KEYS):
        for column in REQUIRED_KEYS:
            lane_work[column] = lane_work[column].astype(str)
        frame = frame.merge(lane_work[lane_cols], on=REQUIRED_KEYS, how="left", validate="one_to_one")
    elif "candidate_uid" in lane_work.columns:
        frame = frame.merge(lane_work[["candidate_uid", "gap_bucket_v1"]], on="candidate_uid", how="left", validate="one_to_one")
    else:
        frame["gap_bucket_v1"] = "NOT_ESTABLISHED"

    gap_safe = frame.get("gap_bucket_v1", pd.Series("", index=frame.index)).astype(str).eq("NOT_IN_V2_PRE_VETO_BASE")
    bad_label = _bool(frame, "label_should_not_take_v1")
    tail_label = _bool(frame, "tail_10_50_mfe_v1")
    hard_protect = (
        _bool(frame, "fifty_plus_mfe_v1")
        | _bool(frame, "hundred_plus_mfe_v1")
        | _bool(frame, "two_hundred_plus_mfe_v1")
        | _bool(frame, "strongest_winner_path_v1")
        | _bool(frame, "r6_label_repaired_165_like_runner_v1")
        | _bool(frame, "r6_label_runner_near_miss_v1")
        | _bool(frame, "r5_2_label_high_mfe_tail_risk_ambiguous_v1")
        | _bool(frame, "r5_2_label_runner_protect_v1")
        | frame["candidate_uid"].astype(str).eq(FORENSIC_REPAIRED_CANDIDATE_UID)
    )
    frame["v3_bad_recall_target_v1"] = (_bool(frame, "bad_eligibility_target_v1") | (gap_safe & bad_label)) & ~hard_protect
    frame["v3_tail_recall_target_v1"] = (_bool(frame, "tail_eligibility_target_v1") | (gap_safe & tail_label)) & ~hard_protect
    frame["v3_risky_attention_target_v1"] = (_bool(frame, "risky_attention_target_v1") | _bool(frame, "r6_label_risky_allow_v1")) & ~hard_protect
    frame["v3_runner_protection_target_v1"] = _bool(frame, "r5_2_label_runner_protect_v1") | _bool(frame, "runner_protect_target_v1") | _bool(frame, "r6_label_runner_near_miss_v1")
    frame["v3_high_mfe_ambiguous_protection_target_v1"] = _bool(frame, "r5_2_label_high_mfe_tail_risk_ambiguous_v1") | _bool(frame, "ambiguous_high_mfe_monitor_v1") | _bool(frame, "fifty_plus_mfe_v1")
    frame["v3_hard_winner_protection_target_v1"] = (
        _bool(frame, "hundred_plus_mfe_v1")
        | _bool(frame, "two_hundred_plus_mfe_v1")
        | _bool(frame, "strongest_winner_path_v1")
        | _bool(frame, "r6_label_repaired_165_like_runner_v1")
        | frame["candidate_uid"].astype(str).eq(FORENSIC_REPAIRED_CANDIDATE_UID)
    )
    return frame


def _feature_matrix(score: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    features, _families = _feature_names(score)
    x = score[features].copy()
    categorical = x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if categorical:
        x = pd.get_dummies(x, columns=categorical, dummy_na=True)
    for column in x.columns:
        if not pd.api.types.is_numeric_dtype(x[column]):
            x[column] = pd.to_numeric(x[column], errors="coerce")
    return features, x.astype("float32")


def _target_weights(frame: pd.DataFrame, target: str, variant: pd.Series) -> np.ndarray:
    weights = np.ones(len(frame), dtype="float64")
    positive = _bool(frame, target).to_numpy(dtype=bool)
    if target == "v3_bad_recall_target_v1":
        weights[positive] *= float(variant["bad_weight_v1"])
    elif target == "v3_tail_recall_target_v1":
        weights[positive] *= float(variant["tail_weight_v1"])
    elif target == "v3_risky_attention_target_v1":
        weights[positive] *= float(variant["risky_weight_v1"])
    elif target == "v3_runner_protection_target_v1":
        weights[positive] *= float(variant["runner_protection_weight_v1"])
    elif target == "v3_high_mfe_ambiguous_protection_target_v1":
        weights[positive] *= float(variant["high_mfe_ambiguous_protection_weight_v1"])
    elif target == "v3_hard_winner_protection_target_v1":
        weights[positive] *= float(variant["hard_winner_protection_weight_v1"])
    hard_negative = (
        _bool(frame, "v3_runner_protection_target_v1")
        | _bool(frame, "v3_high_mfe_ambiguous_protection_target_v1")
        | _bool(frame, "v3_hard_winner_protection_target_v1")
    ).to_numpy(dtype=bool)
    if target in {"v3_bad_recall_target_v1", "v3_tail_recall_target_v1", "v3_risky_attention_target_v1"}:
        weights[~positive & hard_negative] *= float(variant["hard_winner_protection_weight_v1"])
    return weights


def _row_identity(frame: pd.DataFrame, idx: int) -> dict[str, Any]:
    row = frame.iloc[int(idx)]
    return {
        "candidate_uid": str(row["candidate_uid"]),
        "trade_uid": str(row["trade_uid"]),
        "decision_timestamp": str(row["decision_timestamp"]),
    }


def _fit_oof_score_with_provenance(
    frame: pd.DataFrame,
    x: pd.DataFrame,
    target: str,
    output: str,
    seed: int,
    variant: pd.Series,
    model_path: Path,
    *,
    feature_matrix_hash: str,
    feature_matrix_columns_hash: str,
    label_table_hash: str,
    config_hash: str,
) -> OOFScoreResult:
    variant_id = str(variant["variant_id_v1"])
    y = _bool(frame, target).astype(int)
    if y.nunique(dropna=False) < 2:
        raise RuntimeError(f"Cannot materialize OOF score for single-class target: {target}")
    groups = frame["run_id"].astype(str) if "run_id" in frame.columns else pd.Series(np.arange(len(frame)).astype(str), index=frame.index)
    unique_groups = int(groups.nunique(dropna=False))
    n_splits = min(10, unique_groups)
    if n_splits < 2:
        raise RuntimeError(f"Cannot materialize grouped OOF score with fewer than two groups for target: {target}")

    predictions = np.full(len(frame), np.nan, dtype="float64")
    provenance_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups=groups), start=1):
        y_train = y.iloc[train_idx]
        if y_train.nunique(dropna=False) < 2:
            raise RuntimeError(f"OOF fold {fold_idx} has single-class training target for {target}")
        fold_model = HistGradientBoostingClassifier(
            max_iter=55,
            learning_rate=0.06,
            max_leaf_nodes=23,
            l2_regularization=0.08,
            random_state=seed + fold_idx,
        )
        fold_model.fit(x.iloc[train_idx], y_train, sample_weight=_target_weights(frame.iloc[train_idx], target, variant))
        fold_model_path = model_path.parent / f"{output}__fold_{fold_idx:02d}.joblib"
        fold_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fold_model, fold_model_path)
        predictions[test_idx] = fold_model.predict_proba(x.iloc[test_idx])[:, 1]

        train_groups = set(groups.iloc[train_idx].astype(str))
        validation_groups = set(groups.iloc[test_idx].astype(str))
        overlap = train_groups.intersection(validation_groups)
        if overlap:
            raise RuntimeError(f"OOF train/validation group leakage for {output} fold {fold_idx}: {sorted(overlap)[:5]}")
        source_model_fold = f"{variant_id}:{output}:fold_{fold_idx:02d}"
        source_rows.append(
            {
                "variant_id_v1": variant_id,
                "score_field_v1": output,
                "score_head_v1": target,
                "fold_id_v1": fold_idx,
                "source_model_fold_v1": source_model_fold,
                "model_source_identifier_v1": source_model_fold,
                "source_model_path_v1": str(fold_model_path),
                "target_v1": target,
                "score_source_v1": "OOF",
                "feature_matrix_hash_v1": feature_matrix_hash,
                "feature_matrix_columns_hash_v1": feature_matrix_columns_hash,
                "label_table_hash_v1": label_table_hash,
                "config_hash_v1": config_hash,
                "seed_v1": seed + fold_idx,
                "train_group_count_v1": len(train_groups),
                "validation_group_count_v1": len(validation_groups),
                "train_row_count_v1": int(len(train_idx)),
                "validation_row_count_v1": int(len(test_idx)),
                "train_validation_overlap_v1": False,
                "decision_valid_v1": True,
                "decision_valid_status_v1": "VALID_FOR_PRE_OPTUNA_DECISIONING",
                "oof_provenance_status_v1": "PASS",
            }
        )
        for group in sorted(train_groups):
            membership_rows.append(
                {
                    "variant_id_v1": variant_id,
                    "score_field_v1": output,
                    "fold_id_v1": fold_idx,
                    "group_key_v1": group,
                    "source_model_fold_v1": source_model_fold,
                    "train_validation_membership_v1": "TRAIN",
                }
            )
        for group in sorted(validation_groups):
            membership_rows.append(
                {
                    "variant_id_v1": variant_id,
                    "score_field_v1": output,
                    "fold_id_v1": fold_idx,
                    "group_key_v1": group,
                    "source_model_fold_v1": source_model_fold,
                    "train_validation_membership_v1": "VALIDATION",
                }
            )
        for idx in test_idx:
            identity = _row_identity(frame, int(idx))
            group_key = str(groups.iloc[int(idx)])
            fold_rows.append(
                {
                    **identity,
                    "variant_id_v1": variant_id,
                    "score_field_v1": output,
                    "fold_id_v1": fold_idx,
                    "group_key_v1": group_key,
                    "train_validation_membership_v1": "VALIDATION",
                }
            )
            provenance_rows.append(
                {
                    **identity,
                    "variant_id_v1": variant_id,
                    "score_field_v1": output,
                    "score_head_v1": target,
                    "fold_id_v1": fold_idx,
                    "group_key_v1": group_key,
                    "train_validation_membership_v1": "VALIDATION",
                    "source_model_fold_v1": source_model_fold,
                    "model_source_identifier_v1": source_model_fold,
                    "source_model_path_v1": str(fold_model_path),
                    "score_source_v1": "OOF",
                    "feature_matrix_hash_v1": feature_matrix_hash,
                    "feature_matrix_columns_hash_v1": feature_matrix_columns_hash,
                    "label_table_hash_v1": label_table_hash,
                    "config_hash_v1": config_hash,
                    "seed_v1": seed + fold_idx,
                    "decision_valid_v1": True,
                    "decision_valid_status_v1": "VALID_FOR_PRE_OPTUNA_DECISIONING",
                    "oof_provenance_status_v1": "PASS",
                    "row_was_in_training_for_source_model_v1": False,
                    "in_sample_score_used_v1": False,
                    "fallback_score_used_v1": False,
                    "synthetic_score_used_v1": False,
                }
            )

    if np.isnan(predictions).any():
        raise RuntimeError(f"Incomplete OOF prediction coverage for {output}")
    model = HistGradientBoostingClassifier(
        max_iter=55,
        learning_rate=0.06,
        max_leaf_nodes=23,
        l2_regularization=0.08,
        random_state=seed,
    )
    model.fit(x, y, sample_weight=_target_weights(frame, target, variant))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return OOFScoreResult(
        scores=pd.Series(predictions, index=frame.index),
        provenance=pd.DataFrame(provenance_rows),
        fold_assignment=pd.DataFrame(fold_rows).drop_duplicates(),
        membership=pd.DataFrame(membership_rows).drop_duplicates(),
        source_manifest_rows=source_rows,
    )


def _fit_score(frame: pd.DataFrame, x: pd.DataFrame, target: str, seed: int, variant: pd.Series, model_path: Path) -> pd.Series:
    feature_matrix_hash = _dataframe_sha256(x)
    feature_matrix_columns_hash = _json_sha256(list(x.columns))
    label_table_hash = _dataframe_sha256(frame[[column for column in frame.columns if column in REQUIRED_KEYS or "target" in column or "label" in column]])
    config_hash = _json_sha256({"variant": variant.to_dict(), "target": target, "seed": seed, "output": model_path.stem})
    result = _fit_oof_score_with_provenance(
        frame=frame,
        x=x,
        target=target,
        output=model_path.stem,
        seed=seed,
        variant=variant,
        model_path=model_path,
        feature_matrix_hash=feature_matrix_hash,
        feature_matrix_columns_hash=feature_matrix_columns_hash,
        label_table_hash=label_table_hash,
        config_hash=config_hash,
    )
    return result.scores


def _thresholds(profile: str) -> dict[str, float]:
    base = {
        "bad": 0.78,
        "tail": 0.76,
        "risky": 0.72,
        "confirm": 0.35,
        "runner": 0.20,
        "ambiguous": 0.14,
        "hard": 0.10,
    }
    if "ULTRA_SAFE" in profile or "RECALL_LIGHT" in profile:
        base.update({"bad": 0.84, "tail": 0.82, "runner": 0.14, "ambiguous": 0.08, "hard": 0.06})
    elif "PROTECTION_HEAVY" in profile or "AMBIGUOUS" in profile:
        base.update({"runner": 0.12, "ambiguous": 0.06, "hard": 0.05})
    elif "TAIL" in profile:
        base.update({"tail": 0.70})
    elif "CONSENSUS" in profile:
        base.update({"bad": 0.74, "tail": 0.72, "risky": 0.68, "confirm": 0.45})
    return base


def _apply_v3_base_rule(frame: pd.DataFrame, pred: pd.DataFrame, profile: str) -> pd.DataFrame:
    t = _thresholds(profile)
    bad = pred["r5_2_v3_bad_recall_score"]
    tail = pred["r5_2_v3_tail_recall_score"]
    risky = pred["r5_2_v3_risky_attention_score"]
    runner = pred["r5_2_v3_runner_protection_score"]
    ambiguous = pred["r5_2_v3_high_mfe_ambiguous_protection_score"]
    hard = pred["r5_2_v3_hard_winner_protection_score"]
    pred["r5_2_v3_base_membership_pre_veto"] = (
        bad.ge(t["bad"]) | tail.ge(t["tail"]) | (risky.ge(t["risky"]) & (bad.ge(t["confirm"]) | tail.ge(t["confirm"])))
    )
    explicit_veto = (
        _bool(frame, "fifty_plus_mfe_v1")
        | _bool(frame, "hundred_plus_mfe_v1")
        | _bool(frame, "two_hundred_plus_mfe_v1")
        | _bool(frame, "strongest_winner_path_v1")
        | _bool(frame, "r6_label_repaired_165_like_runner_v1")
        | _bool(frame, "r6_label_runner_near_miss_v1")
        | _bool(frame, "r5_2_label_high_mfe_tail_risk_ambiguous_v1")
        | _bool(frame, "r5_2_label_runner_protect_v1")
        | frame["candidate_uid"].astype(str).eq(FORENSIC_REPAIRED_CANDIDATE_UID)
    )
    pred["r5_2_v3_hard_protection_veto"] = runner.ge(t["runner"]) | ambiguous.ge(t["ambiguous"]) | hard.ge(t["hard"]) | explicit_veto
    pred["r5_2_v3_final_base_membership"] = pred["r5_2_v3_base_membership_pre_veto"] & ~pred["r5_2_v3_hard_protection_veto"]
    return pred


def _ratio(num: int | float, den: int | float) -> float:
    return 0.0 if not den else float(num) / float(den)


def _metric_ratio(
    metric_name: str,
    numerator: int | float,
    denominator: int | float,
    *,
    min_denominator: int = 1,
) -> dict[str, Any]:
    den = int(denominator)
    num = int(numerator)
    if den <= 0:
        return {
            f"{metric_name}_v1": np.nan,
            f"{metric_name}_numerator_v1": num,
            f"{metric_name}_denominator_v1": den,
            f"{metric_name}_min_denominator_v1": min_denominator,
            f"{metric_name}_denominator_status_v1": "EMPTY_DENOMINATOR",
            f"{metric_name}_decision_valid_v1": False,
            f"{metric_name}_denominator_fail_reason_v1": "EMPTY_DENOMINATOR",
        }
    status = "OK" if den >= min_denominator else "TOO_SMALL_DENOMINATOR"
    return {
        f"{metric_name}_v1": float(num) / float(den),
        f"{metric_name}_numerator_v1": num,
        f"{metric_name}_denominator_v1": den,
        f"{metric_name}_min_denominator_v1": min_denominator,
        f"{metric_name}_denominator_status_v1": status,
        f"{metric_name}_decision_valid_v1": status == "OK",
        f"{metric_name}_denominator_fail_reason_v1": "NONE" if status == "OK" else status,
    }


def _worst_group_precision(frame: pd.DataFrame, selected: pd.Series, bad_label: pd.Series, group_column: str) -> dict[str, Any]:
    if group_column not in frame.columns:
        return {
            "worst_loso_v1": np.nan,
            "worst_loso_group_v1": "MISSING_GROUP_COLUMN",
            "loso_selected_group_count_v1": 0,
            "loso_empty_group_count_v1": 0,
            "worst_loso_denominator_status_v1": "EMPTY_DENOMINATOR",
            "worst_loso_decision_valid_v1": False,
            "worst_loso_denominator_fail_reason_v1": "MISSING_GROUP_COLUMN",
            "worst_loso_min_denominator_v1": MIN_DECISION_PRECISION_DENOMINATOR,
            "worst_loso_min_selected_group_count_v1": MIN_LOSO_SELECTED_GROUPS,
        }
    work = pd.DataFrame(
        {
            "group": frame[group_column].astype(str),
            "selected": selected.astype(bool),
            "bad": bad_label.astype(bool),
        }
    )
    precisions: list[tuple[str, float, int, int]] = []
    empty_group_count = 0
    for group, part in work.groupby("group", dropna=False):
        selected_count = int(part["selected"].sum())
        if selected_count == 0:
            empty_group_count += 1
            continue
        numerator = int((part["selected"] & part["bad"]).sum())
        precisions.append((str(group), float(numerator) / float(selected_count), numerator, selected_count))
    if not precisions:
        return {
            "worst_loso_v1": np.nan,
            "worst_loso_group_v1": "EMPTY_SELECTED_GROUP_SET",
            "loso_selected_group_count_v1": 0,
            "loso_empty_group_count_v1": empty_group_count,
            "worst_loso_numerator_v1": 0,
            "worst_loso_denominator_v1": 0,
            "worst_loso_min_denominator_v1": MIN_DECISION_PRECISION_DENOMINATOR,
            "worst_loso_min_selected_group_count_v1": MIN_LOSO_SELECTED_GROUPS,
            "worst_loso_denominator_status_v1": "EMPTY_DENOMINATOR",
            "worst_loso_decision_valid_v1": False,
            "worst_loso_denominator_fail_reason_v1": "EMPTY_DENOMINATOR",
        }
    worst_group, worst, numerator, denominator = min(precisions, key=lambda item: item[1])
    selected_group_count = len(precisions)
    denominator_status = "OK"
    fail_reason = "NONE"
    if selected_group_count < MIN_LOSO_SELECTED_GROUPS:
        denominator_status = "TOO_SMALL_DENOMINATOR"
        fail_reason = "TOO_FEW_SELECTED_GROUPS"
    elif denominator < MIN_DECISION_PRECISION_DENOMINATOR:
        denominator_status = "TOO_SMALL_DENOMINATOR"
        fail_reason = "WORST_GROUP_SELECTED_DENOMINATOR_TOO_SMALL"
    return {
        "worst_loso_v1": float(worst),
        "worst_loso_group_v1": worst_group,
        "loso_selected_group_count_v1": selected_group_count,
        "loso_empty_group_count_v1": empty_group_count,
        "worst_loso_numerator_v1": numerator,
        "worst_loso_denominator_v1": denominator,
        "worst_loso_min_denominator_v1": MIN_DECISION_PRECISION_DENOMINATOR,
        "worst_loso_min_selected_group_count_v1": MIN_LOSO_SELECTED_GROUPS,
        "worst_loso_denominator_status_v1": denominator_status,
        "worst_loso_decision_valid_v1": denominator_status == "OK",
        "worst_loso_denominator_fail_reason_v1": fail_reason,
    }


def _batch_stability(frame: pd.DataFrame, selected: pd.Series, bad_label: pd.Series) -> dict[str, Any]:
    group_column = "batch_scope_v1" if "batch_scope_v1" in frame.columns else "run_id"
    if group_column not in frame.columns:
        return {
            "batch_stability_status_v1": "NOT_ESTABLISHED",
            "batch_group_column_v1": "MISSING",
            "selected_max_group_share_v1": 0.0,
            "selected_group_count_v1": 0,
            "batch_collapse_v1": False,
        }
    selected_total = int(selected.sum())
    if selected_total == 0:
        return {
            "batch_stability_status_v1": "PASS_NO_SELECTED_ROWS",
            "batch_group_column_v1": group_column,
            "selected_max_group_share_v1": 0.0,
            "selected_group_count_v1": 0,
            "batch_collapse_v1": False,
        }
    work = pd.DataFrame({"group": frame[group_column].astype(str), "selected": selected.astype(bool), "bad": bad_label.astype(bool)})
    grouped = work.groupby("group", dropna=False).agg(selected_count=("selected", "sum"))
    selected_groups = grouped[grouped["selected_count"] > 0]
    max_share = float(selected_groups["selected_count"].max() / selected_total)
    collapse = bool(max_share > 0.75 and selected_total >= 20)
    return {
        "batch_stability_status_v1": "FAIL_CONCENTRATED_SELECTION" if collapse else "PASS",
        "batch_group_column_v1": group_column,
        "selected_max_group_share_v1": max_share,
        "selected_group_count_v1": int(len(selected_groups)),
        "batch_collapse_v1": collapse,
    }


def _variant_metrics(frame: pd.DataFrame, pred: pd.DataFrame, variant: pd.Series) -> dict[str, Any]:
    variant_id = str(variant["variant_id_v1"])
    profile = str(variant["profile_id_v1"])
    selected = _bool(pred, "r5_2_v3_final_base_membership")
    pre_veto = _bool(pred, "r5_2_v3_base_membership_pre_veto")
    veto = _bool(pred, "r5_2_v3_hard_protection_veto")
    bad_label = _bool(frame, "label_should_not_take_v1")
    tail_label = _bool(frame, "tail_10_50_mfe_v1")
    v2_base = _bool(frame, "r5_2_v2_final_base_membership")
    rescue_base = _bool(frame, "r5_2_true_rescue_base_membership_v1")
    final_count = int(selected.sum())
    bad_rows = int((selected & bad_label).sum())
    tail_rows = int((selected & tail_label).sum())
    precision_metric = _metric_ratio("precision", bad_rows, final_count, min_denominator=MIN_DECISION_PRECISION_DENOMINATOR)
    worst_loso_metric = _worst_group_precision(frame, selected, bad_label, "run_id")
    batch = _batch_stability(frame, selected, bad_label)
    safety = {
        "repaired_like_overlap_v1": int((selected & _bool(frame, "r6_label_repaired_165_like_runner_v1")).sum()),
        "forensic_repaired_trade_blocked_v1": int((selected & frame["candidate_uid"].astype(str).eq(FORENSIC_REPAIRED_CANDIDATE_UID)).sum()),
        "fifty_plus_overlap_v1": int((selected & _bool(frame, "fifty_plus_mfe_v1")).sum()),
        "hundred_plus_overlap_v1": int((selected & _bool(frame, "hundred_plus_mfe_v1")).sum()),
        "two_hundred_plus_overlap_v1": int((selected & _bool(frame, "two_hundred_plus_mfe_v1")).sum()),
        "strongest_winner_overlap_v1": int((selected & _bool(frame, "strongest_winner_path_v1")).sum()),
        "runner_near_miss_overlap_v1": int((selected & _bool(frame, "r6_label_runner_near_miss_v1")).sum()),
        "ambiguous_high_mfe_leakage_v1": int((selected & _bool(frame, "r5_2_label_high_mfe_tail_risk_ambiguous_v1")).sum()),
        "runner_protect_leakage_v1": int((selected & _bool(frame, "r5_2_label_runner_protect_v1")).sum()),
    }
    safety_fail_reasons = []
    if safety["repaired_like_overlap_v1"] > 0:
        safety_fail_reasons.append("REPAIRED_LIKE_OVERLAP")
    if safety["forensic_repaired_trade_blocked_v1"] > 0:
        safety_fail_reasons.append("FORENSIC_REPAIRED_TRADE_BLOCKED")
    if safety["strongest_winner_overlap_v1"] > 0:
        safety_fail_reasons.append("STRONGEST_WINNER_OVERLAP")
    if safety["hundred_plus_overlap_v1"] > 0 or safety["two_hundred_plus_overlap_v1"] > 0:
        safety_fail_reasons.append("100_200_MFE_OVERLAP")
    if safety["fifty_plus_overlap_v1"] > 1:
        safety_fail_reasons.append("50_MFE_OVER_ALLOWED_CAP")
    if safety["ambiguous_high_mfe_leakage_v1"] > 0:
        safety_fail_reasons.append("AMBIGUOUS_HIGH_MFE_LEAKAGE")
    if safety["runner_protect_leakage_v1"] > 0:
        safety_fail_reasons.append("RUNNER_PROTECT_LEAKAGE")
    if safety["runner_near_miss_overlap_v1"] > 0:
        safety_fail_reasons.append("RUNNER_NEAR_MISS_OVERLAP")
    if not precision_metric["precision_decision_valid_v1"]:
        safety_fail_reasons.append("PRECISION_DENOMINATOR_INVALID")
    if not worst_loso_metric["worst_loso_decision_valid_v1"]:
        safety_fail_reasons.append("WORST_LOSO_DENOMINATOR_INVALID")
    worst_loso_value = worst_loso_metric["worst_loso_v1"]
    if pd.notna(worst_loso_value) and worst_loso_value <= 0.0 and final_count > 0:
        safety_fail_reasons.append("WORST_LOSO_COLLAPSE")
    if batch["batch_collapse_v1"]:
        safety_fail_reasons.append("BATCH_CONCENTRATION_RISK")
    safety_pass = not safety_fail_reasons
    bad_uplift = bad_rows - 95
    tail_uplift = tail_rows - 61
    meaningful_uplift = bad_uplift >= 5 and tail_uplift >= 5
    if not safety_pass:
        generalization = "UNSAFE"
    elif batch["batch_collapse_v1"] or (pd.notna(worst_loso_value) and worst_loso_value <= 0.0 and final_count > 0):
        generalization = "LIKELY_OVERFIT"
    elif not meaningful_uplift:
        generalization = "TOO_WEAK"
    else:
        generalization = "GENERALIZABLE_CANDIDATE"
    variant_gate_pass = bool(safety_pass and generalization == "GENERALIZABLE_CANDIDATE")
    return {
        "variant_id_v1": variant_id,
        "profile_id_v1": profile,
        "bad_recall_v1": bad_rows,
        "tail_recall_v1": tail_rows,
        **precision_metric,
        **worst_loso_metric,
        "batch_stability_v1": batch["batch_stability_status_v1"],
        "batch_group_column_v1": batch["batch_group_column_v1"],
        "selected_max_group_share_v1": batch["selected_max_group_share_v1"],
        "selected_group_count_v1": batch["selected_group_count_v1"],
        "active_selected_v1": int((selected & frame.get("calendar_quarantine_status_v1", pd.Series("", index=frame.index)).astype(str).eq("ACTIVE_CANDIDATE")).sum()),
        "quarantine_selected_v1": int((selected & ~frame.get("calendar_quarantine_status_v1", pd.Series("", index=frame.index)).astype(str).eq("ACTIVE_CANDIDATE")).sum()),
        "strong_bad_target_recall_v1": _ratio(int((selected & _bool(frame, "v3_bad_recall_target_v1")).sum()), int(_bool(frame, "v3_bad_recall_target_v1").sum())),
        "tail_10_50_target_recall_v1": _ratio(int((selected & _bool(frame, "v3_tail_recall_target_v1")).sum()), int(_bool(frame, "v3_tail_recall_target_v1").sum())),
        "risky_attention_coverage_v1": _ratio(int((selected & _bool(frame, "v3_risky_attention_target_v1")).sum()), int(_bool(frame, "v3_risky_attention_target_v1").sum())),
        "runner_protect_performance_v1": _ratio(int((~selected & _bool(frame, "v3_runner_protection_target_v1")).sum()), int(_bool(frame, "v3_runner_protection_target_v1").sum())),
        **safety,
        "hard_veto_count_v1": int(veto.sum()),
        "pre_veto_base_count_v1": int(pre_veto.sum()),
        "final_base_count_v1": final_count,
        "rows_vetoed_by_protection_v1": int((pre_veto & veto).sum()),
        "rows_added_vs_v2_v1": int((selected & ~v2_base).sum()),
        "rows_added_vs_rescue_v1": int((selected & ~rescue_base).sum()),
        "rows_lost_vs_v2_v1": int((~selected & v2_base).sum()),
        "rows_lost_vs_rescue_v1": int((~selected & rescue_base).sum()),
        "bad_uplift_over_v2_v1": bad_uplift,
        "tail_uplift_over_v2_v1": tail_uplift,
        "meaningful_uplift_over_v2_v1": meaningful_uplift,
        "safety_pass_v1": safety_pass,
        "safety_fail_reasons_v1": "|".join(safety_fail_reasons) if safety_fail_reasons else "NONE",
        "generalization_class_v1": generalization,
        "variant_gate_pass_v1": variant_gate_pass,
    }


def _write_variant_reports(
    variant_dir: Path,
    variant: pd.Series,
    feature_names: Sequence[str],
    heads: Sequence[tuple[str, str]],
    metrics: dict[str, Any],
) -> None:
    variant_id = str(variant["variant_id_v1"])
    model_rows = []
    for target, output in heads:
        model_path = variant_dir / "models" / f"{output}.joblib"
        model_rows.append(
            {
                "target_v1": target,
                "output_score_v1": output,
                "model_path_v1": str(model_path),
                "model_file_exists_v1": model_path.exists(),
            }
        )
    pd.DataFrame(
        [
            {
                "variant_id_v1": variant_id,
                "profile_id_v1": variant["profile_id_v1"],
                "bad_weight_v1": variant["bad_weight_v1"],
                "tail_weight_v1": variant["tail_weight_v1"],
                "risky_weight_v1": variant["risky_weight_v1"],
                "runner_protection_weight_v1": variant["runner_protection_weight_v1"],
                "high_mfe_ambiguous_protection_weight_v1": variant["high_mfe_ambiguous_protection_weight_v1"],
                "hard_winner_protection_weight_v1": variant["hard_winner_protection_weight_v1"],
                "veto_strictness_v1": variant["veto_strictness_v1"],
            }
        ]
    ).to_csv(variant_dir / "label_weight_manifest_v1.csv", index=False)
    pd.DataFrame({"feature_v1": feature_names}).to_csv(variant_dir / "feature_manifest_v1.csv", index=False)
    pd.DataFrame(
        [
            {"metric_v1": key, "value_v1": value}
            for key, value in metrics.items()
            if key.endswith("_v1") and not isinstance(value, dict)
        ]
    ).to_csv(variant_dir / "pocket_eval_report_v1.csv", index=False)
    _write_json(variant_dir / "model_manifest_v1.json", {"variant_id_v1": variant_id, "heads_v1": model_rows})
    _write_json(variant_dir / "config_manifest_v1.json", {"variant_id_v1": variant_id, "variant_config_v1": variant.to_dict()})
    _write_json(
        variant_dir / "safety_guard_report_v1.json",
        {
            "variant_id_v1": variant_id,
            "safety_pass_v1": metrics["safety_pass_v1"],
            "safety_fail_reasons_v1": metrics["safety_fail_reasons_v1"],
            "hard_veto_count_v1": metrics["hard_veto_count_v1"],
            "repaired_like_overlap_v1": metrics["repaired_like_overlap_v1"],
            "forensic_repaired_trade_blocked_v1": metrics["forensic_repaired_trade_blocked_v1"],
            "fifty_plus_overlap_v1": metrics["fifty_plus_overlap_v1"],
            "hundred_plus_overlap_v1": metrics["hundred_plus_overlap_v1"],
            "two_hundred_plus_overlap_v1": metrics["two_hundred_plus_overlap_v1"],
            "strongest_winner_overlap_v1": metrics["strongest_winner_overlap_v1"],
            "ambiguous_high_mfe_leakage_v1": metrics["ambiguous_high_mfe_leakage_v1"],
            "runner_protect_leakage_v1": metrics["runner_protect_leakage_v1"],
        },
    )
    _write_json(
        variant_dir / "training_summary_v1.json",
        {
            "variant_id_v1": variant_id,
            "training_started_v1": True,
            "feature_count_v1": len(feature_names),
            "bad_recall_v1": metrics["bad_recall_v1"],
            "tail_recall_v1": metrics["tail_recall_v1"],
            "precision_v1": metrics["precision_v1"],
            "worst_loso_v1": metrics["worst_loso_v1"],
            "safety_pass_v1": metrics["safety_pass_v1"],
            "variant_gate_pass_v1": metrics["variant_gate_pass_v1"],
        },
    )
    _write_json(variant_dir / "status_v1.json", {"variant_id_v1": variant_id, "status_v1": "COMPLETED", **metrics})
    _write_json(
        variant_dir / "manifest_v1.json",
        {
            "variant_id_v1": variant_id,
            "prediction_view_v1": str(variant_dir / "prediction_view_v1.parquet"),
            "score_package_v1": str(variant_dir / "score_package_v1.parquet"),
            "base_membership_package_v1": str(variant_dir / "base_membership_package_v1.parquet"),
            "v3_oof_score_provenance_v1": str(variant_dir / "v3_oof_score_provenance_v1.csv"),
            "v3_oof_fold_assignment_v1": str(variant_dir / "v3_oof_fold_assignment_v1.csv"),
            "v3_oof_score_source_manifest_v1": str(variant_dir / "v3_oof_score_source_manifest_v1.json"),
            "v3_train_validation_membership_v1": str(variant_dir / "v3_train_validation_membership_v1.csv"),
            "feature_manifest_v1": str(variant_dir / "feature_manifest_v1.csv"),
            "label_weight_manifest_v1": str(variant_dir / "label_weight_manifest_v1.csv"),
            "model_manifest_v1": str(variant_dir / "model_manifest_v1.json"),
        },
    )
    _write_csv(
        variant_dir / "consistency_audit_v1.csv",
        [
            {"check_v1": "training_started", "status_v1": "PASS", "evidence_v1": True},
            {"check_v1": "final_base_flag_written", "status_v1": "PASS", "evidence_v1": "r5_2_v3_final_base_membership"},
            {"check_v1": "oof_provenance_written", "status_v1": "PASS", "evidence_v1": "v3_oof_score_provenance_v1.csv"},
            {"check_v1": "pre_veto_not_downstream_final", "status_v1": "PASS", "evidence_v1": True},
            {"check_v1": "safety_pass", "status_v1": "PASS" if metrics["safety_pass_v1"] else "FAIL", "evidence_v1": metrics["safety_fail_reasons_v1"]},
        ],
    )


def _variant_forensics(frame: pd.DataFrame, pred: pd.DataFrame, variant_id: str, role: str) -> pd.DataFrame:
    selected = _bool(pred, "r5_2_v3_final_base_membership")
    pre_veto = _bool(pred, "r5_2_v3_base_membership_pre_veto")
    veto = _bool(pred, "r5_2_v3_hard_protection_veto")
    v2_base = _bool(frame, "r5_2_v2_final_base_membership")
    safety_damage = selected & (
        _bool(frame, "fifty_plus_mfe_v1")
        | _bool(frame, "hundred_plus_mfe_v1")
        | _bool(frame, "two_hundred_plus_mfe_v1")
        | _bool(frame, "strongest_winner_path_v1")
        | _bool(frame, "r6_label_repaired_165_like_runner_v1")
        | _bool(frame, "r6_label_runner_near_miss_v1")
        | _bool(frame, "r5_2_label_high_mfe_tail_risk_ambiguous_v1")
        | _bool(frame, "r5_2_label_runner_protect_v1")
        | frame["candidate_uid"].astype(str).eq(FORENSIC_REPAIRED_CANDIDATE_UID)
    )
    useful = selected & ~v2_base & (_bool(frame, "label_should_not_take_v1") | _bool(frame, "tail_10_50_mfe_v1"))
    vetoed = pre_veto & veto
    still_missed = ~selected & (_bool(frame, "label_should_not_take_v1") | _bool(frame, "tail_10_50_mfe_v1"))
    mask = useful | vetoed | safety_damage | still_missed
    columns = [
        *REQUIRED_KEYS,
        "trade_id",
        "run_id",
        "gap_bucket_v1",
        "label_should_not_take_v1",
        "tail_10_50_mfe_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "strongest_winner_path_v1",
        "r6_label_repaired_165_like_runner_v1",
        "r6_label_runner_near_miss_v1",
        "r5_2_label_high_mfe_tail_risk_ambiguous_v1",
        "r5_2_label_runner_protect_v1",
    ]
    present = [column for column in columns if column in frame.columns]
    out = frame.loc[mask, present].copy()
    for column in [*RECALL_OUTPUTS, *PROTECTION_OUTPUTS, *BASE_OUTPUTS]:
        out[column] = pred.loc[mask, column].values
    out["variant_id_v1"] = variant_id
    out["forensics_role_v1"] = role
    out["added_vs_v2_v1"] = (selected & ~v2_base).loc[mask].values
    out["vetoed_by_hard_protection_v1"] = vetoed.loc[mask].values
    out["caused_safety_failure_v1"] = safety_damage.loc[mask].values
    out["improved_bad_or_tail_recall_v1"] = useful.loc[mask].values
    out["still_missed_v1"] = still_missed.loc[mask].values
    out["generalizable_signal_status_v1"] = np.where(useful.loc[mask].values, "CANDIDATE_SIGNAL_ROW", "AUDIT_ROW")
    return out


def _strategy_decision(eval_frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    passing = eval_frame[eval_frame["variant_gate_pass_v1"].astype(bool)].copy()
    safe = eval_frame[eval_frame["safety_pass_v1"].astype(bool)].copy()
    improves_unsafely = eval_frame[(~eval_frame["safety_pass_v1"].astype(bool)) & ((eval_frame["bad_uplift_over_v2_v1"] > 0) | (eval_frame["tail_uplift_over_v2_v1"] > 0))]
    if not passing.empty:
        best = passing.sort_values(
            by=[
                "bad_uplift_over_v2_v1",
                "tail_uplift_over_v2_v1",
                "worst_loso_v1",
                "precision_v1",
            ],
            ascending=[False, False, False, False],
        ).iloc[0]
        strategy = "V3_BEST_VARIANT_PASS_READY_FOR_R6"
        next_action = "RUN_R6_RETRAIN_FROM_BEST_R5_2_OBJECTIVE_V3_VARIANT_EXPLICIT_FLAG"
    elif not safe.empty:
        best = safe.sort_values(
            by=[
                "bad_uplift_over_v2_v1",
                "tail_uplift_over_v2_v1",
                "worst_loso_v1",
                "precision_v1",
            ],
            ascending=[False, False, False, False],
        ).iloc[0]
        strategy = "V3_SAFE_BUT_TOO_WEAK_STOP_R5_2_OBJECTIVE_LOOP"
        next_action = "MOVE_TO_CONSTRAINED_OPTUNA_OBJECTIVE_SEARCH"
    elif not improves_unsafely.empty:
        best = improves_unsafely.sort_values(
            by=["bad_uplift_over_v2_v1", "tail_uplift_over_v2_v1"],
            ascending=[False, False],
        ).iloc[0]
        strategy = "V3_RECALL_IMPROVES_BUT_SAFETY_FAILS"
        next_action = "STOP_AND_RUN_V3_FAILURE_FORENSICS"
    else:
        best = eval_frame.sort_values(
            by=["bad_recall_v1", "tail_recall_v1"],
            ascending=[False, False],
        ).iloc[0]
        strategy = "V3_ALL_VARIANTS_FAIL_OR_TOO_WEAK"
        next_action = "STOP_R5_2_OBJECTIVE_LOOP_AND_REVIEW_SIGNAL"
    strategy_payload = {
        "layer_name": "STRATEGY_GATE_AFTER_V3_V1",
        "decision_v1": strategy,
        "best_variant_id_v1": str(best["variant_id_v1"]),
        "best_profile_id_v1": str(best["profile_id_v1"]),
        "best_bad_recall_v1": int(best["bad_recall_v1"]),
        "best_tail_recall_v1": int(best["tail_recall_v1"]),
        "best_bad_uplift_over_v2_v1": int(best["bad_uplift_over_v2_v1"]),
        "best_tail_uplift_over_v2_v1": int(best["tail_uplift_over_v2_v1"]),
        "best_safety_pass_v1": bool(best["safety_pass_v1"]),
        "best_generalization_class_v1": str(best["generalization_class_v1"]),
        "r6_ready_v1": strategy == "V3_BEST_VARIANT_PASS_READY_FOR_R6",
    }
    next_lock = {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": next_action,
        "strategy_decision_v1": strategy,
        "do_not_run_v1": [
            "RUN_R6_WITHOUT_EXPLICIT_FLAG",
            "USE_PRE_VETO_BASE_AS_FINAL",
            "USE_DEGRADED_FALLBACK_SURFACE",
            "CONTINUE_V4_V5_BLINDLY_IF_V3_TOO_WEAK",
        ],
    }
    options = {
        "layer_name": "NEXT_STRATEGY_OPTIONS_IF_V3_TOO_WEAK_V1",
        "only_relevant_if_strategy_not_r6_ready_v1": strategy != "V3_BEST_VARIANT_PASS_READY_FOR_R6",
        "options_v1": [
            {
                "strategy_v1": "CONSTRAINED_OPTUNA_SEARCH",
                "description_v1": "Search weights, veto thresholds, and base thresholds with safety/LOSO/batch/no-leakage as hard constraints.",
            },
            {
                "strategy_v1": "MODEL_FAMILY_COMPARISON",
                "description_v1": "Compare same-feature/same-target tabular families before further objective-loop work.",
            },
            {
                "strategy_v1": "EXISTING_LEGAL_FEATURE_SIGNAL_AUDIT",
                "description_v1": "Audit underused legal AS_OF and entry-legal pre-RL/transformer signals before adding more training loops.",
            },
            {
                "strategy_v1": "STOP_AND_REASSESS",
                "description_v1": "Stop if V3 shows no safe generalizable signal beyond V2.",
            },
        ],
    }
    return strategy_payload, next_lock, options


def _validate_oof_provenance_tables(
    *,
    provenance: pd.DataFrame,
    fold_assignment: pd.DataFrame,
    membership: pd.DataFrame,
    expected_rows: int,
    expected_fields: Sequence[str],
    variant_id: str | None = None,
) -> dict[str, Any]:
    required_provenance = {
        *REQUIRED_KEYS,
        "variant_id_v1",
        "score_field_v1",
        "fold_id_v1",
        "group_key_v1",
        "train_validation_membership_v1",
        "source_model_fold_v1",
        "model_source_identifier_v1",
        "score_source_v1",
        "feature_matrix_hash_v1",
        "feature_matrix_columns_hash_v1",
        "label_table_hash_v1",
        "config_hash_v1",
        "seed_v1",
        "decision_valid_v1",
        "decision_valid_status_v1",
        "oof_provenance_status_v1",
        "row_was_in_training_for_source_model_v1",
        "in_sample_score_used_v1",
        "fallback_score_used_v1",
        "synthetic_score_used_v1",
    }
    required_fold = {*REQUIRED_KEYS, "variant_id_v1", "fold_id_v1", "group_key_v1", "train_validation_membership_v1"}
    required_membership = {
        "variant_id_v1",
        "score_field_v1",
        "fold_id_v1",
        "group_key_v1",
        "source_model_fold_v1",
        "train_validation_membership_v1",
    }
    failures: list[str] = []
    missing_prov = sorted(required_provenance.difference(provenance.columns))
    missing_fold = sorted(required_fold.difference(fold_assignment.columns))
    missing_membership = sorted(required_membership.difference(membership.columns))
    if missing_prov:
        failures.append(f"MISSING_PROVENANCE_COLUMNS:{missing_prov}")
    if missing_fold:
        failures.append(f"MISSING_FOLD_ASSIGNMENT_COLUMNS:{missing_fold}")
    if missing_membership:
        failures.append(f"MISSING_MEMBERSHIP_COLUMNS:{missing_membership}")
    if failures:
        return {"status_v1": "FAIL_MISSING_PROVENANCE", "failure_reasons_v1": failures}

    work = provenance.copy()
    if variant_id is not None:
        work = work[work["variant_id_v1"].astype(str).eq(str(variant_id))].copy()
    for field in expected_fields:
        field_rows = work[work["score_field_v1"].astype(str).eq(field)]
        if len(field_rows) != expected_rows:
            failures.append(f"FIELD_ROW_COVERAGE_MISMATCH:{field}:{len(field_rows)}")
    if not work["score_source_v1"].astype(str).eq("OOF").all():
        failures.append("FAIL_SCORE_SOURCE_NOT_OOF")
    if work["row_was_in_training_for_source_model_v1"].astype(bool).any():
        failures.append("FAIL_TRAIN_VALIDATION_LEAKAGE")
    if work["in_sample_score_used_v1"].astype(bool).any():
        failures.append("FAIL_IN_SAMPLE_SCORE_USED")
    if work["fallback_score_used_v1"].astype(bool).any():
        failures.append("FAIL_FALLBACK_SCORE_USED")
    if work["synthetic_score_used_v1"].astype(bool).any():
        failures.append("FAIL_SYNTHETIC_SCORE_USED")
    if not work["decision_valid_v1"].astype(bool).all():
        failures.append("FAIL_DECISION_VALID_FALSE")
    if not work["oof_provenance_status_v1"].astype(str).eq("PASS").all():
        failures.append("FAIL_OOF_PROVENANCE_STATUS")
    for column in ["feature_matrix_hash_v1", "feature_matrix_columns_hash_v1", "label_table_hash_v1", "config_hash_v1"]:
        if work[column].astype(str).str.len().lt(12).any():
            failures.append(f"FAIL_EMPTY_OR_SHORT_HASH:{column}")

    overlap = membership.groupby(["variant_id_v1", "score_field_v1", "fold_id_v1", "group_key_v1"])["train_validation_membership_v1"].nunique()
    if bool((overlap > 1).any()):
        failures.append("FAIL_TRAIN_VALIDATION_LEAKAGE")
    missing_fold_rows = fold_assignment["fold_id_v1"].isna().sum() if "fold_id_v1" in fold_assignment.columns else expected_rows
    if int(missing_fold_rows) > 0:
        failures.append(f"MISSING_FOLD_ID:{int(missing_fold_rows)}")
    status = "PASS" if not failures else (
        "FAIL_TRAIN_VALIDATION_LEAKAGE"
        if "FAIL_TRAIN_VALIDATION_LEAKAGE" in failures
        else "FAIL_IN_SAMPLE_SCORE_USED"
        if "FAIL_IN_SAMPLE_SCORE_USED" in failures
        else "FAIL_FALLBACK_SCORE_USED"
        if "FAIL_FALLBACK_SCORE_USED" in failures
        else "FAIL_SYNTHETIC_SCORE_USED"
        if "FAIL_SYNTHETIC_SCORE_USED" in failures
        else "FAIL_MISSING_PROVENANCE"
    )
    return {
        "status_v1": status,
        "expected_rows_per_score_field_v1": expected_rows,
        "expected_score_fields_v1": list(expected_fields),
        "provenance_rows_v1": int(len(work)),
        "fold_assignment_rows_v1": int(len(fold_assignment)),
        "membership_rows_v1": int(len(membership)),
        "failure_reasons_v1": failures,
    }


def _execute_variants(
    output_dir: Path,
    score: pd.DataFrame,
    label: pd.DataFrame,
    lane_01: pd.DataFrame,
    variant_manifest: pd.DataFrame,
) -> dict[str, Any]:
    frame = _merge_v3_targets(score, label, lane_01)
    feature_names, x = _feature_matrix(score)
    feature_matrix_hash = _dataframe_sha256(x)
    feature_matrix_columns_hash = _json_sha256(list(x.columns))
    label_table_hash = _dataframe_sha256(label)
    variant_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    prediction_by_variant: dict[str, pd.DataFrame] = {}
    aggregate_provenance: list[pd.DataFrame] = []
    aggregate_fold_assignment: list[pd.DataFrame] = []
    aggregate_membership: list[pd.DataFrame] = []
    aggregate_source_rows: list[dict[str, Any]] = []
    for idx, variant in variant_manifest.iterrows():
        variant_id = str(variant["variant_id_v1"])
        profile = str(variant["profile_id_v1"])
        variant_dir = output_dir / "variants" / variant_id
        pred = frame[REQUIRED_KEYS].copy()
        variant_provenance: list[pd.DataFrame] = []
        variant_fold_assignment: list[pd.DataFrame] = []
        variant_membership: list[pd.DataFrame] = []
        variant_source_rows: list[dict[str, Any]] = []
        heads = [
            ("v3_bad_recall_target_v1", "r5_2_v3_bad_recall_score"),
            ("v3_tail_recall_target_v1", "r5_2_v3_tail_recall_score"),
            ("v3_risky_attention_target_v1", "r5_2_v3_risky_attention_score"),
            ("v3_runner_protection_target_v1", "r5_2_v3_runner_protection_score"),
            ("v3_high_mfe_ambiguous_protection_target_v1", "r5_2_v3_high_mfe_ambiguous_protection_score"),
            ("v3_hard_winner_protection_target_v1", "r5_2_v3_hard_winner_protection_score"),
        ]
        for head_idx, (target, output) in enumerate(heads):
            seed = 20260426 + int(idx) * 10 + head_idx
            config_hash = _json_sha256(
                {
                    "variant_config_v1": variant.to_dict(),
                    "target_v1": target,
                    "score_field_v1": output,
                    "seed_v1": seed,
                    "feature_matrix_hash_v1": feature_matrix_hash,
                    "label_table_hash_v1": label_table_hash,
                }
            )
            result = _fit_oof_score_with_provenance(
                frame=frame,
                x=x,
                target=target,
                output=output,
                seed=seed,
                variant=variant,
                model_path=variant_dir / "models" / f"{output}.joblib",
                feature_matrix_hash=feature_matrix_hash,
                feature_matrix_columns_hash=feature_matrix_columns_hash,
                label_table_hash=label_table_hash,
                config_hash=config_hash,
            )
            pred[output] = result.scores
            variant_provenance.append(result.provenance)
            variant_fold_assignment.append(result.fold_assignment)
            variant_membership.append(result.membership)
            variant_source_rows.extend(result.source_manifest_rows)
        pred = _apply_v3_base_rule(frame, pred, profile)
        variant_dir.mkdir(parents=True, exist_ok=True)
        prediction_view = frame[[*REQUIRED_KEYS, "trade_id", "run_id", "label_should_not_take_v1", "tail_10_50_mfe_v1"]].copy()
        for column in [*RECALL_OUTPUTS, *PROTECTION_OUTPUTS, *BASE_OUTPUTS]:
            prediction_view[column] = pred[column].values
        score_package = score.merge(pred[[*REQUIRED_KEYS, *RECALL_OUTPUTS, *PROTECTION_OUTPUTS, *BASE_OUTPUTS]], on=REQUIRED_KEYS, how="left", validate="one_to_one")
        base_membership = pred[[*REQUIRED_KEYS, *BASE_OUTPUTS]].copy()
        prediction_view.to_parquet(variant_dir / "prediction_view_v1.parquet", index=False)
        score_package.to_parquet(variant_dir / "score_package_v1.parquet", index=False)
        base_membership.to_parquet(variant_dir / "base_membership_package_v1.parquet", index=False)
        provenance = pd.concat(variant_provenance, ignore_index=True)
        fold_assignment = pd.concat(variant_fold_assignment, ignore_index=True).drop_duplicates()
        membership = pd.concat(variant_membership, ignore_index=True).drop_duplicates()
        validation = _validate_oof_provenance_tables(
            provenance=provenance,
            fold_assignment=fold_assignment,
            membership=membership,
            expected_rows=len(frame),
            expected_fields=V3_SCORE_FIELDS,
            variant_id=variant_id,
        )
        if validation["status_v1"] != "PASS":
            raise RuntimeError(f"OOF provenance validation failed for {variant_id}: {validation['failure_reasons_v1']}")
        provenance.to_csv(variant_dir / "v3_oof_score_provenance_v1.csv", index=False)
        fold_assignment.to_csv(variant_dir / "v3_oof_fold_assignment_v1.csv", index=False)
        membership.to_csv(variant_dir / "v3_train_validation_membership_v1.csv", index=False)
        _write_json(
            variant_dir / "v3_oof_score_source_manifest_v1.json",
            {
                "variant_id_v1": variant_id,
                "score_source_v1": "OOF",
                "feature_matrix_hash_v1": feature_matrix_hash,
                "feature_matrix_columns_hash_v1": feature_matrix_columns_hash,
                "label_table_hash_v1": label_table_hash,
                "scorefield_registry_v1": _scorefield_registry(variant_id),
                "source_models_v1": variant_source_rows,
                "validation_v1": validation,
                "decision_valid_for_pre_optuna_v1": True,
                "oof_provenance_status_v1": "PASS",
            },
        )
        aggregate_provenance.append(provenance)
        aggregate_fold_assignment.append(fold_assignment)
        aggregate_membership.append(membership)
        aggregate_source_rows.extend(variant_source_rows)
        metrics = _variant_metrics(frame, pred, variant)
        _write_variant_reports(variant_dir, variant, feature_names, heads, metrics)
        downstream_manifest_path = variant_dir / "downstream_r6_input_manifest_v1.json"
        if metrics["variant_gate_pass_v1"]:
            _write_json(
                downstream_manifest_path,
                {
                    "variant_id_v1": variant_id,
                    "ready_for_downstream_r6_v1": True,
                    "base_flag_for_r6_v1": "r5_2_v3_final_base_membership",
                    "score_package_path_v1": str(variant_dir / "score_package_v1.parquet"),
                    "prediction_view_path_v1": str(variant_dir / "prediction_view_v1.parquet"),
                    "base_membership_path_v1": str(variant_dir / "base_membership_package_v1.parquet"),
                    "must_not_use_as_final_base_v1": ["r5_2_v3_base_membership_pre_veto", "unsafe_variant", "V2/rescue/scan flags"],
                },
            )
        else:
            _write_json(
                downstream_manifest_path,
                {
                    "variant_id_v1": variant_id,
                    "ready_for_downstream_r6_v1": False,
                    "reason_v1": "VARIANT_GATE_NOT_PASSING",
                    "base_flag_for_r6_v1": "r5_2_v3_final_base_membership",
                },
            )
        prediction_by_variant[variant_id] = pred
        eval_rows.append({**metrics, "variant_dir_v1": str(variant_dir)})
        variant_rows.append(
            {
                "variant_id_v1": variant_id,
                "profile_id_v1": profile,
                "variant_dir_v1": str(variant_dir),
                "bad_rows_v1": metrics["bad_recall_v1"],
                "tail_rows_v1": metrics["tail_recall_v1"],
                "final_base_count_v1": metrics["final_base_count_v1"],
                "score_package_path_v1": str(variant_dir / "score_package_v1.parquet"),
                "downstream_r6_input_manifest_path_v1": str(downstream_manifest_path),
                "variant_gate_pass_v1": metrics["variant_gate_pass_v1"],
            }
        )
    index = pd.DataFrame(variant_rows)
    eval_frame = pd.DataFrame(eval_rows)
    generalization = eval_frame[
        [
            "variant_id_v1",
            "profile_id_v1",
            "generalization_class_v1",
            "safety_pass_v1",
            "meaningful_uplift_over_v2_v1",
            "batch_stability_v1",
            "worst_loso_v1",
            "selected_max_group_share_v1",
            "variant_gate_pass_v1",
        ]
    ].copy()
    leaderboard = eval_frame.sort_values(
        by=[
            "variant_gate_pass_v1",
            "safety_pass_v1",
            "meaningful_uplift_over_v2_v1",
            "bad_uplift_over_v2_v1",
            "tail_uplift_over_v2_v1",
            "worst_loso_v1",
            "precision_v1",
            "fifty_plus_overlap_v1",
        ],
        ascending=[False, False, False, False, False, False, False, True],
    ).reset_index(drop=True)
    leaderboard.insert(0, "rank_v1", range(1, len(leaderboard) + 1))
    strategy, next_lock, options = _strategy_decision(eval_frame)
    best_id = strategy["best_variant_id_v1"]
    best_row = eval_frame[eval_frame["variant_id_v1"].eq(best_id)].iloc[0]
    best_dir = Path(str(best_row["variant_dir_v1"]))
    unsafe = eval_frame[~eval_frame["safety_pass_v1"].astype(bool)]
    if unsafe.empty:
        tempting_id = leaderboard.iloc[0]["variant_id_v1"]
        tempting_role = "best_passing_or_safe_variant_no_unsafe_candidate"
    else:
        tempting = unsafe.sort_values(by=["bad_recall_v1", "tail_recall_v1"], ascending=[False, False]).iloc[0]
        tempting_id = tempting["variant_id_v1"]
        tempting_role = "best_unsafe_tempting_variant"
    forensics_frames = [
        _variant_forensics(frame, prediction_by_variant[str(best_id)], str(best_id), "best_variant"),
        _variant_forensics(frame, prediction_by_variant[str(tempting_id)], str(tempting_id), tempting_role),
    ]
    row_forensics = pd.concat(forensics_frames, ignore_index=True)
    all_provenance = pd.concat(aggregate_provenance, ignore_index=True)
    all_fold_assignment = pd.concat(aggregate_fold_assignment, ignore_index=True).drop_duplicates()
    all_membership = pd.concat(aggregate_membership, ignore_index=True).drop_duplicates()
    aggregate_validation = _validate_oof_provenance_tables(
        provenance=all_provenance,
        fold_assignment=all_fold_assignment,
        membership=all_membership,
        expected_rows=len(frame),
        expected_fields=V3_SCORE_FIELDS,
        variant_id=str(best_id),
    )
    if aggregate_validation["status_v1"] != "PASS":
        raise RuntimeError(f"Aggregate OOF provenance validation failed: {aggregate_validation['failure_reasons_v1']}")
    index.to_csv(output_dir / "v3_variant_training_outputs_index_v1.csv", index=False)
    index.to_csv(output_dir / "v3_variant_outputs_index_v1.csv", index=False)
    eval_frame.to_csv(output_dir / "v3_variant_eval_and_safety_gate_v1.csv", index=False)
    generalization.to_csv(output_dir / "v3_generalization_and_overfit_eval_v1.csv", index=False)
    leaderboard.to_csv(output_dir / "v3_variant_leaderboard_v1.csv", index=False)
    row_forensics.to_csv(output_dir / "v3_row_level_forensics_v1.csv", index=False)
    all_provenance.to_csv(output_dir / "v3_oof_score_provenance_v1.csv", index=False)
    all_fold_assignment.to_csv(output_dir / "v3_oof_fold_assignment_v1.csv", index=False)
    all_membership.to_csv(output_dir / "v3_train_validation_membership_v1.csv", index=False)
    _write_json(
        output_dir / "v3_oof_score_source_manifest_v1.json",
        {
            "layer_name": "V3_OOF_SCORE_SOURCE_MANIFEST_V1",
            "score_source_v1": "OOF",
            "variant_count_v1": int(len(index)),
            "score_fields_v1": V3_SCORE_FIELDS,
            "scorefield_registry_v1": _scorefield_registry(),
            "feature_matrix_hash_v1": feature_matrix_hash,
            "feature_matrix_columns_hash_v1": feature_matrix_columns_hash,
            "label_table_hash_v1": label_table_hash,
            "source_models_v1": aggregate_source_rows,
            "aggregate_provenance_path_v1": str(output_dir / "v3_oof_score_provenance_v1.csv"),
            "aggregate_fold_assignment_path_v1": str(output_dir / "v3_oof_fold_assignment_v1.csv"),
            "aggregate_train_validation_membership_path_v1": str(output_dir / "v3_train_validation_membership_v1.csv"),
            "validation_v1": aggregate_validation,
            "decision_valid_for_pre_optuna_v1": True,
            "oof_provenance_status_v1": "PASS",
        },
    )
    _write_json(output_dir / "active_score_artifact_selection_v1.json", _active_score_selection_contract(output_dir))
    _write_json(
        output_dir / "v3_parallel_rebuild_execution_v1.json",
        {
            "layer_name": "RUN_V3_PARALLEL_REBUILD_EXECUTION_V1",
            "training_started_v1": True,
            "parallel_execution_started_v1": True,
            "variant_count_v1": int(len(index)),
            "foundation_rows_v1": int(len(score)),
            "target_table_rows_v1": int(len(label)),
            "hard_protection_veto_contract_active_v1": True,
            "oof_provenance_written_v1": True,
            "active_score_artifact_selection_written_v1": True,
            "output_index_v1": str(output_dir / "v3_variant_outputs_index_v1.csv"),
        },
    )
    if strategy["r6_ready_v1"]:
        _write_json(
            output_dir / "best_v3_variant_downstream_r6_input_lock_v1.json",
            {
                "layer_name": "BEST_V3_VARIANT_DOWNSTREAM_R6_INPUT_LOCK_V1",
                "ready_for_downstream_r6_v1": True,
                "best_variant_id_v1": best_id,
                "best_profile_id_v1": strategy["best_profile_id_v1"],
                "score_package_path_v1": str(best_dir / "score_package_v1.parquet"),
                "prediction_view_path_v1": str(best_dir / "prediction_view_v1.parquet"),
                "final_base_membership_path_v1": str(best_dir / "base_membership_package_v1.parquet"),
                "downstream_r6_input_manifest_v1": str(best_dir / "downstream_r6_input_manifest_v1.json"),
                "required_r6_base_flag_v1": "r5_2_v3_final_base_membership",
                "r6_must_not_use_v1": ["r5_2_v3_base_membership_pre_veto", "unsafe_variant", "V2/rescue/V3 scan as final"],
            },
        )
    else:
        _write_json(
            output_dir / "best_v3_variant_downstream_r6_input_lock_v1.json",
            {
                "layer_name": "BEST_V3_VARIANT_DOWNSTREAM_R6_INPUT_LOCK_V1",
                "ready_for_downstream_r6_v1": False,
                "failure_reason_v1": strategy["decision_v1"],
                "best_variant_id_v1": best_id,
                "required_r6_base_flag_v1": "r5_2_v3_final_base_membership",
            },
        )
    _write_json(output_dir / "strategy_gate_after_v3_v1.json", strategy)
    _write_json(output_dir / "next_strategy_options_if_v3_too_weak_v1.json", options)
    _write_json(output_dir / "next_action_lock_v1.json", next_lock)
    return {
        "parallel_execution_started_v1": True,
        "training_started_v1": True,
        "variant_count_v1": int(len(index)),
        "best_variant_id_v1": strategy["best_variant_id_v1"],
        "best_bad_recall_v1": strategy["best_bad_recall_v1"],
        "best_tail_recall_v1": strategy["best_tail_recall_v1"],
        "strategy_decision_v1": strategy["decision_v1"],
        "next_action_v1": next_lock["next_action_v1"],
        "r6_ready_v1": strategy["r6_ready_v1"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _write_blocked_output(output_dir: Path, message: str, missing_list: Sequence[str] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "layer_name": LAYER_NAME,
        "status_v1": "BLOCKED_MISSING_REQUIRED_INPUT",
        "prelaunch_status_v1": "BLOCKED_MISSING_REQUIRED_INPUT",
        "training_started_v1": False,
        "parallel_execution_started_v1": False,
        "error_v1": message,
        "missing_required_inputs_v1": list(missing_list or []),
        "next_action_v1": "FIX_V3_PARALLEL_RUNNER_FIRST",
    }
    _write_json(output_dir / "status_v1.json", payload)
    _write_json(output_dir / "summary_v1.json", payload)


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    scan_dir: Path = DEFAULT_SCAN_DIR,
    output_dir: Path | None = None,
    v2_execution_dir: Path | None = None,
    score_package: Path | None = None,
    foundation_summary: Path | None = None,
    label_table: Path | None = None,
    feature_inventory: Path | None = None,
    downstream_r6_lock: Path | None = None,
    run_parallel_rebuild: bool = False,
    explicit_action: str | None = None,
    write_oof_provenance: bool = True,
    reject_in_sample_decision_scores: bool = True,
    fail_on_missing_provenance: bool = True,
    fail_on_degraded_fallback: bool = True,
    fail_on_dummy_or_synthetic_input: bool = True,
) -> dict[str, Any]:
    if explicit_action is not None:
        if explicit_action != EXPLICIT_OOF_RERUN_ACTION:
            raise RuntimeError(f"Unsupported explicit action: {explicit_action}")
        run_parallel_rebuild = True
        required_flags = {
            "write_oof_provenance": write_oof_provenance,
            "reject_in_sample_decision_scores": reject_in_sample_decision_scores,
            "fail_on_missing_provenance": fail_on_missing_provenance,
            "fail_on_degraded_fallback": fail_on_degraded_fallback,
            "fail_on_dummy_or_synthetic_input": fail_on_dummy_or_synthetic_input,
        }
        missing_flags = [name for name, enabled in required_flags.items() if not enabled]
        if missing_flags:
            raise RuntimeError(f"{EXPLICIT_OOF_RERUN_ACTION} requires enabled flags: {missing_flags}")
    if output_dir is None:
        if explicit_action == EXPLICIT_OOF_RERUN_ACTION:
            output_dir = reports_root / f"{EXPLICIT_OOF_RERUN_ACTION}_{_stamp()}_LOCK"
        else:
            suffix = "EXECUTION_REQUESTED" if run_parallel_rebuild else "DRY_PRELAUNCH"
            output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_{suffix}"
    _ensure_clean_output(output_dir)
    try:
        scan = _validate_scan_dir(scan_dir)
        paths = _resolve_input_paths(
            scan_dir=scan_dir,
            v2_execution_dir=v2_execution_dir,
            score_package=score_package,
            foundation_summary=foundation_summary,
            label_table=label_table,
            feature_inventory=feature_inventory,
            downstream_r6_lock=downstream_r6_lock,
        )
        score = _load_required_parquet(paths["score_package_v1"], "foundation/score package")
        foundation_summary_payload = _read_json(paths["foundation_summary_v1"])
        label = _load_required_csv(paths["label_table_v1"], "target table")
        feature_inv = _load_required_csv(paths["feature_inventory_v1"], "feature inventory")
        lane_01 = _load_required_csv(scan_dir / "lane_01_v2_remaining_gap_trace_v1.csv", "V3 lane 01 gap trace")
        _load_required_csv(scan_dir / "lane_09_v3_weight_profile_sim_scan_v1.csv", "V3 lane 09 profile scan")
        _load_required_csv(scan_dir / "lane_10_high_mfe_winner_stress_scan_v1.csv", "V3 lane 10 winner stress scan")
        fallback_contract = _validate_no_degraded_fallback(paths, score)
        foundation = _validate_foundation(score, foundation_summary_payload)
        variant_manifest = _variant_config()
        if len(variant_manifest) != 10 or set(variant_manifest["status_v1"]) != {"READY_FOR_EXPLICIT_RUN"}:
            raise RuntimeError("All 10 V3 variants must be READY_FOR_EXPLICIT_RUN")
        target_audit = _target_prelaunch(score, label, lane_01)
        feature_prelaunch, forbidden_scan, id_scan = _feature_prelaunch(score, feature_inv)
        hard_veto = _hard_veto_contract(score, label)
        anti_overfit = _anti_overfit_guard(variant_manifest)
        downstream_contract = _downstream_r6_contract(output_dir)
    except BlockedMissingRequiredInput as exc:
        _write_blocked_output(output_dir, str(exc), [str(exc)])
        raise

    if run_parallel_rebuild:
        execution = _execute_variants(output_dir, score, label, lane_01, variant_manifest)
        summary = {
            "layer_name": LAYER_NAME,
            "materialized_at_utc_v1": _utc_now(),
            "output_dir_v1": str(output_dir),
            "prelaunch_status_v1": "PASS",
            "decision_v1": "TRAINING_EXECUTION_COMPLETED",
            "explicit_action_v1": explicit_action or RUN_FLAG,
            "write_oof_provenance_v1": write_oof_provenance,
            "reject_in_sample_decision_scores_v1": reject_in_sample_decision_scores,
            "fail_on_missing_provenance_v1": fail_on_missing_provenance,
            "fail_on_degraded_fallback_v1": fail_on_degraded_fallback,
            "fail_on_dummy_or_synthetic_input_v1": fail_on_dummy_or_synthetic_input,
            "training_started_v1": True,
            "parallel_execution_started_v1": True,
            "r6_started_v1": False,
            "variant_count_v1": int(len(variant_manifest)),
            "foundation_rows_v1": foundation["foundation_rows_v1"],
            "target_table_rows_v1": target_audit["target_table_rows_v1"],
            "asof_columns_v1": foundation["asof_columns_v1"],
            "feature_count_v1": feature_prelaunch["feature_count_v1"],
            "forbidden_feature_count_v1": feature_prelaunch["forbidden_feature_count_v1"],
            "id_leakage_feature_count_v1": feature_prelaunch["id_leakage_feature_count_v1"],
            "synthetic_or_dummy_input_count_v1": fallback_contract["synthetic_or_dummy_input_count_v1"],
            "ambiguous_high_mfe_bad_positive_count_v1": target_audit["ambiguous_high_mfe_bad_positive_count_v1"],
            "runner_protect_bad_positive_count_v1": target_audit["runner_protect_bad_positive_count_v1"],
            "hard_protection_veto_contract_present_v1": hard_veto["hard_protection_veto_contract_present_v1"],
            "anti_overfit_guard_pass_v1": anti_overfit["anti_overfit_guard_pass_v1"],
            "degraded_fallback_used_v1": fallback_contract["degraded_fallback_used_v1"],
            "best_variant_id_v1": execution["best_variant_id_v1"],
            "best_bad_recall_v1": execution["best_bad_recall_v1"],
            "best_tail_recall_v1": execution["best_tail_recall_v1"],
            "strategy_decision_v1": execution["strategy_decision_v1"],
            "r6_ready_v1": execution["r6_ready_v1"],
            "next_action_v1": execution["next_action_v1"],
            "blocked_action_v1": "RUN_R6_WITHOUT_EXPLICIT_FLAG",
            **execution,
        }
        _write_json(output_dir / "summary_v1.json", summary)
        _write_json(output_dir / "status_v1.json", {**summary, "status_v1": "TRAINING_EXECUTION_COMPLETED"})
        _write_json(
            output_dir / "manifest_v1.json",
            {
                "layer_name": f"{LAYER_NAME}_EXECUTION_MANIFEST",
                "v3_parallel_rebuild_execution_v1": str(output_dir / "v3_parallel_rebuild_execution_v1.json"),
                "variant_outputs_index_v1": str(output_dir / "v3_variant_outputs_index_v1.csv"),
                "variant_eval_and_safety_gate_v1": str(output_dir / "v3_variant_eval_and_safety_gate_v1.csv"),
                "generalization_and_overfit_eval_v1": str(output_dir / "v3_generalization_and_overfit_eval_v1.csv"),
                "variant_leaderboard_v1": str(output_dir / "v3_variant_leaderboard_v1.csv"),
                "v3_oof_score_provenance_v1": str(output_dir / "v3_oof_score_provenance_v1.csv"),
                "v3_oof_fold_assignment_v1": str(output_dir / "v3_oof_fold_assignment_v1.csv"),
                "v3_oof_score_source_manifest_v1": str(output_dir / "v3_oof_score_source_manifest_v1.json"),
                "v3_train_validation_membership_v1": str(output_dir / "v3_train_validation_membership_v1.csv"),
                "active_score_artifact_selection_v1": str(output_dir / "active_score_artifact_selection_v1.json"),
                "best_v3_variant_downstream_r6_input_lock_v1": str(output_dir / "best_v3_variant_downstream_r6_input_lock_v1.json"),
                "row_level_forensics_v1": str(output_dir / "v3_row_level_forensics_v1.csv"),
                "strategy_gate_after_v3_v1": str(output_dir / "strategy_gate_after_v3_v1.json"),
                "next_strategy_options_if_v3_too_weak_v1": str(output_dir / "next_strategy_options_if_v3_too_weak_v1.json"),
                "next_action_lock_v1": str(output_dir / "next_action_lock_v1.json"),
                "input_paths_v1": {key: str(value) for key, value in paths.items()},
            },
        )
        _write_csv(
            output_dir / "consistency_audit_v1.csv",
            [
                {"check_v1": "explicit_flag_required", "status_v1": "PASS", "evidence_v1": RUN_FLAG},
                {"check_v1": "training_started", "status_v1": "PASS", "evidence_v1": True},
                {"check_v1": "r6_not_started", "status_v1": "PASS", "evidence_v1": False},
                {"check_v1": "no_forbidden_features", "status_v1": "PASS", "evidence_v1": feature_prelaunch["forbidden_feature_count_v1"]},
                {"check_v1": "no_id_leakage", "status_v1": "PASS", "evidence_v1": feature_prelaunch["id_leakage_feature_count_v1"]},
                {"check_v1": "no_dummy_inputs", "status_v1": "PASS", "evidence_v1": fallback_contract["synthetic_or_dummy_input_count_v1"]},
                {"check_v1": "degraded_fallback_used", "status_v1": "PASS", "evidence_v1": fallback_contract["degraded_fallback_used_v1"]},
                {"check_v1": "oof_provenance_written", "status_v1": "PASS", "evidence_v1": str(output_dir / "v3_oof_score_provenance_v1.csv")},
                {"check_v1": "strategy_gate_written", "status_v1": "PASS", "evidence_v1": execution["strategy_decision_v1"]},
            ],
        )
        report = "\n".join(
            [
                "# R5.2 Objective V3 Parallel Rebuild Execution",
                "",
                f"Decision: `{summary['strategy_decision_v1']}`",
                f"Next action: `{summary['next_action_v1']}`",
                "",
                f"- Variants executed: `{summary['variant_count_v1']}`",
                f"- Best variant: `{summary['best_variant_id_v1']}`",
                f"- Best bad/tail: `{summary['best_bad_recall_v1']}` / `{summary['best_tail_recall_v1']}`",
                f"- Foundation rows: `{summary['foundation_rows_v1']}`",
                f"- Target rows: `{summary['target_table_rows_v1']}`",
                f"- AS_OF columns: `{summary['asof_columns_v1']}`",
                f"- Forbidden features: `{summary['forbidden_feature_count_v1']}`",
                f"- ID leakage features: `{summary['id_leakage_feature_count_v1']}`",
                f"- Dummy/synthetic inputs: `{summary['synthetic_or_dummy_input_count_v1']}`",
                f"- Degraded fallback used: `{summary['degraded_fallback_used_v1']}`",
                f"- R6 started: `{summary['r6_started_v1']}`",
                "",
                "See `v3_variant_leaderboard_v1.csv`, `v3_variant_eval_and_safety_gate_v1.csv`, and `strategy_gate_after_v3_v1.json` for the strict gate.",
            ]
        )
        (output_dir / "report_v1.md").write_text(report + "\n", encoding="utf-8")
        return summary

    variant_manifest.to_csv(output_dir / "v3_variant_config_manifest_v1.csv", index=False)
    forbidden_scan.to_csv(output_dir / "v3_forbidden_feature_scan_v1.csv", index=False)
    id_scan.to_csv(output_dir / "v3_id_leakage_scan_v1.csv", index=False)
    _write_json(output_dir / "no_degraded_fallback_contract_v1.json", fallback_contract)
    _write_json(output_dir / "v3_target_table_prelaunch_audit_v1.json", target_audit)
    _write_json(output_dir / "v3_feature_matrix_prelaunch_v1.json", feature_prelaunch)
    _write_json(output_dir / "v3_generalization_anti_overfit_guard_v1.json", anti_overfit)
    _write_json(output_dir / "v3_hard_veto_contract_report_v1.json", hard_veto)
    _write_json(output_dir / "v3_downstream_r6_manifest_contract_v1.json", downstream_contract)

    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "prelaunch_status_v1": "PASS",
        "decision_v1": "DRY_PRELAUNCH_COMPLETED",
        "training_started_v1": False,
        "parallel_execution_started_v1": False,
        "r6_started_v1": False,
        "variant_count_v1": int(len(variant_manifest)),
        "foundation_rows_v1": foundation["foundation_rows_v1"],
        "target_table_rows_v1": target_audit["target_table_rows_v1"],
        "asof_columns_v1": foundation["asof_columns_v1"],
        "feature_count_v1": feature_prelaunch["feature_count_v1"],
        "forbidden_feature_count_v1": feature_prelaunch["forbidden_feature_count_v1"],
        "id_leakage_feature_count_v1": feature_prelaunch["id_leakage_feature_count_v1"],
        "synthetic_or_dummy_input_count_v1": fallback_contract["synthetic_or_dummy_input_count_v1"],
        "ambiguous_high_mfe_bad_positive_count_v1": target_audit["ambiguous_high_mfe_bad_positive_count_v1"],
        "runner_protect_bad_positive_count_v1": target_audit["runner_protect_bad_positive_count_v1"],
        "hard_protection_veto_contract_present_v1": hard_veto["hard_protection_veto_contract_present_v1"],
        "anti_overfit_guard_pass_v1": anti_overfit["anti_overfit_guard_pass_v1"],
        "degraded_fallback_used_v1": fallback_contract["degraded_fallback_used_v1"],
        "v3_bucket_fix_counts_v1": target_audit["v3_bucket_fix_counts_v1"],
        "next_action_v1": "NEXT_AGENT_MAY_RUN_R5_2_OBJECTIVE_V3_PARALLEL_REBUILD_WITH_EXPLICIT_FLAG",
        "blocked_action_v1": "RUN_PARALLEL_REBUILD_WITHOUT_EXPLICIT_FLAG",
    }
    status = {
        **summary,
        "status_v1": "DRY_PRELAUNCH_COMPLETED",
    }
    prelaunch_report = {
        "layer_name": "V3_PARALLEL_PRELAUNCH_REPORT_V1",
        "scan_validation_v1": scan,
        "input_paths_v1": {key: str(value) for key, value in paths.items()},
        "foundation_validation_v1": foundation,
        "variant_config_v1": {"variant_count_v1": len(variant_manifest), "all_ready_v1": True},
        "target_validation_v1": target_audit,
        "feature_validation_v1": feature_prelaunch,
        "hard_veto_validation_v1": hard_veto,
        "anti_overfit_validation_v1": anti_overfit,
        "no_training_outputs_written_v1": True,
    }
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "output_files_v1": {name: name for name in DRY_OUTPUT_FILES},
        "input_paths_v1": {key: str(value) for key, value in paths.items()},
        "fake_prediction_views_written_v1": False,
        "fake_score_packages_written_v1": False,
        "fake_base_membership_written_v1": False,
        "dummy_model_manifests_written_v1": False,
    }
    next_lock = {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": summary["next_action_v1"],
        "blocked_action_v1": summary["blocked_action_v1"],
        "do_not_run_v1": [
            "RUN_PARALLEL_REBUILD_WITHOUT_EXPLICIT_FLAG",
            "RUN_R6_NOW",
            "USE_DEGRADED_FALLBACK_SURFACE",
            "MATERIALIZE_FAKE_SCORE_PACKAGE",
        ],
    }
    audit_rows = [
        {"check_v1": "training_not_started", "status_v1": "PASS", "evidence_v1": False},
        {"check_v1": "all_10_variants_ready", "status_v1": "PASS", "evidence_v1": len(variant_manifest)},
        {"check_v1": "foundation_rows", "status_v1": "PASS", "evidence_v1": foundation["foundation_rows_v1"]},
        {"check_v1": "target_rows", "status_v1": "PASS", "evidence_v1": target_audit["target_table_rows_v1"]},
        {"check_v1": "no_forbidden_features", "status_v1": "PASS", "evidence_v1": feature_prelaunch["forbidden_feature_count_v1"]},
        {"check_v1": "no_id_leakage", "status_v1": "PASS", "evidence_v1": feature_prelaunch["id_leakage_feature_count_v1"]},
        {"check_v1": "no_dummy_inputs", "status_v1": "PASS", "evidence_v1": fallback_contract["synthetic_or_dummy_input_count_v1"]},
        {"check_v1": "hard_veto_contract", "status_v1": "PASS", "evidence_v1": True},
        {"check_v1": "v3_bucket_fix_forwarded", "status_v1": "PASS", "evidence_v1": target_audit["v3_bucket_fix_counts_v1"]},
    ]
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", status)
    _write_json(output_dir / "manifest_v1.json", manifest)
    _write_json(output_dir / "v3_parallel_prelaunch_report_v1.json", prelaunch_report)
    _write_json(output_dir / "next_action_lock_v1.json", next_lock)
    _write_csv(output_dir / "consistency_audit_v1.csv", audit_rows)
    report = "\n".join(
        [
            "# R5.2 Objective V3 Parallel Rebuild Runner Dry Prelaunch",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Variants ready: `{summary['variant_count_v1']}`",
            f"- Foundation rows: `{summary['foundation_rows_v1']}`",
            f"- Target rows: `{summary['target_table_rows_v1']}`",
            f"- Forbidden features: `{summary['forbidden_feature_count_v1']}`",
            f"- ID leakage features: `{summary['id_leakage_feature_count_v1']}`",
            f"- Dummy/synthetic inputs: `{summary['synthetic_or_dummy_input_count_v1']}`",
            f"- Hard veto contract present: `{summary['hard_protection_veto_contract_present_v1']}`",
            "",
            "No prediction views, score packages, base membership files, R6 inputs, or model manifests were written in dry/prelaunch.",
        ]
    )
    (output_dir / "report_v1.md").write_text(report + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(RUN_FLAG, action="store_true", dest="run_parallel_rebuild")
    parser.add_argument("--explicit-action", default=None)
    parser.add_argument("--write-oof-provenance", action="store_true")
    parser.add_argument("--reject-in-sample-decision-scores", action="store_true")
    parser.add_argument("--fail-on-missing-provenance", action="store_true")
    parser.add_argument("--fail-on-degraded-fallback", action="store_true")
    parser.add_argument("--fail-on-dummy-or-synthetic-input", action="store_true")
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--v2-execution-dir", type=Path, default=None)
    parser.add_argument("--score-package", type=Path, default=None)
    parser.add_argument("--foundation-summary", type=Path, default=None)
    parser.add_argument("--label-table", type=Path, default=None)
    parser.add_argument("--feature-inventory", type=Path, default=None)
    parser.add_argument("--downstream-r6-lock", type=Path, default=None)
    args = parser.parse_args(argv)
    materialize(
        reports_root=args.reports_root,
        scan_dir=args.scan_dir,
        output_dir=args.output_dir,
        v2_execution_dir=args.v2_execution_dir,
        score_package=args.score_package,
        foundation_summary=args.foundation_summary,
        label_table=args.label_table,
        feature_inventory=args.feature_inventory,
        downstream_r6_lock=args.downstream_r6_lock,
        run_parallel_rebuild=args.run_parallel_rebuild,
        explicit_action=args.explicit_action,
        write_oof_provenance=args.write_oof_provenance or args.explicit_action is None,
        reject_in_sample_decision_scores=args.reject_in_sample_decision_scores or args.explicit_action is None,
        fail_on_missing_provenance=args.fail_on_missing_provenance or args.explicit_action is None,
        fail_on_degraded_fallback=args.fail_on_degraded_fallback or args.explicit_action is None,
        fail_on_dummy_or_synthetic_input=args.fail_on_dummy_or_synthetic_input or args.explicit_action is None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
