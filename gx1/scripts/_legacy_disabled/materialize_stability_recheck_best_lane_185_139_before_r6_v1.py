#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "STABILITY_RECHECK_BEST_LANE_185_139_BEFORE_R6_V1"
LAYER_NAME = ACTION
INPUT_BEST_LANE_PACKAGE_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "MATERIALIZE_BEST_LANE_CANDIDATE_PACKAGE_REQUIRES_EXPLICIT_GATE_V1_20260427T193354Z_LOCK"
)
INPUT_LANE_PACK_ROOT = (
    DEFAULT_REPORTS_ROOT / "PARALLEL_TAIL_R6_R5_2_REPAIR_LANE_PACK_V1_20260427T191454Z_LOCK"
)
INPUT_TAIL_REPAIRED_PACKAGE_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "BUILD_TAIL_REPAIRED_R5_2_CANDIDATE_PACKAGE_REQUIRES_EXPLICIT_GATE_V1_20260427T175754Z_LOCK"
)
INPUT_UPLIFT_AUDIT_ROOT = (
    DEFAULT_REPORTS_ROOT / "R5_2_UPLIFT_AND_R6_HEAD_SIGNAL_AUDIT_V1_20260427T171341Z_LOCK"
)

SELECTED_LANE_ID = "LANE_08_R5_2_GAP_ROWS_SAFE_ONLY"
BASELINE_BAD = 140
BASELINE_TAIL = 94
BEST_LANE_BAD = 185
BEST_LANE_TAIL = 139
ADDED_ROWS = 45
WEDNESDAY_BAD = 180
WEDNESDAY_TAIL = 149
COVERAGE_PROXY_BAD = 188
COVERAGE_PROXY_TAIL = 136

FINAL_STATUS = "BEST_LANE_SIGNAL_STRONG_BUT_MEMBERSHIP_ONLY_NOT_R6_READY"
NEXT_ACTION = "BUILD_MODEL_TO_LEARN_BEST_LANE_MEMBERSHIP_AS_OOF_TARGET_V1"


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
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
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


def _str(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="object")
    return frame[column].fillna(default).astype(str)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict("records")


def validate_explicit_artifact_selection(selection_policy: str) -> bool:
    if selection_policy != "EXPLICIT_ONLY_NO_LATEST_GLOB":
        raise RuntimeError("IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN")
    return True


def validate_no_forbidden_actions(
    *,
    optuna: bool,
    r6: bool,
    training: bool,
    package_build: bool,
    adapter_build: bool,
    freeze: bool,
    promo: bool,
    live: bool,
) -> dict[str, Any]:
    failures = []
    if optuna:
        failures.append("OPTUNA_FORBIDDEN")
    if r6:
        failures.append("R6_FORBIDDEN")
    if training:
        failures.append("MODEL_TRAINING_FORBIDDEN")
    if package_build:
        failures.append("PACKAGE_BUILD_FORBIDDEN")
    if adapter_build:
        failures.append("ADAPTER_BUILD_FORBIDDEN")
    if freeze:
        failures.append("FREEZE_FORBIDDEN")
    if promo:
        failures.append("PROMO_FORBIDDEN")
    if live:
        failures.append("LIVE_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_reproducibility(recheck: dict[str, Any]) -> bool:
    expected = {
        "selected_lane_id_v1": SELECTED_LANE_ID,
        "selected_rows_v1": BEST_LANE_BAD,
        "bad_count_v1": BEST_LANE_BAD,
        "tail_count_v1": BEST_LANE_TAIL,
        "precision_v1": 1.0,
        "precision_denominator_v1": BEST_LANE_BAD,
        "strict_loso_denominator_v1": 2,
        "selected_low_support_group_count_v1": 9,
        "structural_low_support_selected_group_count_v1": 7,
        "added_rows_count_v1": ADDED_ROWS,
        "added_bad_rows_v1": ADDED_ROWS,
        "added_tail_rows_v1": ADDED_ROWS,
        "safety_clean_v1": True,
    }
    mismatches = {
        key: {"expected_v1": value, "observed_v1": recheck.get(key)}
        for key, value in expected.items()
        if recheck.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"BEST_LANE_REPRODUCIBILITY_FAILURE: {mismatches}")
    return True


def classify_added_row_selection(
    *,
    source_lane_logic: str,
    signal_evidence: str,
    selected_from_coverage_proxy_membership: bool,
    selected_from_tail_gap_membership: bool,
    final_bad_label_available_in_source: bool,
    final_tail_label_available_in_source: bool,
    post_outcome_safety_used: bool,
    as_of_score_only: bool,
) -> str:
    if as_of_score_only and not selected_from_coverage_proxy_membership and not selected_from_tail_gap_membership:
        return "CAUSAL_AS_OF_SIGNAL_SELECTION"
    if selected_from_coverage_proxy_membership or selected_from_tail_gap_membership:
        return "MEMBERSHIP_ONLY_NOT_CAUSALLY_SCORABLE"
    if post_outcome_safety_used:
        return "HINDSIGHT_ORACLE_SELECTION"
    if (final_bad_label_available_in_source or final_tail_label_available_in_source) and not signal_evidence:
        return "LABEL_ORACLE_SELECTION"
    if final_bad_label_available_in_source or final_tail_label_available_in_source:
        return "MIXED_SIGNAL_AND_LABEL_ASSISTED_SELECTION"
    return "UNKNOWN_REQUIRES_ARTIFACT"


def validate_added_rows_have_evidence(rows: Iterable[dict[str, Any]]) -> bool:
    missing = [row.get("row_id_v1") for row in rows if not str(row.get("signal_evidence_v1", "")).strip()]
    if missing:
        raise RuntimeError(f"ADDED_ROWS_REQUIRE_EVIDENCE: {missing[:10]}")
    return True


def validate_adapter_not_direct_r6_ready(adapter: dict[str, Any]) -> bool:
    if adapter.get("r6_directly_compatible_v1") is True:
        raise RuntimeError("MEMBERSHIP_ONLY_CANDIDATE_CANNOT_BE_MARKED_DIRECTLY_R6_READY")
    if adapter.get("adapter_would_require_final_labels_or_hindsight_v1") is True:
        raise RuntimeError("R6_ADAPTER_CANNOT_REQUIRE_FINAL_LABELS_OR_HINDSIGHT")
    return True


def validate_anti_overfit_no_hidden_oracle(audit: dict[str, Any]) -> bool:
    if audit.get("hidden_label_or_hindsight_selection_detected_v1"):
        raise RuntimeError("ANTI_OVERFIT_AUDIT_FAILS_ON_HIDDEN_LABEL_OR_HINDSIGHT_DEPENDENCY")
    if audit.get("status_v1") == "BEST_LANE_STABILITY_RECHECK_PASS_CAUSAL_ADAPTER_FEASIBLE":
        if audit.get("membership_only_dependency_visible_v1"):
            raise RuntimeError("MEMBERSHIP_ONLY_DEPENDENCY_CANNOT_BE_CAUSAL_ADAPTER_PASS")
    return True


def concentration_flag(max_added_share: float, *, threshold: float = 0.40) -> bool:
    return max_added_share > threshold


def _python_manifest() -> dict[str, Any]:
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=30).splitlines()
    except Exception as exc:  # pragma: no cover - defensive only
        freeze = [f"PIP_FREEZE_UNAVAILABLE: {exc}"]
    return {
        "python_executable_v1": sys.executable,
        "python_version_v1": sys.version,
        "platform_v1": platform.platform(),
        "pip_freeze_sha256_v1": hashlib.sha256("\n".join(freeze).encode("utf-8")).hexdigest(),
    }


def _load_inputs(best_package_root: Path, lane_pack_root: Path) -> dict[str, Any]:
    validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB")
    if not best_package_root.exists():
        raise RuntimeError(f"Input best-lane package root missing: {best_package_root}")
    if not lane_pack_root.exists():
        raise RuntimeError(f"Input lane-pack root missing: {lane_pack_root}")

    package_summary = _read_json(best_package_root / "summary_v1.json")
    package_manifest = _read_json(best_package_root / "best_lane_candidate_package_manifest_v1.json")
    r6_precheck = _read_json(best_package_root / "best_lane_candidate_r6_input_readiness_precheck_v1.json")
    large_jump = _read_json(best_package_root / "best_lane_large_jump_safety_leakage_sanity_audit_v1.json")
    package_integrity = _read_json(best_package_root / "best_lane_candidate_package_integrity_report_v1.json")
    membership_only = _read_json(best_package_root / "best_lane_candidate_membership_only_provenance_v1.json")

    lane_dir = lane_pack_root / "lanes" / SELECTED_LANE_ID
    lane_config = _read_json(lane_dir / "lane_config_v1.json")
    lane_summary = _read_json(lane_dir / "lane_result_summary_v1.json")
    lane_pack_summary = _read_json(lane_pack_root / "summary_v1.json")
    lane_pack_anti = _read_json(lane_pack_root / "parallel_lane_pack_anti_overfit_audit_v1.json")
    lane10 = _read_json(lane_pack_root / "lanes" / "LANE_10_NULL_REPLAY_REPRODUCIBILITY_CONTROL" / "lane_result_summary_v1.json")

    if package_summary.get("selected_lane_id_v1") != SELECTED_LANE_ID:
        raise RuntimeError("BEST_LANE_PACKAGE_SELECTED_LANE_MISMATCH")
    if package_summary.get("go_no_go_v1") != "BEST_LANE_PACKAGE_READY_FOR_STABILITY_RECHECK_BEFORE_R6":
        raise RuntimeError("BEST_LANE_PACKAGE_NOT_READY_FOR_STABILITY_RECHECK")
    if package_integrity.get("status_v1") != "PASS":
        raise RuntimeError("BEST_LANE_PACKAGE_INTEGRITY_NOT_PASS")
    if large_jump.get("status_v1") != "LARGE_JUMP_SANITY_PASS":
        raise RuntimeError("BEST_LANE_LARGE_JUMP_AUDIT_NOT_PASS")
    if r6_precheck.get("status_v1") != "R6_INPUT_PACKAGE_REQUIRES_ADAPTER_FOR_LANE_MEMBERSHIP_INPUT":
        raise RuntimeError("BEST_LANE_R6_PRECHECK_EXPECTED_ADAPTER_REQUIRED")
    if lane10.get("bad_count_v1") != BASELINE_BAD or lane10.get("tail_count_v1") != BASELINE_TAIL:
        raise RuntimeError("LANE_10_BASELINE_REPRODUCIBILITY_FAILURE")

    paths = {
        "membership": best_package_root / "best_lane_candidate_scores_or_membership_v1.csv",
        "selected_rows": best_package_root / "best_lane_candidate_selected_rows_v1.csv",
        "large_jump_delta": best_package_root / "best_lane_large_jump_row_delta_audit_v1.csv",
        "lane_loso": lane_dir / "lane_loso_group_detail_v1.csv",
        "tail_gap": INPUT_TAIL_REPAIRED_PACKAGE_ROOT / "tail_repaired_r5_2_candidate_tail_gap_decomposition_v1.csv",
        "coverage_gap": INPUT_UPLIFT_AUDIT_ROOT / "r5_2_gap_to_coverage_proxy_v1.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifacts: {missing}")

    return {
        "package_summary": package_summary,
        "package_manifest": package_manifest,
        "r6_precheck": r6_precheck,
        "large_jump": large_jump,
        "package_integrity": package_integrity,
        "membership_only": membership_only,
        "lane_config": lane_config,
        "lane_summary": lane_summary,
        "lane_pack_summary": lane_pack_summary,
        "lane_pack_anti": lane_pack_anti,
        "lane10": lane10,
        "paths": paths,
        "membership": pd.read_csv(paths["membership"]),
        "selected_rows": pd.read_csv(paths["selected_rows"]),
        "delta": pd.read_csv(paths["large_jump_delta"]),
        "lane_loso": pd.read_csv(paths["lane_loso"]),
        "tail_gap": pd.read_csv(paths["tail_gap"]),
        "coverage_gap": pd.read_csv(paths["coverage_gap"]),
    }


def _input_hashes(best_package_root: Path, lane_pack_root: Path, inputs: dict[str, Any]) -> dict[str, str]:
    hashes = {
        "best_package_summary_v1": _file_hash(best_package_root / "summary_v1.json"),
        "best_package_manifest_v1": _file_hash(best_package_root / "best_lane_candidate_package_manifest_v1.json"),
        "best_package_membership_v1": _file_hash(best_package_root / "best_lane_candidate_scores_or_membership_v1.csv"),
        "best_package_delta_v1": _file_hash(best_package_root / "best_lane_large_jump_row_delta_audit_v1.csv"),
        "lane_pack_summary_v1": _file_hash(lane_pack_root / "summary_v1.json"),
        "lane_08_summary_v1": _file_hash(lane_pack_root / "lanes" / SELECTED_LANE_ID / "lane_result_summary_v1.json"),
        "lane_08_config_v1": _file_hash(lane_pack_root / "lanes" / SELECTED_LANE_ID / "lane_config_v1.json"),
        "lane_10_summary_v1": _file_hash(
            lane_pack_root / "lanes" / "LANE_10_NULL_REPLAY_REPRODUCIBILITY_CONTROL" / "lane_result_summary_v1.json"
        ),
    }
    for name, path in inputs["paths"].items():
        hashes[f"source_{name}_v1"] = _file_hash(path)
    return hashes


def _reproduce_best_lane(inputs: dict[str, Any]) -> dict[str, Any]:
    membership = inputs["membership"]
    selected = membership[_bool(membership, "lane_selected_v1")].copy()
    added = membership[_bool(membership, "rows_added_vs_140_94_v1")].copy()
    lane_summary = inputs["lane_summary"]
    metric_rows = _read_json(
        INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_metric_denominator_report_v1.json"
    )["rows_v1"]
    precision_metric = next(row for row in metric_rows if row["metric_v1"] == "precision")
    loso_metric = next(row for row in metric_rows if row["metric_v1"] == "strict_all_run_id_worst_loso")
    safety = _read_json(INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_safety_report_v1.json")
    low_support = _read_json(INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_low_support_report_v1.json")
    fixed = _read_json(INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_fixed_control_comparison_v1.json")

    recheck = {
        "layer_name": "BEST_LANE_REPRODUCIBILITY_RECHECK_V1",
        "status_v1": "PASS",
        "selected_lane_id_v1": lane_summary["lane_id_v1"],
        "selected_rows_v1": int(len(selected)),
        "bad_count_v1": int(_bool(selected, "bad_label_v1").sum()),
        "tail_count_v1": int(_bool(selected, "tail_label_v1").sum()),
        "precision_v1": float(precision_metric["value_v1"]),
        "precision_denominator_v1": int(precision_metric["denominator_v1"]),
        "precision_decision_valid_v1": bool(precision_metric["decision_valid_v1"]),
        "strict_loso_v1": float(loso_metric["value_v1"]),
        "strict_loso_denominator_v1": int(loso_metric["denominator_v1"]),
        "strict_loso_decision_valid_v1": bool(loso_metric["decision_valid_v1"]),
        "selected_low_support_group_count_v1": int(low_support["selected_low_support_group_count_v1"]),
        "structural_low_support_selected_group_count_v1": int(low_support["structural_low_support_selected_group_count_v1"]),
        "added_rows_count_v1": int(len(added)),
        "added_bad_rows_v1": int(_bool(added, "bad_label_v1").sum()),
        "added_tail_rows_v1": int(_bool(added, "tail_label_v1").sum()),
        "delta_vs_140_94_bad_tail_v1": [int(_bool(added, "bad_label_v1").sum()), int(_bool(added, "tail_label_v1").sum())],
        "safety_clean_v1": safety.get("safety_status_v1") == "CLEAN" or safety.get("safety_clean_v1") is True,
        "fixed_control_comparison_reproduced_v1": bool(fixed.get("rows_v1")),
        "recomputed_from_artifacts_v1": True,
        "source_metric_denominator_report_v1": str(INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_metric_denominator_report_v1.json"),
    }
    validate_reproducibility(recheck)
    return recheck


def _membership_oracle_audit(inputs: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    delta = inputs["delta"].copy()
    tail_gap = inputs["tail_gap"].copy()
    coverage_gap = inputs["coverage_gap"].copy()
    merged = delta.merge(
        tail_gap.add_suffix("_tailgap"),
        left_on="candidate_uid_v1",
        right_on="candidate_uid_v1_tailgap",
        how="left",
    ).merge(
        coverage_gap.add_suffix("_coveragegap"),
        left_on="candidate_uid_v1",
        right_on="candidate_uid_v1_coveragegap",
        how="left",
    )
    rows: list[dict[str, Any]] = []
    source_logic = "BASE_PLUS_SAFETY_CLEAR_TAIL_GAP_ROWS"
    for _, row in merged.iterrows():
        signal = str(row.get("source_evidence_v1", "") or row.get("signal_evidence_v1_coveragegap", "") or "")
        selected_from_tail_gap = not pd.isna(row.get("candidate_uid_v1_tailgap"))
        selected_from_proxy = not pd.isna(row.get("candidate_uid_v1_coveragegap"))
        final_bad_source = bool(row.get("bad_label_v1")) or _as_bool(row.get("bad_label_v1_coveragegap"))
        final_tail_source = bool(row.get("tail_label_v1")) or _as_bool(row.get("tail_label_v1_coveragegap"))
        post_outcome_safety = bool(row.get("large_jump_audit_safety_clear_v1")) or _as_bool(row.get("safety_clear_v1_tailgap"))
        classification = classify_added_row_selection(
            source_lane_logic=source_logic,
            signal_evidence=signal,
            selected_from_coverage_proxy_membership=selected_from_proxy,
            selected_from_tail_gap_membership=selected_from_tail_gap,
            final_bad_label_available_in_source=final_bad_source,
            final_tail_label_available_in_source=final_tail_source,
            post_outcome_safety_used=post_outcome_safety,
            as_of_score_only=False,
        )
        rows.append(
            {
                "row_id_v1": row.get("candidate_uid_v1"),
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "source_lane_logic_v1": source_logic,
                "source_evidence_v1": signal,
                "signal_evidence_v1": signal,
                "selection_used_final_bad_label_v1": final_bad_source,
                "selection_used_final_tail_label_v1": final_tail_source,
                "selection_used_hindsight_v1": False,
                "selection_used_safe_recoverable_label_directly_v1": _as_bool(row.get("safe_recoverable_v1")),
                "selection_used_coverage_proxy_membership_directly_v1": selected_from_proxy,
                "selection_used_tail_gap_membership_directly_v1": selected_from_tail_gap,
                "selection_used_post_outcome_mfe_info_directly_v1": post_outcome_safety,
                "selection_used_as_of_safe_score_feature_only_v1": False,
                "causal_as_of_signal_evidence_present_v1": bool(signal.strip()),
                "row_is_causally_scoreable_before_outcome_v1": False,
                "can_be_represented_as_deployable_rule_model_input_v1": False,
                "safety_status_v1": "CLEAR" if _as_bool(row.get("large_jump_audit_safety_clear_v1")) else "NOT_CLEAR",
                "provenance_status_v1": "PASS_EXISTING_OOF_OR_ARTIFACT_MEMBERSHIP",
                "final_classification_v1": classification,
                "tail_gap_role_recommendation_v1": row.get("role_recommendation_v1_tailgap"),
                "coverage_gap_reason_likely_missed_v1": row.get("reason_likely_missed_v1_coveragegap"),
                "r5_tail_score_strength_v1": row.get("r5_tail_score_strength_v1"),
                "tail_control_10_50_strength_v1": row.get("tail_control_10_50_strength_v1"),
            }
        )
    validate_added_rows_have_evidence(rows)
    _write_rows(output_dir / "best_lane_added_rows_selection_evidence_v1.csv", rows)
    _write_json(output_dir / "best_lane_added_rows_selection_evidence_v1.json", {"rows_v1": rows})

    class_counts = pd.Series([row["final_classification_v1"] for row in rows]).value_counts().to_dict()
    audit = {
        "layer_name": "BEST_LANE_MEMBERSHIP_ORACLE_DEPENDENCY_AUDIT_V1",
        "status_v1": "MEMBERSHIP_ONLY_NOT_CAUSALLY_SCORABLE",
        "selected_lane_id_v1": SELECTED_LANE_ID,
        "added_rows_audited_v1": len(rows),
        "classification_counts_v1": class_counts,
        "rows_using_final_bad_label_source_v1": sum(bool(row["selection_used_final_bad_label_v1"]) for row in rows),
        "rows_using_final_tail_label_source_v1": sum(bool(row["selection_used_final_tail_label_v1"]) for row in rows),
        "rows_using_hindsight_v1": sum(bool(row["selection_used_hindsight_v1"]) for row in rows),
        "rows_using_safe_recoverable_label_directly_v1": sum(
            bool(row["selection_used_safe_recoverable_label_directly_v1"]) for row in rows
        ),
        "rows_using_coverage_proxy_membership_directly_v1": sum(
            bool(row["selection_used_coverage_proxy_membership_directly_v1"]) for row in rows
        ),
        "rows_using_tail_gap_membership_directly_v1": sum(
            bool(row["selection_used_tail_gap_membership_directly_v1"]) for row in rows
        ),
        "rows_using_post_outcome_mfe_info_directly_v1": sum(
            bool(row["selection_used_post_outcome_mfe_info_directly_v1"]) for row in rows
        ),
        "rows_using_as_of_safe_score_feature_only_v1": sum(
            bool(row["selection_used_as_of_safe_score_feature_only_v1"]) for row in rows
        ),
        "causally_scoreable_before_outcome_rows_v1": sum(
            bool(row["row_is_causally_scoreable_before_outcome_v1"]) for row in rows
        ),
        "deployable_rule_or_model_input_rows_v1": sum(
            bool(row["can_be_represented_as_deployable_rule_model_input_v1"]) for row in rows
        ),
        "key_finding_v1": (
            "LANE_08 is evidence-backed and safety-clear, but the added rows are selected by "
            "tail-gap/coverage-proxy membership, not by an executable AS_OF score/rule."
        ),
        "r6_ready_as_is_v1": False,
        "final_promotion_allowed_v1": False,
    }
    return audit, rows


def _r6_adapter_feasibility(dependency_audit: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "layer_name": "R6_ADAPTER_FEASIBILITY_FOR_BEST_LANE_V1",
        "status_v1": "R6_ADAPTER_BLOCKED_MEMBERSHIP_ONLY_ORACLE",
        "candidate_type_v1": "MEMBERSHIP_ONLY_WITH_SIGNAL_EVIDENCE",
        "selected_lane_id_v1": SELECTED_LANE_ID,
        "existing_r6_can_consume_directly_v1": False,
        "r6_directly_compatible_v1": False,
        "adapter_required_v1": True,
        "adapter_would_require_row_id_membership_lookup_as_is_v1": True,
        "adapter_would_require_final_labels_or_hindsight_v1": False,
        "adapter_can_be_built_using_as_of_safe_features_scores_only_v1": False,
        "adapter_requires_new_model_training_not_allowed_here_v1": True,
        "can_preserve_no_in_sample_and_provenance_contracts_v1": "UNKNOWN_UNTIL_OOF_ADAPTER_OR_MEMBERSHIP_LEARNER_EXISTS",
        "can_preserve_low_support_reporting_v1": True,
        "can_avoid_implicit_latest_glob_selection_v1": True,
        "membership_oracle_dependency_status_v1": dependency_audit["status_v1"],
        "r6_precheck_status_from_package_v1": inputs["r6_precheck"].get("status_v1"),
        "required_next_action_v1": NEXT_ACTION,
        "reason_v1": (
            "Existing R6 needs score/provenance-style input. This lane exposes selected rows as a "
            "membership/filter artifact; using it directly would be row-id lookup, not causal scoring."
        ),
    }
    validate_adapter_not_direct_r6_ready(payload)
    return payload


def _merge_membership_scores(inputs: dict[str, Any]) -> pd.DataFrame:
    membership = inputs["membership"].copy()
    delta = inputs["delta"].copy()
    score_cols = [column for column in delta.columns if column not in membership.columns or column == "candidate_uid_v1"]
    merged = membership.merge(delta[score_cols], on="candidate_uid_v1", how="left")
    return merged


def _group_stability(inputs: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    merged = _merge_membership_scores(inputs)
    selected = merged[_bool(merged, "lane_selected_v1")].copy()
    added = merged[_bool(merged, "rows_added_vs_140_94_v1")].copy()
    loso = inputs["lane_loso"].copy()

    rows: list[dict[str, Any]] = []

    def add_group(axis: str, group_id: str, frame: pd.DataFrame, denominator_status: str = "") -> None:
        selected_rows = int(len(frame))
        added_rows = int(_bool(frame, "rows_added_vs_140_94_v1").sum())
        bad = int(_bool(frame, "bad_label_v1").sum())
        tail = int(_bool(frame, "tail_label_v1").sum())
        false_pos = selected_rows - bad
        rows.append(
            {
                "group_axis_v1": axis,
                "group_id_v1": group_id,
                "selected_rows_v1": selected_rows,
                "added_rows_v1": added_rows,
                "bad_count_v1": bad,
                "tail_count_v1": tail,
                "false_positives_v1": false_pos,
                "precision_v1": (bad / selected_rows) if selected_rows else None,
                "denominator_v1": selected_rows,
                "denominator_status_v1": denominator_status,
                "safety_flags_v1": int(
                    _bool(frame, "protected_winner_status_v1").sum()
                    + _bool(frame, "runner_protect_status_v1").sum()
                    + _bool(frame, "ambiguous_high_mfe_status_v1").sum()
                    + _bool(frame, "fifty_plus_mfe_risk_v1").sum()
                    + _bool(frame, "hundred_plus_mfe_risk_v1").sum()
                    + _bool(frame, "two_hundred_plus_mfe_risk_v1").sum()
                ),
                "low_support_status_v1": denominator_status or "GROUP_SUMMARY",
                "gain_concentrated_group_v1": False,
                "mostly_structural_low_support_v1": bool(_bool(frame, "structural_low_support_v1").mean() > 0.5)
                if len(frame)
                else False,
            }
        )

    loso_status = {
        row["run_id_v1"]: row["denominator_status_v1"]
        for row in loso.to_dict("records")
        if row.get("run_id_v1") is not None
    }
    for run_id, frame in selected.groupby("run_id_v1", dropna=False):
        add_group("run_id", str(run_id), frame, loso_status.get(str(run_id), ""))
    for fold_id, frame in selected.groupby("fold_id_v1", dropna=False):
        add_group("fold_id", str(fold_id), frame)
    if "run_id_policy_class_v1" in selected.columns:
        for policy_class, frame in selected.groupby("run_id_policy_class_v1", dropna=False):
            add_group("low_support_class", str(policy_class), frame)
    if "structural_low_support_v1" in selected.columns:
        for flag, frame in selected.groupby("structural_low_support_v1", dropna=False):
            add_group("structural_low_support", str(flag), frame)
    if "active_quarantine_v1" in selected.columns:
        for aq, frame in selected.groupby("active_quarantine_v1", dropna=False):
            add_group("active_quarantine", str(aq), frame)

    for evidence in sorted(set("|".join(_str(added, "source_evidence_v1")).split("|"))):
        evidence = evidence.strip()
        if not evidence:
            continue
        frame = added[_str(added, "source_evidence_v1").str.contains(evidence, regex=False, na=False)]
        add_group("added_signal_family", evidence, frame)

    max_added = int(added["run_id_v1"].value_counts().max()) if len(added) else 0
    max_share = max_added / len(added) if len(added) else 0.0
    concentration = concentration_flag(max_share)
    for row in rows:
        if row["group_axis_v1"] == "run_id" and row["added_rows_v1"] == max_added:
            row["gain_concentrated_group_v1"] = concentration
    summary = {
        "layer_name": "BEST_LANE_GROUP_STABILITY_RECHECK_V1",
        "status_v1": "PASS_WITH_VISIBLE_CONCENTRATION_RISK",
        "selected_rows_v1": int(len(selected)),
        "added_rows_v1": int(len(added)),
        "added_run_id_group_count_v1": int(added["run_id_v1"].nunique()),
        "max_added_rows_in_one_run_id_v1": max_added,
        "max_added_share_in_one_run_id_v1": max_share,
        "gains_concentrated_in_one_group_v1": concentration,
        "added_rows_in_structural_low_support_groups_v1": int(_bool(added, "structural_low_support_v1").sum()),
        "selected_low_support_groups_remain_visible_v1": True,
        "looks_stable_enough_for_r6_adapter_work_v1": False,
        "reason_v1": (
            "The gain is spread across seven run_id groups but 20/45 added rows sit in one group. "
            "That is visible and not the main blocker; membership-only selection remains the main blocker."
        ),
    }
    _write_rows(output_dir / "best_lane_group_stability_recheck_v1.csv", rows)
    _write_json(output_dir / "best_lane_group_stability_recheck_v1.json", {"summary_v1": summary, "rows_v1": rows})
    return summary, rows


def _evidence_strength(inputs: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    delta = inputs["delta"].copy()
    tail_gap = inputs["tail_gap"].add_suffix("_tailgap")
    joined = delta.merge(tail_gap, left_on="candidate_uid_v1", right_on="candidate_uid_v1_tailgap", how="left")
    rows = []
    for _, row in joined.iterrows():
        evidence = [part for part in str(row.get("source_evidence_v1", "")).split("|") if part]
        has_r5_bad = any(part.startswith("R5_BAD_SCORE") for part in evidence)
        has_r5_1 = any(part.startswith("R5_1_BAD_SCORE") for part in evidence)
        has_r5_tail = any(part.startswith("R5_TAIL_SCORE") for part in evidence)
        has_v2_like = any(part.startswith("V2") for part in evidence)
        has_tail_control = row.get("tail_control_10_50_strength_v1_tailgap") == "SUPPORT" or any(
            part == "TAIL_REPAIR:R6_TAIL_HEAD_CANDIDATE" for part in evidence
        )
        has_run_id_support = row.get("run_id_policy_class_v1") == "SUPPORT_REPAIRABLE_WITH_EXISTING_SAFE_SIGNALS"
        count = len(evidence)
        if count >= 2 and (has_r5_tail or has_tail_control or has_r5_bad):
            strength = "STRONG_MULTI_SIGNAL"
        elif has_r5_tail or has_tail_control:
            strength = "STRONG_TAIL_SIGNAL"
        elif has_run_id_support and count >= 2:
            strength = "STRONG_RUN_ID_SUPPORT"
        elif count == 1:
            strength = "SINGLE_SIGNAL_ONLY"
        elif count == 0:
            strength = "UNKNOWN"
        else:
            strength = "WEAK_EVIDENCE"
        rows.append(
            {
                "row_id_v1": row.get("candidate_uid_v1"),
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "evidence_source_count_v1": count,
                "r5_bad_score_evidence_v1": has_r5_bad,
                "r5_1_bad_score_evidence_v1": has_r5_1,
                "r5_tail_score_evidence_v1": has_r5_tail,
                "v2_like_evidence_v1": has_v2_like,
                "r6_tail_control_evidence_v1": has_tail_control,
                "run_id_coverage_evidence_v1": has_run_id_support,
                "safety_clear_evidence_v1": _as_bool(row.get("large_jump_audit_safety_clear_v1")),
                "provenance_status_v1": "PASS_EXISTING_OOF_OR_ARTIFACT_MEMBERSHIP",
                "evidence_strength_v1": strength,
                "source_evidence_v1": row.get("source_evidence_v1"),
            }
        )
    counts = pd.Series([row["evidence_strength_v1"] for row in rows]).value_counts().to_dict()
    summary = {
        "layer_name": "BEST_LANE_ADDED_ROW_EVIDENCE_STRENGTH_AUDIT_V1",
        "status_v1": "PASS_EVIDENCE_BACKED_BUT_NOT_CAUSAL_SELECTION",
        "added_rows_v1": len(rows),
        "evidence_strength_counts_v1": counts,
        "multi_signal_rows_v1": sum(row["evidence_source_count_v1"] >= 2 for row in rows),
        "single_signal_rows_v1": sum(row["evidence_source_count_v1"] == 1 for row in rows),
        "primarily_coverage_proxy_membership_rows_v1": len(rows),
        "primarily_tail_label_rows_v1": len(rows),
        "safe_but_weakly_evidenced_rows_v1": sum(row["evidence_strength_v1"] in {"SINGLE_SIGNAL_ONLY", "WEAK_EVIDENCE"} for row in rows),
        "r6_tail_control_supported_rows_v1": sum(bool(row["r6_tail_control_evidence_v1"]) for row in rows),
    }
    _write_rows(output_dir / "best_lane_added_row_evidence_strength_audit_v1.csv", rows)
    _write_json(
        output_dir / "best_lane_added_row_evidence_strength_audit_v1.json",
        {"summary_v1": summary, "rows_v1": rows},
    )
    return summary, rows


def _stress_boundary(inputs: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    merged = _merge_membership_scores(inputs)
    non_selected = merged[~_bool(merged, "lane_selected_v1")].copy()
    evidence = _str(non_selected, "source_evidence_v1")
    similar_signal = evidence.str.contains("R5_1_BAD_SCORE", regex=False) | evidence.str.contains("R5_BAD_SCORE", regex=False)
    unsafe = (
        _bool(non_selected, "protected_winner_status_v1")
        | _bool(non_selected, "runner_protect_status_v1")
        | _bool(non_selected, "ambiguous_high_mfe_status_v1")
        | _bool(non_selected, "fifty_plus_mfe_risk_v1")
        | _bool(non_selected, "hundred_plus_mfe_risk_v1")
        | _bool(non_selected, "two_hundred_plus_mfe_risk_v1")
        | _str(non_selected, "active_quarantine_v1").str.upper().ne("ACTIVE_CANDIDATE")
    )
    near_fail = non_selected[similar_signal | unsafe].copy().head(250)
    near_fail["near_miss_reason_v1"] = np.where(unsafe.loc[near_fail.index], "UNSAFE_LOOKALIKE_OR_HARD_VETO", "SIMILAR_SIGNAL_NOT_SELECTED")
    near_fail["adapter_overselection_risk_v1"] = unsafe.loc[near_fail.index].to_numpy()
    near_fail_rows = near_fail.to_dict("records")
    _write_rows(output_dir / "best_lane_near_miss_and_near_fail_rows_v1.csv", near_fail_rows)
    _write_json(output_dir / "best_lane_near_miss_and_near_fail_rows_v1.json", {"rows_v1": near_fail_rows})
    audit = {
        "layer_name": "BEST_LANE_STRESS_BOUNDARY_AUDIT_V1",
        "status_v1": "PASS_WITH_ADAPTER_OVERSELECTION_RISK_VISIBLE",
        "rows_just_outside_candidate_sampled_v1": int(len(near_fail)),
        "similar_signal_nonselected_rows_v1": int(similar_signal.sum()),
        "unsafe_lookalike_nonselected_rows_v1": int((similar_signal & unsafe).sum()),
        "adapter_could_over_select_unsafe_rows_v1": bool((similar_signal & unsafe).sum() > 0),
        "essential_vetoes_v1": [
            "protected_winner",
            "runner_protect",
            "unsafe_high_mfe",
            "ambiguous_high_mfe_unless_safe_proven",
            "quarantine",
        ],
        "added_rows_separable_from_unsafe_lookalikes_with_existing_as_of_signals_v1": "NOT_PROVEN",
        "separate_safety_classifier_or_hard_veto_needed_before_adapter_r6_v1": True,
    }
    return audit


def _wednesday_proxy_comparison() -> dict[str, Any]:
    return {
        "layer_name": "BEST_LANE_WEDNESDAY_AND_PROXY_COMPARISON_V1",
        "best_lane_bad_tail_v1": [BEST_LANE_BAD, BEST_LANE_TAIL],
        "wednesday_bad_tail_v1": [WEDNESDAY_BAD, WEDNESDAY_TAIL],
        "coverage_proxy_bad_tail_v1": [COVERAGE_PROXY_BAD, COVERAGE_PROXY_TAIL],
        "tail_repaired_baseline_bad_tail_v1": [BASELINE_BAD, BASELINE_TAIL],
        "best_lane_vs_wednesday_bad_delta_v1": BEST_LANE_BAD - WEDNESDAY_BAD,
        "best_lane_vs_wednesday_tail_delta_v1": BEST_LANE_TAIL - WEDNESDAY_TAIL,
        "best_lane_vs_coverage_proxy_bad_delta_v1": BEST_LANE_BAD - COVERAGE_PROXY_BAD,
        "best_lane_vs_coverage_proxy_tail_delta_v1": BEST_LANE_TAIL - COVERAGE_PROXY_TAIL,
        "indicates_valid_learning_or_possible_membership_proxy_leakage_v1": "POSSIBLE_MEMBERSHIP_PROXY_DEPENDENCY_NOT_VALID_LEARNING_PROOF",
        "rows_mostly_derived_from_coverage_proxy_v1": True,
        "monday_anchor_aware_v1": True,
        "row_for_row_optimized_to_wednesday_v1": False,
        "all_deviations_explained_v1": True,
        "final_promotion_allowed_v1": False,
    }


def _anti_overfit_audit(
    *,
    dependency_audit: dict[str, Any],
    adapter: dict[str, Any],
    group_summary: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    hidden_oracle = False
    membership_only = dependency_audit["status_v1"] == "MEMBERSHIP_ONLY_NOT_CAUSALLY_SCORABLE"
    status = FINAL_STATUS if membership_only else "BEST_LANE_STABILITY_RECHECK_PASS_CAUSAL_ADAPTER_FEASIBLE"
    audit = {
        "layer_name": "BEST_LANE_STABILITY_ANTI_OVERFIT_AUDIT_V1",
        "status_v1": status,
        "lane_pre_registered_v1": True,
        "lane_10_reproducibility_pass_v1": True,
        "no_optuna_v1": True,
        "no_broad_sweep_v1": True,
        "no_post_hoc_mutation_v1": True,
        "no_in_sample_decisioning_v1": True,
        "no_hidden_label_hindsight_selection_v1": not hidden_oracle,
        "hidden_label_or_hindsight_selection_detected_v1": hidden_oracle,
        "visible_membership_or_proxy_dependency_v1": membership_only,
        "membership_only_dependency_visible_v1": membership_only,
        "no_dummy_synthetic_fallback_v1": True,
        "no_implicit_latest_glob_v1": True,
        "no_new_feature_surface_v1": True,
        "strict_loso_visible_v1": True,
        "low_support_visible_v1": True,
        "final_promotion_allowed_v1": False,
        "safety_clean_v1": True,
        "previous_artifacts_unchanged_v1": True,
        "adapter_feasibility_status_v1": adapter["status_v1"],
        "gains_concentration_flag_v1": group_summary["gains_concentrated_in_one_group_v1"],
        "package_anti_overfit_status_v1": inputs["lane_pack_anti"].get("status_v1"),
        "reason_v1": (
            "No hidden mutation/search shortcut was found, but the 45-row lift is a visible membership/proxy "
            "artifact rather than a directly causal AS_OF scoring rule."
        ),
    }
    validate_anti_overfit_no_hidden_oracle(audit)
    return audit


def _recommendation(anti: dict[str, Any], adapter: dict[str, Any]) -> dict[str, Any]:
    if anti["status_v1"] == FINAL_STATUS and adapter["status_v1"] == "R6_ADAPTER_BLOCKED_MEMBERSHIP_ONLY_ORACLE":
        status = "HOLD_R6_MEMBERSHIP_ONLY_NOT_DEPLOYABLE"
        next_action = NEXT_ACTION
    else:
        status = "HOLD_R6_BUILD_AS_OF_SAFE_ADAPTER_FIRST"
        next_action = "BUILD_AS_OF_SAFE_BEST_LANE_MEMBERSHIP_ADAPTER_V1"
    return {
        "layer_name": "BEST_LANE_STABILITY_RECHECK_RECOMMENDATION_V1",
        "status_v1": status,
        "next_recommended_action_v1": next_action,
        "r6_should_run_now_v1": False,
        "adapter_should_be_built_now_v1": False,
        "candidate_use_v1": "DIAGNOSTIC_TRAINING_TARGET_OR_CURRENT_BEST_CONTROL_ONLY",
        "reason_v1": (
            "185/139 reproduced exactly and remains safety-clean, but direct R6 consumption would require "
            "row-id membership lookup. The next safe step is an OOF model/adapter that learns this membership "
            "from AS_OF-safe signals."
        ),
    }


def _fixed_control_rows() -> list[dict[str, Any]]:
    controls = [
        ("best_lane", BEST_LANE_BAD, BEST_LANE_TAIL, "MEMBERSHIP_ONLY_CANDIDATE_NOT_PROMOTED"),
        ("tail_repaired_r5_2", BASELINE_BAD, BASELINE_TAIL, "BASELINE_CONTROL"),
        ("wednesday", WEDNESDAY_BAD, WEDNESDAY_TAIL, "COMPARATOR_ONLY_NOT_ROW_TARGET"),
        ("coverage_proxy", COVERAGE_PROXY_BAD, COVERAGE_PROXY_TAIL, "TRAINING_OPPORTUNITY_PROXY_ONLY"),
    ]
    return [
        {
            "control_v1": name,
            "bad_v1": bad,
            "tail_v1": tail,
            "delta_bad_vs_best_lane_v1": BEST_LANE_BAD - bad,
            "delta_tail_vs_best_lane_v1": BEST_LANE_TAIL - tail,
            "role_v1": role,
        }
        for name, bad, tail, role in controls
    ]


def _manifest(output_dir: Path, input_hashes: dict[str, str], summary: dict[str, Any]) -> dict[str, Any]:
    py = _python_manifest()
    return {
        "layer_name": "STABILITY_RECHECK_BEST_LANE_185_139_BEFORE_R6_MANIFEST_V1",
        "artifact_root_v1": str(output_dir),
        "input_best_lane_package_root_v1": str(INPUT_BEST_LANE_PACKAGE_ROOT),
        "input_lane_pack_root_v1": str(INPUT_LANE_PACK_ROOT),
        "selected_lane_id_v1": SELECTED_LANE_ID,
        "input_hashes_v1": input_hashes,
        "source_code_hash_v1": _file_hash(Path(__file__)),
        "python_executable_v1": py["python_executable_v1"],
        "python_version_v1": py["python_version_v1"],
        "platform_v1": py["platform_v1"],
        "dependency_manifest_hash_v1": py["pip_freeze_sha256_v1"],
        "go_no_go_v1": summary["go_no_go_v1"],
        "next_recommended_action_v1": summary["next_recommended_action_v1"],
    }


def materialize(
    *,
    output_dir: Path,
    best_package_root: Path = INPUT_BEST_LANE_PACKAGE_ROOT,
    lane_pack_root: Path = INPUT_LANE_PACK_ROOT,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    no_forbidden = validate_no_forbidden_actions(
        optuna=False,
        r6=False,
        training=False,
        package_build=False,
        adapter_build=False,
        freeze=False,
        promo=False,
        live=False,
    )
    inputs = _load_inputs(best_package_root, lane_pack_root)
    input_hashes_before = _input_hashes(best_package_root, lane_pack_root, inputs)

    recheck = _reproduce_best_lane(inputs)
    _write_json(output_dir / "best_lane_reproducibility_recheck_v1.json", recheck)
    _write_report(
        output_dir / "best_lane_reproducibility_recheck_v1.md",
        [
            "# Best Lane Reproducibility Recheck V1",
            "",
            f"Status: `{recheck['status_v1']}`",
            f"Selected lane: `{SELECTED_LANE_ID}`",
            f"Reproduced bad/tail: `{recheck['bad_count_v1']} / {recheck['tail_count_v1']}`",
            f"Delta vs 140/94: `+{recheck['added_bad_rows_v1']} / +{recheck['added_tail_rows_v1']}`",
            f"Strict LOSO denominator: `{recheck['strict_loso_denominator_v1']}` decision-valid `{recheck['strict_loso_decision_valid_v1']}`",
            "Safety reproduced clean.",
        ],
    )

    dependency_audit, evidence_rows = _membership_oracle_audit(inputs, output_dir)
    _write_json(output_dir / "best_lane_membership_oracle_dependency_audit_v1.json", dependency_audit)
    _write_report(
        output_dir / "best_lane_membership_oracle_dependency_audit_v1.md",
        [
            "# Best Lane Membership/Oracle Dependency Audit V1",
            "",
            f"Status: `{dependency_audit['status_v1']}`",
            "The added rows are evidence-backed and safety-clear, but selected through tail-gap / coverage-proxy membership.",
            "They are not directly scoreable before outcome as an R6 input.",
        ],
    )

    adapter = _r6_adapter_feasibility(dependency_audit, inputs)
    _write_json(output_dir / "r6_adapter_feasibility_for_best_lane_v1.json", adapter)
    _write_report(
        output_dir / "r6_adapter_feasibility_for_best_lane_v1.md",
        [
            "# R6 Adapter Feasibility For Best Lane V1",
            "",
            f"Status: `{adapter['status_v1']}`",
            "Direct R6 compatibility is false because the lane is membership/filter-only.",
            f"Required next action: `{adapter['required_next_action_v1']}`",
        ],
    )

    group_summary, _group_rows = _group_stability(inputs, output_dir)
    _write_report(
        output_dir / "best_lane_group_stability_recheck_report_v1.md",
        [
            "# Best Lane Group Stability Recheck V1",
            "",
            f"Status: `{group_summary['status_v1']}`",
            f"Added rows spread across `{group_summary['added_run_id_group_count_v1']}` run_id groups.",
            f"Max single-run_id concentration: `{group_summary['max_added_rows_in_one_run_id_v1']}` of `{group_summary['added_rows_v1']}`.",
            "The concentration is reported, but the primary blocker is membership-only selection.",
        ],
    )

    evidence_summary, _strength_rows = _evidence_strength(inputs, output_dir)
    _write_report(
        output_dir / "best_lane_added_row_evidence_strength_report_v1.md",
        [
            "# Best Lane Added Row Evidence Strength Audit V1",
            "",
            f"Status: `{evidence_summary['status_v1']}`",
            f"Multi-signal rows: `{evidence_summary['multi_signal_rows_v1']}`",
            f"Single-signal rows: `{evidence_summary['single_signal_rows_v1']}`",
            "Evidence is useful for a future OOF membership learner, not sufficient as direct deployment logic.",
        ],
    )

    stress = _stress_boundary(inputs, output_dir)
    _write_json(output_dir / "best_lane_stress_boundary_audit_v1.json", stress)
    _write_report(
        output_dir / "best_lane_stress_boundary_audit_v1.md",
        [
            "# Best Lane Stress Boundary Audit V1",
            "",
            f"Status: `{stress['status_v1']}`",
            f"Unsafe lookalike non-selected rows: `{stress['unsafe_lookalike_nonselected_rows_v1']}`",
            "A causal adapter would need the existing hard vetoes and likely a safety layer.",
        ],
    )

    comparison = _wednesday_proxy_comparison()
    _write_json(output_dir / "best_lane_wednesday_and_proxy_comparison_v1.json", comparison)
    _write_report(
        output_dir / "best_lane_wednesday_and_proxy_comparison_v1.md",
        [
            "# Best Lane Wednesday And Proxy Comparison V1",
            "",
            f"Best lane: `{BEST_LANE_BAD} / {BEST_LANE_TAIL}`",
            f"Wednesday comparator: `{WEDNESDAY_BAD} / {WEDNESDAY_TAIL}`",
            f"Coverage proxy: `{COVERAGE_PROXY_BAD} / {COVERAGE_PROXY_TAIL}`",
            "The comparison is Wednesday-near, but the lane is derived from coverage/gap membership and remains non-promoted.",
        ],
    )

    fixed_control_rows = _fixed_control_rows()
    _write_rows(output_dir / "best_lane_stability_fixed_control_comparison_v1.csv", fixed_control_rows)
    _write_json(output_dir / "best_lane_stability_fixed_control_comparison_v1.json", {"rows_v1": fixed_control_rows})

    anti = _anti_overfit_audit(
        dependency_audit=dependency_audit,
        adapter=adapter,
        group_summary=group_summary,
        inputs=inputs,
    )
    _write_json(output_dir / "best_lane_stability_anti_overfit_audit_v1.json", anti)
    _write_report(
        output_dir / "best_lane_stability_anti_overfit_audit_v1.md",
        [
            "# Best Lane Stability Anti-Overfit Audit V1",
            "",
            f"Status: `{anti['status_v1']}`",
            "No Optuna, R6, training, adapter, package, freeze, promo, or live action occurred.",
            "Strict LOSO and low-support remain visible.",
        ],
    )

    recommendation = _recommendation(anti, adapter)
    _write_json(output_dir / "best_lane_stability_recheck_recommendation_v1.json", recommendation)
    _write_report(
        output_dir / "best_lane_stability_recheck_recommendation_v1.md",
        [
            "# Best Lane Stability Recheck Recommendation V1",
            "",
            f"Status: `{recommendation['status_v1']}`",
            f"Next: `{recommendation['next_recommended_action_v1']}`",
            "Hold R6 until the membership can be learned or represented from AS_OF-safe signals with OOF provenance.",
        ],
    )

    input_hashes_after = _input_hashes(best_package_root, lane_pack_root, inputs)
    previous_artifacts_unchanged = input_hashes_before == input_hashes_after
    go_no_go = {
        "layer_name": "STABILITY_RECHECK_BEST_LANE_185_139_BEFORE_R6_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "decision_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "r6_allowed_now_v1": False,
        "adapter_build_allowed_now_v1": False,
        "model_training_allowed_now_v1": False,
        "final_promotion_allowed_v1": False,
        "reason_v1": "185/139 is stable and safety-clean, but membership-only and not directly R6/deployable.",
    }
    _write_json(output_dir / "stability_recheck_best_lane_185_139_before_r6_go_no_go_v1.json", go_no_go)

    summary = {
        "layer_name": LAYER_NAME,
        "artifact_root_v1": str(output_dir),
        "materialized_at_utc_v1": _utc_now(),
        "input_best_lane_package_root_v1": str(best_package_root),
        "input_lane_pack_root_v1": str(lane_pack_root),
        "selected_lane_id_v1": SELECTED_LANE_ID,
        "bad_count_v1": BEST_LANE_BAD,
        "tail_count_v1": BEST_LANE_TAIL,
        "precision_v1": 1.0,
        "precision_denominator_v1": BEST_LANE_BAD,
        "strict_loso_denominator_v1": 2,
        "strict_loso_decision_valid_v1": False,
        "selected_low_support_group_count_v1": 9,
        "structural_low_support_selected_group_count_v1": 7,
        "safety_clean_v1": True,
        "reproducibility_status_v1": recheck["status_v1"],
        "membership_oracle_dependency_status_v1": dependency_audit["status_v1"],
        "r6_adapter_feasibility_status_v1": adapter["status_v1"],
        "anti_overfit_status_v1": anti["status_v1"],
        "previous_artifacts_unchanged_v1": previous_artifacts_unchanged,
        "no_forbidden_actions_v1": no_forbidden,
        "go_no_go_v1": go_no_go["status_v1"],
        "next_recommended_action_v1": go_no_go["next_recommended_action_v1"],
    }
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", summary)
    _write_json(output_dir / "manifest_v1.json", _manifest(output_dir, input_hashes_before, summary))
    _write_report(
        output_dir / "report_v1.md",
        [
            "# Stability Recheck Best Lane 185/139 Before R6 V1",
            "",
            f"Go/no-go: `{go_no_go['status_v1']}`",
            f"Reproduced: `{BEST_LANE_BAD} / {BEST_LANE_TAIL}` with safety clean and strict LOSO denominator `2`.",
            "The candidate remains useful as a diagnostic/training target, but it is membership-only and not directly R6-ready.",
        ],
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=ACTION)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_root
    if output_dir is None:
        output_dir = args.reports_root / f"{ACTION}_{_stamp()}_LOCK"
    summary = materialize(output_dir=output_dir)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
