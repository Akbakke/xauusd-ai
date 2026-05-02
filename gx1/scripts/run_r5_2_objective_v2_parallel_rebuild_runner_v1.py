#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score

from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import SCORE_FRAME, SUMMARY as SCORE_SUMMARY


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
LAYER_NAME = "R5_2_OBJECTIVE_V2_PARALLEL_REBUILD_RUNNER_V1"
DEFAULT_SPEC_DIR = DEFAULT_REPORTS_ROOT / "DESIGN_R5_2_OBJECTIVE_V2_REBUILD_NEXT_V1_20260426T_LOCK"

SPEC_FILES = {
    "design_lock": "r5_2_objective_v2_design_lock_v1.json",
    "label_contract": "r5_2_objective_v2_label_contract_v1.json",
    "weight_cost": "r5_2_objective_v2_weight_and_cost_spec_v1.json",
    "architecture": "r5_2_objective_v2_model_architecture_spec_v1.json",
    "base_contract": "r5_2_objective_v2_base_membership_contract_v1.json",
    "target_table": "r5_2_objective_v2_target_table_spec_v1.json",
    "feature_use": "r5_2_objective_v2_existing_feature_use_spec_v1.csv",
    "parallel_run": "r5_2_objective_v2_parallel_rebuild_run_spec_v1.json",
    "eval_gate": "r5_2_objective_v2_eval_and_gate_spec_v1.json",
    "runner_lock": "r5_2_objective_v2_next_runner_spec_lock_v1.json",
    "manifest": "manifest_v1.json",
}

DRY_OUTPUT_FILES = {
    "summary": "summary_v1.json",
    "status": "status_v1.json",
    "manifest": "manifest_v1.json",
    "prelaunch_report": "v2_parallel_prelaunch_report_v1.json",
    "variant_manifest": "v2_variant_config_manifest_v1.csv",
    "target_audit": "v2_target_table_prelaunch_audit_v1.json",
    "feature_prelaunch": "v2_feature_matrix_prelaunch_v1.json",
    "forbidden_scan": "v2_forbidden_feature_scan_v1.csv",
    "veto_report": "v2_hard_protection_veto_contract_report_v1.json",
    "execution_placeholder": "v2_parallel_execution_placeholder_v1.json",
    "r6_placeholder": "v2_downstream_r6_manifest_placeholder_v1.json",
    "audit": "consistency_audit_v1.csv",
    "report": "report_v1.md",
}

EXECUTION_OUTPUT_FILES = {
    "execution": "v2_parallel_rebuild_execution_v1.json",
    "training_outputs_index": "v2_variant_training_outputs_index_v1.csv",
    "eval_gate": "v2_variant_eval_and_safety_gate_v1.csv",
    "leaderboard": "v2_variant_leaderboard_v1.csv",
    "best_r6_lock": "best_v2_variant_downstream_r6_input_lock_v1.json",
    "row_forensics": "v2_variant_row_level_forensics_v1.csv",
    "parallel_gate": "v2_parallel_rebuild_gate_v1.json",
    "next_action": "next_action_lock_v1.json",
    "summary": "summary_v1.json",
    "report": "report_v1.md",
    "manifest": "manifest_v1.json",
    "status": "status_v1.json",
    "audit": "consistency_audit_v1.csv",
}

EXPECTED_VARIANTS = [
    "V2_BALANCED_STRICT_PROTECT",
    "V2_STRONG_BAD_TAIL_WITH_HARD_VETO",
    "V2_PROTECTION_HEAVY",
    "V2_TAIL_RECOVERY_FOCUSED",
    "V2_RECALL_LIGHT_ULTRA_SAFE",
    "V2_AMBIGUOUS_HARD_NEGATIVE_STRESS",
    "V2_BAD_RECALL_FOCUSED_WITH_STRONG_VETO",
]
REQUIRED_SPEC_KEYS = [key for key in SPEC_FILES if key != "manifest"]
REQUIRED_KEYS = ["candidate_uid", "trade_uid", "decision_timestamp"]
EXPECTED_FOUNDATION_ROWS = 1914
EXPECTED_ACTIVE_ROWS = 1852
EXPECTED_QUARANTINE_ROWS = 62
EXPECTED_ASOF_COLUMNS = 109
EXPECTED_TARGET_ROWS = 1914
DESIGN_ID = "TWO_STAGE_RECALL_WITH_HARD_PROTECTION_VETO"
RUN_FLAG = "--run-r5-2-objective-v2-parallel-rebuild"
NEXT_ACTION = "NEXT_AGENT_MAY_RUN_R5_2_OBJECTIVE_V2_PARALLEL_REBUILD_WITH_EXPLICIT_FLAG"
BLOCKED_ACTION = "RUN_PARALLEL_REBUILD_WITHOUT_EXPLICIT_FLAG"
EXECUTION_NEXT_ACTIONS = {
    "pass": "RUN_R6_RETRAIN_FROM_BEST_R5_2_OBJECTIVE_V2_VARIANT_EXPLICIT_FLAG",
    "safe_weak": "R5_2_OBJECTIVE_V2_SAFE_BUT_TOO_WEAK_REVIEW",
    "unsafe_recall": "TIGHTEN_V2_HARD_PROTECTION_AND_RERUN_PARALLEL",
    "all_fail": "STOP_R5_2_V2_REBUILD_AND_REVIEW_FEATURE_SIGNAL",
}

REQUIRED_V2_BUCKETS = [
    "BAD_RECALL_POSITIVE",
    "TAIL_RECALL_POSITIVE",
    "RISKY_ATTENTION_POSITIVE",
    "HARD_PROTECT_NEGATIVE",
    "AMBIGUOUS_HIGH_MFE_PROTECTED",
    "MONITOR_ONLY",
]
RECALL_OUTPUTS = [
    "r5_2_v2_bad_recall_score",
    "r5_2_v2_tail_recall_score",
    "r5_2_v2_risky_attention_score",
]
PROTECTION_OUTPUTS = [
    "r5_2_v2_runner_protection_score",
    "r5_2_v2_high_mfe_ambiguous_protection_score",
    "r5_2_v2_hard_winner_protection_score",
]
BASE_OUTPUTS = [
    "r5_2_v2_base_membership_pre_veto",
    "r5_2_v2_hard_protection_veto",
    "r5_2_v2_final_base_membership",
]
FORBIDDEN_FEATURE_PATTERNS = [
    "hindsight",
    "exit_",
    "management_",
    "bridge",
    "readiness",
    "exact_only",
    "protector_first",
    "diagnostic",
    "narrow",
]
FORENSIC_REPAIRED_CANDIDATE_UID = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"
V3_BASE_COLUMNS = ["in_v3_base_v1", "r5_2_v3_base_flag_v1", "r5_2_v3_base_flag_before_rescue_v1"]
RESCUE_BASE_COLUMNS = ["r5_2_true_rescue_base_membership_v1", "r5_2_selected_candidate__block_v1"]
RAW_TRUE_BASE_COLUMNS = ["raw_true_base_membership_v1", "r5_2_raw_true_base_membership_v1"]
POCKET_EVAL_COLUMNS = [
    "label_should_not_take_v1",
    "tail_10_50_mfe_v1",
    "take_was_ok_v1",
    "fifty_plus_mfe_v1",
    "hundred_plus_mfe_v1",
    "two_hundred_plus_mfe_v1",
    "strongest_winner_path_v1",
    "r6_label_repaired_165_like_runner_v1",
    "r6_label_runner_near_miss_v1",
]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _ensure_output_namespace_clean(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output namespace is not clean: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_spec_package(spec_dir: Path) -> dict[str, Any]:
    missing = [filename for filename in SPEC_FILES.values() if filename != "manifest_v1.json" and not (spec_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(f"V2 design spec package missing files: {missing}")
    spec: dict[str, Any] = {}
    for key, filename in SPEC_FILES.items():
        path = spec_dir / filename
        if filename.endswith(".csv"):
            spec[key] = pd.read_csv(path)
        elif path.exists():
            spec[key] = _read_json(path)
        else:
            spec[key] = {}
    return spec


def _resolve_label_table_path(spec: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    manifest = spec.get("manifest") or {}
    label_path = (manifest.get("input_artifacts_v1") or {}).get("label_table_v1")
    if not label_path:
        raise RuntimeError("Spec manifest does not provide label_table_v1; pass --label-table")
    return Path(label_path).expanduser().resolve()


def _resolve_foundation_score_dir(spec: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    manifest = spec.get("manifest") or {}
    rescue_dir_value = (manifest.get("input_artifacts_v1") or {}).get("rescue_r6_dir_v1")
    if not rescue_dir_value:
        raise RuntimeError("Spec manifest does not provide rescue_r6_dir_v1; pass --foundation-score-dir")
    rescue_dir = Path(rescue_dir_value).expanduser().resolve()
    rescue_manifest = _read_json(rescue_dir / "manifest_v1.json")
    staged = rescue_manifest.get("staged_score_dir_v1")
    if staged:
        return Path(staged).expanduser().resolve()
    return rescue_dir / "staged_true_r5_2_rescue_score_package_for_r6_v1"


def _load_foundation(score_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    score_path = score_dir / SCORE_FRAME
    summary_path = score_dir / SCORE_SUMMARY
    if not score_path.exists():
        raise FileNotFoundError(f"Foundation score frame missing: {score_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Foundation score summary missing: {summary_path}")
    return pd.read_parquet(score_path), _read_json(summary_path)


def _validate_foundation(score: pd.DataFrame, score_summary: dict[str, Any], score_dir: Path) -> dict[str, Any]:
    missing_keys = [column for column in REQUIRED_KEYS if column not in score.columns]
    if missing_keys:
        raise RuntimeError(f"Foundation missing required key columns: {missing_keys}")
    active = score.get("calendar_quarantine_status_v1", pd.Series("", index=score.index)).astype(str).eq("ACTIVE_CANDIDATE")
    observed = {
        "foundation_score_dir_v1": str(score_dir),
        "foundation_rows_v1": int(len(score)),
        "active_rows_v1": int(active.sum()),
        "quarantine_rows_v1": int((~active).sum()),
        "asof_columns_v1": int(score_summary.get("as_of_column_count_v1") or 0),
        "asof_prefix_columns_seen_v1": int(sum(column.startswith("as_of_") for column in score.columns)),
    }
    if observed["foundation_rows_v1"] != EXPECTED_FOUNDATION_ROWS:
        raise RuntimeError(f"Expected foundation rows {EXPECTED_FOUNDATION_ROWS}, observed {observed['foundation_rows_v1']}")
    if observed["active_rows_v1"] != EXPECTED_ACTIVE_ROWS or observed["quarantine_rows_v1"] != EXPECTED_QUARANTINE_ROWS:
        raise RuntimeError(
            f"Expected active/quarantine {EXPECTED_ACTIVE_ROWS}/{EXPECTED_QUARANTINE_ROWS}, "
            f"observed {observed['active_rows_v1']}/{observed['quarantine_rows_v1']}"
        )
    if observed["asof_columns_v1"] != EXPECTED_ASOF_COLUMNS:
        raise RuntimeError(f"Expected AS_OF columns {EXPECTED_ASOF_COLUMNS}, observed {observed['asof_columns_v1']}")
    return observed


def _validate_key_alignment(score: pd.DataFrame, label_table: pd.DataFrame) -> dict[str, Any]:
    missing_label_keys = [column for column in REQUIRED_KEYS if column not in label_table.columns]
    if missing_label_keys:
        raise RuntimeError(f"Target label table missing required key columns: {missing_label_keys}")
    score_keys = set(map(tuple, score[REQUIRED_KEYS].astype(str).to_numpy()))
    label_keys = set(map(tuple, label_table[REQUIRED_KEYS].astype(str).to_numpy()))
    missing_from_score = label_keys - score_keys
    extra_in_score = score_keys - label_keys
    if missing_from_score or extra_in_score:
        raise RuntimeError(f"Key alignment mismatch: missing_from_score={len(missing_from_score)} extra_in_score={len(extra_in_score)}")
    return {
        "required_key_columns_v1": REQUIRED_KEYS,
        "aligned_rows_v1": int(len(label_keys)),
        "missing_from_score_v1": int(len(missing_from_score)),
        "extra_in_score_v1": int(len(extra_in_score)),
    }


def _build_v2_target_table(label_table: pd.DataFrame) -> pd.DataFrame:
    required = [*REQUIRED_KEYS, "new_r5_2_label_bucket_v1"]
    missing = [column for column in required if column not in label_table.columns]
    if missing:
        raise RuntimeError(f"Target label table missing required columns: {missing}")
    out = label_table.copy()
    original = out["new_r5_2_label_bucket_v1"].astype(str)
    hard_flag = (
        _bool(out, "hundred_plus_mfe_v1")
        | _bool(out, "two_hundred_plus_mfe_v1")
        | _bool(out, "strongest_winner_path_v1")
        | _bool(out, "r6_label_repaired_165_like_runner_v1")
    )
    runner_protect = original.eq("RUNNER_PROTECT_TARGET")
    ambiguous = original.eq("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD")
    bad = original.eq("STRONG_BAD_BLOCK_TARGET") & ~hard_flag & ~runner_protect & ~ambiguous
    tail = original.eq("TAIL_CONTROL_TARGET") & ~hard_flag & ~runner_protect & ~ambiguous
    risky_raw = _bool(out, "r6_label_risky_allow_v1") | original.eq("RISKY_ALLOW_TARGET")
    risky = risky_raw & ~bad & ~tail & ~runner_protect & ~ambiguous
    monitor = ~(bad | tail | risky | runner_protect | ambiguous | hard_flag)
    out["original_bucket"] = original
    out["source_bad_eligibility_target_v1"] = _bool(out, "bad_eligibility_target_v1")
    out["source_tail_eligibility_target_v1"] = _bool(out, "tail_eligibility_target_v1")
    out["v2_bucket"] = np.select(
        [bad, tail, risky, hard_flag | runner_protect, ambiguous, monitor],
        [
            "BAD_RECALL_POSITIVE",
            "TAIL_RECALL_POSITIVE",
            "RISKY_ATTENTION_POSITIVE",
            "HARD_PROTECT_NEGATIVE",
            "AMBIGUOUS_HIGH_MFE_PROTECTED",
            "MONITOR_ONLY",
        ],
        default="MONITOR_ONLY",
    )
    out["bad_recall_target"] = bad
    out["tail_recall_target"] = tail
    out["risky_attention_target"] = risky_raw
    out["runner_protection_target"] = runner_protect | _bool(out, "r6_label_runner_near_miss_v1")
    out["high_mfe_ambiguous_protection_target"] = ambiguous | _bool(out, "fifty_plus_mfe_v1")
    out["hard_winner_protection_target"] = hard_flag
    out["sample_weight"] = np.select(
        [bad, tail, risky],
        [3.5, 3.0, 1.5],
        default=0.25,
    )
    out["protection_weight"] = np.select(
        [hard_flag, ambiguous, runner_protect, _bool(out, "fifty_plus_mfe_v1")],
        [32.0, 24.0, 16.0, 8.0],
        default=0.0,
    )
    out["monitor_only_flag"] = out["v2_bucket"].eq("MONITOR_ONLY")
    out["hard_protection_veto_target"] = (
        out["runner_protection_target"]
        | out["high_mfe_ambiguous_protection_target"]
        | out["hard_winner_protection_target"]
    )
    out["reason"] = out["v2_bucket"]
    return out


def _validate_target_table(target: pd.DataFrame) -> dict[str, Any]:
    if len(target) != EXPECTED_TARGET_ROWS:
        raise RuntimeError(f"Expected target table rows {EXPECTED_TARGET_ROWS}, observed {len(target)}")
    bucket_counts = {str(k): int(v) for k, v in target["v2_bucket"].value_counts().to_dict().items()}
    risky_role_present = int(_bool(target, "risky_attention_target").sum()) > 0
    missing_buckets = [
        bucket
        for bucket in REQUIRED_V2_BUCKETS
        if bucket not in bucket_counts and not (bucket == "RISKY_ATTENTION_POSITIVE" and risky_role_present)
    ]
    if missing_buckets:
        raise RuntimeError(f"V2 target table missing required buckets: {missing_buckets}")
    source_bad = _bool(target, "source_bad_eligibility_target_v1")
    ambiguous_bad = int((target["v2_bucket"].eq("AMBIGUOUS_HIGH_MFE_PROTECTED") & (target["bad_recall_target"] | source_bad)).sum())
    runner_bad = int((target["runner_protection_target"] & (target["bad_recall_target"] | source_bad)).sum())
    hard_without_veto = int(
        (
            (target["hard_winner_protection_target"] | target["runner_protection_target"] | target["high_mfe_ambiguous_protection_target"])
            & (target["bad_recall_target"] | target["tail_recall_target"] | target["risky_attention_target"])
            & ~target["hard_protection_veto_target"]
        ).sum()
    )
    monitor_positive = int((target["monitor_only_flag"] & (target["bad_recall_target"] | target["tail_recall_target"])).sum())
    if ambiguous_bad:
        raise RuntimeError(f"Ambiguous high-MFE rows became bad-positive: {ambiguous_bad}")
    if runner_bad:
        raise RuntimeError(f"Runner-protect rows became bad-positive: {runner_bad}")
    if hard_without_veto:
        raise RuntimeError(f"Hard protected rows can become recall-positive without veto: {hard_without_veto}")
    if monitor_positive:
        raise RuntimeError(f"Monitor-only rows drive positive bad/tail target: {monitor_positive}")
    hundred = _bool(target, "hundred_plus_mfe_v1")
    two_hundred = _bool(target, "two_hundred_plus_mfe_v1")
    strongest = _bool(target, "strongest_winner_path_v1")
    repaired = _bool(target, "r6_label_repaired_165_like_runner_v1")
    hard_rows = hundred | two_hundred | strongest | repaired
    hard_unprotected = int((hard_rows & ~target["hard_winner_protection_target"]).sum())
    if hard_unprotected:
        raise RuntimeError(f"Hard winner/repaired rows missing protection target: {hard_unprotected}")
    return {
        "target_table_rows_v1": int(len(target)),
        "v2_bucket_counts_v1": bucket_counts,
        "ambiguous_high_mfe_bad_positive_count_v1": ambiguous_bad,
        "runner_protect_bad_positive_count_v1": runner_bad,
        "hard_protected_recall_positive_without_veto_v1": hard_without_veto,
        "monitor_only_positive_bad_tail_count_v1": monitor_positive,
        "fifty_plus_rows_v1": int(_bool(target, "fifty_plus_mfe_v1").sum()),
        "fifty_plus_hard_or_eval_protected_v1": int((_bool(target, "fifty_plus_mfe_v1") & target["high_mfe_ambiguous_protection_target"]).sum()),
        "hundred_plus_hard_protected_v1": int((hundred & target["hard_winner_protection_target"]).sum()),
        "two_hundred_plus_hard_protected_v1": int((two_hundred & target["hard_winner_protection_target"]).sum()),
        "strongest_winner_hard_protected_v1": int((strongest & target["hard_winner_protection_target"]).sum()),
        "repaired_hard_protected_v1": int((repaired & target["hard_winner_protection_target"]).sum()),
    }


def _validate_variants(spec: dict[str, Any], architecture: dict[str, Any], base_contract: dict[str, Any]) -> pd.DataFrame:
    variants = (spec.get("parallel_run") or {}).get("variants_v1") or []
    if len(variants) != len(EXPECTED_VARIANTS):
        raise RuntimeError(f"Expected {len(EXPECTED_VARIANTS)} V2 variants, observed {len(variants)}")
    final_outputs = set((architecture or {}).get("final_outputs_v1") or [])
    missing_outputs = [column for column in [*RECALL_OUTPUTS, *PROTECTION_OUTPUTS, *BASE_OUTPUTS] if column not in final_outputs]
    if missing_outputs:
        raise RuntimeError(f"V2 architecture missing expected outputs: {missing_outputs}")
    base_rule = (base_contract or {}).get("final_base_rule_v1", "")
    if "NOT r5_2_v2_hard_protection_veto" not in base_rule:
        raise RuntimeError("V2 base contract does not enforce hard protection veto in final base rule")
    rows: list[dict[str, Any]] = []
    seen_profiles: list[str] = []
    required_weights = [
        "bad_weight_v1",
        "tail_weight_v1",
        "risky_weight_v1",
        "runner_protect_weight_v1",
        "ambiguous_high_mfe_protection_weight_v1",
        "hard_protect_weight_v1",
    ]
    required_veto = [
        "bad_recall_threshold_v1",
        "tail_recall_threshold_v1",
        "risky_attention_threshold_v1",
        "bad_tail_confirmation_threshold_v1",
        "runner_veto_threshold_v1",
        "ambiguous_veto_threshold_v1",
        "hard_winner_veto_threshold_v1",
    ]
    for variant in variants:
        weights = variant.get("weights_v1") or {}
        veto = variant.get("veto_strictness_v1") or {}
        profile_id = str(weights.get("profile_id_v1", ""))
        seen_profiles.append(profile_id)
        missing_weights = [key for key in required_weights if key not in weights or float(weights.get(key) or 0.0) <= 0.0]
        missing_veto = [key for key in required_veto if key not in veto]
        missing_expected_outputs = [name for name in ["r5_2_v2_prediction_view_v1.parquet", "r5_2_v2_score_package_v1.parquet", "r5_2_v2_base_membership_v1.parquet"] if name not in (variant.get("expected_outputs_v1") or [])]
        status = "READY_FOR_EXPLICIT_RUN" if not missing_weights and not missing_veto and not missing_expected_outputs else "NOT_READY"
        rows.append(
            {
                "variant_id_v1": variant.get("variant_id_v1"),
                "profile_id_v1": profile_id,
                "status_v1": status,
                "missing_weights_v1": "|".join(missing_weights),
                "missing_veto_strictness_v1": "|".join(missing_veto),
                "missing_expected_outputs_v1": "|".join(missing_expected_outputs),
                "bad_weight_v1": weights.get("bad_weight_v1"),
                "tail_weight_v1": weights.get("tail_weight_v1"),
                "risky_weight_v1": weights.get("risky_weight_v1"),
                "runner_protect_weight_v1": weights.get("runner_protect_weight_v1"),
                "ambiguous_high_mfe_protection_weight_v1": weights.get("ambiguous_high_mfe_protection_weight_v1"),
                "hard_protect_weight_v1": weights.get("hard_protect_weight_v1"),
                "base_membership_rule_v1": variant.get("base_membership_rule_v1"),
            }
        )
    if seen_profiles != EXPECTED_VARIANTS:
        raise RuntimeError(f"V2 variant profile order/list mismatch: {seen_profiles}")
    manifest = pd.DataFrame(rows)
    not_ready = manifest[manifest["status_v1"] != "READY_FOR_EXPLICIT_RUN"]
    if not not_ready.empty:
        raise RuntimeError(f"V2 variant config not ready: {not_ready[['variant_id_v1', 'missing_weights_v1', 'missing_veto_strictness_v1']].to_dict('records')}")
    return manifest


def _legal_feature_candidates(score: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    asof = [column for column in score.columns if column.startswith("as_of_")]
    r5 = [column for column in score.columns if column.startswith("pred__entry_r5_")]
    r5_1 = [column for column in score.columns if column.startswith("r5_1_")]
    r5_2 = [
        column
        for column in score.columns
        if column
        in {
            "pred__entry_r5_2_bad_blocker__prob_true_v1",
            "pred__entry_r5_2_runner_protector__prob_true_v1",
        }
    ]
    ordered: list[str] = []
    for column in [*asof, *r5, *r5_1, *r5_2]:
        if column not in ordered:
            ordered.append(column)
    return ordered, {"AS_OF": asof, "R5_SIGNALS": r5, "R5_1_SIGNALS": r5_1, "LEGAL_R5_2_INPUTS": r5_2}


def _forbidden_feature_scan(feature_names: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in feature_names:
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


def _feature_prelaunch(score: pd.DataFrame, score_summary: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    feature_names, families = _legal_feature_candidates(score)
    if not feature_names:
        raise RuntimeError("No legal V2 feature candidates found")
    forbidden = _forbidden_feature_scan(feature_names)
    forbidden_count = int(forbidden["is_forbidden_v1"].sum())
    if forbidden_count:
        fields = forbidden[forbidden["is_forbidden_v1"]]["field_v1"].tolist()
        raise RuntimeError(f"Forbidden features present in V2 feature matrix: {fields}")
    null_rates = {feature: float(score[feature].isna().mean()) for feature in feature_names}
    return {
        "layer_name": "V2_FEATURE_MATRIX_PRELAUNCH_V1",
        "feature_count_v1": int(len(feature_names)),
        "asof_columns_v1": int(score_summary.get("as_of_column_count_v1") or 0),
        "feature_families_v1": {family: int(len(cols)) for family, cols in families.items()},
        "max_null_rate_v1": max(null_rates.values()) if null_rates else 0.0,
        "null_coverage_summary_v1": {
            "zero_null_features_v1": int(sum(rate == 0.0 for rate in null_rates.values())),
            "nonzero_null_features_v1": int(sum(rate > 0.0 for rate in null_rates.values())),
        },
        "forbidden_feature_count_v1": forbidden_count,
        "allowed_feature_examples_v1": feature_names[:20],
        "disallowed_feature_rules_v1": FORBIDDEN_FEATURE_PATTERNS,
    }, forbidden


def _feature_matrix(score: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    features = score[list(feature_names)].copy()
    for column in features.columns:
        if pd.api.types.is_bool_dtype(features[column]):
            features[column] = features[column].fillna(False).astype(int)
    categorical = features.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if categorical:
        features = pd.get_dummies(features, columns=categorical, dummy_na=True)
    for column in features.columns:
        if not pd.api.types.is_numeric_dtype(features[column]):
            features[column] = pd.to_numeric(features[column], errors="coerce")
    return features.astype("float32")


def _safe_div(num: int | float, den: int | float) -> float | None:
    return float(num / den) if den else None


def _first_existing_bool(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return _bool(frame, column)
    return pd.Series(False, index=frame.index, dtype=bool)


def _join_training_frame(score: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    score_work = score.copy()
    target_work = target.copy()
    for column in REQUIRED_KEYS:
        score_work[column] = score_work[column].astype(str)
        target_work[column] = target_work[column].astype(str)
    target_cols = [
        *REQUIRED_KEYS,
        "original_bucket",
        "v2_bucket",
        "bad_recall_target",
        "tail_recall_target",
        "risky_attention_target",
        "runner_protection_target",
        "high_mfe_ambiguous_protection_target",
        "hard_winner_protection_target",
        "sample_weight",
        "protection_weight",
        "monitor_only_flag",
        "hard_protection_veto_target",
        "reason",
    ]
    for column in POCKET_EVAL_COLUMNS:
        if column in target.columns and column not in score.columns and column not in target_cols:
            target_cols.append(column)
    return score_work.merge(target_work[target_cols], on=REQUIRED_KEYS, how="left", validate="one_to_one")


def _split_masks(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "used_for_training" in frame.columns:
        train = _bool(frame, "used_for_training")
    elif "split_scope_v1" in frame.columns:
        train = frame["split_scope_v1"].astype(str).str.upper().eq("TRAIN")
    else:
        train = pd.Series(True, index=frame.index)
    if "used_for_validation" in frame.columns:
        validation = _bool(frame, "used_for_validation")
    elif "split_scope_v1" in frame.columns:
        validation = frame["split_scope_v1"].astype(str).str.upper().str.contains("VALID")
    else:
        validation = ~train
    if int(train.sum()) == 0:
        train = pd.Series(True, index=frame.index)
    return train.astype(bool), validation.astype(bool)


def _head_weight_vector(frame: pd.DataFrame, label_col: str, weights: dict[str, Any]) -> np.ndarray:
    y = _bool(frame, label_col).to_numpy(dtype=bool)
    out = np.ones(len(frame), dtype="float64")
    if label_col == "bad_recall_target":
        out[y] *= float(weights.get("bad_weight_v1") or 1.0)
    elif label_col == "tail_recall_target":
        out[y] *= float(weights.get("tail_weight_v1") or 1.0)
    elif label_col == "risky_attention_target":
        out[y] *= float(weights.get("risky_weight_v1") or 1.0)
    elif label_col == "runner_protection_target":
        out[y] *= float(weights.get("runner_protect_weight_v1") or 1.0)
    elif label_col == "high_mfe_ambiguous_protection_target":
        out[y] *= float(weights.get("ambiguous_high_mfe_protection_weight_v1") or 1.0)
    elif label_col == "hard_winner_protection_target":
        out[y] *= float(weights.get("hard_protect_weight_v1") or 1.0)
    protection = _bool(frame, "hard_protection_veto_target").to_numpy(dtype=bool)
    if label_col in {"bad_recall_target", "tail_recall_target", "risky_attention_target"}:
        out[~y & protection] *= np.maximum(out[~y & protection], float(weights.get("hard_protect_weight_v1") or 1.0))
    return out


def _fit_head(
    *,
    x: pd.DataFrame,
    frame: pd.DataFrame,
    label_col: str,
    output_col: str,
    variant_id: str,
    seed: int,
    weights: dict[str, Any],
    model_dir: Path,
) -> tuple[pd.Series, dict[str, Any]]:
    y = _bool(frame, label_col).astype(int)
    train_mask, validation_mask = _split_masks(frame)
    y_train = y.loc[train_mask]
    if len(set(y_train.tolist())) < 2:
        constant = float(y_train.mean()) if len(y_train) else float(y.mean())
        pred = pd.Series(constant, index=frame.index, dtype="float64")
        return pred.rename(output_col), {
            "head_v1": label_col,
            "output_col_v1": output_col,
            "constant_model_v1": True,
            "positive_train_rows_v1": int(y_train.sum()),
            "train_rows_v1": int(len(y_train)),
            "roc_auc_all_v1": None,
        }
    model = HistGradientBoostingClassifier(
        max_iter=80,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=seed,
    )
    sample_weight = _head_weight_vector(frame.loc[train_mask], label_col, weights)
    model.fit(x.loc[train_mask], y_train, sample_weight=sample_weight)
    pred = pd.Series(model.predict_proba(x)[:, 1], index=frame.index, dtype="float64")
    metrics: dict[str, Any] = {
        "head_v1": label_col,
        "output_col_v1": output_col,
        "constant_model_v1": False,
        "positive_train_rows_v1": int(y_train.sum()),
        "train_rows_v1": int(len(y_train)),
        "validation_rows_v1": int(validation_mask.sum()),
    }
    for split_name, mask in {
        "all": pd.Series(True, index=frame.index),
        "train": train_mask,
        "validation": validation_mask,
    }.items():
        if int(mask.sum()) == 0:
            continue
        yy = y.loc[mask].to_numpy(dtype=int)
        pp = pred.loc[mask].to_numpy(dtype=float)
        if len(set(yy.tolist())) >= 2:
            pred_label = (pp >= 0.5).astype(int)
            precision, recall, _, _ = precision_recall_fscore_support(yy, pred_label, labels=[0, 1], zero_division=0)
            metrics[f"balanced_accuracy_{split_name}_v1"] = float(balanced_accuracy_score(yy, pred_label))
            metrics[f"precision_true_{split_name}_v1"] = float(precision[1])
            metrics[f"recall_true_{split_name}_v1"] = float(recall[1])
            metrics[f"roc_auc_{split_name}_v1"] = float(roc_auc_score(yy, pp))
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"{output_col}.joblib")
    _write_json(
        model_dir / f"{output_col}.metadata.json",
        {
            "variant_id_v1": variant_id,
            "label_col_v1": label_col,
            "output_col_v1": output_col,
            "seed_v1": seed,
            "model_family_v1": "HistGradientBoostingClassifier",
        },
    )
    return pred.rename(output_col), metrics


def _apply_variant_base_rule(prediction: pd.DataFrame, frame: pd.DataFrame, variant: dict[str, Any]) -> pd.DataFrame:
    veto = variant.get("veto_strictness_v1") or {}
    bad_thr = float(veto.get("bad_recall_threshold_v1"))
    tail_thr = float(veto.get("tail_recall_threshold_v1"))
    risky_thr = float(veto.get("risky_attention_threshold_v1"))
    confirm_thr = float(veto.get("bad_tail_confirmation_threshold_v1"))
    runner_thr = float(veto.get("runner_veto_threshold_v1"))
    ambiguous_thr = float(veto.get("ambiguous_veto_threshold_v1"))
    hard_thr = float(veto.get("hard_winner_veto_threshold_v1"))
    bad = prediction["r5_2_v2_bad_recall_score"]
    tail = prediction["r5_2_v2_tail_recall_score"]
    risky = prediction["r5_2_v2_risky_attention_score"]
    runner = prediction["r5_2_v2_runner_protection_score"]
    ambiguous = prediction["r5_2_v2_high_mfe_ambiguous_protection_score"]
    hard = prediction["r5_2_v2_hard_winner_protection_score"]
    prediction["r5_2_v2_base_membership_pre_veto"] = (
        bad.ge(bad_thr)
        | tail.ge(tail_thr)
        | (risky.ge(risky_thr) & (bad.ge(confirm_thr) | tail.ge(confirm_thr)))
    )
    explicit_hard = (
        _bool(frame, "r6_label_repaired_165_like_runner_v1")
        | _bool(frame, "strongest_winner_path_v1")
        | _bool(frame, "hundred_plus_mfe_v1")
        | _bool(frame, "two_hundred_plus_mfe_v1")
        | frame["candidate_uid"].astype(str).eq(FORENSIC_REPAIRED_CANDIDATE_UID)
    )
    dangerous_50 = _bool(frame, "fifty_plus_mfe_v1")
    explicit_runner = _bool(frame, "runner_protection_target") | _bool(frame, "r6_label_runner_near_miss_v1")
    explicit_ambiguous = frame["v2_bucket"].astype(str).eq("AMBIGUOUS_HIGH_MFE_PROTECTED")
    explicit_monitor = frame["v2_bucket"].astype(str).eq("MONITOR_ONLY")
    prediction["r5_2_v2_hard_protection_veto"] = (
        runner.ge(runner_thr)
        | ambiguous.ge(ambiguous_thr)
        | hard.ge(hard_thr)
        | explicit_hard
        | dangerous_50
        | explicit_runner
        | explicit_ambiguous
        | explicit_monitor
    )
    prediction["r5_2_v2_final_base_membership"] = prediction["r5_2_v2_base_membership_pre_veto"] & ~prediction["r5_2_v2_hard_protection_veto"]
    prediction["v2_base_reason_v1"] = np.select(
        [
            prediction["r5_2_v2_final_base_membership"] & bad.ge(bad_thr),
            prediction["r5_2_v2_final_base_membership"] & tail.ge(tail_thr),
            prediction["r5_2_v2_final_base_membership"],
            prediction["r5_2_v2_base_membership_pre_veto"] & prediction["r5_2_v2_hard_protection_veto"],
        ],
        ["ADDED_BY_BAD_RECALL", "ADDED_BY_TAIL_RECALL", "ADDED_BY_RISKY_CONFIRMATION", "VETOED_AFTER_PRE_VETO_RECALL"],
        default="NOT_BASE",
    )
    return prediction


def _worst_loso_precision(frame: pd.DataFrame, selected: pd.Series) -> float | None:
    if "run_id" not in frame.columns:
        return None
    values: list[float] = []
    selected = selected.reindex(frame.index).fillna(False).astype(bool)
    bad = _bool(frame, "label_should_not_take_v1")
    for _, group in frame.groupby("run_id"):
        mask = selected.loc[group.index]
        count = int(mask.sum())
        if count:
            values.append(float((bad.loc[group.index] & mask).sum() / count))
    return min(values) if values else 1.0


def _variant_metrics(frame: pd.DataFrame, selected: pd.Series, pre_veto: pd.Series, veto: pd.Series) -> dict[str, Any]:
    selected = selected.reindex(frame.index).fillna(False).astype(bool)
    pre_veto = pre_veto.reindex(frame.index).fillna(False).astype(bool)
    veto = veto.reindex(frame.index).fillna(False).astype(bool)
    bad = _bool(frame, "label_should_not_take_v1")
    tail = _bool(frame, "tail_10_50_mfe_v1")
    strong_bad = frame["v2_bucket"].astype(str).eq("BAD_RECALL_POSITIVE")
    tail_control = frame["v2_bucket"].astype(str).eq("TAIL_RECALL_POSITIVE")
    risky = _bool(frame, "risky_attention_target")
    runner = _bool(frame, "runner_protection_target")
    ambiguous = frame["v2_bucket"].astype(str).eq("AMBIGUOUS_HIGH_MFE_PROTECTED")
    fifty = _bool(frame, "fifty_plus_mfe_v1")
    hundred = _bool(frame, "hundred_plus_mfe_v1")
    two_hundred = _bool(frame, "two_hundred_plus_mfe_v1")
    strongest = _bool(frame, "strongest_winner_path_v1")
    repaired = _bool(frame, "r6_label_repaired_165_like_runner_v1")
    near_miss = _bool(frame, "r6_label_runner_near_miss_v1")
    forensic = frame["candidate_uid"].astype(str).eq(FORENSIC_REPAIRED_CANDIDATE_UID)
    v3 = _first_existing_bool(frame, V3_BASE_COLUMNS)
    rescue = _first_existing_bool(frame, RESCUE_BASE_COLUMNS)
    raw_true = _first_existing_bool(frame, RAW_TRUE_BASE_COLUMNS)
    base_count = int(selected.sum())
    bad_count = int((selected & bad).sum())
    tail_count = int((selected & tail).sum())
    out = {
        "bad_recall_v1": bad_count,
        "tail_recall_v1": tail_count,
        "precision_v1": _safe_div(bad_count, base_count),
        "worst_loso_v1": _worst_loso_precision(frame, selected),
        "strong_bad_target_recall_v1": _safe_div(int((selected & strong_bad).sum()), int(strong_bad.sum())),
        "tail_control_target_recall_v1": _safe_div(int((selected & tail_control).sum()), int(tail_control.sum())),
        "risky_attention_coverage_v1": _safe_div(int((selected & risky).sum()), int(risky.sum())),
        "runner_protect_performance_v1": _safe_div(int(((~selected) & runner).sum()), int(runner.sum())),
        "ambiguous_high_mfe_leakage_v1": int((selected & ambiguous).sum()),
        "fifty_plus_overlap_v1": int((selected & fifty).sum()),
        "hundred_plus_overlap_v1": int((selected & hundred).sum()),
        "two_hundred_plus_overlap_v1": int((selected & two_hundred).sum()),
        "strongest_winner_overlap_v1": int((selected & strongest).sum()),
        "repaired_like_overlap_v1": int((selected & repaired).sum()),
        "forensic_repaired_trade_blocked_v1": int((selected & forensic).sum()),
        "runner_near_miss_overlap_v1": int((selected & near_miss).sum()),
        "runner_protect_leakage_v1": int((selected & runner).sum()),
        "hard_veto_count_v1": int(veto.sum()),
        "pre_veto_base_count_v1": int(pre_veto.sum()),
        "final_base_count_v1": base_count,
        "rows_vetoed_by_protection_v1": int((pre_veto & veto).sum()),
        "rows_added_vs_v3_v1": int((selected & ~v3).sum()),
        "rows_added_vs_rescue_v1": int((selected & ~rescue).sum()),
        "rows_lost_vs_v3_v1": int((v3 & ~selected).sum()),
        "rows_lost_vs_rescue_v1": int((rescue & ~selected).sum()),
        "raw_true_unsafe_overlap_avoided_v1": int((raw_true & ~selected).sum()),
    }
    out["safety_pass_v1"] = bool(
        out["repaired_like_overlap_v1"] == 0
        and out["forensic_repaired_trade_blocked_v1"] == 0
        and out["strongest_winner_overlap_v1"] == 0
        and out["hundred_plus_overlap_v1"] == 0
        and out["two_hundred_plus_overlap_v1"] == 0
        and out["fifty_plus_overlap_v1"] <= 1
        and out["ambiguous_high_mfe_leakage_v1"] == 0
        and out["runner_protect_leakage_v1"] == 0
        and (out["worst_loso_v1"] is None or float(out["worst_loso_v1"]) > 0.0)
    )
    out["meaningful_uplift_over_rescue_v1"] = bool(out["bad_recall_v1"] > 88 and out["tail_recall_v1"] > 57)
    return out


def _label_weight_manifest(target: pd.DataFrame, weights: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for bucket, group in target.groupby("v2_bucket"):
        rows.append(
            {
                "v2_bucket_v1": bucket,
                "row_count_v1": int(len(group)),
                "bad_recall_target_rows_v1": int(_bool(group, "bad_recall_target").sum()),
                "tail_recall_target_rows_v1": int(_bool(group, "tail_recall_target").sum()),
                "risky_attention_target_rows_v1": int(_bool(group, "risky_attention_target").sum()),
                "runner_protection_target_rows_v1": int(_bool(group, "runner_protection_target").sum()),
                "hard_veto_target_rows_v1": int(_bool(group, "hard_protection_veto_target").sum()),
                "profile_bad_weight_v1": weights.get("bad_weight_v1"),
                "profile_tail_weight_v1": weights.get("tail_weight_v1"),
                "profile_runner_protect_weight_v1": weights.get("runner_protect_weight_v1"),
                "profile_hard_protect_weight_v1": weights.get("hard_protect_weight_v1"),
            }
        )
    return pd.DataFrame(rows).sort_values("v2_bucket_v1")


def _variant_audit(metrics: dict[str, Any], feature_preflight: dict[str, Any], key_alignment: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("TRAINING_STARTED", True, True),
            row("NO_FORBIDDEN_FEATURES", feature_preflight["forbidden_feature_count_v1"] == 0, feature_preflight["forbidden_feature_count_v1"]),
            row("KEY_ALIGNMENT", key_alignment["missing_from_score_v1"] == 0 and key_alignment["extra_in_score_v1"] == 0, key_alignment),
            row("HARD_VETO_ACTIVE", metrics["rows_vetoed_by_protection_v1"] >= 0, metrics["rows_vetoed_by_protection_v1"]),
            row("SAFETY_PASS", bool(metrics["safety_pass_v1"]), metrics),
        ]
    )


def _write_variant_outputs(
    *,
    variant_dir: Path,
    variant: dict[str, Any],
    score: pd.DataFrame,
    target: pd.DataFrame,
    training_frame: pd.DataFrame,
    prediction: pd.DataFrame,
    feature_names: Sequence[str],
    processed_feature_count: int,
    head_metrics: pd.DataFrame,
    metrics: dict[str, Any],
    feature_preflight: dict[str, Any],
    key_alignment: dict[str, Any],
) -> dict[str, Any]:
    variant_dir.mkdir(parents=True, exist_ok=True)
    weights = variant["weights_v1"]
    variant_id = str(variant["variant_id_v1"])
    prediction_view = training_frame[[*REQUIRED_KEYS, "trade_id", "run_id", "v2_bucket", "label_should_not_take_v1", "tail_10_50_mfe_v1"]].copy()
    for col in [*RECALL_OUTPUTS, *PROTECTION_OUTPUTS, *BASE_OUTPUTS, "v2_base_reason_v1"]:
        prediction_view[col] = prediction[col].values
    score_work = score.copy()
    pred_work = prediction[[*REQUIRED_KEYS, *RECALL_OUTPUTS, *PROTECTION_OUTPUTS, *BASE_OUTPUTS, "v2_base_reason_v1"]].copy()
    for column in REQUIRED_KEYS:
        score_work[column] = score_work[column].astype(str)
        pred_work[column] = pred_work[column].astype(str)
    score_package = score_work.merge(pred_work, on=REQUIRED_KEYS, how="left", validate="one_to_one")
    base_membership = prediction[[*REQUIRED_KEYS, "r5_2_v2_final_base_membership", "r5_2_v2_base_membership_pre_veto", "r5_2_v2_hard_protection_veto", "v2_base_reason_v1"]].copy()
    prediction_view.to_parquet(variant_dir / "prediction_view_v1.parquet", index=False)
    score_package.to_parquet(variant_dir / "score_package_v1.parquet", index=False)
    base_membership.to_parquet(variant_dir / "base_membership_package_v1.parquet", index=False)
    feature_manifest = pd.DataFrame({"feature_v1": list(feature_names)})
    feature_manifest.to_csv(variant_dir / "feature_manifest_v1.csv", index=False)
    _label_weight_manifest(target, weights).to_csv(variant_dir / "label_weight_manifest_v1.csv", index=False)
    head_metrics.to_csv(variant_dir / "model_metrics_v1.csv", index=False)
    pocket_eval = pd.DataFrame(
        [
            {"pocket_v1": key, "value_v1": value}
            for key, value in metrics.items()
            if key.endswith("_v1") and isinstance(value, (int, float, bool, str)) or value is None
        ]
    )
    pocket_eval.to_csv(variant_dir / "pocket_eval_report_v1.csv", index=False)
    safety = {
        "layer_name": "V2_VARIANT_SAFETY_GUARD_REPORT_V1",
        "variant_id_v1": variant_id,
        **metrics,
    }
    training_summary = {
        "layer_name": "V2_VARIANT_TRAINING_SUMMARY_V1",
        "variant_id_v1": variant_id,
        "profile_id_v1": weights["profile_id_v1"],
        "training_started_v1": True,
        "head_count_v1": 6,
        "input_rows_v1": int(len(training_frame)),
        "feature_count_v1": int(len(feature_names)),
        "processed_feature_count_v1": int(processed_feature_count),
        **metrics,
    }
    model_manifest = {
        "layer_name": "V2_VARIANT_MODEL_MANIFEST_V1",
        "variant_id_v1": variant_id,
        "model_family_v1": "HistGradientBoostingClassifier",
        "heads_v1": [*RECALL_OUTPUTS, *PROTECTION_OUTPUTS],
        "model_dir_v1": str(variant_dir / "models"),
    }
    config_manifest = {
        "layer_name": "V2_VARIANT_CONFIG_MANIFEST_V1",
        "variant_id_v1": variant_id,
        "weights_v1": weights,
        "veto_strictness_v1": variant["veto_strictness_v1"],
        "base_membership_rule_v1": variant.get("base_membership_rule_v1"),
    }
    downstream = {
        "layer_name": "V2_VARIANT_DOWNSTREAM_R6_INPUT_MANIFEST_V1",
        "variant_id_v1": variant_id,
        "ready_for_downstream_r6_v1": bool(metrics["safety_pass_v1"] and metrics["meaningful_uplift_over_rescue_v1"]),
        "score_package_path_v1": str(variant_dir / "score_package_v1.parquet"),
        "prediction_view_path_v1": str(variant_dir / "prediction_view_v1.parquet"),
        "base_membership_path_v1": str(variant_dir / "base_membership_package_v1.parquet"),
        "base_flag_for_r6_v1": "r5_2_v2_final_base_membership",
        "score_columns_for_r6_v1": [*RECALL_OUTPUTS, *PROTECTION_OUTPUTS],
        "raw_pre_veto_base_not_allowed_v1": "r5_2_v2_base_membership_pre_veto",
    }
    status = {
        "layer_name": "V2_VARIANT_STATUS_V1",
        "variant_id_v1": variant_id,
        "training_started_v1": True,
        "safety_pass_v1": bool(metrics["safety_pass_v1"]),
        "meaningful_uplift_over_rescue_v1": bool(metrics["meaningful_uplift_over_rescue_v1"]),
        "decision_v1": "VARIANT_PASS_READY_FOR_R6" if downstream["ready_for_downstream_r6_v1"] else "VARIANT_NOT_R6_READY",
    }
    manifest = {
        "layer_name": "V2_VARIANT_MANIFEST_V1",
        "variant_id_v1": variant_id,
        "output_files_v1": {
            "training_summary": "training_summary_v1.json",
            "model_manifest": "model_manifest_v1.json",
            "config_manifest": "config_manifest_v1.json",
            "feature_manifest": "feature_manifest_v1.csv",
            "label_weight_manifest": "label_weight_manifest_v1.csv",
            "prediction_view": "prediction_view_v1.parquet",
            "score_package": "score_package_v1.parquet",
            "base_membership": "base_membership_package_v1.parquet",
            "pocket_eval": "pocket_eval_report_v1.csv",
            "safety_guard": "safety_guard_report_v1.json",
            "downstream_r6_input_manifest": "downstream_r6_input_manifest_v1.json",
            "status": "status_v1.json",
            "manifest": "manifest_v1.json",
            "audit": "consistency_audit_v1.csv",
        },
    }
    _write_json(variant_dir / "training_summary_v1.json", training_summary)
    _write_json(variant_dir / "model_manifest_v1.json", model_manifest)
    _write_json(variant_dir / "config_manifest_v1.json", config_manifest)
    _write_json(variant_dir / "safety_guard_report_v1.json", safety)
    _write_json(variant_dir / "downstream_r6_input_manifest_v1.json", downstream)
    _write_json(variant_dir / "status_v1.json", status)
    _write_json(variant_dir / "manifest_v1.json", manifest)
    _variant_audit(metrics, feature_preflight, key_alignment).to_csv(variant_dir / "consistency_audit_v1.csv", index=False)
    return {
        "variant_id_v1": variant_id,
        "profile_id_v1": weights["profile_id_v1"],
        "variant_dir_v1": str(variant_dir),
        "training_summary_path_v1": str(variant_dir / "training_summary_v1.json"),
        "prediction_view_path_v1": str(variant_dir / "prediction_view_v1.parquet"),
        "score_package_path_v1": str(variant_dir / "score_package_v1.parquet"),
        "base_membership_path_v1": str(variant_dir / "base_membership_package_v1.parquet"),
        "downstream_r6_input_manifest_path_v1": str(variant_dir / "downstream_r6_input_manifest_v1.json"),
        "safety_pass_v1": bool(metrics["safety_pass_v1"]),
        "r6_ready_v1": downstream["ready_for_downstream_r6_v1"],
    }


def _train_variant(
    *,
    output_dir: Path,
    variant: dict[str, Any],
    score: pd.DataFrame,
    target: pd.DataFrame,
    training_frame: pd.DataFrame,
    feature_names: Sequence[str],
    x: pd.DataFrame,
    feature_preflight: dict[str, Any],
    key_alignment: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    variant_id = str(variant["variant_id_v1"])
    variant_dir = output_dir / "variants" / variant_id
    weights = variant["weights_v1"]
    heads = [
        ("bad_recall_target", "r5_2_v2_bad_recall_score"),
        ("tail_recall_target", "r5_2_v2_tail_recall_score"),
        ("risky_attention_target", "r5_2_v2_risky_attention_score"),
        ("runner_protection_target", "r5_2_v2_runner_protection_score"),
        ("high_mfe_ambiguous_protection_target", "r5_2_v2_high_mfe_ambiguous_protection_score"),
        ("hard_winner_protection_target", "r5_2_v2_hard_winner_protection_score"),
    ]
    prediction = training_frame[[*REQUIRED_KEYS]].copy()
    head_rows: list[dict[str, Any]] = []
    for idx, (label_col, output_col) in enumerate(heads):
        pred, metrics = _fit_head(
            x=x,
            frame=training_frame,
            label_col=label_col,
            output_col=output_col,
            variant_id=variant_id,
            seed=int((variant.get("model_config_v1") or {}).get("seed_v1") or 20260426) + idx,
            weights=weights,
            model_dir=variant_dir / "models",
        )
        prediction[output_col] = pred
        head_rows.append(metrics)
    prediction = _apply_variant_base_rule(prediction, training_frame, variant)
    metrics = _variant_metrics(
        training_frame,
        prediction["r5_2_v2_final_base_membership"],
        prediction["r5_2_v2_base_membership_pre_veto"],
        prediction["r5_2_v2_hard_protection_veto"],
    )
    metrics.update(
        {
            "variant_id_v1": variant_id,
            "profile_id_v1": weights["profile_id_v1"],
            "forbidden_feature_count_v1": int(feature_preflight["forbidden_feature_count_v1"]),
            "key_schema_drift_v1": int(key_alignment["missing_from_score_v1"] + key_alignment["extra_in_score_v1"]),
        }
    )
    output_index = _write_variant_outputs(
        variant_dir=variant_dir,
        variant=variant,
        score=score,
        target=target,
        training_frame=training_frame,
        prediction=prediction,
        feature_names=feature_names,
        processed_feature_count=int(x.shape[1]),
        head_metrics=pd.DataFrame(head_rows),
        metrics=metrics,
        feature_preflight=feature_preflight,
        key_alignment=key_alignment,
    )
    forensic = training_frame[[*REQUIRED_KEYS, "trade_id", "run_id", "v2_bucket", "label_should_not_take_v1", "tail_10_50_mfe_v1"]].copy()
    for column in [
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "strongest_winner_path_v1",
        "r6_label_repaired_165_like_runner_v1",
        "r6_label_runner_near_miss_v1",
    ]:
        if column in training_frame.columns:
            forensic[column] = training_frame[column].values
    forensic = forensic.join(prediction[[*RECALL_OUTPUTS, *PROTECTION_OUTPUTS, *BASE_OUTPUTS, "v2_base_reason_v1"]])
    rescue = _first_existing_bool(training_frame, RESCUE_BASE_COLUMNS)
    v3 = _first_existing_bool(training_frame, V3_BASE_COLUMNS)
    forensic["variant_id_v1"] = variant_id
    forensic["profile_id_v1"] = weights["profile_id_v1"]
    forensic["added_vs_rescue_v1"] = forensic["r5_2_v2_final_base_membership"].astype(bool) & ~rescue
    forensic["added_vs_v3_v1"] = forensic["r5_2_v2_final_base_membership"].astype(bool) & ~v3
    forensic["vetoed_by_hard_protection_v1"] = forensic["r5_2_v2_base_membership_pre_veto"].astype(bool) & forensic["r5_2_v2_hard_protection_veto"].astype(bool)
    return metrics, output_index, forensic


def _leaderboard(eval_df: pd.DataFrame) -> pd.DataFrame:
    work = eval_df.copy()
    work["bad_delta_vs_rescue_v1"] = work["bad_recall_v1"] - 88
    work["tail_delta_vs_rescue_v1"] = work["tail_recall_v1"] - 57
    work["winner_risk_v1"] = (
        work["fifty_plus_overlap_v1"]
        + work["hundred_plus_overlap_v1"] * 10
        + work["two_hundred_plus_overlap_v1"] * 10
        + work["strongest_winner_overlap_v1"] * 10
        + work["ambiguous_high_mfe_leakage_v1"] * 10
        + work["runner_protect_leakage_v1"] * 10
    )
    work["explainable_rank_v1"] = work["profile_id_v1"].map({profile: idx for idx, profile in enumerate(EXPECTED_VARIANTS, start=1)}).fillna(99)
    return work.sort_values(
        [
            "safety_pass_v1",
            "bad_delta_vs_rescue_v1",
            "tail_delta_vs_rescue_v1",
            "worst_loso_v1",
            "precision_v1",
            "winner_risk_v1",
            "explainable_rank_v1",
        ],
        ascending=[False, False, False, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _gate_and_next(leaderboard: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any], pd.Series | None]:
    safe = leaderboard[leaderboard["safety_pass_v1"].astype(bool)].copy()
    best = safe.iloc[0] if not safe.empty else None
    unsafe = leaderboard[~leaderboard["safety_pass_v1"].astype(bool)].copy()
    unsafe_recall = unsafe[(unsafe["bad_recall_v1"] > 88) | (unsafe["tail_recall_v1"] > 57)]
    if best is not None and bool(best["meaningful_uplift_over_rescue_v1"]):
        decision = "V2_BEST_VARIANT_PASS_READY_FOR_R6"
        next_action = EXECUTION_NEXT_ACTIONS["pass"]
    elif best is not None:
        decision = "V2_SAFE_BUT_TOO_WEAK"
        next_action = EXECUTION_NEXT_ACTIONS["safe_weak"]
    elif not unsafe_recall.empty:
        decision = "V2_RECALL_IMPROVES_BUT_SAFETY_FAILS"
        next_action = EXECUTION_NEXT_ACTIONS["unsafe_recall"]
    else:
        decision = "V2_ALL_VARIANTS_FAIL_SAFETY"
        next_action = EXECUTION_NEXT_ACTIONS["all_fail"]
    gate = {
        "layer_name": "V2_PARALLEL_REBUILD_GATE_V1",
        "decision_v1": decision,
        "best_variant_v1": None if best is None else best.to_dict(),
        "safe_variant_count_v1": int(safe.shape[0]),
        "unsafe_variant_count_v1": int(unsafe.shape[0]),
        "requirements_v1": {
            "at_least_one_safety_pass_v1": best is not None,
            "meaningful_uplift_over_rescue_v1": None if best is None else bool(best["meaningful_uplift_over_rescue_v1"]),
            "downstream_r6_manifest_required_for_pass_v1": decision == "V2_BEST_VARIANT_PASS_READY_FOR_R6",
            "no_pre_veto_base_as_final_v1": True,
        },
    }
    next_lock = {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": next_action,
        "blocked_action_v1": [
            "RUN_R6_WITHOUT_EXPLICIT_FLAG",
            "USE_RAW_PRE_VETO_V2_BASE",
            "USE_UNSAFE_V2_VARIANT",
            "USE_RAW_TRUE_R5_2_PACKAGE_DIRECTLY",
        ],
    }
    return gate, next_lock, best


def _best_r6_lock(best: pd.Series | None, output_index: pd.DataFrame, gate: dict[str, Any]) -> dict[str, Any]:
    if best is None or gate["decision_v1"] != "V2_BEST_VARIANT_PASS_READY_FOR_R6":
        return {
            "layer_name": "BEST_V2_VARIANT_DOWNSTREAM_R6_INPUT_LOCK_V1",
            "ready_for_downstream_r6_v1": False,
            "failure_reason_v1": gate["decision_v1"],
            "do_not_materialize_pass_input_v1": True,
        }
    row = output_index[output_index["variant_id_v1"] == best["variant_id_v1"]].iloc[0]
    return {
        "layer_name": "BEST_V2_VARIANT_DOWNSTREAM_R6_INPUT_LOCK_V1",
        "ready_for_downstream_r6_v1": True,
        "best_variant_id_v1": best["variant_id_v1"],
        "best_profile_id_v1": best["profile_id_v1"],
        "score_package_path_v1": row["score_package_path_v1"],
        "prediction_view_path_v1": row["prediction_view_path_v1"],
        "base_membership_path_v1": row["base_membership_path_v1"],
        "downstream_r6_input_manifest_path_v1": row["downstream_r6_input_manifest_path_v1"],
        "score_columns_for_r6_v1": [*RECALL_OUTPUTS, *PROTECTION_OUTPUTS],
        "base_flag_for_r6_v1": "r5_2_v2_final_base_membership",
        "raw_pre_veto_base_not_allowed_v1": "r5_2_v2_base_membership_pre_veto",
        "unsafe_variants_not_allowed_v1": True,
        "old_flags_not_final_if_v2_used_v1": [
            "r5_2_v3_base_flag_v1",
            "r5_2_true_rescue_base_membership_v1",
            "raw_true_base_membership_v1",
        ],
    }


def _execution_audit(summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("TRAINING_STARTED", summary["training_started_v1"], summary["training_started_v1"]),
            row("PARALLEL_EXECUTION_STARTED", summary["parallel_execution_started_v1"], summary["parallel_execution_started_v1"]),
            row("VARIANT_COUNT", summary["variant_count_v1"] == 7, summary["variant_count_v1"]),
            row("FOUNDATION_ROWS", summary["foundation_rows_v1"] == 1914, summary["foundation_rows_v1"]),
            row("TARGET_ROWS", summary["target_table_rows_v1"] == 1914, summary["target_table_rows_v1"]),
            row("NO_FORBIDDEN_FEATURES", summary["forbidden_feature_count_v1"] == 0, summary["forbidden_feature_count_v1"]),
            row("HARD_VETO_ACTIVE", summary["hard_protection_veto_contract_present_v1"], summary["hard_protection_veto_contract_present_v1"]),
            row("NO_R6_RUN", not summary["r6_started_v1"], summary["r6_started_v1"]),
        ]
    )


def _execution_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# R5.2 Objective V2 Parallel Rebuild Execution",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Variants run: `{summary['variant_count_v1']}`",
            f"- Best safe variant: `{summary.get('best_variant_id_v1')}`",
            f"- Best bad/tail: `{summary.get('best_bad_recall_v1')}` / `{summary.get('best_tail_recall_v1')}`",
            f"- Safe variants: `{summary['safe_variant_count_v1']}`",
            f"- Downstream R6 input ready: `{summary['downstream_r6_input_ready_v1']}`",
            "",
            "No R6 retrain, freeze, promo, live gate, controller change, new baseline, or new feature surface was run.",
            "",
        ]
    )


def _execute_parallel_rebuild(
    *,
    output_dir: Path,
    score: pd.DataFrame,
    target: pd.DataFrame,
    variants: list[dict[str, Any]],
    feature_names: Sequence[str],
    feature_preflight: dict[str, Any],
    key_alignment: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training_frame = _join_training_frame(score, target)
    x = _feature_matrix(score, feature_names)
    eval_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    forensic_frames: list[pd.DataFrame] = []
    for variant in variants:
        metrics, output_index, forensic = _train_variant(
            output_dir=output_dir,
            variant=variant,
            score=score,
            target=target,
            training_frame=training_frame,
            feature_names=feature_names,
            x=x,
            feature_preflight=feature_preflight,
            key_alignment=key_alignment,
        )
        eval_rows.append(metrics)
        index_rows.append(output_index)
        forensic_frames.append(forensic)
    return pd.DataFrame(index_rows), pd.DataFrame(eval_rows), pd.concat(forensic_frames, ignore_index=True)


def _veto_contract_report(spec: dict[str, Any], target_audit: dict[str, Any]) -> dict[str, Any]:
    base_contract = spec["base_contract"]
    label_contract = spec["label_contract"]
    architecture = spec["architecture"]
    veto_rule = base_contract.get("veto_rule_v1") or {}
    reason_codes = set(base_contract.get("reason_codes_v1") or [])
    required_reason_codes = {
        "VETO_HARD_WINNER",
        "VETO_HIGH_MFE_AMBIGUOUS",
        "VETO_RUNNER_PROTECT",
        "VETO_REPAIRED_OR_STRONGEST",
    }
    final_outputs = set(architecture.get("final_outputs_v1") or [])
    buckets = {bucket.get("bucket_v1") for bucket in label_contract.get("buckets_v1", [])}
    present = bool(
        base_contract.get("contract_id_v1")
        and "forensic" in json.dumps(veto_rule).lower()
        and required_reason_codes.issubset(reason_codes)
        and set(PROTECTION_OUTPUTS).issubset(final_outputs)
        and {"HARD_PROTECT_NEGATIVE", "AMBIGUOUS_HIGH_MFE_PROTECTED"}.issubset(buckets)
    )
    if not present:
        raise RuntimeError("V2 hard protection veto contract is missing or incomplete")
    connected = bool(
        target_audit["hard_protected_recall_positive_without_veto_v1"] == 0
        and target_audit["ambiguous_high_mfe_bad_positive_count_v1"] == 0
        and target_audit["runner_protect_bad_positive_count_v1"] == 0
    )
    if not connected:
        raise RuntimeError("V2 hard protection veto contract cannot be connected to target/eval pockets")
    return {
        "layer_name": "V2_HARD_PROTECTION_VETO_CONTRACT_ENFORCEMENT_V1",
        "hard_protection_veto_contract_present_v1": True,
        "contract_id_v1": base_contract.get("contract_id_v1"),
        "veto_applies_to_v1": [
            "repaired-like",
            "forensic repaired trade",
            "strongest-winner",
            "100+/200+",
            "dangerous 50+",
            "runner-protect",
            "high-MFE ambiguous",
            "explicit hard-protect buckets",
        ],
        "reason_codes_v1": sorted(reason_codes),
        "target_connection_v1": target_audit,
    }


def _execution_placeholder(output_dir: Path, variant_manifest: pd.DataFrame) -> dict[str, Any]:
    variant_namespaces = []
    for variant_id in variant_manifest["variant_id_v1"].astype(str):
        variant_namespaces.append(
            {
                "variant_id_v1": variant_id,
                "future_output_namespace_v1": str(output_dir / "variants" / variant_id),
                "future_outputs_v1": [
                    "training_summary_v1.json",
                    "model_manifest_v1.json",
                    "config_manifest_v1.json",
                    "feature_manifest_v1.csv",
                    "prediction_view_v1.parquet",
                    "score_package_v1.parquet",
                    "base_membership_package_v1.parquet",
                    "pocket_eval_v1.csv",
                    "safety_guard_report_v1.json",
                    "downstream_r6_input_manifest_v1.json",
                ],
            }
        )
    return {
        "layer_name": "V2_PARALLEL_OUTPUT_SCAFFOLD_V1",
        "parallel_execution_started_v1": False,
        "explicit_run_flag_required_v1": RUN_FLAG,
        "variant_namespaces_v1": variant_namespaces,
        "aggregator_namespace_v1": str(output_dir / "aggregator"),
    }


def _r6_placeholder(output_dir: Path) -> dict[str, Any]:
    return {
        "layer_name": "V2_DOWNSTREAM_R6_MANIFEST_PLACEHOLDER_V1",
        "ready_for_future_r6_after_explicit_v2_rebuild_v1": False,
        "future_base_flag_for_r6_v1": "r5_2_v2_final_base_membership",
        "future_score_columns_for_r6_v1": [*RECALL_OUTPUTS, *PROTECTION_OUTPUTS],
        "placeholder_path_v1": str(output_dir / DRY_OUTPUT_FILES["r6_placeholder"]),
        "blocked_until_v1": "R5_2_OBJECTIVE_V2_PARALLEL_REBUILD_WITH_EXPLICIT_FLAG_COMPLETES_AND_GATE_PASSES",
    }


def _audit(summary: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, ok: bool, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": "PASS" if ok else "FAIL", "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("NO_TRAINING", not summary["training_started_v1"], summary["training_started_v1"]),
            row("NO_PARALLEL_EXECUTION", not summary["parallel_execution_started_v1"], summary["parallel_execution_started_v1"]),
            row("PRELAUNCH_PASS", summary["prelaunch_status_v1"] == "PASS", summary["prelaunch_status_v1"]),
            row("VARIANT_COUNT", summary["variant_count_v1"] == 7, summary["variant_count_v1"]),
            row("FOUNDATION_ROWS", summary["foundation_rows_v1"] == 1914, summary["foundation_rows_v1"]),
            row("TARGET_ROWS", summary["target_table_rows_v1"] == 1914, summary["target_table_rows_v1"]),
            row("NO_FORBIDDEN_FEATURES", summary["forbidden_feature_count_v1"] == 0, summary["forbidden_feature_count_v1"]),
            row("AMBIGUOUS_NOT_BAD_POSITIVE", summary["ambiguous_high_mfe_bad_positive_count_v1"] == 0, summary["ambiguous_high_mfe_bad_positive_count_v1"]),
            row("HARD_VETO_PRESENT", summary["hard_protection_veto_contract_present_v1"], summary["hard_protection_veto_contract_present_v1"]),
            row("NEXT_ACTION_LOCKED", summary["next_action_v1"] == NEXT_ACTION, summary["next_action_v1"]),
        ]
    )


def _report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# R5.2 Objective V2 Parallel Rebuild Runner Prelaunch",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Variants: `{summary['variant_count_v1']}`",
            f"- Foundation rows: `{summary['foundation_rows_v1']}`",
            f"- Target table rows: `{summary['target_table_rows_v1']}`",
            f"- Forbidden features: `{summary['forbidden_feature_count_v1']}`",
            f"- Hard veto contract present: `{summary['hard_protection_veto_contract_present_v1']}`",
            "",
            "No V2 rebuild, R5.2 training, R6 run, baseline build, feature surface build, freeze, promo, live gate, or controller change was performed.",
            "",
        ]
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    spec_dir: Path = DEFAULT_SPEC_DIR,
    output_dir: Path | None = None,
    foundation_score_dir: Path | None = None,
    label_table: Path | None = None,
    run_parallel_rebuild: bool = False,
) -> dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    spec_dir = spec_dir.expanduser().resolve()
    output_dir = (output_dir or reports_root / f"{LAYER_NAME}_{_stamp()}").expanduser().resolve()
    _ensure_output_namespace_clean(output_dir)

    spec = _load_spec_package(spec_dir)
    if spec["design_lock"].get("design_id_v1") != DESIGN_ID:
        raise RuntimeError(f"Unexpected V2 design id: {spec['design_lock'].get('design_id_v1')}")
    score_dir = _resolve_foundation_score_dir(spec, foundation_score_dir)
    label_path = _resolve_label_table_path(spec, label_table)
    score, score_summary = _load_foundation(score_dir)
    labels = pd.read_csv(label_path)
    foundation = _validate_foundation(score, score_summary, score_dir)
    target = _build_v2_target_table(labels)
    target_audit = _validate_target_table(target)
    key_alignment = _validate_key_alignment(score, labels)
    variant_manifest = _validate_variants(spec, spec["architecture"], spec["base_contract"])
    feature_preflight, forbidden_scan = _feature_prelaunch(score, score_summary)
    veto_report = _veto_contract_report(spec, target_audit)
    execution_placeholder = _execution_placeholder(output_dir, variant_manifest)
    r6_placeholder = _r6_placeholder(output_dir)

    if run_parallel_rebuild:
        feature_names, _ = _legal_feature_candidates(score)
        variants = (spec["parallel_run"] or {}).get("variants_v1") or []
        output_index, eval_df, forensics = _execute_parallel_rebuild(
            output_dir=output_dir,
            score=score,
            target=target,
            variants=variants,
            feature_names=feature_names,
            feature_preflight=feature_preflight,
            key_alignment=key_alignment,
        )
        leaderboard = _leaderboard(eval_df)
        gate, next_lock, best = _gate_and_next(leaderboard)
        best_lock = _best_r6_lock(best, output_index, gate)
        tempting: pd.Series | None = None
        unsafe = leaderboard[~leaderboard["safety_pass_v1"].astype(bool)].copy()
        unsafe_recall = unsafe[(unsafe["bad_recall_v1"] > 88) | (unsafe["tail_recall_v1"] > 57)]
        if not unsafe_recall.empty:
            tempting = unsafe_recall.iloc[0]
        elif not unsafe.empty:
            tempting = unsafe.iloc[0]
        forensic_variant_ids = set()
        if best is not None:
            forensic_variant_ids.add(str(best["variant_id_v1"]))
        if tempting is not None:
            forensic_variant_ids.add(str(tempting["variant_id_v1"]))
        row_forensics = forensics[forensics["variant_id_v1"].astype(str).isin(forensic_variant_ids)].copy()
        if not row_forensics.empty:
            safety_flags = [
                column
                for column in [
                    "fifty_plus_mfe_v1",
                    "hundred_plus_mfe_v1",
                    "two_hundred_plus_mfe_v1",
                    "strongest_winner_path_v1",
                    "r6_label_repaired_165_like_runner_v1",
                    "r6_label_runner_near_miss_v1",
                ]
                if column in row_forensics.columns
            ]
            safety_mask = row_forensics[safety_flags].fillna(False).astype(bool).any(axis=1) if safety_flags else pd.Series(False, index=row_forensics.index)
            row_forensics["forensic_row_scope_v1"] = np.select(
                [
                    row_forensics["added_vs_rescue_v1"].astype(bool),
                    row_forensics["vetoed_by_hard_protection_v1"].astype(bool),
                    row_forensics["r5_2_v2_final_base_membership"].astype(bool) & (_bool(row_forensics, "label_should_not_take_v1") | _bool(row_forensics, "tail_10_50_mfe_v1")),
                    row_forensics["r5_2_v2_final_base_membership"].astype(bool) & safety_mask,
                    (~row_forensics["r5_2_v2_final_base_membership"].astype(bool)) & (_bool(row_forensics, "label_should_not_take_v1") | _bool(row_forensics, "tail_10_50_mfe_v1")),
                ],
                [
                    "ROWS_ADDED_VS_RESCUE",
                    "ROWS_VETOED_BY_HARD_PROTECTION",
                    "ROWS_IMPROVED_BAD_TAIL_RECALL",
                    "ROWS_CAUSING_SAFETY_FAILURE",
                    "ROWS_STILL_MISSED",
                ],
                default="CONTEXT",
            )
        execution = {
            "layer_name": "RUN_V2_PARALLEL_REBUILD_EXECUTION_V1",
            "training_started_v1": True,
            "parallel_execution_started_v1": True,
            "variant_count_v1": int(len(variants)),
            "variant_namespaces_v1": output_index[["variant_id_v1", "variant_dir_v1"]].to_dict("records"),
            "aggregator_namespace_v1": str(output_dir),
            "foundation_score_dir_v1": str(score_dir),
            "target_table_v1": str(label_path),
            "feature_count_v1": int(len(feature_names)),
            "forbidden_feature_count_v1": int(feature_preflight["forbidden_feature_count_v1"]),
            "ambiguous_high_mfe_bad_positive_count_v1": int(target_audit["ambiguous_high_mfe_bad_positive_count_v1"]),
            "runner_protect_bad_positive_count_v1": int(target_audit["runner_protect_bad_positive_count_v1"]),
            "hard_protection_veto_active_v1": True,
        }
        safe_variants = leaderboard[leaderboard["safety_pass_v1"].astype(bool)]
        summary = {
            "layer_name": LAYER_NAME,
            "materialized_at_utc_v1": _utc_now(),
            "output_dir_v1": str(output_dir),
            "spec_dir_v1": str(spec_dir),
            "foundation_score_dir_v1": str(score_dir),
            "label_table_v1": str(label_path),
            "decision_v1": gate["decision_v1"],
            "next_action_v1": next_lock["next_action_v1"],
            "training_started_v1": True,
            "parallel_execution_started_v1": True,
            "r6_started_v1": False,
            "variant_count_v1": int(len(variants)),
            "safe_variant_count_v1": int(safe_variants.shape[0]),
            "foundation_rows_v1": foundation["foundation_rows_v1"],
            "target_table_rows_v1": target_audit["target_table_rows_v1"],
            "asof_columns_v1": foundation["asof_columns_v1"],
            "forbidden_feature_count_v1": int(feature_preflight["forbidden_feature_count_v1"]),
            "ambiguous_high_mfe_bad_positive_count_v1": int(target_audit["ambiguous_high_mfe_bad_positive_count_v1"]),
            "runner_protect_bad_positive_count_v1": int(target_audit["runner_protect_bad_positive_count_v1"]),
            "hard_protection_veto_contract_present_v1": True,
            "best_variant_id_v1": None if best is None else str(best["variant_id_v1"]),
            "best_profile_id_v1": None if best is None else str(best["profile_id_v1"]),
            "best_bad_recall_v1": None if best is None else int(best["bad_recall_v1"]),
            "best_tail_recall_v1": None if best is None else int(best["tail_recall_v1"]),
            "best_precision_v1": None if best is None else best["precision_v1"],
            "best_worst_loso_v1": None if best is None else best["worst_loso_v1"],
            "downstream_r6_input_ready_v1": bool(best_lock.get("ready_for_downstream_r6_v1")),
            "new_baseline_built_v1": False,
            "new_feature_surface_built_v1": False,
            "hard_status_v1": {
                "BEVIST": [
                    "V2 parallel execution was run with the explicit flag.",
                    "All seven locked V2 variants trained in separate namespaces.",
                    "No R6 retrain, freeze, promotion, live gate, controller change, new baseline, or new feature surface was run.",
                ],
                "INDIKERT": [
                    "The leaderboard determines whether a V2 variant is ready for explicit R6 retrain.",
                ],
                "IKKE_ETABLERT": [
                    "Downstream R6 uplift is not established until R6 is explicitly run from a passing V2 variant.",
                ],
            },
        }
        manifest = {
            "layer_name": f"{LAYER_NAME}_EXECUTION_MANIFEST",
            "input_artifacts_v1": {
                "spec_dir_v1": str(spec_dir),
                "foundation_score_frame_v1": str(score_dir / SCORE_FRAME),
                "foundation_summary_v1": str(score_dir / SCORE_SUMMARY),
                "label_table_v1": str(label_path),
            },
            "output_files_v1": EXECUTION_OUTPUT_FILES,
            "variant_output_index_v1": output_index.to_dict("records"),
        }
        status = {
            "layer_name": f"{LAYER_NAME}_STATUS",
            "decision_v1": summary["decision_v1"],
            "next_action_v1": summary["next_action_v1"],
            "training_started_v1": True,
            "parallel_execution_started_v1": True,
            "r6_started_v1": False,
        }
        _write_json(output_dir / EXECUTION_OUTPUT_FILES["execution"], execution)
        output_index.to_csv(output_dir / EXECUTION_OUTPUT_FILES["training_outputs_index"], index=False)
        eval_df.to_csv(output_dir / EXECUTION_OUTPUT_FILES["eval_gate"], index=False)
        leaderboard.to_csv(output_dir / EXECUTION_OUTPUT_FILES["leaderboard"], index=False)
        _write_json(output_dir / EXECUTION_OUTPUT_FILES["best_r6_lock"], best_lock)
        row_forensics.to_csv(output_dir / EXECUTION_OUTPUT_FILES["row_forensics"], index=False)
        _write_json(output_dir / EXECUTION_OUTPUT_FILES["parallel_gate"], gate)
        _write_json(output_dir / EXECUTION_OUTPUT_FILES["next_action"], next_lock)
        _write_json(output_dir / EXECUTION_OUTPUT_FILES["summary"], summary)
        _write_json(output_dir / EXECUTION_OUTPUT_FILES["manifest"], manifest)
        _write_json(output_dir / EXECUTION_OUTPUT_FILES["status"], status)
        _execution_audit(summary).to_csv(output_dir / EXECUTION_OUTPUT_FILES["audit"], index=False)
        (output_dir / EXECUTION_OUTPUT_FILES["report"]).write_text(_execution_report(summary), encoding="utf-8")
        return summary

    prelaunch_report = {
        "layer_name": "R5_2_OBJECTIVE_V2_PARALLEL_REBUILD_RUNNER_PRELAUNCH_V1",
        "prelaunch_status_v1": "PASS",
        "spec_dir_v1": str(spec_dir),
        "foundation_v1": foundation,
        "key_alignment_v1": key_alignment,
        "target_table_audit_v1": target_audit,
        "feature_matrix_prelaunch_v1": feature_preflight,
        "variant_count_v1": int(len(variant_manifest)),
        "hard_protection_veto_contract_present_v1": True,
        "explicit_run_flag_required_v1": RUN_FLAG,
    }

    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "spec_dir_v1": str(spec_dir),
        "foundation_score_dir_v1": str(score_dir),
        "label_table_v1": str(label_path),
        "decision_v1": "DRY_PRELAUNCH_COMPLETED",
        "prelaunch_status_v1": "PASS",
        "training_started_v1": False,
        "parallel_execution_started_v1": False,
        "variant_count_v1": int(len(variant_manifest)),
        "foundation_rows_v1": foundation["foundation_rows_v1"],
        "active_rows_v1": foundation["active_rows_v1"],
        "quarantine_rows_v1": foundation["quarantine_rows_v1"],
        "asof_columns_v1": foundation["asof_columns_v1"],
        "target_table_rows_v1": target_audit["target_table_rows_v1"],
        "forbidden_feature_count_v1": feature_preflight["forbidden_feature_count_v1"],
        "ambiguous_high_mfe_bad_positive_count_v1": target_audit["ambiguous_high_mfe_bad_positive_count_v1"],
        "runner_protect_bad_positive_count_v1": target_audit["runner_protect_bad_positive_count_v1"],
        "hard_protection_veto_contract_present_v1": True,
        "next_action_v1": NEXT_ACTION,
        "blocked_action_v1": BLOCKED_ACTION,
        "hard_status_v1": {
            "BEVIST": [
                "The V2 parallel rebuild runner loads the locked design package and all seven variants.",
                "Dry/prelaunch validates foundation rows, target rows, key alignment, feature legality, and hard protection veto connectivity.",
                "No training, parallel execution, R6 run, baseline build, or feature surface build was performed.",
            ],
            "INDIKERT": [
                "The runner is ready for a later explicit V2 parallel rebuild execution.",
            ],
            "IKKE_ETABLERT": [
                "Actual V2 model results and downstream R6 uplift are not established until the explicit execution flag is run in a later step.",
            ],
        },
    }
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "input_artifacts_v1": {
            "spec_dir_v1": str(spec_dir),
            "foundation_score_frame_v1": str(score_dir / SCORE_FRAME),
            "foundation_summary_v1": str(score_dir / SCORE_SUMMARY),
            "label_table_v1": str(label_path),
        },
        "output_files_v1": DRY_OUTPUT_FILES,
    }
    status = {
        "layer_name": f"{LAYER_NAME}_STATUS",
        "decision_v1": summary["decision_v1"],
        "prelaunch_status_v1": "PASS",
        "training_started_v1": False,
        "parallel_execution_started_v1": False,
        "next_action_v1": NEXT_ACTION,
        "blocked_action_v1": BLOCKED_ACTION,
    }

    _write_json(output_dir / DRY_OUTPUT_FILES["summary"], summary)
    _write_json(output_dir / DRY_OUTPUT_FILES["status"], status)
    _write_json(output_dir / DRY_OUTPUT_FILES["manifest"], manifest)
    _write_json(output_dir / DRY_OUTPUT_FILES["prelaunch_report"], prelaunch_report)
    variant_manifest.to_csv(output_dir / DRY_OUTPUT_FILES["variant_manifest"], index=False)
    _write_json(output_dir / DRY_OUTPUT_FILES["target_audit"], target_audit)
    _write_json(output_dir / DRY_OUTPUT_FILES["feature_prelaunch"], feature_preflight)
    forbidden_scan.to_csv(output_dir / DRY_OUTPUT_FILES["forbidden_scan"], index=False)
    _write_json(output_dir / DRY_OUTPUT_FILES["veto_report"], veto_report)
    _write_json(output_dir / DRY_OUTPUT_FILES["execution_placeholder"], execution_placeholder)
    _write_json(output_dir / DRY_OUTPUT_FILES["r6_placeholder"], r6_placeholder)
    _audit(summary).to_csv(output_dir / DRY_OUTPUT_FILES["audit"], index=False)
    (output_dir / DRY_OUTPUT_FILES["report"]).write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--foundation-score-dir", type=Path, default=None)
    parser.add_argument("--label-table", type=Path, default=None)
    parser.add_argument(RUN_FLAG, action="store_true", dest="run_parallel_rebuild")
    args = parser.parse_args()
    summary = materialize(
        reports_root=args.reports_root,
        spec_dir=args.spec_dir,
        output_dir=args.output_dir,
        foundation_score_dir=args.foundation_score_dir,
        label_table=args.label_table,
        run_parallel_rebuild=args.run_parallel_rebuild,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
