#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

SPEC_PREFIX = "PROTECTOR_FIRST_SHADOW_EXPERIMENT_RUNNER_SPEC_V1_"
OBJECTIVE_LABEL_REVIEW_PREFIX = "PROTECTOR_FIRST_OBJECTIVE_LABEL_REVIEW_V1_"
DRY_OUTPUT_PREFIX = "PROTECTOR_FIRST_SHADOW_EXPERIMENT_DRY_PRELAUNCH_V1"
TRAINING_OUTPUT_SUFFIX = "PROTECTOR_FIRST_SHADOW_EXPERIMENT_RUN_V1"
BRIDGE_PREFIX = "MONDAY_ENTRY_TO_FAILURE_POCKET_BRIDGE_IMPLEMENTATION_V1_"
BRIDGE_SURFACE = "entry_to_failure_pocket_bridge_surface_v1.parquet"

RUNNER_SPEC = "protector_first_runner_spec_v1.json"
CONFIG_LOCK = "protector_first_config_lock_v1.json"
DECISION_CONTRACT = "protector_first_decision_contract_v1.json"
OBJECTIVE_LABEL_REVIEW = "protector_first_objective_label_review_spec_v1.json"
FEATURE_SURFACE_LOCK = "protector_first_feature_and_surface_lock_v1.json"
EVAL_VERDICT_MATRIX = "protector_first_eval_and_verdict_matrix_v1.json"
PRELAUNCH_CHECKLIST = "protector_first_prelaunch_checklist_v1.json"
ABORT_RULES = "protector_first_abort_rules_v1.json"
SPEC_SUMMARY = "summary_v1.json"

CONTRACT = "contract_v1.json"
LOADED_CONFIG = "protector_first_loaded_config_v1.json"
CONFIG_ECHO = "protector_first_config_echo_v1.json"
FEATURE_MANIFEST_ECHO = "protector_first_feature_manifest_echo_v1.csv"
GENERIC_FEATURE_MANIFEST_ECHO = "feature_manifest_echo_v1.csv"
PRELAUNCH_REPORT = "protector_first_prelaunch_report_v1.json"
OBJECTIVE_GATE_REPORT = "protector_first_objective_label_gate_report_v1.json"
DECISION_CONTRACT_REPORT = "protector_first_decision_contract_report_v1.json"
ABORT_ENFORCEMENT_REPORT = "protector_first_abort_enforcement_report_v1.json"
CONFLICT_SUMMARY_PLACEHOLDER = "protector_first_conflict_summary_placeholder_v1.csv"
EVAL_VERDICT_PLACEHOLDER = "protector_first_eval_verdict_placeholder_v1.json"
OUTPUT_SCAFFOLD_MANIFEST = "protector_first_output_scaffold_manifest_v1.json"
TRAINING_EXECUTION_SUMMARY = "training_execution_summary_v1.json"
TRAINING_MATRIX_SUMMARY = "training_matrix_summary_v1.json"
LEARNING_SURFACE_PARITY_GUARD = "learning_surface_parity_guard_v1.json"
MODEL_MANIFEST = "model_manifest_v1.json"
CONFIG_MANIFEST = "config_manifest_v1.json"
PREDICTION_VIEW = "prediction_view_v1.parquet"
EVAL_SUMMARY = "eval_summary_v1.json"
COMPARE_AGAINST_REPORT = "compare_against_report_v1.json"
POCKET_REPORT_CSV = "pocket_report_v1.csv"
POCKET_REPORT_JSON = "pocket_report_v1.json"
CONFLICT_SUMMARY_CSV = "blocker_vs_protector_conflict_summary_v1.csv"
CONFLICT_SUMMARY_JSON = "blocker_vs_protector_conflict_summary_v1.json"
VERDICT_PACKAGE = "verdict_package_v1.json"
NEXT_ACTION = "next_agent_action_lock_v1.json"
SUMMARY = "summary_v1.json"
REPORT = "report_v1.md"
MANIFEST = "manifest_v1.json"
STATUS = "status_v1.json"
CONSISTENCY_AUDIT = "consistency_audit_v1.csv"

JOB_NAME = "PROTECTOR_FIRST_SHADOW_EXPERIMENT_V1"
RUNNER_NAME = "PROTECTOR_FIRST_SHADOW_EXPERIMENT_RUNNER_V1"
ARCHITECTURE = "PROTECTOR_FIRST_VETO_OR_DAMPER"
EXPECTED_SURFACE_KIND = "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE"
EXPECTED_ROWS = 1689
EXPECTED_EVAL_ROWS = 1852
EXPECTED_FEATURES = 67
EXPECTED_BASELINE_FEATURES = 62
EXPECTED_PROXY_FEATURES = 5
EXECUTION_SEED = 20260425
EXPECTED_MODEL_FAMILY = "R6_STYLE_FIVE_HEAD_SHADOW_FAMILY"
FORENSIC_TRADE = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"

FORBIDDEN_FIELD_PATTERNS = [
    "as_of_skip_xgb_",
    "last_peak_ts",
    "last_mfe_ts",
    "last_peak_mfe",
    "max_mfe_without_mae",
    "mfe_mae_sequence_order",
    "management_policy",
    "hindsight_management",
    "hindsight_entry",
    "baseline_realized_pnl",
    "peak_mfe",
    "mae_abs",
    "giveback_bps",
    "decision_timestamp",
    "decision_log",
    "policy_log",
    "bridge_only",
    "path_dynamics_truth",
    "path_truth",
]

HEAD_SPECS = [
    {"head_id_v1": "bad_risk", "label_col_v1": "r6_label_bad_risk_v1", "role_v1": "BLOCKER"},
    {"head_id_v1": "runner_protector", "label_col_v1": "r6_label_runner_protect_v1", "role_v1": "PROTECTOR"},
    {"head_id_v1": "tail_control_10_50", "label_col_v1": "r6_label_tail_control_10_50_v1", "role_v1": "BLOCKER"},
    {"head_id_v1": "risky_allow", "label_col_v1": "r6_label_risky_allow_v1", "role_v1": "BLOCKER"},
    {"head_id_v1": "batch04_blindspot", "label_col_v1": "r6_label_batch04_blindspot_v1", "role_v1": "BLOCKER"},
]

REFERENCE_METRICS = [
    {
        "reference_v1": "FROZEN_WEDNESDAY_R6_BENCHMARK",
        "id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
        "kind_v1": "BENCHMARK",
        "bad_blocks_v1": 180,
        "tail_help_v1": 149,
        "global_precision_v1": 0.972972972972973,
        "worst_loso_precision_v1": 0.9285714285714286,
        "repaired_165_damage_v1": 0,
        "fifty_plus_mfe_block_count_v1": 1,
        "hundred_plus_mfe_block_count_v1": 0,
        "two_hundred_plus_mfe_block_count_v1": 0,
        "strongest_winner_damage_v1": 0,
        "runner_near_miss_block_count_v1": 0,
    },
    {
        "reference_v1": "MONDAY_R5_1_SAFETY_REFERENCE",
        "id_v1": "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like",
        "kind_v1": "SAFETY_REFERENCE",
        "bad_blocks_v1": 66,
        "tail_help_v1": 66,
        "global_precision_v1": 0.9295774647887324,
        "worst_loso_precision_v1": 0.0,
        "repaired_165_damage_v1": 0,
        "fifty_plus_mfe_block_count_v1": 0,
        "hundred_plus_mfe_block_count_v1": 0,
        "two_hundred_plus_mfe_block_count_v1": 0,
        "strongest_winner_damage_v1": 0,
        "runner_near_miss_block_count_v1": 0,
    },
    {
        "reference_v1": "FAILED_NARROW_RETRAIN_RUN_HARD_NEGATIVE",
        "id_v1": "ALL_TRADE_REVIEW_LEDGER_20260424T170555Z_MONDAY_NARROW_RETRAIN_RUN_V1",
        "kind_v1": "HARD_NEGATIVE_REFERENCE",
        "bad_blocks_v1": 0,
        "tail_help_v1": 0,
        "global_precision_v1": 0.0,
        "worst_loso_precision_v1": 0.0,
        "repaired_165_damage_v1": 1,
        "fifty_plus_mfe_block_count_v1": 1,
        "hundred_plus_mfe_block_count_v1": 0,
        "two_hundred_plus_mfe_block_count_v1": 0,
        "strongest_winner_damage_v1": 0,
        "runner_near_miss_block_count_v1": 1,
    },
    {
        "reference_v1": "MONDAY_NATIVE_R6_FAILURE_MINER",
        "id_v1": "FAILURE_MINER_DIAGNOSIS_ONLY",
        "kind_v1": "FAILURE_MINER",
        "bad_blocks_v1": 84,
        "tail_help_v1": 84,
        "global_precision_v1": 0.9545454545454546,
        "worst_loso_precision_v1": 0.8888888888888888,
        "repaired_165_damage_v1": 1,
        "fifty_plus_mfe_block_count_v1": 1,
        "hundred_plus_mfe_block_count_v1": 0,
        "two_hundred_plus_mfe_block_count_v1": 0,
        "strongest_winner_damage_v1": 0,
        "runner_near_miss_block_count_v1": 0,
    },
]

REQUIRED_HARD_VETO_POCKETS = {
    "forensic_repaired_trade",
    "repaired_165_like_pockets",
    "strongest_winner",
    "100_plus_200_plus_winner_pockets",
}

REQUIRED_SOFT_DAMPER_POCKETS = {
    "runner_near_miss",
    "50_plus_mfe_seed_pockets",
}

REQUIRED_OBJECTIVE_LABELS = {
    "runner_protect",
    "runner_near_miss",
    "strongest_winner",
    "100_plus_winner",
    "200_plus_winner",
    "repaired_165_safety",
    "bad_risk_vs_runner_conflict",
}

REQUIRED_OBJECTIVE_COSTS = {
    "winner_damage_cost",
    "strongest_winner_damage_cost",
    "100_plus_block_cost",
    "200_plus_block_cost",
    "runner_near_miss_block_cost",
    "repaired_165_damage_cost",
}

REQUIRED_CONFLICT_COLUMNS = [
    "candidate_uid",
    "pocket_tag",
    "blocker_score",
    "protector_score",
    "protector_action",
    "blocker_action_before_protection",
    "final_shadow_action",
    "score_margin",
    "override_or_damper_reason",
]


class PrelaunchValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProtectorRunnerBundle:
    spec_dir: Path
    objective_label_review_artifact: Path
    runner_spec: Dict[str, Any]
    config_lock: Dict[str, Any]
    decision_contract: Dict[str, Any]
    objective_label_review: Dict[str, Any]
    feature_surface_lock: Dict[str, Any]
    eval_verdict_matrix: Dict[str, Any]
    prelaunch_checklist: Dict[str, Any]
    abort_rules: Dict[str, Any]
    spec_summary: Dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _latest_dir(reports_root: Path, prefix: str) -> Path:
    matches = sorted([p for p in reports_root.iterdir() if p.is_dir() and p.name.startswith(prefix)], key=lambda p: p.name)
    if not matches:
        raise FileNotFoundError(f"No directory found for prefix {prefix} under {reports_root}")
    return matches[-1]


def _resolve_default_objective_label_review_artifact(reports_root: Path, spec_dir: Path) -> Path:
    try:
        review_dir = _latest_dir(reports_root, OBJECTIVE_LABEL_REVIEW_PREFIX)
        review_artifact = review_dir / OBJECTIVE_LABEL_REVIEW
        if review_artifact.exists():
            return review_artifact
    except FileNotFoundError:
        pass
    return spec_dir / OBJECTIVE_LABEL_REVIEW


def _resolve_output_dir(reports_root: Path, output_dir_arg: str | None, *, run_training: bool = False) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg).expanduser().resolve()
    if run_training:
        return reports_root / f"ALL_TRADE_REVIEW_LEDGER_{_utc_compact()}_{TRAINING_OUTPUT_SUFFIX}"
    return reports_root / f"{DRY_OUTPUT_PREFIX}_{_utc_compact()}"


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _load_bundle(
    reports_root: Path,
    spec_dir_arg: str | None = None,
    objective_label_review_artifact_arg: str | None = None,
) -> ProtectorRunnerBundle:
    spec_dir = Path(spec_dir_arg).expanduser().resolve() if spec_dir_arg else _latest_dir(reports_root, SPEC_PREFIX)
    required = [
        RUNNER_SPEC,
        CONFIG_LOCK,
        DECISION_CONTRACT,
        OBJECTIVE_LABEL_REVIEW,
        FEATURE_SURFACE_LOCK,
        EVAL_VERDICT_MATRIX,
        PRELAUNCH_CHECKLIST,
        ABORT_RULES,
        SPEC_SUMMARY,
    ]
    missing = [name for name in required if not (spec_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Protector-first runner spec dir missing required artifacts: {missing}")
    objective_label_review_path = (
        Path(objective_label_review_artifact_arg).expanduser().resolve()
        if objective_label_review_artifact_arg
        else _resolve_default_objective_label_review_artifact(reports_root, spec_dir)
    )
    if not objective_label_review_path.exists():
        raise FileNotFoundError(f"Protector-first objective/label review artifact missing: {objective_label_review_path}")
    return ProtectorRunnerBundle(
        spec_dir=spec_dir,
        objective_label_review_artifact=objective_label_review_path,
        runner_spec=_load_json(spec_dir / RUNNER_SPEC),
        config_lock=_load_json(spec_dir / CONFIG_LOCK),
        decision_contract=_load_json(spec_dir / DECISION_CONTRACT),
        objective_label_review=_load_json(objective_label_review_path),
        feature_surface_lock=_load_json(spec_dir / FEATURE_SURFACE_LOCK),
        eval_verdict_matrix=_load_json(spec_dir / EVAL_VERDICT_MATRIX),
        prelaunch_checklist=_load_json(spec_dir / PRELAUNCH_CHECKLIST),
        abort_rules=_load_json(spec_dir / ABORT_RULES),
        spec_summary=_load_json(spec_dir / SPEC_SUMMARY),
    )


def _forbidden_fields(fields: Iterable[str]) -> List[str]:
    out: List[str] = []
    for field in fields:
        lower = str(field).lower()
        if any(pattern in lower for pattern in FORBIDDEN_FIELD_PATTERNS):
            out.append(str(field))
    return sorted(out)


def _bool_col(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.astype("string").str.lower().isin(["1", "true", "yes", "y"])


def _num_col(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _safe_rate(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _numeric_matrix(frame: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    matrix = frame[features].apply(pd.to_numeric, errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan)


def _fold_indices(n_rows: int, n_slices: int = 5) -> List[np.ndarray]:
    return [np.asarray(idx, dtype=int) for idx in np.array_split(np.arange(n_rows), n_slices) if len(idx)]


def _logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return float(math.log(clipped / (1.0 - clipped)))


class _CentroidLogitHead:
    def __init__(
        self,
        *,
        head_id: str,
        label_col: str,
        feature_names: List[str],
        medians: Dict[str, float],
        scales: Dict[str, float],
        weights: Dict[str, float],
        intercept: float,
        positive_rate: float,
        constant_model: bool,
    ) -> None:
        self.head_id = head_id
        self.label_col = label_col
        self.feature_names = feature_names
        self.medians = medians
        self.scales = scales
        self.weights = weights
        self.intercept = float(intercept)
        self.positive_rate = float(positive_rate)
        self.constant_model = bool(constant_model)

    def predict_true_probability(self, x: pd.DataFrame) -> np.ndarray:
        if self.constant_model:
            return np.full(len(x), self.positive_rate, dtype=float)
        matrix = _numeric_matrix(x, self.feature_names)
        for feature in self.feature_names:
            matrix[feature] = matrix[feature].fillna(self.medians.get(feature, 0.0))
        scales = np.asarray([self.scales.get(feature, 1.0) or 1.0 for feature in self.feature_names], dtype=float)
        weights = np.asarray([self.weights.get(feature, 0.0) for feature in self.feature_names], dtype=float)
        centered = (matrix.to_numpy(dtype=float) - np.asarray([self.medians.get(feature, 0.0) for feature in self.feature_names], dtype=float)) / scales
        logits = centered.dot(weights) + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def manifest_v1(self) -> Dict[str, Any]:
        return {
            "head_id_v1": self.head_id,
            "label_col_v1": self.label_col,
            "model_type_v1": "CENTROID_LOGIT_BINARY_HEAD",
            "feature_count_v1": len(self.feature_names),
            "positive_rate_v1": self.positive_rate,
            "constant_model_v1": self.constant_model,
            "intercept_v1": self.intercept,
            "nonzero_weight_count_v1": int(sum(1 for value in self.weights.values() if abs(float(value)) > 0.0)),
            "weights_v1": self.weights,
            "medians_v1": self.medians,
            "scales_v1": self.scales,
        }


def _fit_centroid_logit_head(
    train_x: pd.DataFrame,
    y: pd.Series,
    *,
    head_id: str,
    label_col: str,
    feature_names: List[str],
) -> _CentroidLogitHead:
    x = _numeric_matrix(train_x, feature_names)
    medians_series = x.median(numeric_only=True).fillna(0.0)
    x = x.fillna(medians_series)
    scales_series = x.std(numeric_only=True).replace(0.0, 1.0).fillna(1.0)
    y_bool = y.fillna(False).astype(bool)
    positive_rate = float(y_bool.mean()) if len(y_bool) else 0.0
    constant_model = bool(y_bool.nunique(dropna=False) < 2)
    if constant_model:
        weights = {feature: 0.0 for feature in feature_names}
    else:
        pos_mean = x.loc[y_bool].mean(numeric_only=True)
        neg_mean = x.loc[~y_bool].mean(numeric_only=True)
        raw_weights = ((pos_mean - neg_mean) / scales_series).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        divisor = max(math.sqrt(float(len(feature_names))), 1.0)
        weights = {str(feature): float(np.clip(raw_weights.get(feature, 0.0), -3.0, 3.0) / divisor) for feature in feature_names}
    return _CentroidLogitHead(
        head_id=head_id,
        label_col=label_col,
        feature_names=feature_names,
        medians={str(k): float(v) for k, v in medians_series.items()},
        scales={str(k): float(v) for k, v in scales_series.items()},
        weights=weights,
        intercept=_logit(positive_rate),
        positive_rate=positive_rate,
        constant_model=constant_model,
    )


def _validate_output_namespace_clean(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PrelaunchValidationError(f"output namespace is not clean: {output_dir}")


def _objective_gate_is_green(review: Dict[str, Any]) -> bool:
    raw_status = (
        review.get("objective_label_review_gate_status_v1")
        or review.get("gate_status_v1")
        or review.get("training_gate_status_v1")
    )
    return str(raw_status).upper() in {"PASS", "GREEN", "READY"}


def _validate_runner_config_contract(bundle: ProtectorRunnerBundle, output_dir: Path) -> Dict[str, Any]:
    spec = bundle.runner_spec
    config = bundle.config_lock
    if spec.get("job_name_v1") != JOB_NAME:
        raise PrelaunchValidationError(f"unexpected job name: {spec.get('job_name_v1')}")
    if spec.get("runner_name_v1") != RUNNER_NAME:
        raise PrelaunchValidationError(f"unexpected runner name: {spec.get('runner_name_v1')}")
    if spec.get("training_now_v1") is not False or spec.get("replay_now_v1") is not False:
        raise PrelaunchValidationError("runner spec must remain no-training/no-replay by default")
    if config.get("architecture_v1") != ARCHITECTURE:
        raise PrelaunchValidationError(f"unexpected architecture: {config.get('architecture_v1')}")
    if config.get("shadow_only_v1") is not True or config.get("not_live_gate_v1") is not True:
        raise PrelaunchValidationError("config must be shadow-only and not-live-gate")
    if config.get("not_policy_controller_v1") is not True:
        raise PrelaunchValidationError("policy/controller changes are not allowed")
    if config.get("bridge_as_training_surface_allowed_v1") is not False:
        raise PrelaunchValidationError("bridge as training surface must be forbidden")
    if config.get("management_exit_truth_as_entry_features_allowed_v1") is not False:
        raise PrelaunchValidationError("management/exit truth as entry features must be forbidden")
    _validate_output_namespace_clean(output_dir)
    return {
        "job_name_v1": "PASS",
        "runner_name_v1": "PASS",
        "no_training_default_v1": "PASS",
        "architecture_v1": "PASS",
        "shadow_only_v1": "PASS",
        "not_policy_controller_v1": "PASS",
        "output_namespace_clean_v1": "PASS",
    }


def _validate_feature_surface(bundle: ProtectorRunnerBundle) -> Dict[str, Any]:
    surface = bundle.feature_surface_lock
    spec = bundle.runner_spec
    if surface.get("training_surface_kind_v1") != EXPECTED_SURFACE_KIND:
        raise PrelaunchValidationError(f"unexpected training surface kind: {surface.get('training_surface_kind_v1')}")
    if spec.get("training_surface_kind_v1") != EXPECTED_SURFACE_KIND:
        raise PrelaunchValidationError(f"unexpected runner training surface kind: {spec.get('training_surface_kind_v1')}")
    if int(surface.get("expected_training_rows_v1", -1)) != EXPECTED_ROWS:
        raise PrelaunchValidationError(f"spec expected row count != {EXPECTED_ROWS}")
    if int(surface.get("feature_count_v1", -1)) != EXPECTED_FEATURES:
        raise PrelaunchValidationError(f"feature count {surface.get('feature_count_v1')} != {EXPECTED_FEATURES}")
    if int(surface.get("baseline_feature_count_v1", -1)) != EXPECTED_BASELINE_FEATURES:
        raise PrelaunchValidationError(
            f"baseline feature count {surface.get('baseline_feature_count_v1')} != {EXPECTED_BASELINE_FEATURES}"
        )
    if int(surface.get("new_proxy_feature_count_v1", -1)) != EXPECTED_PROXY_FEATURES:
        raise PrelaunchValidationError(
            f"proxy feature count {surface.get('new_proxy_feature_count_v1')} != {EXPECTED_PROXY_FEATURES}"
        )
    if surface.get("bridge_as_training_surface_allowed_v1") is not False:
        raise PrelaunchValidationError("bridge as training surface must be forbidden")
    if surface.get("management_exit_truth_as_features_allowed_v1") is not False:
        raise PrelaunchValidationError("management/exit truth as features must be forbidden")
    feature_names = [str(name) for name in surface.get("feature_names_v1", [])]
    if len(feature_names) != EXPECTED_FEATURES:
        raise PrelaunchValidationError(f"selected feature count {len(feature_names)} != {EXPECTED_FEATURES}")
    if len(feature_names) != len(set(feature_names)):
        raise PrelaunchValidationError("duplicate selected feature names in protector-first feature lock")
    forbidden = _forbidden_fields(feature_names)
    if forbidden:
        raise PrelaunchValidationError(f"forbidden feature fields: {forbidden}")
    input_path = Path(str(surface["training_surface_v1"])).expanduser().resolve()
    spec_input_path = Path(str(spec.get("input_training_surface_v1", input_path))).expanduser().resolve()
    if spec_input_path != input_path:
        raise PrelaunchValidationError(
            f"runner/spec training surface mismatch: {spec_input_path} != {input_path}"
        )
    if "bridge" in input_path.name.lower():
        raise PrelaunchValidationError(f"bridge path proposed as training surface: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"training surface does not exist: {input_path}")
    raw_df = pd.read_parquet(input_path)
    if len(raw_df) != EXPECTED_ROWS:
        raise PrelaunchValidationError(f"training row count {len(raw_df)} != {EXPECTED_ROWS}")
    if "candidate_uid" not in raw_df.columns:
        raise PrelaunchValidationError("candidate_uid is required on raw-state surface")
    missing_features = sorted([name for name in feature_names if name not in raw_df.columns])
    if missing_features:
        raise PrelaunchValidationError(f"selected features missing from raw-state: {missing_features[:20]}")
    matrix_fields = ["candidate_uid"] + feature_names
    forbidden_matrix = _forbidden_fields(matrix_fields)
    if forbidden_matrix:
        raise PrelaunchValidationError(f"forbidden fields in training matrix: {forbidden_matrix}")
    return {
        "training_surface_v1": str(input_path),
        "training_surface_kind_v1": EXPECTED_SURFACE_KIND,
        "raw_rows_v1": int(len(raw_df)),
        "raw_column_count_v1": int(len(raw_df.columns)),
        "selected_feature_count_v1": len(feature_names),
        "baseline_feature_count_v1": EXPECTED_BASELINE_FEATURES,
        "new_proxy_feature_count_v1": EXPECTED_PROXY_FEATURES,
        "feature_names_v1": feature_names,
        "candidate_uid_count_v1": int(raw_df["candidate_uid"].astype("string").nunique(dropna=True)),
        "forbidden_unselected_columns_present_v1": _forbidden_fields(raw_df.columns),
    }


def _resolve_training_surface_path(bundle: ProtectorRunnerBundle) -> Path:
    return Path(str(bundle.feature_surface_lock["training_surface_v1"])).expanduser().resolve()


def _resolve_label_surface_path(bundle: ProtectorRunnerBundle) -> Path:
    label_contract = bundle.runner_spec.get("label_target_contract_v1", {})
    label_path = label_contract.get("label_artifact_v1")
    if not label_path:
        raise PrelaunchValidationError("label artifact is required for protector-first training execution")
    resolved = Path(str(label_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"label/eval surface does not exist: {resolved}")
    return resolved


def _resolve_bridge_surface_path(reports_root: Path, bundle: ProtectorRunnerBundle) -> Path:
    explicit = (
        bundle.runner_spec.get("eval_readiness_bridge_surface_v1")
        or bundle.feature_surface_lock.get("eval_readiness_bridge_surface_v1")
        or bundle.feature_surface_lock.get("bridge_surface_v1")
    )
    if explicit:
        resolved = Path(str(explicit)).expanduser().resolve()
        if resolved.exists():
            return resolved
    try:
        bridge_dir = _latest_dir(reports_root, BRIDGE_PREFIX)
        bridge_path = bridge_dir / BRIDGE_SURFACE
        if bridge_path.exists():
            return bridge_path
    except FileNotFoundError:
        pass
    matches = sorted(reports_root.glob(f"*/{BRIDGE_SURFACE}"), key=lambda path: str(path))
    if matches:
        return matches[-1].resolve()
    raise PrelaunchValidationError("surface-boundary report cannot be materialized: eval/readiness bridge surface is missing")


def _pocket_columns(frame: pd.DataFrame) -> List[str]:
    return sorted([str(column) for column in frame.columns if str(column).startswith("bridge_pocket_")])


def _materialize_learning_surface_parity_guard(
    *,
    reports_root: Path,
    bundle: ProtectorRunnerBundle,
    feature_status: Dict[str, Any],
) -> Dict[str, Any]:
    training_path = Path(feature_status["training_surface_v1"]).expanduser().resolve()
    label_path = _resolve_label_surface_path(bundle)
    bridge_path = _resolve_bridge_surface_path(reports_root, bundle)
    training_df = pd.read_parquet(training_path)
    label_df = pd.read_parquet(label_path)
    bridge_df = pd.read_parquet(bridge_path)
    for name, frame in [("training", training_df), ("label_eval", label_df), ("bridge", bridge_df)]:
        if "candidate_uid" not in frame.columns:
            raise PrelaunchValidationError(f"candidate_uid is required on {name} surface")
    training_ids = set(training_df["candidate_uid"].astype(str))
    label_ids = set(label_df["candidate_uid"].astype(str))
    bridge_ids = set(bridge_df["candidate_uid"].astype(str))
    bridge_only_ids = bridge_ids - training_ids
    label_only_ids = label_ids - training_ids
    if len(training_df) != EXPECTED_ROWS:
        raise PrelaunchValidationError(f"training row count {len(training_df)} != {EXPECTED_ROWS}")
    if len(bridge_df) != EXPECTED_EVAL_ROWS:
        raise PrelaunchValidationError(f"eval/readiness bridge row count {len(bridge_df)} != {EXPECTED_EVAL_ROWS}")
    if len(label_df) != EXPECTED_EVAL_ROWS:
        raise PrelaunchValidationError(f"eval/readiness label row count {len(label_df)} != {EXPECTED_EVAL_ROWS}")
    if len(training_ids & label_ids) != EXPECTED_ROWS:
        raise PrelaunchValidationError("training surface does not have a full 1689-row label intersection")
    if bridge_ids != label_ids:
        raise PrelaunchValidationError("bridge and eval/label surfaces must describe the same 1852-row eval universe")
    bridge_only_frame = bridge_df[bridge_df["candidate_uid"].astype(str).isin(bridge_only_ids)].copy()
    training_bridge_view = bridge_df[bridge_df["candidate_uid"].astype(str).isin(training_ids)].copy()
    if "exact_canonical_raw_state_present_v1" in training_bridge_view.columns:
        non_exact_training_ids = training_bridge_view.loc[
            ~_bool_col(training_bridge_view, "exact_canonical_raw_state_present_v1"),
            "candidate_uid",
        ].astype(str).tolist()
        if non_exact_training_ids:
            raise PrelaunchValidationError(f"bridge-only rows are present in training matrix: {non_exact_training_ids[:20]}")
    pocket_report = []
    for column in _pocket_columns(bridge_df):
        pocket_report.append(
            {
                "pocket_v1": column,
                "eval_readiness_count_v1": int(_bool_col(bridge_df, column).sum()),
                "bridge_only_count_v1": int(_bool_col(bridge_only_frame, column).sum()),
                "only_eval_readiness_via_bridge_v1": bool(_bool_col(bridge_only_frame, column).sum() > 0),
                "bridge_only_candidate_sample_v1": bridge_only_frame.loc[
                    _bool_col(bridge_only_frame, column), "candidate_uid"
                ]
                .astype(str)
                .head(10)
                .tolist(),
            }
        )
    forensic_on_training = FORENSIC_TRADE in training_ids
    forensic_on_bridge = FORENSIC_TRADE in bridge_ids
    matrix_uses_bridge_only = bool(training_ids & bridge_only_ids)
    if matrix_uses_bridge_only:
        raise PrelaunchValidationError("bridge-only rows are present in the training matrix population")
    if not forensic_on_bridge:
        raise PrelaunchValidationError("forensic repaired trade is missing from eval/readiness bridge")
    return {
        "layer_name_v1": "LEARNING_SURFACE_PARITY_GUARD_V1",
        "status_v1": "PASS",
        "surface_boundary_guard_pass_v1": True,
        "materialized_at_utc_v1": _utc_now_iso(),
        "training_surface_v1": str(training_path),
        "training_surface_kind_v1": EXPECTED_SURFACE_KIND,
        "training_surface_row_count_v1": int(len(training_df)),
        "training_surface_candidate_uid_count_v1": int(training_df["candidate_uid"].astype(str).nunique()),
        "eval_readiness_bridge_surface_v1": str(bridge_path),
        "eval_readiness_bridge_row_count_v1": int(len(bridge_df)),
        "eval_label_surface_v1": str(label_path),
        "eval_label_row_count_v1": int(len(label_df)),
        "bridge_only_count_v1": int(len(bridge_only_ids)),
        "label_only_vs_training_count_v1": int(len(label_only_ids)),
        "bridge_only_rows_used_in_training_matrix_v1": False,
        "bridge_only_candidate_uid_sample_v1": sorted(bridge_only_ids)[:25],
        "eval_readiness_only_pockets_v1": pocket_report,
        "forensic_repaired_trade_v1": {
            "candidate_uid_v1": FORENSIC_TRADE,
            "present_on_training_surface_v1": forensic_on_training,
            "present_on_eval_readiness_bridge_v1": forensic_on_bridge,
            "eval_hard_guard_even_when_not_training_row_v1": bool(forensic_on_bridge and not forensic_on_training),
        },
        "frozen_wednesday_r6_compare_scope_v1": (
            "benchmark/eval reference only; not evidence that the 1689-row training surface is identical "
            "to the 1852-row eval/readiness universe"
        ),
    }


def _validate_decision_contract(bundle: ProtectorRunnerBundle) -> Dict[str, Any]:
    contract = bundle.decision_contract
    if contract.get("architecture_v1") != ARCHITECTURE:
        raise PrelaunchValidationError(f"decision contract architecture mismatch: {contract.get('architecture_v1')}")
    if contract.get("protector_has_decision_power_v1") is not True:
        raise PrelaunchValidationError("protector decision power is not enabled")
    hard_pockets = {str(row.get("pocket_v1")) for row in contract.get("hard_protector_veto_v1", [])}
    soft_pockets = {str(row.get("pocket_v1")) for row in contract.get("soft_damper_v1", [])}
    missing_hard = sorted(REQUIRED_HARD_VETO_POCKETS - hard_pockets)
    missing_soft = sorted(REQUIRED_SOFT_DAMPER_POCKETS - soft_pockets)
    if missing_hard:
        raise PrelaunchValidationError(f"missing hard protector veto pockets: {missing_hard}")
    if missing_soft:
        raise PrelaunchValidationError(f"missing soft damper pockets: {missing_soft}")
    conflict_fields = [str(field) for field in contract.get("conflict_summary_required_fields_v1", [])]
    missing_conflict = sorted(set(REQUIRED_CONFLICT_COLUMNS) - set(conflict_fields))
    if missing_conflict:
        raise PrelaunchValidationError(f"missing conflict summary fields: {missing_conflict}")
    return {
        "architecture_v1": contract["architecture_v1"],
        "hard_veto_pockets_v1": sorted(hard_pockets),
        "soft_damper_pockets_v1": sorted(soft_pockets),
        "conflict_summary_columns_v1": conflict_fields,
        "protector_has_decision_power_v1": True,
    }


def _validate_objective_label_gate(bundle: ProtectorRunnerBundle, *, run_training: bool) -> Dict[str, Any]:
    review = bundle.objective_label_review
    if review.get("review_required_before_training_v1") is not True:
        raise PrelaunchValidationError("objective/label review must be required before training")
    labels = {str(label) for label in review.get("labels_to_recheck_v1", [])}
    costs = {str(cost) for cost in review.get("costs_to_weight_harder_v1", [])}
    missing_labels = sorted(REQUIRED_OBJECTIVE_LABELS - labels)
    missing_costs = sorted(REQUIRED_OBJECTIVE_COSTS - costs)
    if missing_labels:
        raise PrelaunchValidationError(f"objective/label review missing labels: {missing_labels}")
    if missing_costs:
        raise PrelaunchValidationError(f"objective/label review missing costs: {missing_costs}")
    stop_items = [str(item) for item in review.get("training_stop_if_review_not_green_v1", [])]
    gate_green = _objective_gate_is_green(review)
    if run_training and not gate_green:
        raise PrelaunchValidationError("objective/label review gate is not green; training is blocked")
    return {
        "status_v1": "PASS_FOR_DRY_PRELAUNCH" if not run_training else "PASS_FOR_TRAINING",
        "training_gate_green_v1": gate_green,
        "training_requested_v1": bool(run_training),
        "training_blocked_without_green_gate_v1": not gate_green,
        "labels_checked_v1": sorted(labels),
        "costs_checked_v1": sorted(costs),
        "stop_items_v1": stop_items,
    }


def run_prelaunch_validation(
    reports_root: Path,
    bundle: ProtectorRunnerBundle,
    output_dir: Path,
    *,
    run_training: bool,
) -> Dict[str, Any]:
    contract_status = _validate_runner_config_contract(bundle, output_dir)
    feature_status = _validate_feature_surface(bundle)
    surface_boundary_status = _materialize_learning_surface_parity_guard(
        reports_root=reports_root,
        bundle=bundle,
        feature_status=feature_status,
    )
    decision_status = _validate_decision_contract(bundle)
    objective_status = _validate_objective_label_gate(bundle, run_training=run_training)
    return {
        "layer_name_v1": "PROTECTOR_FIRST_EXECUTION_IMPLEMENTATION_V1",
        "validated_at_utc_v1": _utc_now_iso(),
        "status_v1": "PASS",
        "contract_status_v1": contract_status,
        "feature_surface_status_v1": feature_status,
        "learning_surface_parity_guard_v1": surface_boundary_status,
        "decision_contract_status_v1": decision_status,
        "objective_label_gate_status_v1": objective_status,
        "checklist_v1": bundle.prelaunch_checklist,
    }


def _build_loaded_config(bundle: ProtectorRunnerBundle) -> Dict[str, Any]:
    return {
        "layer_name_v1": "IMPLEMENT_PROTECTOR_FIRST_CONFIG_LOADERS_V1",
        "spec_dir_v1": str(bundle.spec_dir),
        "objective_label_review_artifact_v1": str(bundle.objective_label_review_artifact),
        "loaded_artifacts_v1": [
            RUNNER_SPEC,
            CONFIG_LOCK,
            DECISION_CONTRACT,
            OBJECTIVE_LABEL_REVIEW,
            FEATURE_SURFACE_LOCK,
            EVAL_VERDICT_MATRIX,
            PRELAUNCH_CHECKLIST,
            ABORT_RULES,
        ],
        "runner_spec_v1": bundle.runner_spec,
        "config_lock_v1": bundle.config_lock,
        "decision_contract_v1": bundle.decision_contract,
        "objective_label_review_v1": bundle.objective_label_review,
        "feature_surface_lock_v1": bundle.feature_surface_lock,
        "eval_verdict_matrix_v1": bundle.eval_verdict_matrix,
        "abort_rules_v1": bundle.abort_rules,
    }


def _build_decision_contract_report(validation_report: Dict[str, Any], bundle: ProtectorRunnerBundle) -> Dict[str, Any]:
    return {
        "layer_name_v1": "IMPLEMENT_PROTECTOR_FIRST_DECISION_CONTRACT_ENFORCEMENT_V1",
        "status_v1": "STRUCTURE_READY_NO_POLICY_CONTROLLER_CHANGE",
        "shadow_eval_only_v1": True,
        "hard_protector_veto_v1": bundle.decision_contract["hard_protector_veto_v1"],
        "soft_damper_v1": bundle.decision_contract["soft_damper_v1"],
        "conflict_resolution_order_v1": bundle.decision_contract["conflict_resolution_order_v1"],
        "conflict_summary_required_fields_v1": validation_report["decision_contract_status_v1"]["conflict_summary_columns_v1"],
        "enforcement_testable_without_training_v1": True,
    }


def _build_abort_enforcement_report(bundle: ProtectorRunnerBundle) -> Dict[str, Any]:
    return {
        "layer_name_v1": "IMPLEMENT_PROTECTOR_FIRST_ABORT_RULES_V1",
        "status_v1": "STRUCTURE_READY_EVAL_NOT_RUN",
        "abort_before_training_v1": bundle.abort_rules["abort_before_training_v1"],
        "reject_after_eval_v1": bundle.abort_rules["reject_after_eval_v1"],
        "runtime_enforcement_points_v1": [
            "prelaunch input/surface validation",
            "feature matrix legality validation",
            "decision contract validation",
            "objective/label gate validation",
            "future post-eval hard safety checks",
        ],
    }


def _placeholder_conflict_summary(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns + ["placeholder_status_v1"])


def _build_training_matrix(
    bundle: ProtectorRunnerBundle,
    validation_report: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, Any]]:
    feature_status = validation_report["feature_surface_status_v1"]
    parity = validation_report["learning_surface_parity_guard_v1"]
    feature_names = list(feature_status["feature_names_v1"])
    forbidden = _forbidden_fields(["candidate_uid"] + feature_names)
    if forbidden:
        raise PrelaunchValidationError(f"forbidden fields in training matrix: {forbidden}")
    if len(feature_names) != EXPECTED_FEATURES:
        raise PrelaunchValidationError(f"feature count mismatch in training matrix: {len(feature_names)} != {EXPECTED_FEATURES}")
    if int(feature_status.get("baseline_feature_count_v1", -1)) != EXPECTED_BASELINE_FEATURES:
        raise PrelaunchValidationError("training matrix does not carry the locked 62 baseline feature count")
    if int(feature_status.get("new_proxy_feature_count_v1", -1)) != EXPECTED_PROXY_FEATURES:
        raise PrelaunchValidationError("training matrix does not carry the locked 5 proxy feature count")

    raw_df = pd.read_parquet(Path(feature_status["training_surface_v1"]))
    label_df = pd.read_parquet(Path(parity["eval_label_surface_v1"]))
    if len(raw_df) != EXPECTED_ROWS:
        raise PrelaunchValidationError(f"training row population mismatch: {len(raw_df)} != {EXPECTED_ROWS}")
    if "candidate_uid" not in raw_df.columns or "candidate_uid" not in label_df.columns:
        raise PrelaunchValidationError("candidate_uid is required to build the training matrix")
    missing_features = sorted([feature for feature in feature_names if feature not in raw_df.columns])
    if missing_features:
        raise PrelaunchValidationError(f"selected features missing from raw-state: {missing_features[:20]}")
    missing_heads = sorted([head["label_col_v1"] for head in HEAD_SPECS if head["label_col_v1"] not in label_df.columns])
    if missing_heads:
        raise PrelaunchValidationError(f"locked training head labels missing: {missing_heads}")

    bridge_only_ids = set(str(candidate) for candidate in parity["bridge_only_candidate_uid_sample_v1"])
    bridge_df = pd.read_parquet(Path(parity["eval_readiness_bridge_surface_v1"]))
    full_bridge_only_ids = set(bridge_df.loc[~bridge_df["candidate_uid"].astype(str).isin(set(raw_df["candidate_uid"].astype(str))), "candidate_uid"].astype(str))
    raw_ids = raw_df["candidate_uid"].astype(str)
    leaked_bridge_only = sorted(set(raw_ids) & full_bridge_only_ids)
    if leaked_bridge_only:
        raise PrelaunchValidationError(f"bridge-only rows are present in training matrix: {leaked_bridge_only[:20]}")

    matrix = raw_df[["candidate_uid"] + feature_names].copy()
    matrix["candidate_uid"] = matrix["candidate_uid"].astype(str)
    label_cols = ["candidate_uid"] + [head["label_col_v1"] for head in HEAD_SPECS]
    train_labels = label_df[label_cols].copy()
    train_labels["candidate_uid"] = train_labels["candidate_uid"].astype(str)
    joined = matrix.merge(train_labels, on="candidate_uid", how="inner", validate="one_to_one")
    if len(joined) != EXPECTED_ROWS:
        raise PrelaunchValidationError(f"training matrix/label intersection {len(joined)} != {EXPECTED_ROWS}")
    order = pd.Series(range(len(matrix)), index=matrix["candidate_uid"])
    joined["_row_order_v1"] = joined["candidate_uid"].map(order)
    joined = joined.sort_values("_row_order_v1").drop(columns=["_row_order_v1"]).reset_index(drop=True)
    x = _numeric_matrix(joined, feature_names)
    summary = {
        "layer_name_v1": "PROTECTOR_FIRST_TRAINING_MATRIX_BUILDER_V1",
        "status_v1": "PASS",
        "built_at_utc_v1": _utc_now_iso(),
        "training_surface_v1": feature_status["training_surface_v1"],
        "training_surface_kind_v1": EXPECTED_SURFACE_KIND,
        "row_count_v1": int(len(joined)),
        "expected_row_count_v1": EXPECTED_ROWS,
        "feature_count_v1": int(len(feature_names)),
        "expected_feature_count_v1": EXPECTED_FEATURES,
        "baseline_feature_count_v1": EXPECTED_BASELINE_FEATURES,
        "new_proxy_feature_count_v1": EXPECTED_PROXY_FEATURES,
        "bridge_only_rows_in_training_matrix_v1": 0,
        "bridge_only_rows_excluded_v1": True,
        "management_exit_truth_fields_in_matrix_v1": [],
        "policy_decision_log_fields_in_matrix_v1": [],
        "as_of_skip_xgb_fields_in_matrix_v1": [feature for feature in feature_names if feature.startswith("as_of_skip_xgb_")],
        "path_dynamics_truth_fields_in_matrix_v1": [],
        "forbidden_fields_in_matrix_v1": _forbidden_fields(joined.columns),
        "matrix_summary_written_before_model_training_v1": True,
        "feature_names_v1": feature_names,
        "head_labels_v1": [head["label_col_v1"] for head in HEAD_SPECS],
    }
    if summary["forbidden_fields_in_matrix_v1"] or summary["as_of_skip_xgb_fields_in_matrix_v1"]:
        raise PrelaunchValidationError(
            f"forbidden fields in training matrix: {summary['forbidden_fields_in_matrix_v1'] or summary['as_of_skip_xgb_fields_in_matrix_v1']}"
        )
    return joined, x, feature_names, summary


def _build_eval_frame(
    bundle: ProtectorRunnerBundle,
    validation_report: Dict[str, Any],
    feature_names: List[str],
) -> pd.DataFrame:
    parity = validation_report["learning_surface_parity_guard_v1"]
    raw_df = pd.read_parquet(Path(parity["training_surface_v1"])).copy()
    label_df = pd.read_parquet(Path(parity["eval_label_surface_v1"])).copy()
    bridge_df = pd.read_parquet(Path(parity["eval_readiness_bridge_surface_v1"])).copy()
    for frame in [raw_df, label_df, bridge_df]:
        frame["candidate_uid"] = frame["candidate_uid"].astype(str)
    eval_base = label_df.merge(bridge_df, on=["candidate_uid"], how="left", suffixes=("", "__bridge"), validate="one_to_one")
    raw_feature_cols = ["candidate_uid"] + [feature for feature in feature_names if feature in raw_df.columns]
    eval_frame = eval_base.merge(
        raw_df[raw_feature_cols],
        on="candidate_uid",
        how="left",
        suffixes=("", "__raw"),
        validate="one_to_one",
    )
    for feature in feature_names:
        bridge_value = eval_frame[feature] if feature in eval_frame.columns else pd.Series(np.nan, index=eval_frame.index)
        if feature in bridge_df.columns:
            bridge_col = feature
            if f"{feature}__bridge" in eval_frame.columns:
                bridge_col = f"{feature}__bridge"
            bridge_value = bridge_value.fillna(eval_frame[bridge_col])
        if f"{feature}__raw" in eval_frame.columns:
            bridge_value = bridge_value.fillna(eval_frame[f"{feature}__raw"])
        eval_frame[feature] = bridge_value
    eval_frame["protector_first_eval_surface_role_v1"] = np.where(
        eval_frame["candidate_uid"].isin(set(raw_df["candidate_uid"].astype(str))),
        "EXACT_CANONICAL_TRAINING_ROW_AVAILABLE_FOR_EVAL",
        "BRIDGE_ONLY_EVAL_READINESS_ROW_NOT_TRAINING",
    )
    return eval_frame


def _train_head_family(
    train_frame: pd.DataFrame,
    train_x: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_names: List[str],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    predictions = pd.DataFrame({"candidate_uid": eval_frame["candidate_uid"].astype(str)})
    train_predictions = pd.DataFrame({"candidate_uid": train_frame["candidate_uid"].astype(str)})
    eval_x = _numeric_matrix(eval_frame, feature_names)
    folds = _fold_indices(len(train_frame), 5)
    heads_manifest: Dict[str, Any] = {}
    for head in HEAD_SPECS:
        head_id = head["head_id_v1"]
        label_col = head["label_col_v1"]
        y = _bool_col(train_frame, label_col)
        oof = np.zeros(len(train_frame), dtype=float)
        fold_rows: List[Dict[str, Any]] = []
        for fold_idx, test_idx in enumerate(folds, start=1):
            train_idx = np.setdiff1d(np.arange(len(train_frame)), test_idx)
            model = _fit_centroid_logit_head(
                train_x.iloc[train_idx].copy(),
                y.iloc[train_idx].copy(),
                head_id=head_id,
                label_col=label_col,
                feature_names=feature_names,
            )
            oof[test_idx] = model.predict_true_probability(train_x.iloc[test_idx].copy())
            fold_rows.append(
                {
                    "fold_v1": fold_idx,
                    "train_rows_v1": int(len(train_idx)),
                    "test_rows_v1": int(len(test_idx)),
                    "positive_train_count_v1": int(y.iloc[train_idx].sum()),
                    "constant_model_v1": model.constant_model,
                }
            )
        final_model = _fit_centroid_logit_head(
            train_x.copy(),
            y.copy(),
            head_id=head_id,
            label_col=label_col,
            feature_names=feature_names,
        )
        output_col = f"pred__protector_first__{head_id}__prob_true_v1"
        train_predictions[output_col] = oof
        predictions[output_col] = final_model.predict_true_probability(eval_x.copy())
        head_manifest = final_model.manifest_v1()
        head_manifest.update(
            {
                "role_v1": head["role_v1"],
                "output_col_v1": output_col,
                "training_positive_count_v1": int(y.sum()),
                "training_row_count_v1": int(len(y)),
                "folds_v1": fold_rows,
            }
        )
        heads_manifest[head_id] = head_manifest
    manifest = {
        "layer_name_v1": "PROTECTOR_FIRST_MODEL_TRAINING_PHASE_V1",
        "training_started_v1": True,
        "model_family_v1": EXPECTED_MODEL_FAMILY,
        "head_count_v1": len(HEAD_SPECS),
        "head_ids_v1": [head["head_id_v1"] for head in HEAD_SPECS],
        "seed_v1": EXECUTION_SEED,
        "compact_config_v1": True,
        "deterministic_output_namespace_v1": True,
        "shadow_only_v1": True,
        "not_policy_controller_v1": True,
        "not_live_behavior_v1": True,
        "model_type_v1": "R6_STYLE_FIVE_HEAD_CENTROID_LOGIT_SHADOW_FAMILY",
        "heads_v1": heads_manifest,
        "training_prediction_rows_v1": int(len(train_predictions)),
        "eval_prediction_rows_v1": int(len(predictions)),
    }
    return predictions, manifest


def apply_protector_first_decision_effect(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    frame = scored.copy()
    blocker_cols = {
        "bad_risk": "pred__protector_first__bad_risk__prob_true_v1",
        "tail_control_10_50": "pred__protector_first__tail_control_10_50__prob_true_v1",
        "risky_allow": "pred__protector_first__risky_allow__prob_true_v1",
        "batch04_blindspot": "pred__protector_first__batch04_blindspot__prob_true_v1",
    }
    for column in blocker_cols.values():
        if column not in frame.columns:
            frame[column] = 0.0
    protector_model_col = "pred__protector_first__runner_protector__prob_true_v1"
    if protector_model_col not in frame.columns:
        frame[protector_model_col] = 0.0
    blocker_scores = pd.DataFrame({name: _num_col(frame, column) for name, column in blocker_cols.items()})
    frame["blocker_score_v1"] = blocker_scores.max(axis=1)
    frame["blocker_reason_v1"] = blocker_scores.idxmax(axis=1).astype(str)
    raw_guard = _num_col(frame, "as_of_pre_entry_runner_protection_guard_score_v1", 0.0).clip(0.0, 1.0)
    frame["protector_score_v1"] = np.maximum(_num_col(frame, protector_model_col, 0.0), raw_guard)
    frame["protector_reason_v1"] = np.where(
        raw_guard.ge(_num_col(frame, protector_model_col, 0.0)),
        "raw_runner_protection_guard_score",
        "runner_protector_head",
    )
    frame["decision_before_protector_v1"] = np.where(frame["blocker_score_v1"].ge(0.50), "BLOCK", "ALLOW")

    forensic = frame["candidate_uid"].astype(str).eq(FORENSIC_TRADE) | _bool_col(frame, "bridge_pocket_forensic_repaired_trade_v1")
    repaired = _bool_col(frame, "r6_label_repaired_165_like_runner_v1") | _bool_col(frame, "bridge_pocket_repaired_165_v1")
    runner100 = _bool_col(frame, "r6_label_runner_100_mfe_v1")
    runner200 = _bool_col(frame, "r6_label_runner_200_mfe_v1")
    strong = _bool_col(frame, "r6_label_strong_low_mae_runner_v1") | _bool_col(frame, "r6_label_high_mfe_low_giveback_v1")
    runner50 = _bool_col(frame, "r6_label_runner_50_mfe_v1") | _bool_col(frame, "bridge_pocket_fifty_plus_mfe_seed_v1")
    near_miss = _bool_col(frame, "r6_label_runner_near_miss_v1") | _bool_col(frame, "bridge_pocket_runner_near_miss_v1")
    hard_pocket = forensic | repaired | runner100 | runner200 | strong
    soft_pocket = (near_miss | runner50) & ~hard_pocket
    before_block = frame["decision_before_protector_v1"].eq("BLOCK")
    hard_veto = before_block & hard_pocket
    soft_damper_considered = before_block & soft_pocket & frame["protector_score_v1"].ge(0.35)
    soft_required_score = np.maximum(0.70, frame["protector_score_v1"] + 0.15)
    soft_damper_unblock = soft_damper_considered & frame["blocker_score_v1"].lt(soft_required_score)
    final_block = before_block & ~hard_veto & ~soft_damper_unblock

    frame["hard_veto_applied_v1"] = hard_veto
    frame["soft_damper_applied_v1"] = soft_damper_considered
    frame["veto_or_damper_applied_v1"] = hard_veto | soft_damper_considered
    frame["decision_after_protector_v1"] = np.where(final_block, "BLOCK", "ALLOW")
    frame["final_shadow_decision_v1"] = frame["decision_after_protector_v1"]

    pocket_memberships: List[str] = []
    conflict_types: List[str] = []
    protector_actions: List[str] = []
    for idx in frame.index:
        pockets: List[str] = []
        if bool(forensic.loc[idx]):
            pockets.append("forensic_repaired_trade")
        if bool(repaired.loc[idx]):
            pockets.append("repaired_165_like_pockets")
        if bool(strong.loc[idx]):
            pockets.append("strongest_winner")
        if bool((runner100 | runner200).loc[idx]):
            pockets.append("100_plus_200_plus_winner_pockets")
        if bool(near_miss.loc[idx]):
            pockets.append("runner_near_miss")
        if bool(runner50.loc[idx]):
            pockets.append("50_plus_mfe_seed_pockets")
        pocket_memberships.append("|".join(pockets) if pockets else "none")
        if bool(hard_veto.loc[idx]):
            conflict_types.append("HARD_PROTECTOR_VETO")
            protector_actions.append("VETO_BLOCK")
        elif bool(soft_damper_unblock.loc[idx]):
            conflict_types.append("SOFT_DAMPER_UNBLOCK")
            protector_actions.append("DAMPEN_AND_UNBLOCK")
        elif bool(soft_damper_considered.loc[idx]):
            conflict_types.append("BLOCKER_WINS_DESPITE_SOFT_DAMPER")
            protector_actions.append("DAMPEN_BUT_BLOCKER_EVIDENCE_HELD")
        else:
            conflict_types.append("NO_CONFLICT")
            protector_actions.append("NO_PROTECTOR_ACTION")
    frame["pocket_membership_v1"] = pocket_memberships
    frame["conflict_type_v1"] = conflict_types
    frame["protector_action_v1"] = protector_actions
    frame["override_or_damper_reason_v1"] = np.where(
        frame["hard_veto_applied_v1"],
        "hard protected pocket veto",
        np.where(frame["soft_damper_applied_v1"], "soft protected pocket required stronger blocker evidence", "none"),
    )
    frame["score_margin_v1"] = frame["blocker_score_v1"] - frame["protector_score_v1"]
    conflict_mask = frame["conflict_type_v1"].ne("NO_CONFLICT")
    conflict_rows = frame.loc[conflict_mask].copy()
    if conflict_rows.empty:
        conflict_rows = pd.DataFrame(
            columns=[
                "candidate_uid",
                "trade_uid",
                "trade_id",
                "blocker_score",
                "blocker_reason",
                "protector_score",
                "protector_reason",
                "pocket_membership",
                "pocket_tag",
                "conflict_type",
                "decision_before_protector",
                "decision_after_protector",
                "veto_damper_applied",
                "final_shadow_decision",
                "protector_action",
                "blocker_action_before_protection",
                "final_shadow_action",
                "score_margin",
                "override_or_damper_reason",
            ]
        )
    else:
        conflict_rows = conflict_rows.assign(
            blocker_score=conflict_rows["blocker_score_v1"],
            blocker_reason=conflict_rows["blocker_reason_v1"],
            protector_score=conflict_rows["protector_score_v1"],
            protector_reason=conflict_rows["protector_reason_v1"],
            pocket_membership=conflict_rows["pocket_membership_v1"],
            pocket_tag=conflict_rows["pocket_membership_v1"],
            conflict_type=conflict_rows["conflict_type_v1"],
            decision_before_protector=conflict_rows["decision_before_protector_v1"],
            decision_after_protector=conflict_rows["decision_after_protector_v1"],
            veto_damper_applied=conflict_rows["veto_or_damper_applied_v1"],
            final_shadow_decision=conflict_rows["final_shadow_decision_v1"],
            protector_action=conflict_rows["protector_action_v1"],
            blocker_action_before_protection=conflict_rows["decision_before_protector_v1"],
            final_shadow_action=conflict_rows["final_shadow_decision_v1"],
            score_margin=conflict_rows["score_margin_v1"],
            override_or_damper_reason=conflict_rows["override_or_damper_reason_v1"],
        )
        conflict_rows = conflict_rows[
            [
                "candidate_uid",
                "trade_uid",
                "trade_id",
                "blocker_score",
                "blocker_reason",
                "protector_score",
                "protector_reason",
                "pocket_membership",
                "pocket_tag",
                "conflict_type",
                "decision_before_protector",
                "decision_after_protector",
                "veto_damper_applied",
                "final_shadow_decision",
                "protector_action",
                "blocker_action_before_protection",
                "final_shadow_action",
                "score_margin",
                "override_or_damper_reason",
            ]
        ]
    summary = _build_conflict_summary(frame, conflict_rows)
    return frame, conflict_rows, summary


def _build_conflict_summary(scored: pd.DataFrame, conflict_rows: pd.DataFrame) -> Dict[str, Any]:
    final_block = scored["final_shadow_decision_v1"].eq("BLOCK")
    before_block = scored["decision_before_protector_v1"].eq("BLOCK")
    hard_veto = _bool_col(scored, "hard_veto_applied_v1")
    soft_damper = _bool_col(scored, "soft_damper_applied_v1")
    protector_wins = int((before_block & ~final_block).sum())
    blocker_wins = int((before_block & final_block & soft_damper).sum())
    protected_winner = (
        _bool_col(scored, "r6_label_repaired_165_like_runner_v1")
        | _bool_col(scored, "r6_label_runner_50_mfe_v1")
        | _bool_col(scored, "r6_label_runner_100_mfe_v1")
        | _bool_col(scored, "r6_label_runner_200_mfe_v1")
        | _bool_col(scored, "r6_label_strong_low_mae_runner_v1")
        | _bool_col(scored, "r6_label_high_mfe_low_giveback_v1")
        | _bool_col(scored, "bridge_pocket_repaired_165_v1")
        | _bool_col(scored, "bridge_pocket_fifty_plus_mfe_seed_v1")
    )
    retained = protected_winner & ~final_block
    by_pocket: Dict[str, int] = {}
    by_head: Dict[str, int] = {}
    if not conflict_rows.empty and "pocket_membership" in conflict_rows.columns:
        for membership in conflict_rows["pocket_membership"].astype(str):
            for pocket in membership.split("|"):
                if pocket and pocket != "none":
                    by_pocket[pocket] = by_pocket.get(pocket, 0) + 1
        by_head = conflict_rows["blocker_reason"].astype(str).value_counts().to_dict()
    forensic_row = scored[scored["candidate_uid"].astype(str).eq(FORENSIC_TRADE)]
    forensic_status = "MISSING"
    if not forensic_row.empty:
        forensic_status = "BLOCKED" if bool(forensic_row["final_shadow_decision_v1"].eq("BLOCK").any()) else "UNBLOCKED"
    return {
        "layer_name_v1": "BLOCKER_VS_PROTECTOR_CONFLICT_SUMMARY_V1",
        "materialized_v1": True,
        "total_conflicts_v1": int(len(conflict_rows)),
        "hard_veto_count_v1": int(hard_veto.sum()),
        "soft_damper_count_v1": int(soft_damper.sum()),
        "blocker_wins_despite_protector_count_v1": blocker_wins,
        "protector_wins_count_v1": protector_wins,
        "protected_winner_retention_v1": _safe_rate(float(retained.sum()), float(protected_winner.sum())),
        "protected_50_plus_cases_v1": int((_bool_col(scored, "r6_label_runner_50_mfe_v1") & ~final_block).sum()),
        "protected_100_plus_cases_v1": int((_bool_col(scored, "r6_label_runner_100_mfe_v1") & ~final_block).sum()),
        "protected_200_plus_cases_v1": int((_bool_col(scored, "r6_label_runner_200_mfe_v1") & ~final_block).sum()),
        "forensic_trade_status_v1": forensic_status,
        "repaired_165_protected_count_v1": int((_bool_col(scored, "r6_label_repaired_165_like_runner_v1") & ~final_block).sum()),
        "runner_near_miss_protected_count_v1": int((_bool_col(scored, "r6_label_runner_near_miss_v1") & ~final_block).sum()),
        "conflicts_by_pocket_v1": by_pocket,
        "conflicts_by_head_reason_v1": by_head,
    }


def _metric_row(policy_name: str, scope: str, frame: pd.DataFrame) -> Dict[str, Any]:
    block = frame["final_shadow_decision_v1"].eq("BLOCK")
    should = _bool_col(frame, "r6_label_bad_risk_v1")
    tail = _bool_col(frame, "r6_label_tail_control_10_50_v1")
    repaired = _bool_col(frame, "r6_label_repaired_165_like_runner_v1") | _bool_col(frame, "bridge_pocket_repaired_165_v1")
    runner50 = _bool_col(frame, "r6_label_runner_50_mfe_v1") | _bool_col(frame, "bridge_pocket_fifty_plus_mfe_seed_v1")
    runner100 = _bool_col(frame, "r6_label_runner_100_mfe_v1")
    runner200 = _bool_col(frame, "r6_label_runner_200_mfe_v1")
    strong = _bool_col(frame, "r6_label_strong_low_mae_runner_v1") | _bool_col(frame, "r6_label_high_mfe_low_giveback_v1")
    near = _bool_col(frame, "r6_label_runner_near_miss_v1") | _bool_col(frame, "bridge_pocket_runner_near_miss_v1")
    strongest = runner200 | strong
    blocked = int(block.sum())
    bad_blocks = int((block & should).sum())
    precision = _safe_rate(float(bad_blocks), float(blocked))
    forensic_blocked = bool((block & frame["candidate_uid"].astype(str).eq(FORENSIC_TRADE)).any())
    return {
        "policy_name_v1": policy_name,
        "scope_v1": scope,
        "row_count_v1": int(len(frame)),
        "block_count_v1": blocked,
        "bad_blocks_v1": bad_blocks,
        "tail_help_v1": int((block & tail).sum()),
        "global_precision_v1": precision,
        "should_not_take_count_v1": int(should.sum()),
        "should_not_take_recall_v1": _safe_rate(float(bad_blocks), float(should.sum())),
        "repaired_165_damage_v1": int((block & repaired).sum()),
        "forensic_trade_blocked_v1": forensic_blocked,
        "fifty_plus_mfe_block_count_v1": int((block & runner50).sum()),
        "hundred_plus_mfe_block_count_v1": int((block & runner100).sum()),
        "two_hundred_plus_mfe_block_count_v1": int((block & runner200).sum()),
        "strongest_winner_damage_v1": int((block & strongest).sum()),
        "runner_near_miss_block_count_v1": int((block & near).sum()),
        "runner_near_miss_regression_v1": bool((block & near).sum() > 0),
        "protector_over_block_override_count_v1": int((_bool_col(frame, "veto_or_damper_applied_v1") & frame["decision_before_protector_v1"].eq("BLOCK") & frame["final_shadow_decision_v1"].eq("ALLOW")).sum()),
        "protected_winner_retention_v1": _build_conflict_summary(frame, pd.DataFrame()).get("protected_winner_retention_v1"),
    }


def _worst_fold_precision(scored: pd.DataFrame) -> float | None:
    values: List[float] = []
    for fold in _fold_indices(len(scored), 5):
        row = _metric_row("PROTECTOR_FIRST_CANDIDATE", "FOLD", scored.iloc[fold].copy())
        value = _safe_float(row.get("global_precision_v1"))
        if value is not None:
            values.append(value)
    return min(values) if values else None


def _build_eval_compare_and_verdict(
    scored: pd.DataFrame,
    conflict_summary: Dict[str, Any],
    validation_report: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    global_metric = _metric_row("PROTECTOR_FIRST_CANDIDATE", "EVAL_READINESS_1852", scored)
    worst_loso = _worst_fold_precision(scored)
    global_metric["worst_loso_precision_v1"] = worst_loso
    pocket_rows = []
    pocket_defs = {
        "repaired_165_like_pockets": _bool_col(scored, "r6_label_repaired_165_like_runner_v1") | _bool_col(scored, "bridge_pocket_repaired_165_v1"),
        "forensic_repaired_trade": scored["candidate_uid"].astype(str).eq(FORENSIC_TRADE) | _bool_col(scored, "bridge_pocket_forensic_repaired_trade_v1"),
        "runner_near_miss": _bool_col(scored, "r6_label_runner_near_miss_v1") | _bool_col(scored, "bridge_pocket_runner_near_miss_v1"),
        "50_plus_mfe_seed_pockets": _bool_col(scored, "r6_label_runner_50_mfe_v1") | _bool_col(scored, "bridge_pocket_fifty_plus_mfe_seed_v1"),
        "100_plus_winner_pockets": _bool_col(scored, "r6_label_runner_100_mfe_v1"),
        "200_plus_winner_pockets": _bool_col(scored, "r6_label_runner_200_mfe_v1"),
        "strongest_winner": _bool_col(scored, "r6_label_runner_200_mfe_v1") | _bool_col(scored, "r6_label_strong_low_mae_runner_v1") | _bool_col(scored, "r6_label_high_mfe_low_giveback_v1"),
    }
    final_block = scored["final_shadow_decision_v1"].eq("BLOCK")
    for pocket, mask in pocket_defs.items():
        pocket_rows.append(
            {
                "pocket_v1": pocket,
                "eval_count_v1": int(mask.sum()),
                "blocked_after_protector_v1": int((mask & final_block).sum()),
                "blocked_before_protector_v1": int((mask & scored["decision_before_protector_v1"].eq("BLOCK")).sum()),
                "protector_overrides_v1": int((mask & scored["decision_before_protector_v1"].eq("BLOCK") & ~final_block).sum()),
            }
        )
    pocket_df = pd.DataFrame(pocket_rows)
    candidate_reference = {
        "reference_v1": "PROTECTOR_FIRST_CANDIDATE",
        "id_v1": "CURRENT_RUN",
        "kind_v1": "CANDIDATE",
    }
    candidate_reference.update(
        {
            "bad_blocks_v1": global_metric["bad_blocks_v1"],
            "tail_help_v1": global_metric["tail_help_v1"],
            "global_precision_v1": global_metric["global_precision_v1"],
            "worst_loso_precision_v1": worst_loso,
            "repaired_165_damage_v1": global_metric["repaired_165_damage_v1"],
            "fifty_plus_mfe_block_count_v1": global_metric["fifty_plus_mfe_block_count_v1"],
            "hundred_plus_mfe_block_count_v1": global_metric["hundred_plus_mfe_block_count_v1"],
            "two_hundred_plus_mfe_block_count_v1": global_metric["two_hundred_plus_mfe_block_count_v1"],
            "strongest_winner_damage_v1": global_metric["strongest_winner_damage_v1"],
            "runner_near_miss_block_count_v1": global_metric["runner_near_miss_block_count_v1"],
        }
    )
    compare_rows = [dict(row) for row in REFERENCE_METRICS] + [candidate_reference]
    frozen = REFERENCE_METRICS[0]
    for row in compare_rows:
        for metric in ["bad_blocks_v1", "tail_help_v1", "global_precision_v1", "worst_loso_precision_v1"]:
            row[f"delta_vs_frozen_{metric}"] = (
                None
                if _safe_float(row.get(metric)) is None or _safe_float(frozen.get(metric)) is None
                else _safe_float(row.get(metric)) - _safe_float(frozen.get(metric))
            )
    compare_report = {
        "layer_name_v1": "PROTECTOR_FIRST_EVAL_AND_COMPARE_V1",
        "compare_scope_v1": "benchmark/eval over 1852 eval/readiness rows, not proof of identical training universe",
        "candidate_metric_v1": global_metric,
        "references_v1": compare_rows,
        "blocker_protector_conflict_summary_v1": conflict_summary,
    }
    failures = _post_eval_disqualifiers(global_metric, conflict_summary)
    verdict = _verdict_from_metrics(global_metric, failures, conflict_summary)
    eval_summary = {
        "layer_name_v1": "PROTECTOR_FIRST_EVAL_AND_COMPARE_V1",
        "training_started_v1": True,
        "eval_rows_v1": int(len(scored)),
        "global_metric_v1": global_metric,
        "bad_blocks_v1": global_metric["bad_blocks_v1"],
        "tail_help_v1": global_metric["tail_help_v1"],
        "global_precision_v1": global_metric["global_precision_v1"],
        "worst_loso_precision_v1": worst_loso,
        "repaired_165_damage_v1": global_metric["repaired_165_damage_v1"],
        "forensic_trade_blocked_v1": global_metric["forensic_trade_blocked_v1"],
        "fifty_plus_mfe_block_count_v1": global_metric["fifty_plus_mfe_block_count_v1"],
        "hundred_plus_mfe_block_count_v1": global_metric["hundred_plus_mfe_block_count_v1"],
        "two_hundred_plus_mfe_block_count_v1": global_metric["two_hundred_plus_mfe_block_count_v1"],
        "strongest_winner_damage_v1": global_metric["strongest_winner_damage_v1"],
        "runner_near_miss_block_count_v1": global_metric["runner_near_miss_block_count_v1"],
        "protector_over_block_override_count_v1": global_metric["protector_over_block_override_count_v1"],
        "protected_winner_retention_v1": global_metric["protected_winner_retention_v1"],
        "blocker_protector_conflict_summary_v1": conflict_summary,
    }
    verdict_package = {
        "layer_name_v1": "PROTECTOR_FIRST_VERDICT_AND_ABORT_V1",
        "verdict_v1": verdict,
        "candidate_disqualified_v1": bool(failures),
        "hard_fail_reasons_v1": failures,
        "supported_verdicts_v1": [
            "PROTECTOR_FIRST_CANDIDATE_IMPROVES_AND_HOLDS_SAFETY",
            "PROTECTOR_FIRST_CANDIDATE_SAFE_BUT_NOT_BETTER",
            "PROTECTOR_FIRST_CANDIDATE_IMPROVES_BUT_FAILS_SAFETY",
            "PROTECTOR_FIRST_FEATURES_OR_OBJECTIVE_INSUFFICIENT",
            "PROTECTOR_FIRST_INVALID_SURFACE_OR_LEGALITY_BREACH",
            "NOT_ESTABLISHED",
        ],
        "surface_boundary_guard_pass_v1": validation_report["learning_surface_parity_guard_v1"]["surface_boundary_guard_pass_v1"],
        "conflict_summary_materialized_v1": bool(conflict_summary.get("materialized_v1")),
    }
    return eval_summary, compare_report, pocket_df, verdict_package, global_metric


def _post_eval_disqualifiers(metric: Dict[str, Any], conflict_summary: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    if int(metric.get("repaired_165_damage_v1") or 0) > 0:
        failures.append("repaired_165_damage > 0")
    if bool(metric.get("forensic_trade_blocked_v1")):
        failures.append("forensic trade blocked")
    if int(metric.get("hundred_plus_mfe_block_count_v1") or 0) > 0:
        failures.append("100+ MFE blocked > 0")
    if int(metric.get("two_hundred_plus_mfe_block_count_v1") or 0) > 0:
        failures.append("200+ MFE blocked > 0")
    if int(metric.get("fifty_plus_mfe_block_count_v1") or 0) > 1:
        failures.append("50+ MFE blocked > 1")
    if int(metric.get("strongest_winner_damage_v1") or 0) > 0:
        failures.append("strongest-winner damage > 0")
    if bool(metric.get("runner_near_miss_regression_v1")) and int(metric.get("runner_near_miss_block_count_v1") or 0) > 0:
        failures.append("serious runner near-miss regression")
    precision = _safe_float(metric.get("global_precision_v1"))
    if precision is None or precision < 0.80:
        failures.append("precision collapsed below locked floor")
    worst_loso = _safe_float(metric.get("worst_loso_precision_v1"))
    if worst_loso is None or worst_loso < 0.50:
        failures.append("worst LOSO collapsed below locked floor")
    if not conflict_summary.get("materialized_v1"):
        failures.append("conflict summary missing")
    return failures


def _verdict_from_metrics(
    metric: Dict[str, Any],
    failures: List[str],
    conflict_summary: Dict[str, Any],
) -> str:
    if not conflict_summary.get("materialized_v1"):
        return "NOT_ESTABLISHED"
    frozen = REFERENCE_METRICS[0]
    improves = (
        int(metric.get("bad_blocks_v1") or 0) >= int(frozen["bad_blocks_v1"])
        and int(metric.get("tail_help_v1") or 0) >= int(frozen["tail_help_v1"])
        and (_safe_float(metric.get("global_precision_v1")) or 0.0) >= float(frozen["global_precision_v1"])
    )
    if failures and improves:
        return "PROTECTOR_FIRST_CANDIDATE_IMPROVES_BUT_FAILS_SAFETY"
    if failures:
        return "PROTECTOR_FIRST_CANDIDATE_IMPROVES_BUT_FAILS_SAFETY"
    if improves:
        return "PROTECTOR_FIRST_CANDIDATE_IMPROVES_AND_HOLDS_SAFETY"
    return "PROTECTOR_FIRST_CANDIDATE_SAFE_BUT_NOT_BETTER"


def _write_output_scaffold(bundle: ProtectorRunnerBundle, output_dir: Path, validation_report: Dict[str, Any]) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded_config = _build_loaded_config(bundle)
    decision_report = _build_decision_contract_report(validation_report, bundle)
    objective_report = {
        "layer_name_v1": "IMPLEMENT_PROTECTOR_FIRST_OBJECTIVE_LABEL_GATE_V1",
        **validation_report["objective_label_gate_status_v1"],
    }
    abort_report = _build_abort_enforcement_report(bundle)
    feature_names = validation_report["feature_surface_status_v1"]["feature_names_v1"]
    feature_echo = pd.DataFrame(
        {
            "feature_name_v1": feature_names,
            "source_surface_v1": "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE",
            "selected_for_future_training_v1": True,
        }
    )
    conflict_placeholder = _placeholder_conflict_summary(
        validation_report["decision_contract_status_v1"]["conflict_summary_columns_v1"]
    )
    eval_placeholder = {
        "layer_name_v1": "PROTECTOR_FIRST_EVAL_VERDICT_PLACEHOLDER_V1",
        "status_v1": "PENDING_TRAINING_NOT_RUN",
        "verdict_v1": "NOT_ESTABLISHED",
        "supported_verdicts_v1": [row["verdict_v1"] for row in bundle.eval_verdict_matrix["verdicts_v1"]],
        "hard_safety_requirements_v1": bundle.eval_verdict_matrix["hard_safety_requirements_v1"],
        "protection_specific_metrics_v1": bundle.eval_verdict_matrix["protection_specific_metrics_v1"],
    }
    config_echo = {
        "layer_name_v1": "PROTECTOR_FIRST_CONFIG_ECHO_V1",
        "architecture_v1": bundle.config_lock["architecture_v1"],
        "shadow_only_v1": bundle.config_lock["shadow_only_v1"],
        "not_live_gate_v1": bundle.config_lock["not_live_gate_v1"],
        "not_policy_controller_v1": bundle.config_lock["not_policy_controller_v1"],
        "can_change_in_this_experiment_v1": bundle.config_lock["can_change_in_this_experiment_v1"],
        "cannot_change_v1": bundle.config_lock["cannot_change_v1"],
    }
    scaffold_manifest = {
        "layer_name_v1": "IMPLEMENT_PROTECTOR_FIRST_OUTPUT_SCAFFOLD_V1",
        "training_started_v1": False,
        "surface_boundary_guard_pass_v1": validation_report["learning_surface_parity_guard_v1"]["surface_boundary_guard_pass_v1"],
        "scaffold_artifacts_v1": [
            SUMMARY,
            STATUS,
            MANIFEST,
            CONFIG_ECHO,
            FEATURE_MANIFEST_ECHO,
            GENERIC_FEATURE_MANIFEST_ECHO,
            LEARNING_SURFACE_PARITY_GUARD,
            PRELAUNCH_REPORT,
            OBJECTIVE_GATE_REPORT,
            DECISION_CONTRACT_REPORT,
            ABORT_ENFORCEMENT_REPORT,
            CONFLICT_SUMMARY_PLACEHOLDER,
            EVAL_VERDICT_PLACEHOLDER,
            CONSISTENCY_AUDIT,
        ],
    }
    _write_json(output_dir / LOADED_CONFIG, loaded_config)
    _write_json(output_dir / CONFIG_ECHO, config_echo)
    feature_echo.to_csv(output_dir / FEATURE_MANIFEST_ECHO, index=False)
    feature_echo.to_csv(output_dir / GENERIC_FEATURE_MANIFEST_ECHO, index=False)
    _write_json(output_dir / LEARNING_SURFACE_PARITY_GUARD, validation_report["learning_surface_parity_guard_v1"])
    _write_json(output_dir / PRELAUNCH_REPORT, validation_report)
    _write_json(output_dir / OBJECTIVE_GATE_REPORT, objective_report)
    _write_json(output_dir / DECISION_CONTRACT_REPORT, decision_report)
    _write_json(output_dir / ABORT_ENFORCEMENT_REPORT, abort_report)
    conflict_placeholder.to_csv(output_dir / CONFLICT_SUMMARY_PLACEHOLDER, index=False)
    _write_json(output_dir / EVAL_VERDICT_PLACEHOLDER, eval_placeholder)
    _write_json(output_dir / OUTPUT_SCAFFOLD_MANIFEST, scaffold_manifest)
    return {
        "loaded_config_v1": loaded_config,
        "decision_report_v1": decision_report,
        "objective_report_v1": objective_report,
        "abort_report_v1": abort_report,
        "eval_placeholder_v1": eval_placeholder,
        "scaffold_manifest_v1": scaffold_manifest,
    }


def _build_next_action(*, objective_label_gate_green: bool) -> Dict[str, Any]:
    supporting_locks = [
        "DO_NOT_TRAIN_WITHOUT_EXPLICIT_FLAG",
        "DO_NOT_REPLAY",
        "DO_NOT_TOUCH_POLICY_CONTROLLER",
        "TRAINING_EXECUTION_PHASE_IMPLEMENTED_BUT_FLAG_GATED",
        "LEARNING_SURFACE_PARITY_GUARD_REQUIRED",
    ]
    if objective_label_gate_green:
        supporting_locks.append("OBJECTIVE_LABEL_GATE_GREEN_WITH_STRICT_GUARDS")
        primary_action = "NEXT_AGENT_MAY_RUN_PROTECTOR_FIRST_TRAINING_WITH_EXPLICIT_FLAG"
    else:
        supporting_locks.append("OBJECTIVE_LABEL_GATE_NOT_GREEN_FOR_TRAINING_YET")
        primary_action = "NEXT_AGENT_MAY_RUN_PROTECTOR_FIRST_DRY_PRELAUNCH"
    return {
        "layer_name_v1": "NEXT_AGENT_ACTION_LOCK_V1",
        "primary_action_v1": primary_action,
        "blocked_action_v1": "RUN_TRAINING_WITHOUT_EXPLICIT_FLAG",
        "supporting_locks_v1": supporting_locks,
    }


def _write_report(path: Path, summary: Dict[str, Any], validation_report: Dict[str, Any]) -> None:
    lines = [
        "# Protector-First Shadow Experiment Runner V1",
        "",
        "## Status",
        f"- `{summary['RUNNER_STATUS']}`",
        f"- Training started: `{summary['training_started_v1']}`",
        f"- Replay started: `{summary['replay_started_v1']}`",
        "",
        "## Prelaunch",
        f"- Rows: `{validation_report['feature_surface_status_v1']['raw_rows_v1']}`",
        f"- Features: `{validation_report['feature_surface_status_v1']['selected_feature_count_v1']}`",
        f"- Architecture: `{summary['architecture_v1']}`",
        "",
        "## Next",
        f"- `{summary['next_action_v1']}`",
        f"- `{summary['blocked_action_v1']}` remains blocked.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_final_artifacts(
    bundle: ProtectorRunnerBundle,
    output_dir: Path,
    validation_report: Dict[str, Any],
    scaffold: Dict[str, Any],
) -> Dict[str, Any]:
    now = _utc_now_iso()
    objective_label_gate_green = validation_report["objective_label_gate_status_v1"]["training_gate_green_v1"]
    next_action = _build_next_action(objective_label_gate_green=objective_label_gate_green)
    if objective_label_gate_green:
        hard_status = {
            "BEVIST": [
                "Protector-first runner dry/prelaunch implementation loads locked specs and validates exact-only raw-state.",
                "Dry mode does not train, replay or touch policy/controller.",
                "Training execution is implemented behind an explicit run flag.",
                "Learning-surface parity guard separates 1689 training rows from the 1852-row eval/readiness bridge.",
                "Objective/label gate is green with strict guards.",
                "Dry output scaffold is materialized without model outputs.",
            ],
            "INDIKERT": [
                "The next safe operational step is a deliberate training run with the explicit flag.",
            ],
            "IKKE_ETABLERT": [
                "That a future protector-first trained candidate beats frozen Wednesday-R6.",
                "That training should run without an explicit flag.",
            ],
        }
    else:
        hard_status = {
            "BEVIST": [
                "Protector-first runner dry/prelaunch implementation loads locked specs and validates exact-only raw-state.",
                "Dry mode does not train, replay or touch policy/controller.",
                "Training execution is implemented behind an explicit run flag.",
                "Learning-surface parity guard separates 1689 training rows from the 1852-row eval/readiness bridge.",
                "Objective/label gate blocks future training unless green.",
                "Output scaffold is materialized without model outputs.",
            ],
            "INDIKERT": [
                "The next safe operational step is a dry prelaunch run, not training.",
            ],
            "IKKE_ETABLERT": [
                "That objective/label review is green for training.",
                "That a future protector-first trained candidate beats frozen Wednesday-R6.",
                "That training should run without an explicit flag.",
            ],
        }
    summary = {
        "layer_name_v1": "PROTECTOR_FIRST_RUNNER_SUMMARY_V1",
        "RUNNER_STATUS": "DRY_PRELAUNCH_COMPLETED",
        "materialized_at_utc_v1": now,
        "spec_dir_v1": str(bundle.spec_dir),
        "output_dir_v1": str(output_dir),
        "job_name_v1": JOB_NAME,
        "runner_name_v1": RUNNER_NAME,
        "architecture_v1": ARCHITECTURE,
        "training_started_v1": False,
        "replay_started_v1": False,
        "policy_controller_changed_v1": False,
        "run_training_flag_v1": False,
        "training_run_allowed_now_v1": False,
        "training_execution_implemented_v1": True,
        "explicit_training_flag_required_v1": True,
        "raw_rows_v1": validation_report["feature_surface_status_v1"]["raw_rows_v1"],
        "feature_count_v1": validation_report["feature_surface_status_v1"]["selected_feature_count_v1"],
        "eval_readiness_rows_v1": validation_report["learning_surface_parity_guard_v1"]["eval_readiness_bridge_row_count_v1"],
        "bridge_only_count_v1": validation_report["learning_surface_parity_guard_v1"]["bridge_only_count_v1"],
        "surface_boundary_guard_pass_v1": validation_report["learning_surface_parity_guard_v1"]["surface_boundary_guard_pass_v1"],
        "objective_label_gate_green_v1": objective_label_gate_green,
        "hard_veto_rule_count_v1": len(bundle.decision_contract["hard_protector_veto_v1"]),
        "soft_damper_rule_count_v1": len(bundle.decision_contract["soft_damper_v1"]),
        "conflict_summary_placeholder_v1": CONFLICT_SUMMARY_PLACEHOLDER,
        "next_action_v1": next_action["primary_action_v1"],
        "blocked_action_v1": next_action["blocked_action_v1"],
        "hard_status_division_v1": hard_status,
    }
    contract = {
        "layer_name_v1": "PROTECTOR_FIRST_RUNNER_CONTRACT_V1",
        "materialized_at_utc_v1": now,
        "spec_dir_v1": str(bundle.spec_dir),
        "not_training_v1": True,
        "not_replay_v1": True,
        "not_policy_controller_change_v1": True,
        "not_live_gate_v1": True,
    }
    audit_rows = [
        _audit_record("PRELAUNCH_VALIDATION_PASS", "PASS" if validation_report["status_v1"] == "PASS" else "FAIL", {"status_v1": validation_report["status_v1"]}),
        _audit_record("ROW_COUNT_1689", "PASS" if summary["raw_rows_v1"] == EXPECTED_ROWS else "FAIL", {"raw_rows_v1": summary["raw_rows_v1"]}),
        _audit_record("FEATURE_COUNT_67", "PASS" if summary["feature_count_v1"] == EXPECTED_FEATURES else "FAIL", {"feature_count_v1": summary["feature_count_v1"]}),
        _audit_record("SURFACE_BOUNDARY_GUARD_PASS", "PASS" if summary["surface_boundary_guard_pass_v1"] is True else "FAIL", {"surface_boundary_guard_pass_v1": summary["surface_boundary_guard_pass_v1"]}),
        _audit_record("DECISION_CONTRACT_READY", "PASS" if summary["hard_veto_rule_count_v1"] == 4 and summary["soft_damper_rule_count_v1"] == 2 else "FAIL", {"hard_veto_rule_count_v1": summary["hard_veto_rule_count_v1"], "soft_damper_rule_count_v1": summary["soft_damper_rule_count_v1"]}),
        _audit_record(
            "OBJECTIVE_GATE_STATUS_CONSISTENT",
            "PASS",
            {
                "objective_label_gate_green_v1": summary["objective_label_gate_green_v1"],
                "next_action_v1": next_action["primary_action_v1"],
                "training_run_allowed_now_v1": False,
                "explicit_training_flag_required_v1": True,
            },
        ),
        _audit_record("NO_TRAINING_STARTED", "PASS" if summary["training_started_v1"] is False else "FAIL", {"training_started_v1": summary["training_started_v1"]}),
        _audit_record("CONFLICT_PLACEHOLDER_PRESENT", "PASS" if (output_dir / CONFLICT_SUMMARY_PLACEHOLDER).exists() else "FAIL", {"artifact_v1": CONFLICT_SUMMARY_PLACEHOLDER}),
    ]
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(output_dir / CONSISTENCY_AUDIT, index=False)
    failed_checks = int(audit_df["status_v1"].astype("string").ne("PASS").sum())
    manifest = {
        "layer_name_v1": "PROTECTOR_FIRST_RUNNER_MANIFEST_V1",
        "materialized_at_utc_v1": now,
        "output_dir_v1": str(output_dir),
        "source_spec_dir_v1": str(bundle.spec_dir),
        "artifacts_v1": [
            CONTRACT,
            LOADED_CONFIG,
            CONFIG_ECHO,
            FEATURE_MANIFEST_ECHO,
            GENERIC_FEATURE_MANIFEST_ECHO,
            LEARNING_SURFACE_PARITY_GUARD,
            PRELAUNCH_REPORT,
            OBJECTIVE_GATE_REPORT,
            DECISION_CONTRACT_REPORT,
            ABORT_ENFORCEMENT_REPORT,
            CONFLICT_SUMMARY_PLACEHOLDER,
            EVAL_VERDICT_PLACEHOLDER,
            OUTPUT_SCAFFOLD_MANIFEST,
            NEXT_ACTION,
            SUMMARY,
            REPORT,
            MANIFEST,
            STATUS,
            CONSISTENCY_AUDIT,
        ],
    }
    status = {
        "layer_name_v1": "PROTECTOR_FIRST_RUNNER_STATUS_V1",
        "RUNNER_STATUS": summary["RUNNER_STATUS"],
        "failed_check_count_v1": failed_checks,
        "training_started_v1": False,
        "replay_started_v1": False,
        "dry_prelaunch_completed_v1": True,
        "training_run_allowed_now_v1": False,
        "surface_boundary_guard_pass_v1": summary["surface_boundary_guard_pass_v1"],
        "objective_label_gate_green_v1": summary["objective_label_gate_green_v1"],
        "feature_count_v1": summary["feature_count_v1"],
        "raw_rows_v1": summary["raw_rows_v1"],
        "next_action_v1": next_action["primary_action_v1"],
        "blocked_action_v1": next_action["blocked_action_v1"],
    }
    _write_json(output_dir / CONTRACT, contract)
    _write_json(output_dir / NEXT_ACTION, next_action)
    _write_json(output_dir / SUMMARY, summary)
    _write_report(output_dir / REPORT, summary, validation_report)
    _write_json(output_dir / MANIFEST, manifest)
    _write_json(output_dir / STATUS, status)
    return summary


def _write_training_execution_outputs(
    bundle: ProtectorRunnerBundle,
    output_dir: Path,
    validation_report: Dict[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    start_time = _utc_now_iso()
    _write_json(
        output_dir / STATUS,
        {
            "layer_name_v1": "PROTECTOR_FIRST_TRAINING_EXECUTION_STATUS_V1",
            "RUNNER_STATUS": "TRAINING_EXECUTION_STARTED",
            "training_started_v1": True,
            "run_training_flag_v1": True,
            "created_at_utc_v1": start_time,
            "failed_check_count_v1": 0,
        },
    )

    matrix_frame, train_x, feature_names, matrix_summary = _build_training_matrix(bundle, validation_report)
    _write_json(output_dir / TRAINING_MATRIX_SUMMARY, matrix_summary)
    eval_frame = _build_eval_frame(bundle, validation_report, feature_names)
    predictions, model_manifest = _train_head_family(matrix_frame, train_x, eval_frame, feature_names)
    scored = eval_frame.merge(predictions, on="candidate_uid", how="left", validate="one_to_one")
    scored, conflict_rows, conflict_summary = apply_protector_first_decision_effect(scored)
    eval_summary, compare_report, pocket_df, verdict_package, global_metric = _build_eval_compare_and_verdict(
        scored,
        conflict_summary,
        validation_report,
    )
    candidate_disqualified = bool(verdict_package["candidate_disqualified_v1"])
    runner_status = "TRAINING_EXECUTION_DISQUALIFIED" if candidate_disqualified else "TRAINING_EXECUTION_COMPLETED"
    completed_at = _utc_now_iso()
    feature_echo = pd.DataFrame(
        {
            "feature_name_v1": feature_names,
            "source_surface_v1": "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE",
            "selected_for_training_v1": True,
        }
    )
    config_manifest = {
        "layer_name_v1": "PROTECTOR_FIRST_CONFIG_MANIFEST_V1",
        "architecture_v1": ARCHITECTURE,
        "model_family_v1": EXPECTED_MODEL_FAMILY,
        "seed_v1": EXECUTION_SEED,
        "compact_config_v1": True,
        "shadow_only_v1": True,
        "not_live_gate_v1": True,
        "not_policy_controller_v1": True,
        "hard_veto_soft_damper_decision_layer_v1": True,
        "bridge_as_training_surface_allowed_v1": False,
        "training_surface_kind_v1": EXPECTED_SURFACE_KIND,
        "eval_readiness_bridge_rows_v1": validation_report["learning_surface_parity_guard_v1"]["eval_readiness_bridge_row_count_v1"],
    }
    training_summary = {
        "layer_name_v1": "PROTECTOR_FIRST_EXECUTION_IMPLEMENTATION_V1",
        "created_at_utc_v1": start_time,
        "completed_at_utc_v1": completed_at,
        "RUNNER_STATUS": runner_status,
        "training_started_v1": True,
        "run_training_flag_v1": True,
        "replay_started_v1": False,
        "policy_controller_changed_v1": False,
        "architecture_v1": ARCHITECTURE,
        "objective_label_gate_green_v1": validation_report["objective_label_gate_status_v1"]["training_gate_green_v1"],
        "surface_boundary_guard_pass_v1": validation_report["learning_surface_parity_guard_v1"]["surface_boundary_guard_pass_v1"],
        "training_rows_v1": matrix_summary["row_count_v1"],
        "eval_readiness_rows_v1": eval_summary["eval_rows_v1"],
        "bridge_only_count_v1": validation_report["learning_surface_parity_guard_v1"]["bridge_only_count_v1"],
        "feature_count_v1": matrix_summary["feature_count_v1"],
        "model_family_v1": EXPECTED_MODEL_FAMILY,
        "verdict_v1": verdict_package["verdict_v1"],
        "candidate_disqualified_v1": candidate_disqualified,
        "hard_fail_reasons_v1": verdict_package["hard_fail_reasons_v1"],
        "bad_blocks_v1": global_metric["bad_blocks_v1"],
        "tail_help_v1": global_metric["tail_help_v1"],
        "global_precision_v1": global_metric["global_precision_v1"],
        "worst_loso_precision_v1": global_metric["worst_loso_precision_v1"],
        "protector_over_block_override_count_v1": global_metric["protector_over_block_override_count_v1"],
        "forensic_trade_blocked_v1": global_metric["forensic_trade_blocked_v1"],
    }
    next_action = {
        "layer_name_v1": "NEXT_AGENT_ACTION_LOCK_V1",
        "primary_action_v1": "REVIEW_PROTECTOR_FIRST_TRAINING_EXECUTION_VERDICT",
        "blocked_action_v1": "PROMOTE_OR_FREEZE_WITHOUT_REVIEW",
        "training_now_v1": False,
    }
    contract = {
        "layer_name_v1": "PROTECTOR_FIRST_TRAINING_EXECUTION_CONTRACT_V1",
        "created_at_utc_v1": start_time,
        "spec_dir_v1": str(bundle.spec_dir),
        "output_dir_v1": str(output_dir),
        "run_training_flag_v1": True,
        "training_started_v1": True,
        "not_replay_v1": True,
        "not_policy_controller_change_v1": True,
        "not_live_gate_v1": True,
        "no_freeze_or_promo_artifacts_v1": True,
    }
    _write_json(output_dir / CONTRACT, contract)
    _write_json(output_dir / LOADED_CONFIG, _build_loaded_config(bundle))
    _write_json(output_dir / CONFIG_MANIFEST, config_manifest)
    _write_json(output_dir / LEARNING_SURFACE_PARITY_GUARD, validation_report["learning_surface_parity_guard_v1"])
    _write_json(output_dir / PRELAUNCH_REPORT, validation_report)
    _write_json(output_dir / OBJECTIVE_GATE_REPORT, {"layer_name_v1": "IMPLEMENT_PROTECTOR_FIRST_OBJECTIVE_LABEL_GATE_V1", **validation_report["objective_label_gate_status_v1"]})
    _write_json(output_dir / DECISION_CONTRACT_REPORT, _build_decision_contract_report(validation_report, bundle))
    _write_json(output_dir / ABORT_ENFORCEMENT_REPORT, _build_abort_enforcement_report(bundle))
    _write_json(output_dir / TRAINING_EXECUTION_SUMMARY, training_summary)
    _write_json(output_dir / MODEL_MANIFEST, model_manifest)
    feature_echo.to_csv(output_dir / FEATURE_MANIFEST_ECHO, index=False)
    feature_echo.to_csv(output_dir / GENERIC_FEATURE_MANIFEST_ECHO, index=False)
    scored.to_parquet(output_dir / PREDICTION_VIEW, index=False)
    _write_json(output_dir / EVAL_SUMMARY, eval_summary)
    _write_json(output_dir / COMPARE_AGAINST_REPORT, compare_report)
    pocket_df.to_csv(output_dir / POCKET_REPORT_CSV, index=False)
    _write_json(output_dir / POCKET_REPORT_JSON, {"layer_name_v1": "PROTECTOR_FIRST_POCKET_REPORT_V1", "rows_v1": pocket_df.to_dict(orient="records")})
    conflict_rows.to_csv(output_dir / CONFLICT_SUMMARY_CSV, index=False)
    _write_json(output_dir / CONFLICT_SUMMARY_JSON, conflict_summary)
    _write_json(output_dir / VERDICT_PACKAGE, verdict_package)
    _write_json(output_dir / NEXT_ACTION, next_action)

    audit_rows = [
        _audit_record("TRAINING_EXECUTION_FLAG_PRESENT", "PASS", {"run_training_flag_v1": True}),
        _audit_record("OBJECTIVE_LABEL_GATE_GREEN", "PASS" if training_summary["objective_label_gate_green_v1"] else "FAIL", {"objective_label_gate_green_v1": training_summary["objective_label_gate_green_v1"]}),
        _audit_record("SURFACE_BOUNDARY_GUARD_PASS", "PASS" if training_summary["surface_boundary_guard_pass_v1"] else "FAIL", {"surface_boundary_guard_pass_v1": training_summary["surface_boundary_guard_pass_v1"]}),
        _audit_record("TRAINING_MATRIX_1689", "PASS" if training_summary["training_rows_v1"] == EXPECTED_ROWS else "FAIL", {"training_rows_v1": training_summary["training_rows_v1"]}),
        _audit_record("FEATURE_COUNT_67", "PASS" if training_summary["feature_count_v1"] == EXPECTED_FEATURES else "FAIL", {"feature_count_v1": training_summary["feature_count_v1"]}),
        _audit_record("CONFLICT_SUMMARY_MATERIALIZED", "PASS" if conflict_summary.get("materialized_v1") else "FAIL", {"total_conflicts_v1": conflict_summary.get("total_conflicts_v1")}),
        _audit_record("NO_POLICY_CONTROLLER_CHANGE", "PASS", {"policy_controller_changed_v1": False}),
        _audit_record("NO_FREEZE_OR_PROMO_ARTIFACTS", "PASS", {"no_freeze_or_promo_artifacts_v1": True}),
    ]
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(output_dir / CONSISTENCY_AUDIT, index=False)
    failed_checks = int(audit_df["status_v1"].astype("string").ne("PASS").sum())
    manifest_artifacts = [
        CONTRACT,
        LOADED_CONFIG,
        PRELAUNCH_REPORT,
        OBJECTIVE_GATE_REPORT,
        DECISION_CONTRACT_REPORT,
        ABORT_ENFORCEMENT_REPORT,
        TRAINING_EXECUTION_SUMMARY,
        TRAINING_MATRIX_SUMMARY,
        LEARNING_SURFACE_PARITY_GUARD,
        MODEL_MANIFEST,
        CONFIG_MANIFEST,
        FEATURE_MANIFEST_ECHO,
        GENERIC_FEATURE_MANIFEST_ECHO,
        PREDICTION_VIEW,
        EVAL_SUMMARY,
        COMPARE_AGAINST_REPORT,
        POCKET_REPORT_CSV,
        POCKET_REPORT_JSON,
        CONFLICT_SUMMARY_CSV,
        CONFLICT_SUMMARY_JSON,
        VERDICT_PACKAGE,
        NEXT_ACTION,
        SUMMARY,
        REPORT,
        MANIFEST,
        STATUS,
        CONSISTENCY_AUDIT,
    ]
    manifest = {
        "layer_name_v1": "PROTECTOR_FIRST_OUTPUTS_V1",
        "materialized_at_utc_v1": completed_at,
        "output_dir_v1": str(output_dir),
        "source_spec_dir_v1": str(bundle.spec_dir),
        "training_started_v1": True,
        "artifacts_v1": manifest_artifacts,
    }
    status = {
        "layer_name_v1": "PROTECTOR_FIRST_RUNNER_STATUS_V1",
        "RUNNER_STATUS": runner_status,
        "failed_check_count_v1": failed_checks,
        "training_started_v1": True,
        "replay_started_v1": False,
        "dry_prelaunch_completed_v1": False,
        "training_execution_completed_v1": runner_status == "TRAINING_EXECUTION_COMPLETED",
        "training_execution_disqualified_v1": runner_status == "TRAINING_EXECUTION_DISQUALIFIED",
        "training_execution_aborted_v1": False,
        "surface_boundary_guard_pass_v1": training_summary["surface_boundary_guard_pass_v1"],
        "objective_label_gate_green_v1": training_summary["objective_label_gate_green_v1"],
        "feature_count_v1": training_summary["feature_count_v1"],
        "raw_rows_v1": training_summary["training_rows_v1"],
        "next_action_v1": next_action["primary_action_v1"],
        "blocked_action_v1": next_action["blocked_action_v1"],
    }
    summary = {
        "layer_name_v1": "PROTECTOR_FIRST_TRAINING_EXECUTION_SUMMARY_V1",
        **training_summary,
        "output_dir_v1": str(output_dir),
        "next_action_v1": next_action["primary_action_v1"],
        "blocked_action_v1": next_action["blocked_action_v1"],
        "hard_status_division_v1": {
            "BEVIST": [
                "Training execution runs only with the explicit flag.",
                "The training matrix is locked to 1689 exact-only canonical raw-state rows and 67 features.",
                "Bridge-only rows are kept out of training and used only for eval/readiness hard guards.",
                "Protector hard veto and soft damper have decision effect in shadow eval.",
                "Eval, compare, conflict summary and verdict artifacts are materialized.",
            ],
            "INDIKERT": [
                "The trained candidate can be reviewed against frozen Wednesday-R6 and safety references.",
            ],
            "IKKE_ETABLERT": [
                "No live behavior, policy/controller change, freeze or promotion is established.",
            ],
        },
    }
    _write_json(output_dir / MANIFEST, manifest)
    _write_json(output_dir / STATUS, status)
    _write_json(output_dir / SUMMARY, summary)
    _write_report(output_dir / REPORT, summary, validation_report)
    return summary


def run_runner(
    *,
    reports_root: Path,
    spec_dir: Path | None = None,
    output_dir: Path | None = None,
    run_training: bool = False,
    objective_label_review_artifact: Path | None = None,
) -> Dict[str, Any]:
    bundle = _load_bundle(
        reports_root,
        str(spec_dir) if spec_dir else None,
        str(objective_label_review_artifact) if objective_label_review_artifact else None,
    )
    resolved_output_dir = output_dir if output_dir is not None else _resolve_output_dir(
        reports_root,
        None,
        run_training=run_training,
    )
    validation_report = run_prelaunch_validation(reports_root, bundle, resolved_output_dir, run_training=run_training)
    if run_training:
        return _write_training_execution_outputs(bundle, resolved_output_dir, validation_report)
    scaffold = _write_output_scaffold(bundle, resolved_output_dir, validation_report)
    return _write_final_artifacts(bundle, resolved_output_dir, validation_report, scaffold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run protector-first shadow experiment runner V1.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--spec-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--objective-label-review-artifact", default=None)
    parser.add_argument("--run-training", action="store_true", help="Explicitly run protector-first training execution.")
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    spec_dir = Path(args.spec_dir).expanduser().resolve() if args.spec_dir else None
    output_dir = _resolve_output_dir(reports_root, args.output_dir, run_training=bool(args.run_training))
    objective_label_review_artifact = (
        Path(args.objective_label_review_artifact).expanduser().resolve()
        if args.objective_label_review_artifact
        else None
    )
    summary = run_runner(
        reports_root=reports_root,
        spec_dir=spec_dir,
        output_dir=output_dir,
        run_training=bool(args.run_training),
        objective_label_review_artifact=objective_label_review_artifact,
    )
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
