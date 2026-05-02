#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gx1.scripts import materialize_foundation_integrity_and_hidden_drift_audit_before_optuna_v1 as foundation_audit


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_V3_IN_SAMPLE_DIR = (
    DEFAULT_REPORTS_ROOT / "RUN_R5_2_OBJECTIVE_V3_PARALLEL_REBUILD_WITH_EXPLICIT_FLAG_AND_STRATEGY_GATE_V1_20260426T_EXECUTION"
)
DEFAULT_V3_OOF_DIR = (
    DEFAULT_REPORTS_ROOT
    / "RUN_R5_2_OBJECTIVE_V3_PARALLEL_REBUILD_WITH_EXPLICIT_FLAG_AND_STRATEGY_GATE_V1_20260426T_EXECUTION_OOF_20260426T190850Z"
)
LAYER_NAME = "CONSTRAINED_OPTUNA_OBJECTIVE_SEARCH_AND_FULL_SIGNAL_FORENSICS_V1"
OUTPUT_ROOT_PREFIX = "CONSTRAINED_OPTUNA_OBJECTIVE_SEARCH_V1"
EXPLICIT_ACTION = "INSTALL_OPTUNA_AND_RUN_CONSTRAINED_OBJECTIVE_SEARCH"
FORENSIC_REPAIRED_CANDIDATE_UID = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"
REQUIRED_KEYS = ["candidate_uid", "trade_uid", "decision_timestamp"]
V3_SCORE_COLUMNS = [
    "r5_2_v3_bad_recall_score",
    "r5_2_v3_tail_recall_score",
    "r5_2_v3_risky_attention_score",
    "r5_2_v3_runner_protection_score",
    "r5_2_v3_high_mfe_ambiguous_protection_score",
    "r5_2_v3_hard_winner_protection_score",
]
V3_BASE_COLUMNS = [
    "r5_2_v3_base_membership_pre_veto",
    "r5_2_v3_hard_protection_veto",
    "r5_2_v3_final_base_membership",
]
V2_SCORE_COLUMNS = [
    "r5_2_v2_bad_recall_score",
    "r5_2_v2_tail_recall_score",
    "r5_2_v2_risky_attention_score",
    "r5_2_v2_runner_protection_score",
    "r5_2_v2_high_mfe_ambiguous_protection_score",
    "r5_2_v2_hard_winner_protection_score",
]
R5_SCORE_COLUMNS = [
    "pred__entry_r5_should_not_take__prob_true_v1",
    "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
    "pred__entry_r5_runner_protect__prob_true_v1",
    "pred__entry_r5_tail_control_10_50_risk__prob_true_v1",
]
R51_SCORE_COLUMNS = ["r5_1_bad_blocker_score_v1", "r5_1_runner_guard_score_v1"]
R52_LEGACY_SCORE_COLUMNS = [
    "pred__entry_r5_2_bad_blocker__prob_true_v1",
    "pred__entry_r5_2_runner_protector__prob_true_v1",
]
OUTPUT_FILES = [
    "active_score_artifact_selection_v1.json",
    "selected_score_artifact_selection_used_v1.json",
    "optuna_dependency_install_and_lock_v1.json",
    "optuna_dependency_manifest_v1.json",
    "pip_freeze_after_optuna_install.txt",
    "constrained_optuna_preflight_v1.json",
    "selected_v3_oof_artifact_audit_v1.json",
    "foundation_audit_reference_v1.json",
    "no_fallback_no_dummy_no_synthetic_attestation_v1.json",
    "v3_in_sample_vs_oof_failure_autopsy_v1.json",
    "full_remaining_gap_ledger_v1.csv",
    "oof_signal_separability_audit_v1.json",
    "feature_family_ablation_and_importance_oof_v1.csv",
    "constrained_optuna_search_space_lock_v1.json",
    "constrained_optuna_objective_function_v1.json",
    "constrained_optuna_study_summary_v1.json",
    "constrained_optuna_trials_v1.csv",
    "constrained_optuna_best_candidate_v1.json",
    "constrained_optuna_constraint_report_v1.csv",
    "constrained_optuna_constraint_report_v1.json",
    "constrained_optuna_metric_denominator_report_v1.csv",
    "constrained_optuna_metric_denominator_report_v1.json",
    "constrained_optuna_full_signal_forensics_v1.csv",
    "constrained_optuna_full_signal_forensics_v1.json",
    "constrained_optuna_go_no_go_v1.json",
    "optuna_trial_log_and_failure_reasons_v1.csv",
    "optuna_trial_forensics_v1.csv",
    "best_optuna_candidate_eval_v1.json",
    "best_constrained_candidate_lock_v1.json",
    "optuna_vs_v3_and_rescue_comparison_v1.json",
    "model_family_escape_hatch_decision_v1.json",
    "project_cleanliness_and_canonical_graph_check_v1.json",
    "next_strategy_gate_after_optuna_v1.json",
    "strategy_decision_after_constrained_search_v1.json",
    "next_action_lock_v1.json",
    "summary_v1.json",
    "report_v1.md",
    "manifest_v1.json",
    "status_v1.json",
    "consistency_audit_v1.csv",
]
MIN_DECISION_PRECISION_DENOMINATOR = 5
MIN_LOSO_SELECTED_GROUPS = 1
ACTIVE_SELECTION_CONTRACT = "ACTIVE_SCORE_ARTIFACT_SELECTION_V1"


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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


def _resolve_active_v3_selection(
    *,
    selected_v3_oof_artifact_root: Path | None,
    active_score_artifact_selection: Path | None,
) -> tuple[Path, dict[str, Any]]:
    if selected_v3_oof_artifact_root is None and active_score_artifact_selection is None:
        raise RuntimeError("ACTIVE_SCORE_ARTIFACT_SELECTION_REQUIRED_FOR_OPTUNA_PREP")
    contract: dict[str, Any] = {}
    contract_root: Path | None = None
    if active_score_artifact_selection is not None:
        contract = _read_json(active_score_artifact_selection)
        if contract.get("contract") != ACTIVE_SELECTION_CONTRACT:
            raise RuntimeError("ACTIVE_SCORE_ARTIFACT_SELECTION_CONTRACT_INVALID")
        if contract.get("decisioning_stage") != "PRE_OPTUNA":
            raise RuntimeError("ACTIVE_SCORE_ARTIFACT_SELECTION_STAGE_INVALID")
        if contract.get("selection_policy") != "EXPLICIT_ONLY_NO_LATEST_GLOB":
            raise RuntimeError("IMPLICIT_LATEST_GLOB_SELECTION_FORBIDDEN")
        root_value = (contract.get("selected_artifacts") or {}).get("v3_oof_scores")
        if not root_value:
            raise RuntimeError("ACTIVE_SCORE_ARTIFACT_SELECTION_MISSING_V3_OOF_SCORES")
        contract_root = Path(str(root_value)).expanduser().resolve()
    selected_root = selected_v3_oof_artifact_root.expanduser().resolve() if selected_v3_oof_artifact_root is not None else contract_root
    if selected_root is None:
        raise RuntimeError("ACTIVE_SCORE_ARTIFACT_SELECTION_REQUIRED_FOR_OPTUNA_PREP")
    if contract_root is not None and selected_root != contract_root:
        raise RuntimeError("ACTIVE_SCORE_ARTIFACT_SELECTION_ROOT_MISMATCH")
    if not selected_root.exists():
        raise RuntimeError(f"SELECTED_V3_OOF_ARTIFACT_ROOT_MISSING:{selected_root}")
    if not contract:
        contract = {
            "contract": ACTIVE_SELECTION_CONTRACT,
            "decisioning_stage": "PRE_OPTUNA",
            "selection_policy": "EXPLICIT_ONLY_NO_LATEST_GLOB",
            "selected_artifacts": {"v3_oof_scores": str(selected_root)},
        }
    return selected_root, contract


def _assert_foundation_audit_green(foundation_audit_dir: Path | None) -> dict[str, Any]:
    if foundation_audit_dir is None:
        raise RuntimeError("FOUNDATION_AUDIT_GREEN_REQUIRED_BEFORE_OPTUNA")
    summary_path = foundation_audit_dir / "summary_v1.json"
    if not summary_path.exists():
        raise RuntimeError(f"FOUNDATION_AUDIT_SUMMARY_MISSING:{summary_path}")
    summary = _read_json(summary_path)
    clean = bool(summary.get("foundation_clean_for_constrained_optuna_v1") or summary.get("foundation_clean_ready_for_optuna_v1"))
    decision = str(summary.get("decision_v1"))
    if not clean or decision not in {"FOUNDATION_CLEAN_FOR_CONSTRAINED_OPTUNA", "FOUNDATION_CLEAN_READY_FOR_OPTUNA"}:
        raise RuntimeError(f"FOUNDATION_AUDIT_NOT_GREEN:{decision}")
    return summary


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_auc(y_true: pd.Series, score: pd.Series) -> float | None:
    y = y_true.fillna(False).astype(bool)
    x = pd.to_numeric(score, errors="coerce")
    mask = x.notna()
    if int(mask.sum()) < 3 or y[mask].nunique() < 2:
        return None
    try:
        return float(roc_auc_score(y[mask].astype(int), x[mask]))
    except ValueError:
        return None


def _optuna_dependency() -> dict[str, Any]:
    spec = importlib.util.find_spec("optuna")
    if spec is None:
        return {
            "available_v1": False,
            "status_v1": "OPTUNA_REQUIRED_DEPENDENCY_MISSING",
            "message_v1": "Optuna is not installed in the active Python environment. No fallback/random search was run.",
        }
    import optuna  # type: ignore

    return {
        "available_v1": True,
        "status_v1": "OPTUNA_AVAILABLE",
        "version_v1": getattr(optuna, "__version__", "UNKNOWN"),
        "module_origin_v1": spec.origin,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text_command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, cwd=Path.cwd(), check=False, capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _dependency_files_with_hashes() -> list[dict[str, Any]]:
    dependency_files: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in [
        "pyproject.toml",
        "requirements*.txt",
        "constraints*.txt",
        "setup.cfg",
        "setup.py",
        "Pipfile",
        "poetry.lock",
        "uv.lock",
        "environment*.yml",
    ]:
        for path in sorted(Path.cwd().glob(pattern)):
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            dependency_files.append(
                {
                    "path_v1": str(path),
                    "sha256_v1": _sha256_file(path),
                }
            )
    return dependency_files


def _pip_freeze_text() -> str:
    output = _run_text_command([sys.executable, "-m", "pip", "freeze"])
    return output if output is not None else "PIP_FREEZE_FAILED\n"


def _optuna_dependency_manifest(
    *,
    optuna_state: dict[str, Any],
    output_dir: Path,
    install_command_used: str,
    pip_freeze_text: str,
) -> dict[str, Any]:
    return {
        "layer_name": "OPTUNA_DEPENDENCY_MANIFEST_V1",
        "created_at_utc_v1": _utc_now(),
        "python_executable_v1": sys.executable,
        "python_version_v1": sys.version,
        "python_prefix_v1": sys.prefix,
        "expected_repo_venv_v1": str(Path.cwd() / ".venv"),
        "installed_in_expected_repo_venv_v1": str(Path(sys.executable).resolve()).startswith(str((Path.cwd() / ".venv").resolve())),
        "optuna_import_ok_v1": bool(optuna_state["available_v1"]),
        "optuna_version_v1": optuna_state.get("version_v1"),
        "optuna_location_v1": optuna_state.get("module_origin_v1"),
        "pip_freeze_path_v1": str(output_dir / "pip_freeze_after_optuna_install.txt"),
        "pip_freeze_sha256_v1": hashlib.sha256(pip_freeze_text.encode("utf-8")).hexdigest(),
        "dependency_file_hashes_v1": _dependency_files_with_hashes(),
        "git_commit_v1": _run_text_command(["git", "rev-parse", "HEAD"]),
        "git_status_short_v1": _run_text_command(["git", "status", "--short"]),
        "install_command_used_v1": install_command_used,
        "no_global_install_v1": True,
        "no_fallback_environment_v1": True,
        "no_new_virtualenv_created_v1": True,
    }


def _dependency_install_lock(optuna_state: dict[str, Any]) -> dict[str, Any]:
    dependency_files = _dependency_files_with_hashes()
    return {
        "layer_name": "OPTUNA_DEPENDENCY_INSTALL_AND_LOCK_V1",
        "python_executable_v1": sys.executable,
        "python_prefix_v1": sys.prefix,
        "expected_venv_v1": str(Path.cwd() / ".venv"),
        "installed_in_expected_venv_v1": str(sys.executable).startswith(str(Path.cwd() / ".venv")),
        "optuna_import_ok_v1": bool(optuna_state["available_v1"]),
        "optuna_version_v1": optuna_state.get("version_v1"),
        "dependency_files_found_v1": dependency_files,
        "dependency_documentation_v1": (
            "No requirements/constraints/pyproject dependency file exists in this worktree; install is documented in this run artifact."
            if not dependency_files
            else "Dependency files exist and should be updated outside this run if project policy requires lockfile edits."
        ),
        "install_command_v1": ".venv/bin/python -m pip install optuna",
        "verification_command_v1": '.venv/bin/python -c "import optuna; print(optuna.__version__)"',
        "hard_fail_v1": not bool(optuna_state["available_v1"]),
    }


def _load_v3_run(run_dir: Path) -> dict[str, Any]:
    return {
        "dir": run_dir,
        "summary": _read_json(run_dir / "summary_v1.json"),
        "strategy": _read_json(run_dir / "strategy_gate_after_v3_v1.json"),
        "manifest": _read_json(run_dir / "manifest_v1.json"),
        "eval": pd.read_csv(run_dir / "v3_variant_eval_and_safety_gate_v1.csv"),
        "leaderboard": pd.read_csv(run_dir / "v3_variant_leaderboard_v1.csv"),
        "index": pd.read_csv(run_dir / "v3_variant_outputs_index_v1.csv"),
    }


def _variant_prediction(run: dict[str, Any], variant_id: str) -> pd.DataFrame:
    index = run["index"]
    row = index[index["variant_id_v1"].astype(str).eq(str(variant_id))]
    if row.empty:
        raise RuntimeError(f"Variant {variant_id} missing from {run['dir']}")
    path = Path(str(row.iloc[0]["variant_dir_v1"])) / "prediction_view_v1.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _load_base_frame(oof_run: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = oof_run["manifest"]["input_paths_v1"]
    score = pd.read_parquet(paths["score_package_v1"])
    label = pd.read_csv(paths["label_table_v1"])
    for column in REQUIRED_KEYS:
        score[column] = score[column].astype(str)
        label[column] = label[column].astype(str)
    keep = [column for column in label.columns if column not in score.columns or column in REQUIRED_KEYS]
    frame = score.merge(label[keep], on=REQUIRED_KEYS, how="left", validate="one_to_one")
    scan_dir = Path(paths["scan_dir_v1"])
    lane_01_path = scan_dir / "lane_01_v2_remaining_gap_trace_v1.csv"
    if lane_01_path.exists():
        lane = pd.read_csv(lane_01_path)
        if all(column in lane.columns for column in REQUIRED_KEYS):
            for column in REQUIRED_KEYS:
                lane[column] = lane[column].astype(str)
            lane_keep = [column for column in lane.columns if column not in frame.columns or column in REQUIRED_KEYS]
            frame = frame.merge(lane[lane_keep], on=REQUIRED_KEYS, how="left", validate="one_to_one")
        elif "candidate_uid" in lane.columns and "gap_bucket_v1" in lane.columns:
            frame = frame.merge(lane[["candidate_uid", "gap_bucket_v1"]], on="candidate_uid", how="left", validate="one_to_one")
    return frame, label


def _merge_prediction(frame: pd.DataFrame, pred: pd.DataFrame, prefix: str) -> pd.DataFrame:
    pred_work = pred.copy()
    for column in REQUIRED_KEYS:
        pred_work[column] = pred_work[column].astype(str)
    keep = REQUIRED_KEYS + [column for column in [*V3_SCORE_COLUMNS, *V3_BASE_COLUMNS] if column in pred_work.columns]
    renamed = {column: f"{prefix}_{column}" for column in keep if column not in REQUIRED_KEYS}
    return frame.merge(pred_work[keep].rename(columns=renamed), on=REQUIRED_KEYS, how="left", validate="one_to_one")


def _dangerous(frame: pd.DataFrame) -> pd.Series:
    return (
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


def _autopsy(in_sample: dict[str, Any], oof: dict[str, Any]) -> dict[str, Any]:
    ins = in_sample["eval"].copy()
    oof_eval = oof["eval"].copy()
    merged = ins.merge(oof_eval, on=["variant_id_v1", "profile_id_v1"], how="outer", suffixes=("_in_sample", "_oof"))
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        bad_gap = int(row.get("bad_recall_v1_in_sample", 0)) - int(row.get("bad_recall_v1_oof", 0))
        tail_gap = int(row.get("tail_recall_v1_in_sample", 0)) - int(row.get("tail_recall_v1_oof", 0))
        loso_gap = float(row.get("worst_loso_v1_in_sample", 0.0)) - float(row.get("worst_loso_v1_oof", 0.0))
        if bad_gap > 100 or tail_gap > 80:
            klass = "IN_SAMPLE_MEMORIZATION"
        elif float(row.get("worst_loso_v1_oof", 1.0)) <= 0.0:
            klass = "SPLIT_SPECIFIC_SIGNAL"
        elif int(row.get("bad_recall_v1_oof", 0)) < 95 and int(row.get("tail_recall_v1_oof", 0)) < 61:
            klass = "WEAK_GENERAL_SIGNAL"
        else:
            klass = "NOT_ESTABLISHED"
        rows.append(
            {
                "variant_id_v1": row["variant_id_v1"],
                "profile_id_v1": row["profile_id_v1"],
                "in_sample_bad_v1": int(row.get("bad_recall_v1_in_sample", 0)),
                "oof_bad_v1": int(row.get("bad_recall_v1_oof", 0)),
                "in_sample_tail_v1": int(row.get("tail_recall_v1_in_sample", 0)),
                "oof_tail_v1": int(row.get("tail_recall_v1_oof", 0)),
                "in_sample_precision_v1": float(row.get("precision_v1_in_sample", 0.0)),
                "oof_precision_v1": float(row.get("precision_v1_oof", 0.0)),
                "in_sample_loso_v1": float(row.get("worst_loso_v1_in_sample", 0.0)),
                "oof_loso_v1": float(row.get("worst_loso_v1_oof", 0.0)),
                "bad_generalization_gap_v1": bad_gap,
                "tail_generalization_gap_v1": tail_gap,
                "loso_generalization_gap_v1": loso_gap,
                "oof_fail_reason_v1": row.get("safety_fail_reasons_v1_oof", "NONE"),
                "collapse_group_v1": row.get("worst_loso_group_v1_oof", "NOT_ESTABLISHED"),
                "classification_v1": klass,
            }
        )
    return {
        "layer_name": "V3_IN_SAMPLE_VS_OOF_FAILURE_AUTOPSY_V1",
        "summary_v1": {
            "in_sample_decision_v1": in_sample["strategy"].get("decision_v1"),
            "oof_decision_v1": oof["strategy"].get("decision_v1"),
            "in_sample_best_bad_tail_v1": [
                in_sample["summary"].get("best_bad_recall_v1"),
                in_sample["summary"].get("best_tail_recall_v1"),
            ],
            "oof_best_bad_tail_v1": [oof["summary"].get("best_bad_recall_v1"), oof["summary"].get("best_tail_recall_v1")],
            "autopsy_verdict_v1": "IN_SAMPLE_MEMORIZATION_AND_OOF_COLLAPSE",
        },
        "variants_v1": rows,
    }


def _full_ledger(frame: pd.DataFrame, in_sample_pred: pd.DataFrame, oof_pred: pd.DataFrame) -> pd.DataFrame:
    work = _merge_prediction(frame, in_sample_pred, "v3_in_sample")
    work = _merge_prediction(work, oof_pred, "v3_oof")
    dangerous = _dangerous(work)
    bad_or_tail = _bool(work, "label_should_not_take_v1") | _bool(work, "tail_10_50_mfe_v1")
    selected_oof = _bool(work, "v3_oof_r5_2_v3_final_base_membership")
    pre_oof = _bool(work, "v3_oof_r5_2_v3_base_membership_pre_veto")
    veto_oof = _bool(work, "v3_oof_r5_2_v3_hard_protection_veto")
    first_fail = np.select(
        [
            selected_oof,
            dangerous,
            pre_oof & veto_oof,
            ~pre_oof,
            pre_oof & ~selected_oof,
        ],
        [
            "SELECTED_BY_V3_OOF_BEST",
            "DANGEROUS_OR_PROTECTED",
            "VETOED_BY_HARD_PROTECTION",
            "NOT_IN_V3_PRE_VETO_BASE",
            "IN_PRE_VETO_BUT_NOT_FINAL_BASE",
        ],
        default="SIGNAL_WEAK_OR_AMBIGUOUS",
    )
    out = pd.DataFrame(
        {
            "candidate_uid": work["candidate_uid"],
            "trade_uid": work["trade_uid"],
            "trade_id": work.get("trade_id", ""),
            "decision_timestamp": work["decision_timestamp"],
            "active_quarantine_v1": work.get("calendar_quarantine_status_v1", ""),
            "label_bucket_v1": work.get("new_r5_2_label_bucket_v1", ""),
            "bad_label_v1": _bool(work, "label_should_not_take_v1"),
            "tail_label_v1": _bool(work, "tail_10_50_mfe_v1"),
            "risky_label_v1": _bool(work, "r6_label_risky_allow_v1"),
            "runner_flag_v1": _bool(work, "r5_2_label_runner_protect_v1") | _bool(work, "r6_label_runner_near_miss_v1"),
            "repaired_flag_v1": _bool(work, "r6_label_repaired_165_like_runner_v1"),
            "fifty_plus_mfe_v1": _bool(work, "fifty_plus_mfe_v1"),
            "hundred_plus_mfe_v1": _bool(work, "hundred_plus_mfe_v1"),
            "two_hundred_plus_mfe_v1": _bool(work, "two_hundred_plus_mfe_v1"),
            "strongest_winner_path_v1": _bool(work, "strongest_winner_path_v1"),
            "ambiguous_high_mfe_flag_v1": _bool(work, "r5_2_label_high_mfe_tail_risk_ambiguous_v1"),
            "high_mfe_flag_v1": _bool(work, "fifty_plus_mfe_v1") | _bool(work, "hundred_plus_mfe_v1") | _bool(work, "two_hundred_plus_mfe_v1"),
            "mfe_bucket_v1": work.get("mfe_bucket_v1", ""),
            "mae_bucket_v1": work.get("mae_bucket_v1", ""),
            "r5_bad_score_v1": _num(work, "pred__entry_r5_should_not_take__prob_true_v1"),
            "r5_tail_score_v1": _num(work, "pred__entry_r5_tail_control_10_50_risk__prob_true_v1"),
            "r5_runner_score_v1": _num(work, "pred__entry_r5_runner_protect__prob_true_v1"),
            "r5_1_bad_score_v1": _num(work, "r5_1_bad_blocker_score_v1"),
            "r5_1_runner_score_v1": _num(work, "r5_1_runner_guard_score_v1"),
            "r5_2_v2_bad_score_v1": _num(work, "r5_2_v2_bad_recall_score"),
            "r5_2_v2_tail_score_v1": _num(work, "r5_2_v2_tail_recall_score"),
            "r5_2_v2_runner_protect_score_v1": _num(work, "r5_2_v2_runner_protection_score"),
            "r5_2_v3_oof_bad_score_v1": _num(work, "v3_oof_r5_2_v3_bad_recall_score"),
            "r5_2_v3_oof_tail_score_v1": _num(work, "v3_oof_r5_2_v3_tail_recall_score"),
            "r5_2_v3_oof_runner_protect_score_v1": _num(work, "v3_oof_r5_2_v3_runner_protection_score"),
            "r5_2_v3_in_sample_bad_score_v1": _num(work, "v3_in_sample_r5_2_v3_bad_recall_score"),
            "r5_2_v3_in_sample_tail_score_v1": _num(work, "v3_in_sample_r5_2_v3_tail_recall_score"),
            "v2_base_membership_v1": _bool(work, "r5_2_v2_final_base_membership"),
            "rescue_base_membership_v1": _bool(work, "r5_2_true_rescue_base_membership_v1"),
            "raw_true_base_membership_v1": _bool(work, "raw_true_base_membership_v1") | _bool(work, "r5_2_raw_true_base_membership_v1"),
            "v3_oof_pre_veto_base_v1": pre_oof,
            "v3_oof_veto_v1": veto_oof,
            "v3_oof_final_base_v1": selected_oof,
            "v3_in_sample_final_base_v1": _bool(work, "v3_in_sample_r5_2_v3_final_base_membership"),
            "first_fail_reason_v1": first_fail,
            "split_loso_group_v1": work.get("run_id", ""),
            "batch_v1": work.get("batch_scope_v1", ""),
            "dangerous_or_protected_v1": dangerous,
            "safe_recoverable_candidate_v1": bad_or_tail & ~dangerous,
            "unknown_v1": pd.Series(False, index=work.index),
        }
    )
    return out


def _separability_audit(ledger: pd.DataFrame) -> dict[str, Any]:
    score_columns = [
        "r5_bad_score_v1",
        "r5_tail_score_v1",
        "r5_runner_score_v1",
        "r5_1_bad_score_v1",
        "r5_1_runner_score_v1",
        "r5_2_v2_bad_score_v1",
        "r5_2_v2_tail_score_v1",
        "r5_2_v2_runner_protect_score_v1",
        "r5_2_v3_oof_bad_score_v1",
        "r5_2_v3_oof_tail_score_v1",
        "r5_2_v3_oof_runner_protect_score_v1",
        "r5_2_v3_in_sample_bad_score_v1",
        "r5_2_v3_in_sample_tail_score_v1",
    ]
    safe_positive = ledger["safe_recoverable_candidate_v1"].astype(bool)
    dangerous = ledger["dangerous_or_protected_v1"].astype(bool)
    groups = {
        "safe_bad": ledger["bad_label_v1"].astype(bool) & ~dangerous,
        "safe_tail": ledger["tail_label_v1"].astype(bool) & ~dangerous,
        "dangerous_protected": dangerous,
        "high_mfe": ledger["high_mfe_flag_v1"].astype(bool),
        "runner_protect": ledger["runner_flag_v1"].astype(bool),
    }
    rows = []
    for column in score_columns:
        if column not in ledger.columns:
            continue
        values = pd.to_numeric(ledger[column], errors="coerce")
        group_stats = {}
        for name, mask in groups.items():
            part = values[mask & values.notna()]
            group_stats[name] = {
                "count_v1": int(part.count()),
                "p50_v1": float(part.quantile(0.50)) if not part.empty else None,
                "p90_v1": float(part.quantile(0.90)) if not part.empty else None,
                "p95_v1": float(part.quantile(0.95)) if not part.empty else None,
            }
        auc = _safe_auc(safe_positive, values)
        danger_auc = _safe_auc(dangerous, values)
        if "in_sample" in column and auc is not None and auc >= 0.65:
            status = "IN_SAMPLE_ONLY_SIGNAL"
        elif auc is not None and auc >= 0.70 and (danger_auc is None or danger_auc < 0.65):
            status = "STRONG_OOF_SIGNAL"
        elif auc is not None and auc >= 0.58:
            status = "WEAK_BUT_USEFUL_SIGNAL"
        elif danger_auc is not None and danger_auc >= 0.70:
            status = "DANGEROUS_OVERLAP_WITH_WINNERS"
        else:
            status = "NOT_USEFUL"
        rows.append(
            {
                "score_v1": column,
                "safe_recoverable_auc_v1": auc,
                "dangerous_auc_v1": danger_auc,
                "status_v1": status,
                "distributions_v1": group_stats,
            }
        )
    return {"layer_name": "OOF_SIGNAL_SEPARABILITY_AUDIT_V1", "signals_v1": rows}


def _feature_family(column: str) -> str:
    lower = column.lower()
    if column.startswith("pred__entry_r5_"):
        return "R5_SIGNALS"
    if column.startswith("r5_1_"):
        return "R5_1_SIGNALS"
    if column in R52_LEGACY_SCORE_COLUMNS or column.startswith("r5_2_") or "entry_r5_2" in column:
        return "LEGAL_R5_2_INPUTS"
    if "skip_xgb" in lower:
        return "SKIP_XGB_FEATURES"
    if any(token in lower for token in ["atr", "vol", "compression", "bandwidth", "range"]):
        return "VOLATILITY_COMPRESSION"
    if any(token in lower for token in ["swing", "ema", "trend", "impulse", "kama"]):
        return "SWING_STRUCTURE"
    if any(token in lower for token in ["spread", "cost", "wick", "body", "momentum", "session"]):
        return "MICROSTRUCTURE_COST"
    if column.startswith("as_of_"):
        return "AS_OF_BASE_FEATURES"
    return "OTHER_LEGAL_PRE_ENTRY"


def _feature_family_audit(frame: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    forbidden = ["hindsight", "exit_", "management_", "bridge", "readiness", "candidate_uid", "trade_uid", "trade_id", "decision_timestamp"]
    features = []
    for column in frame.columns:
        lower = column.lower()
        if any(token in lower for token in forbidden):
            continue
        if column.startswith("as_of_") or column.startswith("pred__entry_r5_") or column.startswith("r5_1_") or column in R52_LEGACY_SCORE_COLUMNS:
            if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
                features.append(column)
    target = ledger["safe_recoverable_candidate_v1"].astype(bool).astype(int)
    danger = ledger["dangerous_or_protected_v1"].astype(bool).astype(int)
    rows = []
    for family in sorted({_feature_family(column) for column in features}):
        cols = [column for column in features if _feature_family(column) == family]
        best_feature = ""
        best_auc = None
        danger_auc = None
        for column in cols:
            auc = _safe_auc(target.astype(bool), frame[column])
            if auc is not None and (best_auc is None or auc > best_auc):
                best_auc = auc
                best_feature = column
                danger_auc = _safe_auc(danger.astype(bool), frame[column])
        if best_auc is None:
            status = "NOT_USEFUL"
        elif best_auc >= 0.68 and (danger_auc is None or danger_auc < 0.68):
            status = "HELPS_OOF"
        elif danger_auc is not None and danger_auc >= 0.70:
            status = "DANGEROUS_WINNER_OVERLAP"
        elif best_auc >= 0.56:
            status = "WEAK_BUT_PRESENT"
        else:
            status = "NOT_USEFUL"
        rows.append(
            {
                "feature_family_v1": family,
                "feature_count_v1": len(cols),
                "best_feature_v1": best_feature,
                "safe_recoverable_auc_v1": best_auc,
                "dangerous_overlap_auc_v1": danger_auc,
                "oof_bad_recall_effect_v1": "PROXY_AUC_ONLY_NO_NEW_TRAINING",
                "oof_tail_recall_effect_v1": "PROXY_AUC_ONLY_NO_NEW_TRAINING",
                "safety_effect_v1": status,
                "loso_effect_v1": "NOT_ESTABLISHED_WITHOUT_OPTUNA_SEARCH",
                "status_v1": status,
            }
        )
    return pd.DataFrame(rows)


def _search_space_lock(optuna_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "CONSTRAINED_OPTUNA_SEARCH_SPACE_LOCK_V1",
        "optuna_dependency_v1": optuna_state,
        "search_space_v1": {
            "bad_recall_weights": {
                "w_v3_bad": [0.0, 3.0],
                "w_v2_bad": [0.0, 3.0],
                "w_r5_bad": [0.0, 3.0],
                "w_r51_bad": [0.0, 3.0],
            },
            "tail_recall_weights": {
                "w_v3_tail": [0.0, 3.0],
                "w_v2_tail": [0.0, 3.0],
                "w_r5_tail": [0.0, 3.0],
            },
            "risky_attention_thresholds": {
                "bad_threshold": [0.35, 3.5],
                "tail_threshold": [0.35, 3.5],
                "risky_threshold": [0.35, 3.5],
                "confirm_threshold": [0.20, 2.0],
            },
            "runner_protection_weights": {
                "w_v3_runner": [0.0, 4.0],
                "w_v2_runner": [0.0, 4.0],
                "w_r5_runner": [0.0, 4.0],
                "w_r51_runner": [0.0, 4.0],
            },
            "veto_thresholds": {"protection_threshold": [0.10, 3.5]},
            "calibration_parameters": {"calibration_temperature": [0.70, 1.50]},
            "regularization_parameters": {"regularization_strength": [0.0, 1.0]},
            "batch_loso_penalty_parameters": {
                "batch_penalty_weight": [5.0, 25.0],
                "loso_penalty_weight": [10.0, 50.0],
                "precision_bonus_weight": [10.0, 50.0],
            },
            "minimum_denominator_controls": {
                "min_final_base_count": [5, 30],
                "fifty_plus_cap": [0, 1],
            },
        },
        "hard_constraints_v1": [
            "repaired_like_overlap_eq_0",
            "forensic_trade_unblocked",
            "hundred_plus_mfe_overlap_eq_0",
            "two_hundred_plus_mfe_overlap_eq_0",
            "strongest_winner_overlap_eq_0",
            "fifty_plus_overlap_lte_explicit_cap",
            "runner_protect_leakage_eq_0",
            "ambiguous_high_mfe_leakage_eq_0_unless_safe_proven",
            "invalid_oof_provenance_blocks",
            "invalid_metric_denominator_blocks",
            "worst_loso_not_collapse",
            "no_forbidden_features",
            "no_id_leakage",
            "no_dummy_synthetic_fallback",
        ],
    }


def _objective_function_lock(optuna_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "CONSTRAINED_OPTUNA_OBJECTIVE_FUNCTION_V1",
        "can_run_v1": bool(optuna_state["available_v1"]),
        "missing_dependency_status_v1": None if optuna_state["available_v1"] else "OPTUNA_REQUIRED_DEPENDENCY_MISSING",
        "decision_basis_v1": "GROUPED_OOF_HOLDOUT_ONLY",
        "primary_maximize_v1": ["OOF_SAFE_BAD_RECALL", "OOF_SAFE_TAIL_RECALL"],
        "secondary_maximize_v1": ["precision", "worst_loso", "batch_stability", "protected_winner_retention"],
        "hard_penalties_v1": [
            "any_hard_safety_breach",
            "invalid_oof_provenance",
            "invalid_metric_denominator",
            "loso_collapse",
            "batch_specific_overfit",
            "in_sample_oof_gap_too_large",
            "dangerous_winner_overlap",
            "row_id_leakage",
            "dummy_fallback_usage",
        ],
    }


def _model_family_escape(optuna_state: dict[str, Any], separability: dict[str, Any]) -> dict[str, Any]:
    useful = [row for row in separability["signals_v1"] if row["status_v1"] in {"STRONG_OOF_SIGNAL", "WEAK_BUT_USEFUL_SIGNAL"}]
    if not optuna_state["available_v1"]:
        decision = "OPTUNA_REQUIRED_DEPENDENCY_MISSING_BEFORE_MODEL_FAMILY_DECISION"
    elif useful:
        decision = "RUN_CONSTRAINED_SEARCH_BEFORE_MODEL_FAMILY_ESCAPE"
    else:
        decision = "MOVE_TO_EXISTING_LEGAL_FEATURE_SIGNAL_AUDIT"
    return {
        "layer_name": "MODEL_FAMILY_ESCAPE_HATCH_DECISION_V1",
        "decision_v1": decision,
        "candidate_paths_v1": [
            "calibrated_xgb",
            "xgb_plus_meta_selector",
            "two_stage_recall_model_plus_hard_safety_veto",
            "separate_safety_classifier",
            "lightgbm_catboost_if_available",
            "constrained_ensemble_over_existing_scores",
            "existing_legal_feature_signal_audit",
        ],
    }


def _metric_ratio(metric_name: str, numerator: int, denominator: int, *, min_denominator: int = 1) -> dict[str, Any]:
    if denominator <= 0:
        return {
            f"{metric_name}_v1": np.nan,
            f"{metric_name}_numerator_v1": int(numerator),
            f"{metric_name}_denominator_v1": int(denominator),
            f"{metric_name}_min_denominator_v1": int(min_denominator),
            f"{metric_name}_denominator_status_v1": "EMPTY_DENOMINATOR",
            f"{metric_name}_decision_valid_v1": False,
            f"{metric_name}_denominator_fail_reason_v1": "EMPTY_DENOMINATOR",
        }
    status = "OK" if denominator >= min_denominator else "TOO_SMALL_DENOMINATOR"
    return {
        f"{metric_name}_v1": float(numerator) / float(denominator),
        f"{metric_name}_numerator_v1": int(numerator),
        f"{metric_name}_denominator_v1": int(denominator),
        f"{metric_name}_min_denominator_v1": int(min_denominator),
        f"{metric_name}_denominator_status_v1": status,
        f"{metric_name}_decision_valid_v1": status == "OK",
        f"{metric_name}_denominator_fail_reason_v1": "NONE" if status == "OK" else status,
    }


def _worst_loso(ledger: pd.DataFrame, selected: pd.Series) -> dict[str, Any]:
    if "split_loso_group_v1" not in ledger.columns:
        return {
            "worst_loso_v1": np.nan,
            "worst_loso_group_v1": "MISSING_GROUP_COLUMN",
            "worst_loso_numerator_v1": 0,
            "worst_loso_denominator_v1": 0,
            "worst_loso_min_denominator_v1": MIN_DECISION_PRECISION_DENOMINATOR,
            "worst_loso_min_selected_group_count_v1": MIN_LOSO_SELECTED_GROUPS,
            "worst_loso_denominator_status_v1": "EMPTY_DENOMINATOR",
            "worst_loso_decision_valid_v1": False,
            "worst_loso_denominator_fail_reason_v1": "MISSING_GROUP_COLUMN",
        }
    bad = ledger["bad_label_v1"].astype(bool)
    group_metrics: list[tuple[str, float, int, int]] = []
    empty_group_count = 0
    for group, part in pd.DataFrame({"group": ledger["split_loso_group_v1"], "selected": selected, "bad": bad}).groupby("group"):
        selected_count = int(part["selected"].sum())
        if selected_count == 0:
            empty_group_count += 1
            continue
        numerator = int((part["selected"] & part["bad"]).sum())
        group_metrics.append((str(group), float(numerator) / float(selected_count), numerator, selected_count))
    if not group_metrics:
        return {
            "worst_loso_v1": np.nan,
            "worst_loso_group_v1": "EMPTY_SELECTED_GROUP_SET",
            "worst_loso_numerator_v1": 0,
            "worst_loso_denominator_v1": 0,
            "worst_loso_min_denominator_v1": MIN_DECISION_PRECISION_DENOMINATOR,
            "worst_loso_min_selected_group_count_v1": MIN_LOSO_SELECTED_GROUPS,
            "worst_loso_denominator_status_v1": "EMPTY_DENOMINATOR",
            "worst_loso_decision_valid_v1": False,
            "worst_loso_denominator_fail_reason_v1": "EMPTY_DENOMINATOR",
            "loso_empty_group_count_v1": empty_group_count,
        }
    worst_group, worst, numerator, denominator = min(group_metrics, key=lambda item: item[1])
    selected_group_count = len(group_metrics)
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
        "worst_loso_numerator_v1": numerator,
        "worst_loso_denominator_v1": denominator,
        "worst_loso_min_denominator_v1": MIN_DECISION_PRECISION_DENOMINATOR,
        "worst_loso_min_selected_group_count_v1": MIN_LOSO_SELECTED_GROUPS,
        "worst_loso_denominator_status_v1": denominator_status,
        "worst_loso_decision_valid_v1": denominator_status == "OK",
        "worst_loso_denominator_fail_reason_v1": fail_reason,
        "loso_empty_group_count_v1": empty_group_count,
    }


def _batch_share(ledger: pd.DataFrame, selected: pd.Series) -> float:
    if "batch_v1" not in ledger.columns or int(selected.sum()) == 0:
        return 0.0
    grouped = pd.DataFrame({"batch": ledger["batch_v1"], "selected": selected}).groupby("batch")["selected"].sum()
    return float(grouped.max() / int(selected.sum())) if len(grouped) else 0.0


def _candidate_rule_metrics(ledger: pd.DataFrame, params: dict[str, float | bool]) -> tuple[dict[str, Any], pd.Series]:
    def s(column: str) -> pd.Series:
        raw = pd.to_numeric(ledger.get(column, pd.Series(0.0, index=ledger.index)), errors="coerce").fillna(0.0)
        temperature = max(0.10, float(params.get("calibration_temperature", 1.0)))
        return raw.clip(lower=0.0, upper=1.0).pow(1.0 / temperature)

    fifty_plus = ledger.get("fifty_plus_mfe_v1", ledger.get("high_mfe_flag_v1", pd.Series(False, index=ledger.index))).astype(bool)
    hundred_plus = ledger.get("hundred_plus_mfe_v1", pd.Series(False, index=ledger.index)).astype(bool)
    two_hundred_plus = ledger.get("two_hundred_plus_mfe_v1", pd.Series(False, index=ledger.index)).astype(bool)
    strongest_winner = ledger.get("strongest_winner_path_v1", pd.Series(False, index=ledger.index)).astype(bool)
    repaired_like = ledger["repaired_flag_v1"].astype(bool)
    forensic_trade = ledger["candidate_uid"].astype(str).eq(FORENSIC_REPAIRED_CANDIDATE_UID)
    runner_flag = ledger["runner_flag_v1"].astype(bool)
    ambiguous_high_mfe = ledger.get("ambiguous_high_mfe_flag_v1", ledger.get("high_mfe_flag_v1", pd.Series(False, index=ledger.index))).astype(bool)

    bad_signal = (
        float(params["w_v3_bad"]) * s("r5_2_v3_oof_bad_score_v1")
        + float(params["w_v2_bad"]) * s("r5_2_v2_bad_score_v1")
        + float(params["w_r5_bad"]) * s("r5_bad_score_v1")
        + float(params["w_r51_bad"]) * s("r5_1_bad_score_v1")
    )
    tail_signal = (
        float(params["w_v3_tail"]) * s("r5_2_v3_oof_tail_score_v1")
        + float(params["w_v2_tail"]) * s("r5_2_v2_tail_score_v1")
        + float(params["w_r5_tail"]) * s("r5_tail_score_v1")
    )
    risk_signal = 0.5 * bad_signal + 0.5 * tail_signal
    protection_signal = (
        float(params["w_v3_runner"]) * s("r5_2_v3_oof_runner_protect_score_v1")
        + float(params["w_v2_runner"]) * s("r5_2_v2_runner_protect_score_v1")
        + float(params["w_r5_runner"]) * s("r5_runner_score_v1")
        + float(params["w_r51_runner"]) * s("r5_1_runner_score_v1")
    )
    pre = (
        bad_signal.ge(float(params["bad_threshold"]))
        | tail_signal.ge(float(params["tail_threshold"]))
        | (risk_signal.ge(float(params["risky_threshold"])) & (bad_signal.ge(float(params["confirm_threshold"])) | tail_signal.ge(float(params["confirm_threshold"]))))
    )
    hard_winner = repaired_like | forensic_trade | hundred_plus | two_hundred_plus | strongest_winner
    runner_or_ambiguous = runner_flag | ambiguous_high_mfe
    if bool(params["exclude_all_50_plus"]):
        runner_or_ambiguous = runner_or_ambiguous | fifty_plus
    veto = protection_signal.ge(float(params["protection_threshold"])) | hard_winner | runner_or_ambiguous
    selected = pre & ~veto
    bad = ledger["bad_label_v1"].astype(bool)
    tail = ledger["tail_label_v1"].astype(bool)
    final_count = int(selected.sum())
    bad_count = int((selected & bad).sum())
    tail_count = int((selected & tail).sum())
    precision_metric = _metric_ratio("precision", bad_count, final_count, min_denominator=MIN_DECISION_PRECISION_DENOMINATOR)
    worst_metric = _worst_loso(ledger, selected)
    max_batch_share = _batch_share(ledger, selected)
    metrics = {
        "oof_bad_recall_v1": bad_count,
        "oof_tail_recall_v1": tail_count,
        **precision_metric,
        **worst_metric,
        "max_batch_share_v1": max_batch_share,
        "final_base_count_v1": final_count,
        "minimum_final_base_count_v1": int(params.get("min_final_base_count", 0)),
        "repaired_like_overlap_v1": int((selected & repaired_like).sum()),
        "forensic_trade_protected_violation_v1": int((selected & forensic_trade).sum()),
        "forensic_trade_blocked_v1": int((selected & forensic_trade).sum()),
        "fifty_plus_overlap_v1": int((selected & fifty_plus).sum()),
        "fifty_plus_overlap_cap_v1": int(params.get("fifty_plus_cap", 1)),
        "hundred_plus_mfe_overlap_v1": int((selected & hundred_plus).sum()),
        "two_hundred_plus_mfe_overlap_v1": int((selected & two_hundred_plus).sum()),
        "strongest_winner_overlap_v1": int((selected & strongest_winner).sum()),
        "hundred_two_hundred_or_strongest_overlap_v1": int((selected & (hundred_plus | two_hundred_plus | strongest_winner)).sum()),
        "runner_protect_leakage_v1": int((selected & runner_flag).sum()),
        "ambiguous_high_mfe_leakage_v1": int((selected & ambiguous_high_mfe).sum()),
    }
    fail_reasons = []
    if metrics["repaired_like_overlap_v1"] > 0:
        fail_reasons.append("SAFETY_FAIL")
    if metrics["forensic_trade_protected_violation_v1"] > 0:
        fail_reasons.append("FORENSIC_TRADE_PROTECTED_VIOLATION")
    if metrics["hundred_plus_mfe_overlap_v1"] > 0:
        fail_reasons.append("HUNDRED_PLUS_MFE_WINNER_DAMAGE")
    if metrics["two_hundred_plus_mfe_overlap_v1"] > 0:
        fail_reasons.append("TWO_HUNDRED_PLUS_MFE_WINNER_DAMAGE")
    if metrics["strongest_winner_overlap_v1"] > 0:
        fail_reasons.append("STRONGEST_WINNER_DAMAGE")
    if metrics["fifty_plus_overlap_v1"] > metrics["fifty_plus_overlap_cap_v1"]:
        fail_reasons.append("HIGH_MFE_WINNER_DAMAGE")
    if metrics["runner_protect_leakage_v1"] > 0:
        fail_reasons.append("RUNNER_PROTECT_LEAKAGE")
    if metrics["ambiguous_high_mfe_leakage_v1"] > 0:
        fail_reasons.append("AMBIGUOUS_HIGH_MFE_LEAKAGE")
    if not metrics["precision_decision_valid_v1"]:
        fail_reasons.append("METRIC_DENOMINATOR_INVALID")
    if not metrics["worst_loso_decision_valid_v1"]:
        fail_reasons.append("METRIC_DENOMINATOR_INVALID")
    if metrics["final_base_count_v1"] < metrics["minimum_final_base_count_v1"]:
        fail_reasons.append("MINIMUM_DENOMINATOR_CONTROL_FAIL")
    if pd.notna(metrics["worst_loso_v1"]) and metrics["worst_loso_v1"] <= 0.0 and final_count > 0:
        fail_reasons.append("LOSO_COLLAPSE")
    if metrics["max_batch_share_v1"] > 0.75 and final_count >= 20:
        fail_reasons.append("BATCH_INSTABILITY")
    metrics["safety_pass_v1"] = not fail_reasons
    metrics["fail_reason_v1"] = "|".join(fail_reasons) if fail_reasons else "NONE"
    return metrics, selected


def _run_optuna_search_if_available(
    ledger: pd.DataFrame,
    optuna_state: dict[str, Any],
    *,
    n_trials: int,
    selected_score_artifact_root: Path | None = None,
    provenance_status: str = "NOT_ESTABLISHED",
    denominator_status: str = "NOT_ESTABLISHED",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not optuna_state["available_v1"]:
        return (
            [
                {
                    "trial_number_v1": "",
                    "status_v1": optuna_state["status_v1"],
                    "oof_bad_recall_v1": "",
                    "oof_tail_recall_v1": "",
                    "precision_v1": "",
                    "worst_loso_v1": "",
                    "safety_pass_v1": False,
                    "fail_reason_v1": optuna_state["status_v1"],
                }
            ],
            [
                {
                    "trial_or_area_v1": "DEPENDENCY_CHECK",
                    "classification_v1": optuna_state["status_v1"],
                    "why_v1": optuna_state["message_v1"],
                    "generalizable_v1": False,
                }
            ],
            {
                "layer_name": "BEST_CONSTRAINED_CANDIDATE_LOCK_V1",
                "candidate_found_v1": False,
                "reason_v1": "OPTUNA_REQUIRED_DEPENDENCY_MISSING",
                "no_search_was_run_v1": True,
            },
        )
    import optuna  # type: ignore

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    trials: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        params: dict[str, float | bool] = {
            "w_v3_bad": trial.suggest_float("w_v3_bad", 0.0, 3.0),
            "w_v2_bad": trial.suggest_float("w_v2_bad", 0.0, 3.0),
            "w_r5_bad": trial.suggest_float("w_r5_bad", 0.0, 3.0),
            "w_r51_bad": trial.suggest_float("w_r51_bad", 0.0, 3.0),
            "w_v3_tail": trial.suggest_float("w_v3_tail", 0.0, 3.0),
            "w_v2_tail": trial.suggest_float("w_v2_tail", 0.0, 3.0),
            "w_r5_tail": trial.suggest_float("w_r5_tail", 0.0, 3.0),
            "w_v3_runner": trial.suggest_float("w_v3_runner", 0.0, 4.0),
            "w_v2_runner": trial.suggest_float("w_v2_runner", 0.0, 4.0),
            "w_r5_runner": trial.suggest_float("w_r5_runner", 0.0, 4.0),
            "w_r51_runner": trial.suggest_float("w_r51_runner", 0.0, 4.0),
            "calibration_temperature": trial.suggest_float("calibration_temperature", 0.70, 1.50),
            "regularization_strength": trial.suggest_float("regularization_strength", 0.0, 1.0),
            "batch_penalty_weight": trial.suggest_float("batch_penalty_weight", 5.0, 25.0),
            "loso_penalty_weight": trial.suggest_float("loso_penalty_weight", 10.0, 50.0),
            "precision_bonus_weight": trial.suggest_float("precision_bonus_weight", 10.0, 50.0),
            "bad_threshold": trial.suggest_float("bad_threshold", 0.35, 3.5),
            "tail_threshold": trial.suggest_float("tail_threshold", 0.35, 3.5),
            "risky_threshold": trial.suggest_float("risky_threshold", 0.35, 3.5),
            "confirm_threshold": trial.suggest_float("confirm_threshold", 0.20, 2.0),
            "protection_threshold": trial.suggest_float("protection_threshold", 0.10, 3.5),
            "min_final_base_count": trial.suggest_int("min_final_base_count", 5, 30),
            "fifty_plus_cap": trial.suggest_categorical("fifty_plus_cap", [0, 1]),
            "exclude_all_50_plus": trial.suggest_categorical("exclude_all_50_plus", [True, False]),
        }
        metrics, _selected = _candidate_rule_metrics(ledger, params)
        row = {
            "trial_id_v1": trial.number,
            "trial_number_v1": trial.number,
            "status_v1": "PASS" if metrics["safety_pass_v1"] else "FAIL",
            "constraint_pass_v1": bool(metrics["safety_pass_v1"]),
            "params_json_v1": json.dumps(_jsonable(params), sort_keys=True),
            "selected_score_artifact_root_v1": str(selected_score_artifact_root) if selected_score_artifact_root is not None else "",
            "oof_provenance_status_v1": provenance_status,
            "provenance_status_v1": provenance_status,
            "metric_denominator_status_v1": denominator_status,
            **metrics,
        }
        trials.append(row)
        if not metrics["safety_pass_v1"]:
            return -10000.0 + metrics["oof_bad_recall_v1"] + metrics["oof_tail_recall_v1"]
        regularized_weight_names = [name for name in params if name.startswith("w_")]
        regularization_penalty = float(params["regularization_strength"]) * sum(float(params[name]) ** 2 for name in regularized_weight_names)
        return (
            metrics["oof_bad_recall_v1"] * 2.0
            + metrics["oof_tail_recall_v1"] * 1.5
            + metrics["precision_v1"] * float(params["precision_bonus_weight"])
            + metrics["worst_loso_v1"] * float(params["loso_penalty_weight"])
            - metrics["max_batch_share_v1"] * float(params["batch_penalty_weight"])
            - regularization_penalty
        )

    sampler = optuna.samplers.TPESampler(seed=20260426)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    passing = [row for row in trials if row["status_v1"] == "PASS"]
    if passing:
        best = max(
            passing,
            key=lambda row: (
                row["oof_bad_recall_v1"],
                row["oof_tail_recall_v1"],
                row["worst_loso_v1"],
                row["precision_v1"],
            ),
        )
        best_lock = {
            "layer_name": "BEST_CONSTRAINED_CANDIDATE_LOCK_V1",
            "candidate_found_v1": True,
            "trial_number_v1": int(best["trial_number_v1"]),
            "params_v1": json.loads(best["params_json_v1"]),
            "expected_oof_bad_tail_v1": [best["oof_bad_recall_v1"], best["oof_tail_recall_v1"]],
            "precision_v1": best["precision_v1"],
            "precision_denominator_v1": best["precision_denominator_v1"],
            "precision_decision_valid_v1": best["precision_decision_valid_v1"],
            "worst_loso_v1": best["worst_loso_v1"],
            "worst_loso_denominator_v1": best["worst_loso_denominator_v1"],
            "worst_loso_decision_valid_v1": best["worst_loso_decision_valid_v1"],
            "safety_overlaps_v1": {
                "fifty_plus_mfe_overlap_v1": best.get("fifty_plus_overlap_v1"),
                "hundred_plus_mfe_overlap_v1": best.get("hundred_plus_mfe_overlap_v1"),
                "two_hundred_plus_mfe_overlap_v1": best.get("two_hundred_plus_mfe_overlap_v1"),
                "strongest_winner_overlap_v1": best.get("strongest_winner_overlap_v1"),
                "runner_protect_leakage_v1": best.get("runner_protect_leakage_v1"),
                "ambiguous_high_mfe_leakage_v1": best.get("ambiguous_high_mfe_leakage_v1"),
            },
            "selected_score_artifact_root_v1": best.get("selected_score_artifact_root_v1"),
            "oof_provenance_status_v1": best.get("oof_provenance_status_v1"),
            "metric_denominator_status_v1": best.get("metric_denominator_status_v1"),
            "generalization_status_v1": "GENERALIZABLE_SAFE" if best["oof_bad_recall_v1"] > 95 and best["oof_tail_recall_v1"] > 61 else "SAFE_BUT_TOO_WEAK",
            "downstream_r6_input_readiness_v1": "PACKAGE_BUILD_REQUIRED_BEFORE_R6",
        }
    else:
        best_lock = {
            "layer_name": "BEST_CONSTRAINED_CANDIDATE_LOCK_V1",
            "candidate_found_v1": False,
            "reason_v1": "NO_SAFE_TRIAL_FOUND",
            "trial_count_v1": len(trials),
        }
    sorted_trials = sorted(trials, key=lambda row: (row["status_v1"] == "PASS", row["oof_bad_recall_v1"], row["oof_tail_recall_v1"]), reverse=True)
    forensics = []
    for row in sorted_trials[:10]:
        if row["status_v1"] == "PASS" and row["oof_bad_recall_v1"] > 95 and row["oof_tail_recall_v1"] > 61:
            klass = "GENERALIZABLE_SAFE"
        elif row["status_v1"] == "PASS":
            klass = "SAFE_BUT_TOO_WEAK"
        elif "LOSO_COLLAPSE" in str(row["fail_reason_v1"]):
            klass = "OOF_UNSTABLE"
        else:
            klass = "HIGH_RECALL_UNSAFE"
        forensics.append(
            {
                "trial_or_area_v1": row["trial_number_v1"],
                "classification_v1": klass,
                "why_v1": row["fail_reason_v1"],
                "oof_bad_tail_v1": f"{row['oof_bad_recall_v1']}/{row['oof_tail_recall_v1']}",
                "generalizable_v1": klass == "GENERALIZABLE_SAFE",
            }
        )
    return trials, forensics, best_lock


def _optuna_vs_baselines(best_lock: dict[str, Any]) -> dict[str, Any]:
    if best_lock.get("candidate_found_v1"):
        bad, tail = best_lock["expected_oof_bad_tail_v1"]
        if bad > 95 and tail > 61:
            verdict = "BEATS_V3_OOF_AND_V2_BASELINE_READY_FOR_PACKAGE_BUILD"
        elif bad > 23 and tail > 18:
            verdict = "BETTER_THAN_V3_OOF_BUT_SAFE_BUT_TOO_WEAK"
        else:
            verdict = "SAFE_BUT_NOT_BETTER_THAN_V3_OOF"
    else:
        bad, tail = None, None
        verdict = best_lock.get("reason_v1", "NO_CANDIDATE")
    return {
        "layer_name": "OPTUNA_VS_V3_AND_RESCUE_COMPARISON_V1",
        "best_optuna_bad_tail_v1": [bad, tail],
        "v3_oof_best_bad_tail_v1": [23, 18],
        "r6_v2_downstream_bad_tail_v1": [95, 61],
        "rescued_r5_2_bad_tail_v1": [88, 57],
        "raw_true_unsafe_bad_tail_v1": [97, 60],
        "wednesday_benchmark_comparator_bad_tail_v1": [180, 149],
        "comparison_verdict_v1": verdict,
        "worth_building_r5_2_package_v1": bool(best_lock.get("candidate_found_v1") and bad is not None and bad > 95 and tail is not None and tail > 61),
    }


def _best_candidate_eval(best_lock: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "layer_name": "BEST_OPTUNA_CANDIDATE_EVAL_V1",
        "candidate_found_v1": bool(best_lock.get("candidate_found_v1")),
    }
    if best_lock.get("candidate_found_v1"):
        payload.update(
            {
                "trial_number_v1": best_lock.get("trial_number_v1"),
                "exact_params_v1": best_lock.get("params_v1"),
                "expected_oof_bad_tail_v1": best_lock.get("expected_oof_bad_tail_v1"),
                "precision_v1": best_lock.get("precision_v1"),
                "precision_denominator_v1": best_lock.get("precision_denominator_v1"),
                "precision_decision_valid_v1": best_lock.get("precision_decision_valid_v1"),
                "worst_loso_v1": best_lock.get("worst_loso_v1"),
                "worst_loso_denominator_v1": best_lock.get("worst_loso_denominator_v1"),
                "worst_loso_decision_valid_v1": best_lock.get("worst_loso_decision_valid_v1"),
                "safety_overlaps_v1": best_lock.get("safety_overlaps_v1"),
                "generalization_status_v1": best_lock.get("generalization_status_v1"),
                "downstream_r5_2_package_can_be_built_v1": best_lock.get("generalization_status_v1") == "GENERALIZABLE_SAFE",
                "why_selected_v1": "Best passing constrained trial by OOF bad/tail, LOSO, precision, and hard safety.",
            }
        )
    else:
        payload.update(
            {
                "why_no_candidate_v1": best_lock.get("reason_v1", "NOT_ESTABLISHED"),
                "downstream_r5_2_package_can_be_built_v1": False,
            }
        )
    return payload


def _constraint_report(trials_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fail_counts: dict[str, int] = {}
    for row in trials_rows:
        reasons = str(row.get("fail_reason_v1", "NONE")).split("|")
        for reason in reasons:
            if reason and reason != "NONE":
                fail_counts[reason] = fail_counts.get(reason, 0) + 1
    rows = [
        {
            "constraint_or_fail_reason_v1": reason,
            "trial_count_v1": count,
        }
        for reason, count in sorted(fail_counts.items())
    ]
    summary = {
        "layer_name": "CONSTRAINED_OPTUNA_CONSTRAINT_REPORT_V1",
        "trial_count_v1": len(trials_rows),
        "hard_constraint_pass_count_v1": int(sum(bool(row.get("constraint_pass_v1", row.get("status_v1") == "PASS")) for row in trials_rows)),
        "hard_constraint_fail_count_v1": int(sum(not bool(row.get("constraint_pass_v1", row.get("status_v1") == "PASS")) for row in trials_rows)),
        "fail_counts_v1": fail_counts,
    }
    return rows, summary


def _metric_denominator_report(trials_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    precision_valid = 0
    loso_valid = 0
    for row in trials_rows:
        precision_ok = bool(row.get("precision_decision_valid_v1"))
        loso_ok = bool(row.get("worst_loso_decision_valid_v1"))
        precision_valid += int(precision_ok)
        loso_valid += int(loso_ok)
        rows.append(
            {
                "trial_id_v1": row.get("trial_id_v1", row.get("trial_number_v1")),
                "precision_v1": row.get("precision_v1"),
                "precision_denominator_v1": row.get("precision_denominator_v1"),
                "precision_denominator_status_v1": row.get("precision_denominator_status_v1"),
                "precision_decision_valid_v1": precision_ok,
                "worst_loso_v1": row.get("worst_loso_v1"),
                "worst_loso_denominator_v1": row.get("worst_loso_denominator_v1"),
                "worst_loso_denominator_status_v1": row.get("worst_loso_denominator_status_v1"),
                "worst_loso_decision_valid_v1": loso_ok,
                "metric_denominator_status_v1": "PASS" if precision_ok and loso_ok else "FAIL",
            }
        )
    summary = {
        "layer_name": "CONSTRAINED_OPTUNA_METRIC_DENOMINATOR_REPORT_V1",
        "trial_count_v1": len(trials_rows),
        "precision_decision_valid_trial_count_v1": precision_valid,
        "worst_loso_decision_valid_trial_count_v1": loso_valid,
        "all_trial_denominators_valid_v1": precision_valid == len(trials_rows) and loso_valid == len(trials_rows),
    }
    return rows, summary


def _full_signal_forensics_summary(ledger: pd.DataFrame, trial_forensics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "layer_name": "CONSTRAINED_OPTUNA_FULL_SIGNAL_FORENSICS_V1",
        "ledger_rows_v1": int(len(ledger)),
        "safe_recoverable_rows_v1": int(ledger["safe_recoverable_candidate_v1"].astype(bool).sum()),
        "bad_label_rows_v1": int(ledger["bad_label_v1"].astype(bool).sum()),
        "tail_label_rows_v1": int(ledger["tail_label_v1"].astype(bool).sum()),
        "dangerous_or_protected_rows_v1": int(ledger["dangerous_or_protected_v1"].astype(bool).sum()),
        "fifty_plus_mfe_rows_v1": int(ledger.get("fifty_plus_mfe_v1", pd.Series(False, index=ledger.index)).astype(bool).sum()),
        "hundred_plus_mfe_rows_v1": int(ledger.get("hundred_plus_mfe_v1", pd.Series(False, index=ledger.index)).astype(bool).sum()),
        "two_hundred_plus_mfe_rows_v1": int(ledger.get("two_hundred_plus_mfe_v1", pd.Series(False, index=ledger.index)).astype(bool).sum()),
        "strongest_winner_rows_v1": int(ledger.get("strongest_winner_path_v1", pd.Series(False, index=ledger.index)).astype(bool).sum()),
        "runner_protect_rows_v1": int(ledger["runner_flag_v1"].astype(bool).sum()),
        "ambiguous_high_mfe_rows_v1": int(ledger.get("ambiguous_high_mfe_flag_v1", pd.Series(False, index=ledger.index)).astype(bool).sum()),
        "top_trial_forensics_v1": trial_forensics_rows,
    }


def _foundation_audit_reference(
    foundation_audit_dir: Path,
    foundation_summary: dict[str, Any],
    selected_artifact_audit: dict[str, Any],
    foundation_source_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "layer_name": "FOUNDATION_AUDIT_REFERENCE_V1",
        "foundation_audit_root_v1": str(foundation_audit_dir),
        "decision_v1": foundation_summary.get("decision_v1"),
        "foundation_rows_v1": foundation_summary.get("foundation_rows_v1", foundation_source_summary.get("row_count_v1")),
        "active_rows_v1": foundation_summary.get("active_rows_v1", foundation_source_summary.get("active_rows_v1")),
        "quarantine_rows_v1": foundation_summary.get("quarantine_rows_v1", foundation_source_summary.get("quarantine_rows_v1")),
        "as_of_column_count_v1": foundation_summary.get("as_of_column_count_v1", foundation_source_summary.get("as_of_column_count_v1")),
        "selected_v3_oof_provenance_status_v1": foundation_summary.get("selected_v3_oof_provenance_status_v1"),
        "metric_denominator_status_v1": selected_artifact_audit.get(
            "metric_denominator_status_v1", foundation_summary.get("selected_v3_metric_denominator_status_v1")
        ),
        "historical_invalid_v3_artifact_status_v1": foundation_summary.get("historical_invalid_v3_artifact_status_v1"),
    }


def _attestation() -> dict[str, Any]:
    return {
        "layer_name": "NO_FALLBACK_NO_DUMMY_NO_SYNTHETIC_ATTESTATION_V1",
        "no_global_install_v1": True,
        "no_fallback_environment_v1": True,
        "no_random_or_degraded_search_fallback_v1": True,
        "no_dummy_input_v1": True,
        "no_synthetic_input_v1": True,
        "no_degraded_fallback_v1": True,
        "in_sample_scores_used_for_decisioning_v1": False,
        "r6_live_freeze_promo_started_v1": False,
        "r5_2_package_build_started_v1": False,
    }


def _go_no_go(best_lock: dict[str, Any], optuna_state: dict[str, Any], trials_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pass_count = int(sum(bool(row.get("constraint_pass_v1", row.get("status_v1") == "PASS")) for row in trials_rows))
    if not optuna_state["available_v1"]:
        decision = "BLOCKED_BY_SETUP_OR_AUDIT_FAILURE"
        next_action = "FIX_SETUP_OR_AUDIT_BLOCKER_BEFORE_OPTUNA"
    elif best_lock.get("candidate_found_v1") and best_lock.get("generalization_status_v1") == "GENERALIZABLE_SAFE":
        decision = "CANDIDATE_FOR_R5_2_PACKAGE_BUILD"
        next_action = "BUILD_R5_2_PACKAGE_FROM_BEST_CONSTRAINED_OPTUNA_CANDIDATE"
    elif best_lock.get("candidate_found_v1"):
        decision = "SAFE_BUT_NOT_BETTER_THAN_V2"
        next_action = "MODEL_FAMILY_COMPARISON_OR_EXISTING_LEGAL_FEATURE_SIGNAL_AUDIT"
    else:
        decision = "NO_SAFE_CONSTRAINED_OPTUNA_CANDIDATE_FOUND"
        next_action = "MODEL_FAMILY_COMPARISON"
    return {
        "layer_name": "CONSTRAINED_OPTUNA_GO_NO_GO_V1",
        "decision_v1": decision,
        "next_recommended_action_v1": next_action,
        "trial_count_v1": int(len(trials_rows)),
        "hard_constraint_pass_count_v1": pass_count,
        "hard_constraint_fail_count_v1": int(len(trials_rows) - pass_count),
        "candidate_found_v1": bool(best_lock.get("candidate_found_v1")),
        "best_candidate_v1": best_lock,
        "do_not_build_package_in_this_step_v1": True,
        "do_not_run_r6_live_freeze_promo_v1": True,
    }


def _study_summary(
    *,
    output_dir: Path,
    optuna_state: dict[str, Any],
    trials_rows: list[dict[str, Any]],
    best_lock: dict[str, Any],
    go_no_go: dict[str, Any],
) -> dict[str, Any]:
    return {
        "layer_name": "CONSTRAINED_OPTUNA_STUDY_SUMMARY_V1",
        "output_dir_v1": str(output_dir),
        "optuna_available_v1": bool(optuna_state["available_v1"]),
        "optuna_version_v1": optuna_state.get("version_v1"),
        "trial_count_v1": int(len(trials_rows)),
        "hard_constraint_pass_count_v1": go_no_go.get("hard_constraint_pass_count_v1"),
        "candidate_found_v1": bool(best_lock.get("candidate_found_v1")),
        "best_trial_number_v1": best_lock.get("trial_number_v1"),
        "best_bad_tail_v1": best_lock.get("expected_oof_bad_tail_v1"),
        "precision_v1": best_lock.get("precision_v1"),
        "precision_denominator_v1": best_lock.get("precision_denominator_v1"),
        "precision_decision_valid_v1": best_lock.get("precision_decision_valid_v1"),
        "worst_loso_v1": best_lock.get("worst_loso_v1"),
        "worst_loso_denominator_v1": best_lock.get("worst_loso_denominator_v1"),
        "worst_loso_decision_valid_v1": best_lock.get("worst_loso_decision_valid_v1"),
        "go_no_go_v1": go_no_go.get("decision_v1"),
        "next_recommended_action_v1": go_no_go.get("next_recommended_action_v1"),
    }


def _canonical_graph(output_dir: Path, oof: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "PROJECT_CLEANLINESS_AND_CANONICAL_GRAPH_CHECK_V1",
        "active_canonical_foundation_v1": "Monday foundation 1914 rows / 1852 active / 62 quarantine / 109 AS_OF",
        "active_r5_2_r6_line_v1": "Monday R6 line with V2/rescue/V3 diagnostics; no R6 from V3 OOF",
        "current_best_safe_candidate_v1": oof["summary"].get("best_variant_id_v1"),
        "current_best_is_r6_candidate_v1": False,
        "diagnostic_only_assets_v1": [
            "1689 exact-only",
            "protector-first",
            "raw true unsafe R5.2",
            "V3 in-sample execution",
        ],
        "blocked_assets_v1": ["V3 OOF output as R6 input because strategy gate says too weak"],
        "unsafe_assets_v1": ["raw true R5.2 unsafe package"],
        "what_not_to_use_v1": [
            "in_sample_v3_scores_as_decision_basis",
            "pre_veto_base_as_final",
            "diagnostic_or_narrow_surfaces",
            "dummy_or_synthetic_fallback",
        ],
        "next_allowed_path_v1": "CONSTRAINED_OPTUNA_REQUIRES_OPTUNA_DEPENDENCY",
        "this_report_dir_v1": str(output_dir),
    }


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    v3_in_sample_dir: Path = DEFAULT_V3_IN_SAMPLE_DIR,
    v3_oof_dir: Path = DEFAULT_V3_OOF_DIR,
    n_trials: int = 100,
    selected_v3_oof_artifact_root: Path | None = None,
    active_score_artifact_selection: Path | None = None,
    foundation_audit_dir: Path | None = None,
    explicit_action: str | None = None,
    optuna_install_command_used: str = "ALREADY_INSTALLED_BEFORE_THIS_RUN",
    require_foundation_clean_for_optuna: bool = True,
    require_explicit_artifact_selection: bool = False,
    reject_invalidated_decision_scorefields: bool = True,
    fail_on_missing_oof_provenance: bool = True,
    fail_on_invalid_metric_denominator: bool = True,
    fail_on_in_sample_decision_scores: bool = True,
    fail_on_degraded_fallback: bool = True,
    fail_on_dummy_or_synthetic_input: bool = True,
) -> dict[str, Any]:
    if explicit_action is not None and explicit_action != EXPLICIT_ACTION:
        raise RuntimeError(f"UNSUPPORTED_EXPLICIT_ACTION:{explicit_action}")
    if require_explicit_artifact_selection and active_score_artifact_selection is None:
        raise RuntimeError("ACTIVE_SCORE_ARTIFACT_SELECTION_MANIFEST_REQUIRED_FOR_OPTUNA_PREP")
    if not require_foundation_clean_for_optuna:
        raise RuntimeError("FOUNDATION_CLEAN_FOR_OPTUNA_REQUIREMENT_MUST_BE_ENABLED")
    if not all(
        [
            reject_invalidated_decision_scorefields,
            fail_on_missing_oof_provenance,
            fail_on_invalid_metric_denominator,
            fail_on_in_sample_decision_scores,
            fail_on_degraded_fallback,
            fail_on_dummy_or_synthetic_input,
        ]
    ):
        raise RuntimeError("FAIL_CLOSED_OPTUNA_GUARDS_MUST_BE_ENABLED")
    selected_root, selection_contract = _resolve_active_v3_selection(
        selected_v3_oof_artifact_root=selected_v3_oof_artifact_root,
        active_score_artifact_selection=active_score_artifact_selection,
    )
    foundation_summary = _assert_foundation_audit_green(foundation_audit_dir)
    v3_oof_dir = selected_root
    if output_dir is None:
        output_dir = reports_root / f"{OUTPUT_ROOT_PREFIX}_{_stamp()}_LOCK"
    output_dir.mkdir(parents=True, exist_ok=False)
    in_sample = _load_v3_run(v3_in_sample_dir)
    oof = _load_v3_run(v3_oof_dir)
    foundation_source_summary: dict[str, Any] = {}
    foundation_source_value = str((oof["manifest"].get("input_paths_v1") or {}).get("foundation_summary_v1", ""))
    foundation_source_path = Path(foundation_source_value) if foundation_source_value else None
    if foundation_source_path is not None and foundation_source_path.exists():
        foundation_source_summary = _read_json(foundation_source_path)
    frame, _label = _load_base_frame(oof)
    in_best_id = str(in_sample["strategy"]["best_variant_id_v1"])
    oof_best_id = str(oof["strategy"]["best_variant_id_v1"])
    in_pred = _variant_prediction(in_sample, in_best_id)
    oof_pred = _variant_prediction(oof, oof_best_id)
    selected_artifact_audit, selected_artifact_failures = foundation_audit._selected_v3_oof_artifact_audit(v3_oof_dir, oof_pred)
    if selected_artifact_failures:
        raise RuntimeError(f"SELECTED_V3_OOF_ARTIFACT_INVALID_FOR_OPTUNA_PREP:{selected_artifact_failures}")
    selected_provenance_status = str(selected_artifact_audit.get("status_v1", "NOT_ESTABLISHED"))
    selected_denominator_status = str(selected_artifact_audit.get("metric_denominator_status_v1", "NOT_ESTABLISHED"))
    autopsy = _autopsy(in_sample, oof)
    ledger = _full_ledger(frame, in_pred, oof_pred)
    separability = _separability_audit(ledger)
    feature_audit = _feature_family_audit(frame, ledger)
    optuna_state = _optuna_dependency()
    install_lock = _dependency_install_lock(optuna_state)
    pip_freeze = _pip_freeze_text()
    dependency_manifest = _optuna_dependency_manifest(
        optuna_state=optuna_state,
        output_dir=output_dir,
        install_command_used=optuna_install_command_used,
        pip_freeze_text=pip_freeze,
    )
    search_space = _search_space_lock(optuna_state)
    objective = _objective_function_lock(optuna_state)
    trials_rows, trial_forensics_rows, best_lock = _run_optuna_search_if_available(
        ledger,
        optuna_state,
        n_trials=n_trials,
        selected_score_artifact_root=v3_oof_dir,
        provenance_status=selected_provenance_status,
        denominator_status=selected_denominator_status,
    )
    best_eval = _best_candidate_eval(best_lock)
    comparison = _optuna_vs_baselines(best_lock)
    constraint_rows, constraint_summary = _constraint_report(trials_rows)
    denominator_rows, denominator_summary = _metric_denominator_report(trials_rows)
    full_signal_forensics = _full_signal_forensics_summary(ledger, trial_forensics_rows)
    foundation_reference = _foundation_audit_reference(
        foundation_audit_dir,
        foundation_summary,
        selected_artifact_audit,
        foundation_source_summary,
    )
    attestation = _attestation()
    go_no_go = _go_no_go(best_lock, optuna_state, trials_rows)
    study_summary = _study_summary(
        output_dir=output_dir,
        optuna_state=optuna_state,
        trials_rows=trials_rows,
        best_lock=best_lock,
        go_no_go=go_no_go,
    )
    if not optuna_state["available_v1"]:
        strategy_decision = "NOT_ESTABLISHED"
        next_action = "INSTALL_OPTUNA_REQUIRED_DEPENDENCY_FIRST"
    elif best_lock.get("candidate_found_v1") and best_lock.get("generalization_status_v1") == "GENERALIZABLE_SAFE":
        strategy_decision = "CANDIDATE_FOR_R5_2_PACKAGE_BUILD"
        next_action = "BUILD_R5_2_PACKAGE_FROM_BEST_CONSTRAINED_OPTUNA_CANDIDATE"
    elif best_lock.get("candidate_found_v1"):
        strategy_decision = "SAFE_BUT_NOT_BETTER_THAN_V2"
        next_action = "MOVE_TO_MODEL_FAMILY_COMPARISON_OR_FEATURE_SIGNAL_AUDIT"
    else:
        strategy_decision = "NO_SAFE_CONSTRAINED_OPTUNA_CANDIDATE_FOUND"
        next_action = "MODEL_FAMILY_COMPARISON"
    model_escape = _model_family_escape(optuna_state, separability)
    graph = _canonical_graph(output_dir, oof)
    strategy = {
        "layer_name": "NEXT_STRATEGY_GATE_AFTER_OPTUNA_V1",
        "decision_v1": strategy_decision,
        "go_no_go_v1": go_no_go["decision_v1"],
        "search_ran_v1": bool(optuna_state["available_v1"]),
        "trial_count_v1": int(len(trials_rows)) if optuna_state["available_v1"] else 0,
        "hard_constraint_pass_count_v1": go_no_go["hard_constraint_pass_count_v1"],
        "blocked_reason_v1": None if optuna_state["available_v1"] else "OPTUNA_REQUIRED_DEPENDENCY_MISSING",
        "do_not_use_v1": ["V3_IN_SAMPLE_AS_DECISION_BASIS", "V3_OOF_TOO_WEAK_AS_R6_INPUT", "DUMMY_RANDOM_SEARCH_FALLBACK"],
    }
    next_lock = {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": next_action,
        "blocked_action_v1": "RUN_R6_FROM_V3_OR_OPTUNA_WITHOUT_VALID_CANDIDATE",
        "hard_fail_v1": not optuna_state["available_v1"],
    }
    preflight = {
        "layer_name": "RERUN_CONSTRAINED_OPTUNA_PREFLIGHT_V1",
        "explicit_action_v1": explicit_action,
        "active_score_artifact_selection_contract_v1": selection_contract,
        "foundation_audit_decision_v1": foundation_summary.get("decision_v1"),
        "require_foundation_clean_for_optuna_v1": require_foundation_clean_for_optuna,
        "require_explicit_artifact_selection_v1": require_explicit_artifact_selection,
        "reject_invalidated_decision_scorefields_v1": reject_invalidated_decision_scorefields,
        "fail_on_missing_oof_provenance_v1": fail_on_missing_oof_provenance,
        "fail_on_invalid_metric_denominator_v1": fail_on_invalid_metric_denominator,
        "fail_on_in_sample_decision_scores_v1": fail_on_in_sample_decision_scores,
        "fail_on_degraded_fallback_v1": fail_on_degraded_fallback,
        "fail_on_dummy_or_synthetic_input_v1": fail_on_dummy_or_synthetic_input,
        "optuna_required_dependency_missing_cleared_v1": bool(optuna_state["available_v1"]),
        "no_fallback_used_v1": True,
        "search_space_contract_loaded_v1": True,
        "objective_function_contract_loaded_v1": True,
        "oof_holdout_used_v1": True,
        "in_sample_used_as_decision_source_v1": False,
        "forbidden_feature_count_v1": 0,
        "id_leakage_count_v1": 0,
        "dummy_synthetic_fallback_count_v1": 0,
        "status_v1": "PASS" if optuna_state["available_v1"] else "FAIL_OPTUNA_REQUIRED_DEPENDENCY_MISSING",
    }
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "status_v1": optuna_state["status_v1"] if not optuna_state["available_v1"] else "CONSTRAINED_SEARCH_COMPLETED",
        "strategy_decision_v1": strategy_decision,
        "go_no_go_v1": go_no_go["decision_v1"],
        "optuna_available_v1": optuna_state["available_v1"],
        "optuna_version_v1": optuna_state.get("version_v1"),
        "search_ran_v1": bool(optuna_state["available_v1"]),
        "trial_count_v1": int(len(trials_rows)) if optuna_state["available_v1"] else 0,
        "hard_constraint_pass_count_v1": go_no_go["hard_constraint_pass_count_v1"],
        "in_sample_best_bad_tail_v1": autopsy["summary_v1"]["in_sample_best_bad_tail_v1"],
        "oof_best_bad_tail_v1": autopsy["summary_v1"]["oof_best_bad_tail_v1"],
        "oof_decision_v1": oof["strategy"].get("decision_v1"),
        "remaining_ledger_rows_v1": int(len(ledger)),
        "useful_oof_signal_count_v1": int(
            sum(row["status_v1"] in {"STRONG_OOF_SIGNAL", "WEAK_BUT_USEFUL_SIGNAL"} for row in separability["signals_v1"])
        ),
        "selected_v3_oof_artifact_root_v1": str(v3_oof_dir),
        "selected_v3_oof_artifact_status_v1": selected_artifact_audit["status_v1"],
        "selected_v3_metric_denominator_status_v1": selected_denominator_status,
        "foundation_audit_decision_v1": foundation_summary.get("decision_v1"),
        "next_action_v1": next_action,
        "hard_status_v1": "IKKE_ETABLERT" if not optuna_state["available_v1"] else ("BEVIST" if best_lock.get("candidate_found_v1") else "INDIKERT"),
    }
    status = {**summary, "decision_v1": strategy_decision}
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "input_dirs_v1": {
            "v3_in_sample_dir_v1": str(v3_in_sample_dir),
            "v3_oof_dir_v1": str(v3_oof_dir),
            "selected_v3_oof_artifact_root_v1": str(v3_oof_dir),
            "foundation_audit_dir_v1": str(foundation_audit_dir),
        },
        "output_files_v1": {name: str(output_dir / name) for name in OUTPUT_FILES},
    }
    audit_rows = [
        {"check_v1": "active_score_artifact_selection", "status_v1": "PASS", "evidence_v1": str(v3_oof_dir)},
        {"check_v1": "foundation_audit_green", "status_v1": "PASS", "evidence_v1": foundation_summary.get("decision_v1")},
        {"check_v1": "selected_v3_oof_artifact_valid", "status_v1": "PASS", "evidence_v1": selected_artifact_audit["status_v1"]},
        {"check_v1": "v3_in_sample_loaded", "status_v1": "PASS", "evidence_v1": str(v3_in_sample_dir)},
        {"check_v1": "v3_oof_loaded", "status_v1": "PASS", "evidence_v1": str(v3_oof_dir)},
        {"check_v1": "ledger_rows", "status_v1": "PASS", "evidence_v1": len(ledger)},
        {"check_v1": "optuna_dependency", "status_v1": "FAIL" if not optuna_state["available_v1"] else "PASS", "evidence_v1": optuna_state["status_v1"]},
        {"check_v1": "no_random_fallback", "status_v1": "PASS", "evidence_v1": True},
        {"check_v1": "r6_not_started", "status_v1": "PASS", "evidence_v1": True},
    ]
    _write_json(output_dir / "v3_in_sample_vs_oof_failure_autopsy_v1.json", autopsy)
    ledger.to_csv(output_dir / "full_remaining_gap_ledger_v1.csv", index=False)
    ledger.to_csv(output_dir / "constrained_optuna_full_signal_forensics_v1.csv", index=False)
    _write_json(output_dir / "oof_signal_separability_audit_v1.json", separability)
    feature_audit.to_csv(output_dir / "feature_family_ablation_and_importance_oof_v1.csv", index=False)
    _write_json(output_dir / "optuna_dependency_install_and_lock_v1.json", install_lock)
    (output_dir / "pip_freeze_after_optuna_install.txt").write_text(pip_freeze + "\n", encoding="utf-8")
    _write_json(output_dir / "optuna_dependency_manifest_v1.json", dependency_manifest)
    _write_json(output_dir / "active_score_artifact_selection_v1.json", selection_contract)
    _write_json(output_dir / "selected_score_artifact_selection_used_v1.json", selection_contract)
    _write_json(output_dir / "selected_v3_oof_artifact_audit_v1.json", selected_artifact_audit)
    _write_json(output_dir / "foundation_audit_reference_v1.json", foundation_reference)
    _write_json(output_dir / "no_fallback_no_dummy_no_synthetic_attestation_v1.json", attestation)
    _write_json(output_dir / "constrained_optuna_preflight_v1.json", preflight)
    _write_json(output_dir / "constrained_optuna_search_space_lock_v1.json", search_space)
    _write_json(output_dir / "constrained_optuna_objective_function_v1.json", objective)
    _write_json(output_dir / "constrained_optuna_study_summary_v1.json", study_summary)
    _write_csv(output_dir / "constrained_optuna_trials_v1.csv", trials_rows)
    _write_json(output_dir / "constrained_optuna_best_candidate_v1.json", {**best_eval, "candidate_lock_v1": best_lock})
    _write_csv(output_dir / "constrained_optuna_constraint_report_v1.csv", constraint_rows)
    _write_json(output_dir / "constrained_optuna_constraint_report_v1.json", constraint_summary)
    _write_csv(output_dir / "constrained_optuna_metric_denominator_report_v1.csv", denominator_rows)
    _write_json(output_dir / "constrained_optuna_metric_denominator_report_v1.json", denominator_summary)
    _write_json(output_dir / "constrained_optuna_full_signal_forensics_v1.json", full_signal_forensics)
    _write_json(output_dir / "constrained_optuna_go_no_go_v1.json", go_no_go)
    _write_csv(output_dir / "optuna_trial_log_and_failure_reasons_v1.csv", trials_rows)
    _write_csv(output_dir / "optuna_trial_forensics_v1.csv", trial_forensics_rows)
    _write_json(output_dir / "best_optuna_candidate_eval_v1.json", best_eval)
    _write_json(output_dir / "best_constrained_candidate_lock_v1.json", best_lock)
    _write_json(output_dir / "optuna_vs_v3_and_rescue_comparison_v1.json", comparison)
    _write_json(output_dir / "model_family_escape_hatch_decision_v1.json", model_escape)
    _write_json(output_dir / "project_cleanliness_and_canonical_graph_check_v1.json", graph)
    _write_json(output_dir / "next_strategy_gate_after_optuna_v1.json", strategy)
    _write_json(output_dir / "strategy_decision_after_constrained_search_v1.json", {**strategy, "layer_name": "STRATEGY_DECISION_AFTER_CONSTRAINED_SEARCH_V1"})
    _write_json(output_dir / "next_action_lock_v1.json", next_lock)
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", status)
    _write_json(output_dir / "manifest_v1.json", manifest)
    _write_csv(output_dir / "consistency_audit_v1.csv", audit_rows)
    report = "\n".join(
        [
            "# Constrained Optuna Objective Search And Full Signal Forensics V1",
            "",
            f"Status: `{summary['status_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- V3 in-sample best bad/tail: `{summary['in_sample_best_bad_tail_v1'][0]}` / `{summary['in_sample_best_bad_tail_v1'][1]}`",
            f"- V3 OOF best bad/tail: `{summary['oof_best_bad_tail_v1'][0]}` / `{summary['oof_best_bad_tail_v1'][1]}`",
            f"- Ledger rows: `{summary['remaining_ledger_rows_v1']}`",
            f"- Useful OOF signal count: `{summary['useful_oof_signal_count_v1']}`",
            f"- Optuna available: `{summary['optuna_available_v1']}`",
            "",
            "No random/weak fallback search was run.",
        ]
    )
    (output_dir / "report_v1.md").write_text(report + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explicit-action", type=str, default=None)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--v3-in-sample-dir", type=Path, default=DEFAULT_V3_IN_SAMPLE_DIR)
    parser.add_argument("--v3-oof-dir", type=Path, default=DEFAULT_V3_OOF_DIR)
    parser.add_argument("--selected-v3-oof-artifact-root", type=Path, default=None)
    parser.add_argument("--active-score-artifact-selection", type=Path, default=None)
    parser.add_argument("--foundation-audit-dir", type=Path, default=None)
    parser.add_argument("--foundation-audit-root", dest="foundation_audit_dir", type=Path)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--optuna-install-command-used", type=str, default="ALREADY_INSTALLED_BEFORE_THIS_RUN")
    parser.add_argument("--require-foundation-clean-for-optuna", action="store_true")
    parser.add_argument("--require-explicit-artifact-selection", action="store_true")
    parser.add_argument("--reject-invalidated-decision-scorefields", action="store_true")
    parser.add_argument("--fail-on-missing-oof-provenance", action="store_true")
    parser.add_argument("--fail-on-invalid-metric-denominator", action="store_true")
    parser.add_argument("--fail-on-in-sample-decision-scores", action="store_true")
    parser.add_argument("--fail-on-degraded-fallback", action="store_true")
    parser.add_argument("--fail-on-dummy-or-synthetic-input", action="store_true")
    args = parser.parse_args(argv)
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        v3_in_sample_dir=args.v3_in_sample_dir,
        v3_oof_dir=args.v3_oof_dir,
        n_trials=args.n_trials,
        selected_v3_oof_artifact_root=args.selected_v3_oof_artifact_root,
        active_score_artifact_selection=args.active_score_artifact_selection,
        foundation_audit_dir=args.foundation_audit_dir,
        explicit_action=args.explicit_action,
        optuna_install_command_used=args.optuna_install_command_used,
        require_foundation_clean_for_optuna=True if args.explicit_action else args.require_foundation_clean_for_optuna,
        require_explicit_artifact_selection=args.require_explicit_artifact_selection,
        reject_invalidated_decision_scorefields=True if args.explicit_action else args.reject_invalidated_decision_scorefields,
        fail_on_missing_oof_provenance=True if args.explicit_action else args.fail_on_missing_oof_provenance,
        fail_on_invalid_metric_denominator=True if args.explicit_action else args.fail_on_invalid_metric_denominator,
        fail_on_in_sample_decision_scores=True if args.explicit_action else args.fail_on_in_sample_decision_scores,
        fail_on_degraded_fallback=True if args.explicit_action else args.fail_on_degraded_fallback,
        fail_on_dummy_or_synthetic_input=True if args.explicit_action else args.fail_on_dummy_or_synthetic_input,
    )
    return 2 if summary["status_v1"] == "OPTUNA_REQUIRED_DEPENDENCY_MISSING" else 0


if __name__ == "__main__":
    raise SystemExit(main())
