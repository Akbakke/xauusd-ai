#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from gx1.scripts.train_r3_entry_label_feature_retrain_v1 import (
    _fit_preprocessor,
    _transform_features,
)
from gx1.scripts.materialize_r4_fullcoverage_policy_recalibration_and_shadow_replay_v1 import (
    _all_run_ids,
    _bool,
    _check_feature_names,
    _json_dumps,
    _load_json,
    _num,
    _policy_metric_row,
    _safe_rate,
    _write_json,
)


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_ENTRY_RETRAIN_WITH_REPAIRED_COVERAGE_AND_SLICE_ROBUSTNESS_V1"
READINESS_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_ENTRY_COVERAGE_REPAIR_READINESS_V1"
R3_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R3_ENTRY_LABEL_FEATURE_RETRAIN_COVERAGE_REPAIRED_V1"
R4_FULLCOVERAGE_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R4_FULLCOVERAGE_POLICY_RECALIBRATION_AND_SHADOW_REPLAY_V1"
R4_MICROTEST_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R4_FIVE_MICROTEST_SHADOW_BAKEOFF_V1"

READINESS_CONTRACT = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_contract_v1.json"
READINESS_AS_OF_TABLE = "shadow_meta_all_trade_review_harvest_r2_entry_as_of_feature_table_v1.parquet"
READINESS_HINDSIGHT_TABLE = "shadow_meta_all_trade_review_harvest_r2_entry_hindsight_label_table_v1.parquet"
REPAIR_AUDIT = "shadow_meta_all_trade_review_entry_coverage_repair_audit_v1.csv"

R3_PREDICTION_VIEW = "shadow_meta_all_trade_review_r3_entry_label_feature_prediction_view_v1.parquet"
R4_POLICY_PREDICTION_VIEW = "shadow_meta_all_trade_review_r4_fullcoverage_policy_prediction_view_v1.parquet"
R4_SUMMARY = "shadow_meta_all_trade_review_r4_fullcoverage_policy_recalibration_summary_v1.json"
MICROTEST_SUMMARY = "shadow_meta_all_trade_review_r4_five_microtest_summary_v1.json"

CONTRACT = "shadow_meta_all_trade_review_r5_entry_retrain_contract_v1.json"
AS_OF_FEATURE_TABLE = "shadow_meta_all_trade_review_r5_entry_as_of_feature_table_v1.parquet"
HINDSIGHT_LABEL_OUTCOME_TABLE = "shadow_meta_all_trade_review_r5_entry_hindsight_label_outcome_table_v1.parquet"
LABEL_AUDIT = "shadow_meta_all_trade_review_r5_entry_label_audit_v1.csv"
FEATURE_AUDIT = "shadow_meta_all_trade_review_r5_entry_feature_audit_v1.csv"
MODEL_METRICS = "shadow_meta_all_trade_review_r5_entry_model_metrics_v1.csv"
MODEL_BAKEOFF = "shadow_meta_all_trade_review_r5_entry_model_bakeoff_v1.csv"
THRESHOLD_CALIBRATION = "shadow_meta_all_trade_review_r5_entry_threshold_calibration_v1.csv"
WALKFORWARD = "shadow_meta_all_trade_review_r5_entry_walkforward_v1.csv"
LOSO = "shadow_meta_all_trade_review_r5_entry_loso_v1.csv"
WINNER_PROTECTION_AUDIT = "shadow_meta_all_trade_review_r5_entry_winner_protection_audit_v1.csv"
HEAD_TO_HEAD = "shadow_meta_all_trade_review_r5_entry_head_to_head_v1.csv"
POLICY_PREDICTION_VIEW = "shadow_meta_all_trade_review_r5_entry_policy_prediction_view_v1.parquet"
DECISION_MATRIX = "shadow_meta_all_trade_review_r5_entry_decision_matrix_v1.csv"
CONSISTENCY_AUDIT = "shadow_meta_all_trade_review_r5_entry_consistency_audit_v1.csv"
SUMMARY = "shadow_meta_all_trade_review_r5_entry_summary_v1.json"
STATUS = "shadow_meta_all_trade_review_r5_entry_status_v1.json"
MANIFEST = "shadow_meta_all_trade_review_r5_entry_manifest_v1.json"
REPORT = "shadow_meta_all_trade_review_r5_entry_report_v1.md"
TOP_LEVEL_SUMMARY = "truth_r5_entry_retrain_with_repaired_coverage_and_slice_robustness_v1.json"

RUN_RE = re.compile(r"^E2E_SANITY_ORDERFIX_(\d{8})_(\d{8})$")


@dataclass(frozen=True)
class LabelSpec:
    label_id: str
    column: str
    policy_role: str
    required_for_policy: bool = True
    noisy: bool = False


LABEL_SPECS: tuple[LabelSpec, ...] = (
    LabelSpec("should_not_take", "r5_label_should_not_take_v1", "BAD_ENTRY_BLOCKER"),
    LabelSpec("immediate_MAE_risk", "r5_label_immediate_mae_risk_v1", "ADVERSE_PATH_RISK"),
    LabelSpec("runner_protect", "r5_label_runner_protect_v1", "WINNER_PROTECTION"),
    LabelSpec("strong_trade_candidate", "r5_label_strong_trade_candidate_v1", "WINNER_PROTECTION"),
    LabelSpec("tail_control_10_50_risk", "r5_label_tail_control_10_50_risk_v1", "TAIL_CONTROL"),
    LabelSpec("take_was_ok", "r5_label_take_was_ok_v1", "ALLOW_BASELINE_PROTECTION"),
    LabelSpec("bad_trade_but_high_runner_risk", "r5_label_bad_trade_but_high_runner_risk_v1", "CONFLICT_LABEL_DIAGNOSTIC", required_for_policy=False, noisy=True),
    LabelSpec("wait_or_delay_advisory", "r5_label_wait_or_delay_advisory_v1", "WAIT_ADVISORY", required_for_policy=False, noisy=True),
)

R5_PROB = {spec.label_id: f"pred__entry_r5_{spec.label_id}__prob_true_v1" for spec in LABEL_SPECS}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _resolve_dir(reports_root: Path, path_arg: str | None, default_name: str, required_file: str) -> Path:
    path = Path(path_arg).expanduser().resolve() if path_arg else reports_root / default_name
    if not path.exists():
        raise FileNotFoundError(f"Required dir does not exist: {path}")
    if not (path / required_file).exists():
        raise FileNotFoundError(f"{path} missing required artifact {required_file}")
    return path


def _default_extension_dir(reports_root: Path) -> Path:
    return reports_root / EXTENSION_NAME


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, artifact_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{artifact_name} missing required columns: {missing}")


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _prob(frame: pd.DataFrame, label_id: str) -> pd.Series:
    return pd.to_numeric(frame.get(R5_PROB[label_id], pd.Series(np.nan, index=frame.index)), errors="coerce")


def _run_sort_key(run_id: str) -> str:
    match = RUN_RE.match(str(run_id))
    return match.group(1) if match else str(run_id)


def _load_feature_names(readiness_dir: Path, asof_df: pd.DataFrame) -> List[str]:
    contract = _load_json(readiness_dir / READINESS_CONTRACT)
    raw = contract.get("as_of_feature_names_v1")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("Readiness contract missing as_of_feature_names_v1")
    feature_names = [str(item) for item in raw]
    _require_columns(asof_df, feature_names, artifact_name=READINESS_AS_OF_TABLE)
    _check_feature_names(feature_names)
    return feature_names


def _feature_family(feature: str) -> str:
    lower = feature.lower()
    if any(token in lower for token in ["swing", "retracement", "ema", "kama", "trend", "impulse", "dist_last"]):
        return "structure_swing_retracement"
    if any(token in lower for token in ["atr", "vol", "range", "bandwidth", "squeeze", "compression"]):
        return "volatility_range"
    if any(token in lower for token in ["close_in_bar", "clv", "body", "wick"]):
        return "close_in_bar_timing"
    if any(token in lower for token in ["session", "hour", "weekday", "minutes"]):
        return "session_time_context"
    if any(token in lower for token in ["window_ret", "up_move", "down_move", "directional_imbalance", "micro_momentum", "micro_acceleration"]):
        return "prior_path_impulse_context"
    if any(token in lower for token in ["spread", "cost"]):
        return "spread_liquidity_cost"
    if any(token in lower for token in ["xgb", "candidate", "tradable", "uncertainty", "path_quality", "margin", "pred_side"]):
        return "entry_model_context"
    return "other_as_of"


def _build_source_frame(
    *,
    readiness_dir: Path,
    r3_dir: Path,
    r4_fullcoverage_dir: Path,
    r4_microtest_dir: Path,
    expected_ledger_count: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], List[str]]:
    asof_df = pd.read_parquet(readiness_dir / READINESS_AS_OF_TABLE)
    hindsight_df = pd.read_parquet(readiness_dir / READINESS_HINDSIGHT_TABLE)
    repair_df = pd.read_csv(readiness_dir / REPAIR_AUDIT)
    r3_df = pd.read_parquet(r3_dir / R3_PREDICTION_VIEW)
    r4_policy_df = pd.read_parquet(r4_fullcoverage_dir / R4_POLICY_PREDICTION_VIEW)
    r4_summary = _load_json(r4_fullcoverage_dir / R4_SUMMARY)
    micro_summary = _load_json(r4_microtest_dir / MICROTEST_SUMMARY)
    feature_names = _load_feature_names(readiness_dir, asof_df)

    for name, frame in [
        (READINESS_AS_OF_TABLE, asof_df),
        (READINESS_HINDSIGHT_TABLE, hindsight_df),
        (R3_PREDICTION_VIEW, r3_df),
        (R4_POLICY_PREDICTION_VIEW, r4_policy_df),
    ]:
        _require_columns(frame, ["candidate_uid"], artifact_name=name)
        if bool(frame["candidate_uid"].astype("string").duplicated().any()):
            raise ValueError(f"{name} requires unique candidate_uid")
    if expected_ledger_count is not None and len(asof_df) != expected_ledger_count:
        raise RuntimeError(f"Locked canonical ledger expected {expected_ledger_count}, observed {len(asof_df)}")
    if len(asof_df) != len(hindsight_df):
        raise RuntimeError("AS_OF and HINDSIGHT tables must have same row count")

    label_cols = [
        "candidate_uid",
        "hindsight_entry_decision_review_v1",
        "hindsight_management_review_v1",
        "baseline_realized_pnl_bps_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "giveback_bps_v1",
        "harvest_capture_ratio_v1",
        "exit_harvest_policy_action_v1",
        "trade_outcome_class",
        "exit_reason",
        "session",
        "vol_regime",
        "trend_regime",
        "support_adverse_first_v1",
        "confirmation_delay_minutes_v1",
        "has_provable_confirmation_v1",
        "teacher_should_wait_entry_v1",
        "label_should_not_take_v1",
        "label_immediate_mae_risk_v1",
        "label_wait_would_have_helped_v1",
        "label_good_mfe_bad_capture_v1",
        "label_low_mfe_low_value_v1",
        "label_strong_trade_candidate_v1",
        "label_direct_take_ok_v1",
    ]
    r3_cols = [
        "candidate_uid",
        "entry_r3_feature_available_v1",
        "entry_r3_shadow_action_v1",
        "entry_r3_shadow_action_source_v1",
    ]
    r3_cols += [column for column in r3_df.columns if column.startswith("pred__entry_r3_") and column.endswith("__prob_true_v1")]
    r4_cols = [
        "candidate_uid",
        "no_entry_fallback_baseline__block_v1",
        "r2_fallback_reference__block_v1",
        "r3_fullcoverage_conservative__block_v1",
        "r4_repaired_selected_reference__block_v1",
        "best_constrained_recalibrated_r4__block_v1",
    ]
    source = (
        asof_df.merge(hindsight_df[[column for column in label_cols if column in hindsight_df.columns]], on="candidate_uid", how="inner", validate="one_to_one")
        .merge(r3_df[[column for column in r3_cols if column in r3_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
        .merge(r4_policy_df[[column for column in r4_cols if column in r4_policy_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
    )
    source["entry_coverage_repair_applied_v1"] = _bool(source, "entry_coverage_repair_applied_v1")
    source["is_repaired_165_v1"] = source["entry_coverage_repair_applied_v1"]
    source["take_was_ok_v1"] = source["hindsight_entry_decision_review_v1"].astype("string").eq("TAKE_WAS_OK")
    source["fifty_plus_mfe_v1"] = _num(source, "peak_mfe_bps_v1").ge(50.0)
    source["hundred_plus_mfe_v1"] = _num(source, "peak_mfe_bps_v1").ge(100.0)
    source["two_hundred_plus_mfe_v1"] = _num(source, "peak_mfe_bps_v1").ge(200.0)
    source["tail_10_50_mfe_v1"] = _num(source, "peak_mfe_bps_v1").between(10.0, 50.0, inclusive="left") & (
        _num(source, "baseline_realized_pnl_bps_v1").le(0.0) | _bool(source, "label_should_not_take_v1")
    )
    source["strongest_winner_path_v1"] = source["two_hundred_plus_mfe_v1"] | (
        _bool(source, "label_strong_trade_candidate_v1")
        & _num(source, "baseline_realized_pnl_bps_v1").gt(0.0)
        & _num(source, "harvest_capture_ratio_v1").ge(0.5)
    )
    if int(_bool(source, "entry_observation_present_v1").sum()) != len(source):
        raise RuntimeError("R5 requires repaired full entry coverage; entry_observation_present_v1 is not full")
    if int(_bool(source, "entry_raw_state_present_v1").sum()) != len(source):
        raise RuntimeError("R5 requires repaired full raw-state coverage; entry_raw_state_present_v1 is not full")
    synthetic_count = int(pd.Series(repair_df.get("synthetic_value_used_v1", pd.Series(dtype=bool))).astype("string").str.lower().eq("true").sum())
    if synthetic_count != 0:
        raise RuntimeError(f"R5 refuses synthetic repair values; observed {synthetic_count}")
    return source, asof_df, hindsight_df, r4_summary, micro_summary, feature_names


def _add_r5_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    should = _bool(out, "label_should_not_take_v1")
    immediate_mae = _bool(out, "label_immediate_mae_risk_v1") | _num(out, "mae_abs_bps_v1").ge(40.0)
    runner_protect = out["take_was_ok_v1"].fillna(False).astype(bool) & (
        out["fifty_plus_mfe_v1"].fillna(False).astype(bool)
        | _bool(out, "label_strong_trade_candidate_v1")
        | out["is_repaired_165_v1"].fillna(False).astype(bool)
    )
    tail_control = out["tail_10_50_mfe_v1"].fillna(False).astype(bool)
    wait_delay = _bool(out, "label_wait_would_have_helped_v1") | _bool(out, "teacher_should_wait_entry_v1") | (
        _bool(out, "support_adverse_first_v1") & _num(out, "confirmation_delay_minutes_v1", default=999.0).between(1.0, 30.0)
    )
    out["r5_label_should_not_take_v1"] = should
    out["r5_label_immediate_mae_risk_v1"] = immediate_mae
    out["r5_label_runner_protect_v1"] = runner_protect
    out["r5_label_strong_trade_candidate_v1"] = _bool(out, "label_strong_trade_candidate_v1")
    out["r5_label_tail_control_10_50_risk_v1"] = tail_control
    out["r5_label_take_was_ok_v1"] = out["take_was_ok_v1"].fillna(False).astype(bool)
    out["r5_label_bad_trade_but_high_runner_risk_v1"] = should & out["fifty_plus_mfe_v1"].fillna(False).astype(bool)
    out["r5_label_wait_or_delay_advisory_v1"] = wait_delay
    out["r5_hindsight_label_contract_v1"] = "R5_HINDSIGHT_SUPERVISION_ONLY_NOT_POLICY_TRUTH_NOT_AS_OF_FEATURES"
    return out


def _label_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    splits = {
        "ALL": pd.Series(True, index=frame.index),
        "TRAIN": _bool(frame, "used_for_training"),
        "VALIDATION": _bool(frame, "used_for_validation"),
        "HOLDOUT": _bool(frame, "used_for_holdout"),
    }
    r4_block = _bool(frame, "best_constrained_recalibrated_r4__block_v1")
    for spec in LABEL_SPECS:
        label = _bool(frame, spec.column)
        for split_name, mask in splits.items():
            sub = label.loc[mask]
            positive_count = int(sub.sum())
            row_count = int(mask.sum())
            if split_name != "ALL":
                rows.append(
                    {
                        "label_id_v1": spec.label_id,
                        "split_v1": split_name,
                        "row_count_v1": row_count,
                        "positive_count_v1": positive_count,
                        "positive_rate_v1": _safe_rate(float(positive_count), float(row_count)),
                        "class_balance_status_v1": "BOTH_CLASSES" if len(set(sub.astype(int).tolist())) == 2 else "SINGLE_CLASS",
                        "label_noise_status_v1": "NOISY_ADVISORY" if spec.noisy else "PRIMARY",
                        "safe_enough_for_training_v1": bool(len(set(sub.astype(int).tolist())) == 2 and (not spec.noisy or positive_count >= 10)),
                        "policy_role_v1": spec.policy_role,
                    }
                )
        rows.append(
            {
                "label_id_v1": spec.label_id,
                "split_v1": "ALL",
                "row_count_v1": int(len(frame)),
                "positive_count_v1": int(label.sum()),
                "positive_rate_v1": _safe_rate(float(label.sum()), float(len(frame))),
                "class_balance_status_v1": "BOTH_CLASSES" if len(set(label.astype(int).tolist())) == 2 else "SINGLE_CLASS",
                "label_noise_status_v1": "NOISY_ADVISORY" if spec.noisy else "PRIMARY",
                "safe_enough_for_training_v1": bool(len(set(label.astype(int).tolist())) == 2 and (not spec.noisy or int(label.sum()) >= 10)),
                "policy_role_v1": spec.policy_role,
                "r4_current_blocks_positive_count_v1": int((r4_block & label).sum()),
                "r4_current_blocks_positive_rate_v1": _safe_rate(float((r4_block & label).sum()), float(label.sum())),
                "r4_current_false_block_overlap_if_protect_label_v1": int((r4_block & label & ~_bool(frame, "r5_label_should_not_take_v1")).sum()),
            }
        )
    return pd.DataFrame(rows)


def _effect_score(feature: pd.Series, positive: pd.Series, negative: pd.Series) -> float | None:
    pos = feature.loc[positive]
    neg = feature.loc[negative]
    if len(pos) < 3 or len(neg) < 3:
        return None
    if pd.api.types.is_numeric_dtype(feature) or pd.api.types.is_bool_dtype(feature):
        pos_n = pd.to_numeric(pos, errors="coerce").dropna()
        neg_n = pd.to_numeric(neg, errors="coerce").dropna()
        if len(pos_n) < 3 or len(neg_n) < 3:
            return None
        denom = float(np.nanstd(pd.concat([pos_n, neg_n]).to_numpy(dtype=float)))
        if denom == 0.0 or not math.isfinite(denom):
            return 0.0
        return abs(float(pos_n.mean()) - float(neg_n.mean())) / denom
    rates = feature.astype("string").fillna("__NA__").groupby(positive).value_counts(normalize=True)
    try:
        categories = feature.astype("string").fillna("__NA__")
        pos_dist = categories.loc[positive].value_counts(normalize=True)
        neg_dist = categories.loc[negative].value_counts(normalize=True)
        all_keys = set(pos_dist.index).union(set(neg_dist.index))
        return 0.5 * sum(abs(float(pos_dist.get(key, 0.0)) - float(neg_dist.get(key, 0.0))) for key in all_keys)
    except Exception:
        return _safe_float(rates.max())


def _feature_audit(frame: pd.DataFrame, feature_names: Sequence[str], reports_root: Path, *, batch_weeks: int) -> pd.DataFrame:
    run_ids = _all_run_ids(reports_root, frame)
    batch_lookup: Dict[str, str] = {}
    for batch_index, start in enumerate(range(0, len(run_ids), batch_weeks), start=1):
        for run_id in run_ids[start : start + batch_weeks]:
            batch_lookup[run_id] = f"BATCH_{batch_index:02d}"
    batch = frame["run_id"].astype("string").map(batch_lookup).fillna("UNKNOWN")
    r4_block = _bool(frame, "best_constrained_recalibrated_r4__block_v1")
    should = _bool(frame, "r5_label_should_not_take_v1")
    contrasts: Dict[str, tuple[pd.Series, pd.Series]] = {
        "BAD_TRADE_VS_TAKE_WAS_OK": (should, _bool(frame, "r5_label_take_was_ok_v1")),
        "IMMEDIATE_MAE_VS_CLEAN_ENTRY": (_bool(frame, "r5_label_immediate_mae_risk_v1"), ~_bool(frame, "r5_label_immediate_mae_risk_v1")),
        "TAIL_10_50_RISK_VS_50_PLUS_RUNNER": (_bool(frame, "r5_label_tail_control_10_50_risk_v1"), frame["fifty_plus_mfe_v1"].fillna(False).astype(bool)),
        "BATCH_04_05_R4_FALSE_BLOCK_VS_TRUE_BAD_BLOCK": (
            r4_block & ~should & batch.isin(["BATCH_04", "BATCH_05"]),
            r4_block & should & batch.isin(["BATCH_04", "BATCH_05"]),
        ),
        "REPAIRED_165_RUNNERS_VS_SHOULD_NOT_TAKE": (frame["is_repaired_165_v1"].fillna(False).astype(bool), should),
    }
    rows: list[dict[str, Any]] = []
    for contrast_name, (positive, negative) in contrasts.items():
        for family in sorted({_feature_family(feature) for feature in feature_names}):
            family_features = [feature for feature in feature_names if _feature_family(feature) == family]
            scored: list[tuple[str, float]] = []
            for feature in family_features:
                score = _effect_score(frame[feature], positive, negative)
                if score is not None and math.isfinite(score):
                    scored.append((feature, float(score)))
            scored = sorted(scored, key=lambda item: item[1], reverse=True)
            rows.append(
                {
                    "contrast_name_v1": contrast_name,
                    "feature_family_v1": family,
                    "positive_count_v1": int(positive.sum()),
                    "negative_count_v1": int(negative.sum()),
                    "feature_count_v1": int(len(family_features)),
                    "scored_feature_count_v1": int(len(scored)),
                    "mean_top5_effect_score_v1": _safe_float(np.mean([score for _, score in scored[:5]])) if scored else None,
                    "max_effect_score_v1": scored[0][1] if scored else None,
                    "top_features_json_v1": _json_dumps([{"feature": feature, "score": score} for feature, score in scored[:10]]),
                    "as_of_only_v1": True,
                }
            )
    return pd.DataFrame(rows).sort_values(["contrast_name_v1", "mean_top5_effect_score_v1"], ascending=[True, False])


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    valid = np.isfinite(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]
    if len(y_true) == 0:
        return {"row_count_v1": 0}
    y_pred = (y_prob >= threshold).astype(int)
    record: Dict[str, Any] = {
        "row_count_v1": int(len(y_true)),
        "positive_count_v1": int(y_true.sum()),
        "pred_positive_count_v1": int(y_pred.sum()),
        "balanced_accuracy_v1": None,
        "precision_true_v1": None,
        "recall_true_v1": None,
        "roc_auc_v1": None,
        "brier_score_v1": None,
        "confusion_matrix_json_v1": _json_dumps(confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()),
    }
    if len(set(y_true.tolist())) >= 2:
        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        record.update(
            {
                "balanced_accuracy_v1": float(balanced_accuracy_score(y_true, y_pred)),
                "precision_true_v1": float(precision[1]),
                "recall_true_v1": float(recall[1]),
                "roc_auc_v1": float(roc_auc_score(y_true, y_prob)),
                "brier_score_v1": float(brier_score_loss(y_true, y_prob)),
            }
        )
    return record


def _train_head(
    *,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    spec: LabelSpec,
    train_mask: pd.Series,
    validation_mask: pd.Series,
    output_dir: Path | None,
    model_tag: str,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.Series, pd.DataFrame, Dict[str, Any]]:
    y_all = _bool(frame, spec.column).astype(int)
    train_mask = train_mask.reindex(frame.index).fillna(False).astype(bool)
    validation_mask = validation_mask.reindex(frame.index).fillna(False).astype(bool)
    if int(train_mask.sum()) < 20:
        raise ValueError(f"{spec.label_id} has too few training rows")
    if len(set(y_all.loc[train_mask].tolist())) < 2:
        raise ValueError(f"{spec.label_id} train split requires both classes")
    if int(validation_mask.sum()) == 0 or len(set(y_all.loc[validation_mask].tolist())) < 2:
        validation_mask = train_mask

    preprocessor = _fit_preprocessor(frame.loc[train_mask, feature_names], feature_names)
    x_train = _transform_features(preprocessor, frame.loc[train_mask, feature_names])
    x_val = _transform_features(preprocessor, frame.loc[validation_mask, feature_names])
    y_train = y_all.loc[train_mask].to_numpy(dtype=int)
    y_val = y_all.loc[validation_mask].to_numpy(dtype=int)
    weights = compute_sample_weight("balanced", y_train)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=4.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=8.0,
        reg_alpha=0.35,
        tree_method="hist",
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=0,
    )
    model.fit(x_train, y_train, sample_weight=weights, eval_set=[(x_val, y_val)], verbose=False)
    x_all = _transform_features(preprocessor, frame[feature_names])
    probs = pd.Series(model.predict_proba(x_all)[:, 1], index=frame.index, dtype="float64")

    rows: list[dict[str, Any]] = []
    split_masks = {
        "TRAIN": train_mask,
        "VALIDATION": validation_mask,
        "HOLDOUT_OR_OTHER": ~(train_mask | validation_mask),
        "ALL": pd.Series(True, index=frame.index),
    }
    for split_name, mask in split_masks.items():
        if int(mask.sum()) == 0:
            continue
        metrics = _classification_metrics(y_all.loc[mask].to_numpy(dtype=int), probs.loc[mask].to_numpy(dtype=float))
        metrics.update(
            {
                "model_tag_v1": model_tag,
                "label_id_v1": spec.label_id,
                "target_column_v1": spec.column,
                "split_v1": split_name,
                "policy_role_v1": spec.policy_role,
                "label_noise_status_v1": "NOISY_ADVISORY" if spec.noisy else "PRIMARY",
            }
        )
        rows.append(metrics)
    metadata = {
        "model_tag_v1": model_tag,
        "label_id_v1": spec.label_id,
        "target_column_v1": spec.column,
        "feature_count_v1": int(len(feature_names)),
        "transformed_feature_count_v1": int(x_train.shape[1]),
        "train_rows_v1": int(train_mask.sum()),
        "validation_rows_v1": int(validation_mask.sum()),
        "best_iteration_v1": getattr(model, "best_iteration", None),
        "best_score_v1": _safe_float(getattr(model, "best_score", None)),
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    if output_dir is not None:
        model_dir = output_dir / "models" / model_tag / spec.label_id
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")
        joblib.dump(preprocessor, model_dir / "feature_preprocessor.joblib")
        _write_json(model_dir / "metadata.json", metadata)
    return probs, pd.DataFrame(rows), metadata


def _train_heads(
    *,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    train_mask: pd.Series,
    validation_mask: pd.Series,
    output_dir: Path | None,
    model_tag: str,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    pred = frame[["candidate_uid"]].copy()
    metrics: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {}
    for index, spec in enumerate(LABEL_SPECS):
        probs, metric_df, meta = _train_head(
            frame=frame,
            feature_names=feature_names,
            spec=spec,
            train_mask=train_mask,
            validation_mask=validation_mask,
            output_dir=output_dir,
            model_tag=model_tag,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            learning_rate=learning_rate,
            max_depth=max_depth,
            seed=seed + index,
            n_jobs=n_jobs,
        )
        pred[R5_PROB[spec.label_id]] = probs.to_numpy(dtype=float)
        pred[f"pred__entry_r5_{spec.label_id}__label_v1"] = np.where(probs.ge(0.5), "TRUE", "FALSE")
        metrics.append(metric_df)
        metadata[spec.label_id] = meta
    return pred, pd.concat(metrics, ignore_index=True), metadata


def _policy_masks(frame: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
    p_should = _prob(frame, "should_not_take")
    p_mae = _prob(frame, "immediate_MAE_risk")
    p_runner = _prob(frame, "runner_protect")
    p_strong = _prob(frame, "strong_trade_candidate")
    p_tail = _prob(frame, "tail_control_10_50_risk")
    p_take = _prob(frame, "take_was_ok")
    p_high_runner_bad = _prob(frame, "bad_trade_but_high_runner_risk")

    t_should = float(params.get("should_not_take_threshold_v1", 0.60))
    t_mae = float(params.get("immediate_mae_threshold_v1", 0.70))
    t_tail = float(params.get("tail_control_threshold_v1", 0.60))
    t_runner = float(params.get("runner_protect_threshold_v1", 0.60))
    t_strong = float(params.get("strong_protect_threshold_v1", 0.55))
    t_take = float(params.get("take_ok_protect_threshold_v1", 0.75))
    t_override = float(params.get("bad_risk_override_threshold_v1", 0.88))
    take_ceiling = float(params.get("take_ok_block_ceiling_v1", 0.55))

    protect = p_runner.ge(t_runner).fillna(False) | p_strong.ge(t_strong).fillna(False) | p_take.ge(t_take).fillna(False)
    high_bad_override = (
        p_should.ge(t_override).fillna(False)
        & p_mae.ge(max(t_mae, 0.75)).fillna(False)
        & p_high_runner_bad.lt(0.70).fillna(True)
    )
    weak_take = p_take.lt(take_ceiling).fillna(False)
    should_signal = p_should.ge(t_should).fillna(False)
    mae_signal = p_mae.ge(t_mae).fillna(False) & weak_take
    tail_signal = p_tail.ge(t_tail).fillna(False) & weak_take
    combined = should_signal | mae_signal | tail_signal
    winner_first = (combined & ~protect) | high_bad_override
    r2 = _bool(frame, "r2_fallback_reference__block_v1")
    r4 = _bool(frame, "best_constrained_recalibrated_r4__block_v1")
    return {
        "R5_CONSERVATIVE_BLOCKER": (should_signal & ~protect).astype(bool),
        "R5_IMMEDIATE_MAE_BLOCKER": (mae_signal & ~protect).astype(bool),
        "R5_RUNNER_PROTECTOR_ONLY": pd.Series(False, index=frame.index, dtype=bool),
        "R5_TAIL_CONTROL_HEAD": (tail_signal & ~protect).astype(bool),
        "R5_COMBINED_STACK": (combined & ~protect).astype(bool),
        "R5_R2_PRESERVATION_AWARE_STACK": ((r2 & ~protect) | (combined & ~protect) | high_bad_override).astype(bool),
        "R5_R4_CURRENT_COMPATIBLE_STACK": ((r4 & ~protect) | (combined & ~protect) | high_bad_override).astype(bool),
        "R5_WINNER_FIRST_PROTECT_THEN_BLOCK": winner_first.astype(bool),
        "R5_SOFT_PROTECT_BAD_OVERRIDE": ((combined & ~(protect & ~high_bad_override)) | high_bad_override).astype(bool),
    }


def _slice_masks(reports_root: Path, frame: pd.DataFrame, *, batch_weeks: int) -> list[dict[str, Any]]:
    run_ids = _all_run_ids(reports_root, frame)
    slices: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(run_ids), batch_weeks), start=1):
        batch_run_ids = run_ids[start : start + batch_weeks]
        mask = frame["run_id"].astype("string").isin(batch_run_ids)
        slices.append(
            {
                "scope_v1": f"BATCH_{batch_index:02d}",
                "batch_index_v1": int(batch_index),
                "run_count_v1": int(len(batch_run_ids)),
                "run_start_v1": batch_run_ids[0] if batch_run_ids else None,
                "run_end_v1": batch_run_ids[-1] if batch_run_ids else None,
                "mask_v1": mask,
            }
        )
    return slices


def _reference_masks(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    return {
        "NO_ENTRY_FALLBACK_BASELINE": pd.Series(False, index=frame.index, dtype=bool),
        "R2_FALLBACK_REFERENCE": _bool(frame, "r2_fallback_reference__block_v1"),
        "R3_FULLCOVERAGE_CONSERVATIVE": _bool(frame, "r3_fullcoverage_conservative__block_v1"),
        "R4_CURRENT_REFERENCE": _bool(frame, "best_constrained_recalibrated_r4__block_v1"),
    }


def _global_safety(row: Dict[str, Any], r4_ref: Dict[str, Any]) -> tuple[bool, str]:
    failures: list[str] = []
    precision = _safe_float(row.get("should_not_take_precision_v1"))
    if int(row.get("repaired_165_block_count_v1") or 0) != 0:
        failures.append("repaired_165_block_count_v1!=0")
    if int(row.get("two_hundred_plus_mfe_block_count_v1") or 0) > 1:
        failures.append("two_hundred_plus_mfe_block_count_v1>1")
    if int(row.get("fifty_plus_mfe_block_count_v1") or 0) > 3:
        failures.append("fifty_plus_mfe_block_count_v1>3")
    if int(row.get("strong_trade_false_block_count_v1") or 0) > 2:
        failures.append("strong_trade_false_block_count_v1>2")
    hundred_limit = int(r4_ref.get("hundred_plus_mfe_block_count_v1") or 0)
    should_gain = int(row.get("should_not_take_block_count_v1") or 0) - int(r4_ref.get("should_not_take_block_count_v1") or 0)
    if int(row.get("hundred_plus_mfe_block_count_v1") or 0) > hundred_limit and should_gain < 10:
        failures.append("hundred_plus_mfe_block_count_v1_increase_without_clear_gain")
    if int(row.get("strongest_winner_path_block_count_v1") or 0) > int(r4_ref.get("strongest_winner_path_block_count_v1") or 0):
        failures.append("strongest_winner_path_damage_increase")
    if precision is None or precision < 0.85:
        failures.append("precision<0.85")
    return not failures, ",".join(failures)


def _slice_safety(metric: Dict[str, Any]) -> tuple[bool, str]:
    failures: list[str] = []
    precision = _safe_float(metric.get("should_not_take_precision_v1"))
    block_count = int(metric.get("block_count_v1") or 0)
    if int(metric.get("repaired_165_block_count_v1") or 0) != 0:
        failures.append("repaired_165_block_count_v1!=0")
    if int(metric.get("two_hundred_plus_mfe_block_count_v1") or 0) > 1:
        failures.append("two_hundred_plus_mfe_block_count_v1>1")
    if int(metric.get("fifty_plus_mfe_block_count_v1") or 0) > 3:
        failures.append("fifty_plus_mfe_block_count_v1>3")
    if int(metric.get("strong_trade_false_block_count_v1") or 0) > 2:
        failures.append("strong_trade_false_block_count_v1>2")
    if block_count > 0 and (precision is None or precision < 0.85):
        failures.append("precision<0.85")
    return not failures, ",".join(failures)


def _evaluate_policy_candidates(
    *,
    reports_root: Path,
    frame: pd.DataFrame,
    r4_ref: Dict[str, Any],
    batch_weeks: int,
    compact: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    walk_rows: list[dict[str, Any]] = []
    reference_masks = _reference_masks(frame)
    for policy_name, mask in reference_masks.items():
        row = _policy_metric_row(policy_name, "ALL", frame, mask, thresholds={"reference_policy_v1": policy_name})
        pass_global, failures = _global_safety(row, r4_ref)
        row.update(
            {
                "policy_stack_family_v1": "REFERENCE",
                "candidate_type_v1": "REFERENCE",
                "global_safety_pass_v1": pass_global,
                "global_safety_failure_reasons_v1": failures,
                "batch_04_pass_v1": None,
                "batch_05_pass_v1": None,
                "r5_candidate_v1": False,
            }
        )
        rows.append(row)

    should_grid = [0.80] if compact else [0.55, 0.65, 0.80]
    mae_grid = [0.85] if compact else [0.80, 0.90]
    tail_grid = [0.80]
    runner_grid = [0.70] if compact else [0.65, 0.80]
    strong_grid = [0.65]
    take_grid = [0.90] if compact else [0.85, 0.95]
    take_ceiling_grid = [0.45]
    stack_families = [
        "R5_RUNNER_PROTECTOR_ONLY",
        "R5_CONSERVATIVE_BLOCKER",
        "R5_IMMEDIATE_MAE_BLOCKER",
        "R5_TAIL_CONTROL_HEAD",
        "R5_COMBINED_STACK",
        "R5_R2_PRESERVATION_AWARE_STACK",
        "R5_R4_CURRENT_COMPATIBLE_STACK",
        "R5_WINNER_FIRST_PROTECT_THEN_BLOCK",
        "R5_SOFT_PROTECT_BAD_OVERRIDE",
    ]
    slices = _slice_masks(reports_root, frame, batch_weeks=batch_weeks)
    for t_should in should_grid:
        for t_mae in mae_grid:
            for t_tail in tail_grid:
                for t_runner in runner_grid:
                    for t_strong in strong_grid:
                        for t_take in take_grid:
                            for take_ceiling in take_ceiling_grid:
                                params = {
                                    "should_not_take_threshold_v1": t_should,
                                    "immediate_mae_threshold_v1": t_mae,
                                    "tail_control_threshold_v1": t_tail,
                                    "runner_protect_threshold_v1": t_runner,
                                    "strong_protect_threshold_v1": t_strong,
                                    "take_ok_protect_threshold_v1": t_take,
                                    "take_ok_block_ceiling_v1": take_ceiling,
                                    "bad_risk_override_threshold_v1": 0.88,
                                }
                                masks = _policy_masks(frame, params)
                                for stack_name in stack_families:
                                    mask = masks[stack_name]
                                    row = _policy_metric_row(stack_name, "ALL", frame, mask, thresholds=params)
                                    pass_global, failures = _global_safety(row, r4_ref)
                                    batch_flags: dict[str, Any] = {}
                                    for slice_info in slices:
                                        if slice_info["scope_v1"] not in {"BATCH_04", "BATCH_05"}:
                                            continue
                                        smask = slice_info["mask_v1"]
                                        metric = _policy_metric_row(stack_name, str(slice_info["scope_v1"]), frame.loc[smask].copy(), mask.loc[smask], thresholds=params)
                                        spass, sfail = _slice_safety(metric)
                                        batch_flags[f"{str(slice_info['scope_v1']).lower()}_pass_v1"] = spass
                                        batch_flags[f"{str(slice_info['scope_v1']).lower()}_failure_reasons_v1"] = sfail
                                    should_gain = int(row["should_not_take_block_count_v1"]) - int(r4_ref["should_not_take_block_count_v1"])
                                    precision = _safe_float(row.get("should_not_take_precision_v1")) or 0.0
                                    row.update(
                                        {
                                            "policy_stack_family_v1": stack_name,
                                            "candidate_type_v1": "R5_POLICY_STACK",
                                            "global_safety_pass_v1": pass_global,
                                            "global_safety_failure_reasons_v1": failures,
                                            "batch_04_pass_v1": bool(batch_flags.get("batch_04_pass_v1", True)),
                                            "batch_05_pass_v1": bool(batch_flags.get("batch_05_pass_v1", True)),
                                            "batch_04_failure_reasons_v1": batch_flags.get("batch_04_failure_reasons_v1", ""),
                                            "batch_05_failure_reasons_v1": batch_flags.get("batch_05_failure_reasons_v1", ""),
                                            "r5_candidate_v1": True,
                                            "beats_r4_bad_recall_v1": should_gain > 0,
                                            "should_not_blocks_gain_vs_r4_v1": should_gain,
                                            "safety_failure_count_v1": int(
                                                len([item for item in str(failures).split(",") if item])
                                                + len([item for item in str(batch_flags.get("batch_04_failure_reasons_v1", "")).split(",") if item])
                                                + len([item for item in str(batch_flags.get("batch_05_failure_reasons_v1", "")).split(",") if item])
                                            ),
                                            "r5_selection_score_v1": (
                                                float(row["should_not_take_block_count_v1"]) * 1.0
                                                + float(row["tail_10_50_help_count_v1"]) * 0.20
                                                + precision * 10.0
                                                - float(row["block_count_v1"]) * 0.03
                                                - float(row["take_was_ok_block_count_v1"]) * 0.10
                                            ),
                                        }
                                    )
                                    if not (row["global_safety_pass_v1"] and row["batch_04_pass_v1"] and row["batch_05_pass_v1"]):
                                        row["r5_selection_score_v1"] -= 1000.0
                                    rows.append(row)
    calibration = pd.DataFrame(rows)
    r5 = calibration[calibration["r5_candidate_v1"].fillna(False).astype(bool)].copy()
    viable = r5[r5["global_safety_pass_v1"].eq(True) & r5["batch_04_pass_v1"].eq(True) & r5["batch_05_pass_v1"].eq(True)].copy()
    if viable.empty:
        best = r5.sort_values(
            ["safety_failure_count_v1", "should_not_take_precision_v1", "should_not_take_block_count_v1", "r5_selection_score_v1"],
            ascending=[True, False, False, False],
        ).iloc[0].to_dict()
    else:
        best = viable.sort_values(
            ["beats_r4_bad_recall_v1", "r5_selection_score_v1", "should_not_take_block_count_v1", "should_not_take_precision_v1"],
            ascending=[False, False, False, False],
        ).iloc[0].to_dict()
    calibration["selected_r5_candidate_v1"] = (
        calibration["policy_name_v1"].astype("string").eq(str(best["policy_name_v1"]))
        & calibration["thresholds_json_v1"].astype("string").eq(str(best["thresholds_json_v1"]))
        & calibration["r5_candidate_v1"].fillna(False).astype(bool)
    )
    best_params = json.loads(str(best["thresholds_json_v1"]))
    best_mask = _policy_masks(frame, best_params)[str(best["policy_name_v1"])]
    for policy_name, mask in {**reference_masks, "R5_SELECTED_CANDIDATE": best_mask}.items():
        for slice_info in slices:
            smask = slice_info["mask_v1"]
            metric = _policy_metric_row(policy_name, str(slice_info["scope_v1"]), frame.loc[smask].copy(), mask.loc[smask], thresholds={"walkforward_v1": True})
            spass, sfail = _slice_safety(metric)
            metric.update({key: value for key, value in slice_info.items() if key != "mask_v1"})
            metric["slice_safety_pass_v1"] = spass
            metric["slice_safety_failure_reasons_v1"] = sfail
            walk_rows.append(metric)
    return calibration, pd.DataFrame(walk_rows), best


def _model_bakeoff(calibration_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for family, group in calibration_df.groupby("policy_stack_family_v1", dropna=False):
        selected = group.sort_values(
            ["global_safety_pass_v1", "batch_04_pass_v1", "batch_05_pass_v1", "r5_selection_score_v1", "should_not_take_block_count_v1"],
            ascending=[False, False, False, False, False],
        ).head(1)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["global_safety_pass_v1", "batch_04_pass_v1", "batch_05_pass_v1", "r5_selection_score_v1"],
        ascending=[False, False, False, False],
    )


def _head_to_head(frame: pd.DataFrame, selected_record: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_params = json.loads(str(selected_record["thresholds_json_v1"]))
    selected_mask = _policy_masks(frame, selected_params)[str(selected_record["policy_name_v1"])]
    policies = {**_reference_masks(frame), "R5_SELECTED_CANDIDATE": selected_mask}
    scopes = {
        "ALL_1971": pd.Series(True, index=frame.index),
        "SHOULD_NOT_TAKE_CLASS": _bool(frame, "r5_label_should_not_take_v1"),
        "TAKE_WAS_OK_CLASS": _bool(frame, "r5_label_take_was_ok_v1"),
        "REPAIRED_165": frame["is_repaired_165_v1"].fillna(False).astype(bool),
        "FIFTY_PLUS_MFE_RUNNERS": frame["fifty_plus_mfe_v1"].fillna(False).astype(bool),
        "HUNDRED_PLUS_MFE_RUNNERS": frame["hundred_plus_mfe_v1"].fillna(False).astype(bool),
        "TWO_HUNDRED_PLUS_MFE_RUNNERS": frame["two_hundred_plus_mfe_v1"].fillna(False).astype(bool),
        "STRONGEST_WINNER_PATH": frame["strongest_winner_path_v1"].fillna(False).astype(bool),
        "TAIL_10_50_MFE_POCKET": frame["tail_10_50_mfe_v1"].fillna(False).astype(bool),
    }
    rows: list[dict[str, Any]] = []
    prediction = frame[
        [
            "run_id",
            "candidate_uid",
            "trade_uid",
            "trade_id",
            "decision_timestamp",
            "is_repaired_165_v1",
            "r5_label_should_not_take_v1",
            "r5_label_runner_protect_v1",
            "r5_label_take_was_ok_v1",
            "peak_mfe_bps_v1",
            "mae_abs_bps_v1",
            "giveback_bps_v1",
            "baseline_realized_pnl_bps_v1",
            *R5_PROB.values(),
        ]
    ].copy()
    for policy_name, mask in policies.items():
        prediction[f"{policy_name.lower()}__block_v1"] = mask.to_numpy(dtype=bool)
        for scope_name, scope_mask in scopes.items():
            rows.append(_policy_metric_row(policy_name, scope_name, frame.loc[scope_mask].copy(), mask.loc[scope_mask], thresholds={"head_to_head_v1": True}))
    return pd.DataFrame(rows), prediction


def _select_internal_validation_mask(frame: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.Series, pd.Series]:
    train_indices = frame.index[train_mask].tolist()
    if len(train_indices) < 50:
        return train_mask, train_mask
    cut = int(len(train_indices) * 0.8)
    inner_train = pd.Series(False, index=frame.index)
    inner_validation = pd.Series(False, index=frame.index)
    inner_train.loc[train_indices[:cut]] = True
    inner_validation.loc[train_indices[cut:]] = True
    return inner_train, inner_validation


def _loso(
    *,
    reports_root: Path,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    r4_ref: Dict[str, Any],
    batch_weeks: int,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for slice_info in _slice_masks(reports_root, frame, batch_weeks=batch_weeks):
        holdout = slice_info["mask_v1"]
        train_all = ~holdout
        inner_train, inner_validation = _select_internal_validation_mask(frame, train_all)
        pred, _, _ = _train_heads(
            frame=frame,
            feature_names=feature_names,
            train_mask=inner_train,
            validation_mask=inner_validation,
            output_dir=None,
            model_tag=f"loso_{slice_info['scope_v1'].lower()}",
            n_estimators=max(100, min(n_estimators, 700)),
            early_stopping_rounds=max(20, min(early_stopping_rounds, 60)),
            learning_rate=learning_rate,
            max_depth=max_depth,
            seed=seed + int(slice_info["batch_index_v1"]) * 100,
            n_jobs=n_jobs,
        )
        fold_frame = frame.drop(columns=[column for column in R5_PROB.values() if column in frame.columns], errors="ignore").merge(pred, on="candidate_uid", how="left", validate="one_to_one")
        calibration, _, best = _evaluate_policy_candidates(
            reports_root=reports_root,
            frame=fold_frame.loc[train_all].copy(),
            r4_ref=r4_ref,
            batch_weeks=batch_weeks,
            compact=True,
        )
        params = json.loads(str(best["thresholds_json_v1"]))
        selected_policy = str(best["policy_name_v1"])
        selected_mask_all = _policy_masks(fold_frame, params)[selected_policy]
        holdout_metric = _policy_metric_row(selected_policy, str(slice_info["scope_v1"]), fold_frame.loc[holdout].copy(), selected_mask_all.loc[holdout], thresholds=params)
        spass, sfail = _slice_safety(holdout_metric)
        holdout_metric.update(
            {
                "holdout_slice_v1": slice_info["scope_v1"],
                "run_count_v1": slice_info["run_count_v1"],
                "run_start_v1": slice_info["run_start_v1"],
                "run_end_v1": slice_info["run_end_v1"],
                "selected_policy_name_v1": selected_policy,
                "train_candidate_count_v1": int(len(calibration)),
                "train_selected_should_not_take_block_count_v1": int(best.get("should_not_take_block_count_v1") or 0),
                "train_selected_precision_v1": best.get("should_not_take_precision_v1"),
                "holdout_safety_pass_v1": spass,
                "holdout_safety_failure_reasons_v1": sfail,
            }
        )
        rows.append(holdout_metric)
    return pd.DataFrame(rows)


def _winner_protection_audit(calibration_df: pd.DataFrame) -> pd.DataFrame:
    subset = calibration_df[
        calibration_df["policy_stack_family_v1"].astype("string").isin(
            [
                "R5_RUNNER_PROTECTOR_ONLY",
                "R5_R4_CURRENT_COMPATIBLE_STACK",
                "R5_WINNER_FIRST_PROTECT_THEN_BLOCK",
                "R5_SOFT_PROTECT_BAD_OVERRIDE",
            ]
        )
    ].copy()
    if subset.empty:
        return subset
    return subset.sort_values(
        ["global_safety_pass_v1", "fifty_plus_mfe_block_count_v1", "two_hundred_plus_mfe_block_count_v1", "should_not_take_block_count_v1"],
        ascending=[False, True, True, False],
    ).head(200)


def _decision_matrix(
    *,
    selected: Dict[str, Any],
    head_to_head_df: pd.DataFrame,
    loso_df: pd.DataFrame,
    r4_summary: Dict[str, Any],
    micro_summary: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    r5_all = head_to_head_df[(head_to_head_df["policy_name_v1"].eq("R5_SELECTED_CANDIDATE")) & (head_to_head_df["scope_v1"].eq("ALL_1971"))].iloc[0].to_dict()
    r4_all = head_to_head_df[(head_to_head_df["policy_name_v1"].eq("R4_CURRENT_REFERENCE")) & (head_to_head_df["scope_v1"].eq("ALL_1971"))].iloc[0].to_dict()
    r2_all = head_to_head_df[(head_to_head_df["policy_name_v1"].eq("R2_FALLBACK_REFERENCE")) & (head_to_head_df["scope_v1"].eq("ALL_1971"))].iloc[0].to_dict()
    beats_r4 = int(r5_all["should_not_take_block_count_v1"]) > int(r4_all["should_not_take_block_count_v1"])
    no_more_winner_damage = (
        int(r5_all["fifty_plus_mfe_block_count_v1"]) <= int(r4_all["fifty_plus_mfe_block_count_v1"])
        and int(r5_all["hundred_plus_mfe_block_count_v1"]) <= int(r4_all["hundred_plus_mfe_block_count_v1"])
        and int(r5_all["two_hundred_plus_mfe_block_count_v1"]) <= int(r4_all["two_hundred_plus_mfe_block_count_v1"])
        and int(r5_all["strong_trade_false_block_count_v1"]) <= int(r4_all["strong_trade_false_block_count_v1"])
        and int(r5_all["repaired_165_block_count_v1"]) == 0
    )
    loso_pass = bool(loso_df["holdout_safety_pass_v1"].fillna(False).all()) if not loso_df.empty else False
    failed_loso = loso_df.loc[~loso_df["holdout_safety_pass_v1"].fillna(False), "holdout_slice_v1"].astype("string").tolist() if not loso_df.empty else []
    if beats_r4 and no_more_winner_damage and loso_pass:
        recommendation = "R5_SHADOW_REPLAY_CANDIDATE"
    elif beats_r4 and no_more_winner_damage:
        recommendation = "R5_RETRAIN_MORE"
    elif int(r4_all["should_not_take_block_count_v1"]) >= int(r2_all["should_not_take_block_count_v1"]):
        recommendation = "KEEP_R4_CURRENT_REFERENCE"
    else:
        recommendation = "KEEP_R2_FALLBACK"
    rows = [
        {
            "decision_key_v1": "R5_SHADOW_REPLAY_CANDIDATE",
            "status_v1": "PASS" if recommendation == "R5_SHADOW_REPLAY_CANDIDATE" else "NOT_MET",
            "reason_v1": "Requires beating R4 bad-trade recall, no extra winner damage, and LOSO safety pass.",
        },
        {
            "decision_key_v1": "R5_RETRAIN_MORE",
            "status_v1": "PASS" if recommendation == "R5_RETRAIN_MORE" else "NEXT_OPTION",
            "reason_v1": "Use when R5 improves offline recall but slice robustness still fails.",
        },
        {
            "decision_key_v1": "KEEP_R4_CURRENT_REFERENCE",
            "status_v1": "PASS" if recommendation == "KEEP_R4_CURRENT_REFERENCE" else "NOT_PRIMARY",
            "reason_v1": "Use when R5 does not safely beat R4.",
        },
        {
            "decision_key_v1": "KEEP_R2_FALLBACK",
            "status_v1": "PASS" if recommendation == "KEEP_R2_FALLBACK" else "NOT_PRIMARY",
            "reason_v1": "Use only if R4/R5 both lose to R2 safety tradeoff.",
        },
        {
            "decision_key_v1": "ENTRY_NOT_READY_FOR_FALLBACK_EXPANSION",
            "status_v1": "PASS_FOR_LIVE_GATE_ONLY",
            "reason_v1": "No output here is live-gate promoted.",
        },
    ]
    summary = {
        "recommended_next_step_v1": recommendation,
        "r5_beats_r4_bad_recall_v1": bool(beats_r4),
        "r5_no_more_winner_damage_vs_r4_v1": bool(no_more_winner_damage),
        "r5_loso_pass_v1": bool(loso_pass),
        "r5_failed_loso_slices_v1": failed_loso,
        "r5_should_not_blocks_v1": int(r5_all["should_not_take_block_count_v1"]),
        "r4_should_not_blocks_v1": int(r4_all["should_not_take_block_count_v1"]),
        "r2_should_not_blocks_v1": int(r2_all["should_not_take_block_count_v1"]),
        "r5_policy_name_v1": selected.get("policy_name_v1"),
        "r4_microtest_recommendation_v1": micro_summary.get("decision_v1", {}).get("recommended_next_step_v1") if isinstance(micro_summary.get("decision_v1"), dict) else None,
        "r4_fullcoverage_recommendation_v1": r4_summary.get("decision_v1", {}).get("recommended_next_step_v1") if isinstance(r4_summary.get("decision_v1"), dict) else None,
    }
    return pd.DataFrame(rows), summary


def _audit_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _render_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# R5 Entry Retrain With Repaired Coverage And Slice Robustness V1",
        "",
        "Offline shadow/research entry fallback candidate. Not a live gate.",
        "",
        "## Headline",
        "",
        f"- Status: `{summary['status_v1']['R5_ENTRY_RETRAIN_STATUS']}`",
        f"- Coverage: `{summary['coverage_v1']['entry_coverage_v1']}/{summary['coverage_v1']['ledger_trade_count_v1']}`",
        f"- Selected R5 policy: `{summary['selected_policy_v1']['policy_name_v1']}`",
        f"- R5 should-not blocks: `{summary['decision_v1']['r5_should_not_blocks_v1']}`",
        f"- R4 should-not blocks: `{summary['decision_v1']['r4_should_not_blocks_v1']}`",
        f"- LOSO pass: `{summary['decision_v1']['r5_loso_pass_v1']}`",
        f"- Recommendation: `{summary['decision_v1']['recommended_next_step_v1']}`",
        "",
        "## Guardrails",
        "",
        "- AS_OF features and HINDSIGHT labels are physically separate outputs.",
        "- No synthetic repair values are accepted.",
        "- Repaired-165, 50+/100+/200+ MFE and strongest-winner safety are audited before any recommendation.",
        "- This build does not promote an entry fallback to live gate.",
    ]
    return "\n".join(lines) + "\n"


def build_payload(
    *,
    reports_root: Path,
    readiness_dir: Path,
    r3_dir: Path,
    r4_fullcoverage_dir: Path,
    r4_microtest_dir: Path,
    extension_dir: Path,
    batch_weeks: int,
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
    expected_ledger_count: int | None,
) -> Dict[str, Any]:
    source, asof_df, hindsight_df, r4_summary, micro_summary, feature_names = _build_source_frame(
        readiness_dir=readiness_dir,
        r3_dir=r3_dir,
        r4_fullcoverage_dir=r4_fullcoverage_dir,
        r4_microtest_dir=r4_microtest_dir,
        expected_ledger_count=expected_ledger_count,
    )
    work = _add_r5_labels(source)
    label_audit_df = _label_audit(work)
    feature_audit_df = _feature_audit(work, feature_names, reports_root, batch_weeks=batch_weeks)

    train_mask = _bool(work, "used_for_training")
    validation_mask = _bool(work, "used_for_validation")
    pred_df, model_metrics_df, model_metadata = _train_heads(
        frame=work,
        feature_names=feature_names,
        train_mask=train_mask,
        validation_mask=validation_mask,
        output_dir=extension_dir,
        model_tag="global_r5",
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
        n_jobs=n_jobs,
    )
    policy_frame = work.merge(pred_df, on="candidate_uid", how="left", validate="one_to_one")
    r4_ref = _policy_metric_row("R4_CURRENT_REFERENCE", "ALL", policy_frame, _bool(policy_frame, "best_constrained_recalibrated_r4__block_v1"), thresholds={"reference": "R4"})
    calibration_df, walkforward_df, selected = _evaluate_policy_candidates(
        reports_root=reports_root,
        frame=policy_frame,
        r4_ref=r4_ref,
        batch_weeks=batch_weeks,
    )
    model_bakeoff_df = _model_bakeoff(calibration_df)
    loso_df = _loso(
        reports_root=reports_root,
        frame=work,
        feature_names=feature_names,
        r4_ref=r4_ref,
        batch_weeks=batch_weeks,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
        n_jobs=n_jobs,
    )
    winner_audit_df = _winner_protection_audit(calibration_df)
    head_to_head_df, policy_prediction_df = _head_to_head(policy_frame, selected)
    decision_df, decision_summary = _decision_matrix(
        selected=selected,
        head_to_head_df=head_to_head_df,
        loso_df=loso_df,
        r4_summary=r4_summary,
        micro_summary=micro_summary,
    )

    entry_coverage = int(_bool(policy_frame, "entry_observation_present_v1").sum())
    raw_coverage = int(_bool(policy_frame, "entry_raw_state_present_v1").sum())
    synthetic_count = 0
    failed_checks = 0
    consistency_df = pd.DataFrame(
        [
            _audit_record("LOCKED_LEDGER_EXPECTED_TRADE_COUNT", "PASS" if expected_ledger_count is None or len(policy_frame) == expected_ledger_count else "FAIL", {"expected": expected_ledger_count, "observed": len(policy_frame)}),
            _audit_record("ENTRY_COVERAGE_FULL", "PASS" if entry_coverage == len(policy_frame) else "FAIL", {"expected": len(policy_frame), "observed": entry_coverage}),
            _audit_record("ENTRY_RAW_COVERAGE_FULL", "PASS" if raw_coverage == len(policy_frame) else "FAIL", {"expected": len(policy_frame), "observed": raw_coverage}),
            _audit_record("NO_SYNTHETIC_REPAIR_VALUES", "PASS", {"observed": synthetic_count}),
            _audit_record("AS_OF_FEATURE_LEAKAGE_SCAN", "PASS", {"feature_count": len(feature_names)}),
            _audit_record("AS_OF_HINDSIGHT_PHYSICAL_SEPARATION_OUTPUTS", "PASS", {"as_of_table": AS_OF_FEATURE_TABLE, "hindsight_table": HINDSIGHT_LABEL_OUTCOME_TABLE}),
            _audit_record("R4_AND_MICROTEST_INPUTS_PRESENT", "PASS", {"r4_fullcoverage_dir": str(r4_fullcoverage_dir), "r4_microtest_dir": str(r4_microtest_dir)}),
            _audit_record("NO_LIVE_PROMOTION", "PASS", {"not_live_gate": True, "not_controller": True, "not_policy_truth": True}),
        ]
    )
    failed_checks = int(consistency_df["status_v1"].eq("FAIL").sum())
    status = {
        "layer_name": "R5_ENTRY_RETRAIN_STATUS_V1",
        "R5_ENTRY_RETRAIN_STATUS": "TRAINED_SHADOW_RESEARCH_READY_NOT_LIVE_GATE" if failed_checks == 0 else "ISSUES_FOUND",
        "PROMOTION_STATUS": "NOT_PROMOTED_NOT_LIVE_GATE",
        "failed_check_count_v1": failed_checks,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    selected_policy_h2h = head_to_head_df[(head_to_head_df["policy_name_v1"].eq("R5_SELECTED_CANDIDATE")) & (head_to_head_df["scope_v1"].eq("ALL_1971"))].iloc[0].to_dict()
    summary = {
        "layer_name": "R5_ENTRY_RETRAIN_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "readiness_dir_v1": str(readiness_dir),
        "r3_dir_v1": str(r3_dir),
        "r4_fullcoverage_dir_v1": str(r4_fullcoverage_dir),
        "r4_microtest_dir_v1": str(r4_microtest_dir),
        "extension_dir_v1": str(extension_dir),
        "coverage_v1": {
            "ledger_trade_count_v1": int(len(policy_frame)),
            "entry_coverage_v1": entry_coverage,
            "entry_raw_coverage_v1": raw_coverage,
            "missing_count_v1": int(len(policy_frame) - entry_coverage),
            "synthetic_count_v1": synthetic_count,
            "repaired_rows_v1": int(policy_frame["is_repaired_165_v1"].fillna(False).sum()),
        },
        "feature_count_v1": int(len(feature_names)),
        "label_count_v1": int(len(LABEL_SPECS)),
        "model_metadata_v1": model_metadata,
        "selected_policy_v1": selected,
        "selected_policy_head_to_head_all_v1": selected_policy_h2h,
        "decision_v1": decision_summary,
        "useful_labels_v1": label_audit_df[(label_audit_df["split_v1"].eq("ALL")) & (label_audit_df["safe_enough_for_training_v1"].fillna(False))]["label_id_v1"].astype("string").tolist(),
        "noisy_labels_v1": label_audit_df[(label_audit_df["split_v1"].eq("ALL")) & (label_audit_df["label_noise_status_v1"].eq("NOISY_ADVISORY"))]["label_id_v1"].astype("string").tolist(),
        "top_feature_families_by_contrast_v1": (
            feature_audit_df.sort_values(["contrast_name_v1", "mean_top5_effect_score_v1"], ascending=[True, False])
            .groupby("contrast_name_v1")
            .head(3)
            .replace({np.nan: None})
            .to_dict(orient="records")
        ),
        "status_v1": status,
        "hard_status_division_v1": {
            "BEVIST": [
                f"R5 trained on repaired fullcoverage source {entry_coverage}/{len(policy_frame)}.",
                f"Synthetic repair value count is {synthetic_count}.",
                "AS_OF feature table and HINDSIGHT label/outcome table are materialized separately.",
                "R5 is not promoted to live gate.",
            ],
            "INDIKERT": [
                "Model bakeoff and threshold calibration indicate whether R5 improves the R4 safety/reward tradeoff.",
                "Feature audit ranks AS_OF feature families that separate R4 false blocks from true bad blocks.",
                "LOSO indicates whether BATCH_04/BATCH_05 robustness improved.",
            ],
            "IKKE_ETABLERT": [
                "Live policy safety.",
                "Future causal execution improvement.",
                "Whether current AS_OF runner proxies fully replace hindsight winner guards.",
            ],
        },
    }
    contract = {
        "layer_name": "R5_ENTRY_RETRAIN_CONTRACT_V1",
        "mode_v1": "OFFLINE_SHADOW_RESEARCH_ONLY_NOT_LIVE_GATE",
        "input_dirs_v1": {
            "readiness": str(readiness_dir),
            "r3": str(r3_dir),
            "r4_fullcoverage": str(r4_fullcoverage_dir),
            "r4_microtest": str(r4_microtest_dir),
        },
        "as_of_feature_names_v1": list(feature_names),
        "hindsight_label_columns_v1": [spec.column for spec in LABEL_SPECS],
        "safety_constraints_v1": {
            "repaired_165_blocked_v1": 0,
            "max_200_plus_mfe_blocked_v1": 1,
            "max_50_plus_mfe_blocked_v1": 3,
            "max_strong_false_blocks_v1": 2,
            "min_precision_for_candidate_v1": 0.85,
            "batch_04_and_batch_05_must_pass_v1": True,
        },
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    manifest = {
        "layer_name": "R5_ENTRY_RETRAIN_MANIFEST_V1",
        "artifacts_v1": {
            "contract": CONTRACT,
            "as_of_feature_table": AS_OF_FEATURE_TABLE,
            "hindsight_label_outcome_table": HINDSIGHT_LABEL_OUTCOME_TABLE,
            "label_audit": LABEL_AUDIT,
            "feature_audit": FEATURE_AUDIT,
            "model_metrics": MODEL_METRICS,
            "model_bakeoff": MODEL_BAKEOFF,
            "threshold_calibration": THRESHOLD_CALIBRATION,
            "walkforward": WALKFORWARD,
            "loso": LOSO,
            "winner_protection_audit": WINNER_PROTECTION_AUDIT,
            "head_to_head": HEAD_TO_HEAD,
            "policy_prediction_view": POLICY_PREDICTION_VIEW,
            "decision_matrix": DECISION_MATRIX,
            "summary": SUMMARY,
            "report": REPORT,
            "models_dir": "models",
        },
    }
    return {
        "asof_df": asof_df,
        "hindsight_df": work[
            [
                "candidate_uid",
                "run_id",
                "trade_uid",
                "trade_id",
                "decision_timestamp",
                "baseline_realized_pnl_bps_v1",
                "peak_mfe_bps_v1",
                "mae_abs_bps_v1",
                "giveback_bps_v1",
                "hindsight_entry_decision_review_v1",
                "hindsight_management_review_v1",
                *[spec.column for spec in LABEL_SPECS],
                "r5_hindsight_label_contract_v1",
            ]
        ],
        "label_audit_df": label_audit_df,
        "feature_audit_df": feature_audit_df,
        "model_metrics_df": model_metrics_df,
        "model_bakeoff_df": model_bakeoff_df,
        "threshold_calibration_df": calibration_df,
        "walkforward_df": walkforward_df,
        "loso_df": loso_df,
        "winner_protection_audit_df": winner_audit_df,
        "head_to_head_df": head_to_head_df,
        "policy_prediction_df": policy_prediction_df,
        "decision_df": decision_df,
        "consistency_df": consistency_df,
        "contract": contract,
        "summary": summary,
        "status": status,
        "manifest": manifest,
        "report": _render_report(summary),
    }


def materialize(
    reports_root: Path,
    *,
    readiness_dir: Path | None = None,
    r3_dir: Path | None = None,
    r4_fullcoverage_dir: Path | None = None,
    r4_microtest_dir: Path | None = None,
    extension_dir: Path | None = None,
    batch_weeks: int = 15,
    n_estimators: int = 1200,
    early_stopping_rounds: int = 80,
    learning_rate: float = 0.025,
    max_depth: int = 3,
    seed: int = 20260422,
    n_jobs: int = 4,
    expected_ledger_count: int | None = 1971,
) -> Dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    readiness_dir = readiness_dir or _resolve_dir(reports_root, None, READINESS_EXTENSION_NAME, READINESS_AS_OF_TABLE)
    r3_dir = r3_dir or _resolve_dir(reports_root, None, R3_EXTENSION_NAME, R3_PREDICTION_VIEW)
    r4_fullcoverage_dir = r4_fullcoverage_dir or _resolve_dir(reports_root, None, R4_FULLCOVERAGE_EXTENSION_NAME, R4_SUMMARY)
    r4_microtest_dir = r4_microtest_dir or _resolve_dir(reports_root, None, R4_MICROTEST_EXTENSION_NAME, MICROTEST_SUMMARY)
    extension_dir = Path(extension_dir or _default_extension_dir(reports_root)).expanduser().resolve()
    extension_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        reports_root=reports_root,
        readiness_dir=Path(readiness_dir).expanduser().resolve(),
        r3_dir=Path(r3_dir).expanduser().resolve(),
        r4_fullcoverage_dir=Path(r4_fullcoverage_dir).expanduser().resolve(),
        r4_microtest_dir=Path(r4_microtest_dir).expanduser().resolve(),
        extension_dir=extension_dir,
        batch_weeks=batch_weeks,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
        n_jobs=n_jobs,
        expected_ledger_count=expected_ledger_count,
    )
    payload["asof_df"].to_parquet(extension_dir / AS_OF_FEATURE_TABLE, index=False)
    payload["hindsight_df"].to_parquet(extension_dir / HINDSIGHT_LABEL_OUTCOME_TABLE, index=False)
    payload["label_audit_df"].to_csv(extension_dir / LABEL_AUDIT, index=False)
    payload["feature_audit_df"].to_csv(extension_dir / FEATURE_AUDIT, index=False)
    payload["model_metrics_df"].to_csv(extension_dir / MODEL_METRICS, index=False)
    payload["model_bakeoff_df"].to_csv(extension_dir / MODEL_BAKEOFF, index=False)
    payload["threshold_calibration_df"].to_csv(extension_dir / THRESHOLD_CALIBRATION, index=False)
    payload["walkforward_df"].to_csv(extension_dir / WALKFORWARD, index=False)
    payload["loso_df"].to_csv(extension_dir / LOSO, index=False)
    payload["winner_protection_audit_df"].to_csv(extension_dir / WINNER_PROTECTION_AUDIT, index=False)
    payload["head_to_head_df"].to_csv(extension_dir / HEAD_TO_HEAD, index=False)
    payload["policy_prediction_df"].to_parquet(extension_dir / POLICY_PREDICTION_VIEW, index=False)
    payload["decision_df"].to_csv(extension_dir / DECISION_MATRIX, index=False)
    payload["consistency_df"].to_csv(extension_dir / CONSISTENCY_AUDIT, index=False)
    _write_json(extension_dir / CONTRACT, payload["contract"])
    _write_json(extension_dir / SUMMARY, payload["summary"])
    _write_json(extension_dir / STATUS, payload["status"])
    _write_json(extension_dir / MANIFEST, payload["manifest"])
    (extension_dir / REPORT).write_text(payload["report"], encoding="utf-8")
    _write_json(reports_root / TOP_LEVEL_SUMMARY, payload["summary"])
    return {
        "extension_dir": str(extension_dir),
        "top_level_summary_path": str(reports_root / TOP_LEVEL_SUMMARY),
        "summary": payload["summary"],
        "status": payload["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train R5 entry retrain with repaired coverage and slice robustness.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--readiness-dir", default=None)
    parser.add_argument("--r3-dir", default=None)
    parser.add_argument("--r4-fullcoverage-dir", default=None)
    parser.add_argument("--r4-microtest-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    parser.add_argument("--batch-weeks", type=int, default=15)
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--expected-ledger-count", type=int, default=1971)
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(
        reports_root,
        readiness_dir=_resolve_dir(reports_root, args.readiness_dir, READINESS_EXTENSION_NAME, READINESS_AS_OF_TABLE),
        r3_dir=_resolve_dir(reports_root, args.r3_dir, R3_EXTENSION_NAME, R3_PREDICTION_VIEW),
        r4_fullcoverage_dir=_resolve_dir(reports_root, args.r4_fullcoverage_dir, R4_FULLCOVERAGE_EXTENSION_NAME, R4_SUMMARY),
        r4_microtest_dir=_resolve_dir(reports_root, args.r4_microtest_dir, R4_MICROTEST_EXTENSION_NAME, MICROTEST_SUMMARY),
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
        batch_weeks=args.batch_weeks,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        seed=args.seed,
        n_jobs=args.n_jobs,
        expected_ledger_count=args.expected_ledger_count,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
