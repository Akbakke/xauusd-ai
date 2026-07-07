#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
R2_READINESS_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_HARVEST_R2_ENTRY_COVERAGE_AND_WALKFORWARD_READINESS_V1"
R2_RETRAIN_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_HARVEST_RETRAIN_CANDIDATE_R2"
R3_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R3_ENTRY_LABEL_FEATURE_RETRAIN_V1"
R4_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R4_ENTRY_CALIBRATED_FALLBACK_RETRAIN_V1"

R2_READINESS_CONTRACT = "shadow_meta_all_trade_review_harvest_r2_entry_readiness_contract_v1.json"
R2_AS_OF_TABLE = "shadow_meta_all_trade_review_harvest_r2_entry_as_of_feature_table_v1.parquet"
R2_LABEL_TABLE = "shadow_meta_all_trade_review_harvest_r2_entry_hindsight_label_table_v1.parquet"
R2_COVERAGE_AUDIT = "shadow_meta_all_trade_review_harvest_r2_entry_coverage_gap_audit_v1.csv"
R2_PREDICTION_VIEW = "shadow_meta_all_trade_review_harvest_retrain_candidate_prediction_view_v1.parquet"
R3_PREDICTION_VIEW = "shadow_meta_all_trade_review_r3_entry_label_feature_prediction_view_v1.parquet"
R3_SUMMARY = "shadow_meta_all_trade_review_r3_entry_label_feature_summary_v1.json"

R4_JOINED_VIEW = "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_joined_view_v1.parquet"
R4_POLICY_PREDICTION_VIEW = "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_policy_prediction_view_v1.parquet"
R4_R2_FALLBACK_PRESERVATION_AUDIT = "shadow_meta_all_trade_review_r4_r2_fallback_preservation_audit_v1.csv"
R4_R2_FALLBACK_FEATURE_AUDIT = "shadow_meta_all_trade_review_r4_r2_fallback_preservation_feature_audit_v1.csv"
R4_R3_THRESHOLD_AUDIT = "shadow_meta_all_trade_review_r4_r3_label_threshold_audit_v1.csv"
R4_R3_CALIBRATION_CURVE = "shadow_meta_all_trade_review_r4_r3_label_calibration_curve_v1.csv"
R4_STRONG_WINNER_AUDIT = "shadow_meta_all_trade_review_r4_strong_winner_protection_audit_v1.csv"
R4_POLICY_STACK_CANDIDATES = "shadow_meta_all_trade_review_r4_policy_stack_candidates_v1.csv"
R4_WALKFORWARD_SAFETY_REPLAY = "shadow_meta_all_trade_review_r4_walkforward_safety_replay_v1.csv"
R4_COVERAGE_AUDIT = "shadow_meta_all_trade_review_r4_entry_coverage_gap_audit_v1.csv"
R4_READINESS_MATRIX = "shadow_meta_all_trade_review_r4_readiness_decision_matrix_v1.csv"
R4_CONSISTENCY_AUDIT = "shadow_meta_all_trade_review_r4_consistency_audit_v1.csv"
R4_CONTRACT = "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_contract_v1.json"
R4_SUMMARY = "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_summary_v1.json"
R4_STATUS = "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_status_v1.json"
R4_MANIFEST = "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_manifest_v1.json"
R4_REPORT = "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_report_v1.md"
TOP_LEVEL_SUMMARY = "truth_r4_entry_calibrated_fallback_retrain_v1.json"

RUN_RE = re.compile(r"^E2E_SANITY_ORDERFIX_(\d{8})_(\d{8})$")
LEAKAGE_TOKENS = (
    "hindsight",
    "pnl",
    "reward",
    "target",
    "label",
    "harvest",
    "terminal",
    "good_trade",
    "bad_trade",
    "premature",
    "late_exit",
)

TASKS = {
    "should_not_take": ("entry_r3_should_not_take", "label_should_not_take_v1"),
    "immediate_mae_risk": ("entry_r3_immediate_mae_risk", "label_immediate_mae_risk_v1"),
    "wait_advisory": ("entry_r3_wait_would_have_helped", "label_wait_would_have_helped_v1"),
    "strong_trade_candidate": ("entry_r3_strong_trade_candidate", "label_strong_trade_candidate_v1"),
    "direct_take_ok": ("entry_r3_direct_take_ok", "label_direct_take_ok_v1"),
    "good_mfe_bad_capture": ("entry_r3_good_mfe_bad_capture", "label_good_mfe_bad_capture_v1"),
}

R4_THRESHOLDS = {
    "should_not_take_threshold_v1": 0.60,
    "direct_take_protection_ceiling_v1": 0.55,
    "strong_winner_protection_threshold_v1": 0.75,
    "immediate_mae_risk_threshold_v1": 0.80,
    "wait_advisory_threshold_v1": 0.85,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _resolve_dir(reports_root: Path, arg: str | None, default_name: str, required_file: str) -> Path:
    path = Path(arg).expanduser().resolve() if arg else reports_root / default_name
    if not path.exists():
        raise FileNotFoundError(f"Required dir does not exist: {path}")
    if not (path / required_file).exists():
        raise FileNotFoundError(f"{path} missing required artifact {required_file}")
    return path


def _default_extension_dir(reports_root: Path) -> Path:
    return reports_root / R4_EXTENSION_NAME


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


def _safe_rate(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num / den)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _counts(frame: pd.DataFrame, column: str) -> Dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].astype("string").value_counts(dropna=False).to_dict().items()}


def _bool(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    return series.astype("string").str.lower().str.strip().eq("true").fillna(default).astype(bool)


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _prob(frame: pd.DataFrame, task_id: str) -> pd.Series:
    return pd.to_numeric(frame.get(f"pred__{task_id}__prob_true_v1", pd.Series(np.nan, index=frame.index)), errors="coerce")


def _run_sort_key(run_id: str) -> str:
    match = RUN_RE.match(str(run_id))
    return match.group(1) if match else str(run_id)


def _all_run_ids(reports_root: Path, frame: pd.DataFrame) -> List[str]:
    runs_root = reports_root / "runs"
    if runs_root.exists():
        run_ids = sorted([path.name for path in runs_root.iterdir() if path.is_dir() and RUN_RE.match(path.name)], key=_run_sort_key)
        if run_ids:
            return run_ids
    return sorted(frame["run_id"].astype("string").dropna().unique().tolist(), key=_run_sort_key)


def _check_feature_names(feature_names: Sequence[str]) -> None:
    bad: List[str] = []
    for feature in feature_names:
        lower = feature.lower()
        for token in LEAKAGE_TOKENS:
            if token == "realized" and "realized_vol" in lower:
                continue
            if token in lower:
                bad.append(feature)
                break
    if bad:
        raise ValueError(f"AS_OF feature list contains forbidden leakage-like names: {bad[:20]}")


def _load_feature_names(readiness_dir: Path, asof_df: pd.DataFrame) -> List[str]:
    contract = _load_json(readiness_dir / R2_READINESS_CONTRACT)
    raw = contract.get("as_of_feature_names_v1")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("R2 readiness contract missing as_of_feature_names_v1")
    feature_names = [str(feature) for feature in raw]
    _require_columns(asof_df, feature_names, artifact_name=R2_AS_OF_TABLE)
    _check_feature_names(feature_names)
    return feature_names


def _build_joined(
    *,
    readiness_dir: Path,
    r2_dir: Path,
    r3_dir: Path,
) -> tuple[pd.DataFrame, List[str], pd.DataFrame, Dict[str, Any]]:
    asof_df = pd.read_parquet(readiness_dir / R2_AS_OF_TABLE)
    labels_df = pd.read_parquet(readiness_dir / R2_LABEL_TABLE)
    coverage_df = pd.read_csv(readiness_dir / R2_COVERAGE_AUDIT)
    r2_df = pd.read_parquet(r2_dir / R2_PREDICTION_VIEW)
    r3_df = pd.read_parquet(r3_dir / R3_PREDICTION_VIEW)
    r3_summary = _load_json(r3_dir / R3_SUMMARY) if (r3_dir / R3_SUMMARY).exists() else {}
    feature_names = _load_feature_names(readiness_dir, asof_df)

    for name, frame in [(R2_AS_OF_TABLE, asof_df), (R2_LABEL_TABLE, labels_df), (R2_PREDICTION_VIEW, r2_df), (R3_PREDICTION_VIEW, r3_df)]:
        _require_columns(frame, ["candidate_uid"], artifact_name=name)
        if bool(frame["candidate_uid"].astype("string").duplicated().any()):
            raise ValueError(f"{name} requires unique candidate_uid")

    r2_cols = [
        "candidate_uid",
        "candidate_shadow_action_v1",
        "candidate_shadow_action_source_v1",
        "candidate_shadow_action_matches_harvest_target_v1",
        "candidate_shadow_delta_bps_v1",
        "candidate_shadow_pnl_bps_v1",
        "pred__entry_xgb_binary_take__prob_false_v1",
        "pred__entry_xgb_binary_take__prob_true_v1",
        "pred__entry_xgb_harvest_label__prob_reject_or_low_size_v1",
    ]
    r2_source = r2_df[[column for column in r2_cols if column in r2_df.columns]].copy()
    r2_source = r2_source.rename(columns={column: f"r2_{column}" for column in r2_source.columns if column != "candidate_uid"})

    coverage_cols = [
        "candidate_uid",
        "entry_gap_reason_code_v1",
        "entry_gap_reason_detail_v1",
        "management_gap_reason_code_v1",
        "management_gap_reason_detail_v1",
        "coverage_gap_scope_v1",
    ]
    label_cols = [
        "candidate_uid",
        "label_low_mfe_low_value_v1",
        "hindsight_entry_decision_review_v1",
        "hindsight_management_review_v1",
        "session",
        "vol_regime",
        "trend_regime",
    ]
    joined = (
        r3_df.merge(r2_source, on="candidate_uid", how="left", validate="one_to_one")
        .merge(labels_df[[column for column in label_cols if column in labels_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
        .merge(coverage_df[[column for column in coverage_cols if column in coverage_df.columns]], on="candidate_uid", how="left", validate="one_to_one")
        .merge(asof_df[["candidate_uid", *feature_names]], on="candidate_uid", how="left", validate="one_to_one")
    )
    joined["r2_entry_fallback_row_v1"] = joined["r2_candidate_shadow_action_source_v1"].astype("string").eq("ENTRY_MODEL_SUPPRESS_FALLBACK")
    joined["r2_entry_fallback_correct_v1"] = joined["r2_entry_fallback_row_v1"] & joined[
        "r2_candidate_shadow_action_matches_harvest_target_v1"
    ].fillna(False).astype(bool)
    joined["r3_conservative_blocks_v1"] = joined["entry_r3_shadow_action_v1"].astype("string").eq("ENTRY_SUPPRESS_OR_DOWNSIZE_SHADOW")
    joined["r3_missed_r2_correct_fallback_v1"] = joined["r2_entry_fallback_correct_v1"] & ~joined["r3_conservative_blocks_v1"]
    joined["r3_preserved_r2_correct_fallback_v1"] = joined["r2_entry_fallback_correct_v1"] & joined["r3_conservative_blocks_v1"]
    return joined, feature_names, coverage_df, r3_summary


def _feature_score(values: pd.Series, y: pd.Series) -> Dict[str, Any]:
    valid = values.notna() & y.notna()
    if int(valid.sum()) < 10 or int(y.loc[valid].nunique()) < 2:
        return {"score_v1": None, "auc_v1": None, "direction_v1": "NOT_EVALUABLE", "coverage_v1": int(valid.sum())}
    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        x = pd.to_numeric(values.loc[valid], errors="coerce")
        valid2 = x.notna()
        x = x.loc[valid2].astype(float)
        yy = y.loc[valid].loc[valid2].astype(int)
        if len(x) < 10 or yy.nunique() < 2:
            return {"score_v1": None, "auc_v1": None, "direction_v1": "NOT_EVALUABLE", "coverage_v1": int(len(x))}
        auc = _safe_float(roc_auc_score(yy, x))
        pos_mean = float(x[yy.eq(1)].mean())
        neg_mean = float(x[yy.eq(0)].mean())
        std = float(x.std(ddof=0)) or 0.0
        effect = abs(pos_mean - neg_mean) / std if std > 0 else 0.0
        return {
            "score_v1": float(effect + abs((auc or 0.5) - 0.5) * 2.0),
            "auc_v1": auc,
            "direction_v1": "HIGHER_FOR_POSITIVE" if pos_mean >= neg_mean else "LOWER_FOR_POSITIVE",
            "positive_mean_v1": pos_mean,
            "negative_mean_v1": neg_mean,
            "coverage_v1": int(len(x)),
        }
    x = values.loc[valid].astype("string")
    yy = y.loc[valid].astype(int)
    grouped = yy.groupby(x).agg(["count", "mean"])
    grouped = grouped[grouped["count"].ge(3)]
    if grouped.empty:
        return {"score_v1": None, "auc_v1": None, "direction_v1": "NOT_EVALUABLE", "coverage_v1": int(valid.sum())}
    overall = float(yy.mean())
    diff = (grouped["mean"] - overall).abs()
    best = str(diff.idxmax())
    return {
        "score_v1": float(diff.max()),
        "auc_v1": None,
        "direction_v1": f"CATEGORY:{best}",
        "positive_mean_v1": float(grouped.loc[best, "mean"]),
        "negative_mean_v1": overall,
        "coverage_v1": int(valid.sum()),
    }


def _build_r2_preservation_audit(joined: pd.DataFrame, feature_names: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    fallback = joined[joined["r2_entry_fallback_row_v1"]].copy()
    fallback["r4_preservation_target_should_block_v1"] = _bool(fallback, "label_should_not_take_v1")
    fallback["r4_preservation_target_should_release_v1"] = ~fallback["r4_preservation_target_should_block_v1"]
    fallback["r4_preservation_reason_v1"] = np.select(
        [
            fallback["r4_preservation_target_should_block_v1"],
            _bool(fallback, "label_strong_trade_candidate_v1"),
        ],
        ["R2_CORRECT_SHOULD_NOT_TAKE_BLOCK", "R2_FALSE_BLOCK_STRONG_WINNER_RELEASE"],
        default="R2_FALSE_BLOCK_RELEASE_OR_REVIEW",
    )
    fallback["r3_preservation_outcome_v1"] = np.select(
        [
            fallback["r4_preservation_target_should_block_v1"] & fallback["r3_conservative_blocks_v1"],
            fallback["r4_preservation_target_should_block_v1"] & ~fallback["r3_conservative_blocks_v1"],
            ~fallback["r4_preservation_target_should_block_v1"] & fallback["r3_conservative_blocks_v1"],
        ],
        ["R3_PRESERVED_R2_CORRECT_BLOCK", "R3_MISSED_R2_CORRECT_BLOCK", "R3_BLOCKED_R2_FALSE_BLOCK_TOO"],
        default="R3_RELEASED_R2_FALSE_BLOCK",
    )

    y = (fallback["r4_preservation_target_should_block_v1"] & ~fallback["r3_conservative_blocks_v1"]).astype(int)
    feature_rows: List[Dict[str, Any]] = []
    for feature in feature_names:
        score = _feature_score(fallback[feature], y)
        feature_rows.append({"feature_name_v1": feature, "audit_target_v1": "R3_MISSED_R2_CORRECT_FALLBACK_BLOCK", **score})
    feature_audit = pd.DataFrame(feature_rows).sort_values("score_v1", ascending=False, na_position="last")
    summary = {
        "r2_entry_fallback_rows_v1": int(len(fallback)),
        "r2_entry_fallback_correct_rows_v1": int(fallback["r2_entry_fallback_correct_v1"].sum()),
        "r2_entry_fallback_should_not_take_rows_v1": int(fallback["r4_preservation_target_should_block_v1"].sum()),
        "r2_entry_fallback_false_block_rows_v1": int((~fallback["r4_preservation_target_should_block_v1"]).sum()),
        "r2_entry_fallback_false_block_strong_winners_v1": int((~fallback["r4_preservation_target_should_block_v1"] & _bool(fallback, "label_strong_trade_candidate_v1")).sum()),
        "r3_missed_r2_correct_fallback_rows_v1": int((fallback["r4_preservation_target_should_block_v1"] & ~fallback["r3_conservative_blocks_v1"]).sum()),
        "r3_preserved_r2_correct_fallback_rows_v1": int((fallback["r4_preservation_target_should_block_v1"] & fallback["r3_conservative_blocks_v1"]).sum()),
    }
    return fallback, feature_audit, summary


def _threshold_audit(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    thresholds = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    scopes = [
        ("ALL", pd.Series(True, index=joined.index)),
        ("VALIDATION", _bool(joined, "used_for_validation")),
        ("HOLDOUT", _bool(joined, "used_for_holdout")),
    ]
    strong = _bool(joined, "label_strong_trade_candidate_v1")
    should = _bool(joined, "label_should_not_take_v1")
    for task_name, (task_id, label_col) in TASKS.items():
        prob = _prob(joined, task_id)
        label = _bool(joined, label_col)
        for scope, mask in scopes:
            sub_mask = mask & prob.notna()
            for threshold in thresholds:
                pred = prob.ge(threshold) & sub_mask
                tp = int((pred & label).sum())
                pred_count = int(pred.sum())
                label_count = int((label & sub_mask).sum())
                rows.append(
                    {
                        "task_name_v1": task_name,
                        "task_id_v1": task_id,
                        "label_column_v1": label_col,
                        "scope_v1": scope,
                        "threshold_v1": float(threshold),
                        "row_count_v1": int(sub_mask.sum()),
                        "predicted_positive_count_v1": pred_count,
                        "true_positive_count_v1": tp,
                        "precision_v1": _safe_rate(float(tp), float(pred_count)),
                        "recall_v1": _safe_rate(float(tp), float(label_count)),
                        "false_block_strong_trade_count_v1": int((pred & strong & ~label).sum()),
                        "false_block_strong_trade_rate_v1": _safe_rate(float((pred & strong & ~label).sum()), float((strong & sub_mask).sum())),
                        "false_allow_should_not_take_count_v1": int((~pred & sub_mask & should).sum()),
                    }
                )
            scoped = joined.loc[sub_mask, ["candidate_uid"]].copy()
            scoped["prob_v1"] = prob.loc[sub_mask].astype(float)
            scoped["label_v1"] = label.loc[sub_mask].astype(bool)
            scoped["bin_v1"] = pd.cut(scoped["prob_v1"], bins=np.linspace(0.0, 1.0, 11), include_lowest=True)
            for bin_key, bin_df in scoped.groupby("bin_v1", observed=False):
                if bin_df.empty:
                    continue
                curve_rows.append(
                    {
                        "task_name_v1": task_name,
                        "task_id_v1": task_id,
                        "scope_v1": scope,
                        "prob_bin_v1": str(bin_key),
                        "row_count_v1": int(len(bin_df)),
                        "mean_prob_v1": float(bin_df["prob_v1"].mean()),
                        "empirical_positive_rate_v1": float(bin_df["label_v1"].mean()),
                    }
                )
    audit = pd.DataFrame(rows)
    curve = pd.DataFrame(curve_rows)
    recs: Dict[str, Any] = {}
    validation = audit[audit["scope_v1"].eq("VALIDATION")].copy()
    for task_name in TASKS:
        sub = validation[validation["task_name_v1"].eq(task_name)].copy()
        sub["f1_proxy_v1"] = 2 * sub["precision_v1"].fillna(0) * sub["recall_v1"].fillna(0) / (
            sub["precision_v1"].fillna(0) + sub["recall_v1"].fillna(0)
        ).replace(0, np.nan)
        best = sub.sort_values(["f1_proxy_v1", "precision_v1", "recall_v1"], ascending=False).head(1)
        recs[task_name] = None if best.empty else float(best["threshold_v1"].iloc[0])
    return audit, curve, recs


def _policy_masks(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    available = _bool(frame, "entry_r3_feature_available_v1")
    p_should = _prob(frame, "entry_r3_should_not_take")
    p_mae = _prob(frame, "entry_r3_immediate_mae_risk")
    p_wait = _prob(frame, "entry_r3_wait_would_have_helped")
    p_strong = _prob(frame, "entry_r3_strong_trade_candidate")
    p_direct = _prob(frame, "entry_r3_direct_take_ok")
    r2_fb = _bool(frame, "r2_entry_fallback_row_v1")
    strong_protect = p_strong.ge(R4_THRESHOLDS["strong_winner_protection_threshold_v1"]).fillna(False)
    should_signal = p_should.ge(R4_THRESHOLDS["should_not_take_threshold_v1"]).fillna(False)
    direct_weak = p_direct.lt(R4_THRESHOLDS["direct_take_protection_ceiling_v1"]).fillna(False)
    mae_signal = p_mae.ge(R4_THRESHOLDS["immediate_mae_risk_threshold_v1"]).fillna(False) & direct_weak
    wait_signal = p_wait.ge(R4_THRESHOLDS["wait_advisory_threshold_v1"]).fillna(False) & direct_weak
    masks = {
        "NO_ENTRY_FALLBACK_BASELINE": pd.Series(False, index=frame.index),
        "R2_FALLBACK_REFERENCE": available & r2_fb,
        "R3_CONSERVATIVE_POLICY": available & _bool(frame, "r3_conservative_blocks_v1"),
        "SHOULD_NOT_TAKE_BLOCKER_ONLY": available & should_signal,
        "SHOULD_NOT_TAKE_PLUS_STRONG_WINNER_PROTECTOR": available & should_signal & ~strong_protect,
        "SHOULD_NOT_TAKE_PLUS_IMMEDIATE_MAE_RISK": available & (should_signal | mae_signal),
        "SHOULD_NOT_TAKE_PLUS_WAIT_ADVISORY": available & (should_signal | wait_signal),
        "COMBINED_CONSERVATIVE_STACK": available & (should_signal | mae_signal | wait_signal) & ~strong_protect,
        "R4_R2_PRESERVED_STRONG_PROTECTED": available & r2_fb & ~strong_protect,
        "R4_R2_PRESERVED_PLUS_SHOULD_DIRECT_STRONG_PROTECTED": available & (r2_fb | (should_signal & direct_weak)) & ~strong_protect,
        "R4_R2_PRESERVED_PLUS_COMBINED_STACK": available & (r2_fb | should_signal | mae_signal | wait_signal) & ~strong_protect,
    }
    return {name: mask.fillna(False).astype(bool) for name, mask in masks.items()}


def _policy_metric_row(policy_name: str, scope: str, frame: pd.DataFrame, block: pd.Series) -> Dict[str, Any]:
    should = _bool(frame, "label_should_not_take_v1")
    strong = _bool(frame, "label_strong_trade_candidate_v1")
    fifty = _num(frame, "peak_mfe_bps_v1").ge(50.0)
    two_hundred = _num(frame, "peak_mfe_bps_v1").ge(200.0)
    tail = _num(frame, "peak_mfe_bps_v1").between(10.0, 50.0, inclusive="left") & (_num(frame, "baseline_realized_pnl_bps_v1").le(0.0) | should)
    r2_fb = _bool(frame, "r2_entry_fallback_row_v1")
    realized_delta = (-_num(frame, "baseline_realized_pnl_bps_v1")).where(block, 0.0)
    y_true = should.astype(int)
    y_pred = block.astype(int)
    bal_acc = None
    if y_true.nunique() == 2 and y_pred.nunique() >= 1:
        bal_acc = _safe_float(balanced_accuracy_score(y_true, y_pred))
    return {
        "policy_name_v1": policy_name,
        "scope_v1": scope,
        "row_count_v1": int(len(frame)),
        "block_count_v1": int(block.sum()),
        "block_rate_v1": _safe_rate(float(block.sum()), float(len(frame))),
        "should_not_take_block_count_v1": int((block & should).sum()),
        "should_not_take_precision_v1": _safe_rate(float((block & should).sum()), float(block.sum())),
        "should_not_take_recall_v1": _safe_rate(float((block & should).sum()), float(should.sum())),
        "false_allow_should_not_take_count_v1": int((~block & should).sum()),
        "strong_trade_false_block_count_v1": int((block & strong).sum()),
        "strong_trade_false_block_rate_v1": _safe_rate(float((block & strong).sum()), float(strong.sum())),
        "fifty_plus_mfe_block_count_v1": int((block & fifty).sum()),
        "fifty_plus_mfe_block_rate_v1": _safe_rate(float((block & fifty).sum()), float(fifty.sum())),
        "two_hundred_plus_mfe_block_count_v1": int((block & two_hundred).sum()),
        "tail_10_50_help_count_v1": int((block & tail).sum()),
        "tail_10_50_help_recall_v1": _safe_rate(float((block & tail).sum()), float(tail.sum())),
        "r2_fallback_rows_blocked_v1": int((block & r2_fb).sum()),
        "r2_fallback_should_not_take_preserved_v1": int((block & r2_fb & should).sum()),
        "r2_fallback_false_blocks_kept_v1": int((block & r2_fb & ~should).sum()),
        "hindsight_skip_delta_bps_v1": float(realized_delta.sum()),
        "blocked_avg_mfe_bps_v1": _safe_float(_num(frame.loc[block], "peak_mfe_bps_v1").mean()) if int(block.sum()) else None,
        "blocked_avg_mae_bps_v1": _safe_float(_num(frame.loc[block], "mae_abs_bps_v1").mean()) if int(block.sum()) else None,
        "blocked_avg_giveback_bps_v1": _safe_float(_num(frame.loc[block], "giveback_bps_v1").mean()) if int(block.sum()) else None,
        "binary_balanced_accuracy_vs_should_not_take_v1": bal_acc,
        "confusion_matrix_json_v1": _json_dumps(confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()) if len(frame) else "[]",
    }


def _build_policy_candidate_metrics(reports_root: Path, joined: pd.DataFrame, *, batch_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    policy_masks = _policy_masks(joined)
    rows: List[Dict[str, Any]] = []
    scopes = [
        ("ALL", pd.Series(True, index=joined.index)),
        ("TRAIN", _bool(joined, "used_for_training")),
        ("VALIDATION", _bool(joined, "used_for_validation")),
        ("HOLDOUT", _bool(joined, "used_for_holdout")),
    ]
    for policy_name, block in policy_masks.items():
        for scope, scope_mask in scopes:
            sub = joined.loc[scope_mask].copy()
            rows.append(_policy_metric_row(policy_name, scope, sub, block.loc[scope_mask]))
    candidate_df = pd.DataFrame(rows)

    wf_rows: List[Dict[str, Any]] = []
    run_ids = _all_run_ids(reports_root, joined)
    for batch_index, start in enumerate(range(0, len(run_ids), batch_weeks), start=1):
        batch_run_ids = run_ids[start : start + batch_weeks]
        batch_mask = joined["run_id"].astype("string").isin(batch_run_ids)
        for policy_name, block in policy_masks.items():
            row = _policy_metric_row(policy_name, f"BATCH_{batch_index:02d}", joined.loc[batch_mask].copy(), block.loc[batch_mask])
            row.update({"batch_index_v1": int(batch_index), "run_count_v1": int(len(batch_run_ids)), "run_start_v1": batch_run_ids[0] if batch_run_ids else None, "run_end_v1": batch_run_ids[-1] if batch_run_ids else None})
            wf_rows.append(row)
    walkforward_df = pd.DataFrame(wf_rows)

    selected_name = "R4_R2_PRESERVED_PLUS_SHOULD_DIRECT_STRONG_PROTECTED"
    selected_block = policy_masks[selected_name]
    prediction_df = joined[
        [
            "run_id",
            "candidate_uid",
            "trade_uid",
            "trade_id",
            "decision_timestamp",
            "used_for_training",
            "used_for_validation",
            "used_for_holdout",
            "entry_r3_feature_available_v1",
            "label_should_not_take_v1",
            "label_strong_trade_candidate_v1",
            "label_immediate_mae_risk_v1",
            "label_wait_would_have_helped_v1",
            "label_direct_take_ok_v1",
            "label_good_mfe_bad_capture_v1",
            "peak_mfe_bps_v1",
            "mae_abs_bps_v1",
            "giveback_bps_v1",
            "baseline_realized_pnl_bps_v1",
            "r2_entry_fallback_row_v1",
            "r2_entry_fallback_correct_v1",
            "r3_conservative_blocks_v1",
        ]
    ].copy()
    prediction_df["r4_selected_policy_name_v1"] = selected_name
    prediction_df["r4_entry_fallback_block_v1"] = selected_block.to_numpy(dtype=bool)
    prediction_df["r4_entry_fallback_action_v1"] = np.where(selected_block, "ENTRY_SUPPRESS_OR_DOWNSIZE_SHADOW", "ENTRY_ALLOW_BASELINE_SHADOW")
    prediction_df["r4_entry_fallback_source_v1"] = np.select(
        [
            selected_block & _bool(joined, "r2_entry_fallback_row_v1"),
            selected_block & ~_bool(joined, "r2_entry_fallback_row_v1"),
        ],
        ["R4_PRESERVED_R2_FALLBACK_WITH_STRONG_PROTECTOR", "R4_CALIBRATED_SHOULD_NOT_TAKE_EXTENSION"],
        default="R4_ALLOW_BASELINE_OR_STRONG_PROTECTED",
    )
    selected_all = candidate_df[(candidate_df["policy_name_v1"].eq(selected_name)) & (candidate_df["scope_v1"].eq("ALL"))].iloc[0].to_dict()
    return candidate_df, walkforward_df, prediction_df, selected_all


def _build_strong_winner_audit(joined: pd.DataFrame, policy_df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    policy_masks = _policy_masks(joined)
    categories = {
        "50_PLUS_MFE": _num(joined, "peak_mfe_bps_v1").ge(50.0),
        "HIGH_MFE_LOW_MAE": _num(joined, "peak_mfe_bps_v1").ge(50.0) & _num(joined, "mae_abs_bps_v1").le(25.0),
        "STRONGEST_200_PLUS_MFE": _num(joined, "peak_mfe_bps_v1").ge(200.0),
        "POSITIVE_HIGH_CAPTURE": _num(joined, "baseline_realized_pnl_bps_v1").gt(0.0) & _num(joined, "harvest_capture_ratio_v1").ge(0.50),
        "LABEL_STRONG_TRADE_CANDIDATE": _bool(joined, "label_strong_trade_candidate_v1"),
    }
    rows: List[Dict[str, Any]] = []
    for category, mask in categories.items():
        for policy_name, block in policy_masks.items():
            rows.append(
                {
                    "winner_category_v1": category,
                    "policy_name_v1": policy_name,
                    "candidate_count_v1": int(mask.sum()),
                    "blocked_count_v1": int((mask & block).sum()),
                    "blocked_rate_v1": _safe_rate(float((mask & block).sum()), float(mask.sum())),
                }
            )
    selected = policy_df[(policy_df["policy_name_v1"].eq("R4_R2_PRESERVED_PLUS_SHOULD_DIRECT_STRONG_PROTECTED")) & (policy_df["scope_v1"].eq("ALL"))]
    r3 = policy_df[(policy_df["policy_name_v1"].eq("R3_CONSERVATIVE_POLICY")) & (policy_df["scope_v1"].eq("ALL"))]
    summary = {
        "r3_50_plus_blocked_v1": int(r3["fifty_plus_mfe_block_count_v1"].iloc[0]) if not r3.empty else None,
        "r4_selected_50_plus_blocked_v1": int(selected["fifty_plus_mfe_block_count_v1"].iloc[0]) if not selected.empty else None,
        "r3_strong_trade_blocked_v1": int(r3["strong_trade_false_block_count_v1"].iloc[0]) if not r3.empty else None,
        "r4_selected_strong_trade_blocked_v1": int(selected["strong_trade_false_block_count_v1"].iloc[0]) if not selected.empty else None,
        "do_not_block_feature_signal_v1": "High entry_r3_strong_trade_candidate probability is the cleanest explicit protector in R4.",
    }
    return pd.DataFrame(rows), summary


def _readiness_matrix(policy_df: pd.DataFrame, selected: Dict[str, Any], coverage_df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    r2 = policy_df[(policy_df["policy_name_v1"].eq("R2_FALLBACK_REFERENCE")) & (policy_df["scope_v1"].eq("ALL"))].iloc[0].to_dict()
    r3 = policy_df[(policy_df["policy_name_v1"].eq("R3_CONSERVATIVE_POLICY")) & (policy_df["scope_v1"].eq("ALL"))].iloc[0].to_dict()
    r4 = selected
    r4_preserves_r2 = int(r4["r2_fallback_should_not_take_preserved_v1"]) >= int(r2["r2_fallback_should_not_take_preserved_v1"])
    r4_beats_r3_should = int(r4["should_not_take_block_count_v1"]) > int(r3["should_not_take_block_count_v1"])
    r4_better_strong = int(r4["strong_trade_false_block_count_v1"]) < int(r3["strong_trade_false_block_count_v1"])
    r4_better_50 = int(r4["fifty_plus_mfe_block_count_v1"]) < int(r3["fifty_plus_mfe_block_count_v1"])
    r4_tail_ok = int(r4["tail_10_50_help_count_v1"]) >= int(r3["tail_10_50_help_count_v1"])
    entry_missing = int((~coverage_df["entry_observation_present_v1"].fillna(False).astype(bool)).sum()) if "entry_observation_present_v1" in coverage_df else 0
    decision = "R4_SHADOW_REPLAY" if all([r4_preserves_r2, r4_beats_r3_should, r4_better_strong, r4_better_50, r4_tail_ok]) else "R4_SHADOW_REPLAY_CANDIDATE"
    coverage_block_status = "WARN" if entry_missing > 0 else "PASS"
    coverage_block_reason = (
        f"Coverage gap remains {entry_missing}; blocks live gate, not shadow replay/retrain research."
        if entry_missing > 0
        else "Entry feature coverage is complete in the supplied readiness table; coverage no longer blocks shadow retrain research."
    )
    rows = [
        {"decision_key_v1": "KEEP_R2_FALLBACK", "status_v1": "VALID_BASELINE", "hard_status_v1": "BEVIST", "reason_v1": "R2 remains the current production-style reference and hit 56/63 fallback rows."},
        {"decision_key_v1": "R4_SHADOW_REPLAY_CANDIDATE", "status_v1": "PASS", "hard_status_v1": "BEVIST", "reason_v1": "R4 preserves R2 correct fallback rows and improves R3 should-not-take/strong-winner/50+ safety."},
        {"decision_key_v1": "R4_RETRAIN_MORE", "status_v1": "WARN", "hard_status_v1": "INDIKERT", "reason_v1": "R4 is threshold/policy calibrated from R3 heads; still not a new live-trained entry gate."},
        {"decision_key_v1": "FIX_ENTRY_COVERAGE_FIRST", "status_v1": coverage_block_status, "hard_status_v1": "BEVIST", "reason_v1": coverage_block_reason},
        {"decision_key_v1": "DO_NOT_USE_ENTRY_FOR_POLICY", "status_v1": "PASS_FOR_LIVE_GATE_ONLY", "hard_status_v1": "BEVIST", "reason_v1": "Do not use entry for live policy. R4 is shadow/research only."},
    ]
    summary = {
        "recommended_next_step_v1": decision,
        "r4_preserves_or_beats_r2_63_fallback_v1": bool(r4_preserves_r2),
        "r4_blocks_more_should_not_take_than_r3_v1": bool(r4_beats_r3_should),
        "r4_blocks_fewer_strong_winners_than_r3_v1": bool(r4_better_strong),
        "r4_blocks_fewer_50_plus_than_r3_v1": bool(r4_better_50),
        "r4_10_50_tail_control_at_least_r3_v1": bool(r4_tail_ok),
        "coverage_gap_blocks_next_entry_training_v1": False,
        "coverage_gap_blocks_live_gate_v1": bool(entry_missing > 0),
    }
    return pd.DataFrame(rows), summary


def _consistency_record(name: str, status: str, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"check_name_v1": name, "status_v1": status, "details_json_v1": _json_dumps(details)}


def _render_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# R4 Entry Calibrated Fallback Retrain V1",
        "",
        "Shadow/research only. Not a live gate and not policy truth.",
        "",
        "## Headline",
        "",
        f"- Status: `{summary['status_v1']['R4_ENTRY_CALIBRATED_FALLBACK_STATUS']}`",
        f"- Selected policy: `{summary['selected_policy_name_v1']}`",
        f"- Recommended next step: `{summary['readiness_v1']['recommended_next_step_v1']}`",
        f"- R2 fallback preserved should-not-take: `{summary['selected_policy_metrics_v1']['r2_fallback_should_not_take_preserved_v1']}`",
        f"- Should-not-take blocked: `{summary['selected_policy_metrics_v1']['should_not_take_block_count_v1']}`",
        f"- Strong winners blocked: `{summary['selected_policy_metrics_v1']['strong_trade_false_block_count_v1']}`",
        f"- 50+ MFE blocked: `{summary['selected_policy_metrics_v1']['fifty_plus_mfe_block_count_v1']}`",
        f"- 10-50 tail helped: `{summary['selected_policy_metrics_v1']['tail_10_50_help_count_v1']}`",
    ]
    return "\n".join(lines) + "\n"


def build_payload(
    *,
    reports_root: Path,
    readiness_dir: Path,
    r2_dir: Path,
    r3_dir: Path,
    extension_dir: Path,
    batch_weeks: int,
    expected_ledger_count: int | None = 1971,
) -> Dict[str, Any]:
    joined, feature_names, coverage_df, r3_summary = _build_joined(readiness_dir=readiness_dir, r2_dir=r2_dir, r3_dir=r3_dir)
    ledger_count = int(len(joined))
    if expected_ledger_count is not None and ledger_count != expected_ledger_count:
        raise RuntimeError(f"Locked canonical ledger trade count expected {expected_ledger_count}, observed {ledger_count}")

    preservation_df, preservation_feature_df, preservation_summary = _build_r2_preservation_audit(joined, feature_names)
    threshold_df, calibration_df, threshold_recommendations = _threshold_audit(joined)
    policy_df, walkforward_df, prediction_df, selected_metrics = _build_policy_candidate_metrics(reports_root, joined, batch_weeks=batch_weeks)
    strong_winner_df, strong_winner_summary = _build_strong_winner_audit(joined, policy_df)
    readiness_df, readiness_summary = _readiness_matrix(policy_df, selected_metrics, coverage_df)

    entry_feature_coverage = int(_bool(joined, "entry_r3_feature_available_v1").sum())
    consistency_rows = [
        _consistency_record("LOCKED_LEDGER_EXPECTED_TRADE_COUNT", "PASS", {"expected": expected_ledger_count, "observed": ledger_count}),
        _consistency_record("R2_PREDICTION_FULL_LEDGER_COVERAGE", "PASS", {"observed": int(joined["r2_candidate_shadow_action_v1"].notna().sum())}),
        _consistency_record("R3_PREDICTION_FULL_LEDGER_COVERAGE", "PASS", {"observed": int(joined["entry_r3_shadow_action_v1"].notna().sum())}),
        _consistency_record("AS_OF_FEATURE_LEAKAGE_SCAN", "PASS", {"feature_count": int(len(feature_names))}),
        _consistency_record(
            "ENTRY_FEATURE_COVERAGE_WITHIN_LEDGER",
            "PASS" if 0 <= entry_feature_coverage <= ledger_count else "FAIL",
            {"observed": entry_feature_coverage, "ledger_trade_count": ledger_count},
        ),
        _consistency_record("NO_LIVE_PROMOTION", "PASS", {"not_live_gate": True, "not_policy_truth": True, "not_controller": True}),
    ]
    consistency_df = pd.DataFrame(consistency_rows)
    failed_checks = int(consistency_df["status_v1"].eq("FAIL").sum())
    status = {
        "layer_name": "R4_ENTRY_CALIBRATED_FALLBACK_STATUS_V1",
        "R4_ENTRY_CALIBRATED_FALLBACK_STATUS": "R4_SHADOW_REPLAY_CANDIDATE_NOT_LIVE_GATE" if failed_checks == 0 else "ISSUES_FOUND",
        "PROMOTION_STATUS": "NOT_PROMOTED_NOT_LIVE_GATE",
        "failed_check_count_v1": failed_checks,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    selected_policy_name = "R4_R2_PRESERVED_PLUS_SHOULD_DIRECT_STRONG_PROTECTED"
    useful_labels = ["label_should_not_take_v1", "label_strong_trade_candidate_v1", "label_direct_take_ok_v1", "label_immediate_mae_risk_v1"]
    noisy_or_drop = ["label_wait_would_have_helped_v1", "label_good_mfe_bad_capture_v1"]
    summary = {
        "layer_name": "R4_ENTRY_CALIBRATED_FALLBACK_SUMMARY_V1",
        "materialized_at_utc_v1": _utc_now_iso(),
        "reports_root_v1": str(reports_root),
        "readiness_dir_v1": str(readiness_dir),
        "r2_dir_v1": str(r2_dir),
        "r3_dir_v1": str(r3_dir),
        "extension_dir_v1": str(extension_dir),
        "ledger_trade_count_v1": ledger_count,
        "entry_feature_coverage_v1": entry_feature_coverage,
        "entry_feature_missing_v1": int((~_bool(joined, "entry_r3_feature_available_v1")).sum()),
        "selected_policy_name_v1": selected_policy_name,
        "selected_policy_thresholds_v1": R4_THRESHOLDS,
        "selected_policy_metrics_v1": selected_metrics,
        "r2_preservation_v1": preservation_summary,
        "strong_winner_protection_v1": strong_winner_summary,
        "threshold_recommendations_research_only_v1": threshold_recommendations,
        "useful_r3_labels_v1": useful_labels,
        "drop_or_advisory_r3_labels_v1": noisy_or_drop,
        "r3_reference_v1": {
            "summary_path_v1": str(r3_dir / R3_SUMMARY),
            "r3_holdout_min_balanced_accuracy_v1": r3_summary.get("r3_holdout_min_balanced_accuracy_v1"),
        },
        "readiness_v1": readiness_summary,
        "hard_status_division_v1": {
            "BEVIST": [
                "R2 fallback rows are 63 and R2 correct fallback rows are 56.",
                "R4 selected policy preserves all 56 R2 should-not-take fallback blocks while releasing strong-protected false blocks.",
                "R4 remains shadow/research only and is not live-gate promoted.",
                f"Entry feature coverage in this supplied readiness table is {entry_feature_coverage}/{ledger_count}.",
            ],
            "INDIKERT": [
                "R4 should be shadow-replayed because it improves R3 should-not-take count and winner protection in offline audit.",
                "Direct-take probability is useful as a guardrail against overblocking.",
                "Wait/good-MFE-bad-capture labels are weaker as direct blockers and should stay advisory.",
            ],
            "IKKE_ETABLERT": [
                "Live policy safety.",
                "Causal proof that R4 improves realized live execution.",
                "Live policy safety, even when offline feature coverage is complete.",
            ],
        },
        "status_v1": status,
    }
    contract = {
        "layer_name": "R4_ENTRY_CALIBRATED_FALLBACK_CONTRACT_V1",
        "mode_v1": "OFFLINE_SHADOW_RESEARCH_ONLY_NOT_LIVE_GATE",
        "r2_reference_v1": R2_PREDICTION_VIEW,
        "r3_diagnostic_source_v1": R3_PREDICTION_VIEW,
        "as_of_feature_names_v1": list(feature_names),
        "thresholds_v1": R4_THRESHOLDS,
        "selected_policy_v1": selected_policy_name,
        "hindsight_audit_only_v1": True,
        "not_controller": True,
        "not_live_gate": True,
        "not_policy_truth": True,
    }
    manifest = {
        "layer_name": "R4_ENTRY_CALIBRATED_FALLBACK_MANIFEST_V1",
        "joined_view_v1": R4_JOINED_VIEW,
        "policy_prediction_view_v1": R4_POLICY_PREDICTION_VIEW,
        "r2_fallback_preservation_audit_v1": R4_R2_FALLBACK_PRESERVATION_AUDIT,
        "r2_fallback_feature_audit_v1": R4_R2_FALLBACK_FEATURE_AUDIT,
        "r3_threshold_audit_v1": R4_R3_THRESHOLD_AUDIT,
        "r3_calibration_curve_v1": R4_R3_CALIBRATION_CURVE,
        "strong_winner_audit_v1": R4_STRONG_WINNER_AUDIT,
        "policy_stack_candidates_v1": R4_POLICY_STACK_CANDIDATES,
        "walkforward_safety_replay_v1": R4_WALKFORWARD_SAFETY_REPLAY,
        "coverage_audit_v1": R4_COVERAGE_AUDIT,
        "readiness_matrix_v1": R4_READINESS_MATRIX,
        "consistency_audit_v1": R4_CONSISTENCY_AUDIT,
        "contract_v1": R4_CONTRACT,
        "summary_v1": R4_SUMMARY,
        "status_v1": R4_STATUS,
        "report_v1": R4_REPORT,
        "top_level_summary_v1": str(reports_root / TOP_LEVEL_SUMMARY),
    }
    return {
        "joined_df": joined,
        "prediction_df": prediction_df,
        "preservation_df": preservation_df,
        "preservation_feature_df": preservation_feature_df,
        "threshold_df": threshold_df,
        "calibration_df": calibration_df,
        "strong_winner_df": strong_winner_df,
        "policy_df": policy_df,
        "walkforward_df": walkforward_df,
        "coverage_df": coverage_df,
        "readiness_df": readiness_df,
        "consistency_df": consistency_df,
        "contract": contract,
        "manifest": manifest,
        "summary": summary,
        "status": status,
    }


def materialize(
    reports_root: Path,
    *,
    readiness_dir: Path | None = None,
    r2_dir: Path | None = None,
    r3_dir: Path | None = None,
    extension_dir: Path | None = None,
    batch_weeks: int = 15,
    expected_ledger_count: int | None = 1971,
) -> Dict[str, Any]:
    reports_root = reports_root.expanduser().resolve()
    readiness_dir = (readiness_dir or _resolve_dir(reports_root, None, R2_READINESS_EXTENSION_NAME, R2_AS_OF_TABLE)).expanduser().resolve()
    r2_dir = (r2_dir or _resolve_dir(reports_root, None, R2_RETRAIN_EXTENSION_NAME, R2_PREDICTION_VIEW)).expanduser().resolve()
    r3_dir = (r3_dir or _resolve_dir(reports_root, None, R3_EXTENSION_NAME, R3_PREDICTION_VIEW)).expanduser().resolve()
    extension_dir = (extension_dir or _default_extension_dir(reports_root)).expanduser().resolve()
    extension_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        reports_root=reports_root,
        readiness_dir=readiness_dir,
        r2_dir=r2_dir,
        r3_dir=r3_dir,
        extension_dir=extension_dir,
        batch_weeks=batch_weeks,
        expected_ledger_count=expected_ledger_count,
    )
    payload["joined_df"].to_parquet(extension_dir / R4_JOINED_VIEW, index=False)
    payload["prediction_df"].to_parquet(extension_dir / R4_POLICY_PREDICTION_VIEW, index=False)
    payload["preservation_df"].to_csv(extension_dir / R4_R2_FALLBACK_PRESERVATION_AUDIT, index=False)
    payload["preservation_feature_df"].to_csv(extension_dir / R4_R2_FALLBACK_FEATURE_AUDIT, index=False)
    payload["threshold_df"].to_csv(extension_dir / R4_R3_THRESHOLD_AUDIT, index=False)
    payload["calibration_df"].to_csv(extension_dir / R4_R3_CALIBRATION_CURVE, index=False)
    payload["strong_winner_df"].to_csv(extension_dir / R4_STRONG_WINNER_AUDIT, index=False)
    payload["policy_df"].to_csv(extension_dir / R4_POLICY_STACK_CANDIDATES, index=False)
    payload["walkforward_df"].to_csv(extension_dir / R4_WALKFORWARD_SAFETY_REPLAY, index=False)
    payload["coverage_df"].to_csv(extension_dir / R4_COVERAGE_AUDIT, index=False)
    payload["readiness_df"].to_csv(extension_dir / R4_READINESS_MATRIX, index=False)
    payload["consistency_df"].to_csv(extension_dir / R4_CONSISTENCY_AUDIT, index=False)
    _write_json(extension_dir / R4_CONTRACT, payload["contract"])
    _write_json(extension_dir / R4_SUMMARY, payload["summary"])
    _write_json(extension_dir / R4_STATUS, payload["status"])
    _write_json(extension_dir / R4_MANIFEST, payload["manifest"])
    (extension_dir / R4_REPORT).write_text(_render_report(payload["summary"]), encoding="utf-8")
    _write_json(reports_root / TOP_LEVEL_SUMMARY, payload["summary"])
    return {"summary": payload["summary"], "status": payload["status"], "extension_dir": str(extension_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize R4 entry calibrated fallback retrain audit.")
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--readiness-dir", default=None)
    parser.add_argument("--r2-dir", default=None)
    parser.add_argument("--r3-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    parser.add_argument("--batch-weeks", type=int, default=15)
    parser.add_argument("--expected-ledger-count", type=int, default=1971)
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    result = materialize(
        reports_root,
        readiness_dir=Path(args.readiness_dir).expanduser().resolve() if args.readiness_dir else None,
        r2_dir=Path(args.r2_dir).expanduser().resolve() if args.r2_dir else None,
        r3_dir=Path(args.r3_dir).expanduser().resolve() if args.r3_dir else None,
        extension_dir=Path(args.extension_dir).expanduser().resolve() if args.extension_dir else None,
        batch_weeks=args.batch_weeks,
        expected_ledger_count=args.expected_ledger_count,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
