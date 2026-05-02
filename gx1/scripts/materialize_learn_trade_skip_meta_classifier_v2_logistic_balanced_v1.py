#!/usr/bin/env python3
"""V2 of the trade-skip meta-classifier: logistic regression + class balance.

Background
----------
V1 (`LEARN_TRADE_SKIP_META_CLASSIFIER_AT_TRADE_OPEN_V1`) used closed-form
ridge regression on the binary label. Two limitations surfaced:

  1. Ridge MSE on a binary target predicts toward the class-mean, biasing
     predicted probabilities toward the majority class. With label rate
     14.1%, most p_skip values fell below 0.30, so the [0.30, 0.70] grid
     skipped almost nothing useful.
  2. Even with the V1 extended grid {0.10, 0.15, ...}, val tuning picked
     threshold 0.15 (test lift +120 bps), but threshold 0.10 on test
     would have given +1071 bps - a gap caused by val/test distribution
     differences and the squashed prediction range.

This V2 gate addresses both:

  - Switches to logistic regression with `class_weight='balanced'`. The
    balanced weighting compensates for the 86/14 class imbalance, so
    predicted probabilities are no longer squashed toward 0; the
    decision boundary near 0.5 becomes meaningful.
  - Uses a finer threshold grid {0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
    0.60, 0.65, 0.70, 0.75} that covers the natural range of a balanced
    classifier.
  - Reuses V1's feature set and per-trade pipeline (no schema change).
  - Adds an oracle-fraction-captured headline metric so the test gap to
    oracle is explicit.

Research-only; the trained classifier is NOT promoted to runtime; no
exit_manager / live_features / entry_manager modification.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.linear_model import LogisticRegression

from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)
from gx1.scripts import (
    materialize_learn_trade_skip_meta_classifier_at_trade_open_v1 as v1_gate,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_LOGISTIC_BALANCED_V1"

INPUT_V1_GATE_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "LEARN_TRADE_SKIP_META_CLASSIFIER_AT_TRADE_OPEN_V1_20260430T055350Z_LOCK"
)
INPUT_RECOVERY_ROOT = v1_gate.INPUT_RECOVERY_ROOT
INPUT_SPLIT_ROOT = v1_gate.INPUT_SPLIT_ROOT
INPUT_V2_CONTRACT_ROOT = v1_gate.INPUT_V2_CONTRACT_ROOT
BASE34_M5_FEATURES_PATH = v1_gate.BASE34_M5_FEATURES_PATH

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
SEED_V1 = 20260430

# Logistic regression hyperparameters (closed-form; no manual loop).
LOGREG_C = 1.0  # inverse regularization strength; 1.0 is sklearn default.
LOGREG_MAX_ITER = 200
LOGREG_PENALTY = "l2"
LOGREG_CLASS_WEIGHT = "balanced"

# Finer threshold grid for a properly-calibrated logistic classifier.
THRESHOLD_GRID: list[float] = [
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
]


ALLOWED_FINAL_STATUSES = {
    "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_PASS_LIFTS_V1_BASELINE",
    "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_PASS_TUNED_THRESHOLD_LIFTS_TEST_PNL",
    "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_PARTIAL_TIES_V1_BASELINE",
    "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_PARTIAL_DEGRADES_VS_V1",
    "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_BLOCKED_BY_INPUT_LOCK_MISSING",
}

ALLOWED_NEXT_ACTIONS = {
    "RUN_EXIT_IQL_V3_WITH_PER_BAR_XGB_TRANSFORMER_SIGNAL_V1",
    "COMBINE_SKIP_CLASSIFIER_V2_WITH_EXIT_IQL_V3_V1",
    "REPAIR_SKIP_CLASSIFIER_V2_BEFORE_PROMOTION_V1",
    "HOLD_SKIP_CLASSIFIER_RESEARCH_UNTIL_DATA_FIXED_V1",
}

# Re-use V1's pure helpers
_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows
_write_report = contract_gate._write_report
_read_json = contract_gate._read_json
_file_hash = contract_gate._file_hash
_python_manifest = contract_gate._python_manifest


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_explicit_artifact_roots(paths: Iterable[Path]) -> bool:
    return contract_gate.validate_explicit_artifact_roots(paths)


def validate_no_forbidden_actions(**kwargs: Any) -> dict[str, Any]:
    return contract_gate.validate_no_forbidden_actions(**kwargs)


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_no_deprecated_revival(script_path: Path) -> bool:
    text = script_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.lstrip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        for fragment in QUARANTINE_FORBIDDEN_PATH_FRAGMENTS:
            if fragment in stripped:
                raise RuntimeError("DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN")
    return True


def _load_inputs() -> dict[str, Any]:
    roots = [INPUT_V1_GATE_ROOT, INPUT_RECOVERY_ROOT, INPUT_SPLIT_ROOT, INPUT_V2_CONTRACT_ROOT]
    validate_explicit_artifact_roots(roots)
    required = {
        "v1_gate_summary": INPUT_V1_GATE_ROOT / "summary_v1.json",
        "v1_locked_test_evaluation": INPUT_V1_GATE_ROOT / "locked_test_evaluation_v1.json",
        "recovery_per_trade": INPUT_RECOVERY_ROOT
        / "entry_snapshot_signals_per_trade_v1.parquet",
        "split_locked_dataset": INPUT_SPLIT_ROOT
        / "split_locked_augmented_dataset_v1.parquet",
        "v2_state_contract": INPUT_V2_CONTRACT_ROOT / "state_feature_contract_v2.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_INPUT_LOCKS: {missing}")
    if not BASE34_M5_FEATURES_PATH.exists():
        raise RuntimeError(
            f"BASE34_M5_FEATURES_PATH_NOT_FOUND: {BASE34_M5_FEATURES_PATH}"
        )
    return {
        "required_paths": required,
        "v1_gate_summary": _read_json(required["v1_gate_summary"]),
        "v1_locked_test_evaluation": _read_json(
            required["v1_locked_test_evaluation"]
        ),
        "v2_state_contract": _read_json(required["v2_state_contract"]),
        "base34_path": BASE34_M5_FEATURES_PATH,
    }


def _train_logistic(
    X_train: np.ndarray, y_train: np.ndarray
) -> LogisticRegression:
    """Fit balanced-weight logistic regression on training rows."""
    # Drop the intercept column from the design matrix - sklearn's
    # LogisticRegression learns its own intercept internally.
    X_train_no_intercept = X_train[:, 1:]
    model = LogisticRegression(
        penalty=LOGREG_PENALTY,
        C=LOGREG_C,
        class_weight=LOGREG_CLASS_WEIGHT,
        solver="lbfgs",
        max_iter=LOGREG_MAX_ITER,
        random_state=SEED_V1,
    )
    model.fit(X_train_no_intercept, y_train.astype(int))
    return model


def _predict_p_skip(
    model: LogisticRegression, X: np.ndarray
) -> np.ndarray:
    X_no_intercept = X[:, 1:]
    proba = model.predict_proba(X_no_intercept)
    # Class 1 is should_skip; predict_proba returns columns in classes_ order
    classes = list(model.classes_)
    if 1 not in classes:
        raise RuntimeError("LOGREG_DID_NOT_LEARN_POSITIVE_CLASS")
    pos_idx = classes.index(1)
    return proba[:, pos_idx].astype(float)


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    inputs = _load_inputs()
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    validate_no_deprecated_revival(Path(__file__))
    forbidden_audit = validate_no_forbidden_actions(
        adapter=False,
        r6=False,
        iql_production=False,
        package=False,
        freeze=False,
        promo=False,
        live=False,
        optuna=False,
        broad_sweep=False,
    )

    # Reuse V1 helpers for projection.
    label_formula_audit = v1_gate.validate_label_formula_against_v2_contract(
        inputs["v2_state_contract"]
    )
    _write_json(artifact_root / "label_formula_audit_v1.json", label_formula_audit)

    trades = v1_gate._load_trade_outcomes_concat()
    recovery = pd.read_parquet(inputs["required_paths"]["recovery_per_trade"])
    per_trade, projection_audit = v1_gate._project_per_trade_features(
        trades, recovery, BASE34_M5_FEATURES_PATH
    )
    per_trade_split, split_join_audit = v1_gate._join_split_assignment(
        per_trade, inputs["required_paths"]["split_locked_dataset"]
    )
    split_isolation_audit = v1_gate.audit_split_isolation(per_trade_split)

    per_trade_train = per_trade_split[per_trade_split["primary_split_v1"] == "train"]
    norm = v1_gate._fit_train_normalization(per_trade_train)
    train_only_audit = v1_gate.audit_train_only_normalization(per_trade_split, norm)
    _write_json(artifact_root / "training_normalization_v1.json", norm)

    X_full, feature_names = v1_gate._build_state_matrix(per_trade_split, norm)
    train_mask = (per_trade_split["primary_split_v1"] == "train").to_numpy()
    val_mask = (per_trade_split["primary_split_v1"] == "val").to_numpy()
    test_mask = (per_trade_split["primary_split_v1"] == "test").to_numpy()
    no_shortcut_audit = v1_gate.audit_no_shortcut_at_train_time(feature_names)

    y_full = per_trade_split["should_skip_v1"].astype(int).to_numpy()

    # Train logistic regression with class-balance weighting.
    logreg = _train_logistic(X_full[train_mask], y_full[train_mask])
    p_skip_full = _predict_p_skip(logreg, X_full)

    model_summary = {
        "model_v1": "SKLEARN_LOGISTIC_REGRESSION_BALANCED",
        "feature_count_v1": len(feature_names),
        "feature_names_v1": feature_names,
        "logreg_C_v1": LOGREG_C,
        "logreg_max_iter_v1": LOGREG_MAX_ITER,
        "logreg_penalty_v1": LOGREG_PENALTY,
        "logreg_class_weight_v1": LOGREG_CLASS_WEIGHT,
        "seed_v1": SEED_V1,
        "train_row_count_v1": int(train_mask.sum()),
        "coef_v1": logreg.coef_.tolist(),
        "intercept_v1": logreg.intercept_.tolist(),
        "n_iter_v1": int(logreg.n_iter_[0]),
        "classes_v1": logreg.classes_.tolist(),
    }
    _write_json(artifact_root / "trained_model_v1.json", model_summary)

    # Predicted-probability distribution per split.
    pred_dist: dict[str, dict[str, float]] = {}
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        if not mask.any():
            continue
        p = p_skip_full[mask]
        pred_dist[split_name] = {
            "min_v1": float(p.min()),
            "p05_v1": float(np.quantile(p, 0.05)),
            "p25_v1": float(np.quantile(p, 0.25)),
            "p50_v1": float(np.quantile(p, 0.50)),
            "p75_v1": float(np.quantile(p, 0.75)),
            "p95_v1": float(np.quantile(p, 0.95)),
            "max_v1": float(p.max()),
            "mean_v1": float(p.mean()),
            "std_v1": float(p.std(ddof=0)),
            "n_v1": int(mask.sum()),
        }
    _write_json(
        artifact_root / "predicted_skip_probability_distribution_v1.json", pred_dist
    )

    # Threshold sweep on each split.
    threshold_metrics: dict[str, list[dict[str, Any]]] = {}
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        per_trade_sub = per_trade_split[mask].reset_index(drop=True)
        if per_trade_sub.empty:
            continue
        p_skip_split = p_skip_full[mask]
        rows: list[dict[str, Any]] = []
        for thr in THRESHOLD_GRID:
            m = v1_gate._evaluate_threshold(per_trade_sub, p_skip_split, thr)
            m["split_v1"] = split_name
            rows.append(m)
        threshold_metrics[split_name] = rows
    flat_threshold_rows = [r for rows in threshold_metrics.values() for r in rows]
    _write_rows(
        artifact_root / "threshold_sweep_metrics_v1.csv", flat_threshold_rows
    )
    _write_json(
        artifact_root / "threshold_sweep_metrics_v1.json",
        {"row_count_v1": len(flat_threshold_rows), "rows_v1": flat_threshold_rows},
    )

    # Tune threshold on val.
    val_rows = threshold_metrics.get("val", [])
    if not val_rows:
        raise RuntimeError("EMPTY_VAL_SPLIT_FOR_THRESHOLD_TUNING")
    best_val = max(val_rows, key=lambda r: r["pnl_taken_v1"])
    tuned_threshold = float(best_val["threshold_v1"])

    per_trade_test = per_trade_split[test_mask].reset_index(drop=True)
    p_skip_test = p_skip_full[test_mask]
    test_at_locked = v1_gate._evaluate_threshold(
        per_trade_test, p_skip_test, tuned_threshold
    )
    test_at_locked["split_v1"] = "test"
    _write_json(
        artifact_root / "locked_test_evaluation_v1.json",
        {**test_at_locked, "tuned_threshold_v1": tuned_threshold},
    )

    # Oracle skip per split.
    oracle_per_split = {
        "train": v1_gate._evaluate_oracle_skip(
            per_trade_split[train_mask].reset_index(drop=True)
        ),
        "val": v1_gate._evaluate_oracle_skip(
            per_trade_split[val_mask].reset_index(drop=True)
        ),
        "test": v1_gate._evaluate_oracle_skip(per_trade_test),
    }
    _write_json(artifact_root / "oracle_skip_per_split_v1.json", oracle_per_split)

    # Comparison vs V1 gate's locked test evaluation.
    v1_locked = inputs["v1_locked_test_evaluation"]
    v1_test_lift = float(v1_locked.get("pnl_lift_vs_no_skip_v1", 0.0))
    v2_test_lift = float(test_at_locked["pnl_lift_vs_no_skip_v1"])
    v1_vs_v2_compare = {
        "v1_tuned_threshold_v1": v1_locked.get("tuned_threshold_v1"),
        "v1_test_pnl_no_skip_v1": v1_locked.get("pnl_no_skip_v1"),
        "v1_test_pnl_with_skip_v1": v1_locked.get("pnl_taken_v1"),
        "v1_test_pnl_lift_v1": v1_test_lift,
        "v1_test_precision_v1": v1_locked.get("precision_v1"),
        "v1_test_recall_v1": v1_locked.get("recall_v1"),
        "v1_test_trades_skipped_v1": v1_locked.get("trades_skipped_v1"),
        "v2_tuned_threshold_v1": tuned_threshold,
        "v2_test_pnl_no_skip_v1": test_at_locked["pnl_no_skip_v1"],
        "v2_test_pnl_with_skip_v1": test_at_locked["pnl_taken_v1"],
        "v2_test_pnl_lift_v1": v2_test_lift,
        "v2_test_precision_v1": test_at_locked["precision_v1"],
        "v2_test_recall_v1": test_at_locked["recall_v1"],
        "v2_test_trades_skipped_v1": test_at_locked["trades_skipped_v1"],
        "delta_v2_minus_v1_test_lift_v1": v2_test_lift - v1_test_lift,
    }
    _write_json(
        artifact_root / "v1_vs_v2_comparison_v1.json", v1_vs_v2_compare
    )

    audits = [
        label_formula_audit,
        projection_audit,
        split_join_audit,
        split_isolation_audit,
        train_only_audit,
        no_shortcut_audit,
    ]
    _write_json(
        artifact_root / "training_audits_v1.json",
        {"audit_count_v1": len(audits), "audits_v1": audits},
    )

    repro = {
        "layer_name": "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_REPRODUCIBILITY_AUDIT_V1",
        "model_v1": "SKLEARN_LOGISTIC_REGRESSION_BALANCED",
        "feature_count_v1": len(feature_names),
        "logreg_C_v1": LOGREG_C,
        "seed_v1": SEED_V1,
        "tuned_threshold_v1": tuned_threshold,
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    # go-no-go decision based on V1 vs V2 lift comparison.
    delta = v2_test_lift - v1_test_lift
    oracle_test_lift = float(oracle_per_split["test"]["pnl_lift_vs_no_skip_v1"])
    captured = (
        v2_test_lift / oracle_test_lift if oracle_test_lift else None
    )
    headline = {
        "v2_tuned_threshold_v1": tuned_threshold,
        "v2_test_pnl_lift_v1": v2_test_lift,
        "v1_test_pnl_lift_v1": v1_test_lift,
        "delta_v2_minus_v1_test_lift_v1": delta,
        "v2_captured_fraction_of_oracle_v1": captured,
        "v2_test_precision_v1": test_at_locked["precision_v1"],
        "v2_test_recall_v1": test_at_locked["recall_v1"],
        "v2_test_trades_skipped_v1": test_at_locked["trades_skipped_v1"],
    }

    if delta > 100.0:
        status = "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_PASS_LIFTS_V1_BASELINE"
        next_action = "COMBINE_SKIP_CLASSIFIER_V2_WITH_EXIT_IQL_V3_V1"
        recommendation = (
            f"V2 logistic balanced lifts test PNL by {v2_test_lift:.0f} bps, "
            f"a delta of {delta:+.0f} vs V1 ({v1_test_lift:.0f}). Captured "
            f"{captured*100:.1f}% of oracle. Next: combine V2 skip with V3 exit IQL."
        )
    elif v2_test_lift > 0.0 and abs(delta) <= 100.0:
        status = "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_PASS_TUNED_THRESHOLD_LIFTS_TEST_PNL"
        next_action = "RUN_EXIT_IQL_V3_WITH_PER_BAR_XGB_TRANSFORMER_SIGNAL_V1"
        recommendation = (
            f"V2 logistic balanced test lift {v2_test_lift:.0f} bps, "
            f"~comparable to V1 ({v1_test_lift:.0f}; delta {delta:+.0f}). "
            "Move on to V3 exit IQL."
        )
    elif abs(delta) <= 50.0:
        status = "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_PARTIAL_TIES_V1_BASELINE"
        next_action = "REPAIR_SKIP_CLASSIFIER_V2_BEFORE_PROMOTION_V1"
        recommendation = (
            f"V2 ties V1 (delta {delta:+.0f} bps); not worth promoting "
            "over V1 yet. Investigate before further work."
        )
    else:
        status = "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_PARTIAL_DEGRADES_VS_V1"
        next_action = "REPAIR_SKIP_CLASSIFIER_V2_BEFORE_PROMOTION_V1"
        recommendation = (
            f"V2 degrades vs V1 (delta {delta:+.0f} bps). Investigate "
            "logistic regression hyperparameters or class weighting."
        )
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "model_id_v1": "SKIP_CLASSIFIER_LOGISTIC_BALANCED",
        "feature_count_v1": len(feature_names),
        "tuned_threshold_v1": tuned_threshold,
        "test_at_locked_threshold_v1": test_at_locked,
        "oracle_per_split_v1": oracle_per_split,
        "v1_vs_v2_comparison_v1": v1_vs_v2_compare,
        "predicted_distribution_v1": pred_dist,
        "audits_v1": {a["audit_id_v1"]: a["status_v1"] for a in audits},
        "research_only_v1": True,
        "iql_training_run_v1": False,
        "iql_production_allowed_v1": False,
        "skip_classifier_promoted_to_runtime_v1": False,
        "training_blocked_v1": False,
        "next_research_gate_v1": next_action,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "entry_manager_modified_v1": False,
        "forbidden_actions_audit_v1": forbidden_audit,
    }
    _write_json(artifact_root / "summary_v1.json", summary)

    status_payload = {
        "layer_name": "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_TRAINING_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": True,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_GO_NO_GO_V1",
        "status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
        "skip_classifier_promotion_allowed_v1": False,
        "downstream_block_v1": (
            "Research-only V2 skip classifier. NOT promoted to runtime; "
            "entry_manager / exit_manager / live_features all unmodified."
        ),
    }
    _write_json(
        artifact_root
        / "learn_trade_skip_meta_classifier_v2_logistic_balanced_go_no_go_v1.json",
        go_no_go,
    )

    report_lines = [
        "# Learn Trade-Skip Meta-Classifier V2 Logistic Balanced V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: research-only; classifier NOT promoted to runtime.",
        "",
        "## Headline (test split)",
        f"- Tuned threshold (val-best): {tuned_threshold}",
        f"- V2 test PNL lift: {v2_test_lift:+.0f} bps",
        f"- V1 test PNL lift (reference): {v1_test_lift:+.0f} bps",
        f"- Delta V2 - V1: {delta:+.0f} bps",
        f"- V2 captured fraction of oracle: {captured}",
        f"- V2 test precision: {test_at_locked['precision_v1']}",
        f"- V2 test recall: {test_at_locked['recall_v1']}",
        f"- V2 test F1: {test_at_locked['f1_v1']}",
        f"- V2 trades skipped: {test_at_locked['trades_skipped_v1']} of {test_at_locked['trade_count_v1']}",
        "",
        "## Predicted-probability distribution per split",
    ]
    for split_name, dist in pred_dist.items():
        report_lines.append(
            f"- {split_name}: median={dist['p50_v1']:.3f}, mean={dist['mean_v1']:.3f}, "
            f"p25={dist['p25_v1']:.3f}, p75={dist['p75_v1']:.3f}, "
            f"min={dist['min_v1']:.3f}, max={dist['max_v1']:.3f}"
        )
    report_lines.extend(
        [
            "",
            "## Threshold sweep (test split)",
            "",
            "| Threshold | Skip | Take | PNL no-skip | PNL with-skip | Lift | Precision | Recall | F1 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for r in threshold_metrics.get("test", []):
        report_lines.append(
            f"| {r['threshold_v1']:.2f} | {r['trades_skipped_v1']} | "
            f"{r['trades_taken_v1']} | "
            f"{r['pnl_no_skip_v1']:.0f} | {r['pnl_taken_v1']:.0f} | "
            f"{r['pnl_lift_vs_no_skip_v1']:+.0f} | "
            f"{r['precision_v1']} | {r['recall_v1']} | {r['f1_v1']} |"
        )
    report_lines.extend(
        [
            "",
            "## Threshold sweep (val split)",
            "",
            "| Threshold | Skip | Take | PNL no-skip | PNL with-skip | Lift | Precision | Recall | F1 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for r in threshold_metrics.get("val", []):
        report_lines.append(
            f"| {r['threshold_v1']:.2f} | {r['trades_skipped_v1']} | "
            f"{r['trades_taken_v1']} | "
            f"{r['pnl_no_skip_v1']:.0f} | {r['pnl_taken_v1']:.0f} | "
            f"{r['pnl_lift_vs_no_skip_v1']:+.0f} | "
            f"{r['precision_v1']} | {r['recall_v1']} | {r['f1_v1']} |"
        )
    report_lines.extend(
        [
            "",
            "## Audits",
        ]
    )
    for a in audits:
        report_lines.append(f"- `{a['audit_id_v1']}`: {a['status_v1']}")
    report_lines.extend(["", "## Recommendation", recommendation])
    _write_report(artifact_root / "report_v1.md", report_lines)

    artifact_manifest = {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "append_only_namespace_v1": "truth_e2e_sanity",
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "go_no_go": str(
                artifact_root
                / "learn_trade_skip_meta_classifier_v2_logistic_balanced_go_no_go_v1.json"
            ),
            "trained_model": str(artifact_root / "trained_model_v1.json"),
            "training_normalization": str(artifact_root / "training_normalization_v1.json"),
            "predicted_skip_probability_distribution": str(
                artifact_root / "predicted_skip_probability_distribution_v1.json"
            ),
            "threshold_sweep_metrics_csv": str(
                artifact_root / "threshold_sweep_metrics_v1.csv"
            ),
            "threshold_sweep_metrics_json": str(
                artifact_root / "threshold_sweep_metrics_v1.json"
            ),
            "locked_test_evaluation": str(artifact_root / "locked_test_evaluation_v1.json"),
            "oracle_skip_per_split": str(artifact_root / "oracle_skip_per_split_v1.json"),
            "v1_vs_v2_comparison": str(artifact_root / "v1_vs_v2_comparison_v1.json"),
            "training_audits": str(artifact_root / "training_audits_v1.json"),
            "label_formula_audit": str(artifact_root / "label_formula_audit_v1.json"),
            "reproducibility_audit": str(artifact_root / "reproducibility_audit_v1.json"),
            "report": str(artifact_root / "report_v1.md"),
        },
        "read_only_references_v1": True,
        "trained_model_v1": True,
        "not_controller_v1": True,
        "not_live_gate_v1": True,
    }
    _write_json(artifact_root / "manifest_v1.json", artifact_manifest)

    return {
        "artifact_root": str(artifact_root),
        "summary": summary,
        "status": status_payload,
        "go_no_go": go_no_go,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize LEARN_TRADE_SKIP_META_CLASSIFIER_V2_LOGISTIC_BALANCED_V1."
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = (
        Path(args.out_root).expanduser().resolve() if args.out_root else None
    )
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
