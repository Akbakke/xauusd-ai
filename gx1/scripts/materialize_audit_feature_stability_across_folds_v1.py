#!/usr/bin/env python3
"""Audit per-feature coefficient stability across walk-forward folds.

Re-runs skip-V2 (logistic balanced) and V2 IQL (5 reward variants ridge)
training on each of the 3 walk-forward folds (deterministic seed). For
each feature collects per-fold coefficient and classifies its stability:

  - STABLE: same sign across all folds AND coefficient std < 0.5 * |mean|
  - DIRECTIONAL: same sign across all folds but high variance
  - FLIPS_SIGN: coefficient changes sign between folds
  - DEAD: |mean| < 1e-6 (near-zero across all folds)

Outputs:
  - per_feature_stability_skip_v2_v1.csv: feature, per-fold coefs, classification
  - per_feature_stability_v2_iql_v1.csv: per (feature, reward, head) the same
  - keep/drop/condition recommendation per feature based on classification

This is research-only diagnostic; no model promotion. The output informs
the design of any future regime-conditioned or feature-pruned model.
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

from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)
from gx1.scripts import (
    materialize_walk_forward_validation_v1 as wf_gate,
)
from gx1.scripts import (
    materialize_learn_trade_skip_meta_classifier_at_trade_open_v1 as skip_v1_gate,
)
from gx1.scripts import (
    materialize_learn_trade_skip_meta_classifier_v2_logistic_balanced_v1 as skip_v2_gate,
)
from gx1.scripts import (
    materialize_run_exit_iql_with_v2_state_and_reward_variants_v1 as v2_train_gate,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "AUDIT_FEATURE_STABILITY_ACROSS_FOLDS_V1"

INPUT_WALK_FORWARD_ROOT = wf_gate.INPUT_COMBINED_GATE_ROOT  # share inputs
INPUT_RECOVERY_ROOT = v2_train_gate.INPUT_RECOVERY_ROOT
INPUT_SPLIT_ROOT = v2_train_gate.INPUT_SPLIT_ROOT
INPUT_V2_CONTRACT_ROOT = v2_train_gate.INPUT_V2_CONTRACT_ROOT
INPUT_PROMOTION_CRITERIA_ROOT_GLOB_PATTERN = "DEFINE_PROMOTION_CRITERIA_V1_*_LOCK"
BASE34_M5_FEATURES_PATH = v2_train_gate.BASE34_M5_FEATURES_PATH

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")
SEED_V1 = 20260430

ALLOWED_FINAL_STATUSES = {
    "AUDIT_FEATURE_STABILITY_LOCKED_V1",
    "AUDIT_FEATURE_STABILITY_BLOCKED_BY_INPUT_LOCK_MISSING_V1",
}

ALLOWED_NEXT_ACTIONS = {
    "INVESTIGATE_TRAIL_STOP_DEEP_DIVE_V1",
    "BUILD_REGIME_CONDITIONED_SKIP_V1",
    "REPAIR_FEATURE_SET_BEFORE_FURTHER_WORK_V1",
}


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


# ---------------------------------------------------------------------------
# Stability classification
# ---------------------------------------------------------------------------


def classify_feature_stability(
    feature_name: str, per_fold_coefs: list[float]
) -> dict[str, Any]:
    arr = np.array(per_fold_coefs, dtype=float)
    n = int(len(arr))
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    abs_mean = abs(mean)
    same_sign = bool(np.all(arr > 0)) or bool(np.all(arr < 0))
    near_zero = bool(np.all(np.abs(arr) < 1e-6))

    if near_zero:
        classification = "DEAD"
        recommendation = "DROP"
    elif not same_sign:
        classification = "FLIPS_SIGN"
        recommendation = "REGIME_CONDITION_OR_DROP"
    elif std < 0.5 * abs_mean:
        classification = "STABLE"
        recommendation = "KEEP"
    else:
        classification = "DIRECTIONAL"
        recommendation = "KEEP_BUT_NOTE_HIGH_VARIANCE"

    return {
        "feature_v1": feature_name,
        "n_folds_v1": n,
        "mean_v1": mean,
        "std_v1": std,
        "min_v1": float(arr.min()),
        "max_v1": float(arr.max()),
        "same_sign_across_folds_v1": same_sign,
        "near_zero_v1": near_zero,
        "classification_v1": classification,
        "recommendation_v1": recommendation,
        "per_fold_coefs_v1": per_fold_coefs,
    }


# ---------------------------------------------------------------------------
# Per-fold training (reuse walk-forward gate's helpers)
# ---------------------------------------------------------------------------


def _train_per_fold(
    inputs: dict[str, Any],
    candidate_uid_order: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """For each fold, train skip-V2 and V2 IQL (5 variants) and return:
    - skip_v2_per_fold: list of dicts {fold_id, feature_names, coefs}
    - v2_iql_per_fold: list of dicts {fold_id, reward_id, head, feature_names, coefs}
    """
    skip_v2_per_fold: list[dict[str, Any]] = []
    v2_iql_per_fold: list[dict[str, Any]] = []
    for fold in wf_gate.FOLD_DEFINITIONS:
        fold_id = fold["fold_id_v1"]
        uid_to_split = wf_gate._assign_fold_split(candidate_uid_order, fold)
        per_trade = wf_gate._build_per_trade_for_fold(inputs, uid_to_split)
        per_bar, X_full, feature_names_iql = wf_gate._build_per_bar_for_fold(
            inputs, uid_to_split
        )

        # Skip-V2 training - get the logistic regression coefficients.
        per_trade_train = per_trade[per_trade["primary_split_v1"] == "train"]
        norm_skip = skip_v1_gate._fit_train_normalization(per_trade_train)
        X_skip_full, skip_feature_names = skip_v1_gate._build_state_matrix(
            per_trade, norm_skip
        )
        train_mask_skip = (per_trade["primary_split_v1"] == "train").to_numpy()
        y_full = per_trade["should_skip_v1"].astype(int).to_numpy()
        if y_full[train_mask_skip].sum() > 0:
            logreg = skip_v2_gate._train_logistic(
                X_skip_full[train_mask_skip], y_full[train_mask_skip]
            )
            # logreg.coef_ has shape (1, n_features-1) since _train_logistic
            # drops the intercept column. Re-align to skip_feature_names by
            # prepending the intercept term.
            full_coefs = np.concatenate([logreg.intercept_, logreg.coef_[0]])
        else:
            full_coefs = np.zeros(len(skip_feature_names))
        skip_v2_per_fold.append(
            {
                "fold_id_v1": fold_id,
                "feature_names_v1": list(skip_feature_names),
                "coefs_v1": full_coefs.tolist(),
            }
        )

        # V2 IQL training - 5 reward variants × 2 heads.
        train_mask_iql = (per_bar["primary_split_v1"] == "train").to_numpy()
        per_bar_train = per_bar[per_bar["primary_split_v1"] == "train"]
        for variant in v2_train_gate.REWARD_VARIANTS_V2:
            v_id = variant["reward_id_v1"]
            reward_col = variant["reward_column_v1"]
            targets = v2_train_gate._compute_targets_for_variant(
                per_bar_train, reward_col
            )
            target_hold = targets["__target_hold_v1"].astype(float).to_numpy()
            target_exit_now = (
                targets["__target_exit_now_v1"].astype(float).to_numpy()
            )
            coef_hold = v2_train_gate._ridge_fit(
                X_full[train_mask_iql], target_hold
            )
            coef_exit_now = v2_train_gate._ridge_fit(
                X_full[train_mask_iql], target_exit_now
            )
            v2_iql_per_fold.append(
                {
                    "fold_id_v1": fold_id,
                    "reward_id_v1": v_id,
                    "head_v1": "HOLD",
                    "feature_names_v1": list(feature_names_iql),
                    "coefs_v1": coef_hold.tolist(),
                }
            )
            v2_iql_per_fold.append(
                {
                    "fold_id_v1": fold_id,
                    "reward_id_v1": v_id,
                    "head_v1": "EXIT_NOW",
                    "feature_names_v1": list(feature_names_iql),
                    "coefs_v1": coef_exit_now.tolist(),
                }
            )
    return skip_v2_per_fold, v2_iql_per_fold


def _aggregate_skip_v2_stability(
    skip_v2_per_fold: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """For each feature, collect per-fold coef and classify."""
    if not skip_v2_per_fold:
        return []
    feature_names = skip_v2_per_fold[0]["feature_names_v1"]
    rows: list[dict[str, Any]] = []
    for i, feat in enumerate(feature_names):
        per_fold = [r["coefs_v1"][i] for r in skip_v2_per_fold]
        rows.append(classify_feature_stability(feat, per_fold))
    return rows


def _aggregate_v2_iql_stability(
    v2_iql_per_fold: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per (reward, head, feature) collect per-fold coef and classify."""
    if not v2_iql_per_fold:
        return []
    feature_names = v2_iql_per_fold[0]["feature_names_v1"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in v2_iql_per_fold:
        grouped.setdefault((r["reward_id_v1"], r["head_v1"]), []).append(r)

    rows: list[dict[str, Any]] = []
    for (reward, head), per_fold_models in grouped.items():
        for i, feat in enumerate(feature_names):
            per_fold_coefs = [r["coefs_v1"][i] for r in per_fold_models]
            classified = classify_feature_stability(feat, per_fold_coefs)
            rows.append(
                {
                    **classified,
                    "reward_id_v1": reward,
                    "head_v1": head,
                }
            )
    return rows


def _summarize_classifications(
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        cls = r["classification_v1"]
        out[cls] = out.get(cls, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Materializer
# ---------------------------------------------------------------------------


def _build_input_manifest(
    inputs: dict[str, Any], artifact_root: Path
) -> dict[str, Any]:
    files = [
        {
            "name_v1": name,
            "path_v1": str(path),
            "sha256_v1": _file_hash(path),
        }
        for name, path in inputs["required_paths"].items()
    ]
    files.append(
        {
            "name_v1": "base34_m5_features",
            "path_v1": str(inputs["base34_path"]),
            "sha256_v1": _file_hash(inputs["base34_path"]),
        }
    )
    return {
        "layer_name": "AUDIT_FEATURE_STABILITY_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "recovery_root_v1": str(INPUT_RECOVERY_ROOT),
            "split_root_v1": str(INPUT_SPLIT_ROOT),
            "v2_contract_root_v1": str(INPUT_V2_CONTRACT_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": True,
        "iql_production_allowed_v1": False,
        "skip_classifier_promoted_to_runtime_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "entry_manager_modified_v1": False,
        "v1_state_contract_modified_v1": False,
        "v2_state_contract_modified_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
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

    inputs = wf_gate._load_inputs()
    _write_json(
        artifact_root / "input_manifest_v1.json",
        _build_input_manifest(inputs, artifact_root),
    )

    # Build candidate_uid time-ordering (re-use walk-forward logic).
    trades_all = skip_v1_gate._load_trade_outcomes_concat()
    trades_all["candidate_uid_v1"] = trades_all["candidate_uid"].astype(str)
    trades_all["open_ts_utc"] = pd.to_datetime(
        trades_all["open_ts_utc"], utc=True
    )
    split_df = pd.read_parquet(
        inputs["required_paths"]["split_locked_dataset"],
        columns=["candidate_uid_v1"],
    )
    accepted_uids = set(split_df["candidate_uid_v1"].astype(str).unique())
    trades_accepted = trades_all[
        trades_all["candidate_uid_v1"].isin(accepted_uids)
    ].sort_values(["open_ts_utc", "candidate_uid_v1"], kind="mergesort").reset_index(
        drop=True
    )
    candidate_uid_order = trades_accepted["candidate_uid_v1"].astype(str).tolist()

    # Per-fold training.
    skip_v2_per_fold, v2_iql_per_fold = _train_per_fold(inputs, candidate_uid_order)

    # Aggregate per-feature stability.
    skip_v2_rows = _aggregate_skip_v2_stability(skip_v2_per_fold)
    v2_iql_rows = _aggregate_v2_iql_stability(v2_iql_per_fold)

    _write_rows(
        artifact_root / "per_feature_stability_skip_v2_v1.csv", skip_v2_rows
    )
    _write_json(
        artifact_root / "per_feature_stability_skip_v2_v1.json",
        {"row_count_v1": len(skip_v2_rows), "rows_v1": skip_v2_rows},
    )
    _write_rows(
        artifact_root / "per_feature_stability_v2_iql_v1.csv", v2_iql_rows
    )
    _write_json(
        artifact_root / "per_feature_stability_v2_iql_v1.json",
        {"row_count_v1": len(v2_iql_rows), "rows_v1": v2_iql_rows},
    )

    # Summary classifications.
    skip_v2_classifications = _summarize_classifications(skip_v2_rows)
    v2_iql_classifications = _summarize_classifications(v2_iql_rows)

    # Headline numbers.
    skip_v2_stable_pct = (
        100.0
        * skip_v2_classifications.get("STABLE", 0)
        / max(1, len(skip_v2_rows))
    )
    skip_v2_flips_pct = (
        100.0
        * skip_v2_classifications.get("FLIPS_SIGN", 0)
        / max(1, len(skip_v2_rows))
    )
    v2_iql_stable_pct = (
        100.0
        * v2_iql_classifications.get("STABLE", 0)
        / max(1, len(v2_iql_rows))
    )
    v2_iql_flips_pct = (
        100.0
        * v2_iql_classifications.get("FLIPS_SIGN", 0)
        / max(1, len(v2_iql_rows))
    )

    headline = {
        "skip_v2_total_features_v1": len(skip_v2_rows),
        "skip_v2_classifications_v1": skip_v2_classifications,
        "skip_v2_stable_pct_v1": skip_v2_stable_pct,
        "skip_v2_flips_sign_pct_v1": skip_v2_flips_pct,
        "v2_iql_total_feature_head_reward_v1": len(v2_iql_rows),
        "v2_iql_classifications_v1": v2_iql_classifications,
        "v2_iql_stable_pct_v1": v2_iql_stable_pct,
        "v2_iql_flips_sign_pct_v1": v2_iql_flips_pct,
    }

    repro = {
        "layer_name": "AUDIT_FEATURE_STABILITY_REPRODUCIBILITY_AUDIT_V1",
        "fold_count_v1": len(wf_gate.FOLD_DEFINITIONS),
        "seed_v1": SEED_V1,
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status = "AUDIT_FEATURE_STABILITY_LOCKED_V1"
    next_action = "INVESTIGATE_TRAIL_STOP_DEEP_DIVE_V1"
    recommendation = (
        f"Skip-V2 features: {skip_v2_classifications.get('STABLE', 0)} stable, "
        f"{skip_v2_classifications.get('DIRECTIONAL', 0)} directional, "
        f"{skip_v2_classifications.get('FLIPS_SIGN', 0)} flip sign, "
        f"{skip_v2_classifications.get('DEAD', 0)} dead "
        f"({skip_v2_stable_pct:.0f}% stable, {skip_v2_flips_pct:.0f}% flip). "
        f"V2 IQL feature×head×reward triples: "
        f"{v2_iql_classifications.get('STABLE', 0)} stable, "
        f"{v2_iql_classifications.get('DIRECTIONAL', 0)} directional, "
        f"{v2_iql_classifications.get('FLIPS_SIGN', 0)} flip "
        f"({v2_iql_stable_pct:.0f}% stable, {v2_iql_flips_pct:.0f}% flip). "
        "Next: trail-stop deep-dive."
    )
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "AUDIT_FEATURE_STABILITY_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "fold_count_v1": len(wf_gate.FOLD_DEFINITIONS),
        "research_only_v1": True,
        "iql_training_run_v1": True,
        "iql_production_allowed_v1": False,
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
    _write_json(
        artifact_root / "status_v1.json",
        {
            "layer_name": "AUDIT_FEATURE_STABILITY_STATUS_V1",
            "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
            "final_status_v1": status,
            "next_action_v1": next_action,
            "training_executed_v1": True,
        },
    )
    _write_json(
        artifact_root / "audit_feature_stability_across_folds_go_no_go_v1.json",
        {
            "layer_name": "AUDIT_FEATURE_STABILITY_GO_NO_GO_V1",
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
            "training_allowed_v1": True,
            "downstream_block_v1": (
                "Research-only diagnostic; no model promotion. The output "
                "informs the design of future regime-conditioned or "
                "feature-pruned models."
            ),
        },
    )

    # Build a concise report focused on the most-flipping features (worst).
    skip_v2_sorted = sorted(skip_v2_rows, key=lambda r: r["std_v1"], reverse=True)
    top_unstable_skip = [r for r in skip_v2_sorted if r["classification_v1"] == "FLIPS_SIGN"][:10]
    top_unstable_iql = [r for r in v2_iql_rows if r["classification_v1"] == "FLIPS_SIGN"][:15]

    report_lines = [
        "# Audit Feature Stability Across Folds V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: research-only diagnostic.",
        "",
        "## Headline",
        f"- Skip-V2: {len(skip_v2_rows)} features, "
        f"{skip_v2_classifications.get('STABLE', 0)} stable / "
        f"{skip_v2_classifications.get('DIRECTIONAL', 0)} directional / "
        f"{skip_v2_classifications.get('FLIPS_SIGN', 0)} flips sign / "
        f"{skip_v2_classifications.get('DEAD', 0)} dead.",
        f"- V2 IQL: {len(v2_iql_rows)} (feature × head × reward) triples, "
        f"{v2_iql_classifications.get('STABLE', 0)} stable / "
        f"{v2_iql_classifications.get('DIRECTIONAL', 0)} directional / "
        f"{v2_iql_classifications.get('FLIPS_SIGN', 0)} flips sign / "
        f"{v2_iql_classifications.get('DEAD', 0)} dead.",
        f"- Skip-V2 stable pct: {skip_v2_stable_pct:.0f}% (good if > 70%, alarming if < 50%).",
        f"- V2 IQL stable pct: {v2_iql_stable_pct:.0f}%.",
        "",
        "## Top 10 unstable skip-V2 features (FLIPS_SIGN)",
        "",
        "| Feature | Per-fold coefs | mean | std |",
        "|---|---|---|---|",
    ]
    for r in top_unstable_skip:
        coefs_str = ", ".join(f"{c:+.3f}" for c in r["per_fold_coefs_v1"])
        report_lines.append(
            f"| `{r['feature_v1']}` | {coefs_str} | {r['mean_v1']:+.3f} | {r['std_v1']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## Top 15 unstable V2 IQL (feature × head × reward) triples (FLIPS_SIGN)",
            "",
            "| Feature | Reward | Head | Per-fold coefs | mean | std |",
            "|---|---|---|---|---|---|",
        ]
    )
    for r in top_unstable_iql:
        coefs_str = ", ".join(f"{c:+.3f}" for c in r["per_fold_coefs_v1"])
        report_lines.append(
            f"| `{r['feature_v1']}` | `{r['reward_id_v1']}` | "
            f"`{r['head_v1']}` | {coefs_str} | {r['mean_v1']:+.3f} | {r['std_v1']:.3f} |"
        )
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
                artifact_root / "audit_feature_stability_across_folds_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "per_feature_stability_skip_v2_csv": str(
                artifact_root / "per_feature_stability_skip_v2_v1.csv"
            ),
            "per_feature_stability_skip_v2_json": str(
                artifact_root / "per_feature_stability_skip_v2_v1.json"
            ),
            "per_feature_stability_v2_iql_csv": str(
                artifact_root / "per_feature_stability_v2_iql_v1.csv"
            ),
            "per_feature_stability_v2_iql_json": str(
                artifact_root / "per_feature_stability_v2_iql_v1.json"
            ),
            "reproducibility_audit": str(
                artifact_root / "reproducibility_audit_v1.json"
            ),
            "report": str(artifact_root / "report_v1.md"),
        },
        "read_only_references_v1": True,
        "trained_model_v1": True,
        "not_controller_v1": True,
        "not_live_gate_v1": True,
    }
    _write_json(artifact_root / "manifest_v1.json", artifact_manifest)

    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize AUDIT_FEATURE_STABILITY_ACROSS_FOLDS_V1.")
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
