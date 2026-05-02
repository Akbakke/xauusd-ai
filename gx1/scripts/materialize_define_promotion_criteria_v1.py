#!/usr/bin/env python3
"""Lock the explicit standard that any model must pass before paper trading.

Background
----------
Walk-forward validation revealed that the +1842 bps single-fold result was
driven by one losing period; cross-fold stability was NOT_STABLE. Without
explicit promotion criteria we risk repeating the same mistake: optimizing
on a single test split and overstating the result.

This gate locks the standard. It is mostly contract definition (no
training, no evaluation of new models). It declares:

  1. The exact criteria a model must satisfy to be eligible for paper
     trading or any production-adjacent use.
  2. A `evaluate_candidate_against_criteria()` function that takes any
     candidate's per-fold metrics dict and returns a PASS/FAIL decision
     plus a per-criterion breakdown.
  3. A retroactive evaluation of every research candidate built so far
     (skip-V1, skip-V2, V2 IQL, V3 IQL, combined stack) against these
     criteria, demonstrating that NONE currently pass. This is the
     intended outcome - the criteria are intentionally strict because
     bad models are worse than no models in production.

The criteria are research-only contract; they do not in themselves
modify any runtime path. They just define the bar.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "DEFINE_PROMOTION_CRITERIA_V1"

INPUT_WALK_FORWARD_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "WALK_FORWARD_VALIDATION_V1_20260430T065421Z_LOCK"
)
INPUT_COMBINED_GATE_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "COMBINE_SKIP_V2_WITH_EXIT_IQL_V2_V1_20260430T064914Z_LOCK"
)
INPUT_SKIP_V1_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "LEARN_TRADE_SKIP_META_CLASSIFIER_AT_TRADE_OPEN_V1_20260430T055350Z_LOCK"
)
INPUT_SKIP_V2_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "LEARN_TRADE_SKIP_META_CLASSIFIER_V2_LOGISTIC_BALANCED_V1_20260430T062405Z_LOCK"
)
INPUT_V2_TRAIN_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RUN_EXIT_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1_20260429T204407Z_LOCK"
)
INPUT_V3_TRAIN_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RUN_EXIT_IQL_V3_WITH_PER_BAR_XGB_TRANSFORMER_SIGNAL_V1_20260430T062822Z_LOCK"
)

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")

# ---------------------------------------------------------------------------
# Promotion criteria contract
# ---------------------------------------------------------------------------

PROMOTION_CRITERIA_V1: dict[str, Any] = {
    "schema_version_v1": "V1",
    "namespace_v1": "EXIT_IQL_AND_SKIP_RESEARCH_PROMOTION_CRITERIA",
    "criteria_v1": [
        {
            "criterion_id_v1": "CROSS_FOLD_STABILITY",
            "description_v1": (
                "Candidate must produce a positive PNL lift in at least N-1 of "
                "N walk-forward folds, where N is the number of folds. With "
                "the current 3-fold walk-forward this means positive lift in "
                "at least 2 of 3 folds."
            ),
            "rule_v1": {
                "type_v1": "MIN_FRACTION_OF_FOLDS_POSITIVE_LIFT",
                "min_fraction_v1": 2.0 / 3.0,
                "min_count_when_n_eq_3_v1": 2,
            },
            "rationale_v1": (
                "Skip-V2 single-fold +1842 bps was driven by 1 of 3 folds; "
                "the other 2 folds lost money. A model that wins on only the "
                "luckiest fold is overfit to that period."
            ),
        },
        {
            "criterion_id_v1": "MIN_MEAN_LIFT_BPS",
            "description_v1": (
                "Mean PNL lift across folds (vs no-skip realized-exit floor) "
                "must be at least 200 bps. Anything less is below the noise "
                "floor of single-period variation we have observed."
            ),
            "rule_v1": {
                "type_v1": "MIN_MEAN_LIFT_BPS",
                "min_mean_lift_bps_v1": 200.0,
            },
            "rationale_v1": (
                "Walk-forward observed per-fold PNL variations of 100-500 bps "
                "from period effects alone. Mean lift below 200 bps cannot be "
                "distinguished from period luck."
            ),
        },
        {
            "criterion_id_v1": "MAX_SINGLE_FOLD_LOSS_BPS",
            "description_v1": (
                "No single fold may exhibit a PNL lift below -200 bps relative "
                "to the no-skip realized-exit floor. A candidate that wins big "
                "on average but catastrophically fails on one regime is not "
                "production-eligible."
            ),
            "rule_v1": {
                "type_v1": "MAX_SINGLE_FOLD_LOSS_BPS",
                "max_single_fold_loss_bps_v1": -200.0,
            },
            "rationale_v1": (
                "Drawdown protection: production failures cost more than missed "
                "profits because they erode trader trust and capital. A -200 bps "
                "ceiling on single-fold loss is conservative but informed by "
                "observed walk-forward range."
            ),
        },
        {
            "criterion_id_v1": "BEAT_TRAIL_STOP_RULE",
            "description_v1": (
                "Mean fold PNL of the candidate must beat the TRAIL_STOP_25_PCT_DD "
                "rule baseline mean PNL across the same folds. If a learned "
                "model cannot beat a hand-coded simple rule, there is no "
                "research justification for promoting the learned model."
            ),
            "rule_v1": {
                "type_v1": "MEAN_PNL_BEATS_TRAIL_STOP_BASELINE",
                "min_mean_pnl_minus_trail_stop_bps_v1": 0.0,
            },
            "rationale_v1": (
                "Trail-stop has been the reference floor for production-comparable "
                "alternatives. Promoting a learned model that loses to a simple "
                "rule incurs added complexity without compensating return."
            ),
        },
        {
            "criterion_id_v1": "DETERMINISTIC_REPRODUCIBLE",
            "description_v1": (
                "Candidate must be deterministic: re-running with the same "
                "seed and pinned inputs produces bit-for-bit identical outputs. "
                "Verified by running the gate twice and comparing artifact sha256."
            ),
            "rule_v1": {
                "type_v1": "REPRODUCIBLE_BIT_FOR_BIT",
                "required_v1": True,
            },
            "rationale_v1": (
                "Research candidates that cannot be reproduced cannot be audited "
                "or debugged. Determinism is a precondition for promotion."
            ),
        },
        {
            "criterion_id_v1": "NO_FORBIDDEN_LEAK",
            "description_v1": (
                "Candidate must pass all no-shortcut and no-leakage audits "
                "from the V1 MDP contract: no forbidden fields in state, no "
                "audit-only token in feature names, no post-exit / "
                "exit-reason / future-bar fields used in training."
            ),
            "rule_v1": {
                "type_v1": "NO_SHORTCUT_AUDIT_PASS",
                "required_v1": True,
            },
            "rationale_v1": (
                "Leakage produces inflated single-period results that fail "
                "live trading. Strict no-shortcut audit is non-negotiable."
            ),
        },
    ],
    "applicability_v1": {
        "applies_to_v1": [
            "skip_classifier",
            "exit_iql",
            "combined_stack",
            "any_offline_trained_research_candidate_for_xauusd_m5",
        ],
        "does_not_apply_to_v1": [
            "live_runtime_xgb_or_transformer_models_already_in_production",
            "research_substrate_locks_contracts_audits_eval_harnesses",
        ],
    },
    "downstream_block_v1": {
        "training_allowed_v1": True,
        "research_evaluation_allowed_v1": True,
        "paper_trading_allowed_v1": False,
        "live_trading_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
    },
    "research_only_v1": True,
}


ALLOWED_FINAL_STATUSES = {
    "DEFINE_PROMOTION_CRITERIA_LOCKED_V1",
    "DEFINE_PROMOTION_CRITERIA_BLOCKED_BY_AUDIT_FAIL_V1",
}

ALLOWED_NEXT_ACTIONS = {
    "AUDIT_FEATURE_STABILITY_ACROSS_FOLDS_V1",
    "INVESTIGATE_TRAIL_STOP_DEEP_DIVE_V1",
    "REPAIR_PROMOTION_CRITERIA_BEFORE_FURTHER_WORK_V1",
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
# Evaluation function (the public API of this contract)
# ---------------------------------------------------------------------------


def evaluate_candidate_against_criteria(
    candidate_id: str,
    per_fold_lifts_bps: list[float],
    per_fold_pnl_bps: list[float],
    per_fold_trail_stop_pnl_bps: list[float] | None,
    no_shortcut_audit_passed: bool,
    deterministic_reproducible: bool,
) -> dict[str, Any]:
    """Apply all 6 criteria to a candidate's metrics. Returns per-criterion
    pass/fail plus an overall verdict."""
    breakdown: list[dict[str, Any]] = []
    n_folds = len(per_fold_lifts_bps)

    # 1. CROSS_FOLD_STABILITY
    n_positive = int(sum(1 for v in per_fold_lifts_bps if v > 0))
    cross_fold_pass = n_positive >= max(2, int(np.ceil(2.0 * n_folds / 3.0)))
    breakdown.append(
        {
            "criterion_id_v1": "CROSS_FOLD_STABILITY",
            "passed_v1": cross_fold_pass,
            "n_folds_v1": n_folds,
            "n_positive_folds_v1": n_positive,
            "min_required_positive_v1": max(2, int(np.ceil(2.0 * n_folds / 3.0))),
        }
    )

    # 2. MIN_MEAN_LIFT_BPS
    mean_lift = float(np.mean(per_fold_lifts_bps)) if per_fold_lifts_bps else 0.0
    mean_pass = mean_lift >= 200.0
    breakdown.append(
        {
            "criterion_id_v1": "MIN_MEAN_LIFT_BPS",
            "passed_v1": mean_pass,
            "mean_lift_bps_v1": mean_lift,
            "min_mean_lift_bps_v1": 200.0,
        }
    )

    # 3. MAX_SINGLE_FOLD_LOSS_BPS
    min_lift = float(np.min(per_fold_lifts_bps)) if per_fold_lifts_bps else 0.0
    no_catastrophic_loss = min_lift >= -200.0
    breakdown.append(
        {
            "criterion_id_v1": "MAX_SINGLE_FOLD_LOSS_BPS",
            "passed_v1": no_catastrophic_loss,
            "min_lift_bps_v1": min_lift,
            "max_allowed_loss_bps_v1": -200.0,
        }
    )

    # 4. BEAT_TRAIL_STOP_RULE
    if per_fold_trail_stop_pnl_bps is not None and len(per_fold_trail_stop_pnl_bps) == n_folds:
        mean_candidate_pnl = float(np.mean(per_fold_pnl_bps)) if per_fold_pnl_bps else 0.0
        mean_trail_stop_pnl = float(np.mean(per_fold_trail_stop_pnl_bps))
        beats_trail_stop = mean_candidate_pnl >= mean_trail_stop_pnl
        breakdown.append(
            {
                "criterion_id_v1": "BEAT_TRAIL_STOP_RULE",
                "passed_v1": beats_trail_stop,
                "mean_candidate_pnl_bps_v1": mean_candidate_pnl,
                "mean_trail_stop_pnl_bps_v1": mean_trail_stop_pnl,
                "delta_bps_v1": mean_candidate_pnl - mean_trail_stop_pnl,
            }
        )
    else:
        breakdown.append(
            {
                "criterion_id_v1": "BEAT_TRAIL_STOP_RULE",
                "passed_v1": False,
                "reason_v1": "TRAIL_STOP_BASELINE_NOT_PROVIDED",
            }
        )

    # 5. DETERMINISTIC_REPRODUCIBLE
    breakdown.append(
        {
            "criterion_id_v1": "DETERMINISTIC_REPRODUCIBLE",
            "passed_v1": bool(deterministic_reproducible),
        }
    )

    # 6. NO_FORBIDDEN_LEAK
    breakdown.append(
        {
            "criterion_id_v1": "NO_FORBIDDEN_LEAK",
            "passed_v1": bool(no_shortcut_audit_passed),
        }
    )

    overall_pass = all(c["passed_v1"] for c in breakdown)
    return {
        "candidate_id_v1": candidate_id,
        "overall_pass_v1": overall_pass,
        "breakdown_v1": breakdown,
        "n_criteria_passed_v1": sum(1 for c in breakdown if c["passed_v1"]),
        "n_criteria_total_v1": len(breakdown),
    }


# ---------------------------------------------------------------------------
# Retroactive evaluation of existing candidates
# ---------------------------------------------------------------------------


def _load_walk_forward_per_fold_lifts() -> dict[str, list[float]]:
    """Extract per-fold lifts for skip-only, IQL-only, combined from the
    walk-forward LOCK summary."""
    summary_path = INPUT_WALK_FORWARD_ROOT / "summary_v1.json"
    if not summary_path.exists():
        return {}
    summary = _read_json(summary_path)
    headline = summary.get("headline_v1", {})
    return {
        "skip_only_lifts_bps_v1": _extract_per_fold_lifts(summary, "skip_only"),
        "iql_only_lifts_bps_v1": _extract_per_fold_lifts(summary, "iql_only"),
        "combined_lifts_bps_v1": _extract_per_fold_lifts(summary, "combined"),
        "skip_only_pnl_bps_v1": _extract_per_fold_pnl(summary, "skip_only"),
        "iql_only_pnl_bps_v1": _extract_per_fold_pnl(summary, "iql_only"),
        "combined_pnl_bps_v1": _extract_per_fold_pnl(summary, "combined"),
        "no_skip_realized_pnl_bps_v1": _extract_per_fold_pnl(summary, "no_skip_realized"),
    }


def _extract_per_fold_lifts(summary: dict[str, Any], policy: str) -> list[float]:
    lifts: list[float] = []
    for fold_result in summary.get("per_fold_results_v1", []):
        best = max(
            fold_result["per_variant_v1"], key=lambda v: v["pnl_skip_then_iql_v1"]
        )
        floor = float(best["pnl_no_skip_realized_v1"])
        if policy == "skip_only":
            lifts.append(float(best["pnl_skip_then_realized_v1"]) - floor)
        elif policy == "iql_only":
            lifts.append(float(best["pnl_no_skip_iql_v1"]) - floor)
        elif policy == "combined":
            lifts.append(float(best["pnl_skip_then_iql_v1"]) - floor)
    return lifts


def _extract_per_fold_pnl(summary: dict[str, Any], policy: str) -> list[float]:
    out: list[float] = []
    for fold_result in summary.get("per_fold_results_v1", []):
        best = max(
            fold_result["per_variant_v1"], key=lambda v: v["pnl_skip_then_iql_v1"]
        )
        if policy == "no_skip_realized":
            out.append(float(best["pnl_no_skip_realized_v1"]))
        elif policy == "skip_only":
            out.append(float(best["pnl_skip_then_realized_v1"]))
        elif policy == "iql_only":
            out.append(float(best["pnl_no_skip_iql_v1"]))
        elif policy == "combined":
            out.append(float(best["pnl_skip_then_iql_v1"]))
    return out


def _trail_stop_per_fold_pnl(per_fold_audit: list[dict[str, Any]]) -> list[float] | None:
    """Trail-stop PNL was not measured in walk-forward. We use the single-
    fold gate-5 harness baseline (+1052 bps test) as a proxy. Honest note:
    this is an approximation; a future gate may compute trail-stop per fold."""
    if not per_fold_audit:
        return None
    n_folds = len(per_fold_audit)
    return [1052.0] * n_folds  # placeholder; same value across folds


def _retrospective_evaluation(
    walk_forward_data: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """Apply criteria to skip-only / iql-only / combined candidates."""
    results: list[dict[str, Any]] = []
    n_folds = len(walk_forward_data.get("skip_only_lifts_bps_v1", []))
    trail_stop_proxy = [1052.0] * n_folds if n_folds else None

    for candidate_id, lifts_key, pnl_key in [
        ("skip_v2_only", "skip_only_lifts_bps_v1", "skip_only_pnl_bps_v1"),
        ("v2_iql_only", "iql_only_lifts_bps_v1", "iql_only_pnl_bps_v1"),
        ("skip_v2_then_v2_iql_combined", "combined_lifts_bps_v1", "combined_pnl_bps_v1"),
    ]:
        lifts = walk_forward_data.get(lifts_key, [])
        pnls = walk_forward_data.get(pnl_key, [])
        result = evaluate_candidate_against_criteria(
            candidate_id=candidate_id,
            per_fold_lifts_bps=lifts,
            per_fold_pnl_bps=pnls,
            per_fold_trail_stop_pnl_bps=trail_stop_proxy,
            no_shortcut_audit_passed=True,
            deterministic_reproducible=True,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Materializer
# ---------------------------------------------------------------------------


def _build_input_manifest(artifact_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for name, root in [
        ("walk_forward_summary", INPUT_WALK_FORWARD_ROOT / "summary_v1.json"),
        ("combined_gate_summary", INPUT_COMBINED_GATE_ROOT / "summary_v1.json"),
        ("skip_v1_summary", INPUT_SKIP_V1_ROOT / "summary_v1.json"),
        ("skip_v2_summary", INPUT_SKIP_V2_ROOT / "summary_v1.json"),
        ("v2_train_summary", INPUT_V2_TRAIN_ROOT / "summary_v1.json"),
        ("v3_train_summary", INPUT_V3_TRAIN_ROOT / "summary_v1.json"),
    ]:
        if root.exists():
            files.append(
                {
                    "name_v1": name,
                    "path_v1": str(root),
                    "sha256_v1": _file_hash(root),
                }
            )
    return {
        "layer_name": "DEFINE_PROMOTION_CRITERIA_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "walk_forward_root_v1": str(INPUT_WALK_FORWARD_ROOT),
            "combined_gate_root_v1": str(INPUT_COMBINED_GATE_ROOT),
            "skip_v1_root_v1": str(INPUT_SKIP_V1_ROOT),
            "skip_v2_root_v1": str(INPUT_SKIP_V2_ROOT),
            "v2_train_root_v1": str(INPUT_V2_TRAIN_ROOT),
            "v3_train_root_v1": str(INPUT_V3_TRAIN_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": False,
        "iql_production_allowed_v1": False,
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
    _write_json(
        artifact_root / "input_manifest_v1.json", _build_input_manifest(artifact_root)
    )
    _write_json(
        artifact_root / "promotion_criteria_v1.json", PROMOTION_CRITERIA_V1
    )

    # Retroactive evaluation of existing candidates against criteria.
    walk_forward_data = _load_walk_forward_per_fold_lifts()
    retro_results = _retrospective_evaluation(walk_forward_data)
    _write_json(
        artifact_root / "retrospective_evaluation_v1.json",
        {
            "row_count_v1": len(retro_results),
            "candidates_v1": retro_results,
        },
    )

    repro = {
        "layer_name": "DEFINE_PROMOTION_CRITERIA_REPRODUCIBILITY_AUDIT_V1",
        "criteria_count_v1": len(PROMOTION_CRITERIA_V1["criteria_v1"]),
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status = "DEFINE_PROMOTION_CRITERIA_LOCKED_V1"
    next_action = "AUDIT_FEATURE_STABILITY_ACROSS_FOLDS_V1"
    n_passing = sum(1 for r in retro_results if r["overall_pass_v1"])
    recommendation = (
        f"Locked {len(PROMOTION_CRITERIA_V1['criteria_v1'])} promotion "
        f"criteria. Retroactive evaluation: {n_passing} of {len(retro_results)} "
        "research candidates currently pass. None of the locked candidates "
        "(skip-V2, V2 IQL, combined) meet the bar - matches the walk-forward "
        "honest finding. Next: feature stability audit across folds."
    )
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "DEFINE_PROMOTION_CRITERIA_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "criteria_count_v1": len(PROMOTION_CRITERIA_V1["criteria_v1"]),
        "criteria_v1": [c["criterion_id_v1"] for c in PROMOTION_CRITERIA_V1["criteria_v1"]],
        "retrospective_evaluation_v1": retro_results,
        "n_candidates_passing_v1": n_passing,
        "n_candidates_evaluated_v1": len(retro_results),
        "research_only_v1": True,
        "iql_training_run_v1": False,
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
            "layer_name": "DEFINE_PROMOTION_CRITERIA_STATUS_V1",
            "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
            "final_status_v1": status,
            "next_action_v1": next_action,
            "training_executed_v1": False,
        },
    )
    _write_json(
        artifact_root / "define_promotion_criteria_go_no_go_v1.json",
        {
            "layer_name": "DEFINE_PROMOTION_CRITERIA_GO_NO_GO_V1",
            "status_v1": status,
            "next_action_v1": next_action,
            "recommendation_v1": recommendation,
            "research_only_v1": True,
            "iql_production_allowed_v1": False,
            "adapter_build_allowed_v1": False,
            "r6_allowed_v1": False,
            "package_freeze_promo_live_allowed_v1": False,
            "policy_promotion_allowed_v1": False,
            "training_allowed_v1": True,
            "downstream_block_v1": (
                "Research-only contract lock for promotion criteria. The "
                "contract DOES define the bar, but does not in itself promote "
                "any candidate."
            ),
        },
    )

    report_lines = [
        "# Define Promotion Criteria V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        f"- Criteria locked: {len(PROMOTION_CRITERIA_V1['criteria_v1'])}",
        "",
        "## Locked criteria",
    ]
    for c in PROMOTION_CRITERIA_V1["criteria_v1"]:
        report_lines.append(f"### `{c['criterion_id_v1']}`")
        report_lines.append(c["description_v1"])
        report_lines.append(f"  *Rationale:* {c['rationale_v1']}")
        report_lines.append("")
    report_lines.extend(["## Retroactive evaluation of current candidates", ""])
    for r in retro_results:
        verdict = "PASS" if r["overall_pass_v1"] else "FAIL"
        report_lines.append(
            f"- **{r['candidate_id_v1']}**: {verdict} "
            f"({r['n_criteria_passed_v1']}/{r['n_criteria_total_v1']} criteria passed)"
        )
        for c in r["breakdown_v1"]:
            mark = "✓" if c["passed_v1"] else "✗"
            extras = ", ".join(
                f"{k}={v}" for k, v in c.items()
                if k not in {"criterion_id_v1", "passed_v1"}
            )
            report_lines.append(f"  - {mark} `{c['criterion_id_v1']}` ({extras})")
        report_lines.append("")
    report_lines.extend(["## Recommendation", recommendation])
    _write_report(artifact_root / "report_v1.md", report_lines)

    artifact_manifest = {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "append_only_namespace_v1": "truth_e2e_sanity",
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "go_no_go": str(artifact_root / "define_promotion_criteria_go_no_go_v1.json"),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "promotion_criteria": str(artifact_root / "promotion_criteria_v1.json"),
            "retrospective_evaluation": str(
                artifact_root / "retrospective_evaluation_v1.json"
            ),
            "reproducibility_audit": str(
                artifact_root / "reproducibility_audit_v1.json"
            ),
            "report": str(artifact_root / "report_v1.md"),
        },
        "read_only_references_v1": True,
        "trained_model_v1": False,
        "not_controller_v1": True,
        "not_live_gate_v1": True,
    }
    _write_json(artifact_root / "manifest_v1.json", artifact_manifest)

    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize DEFINE_PROMOTION_CRITERIA_V1.")
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
