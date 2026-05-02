#!/usr/bin/env python3
"""GPU-trained True IQL gate (NN-based V/Q via PyTorch on CUDA).

Mirrors materialize_build_true_implicit_q_learning_v1's contract surface
(folds, reward variants, hyperparameter grid, promotion criteria, artifact
shape) but replaces the closed-form ridge V/Q with small MLPs trained on the
GPU. Same expectile loss for V, same Bellman-MSE for Q, same sigmoid-of-
advantage policy.

Output: <DEFAULT_REPORTS_ROOT>/BUILD_TRUE_IMPLICIT_Q_LEARNING_GPU_V1_<TS>_LOCK/
with the same artifact filenames as V1 so existing audits / readers continue
to work. The summary records device + parameter counts so the difference is
auditable.

Research-only. Not a controller, not a live gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import (
    materialize_build_true_implicit_q_learning_v1 as v1_gate,
)
from gx1.scripts import true_iql_gpu_core_v1 as gpu_core
from gx1.scripts import (
    materialize_define_promotion_criteria_v1 as criteria_gate,
)
from gx1.scripts import materialize_exit_off_policy_eval_harness_v1 as eval_gate
from gx1.scripts import (
    materialize_run_exit_iql_with_v2_state_and_reward_variants_v1 as v2_train_gate,
)
from gx1.scripts import (
    materialize_walk_forward_validation_v1 as wf_gate,
)
from gx1.scripts import (
    materialize_learn_trade_skip_meta_classifier_at_trade_open_v1 as skip_v1_gate,
)
from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)


ACTION = "BUILD_TRUE_IMPLICIT_Q_LEARNING_GPU_V1"
DEFAULT_REPORTS_ROOT = v1_gate.DEFAULT_REPORTS_ROOT

# Reuse V1's hyperparameter grid so results are directly comparable.
TAU_GRID = v1_gate.TAU_GRID
BETA_GRID = v1_gate.BETA_GRID
GAMMA_LOCKED = v1_gate.GAMMA_LOCKED
K_VQ_ITERATIONS = v1_gate.K_VQ_ITERATIONS
ADVANTAGE_CLIP = v1_gate.ADVANTAGE_CLIP
SEED_V1 = v1_gate.SEED_V1

# GPU-specific knobs.
INNER_EPOCHS = 50
HIDDEN_DIM = 64
N_HIDDEN = 2
LR = 1e-3
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 256

ALLOWED_FINAL_STATUSES = {
    "TRUE_IQL_GPU_PASS_MEETS_PROMOTION_CRITERIA",
    "TRUE_IQL_GPU_PASS_BEATS_RIDGE_BUT_FAILS_OTHER_CRITERIA",
    "TRUE_IQL_GPU_PARTIAL_TIES_RIDGE",
    "TRUE_IQL_GPU_PARTIAL_DEGRADES_VS_RIDGE",
    "TRUE_IQL_GPU_BLOCKED_BY_INPUT_LOCK_MISSING",
}
ALLOWED_NEXT_ACTIONS = {
    "BUILD_CONSERVATIVE_Q_LEARNING_GPU_V1",
    "BUILD_DISTRIBUTIONAL_Q_LEARNING_V1",
    "REPAIR_TRUE_IQL_GPU_BEFORE_FURTHER_WORK_V1",
}

_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows
_write_report = contract_gate._write_report
_read_json = contract_gate._read_json
_python_manifest = contract_gate._python_manifest


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def _evaluate_fold_gpu(
    per_bar_full: pd.DataFrame, X_full: np.ndarray, fold_id: str
) -> dict[str, Any]:
    """Same per-fold sweep as V1 but training V/Q on GPU."""
    train_mask = (per_bar_full["primary_split_v1"] == "train").to_numpy()
    val_mask = (per_bar_full["primary_split_v1"] == "val").to_numpy()
    test_mask = (per_bar_full["primary_split_v1"] == "test").to_numpy()
    per_bar_train = per_bar_full[train_mask].reset_index(drop=True)
    X_train = X_full[train_mask].astype(np.float32)
    per_bar_val = per_bar_full[val_mask].reset_index(drop=True)
    X_val = X_full[val_mask].astype(np.float32)
    per_bar_test = per_bar_full[test_mask].reset_index(drop=True)
    X_test = X_full[test_mask].astype(np.float32)

    all_evals: list[dict[str, Any]] = []
    best_val_pnl = -np.inf
    best_combo: tuple[str, float, float] | None = None
    for variant in v2_train_gate.REWARD_VARIANTS_V2:
        v_id = variant["reward_id_v1"]
        reward_col = variant["reward_column_v1"]
        if reward_col not in per_bar_train.columns:
            continue
        trans = v1_gate._build_transition_tuples(per_bar_train, X_train, reward_col)
        a_t = trans["a_v1"].astype(np.int64)
        r_t = trans["r_v1"].astype(np.float32)
        next_idx_t = trans["next_idx_v1"].astype(np.int64)
        done_t = trans["done_v1"].astype(bool)

        for tau in TAU_GRID:
            try:
                model = gpu_core.train_true_iql_gpu(
                    X_train, a_t, r_t, next_idx_t, done_t,
                    tau=tau,
                    gamma=GAMMA_LOCKED,
                    k_iterations=K_VQ_ITERATIONS,
                    inner_epochs=INNER_EPOCHS,
                    lr=LR,
                    weight_decay=WEIGHT_DECAY,
                    hidden_dim=HIDDEN_DIM,
                    n_hidden=N_HIDDEN,
                    batch_size=BATCH_SIZE,
                    seed=SEED_V1,
                    prefer_cuda=True,
                )
            except Exception as exc:  # noqa: BLE001
                all_evals.append({
                    "fold_id_v1": fold_id,
                    "reward_id_v1": v_id,
                    "tau_v1": tau,
                    "status_v1": "TRAIN_ERROR",
                    "error_v1": str(exc)[:200],
                })
                continue
            for beta in BETA_GRID:
                p_val = gpu_core.true_iql_policy_exit_prob_gpu(
                    X_val, model, beta=beta, clip=ADVANTAGE_CLIP,
                )
                val_metric = None
                if not per_bar_val.empty:
                    val_idx = v1_gate._exit_index_from_iql_policy(per_bar_val, p_val)
                    val_metric = eval_gate.evaluate_policy(
                        per_bar_val, val_idx,
                        policy_id=f"TRUE_IQL_GPU_{v_id}_TAU{tau}_BETA{beta}",
                        split="val",
                    )
                p_test = gpu_core.true_iql_policy_exit_prob_gpu(
                    X_test, model, beta=beta, clip=ADVANTAGE_CLIP,
                )
                test_metric = None
                if not per_bar_test.empty:
                    test_idx = v1_gate._exit_index_from_iql_policy(per_bar_test, p_test)
                    test_metric = eval_gate.evaluate_policy(
                        per_bar_test, test_idx,
                        policy_id=f"TRUE_IQL_GPU_{v_id}_TAU{tau}_BETA{beta}",
                        split="test",
                    )
                all_evals.append({
                    "fold_id_v1": fold_id,
                    "reward_id_v1": v_id,
                    "tau_v1": float(tau),
                    "beta_v1": float(beta),
                    "val_metric_v1": val_metric,
                    "test_metric_v1": test_metric,
                })
                if val_metric and val_metric["total_realized_pnl_bps_v1"] > best_val_pnl:
                    best_val_pnl = val_metric["total_realized_pnl_bps_v1"]
                    best_combo = (v_id, float(tau), float(beta))
    if best_combo is None:
        return {
            "fold_id_v1": fold_id,
            "best_variant_v1": None,
            "best_tau_v1": None,
            "best_beta_v1": None,
            "test_at_locked_v1": None,
            "all_evaluations_v1": all_evals,
        }
    bv, btau, bbeta = best_combo
    locked_test = next(
        (e["test_metric_v1"] for e in all_evals
         if e.get("reward_id_v1") == bv
         and e.get("tau_v1") == btau
         and e.get("beta_v1") == bbeta),
        None,
    )
    return {
        "fold_id_v1": fold_id,
        "best_variant_v1": bv,
        "best_tau_v1": btau,
        "best_beta_v1": bbeta,
        "test_at_locked_v1": locked_test,
        "all_evaluations_v1": all_evals,
    }


def _run_per_fold_gpu(
    inputs: dict[str, Any], candidate_uid_order: list[str]
) -> list[dict[str, Any]]:
    fold_results: list[dict[str, Any]] = []
    for fold in wf_gate.FOLD_DEFINITIONS:
        fold_id = fold["fold_id_v1"]
        uid_to_split = wf_gate._assign_fold_split(candidate_uid_order, fold)
        per_bar, X_full, _ = wf_gate._build_per_bar_for_fold(inputs, uid_to_split)
        fold_results.append(_evaluate_fold_gpu(per_bar, X_full, fold_id))
    return fold_results


def _go_no_go_gpu(
    per_fold_results: list[dict[str, Any]],
    promotion: dict[str, Any],
) -> tuple[str, str, str, dict[str, Any]]:
    """Compare GPU IQL to ridge V1 IQL (not AWR). Otherwise mirrors V1."""
    n_pass = int(promotion["n_criteria_passed_v1"])
    n_total = int(promotion["n_criteria_total_v1"])
    overall = bool(promotion["overall_pass_v1"])
    ridge_per_fold_pnl: list[float] = []
    ridge_summary_path = (
        DEFAULT_REPORTS_ROOT / "BUILD_TRUE_IMPLICIT_Q_LEARNING_V1_LATEST_LOCK"
        / "summary_v1.json"
    )
    if not ridge_summary_path.exists():
        for d in sorted(DEFAULT_REPORTS_ROOT.glob("BUILD_TRUE_IMPLICIT_Q_LEARNING_V1_*_LOCK")):
            cand = d / "summary_v1.json"
            if cand.exists():
                ridge_summary_path = cand
    if ridge_summary_path.exists():
        ridge_summary = _read_json(ridge_summary_path)
        for r in ridge_summary.get("per_fold_summary_v1", []) or []:
            if r.get("test_at_locked_v1"):
                ridge_per_fold_pnl.append(
                    float(r["test_at_locked_v1"]["total_realized_pnl_bps_v1"])
                )
            else:
                ridge_per_fold_pnl.append(0.0)
    gpu_total = float(np.sum(promotion["per_fold_pnl_v1"]))
    ridge_total = float(np.sum(ridge_per_fold_pnl)) if ridge_per_fold_pnl else 0.0
    gpu_minus_ridge = gpu_total - ridge_total
    headline = {
        "n_folds_v1": len(per_fold_results),
        "true_iql_gpu_per_fold_pnl_v1": promotion["per_fold_pnl_v1"],
        "true_iql_ridge_per_fold_pnl_v1": ridge_per_fold_pnl,
        "realized_per_fold_pnl_v1": promotion["per_fold_realized_v1"],
        "lifts_vs_realized_v1": promotion["per_fold_lifts_vs_realized_v1"],
        "gpu_total_v1": gpu_total,
        "ridge_total_v1": ridge_total,
        "gpu_minus_ridge_total_v1": gpu_minus_ridge,
        "promotion_pass_v1": overall,
        "promotion_n_passed_v1": n_pass,
        "promotion_n_total_v1": n_total,
        "best_per_fold_v1": [
            {
                "fold_id_v1": r["fold_id_v1"],
                "variant_v1": r["best_variant_v1"],
                "tau_v1": r["best_tau_v1"],
                "beta_v1": r["best_beta_v1"],
            }
            for r in per_fold_results
        ],
    }
    if overall:
        return (
            "TRUE_IQL_GPU_PASS_MEETS_PROMOTION_CRITERIA",
            "BUILD_CONSERVATIVE_Q_LEARNING_GPU_V1",
            (
                f"GPU True IQL passes ALL {n_total} promotion criteria. Total "
                f"{gpu_total:+.0f} bps. Beats ridge V1 by {gpu_minus_ridge:+.0f}. "
                "NN function approximation validated on extended dataset. "
                "Next: add CQL pessimism."
            ),
            headline,
        )
    if gpu_minus_ridge > 200.0:
        return (
            "TRUE_IQL_GPU_PASS_BEATS_RIDGE_BUT_FAILS_OTHER_CRITERIA",
            "BUILD_CONSERVATIVE_Q_LEARNING_GPU_V1",
            (
                f"GPU NN beats ridge V1 by {gpu_minus_ridge:+.0f} bps total but "
                f"fails {n_total - n_pass}/{n_total} criteria. NN expressiveness "
                "validated; promotion gates blocked elsewhere."
            ),
            headline,
        )
    if abs(gpu_minus_ridge) <= 200.0:
        return (
            "TRUE_IQL_GPU_PARTIAL_TIES_RIDGE",
            "BUILD_CONSERVATIVE_Q_LEARNING_GPU_V1",
            (
                f"GPU NN ~= ridge ({gpu_total:+.0f} vs {ridge_total:+.0f}; "
                f"delta {gpu_minus_ridge:+.0f}). NN doesn't help on this "
                "data scale; reconsider after CQL or after dataset is larger."
            ),
            headline,
        )
    return (
        "TRUE_IQL_GPU_PARTIAL_DEGRADES_VS_RIDGE",
        "REPAIR_TRUE_IQL_GPU_BEFORE_FURTHER_WORK_V1",
        (
            f"GPU NN underperforms ridge ({gpu_total:+.0f} vs {ridge_total:+.0f}; "
            f"delta {gpu_minus_ridge:+.0f}). Investigate overfitting / "
            "regularization."
        ),
        headline,
    )


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    inputs = wf_gate._load_inputs()
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    v1_gate.validate_no_deprecated_revival(Path(__file__))
    forbidden_audit = contract_gate.validate_no_forbidden_actions(
        adapter=False, r6=False, iql_production=False, package=False,
        freeze=False, promo=False, live=False, optuna=False, broad_sweep=False,
    )
    _write_json(
        artifact_root / "input_manifest_v1.json",
        v1_gate._build_input_manifest(inputs, artifact_root) | {"action_v1": ACTION},
    )

    trades_all = skip_v1_gate._load_trade_outcomes_concat()
    trades_all["candidate_uid_v1"] = trades_all["candidate_uid"].astype(str)
    trades_all["open_ts_utc"] = pd.to_datetime(trades_all["open_ts_utc"], utc=True)
    split_df = pd.read_parquet(
        inputs["required_paths"]["split_locked_dataset"],
        columns=["candidate_uid_v1"],
    )
    accepted = set(split_df["candidate_uid_v1"].astype(str).unique())
    trades_acc = (
        trades_all[trades_all["candidate_uid_v1"].isin(accepted)]
        .sort_values(["open_ts_utc", "candidate_uid_v1"], kind="mergesort")
        .reset_index(drop=True)
    )
    candidate_uid_order = trades_acc["candidate_uid_v1"].astype(str).tolist()

    print(f"[TRUE_IQL_GPU] cuda_available={torch.cuda.is_available()} "
          f"device_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
          flush=True)
    print("[TRUE_IQL_GPU] Running 3-fold walk-forward...", flush=True)
    per_fold_results = _run_per_fold_gpu(inputs, candidate_uid_order)

    flat_evals: list[dict[str, Any]] = []
    for r in per_fold_results:
        flat_evals.extend(r["all_evaluations_v1"])
    _write_rows(
        artifact_root / "per_fold_per_variant_per_combo_evaluations_v1.csv",
        flat_evals,
    )
    _write_json(
        artifact_root / "per_fold_per_variant_per_combo_evaluations_v1.json",
        {"row_count_v1": len(flat_evals), "rows_v1": flat_evals},
    )
    promotion = v1_gate._apply_promotion_criteria(per_fold_results)
    _write_json(artifact_root / "promotion_criteria_evaluation_v1.json", promotion)

    repro = {
        "layer_name": "TRUE_IQL_GPU_REPRODUCIBILITY_AUDIT_V1",
        "fold_count_v1": len(wf_gate.FOLD_DEFINITIONS),
        "tau_grid_v1": TAU_GRID,
        "beta_grid_v1": BETA_GRID,
        "gamma_locked_v1": GAMMA_LOCKED,
        "k_vq_iterations_v1": K_VQ_ITERATIONS,
        "inner_epochs_v1": INNER_EPOCHS,
        "hidden_dim_v1": HIDDEN_DIM,
        "n_hidden_v1": N_HIDDEN,
        "lr_v1": LR,
        "weight_decay_v1": WEIGHT_DECAY,
        "batch_size_v1": BATCH_SIZE,
        "advantage_clip_v1": ADVANTAGE_CLIP,
        "seed_v1": SEED_V1,
        "device_v1": "cuda" if torch.cuda.is_available() else "cpu",
        "device_name_v1": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "torch_version_v1": torch.__version__,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
    }
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status, next_action, recommendation, headline = _go_no_go_gpu(
        per_fold_results, promotion
    )
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "TRUE_IQL_GPU_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "device_v1": repro["device_v1"],
        "device_name_v1": repro["device_name_v1"],
        "fold_count_v1": len(per_fold_results),
        "promotion_evaluation_v1": promotion,
        "per_fold_summary_v1": [
            {
                "fold_id_v1": r["fold_id_v1"],
                "best_variant_v1": r["best_variant_v1"],
                "best_tau_v1": r["best_tau_v1"],
                "best_beta_v1": r["best_beta_v1"],
                "test_at_locked_v1": r["test_at_locked_v1"],
            }
            for r in per_fold_results
        ],
        "research_only_v1": True,
        "iql_training_run_v1": True,
        "iql_production_allowed_v1": False,
        "training_blocked_v1": False,
        "next_research_gate_v1": next_action,
        "forbidden_actions_audit_v1": forbidden_audit,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {
            "layer_name": "TRUE_IQL_GPU_STATUS_V1",
            "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
            "final_status_v1": status,
            "next_action_v1": next_action,
            "training_executed_v1": True,
            "device_v1": repro["device_v1"],
        },
    )
    _write_json(
        artifact_root / "build_true_implicit_q_learning_gpu_go_no_go_v1.json",
        {
            "layer_name": "TRUE_IQL_GPU_GO_NO_GO_V1",
            "status_v1": status,
            "next_action_v1": next_action,
            "recommendation_v1": recommendation,
            "headline_v1": headline,
            "device_v1": repro["device_v1"],
            "research_only_v1": True,
            "iql_production_allowed_v1": False,
            "adapter_build_allowed_v1": False,
            "r6_allowed_v1": False,
            "package_freeze_promo_live_allowed_v1": False,
            "policy_promotion_allowed_v1": False,
            "training_allowed_v1": True,
            "downstream_block_v1": (
                "Research-only GPU True IQL (Kostrikov 2021 + NN). NOT promoted."
            ),
        },
    )

    report_lines = [
        "# Build True Implicit Q-Learning GPU V1",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        f"- Device: `{repro['device_name_v1']}` (torch {repro['torch_version_v1']})",
        "- Training: research-only; NOT promoted to runtime.",
        "",
        "## Headline",
        f"- Folds: {headline['n_folds_v1']}",
        f"- GPU NN total: **{headline['gpu_total_v1']:+.0f}** bps",
        f"- Ridge V1 total: {headline['ridge_total_v1']:+.0f} bps",
        f"- GPU - Ridge: {headline['gpu_minus_ridge_total_v1']:+.0f}",
        f"- Realized total: {sum(headline['realized_per_fold_pnl_v1']):+.0f} bps",
        f"- Promotion criteria: {headline['promotion_n_passed_v1']}/{headline['promotion_n_total_v1']}",
        "",
        "## Per-fold best (val-tuned variant + tau + beta)",
        "",
        "| Fold | Variant | tau | beta | GPU NN PNL | Realized | Lift |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in per_fold_results:
        if r["test_at_locked_v1"] is None:
            continue
        idx = per_fold_results.index(r)
        gpu_pnl = headline["true_iql_gpu_per_fold_pnl_v1"][idx]
        real_pnl = headline["realized_per_fold_pnl_v1"][idx]
        lift = headline["lifts_vs_realized_v1"][idx]
        report_lines.append(
            f"| `{r['fold_id_v1']}` | `{r['best_variant_v1']}` | "
            f"{r['best_tau_v1']} | {r['best_beta_v1']} | "
            f"**{gpu_pnl:+.0f}** | {real_pnl:+.0f} | {lift:+.0f} |"
        )
    report_lines.extend(["", "## Promotion criteria breakdown"])
    for c in promotion.get("breakdown_v1", []):
        mark = "PASS" if c["passed_v1"] else "FAIL"
        extras = ", ".join(
            f"{k}={v}" for k, v in c.items()
            if k not in {"criterion_id_v1", "passed_v1"}
        )
        report_lines.append(f"- {mark} `{c['criterion_id_v1']}` ({extras})")
    report_lines.extend(["", "## Recommendation", recommendation])
    _write_report(artifact_root / "report_v1.md", report_lines)

    artifact_manifest = {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "append_only_namespace_v1": "truth_e2e_sanity",
        "device_v1": repro["device_v1"],
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "go_no_go": str(
                artifact_root / "build_true_implicit_q_learning_gpu_go_no_go_v1.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "per_fold_per_variant_per_combo_evaluations_csv": str(
                artifact_root / "per_fold_per_variant_per_combo_evaluations_v1.csv"
            ),
            "promotion_criteria_evaluation": str(
                artifact_root / "promotion_criteria_evaluation_v1.json"
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
    parser = argparse.ArgumentParser(
        description="Materialize BUILD_TRUE_IMPLICIT_Q_LEARNING_GPU_V1."
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
