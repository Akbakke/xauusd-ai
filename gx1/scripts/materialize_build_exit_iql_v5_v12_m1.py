#!/usr/bin/env python3
"""V12 Exit-IQL trainer (v5) — drop-in replacement for v3_m1 trener with
V12-specific K_HORIZONS, reward variant, and state features.

Key differences from `materialize_build_exit_iql_v3_m1.py`:
  - K_HORIZONS = [12, 60, 240, 480, 1440]   (V12: 12min → 24h)
  - K_PRIMARY default = 240                  (4h, mid-range)
  - New reward variant R_V12 using PRE-COMPUTED reward columns from V12 builder:
        r_hold_K{N}_v1   (HOLD reward at K-horizon, capital-cost-adjusted)
        r_exit_now_v1    (EXIT_NOW reward, no capital cost — terminal action)
        is_never_fire_v1 (bool flag → applies -1000 penalty to HOLD on those rows)
  - State features extended with:
      V3-tracking (7): v3_should_exit_decision_v1, v3_decision_confidence_v1,
                      v3_max_prob_in_trade_v1, v3_consecutive_exits_v1,
                      v3_signal_acceleration_v1, v3_total_exit_decisions_v1,
                      v3_max_consecutive_exits_v1
      V10-snapshot (4): v10_p_long_at_entry_v1, v10_p_short_at_entry_v1,
                       v10_path_quality_at_entry_v1, v10_mfe_pred_at_entry_v1

Reuses:
  - v3_m1 lazy-join data loader (`load_per_bar_dataset_lazy_join`)
  - v3_m1 state matrix builder (`build_state_matrix`) — operates on whatever
    NUMERIC_STATE_COLS_PER_BAR contains, so we just override the list
  - v2_train fold/oracle/IQL training helpers (via K_HORIZONS-patch trick)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import materialize_build_exit_iql_v2 as v2_train
from gx1.scripts import materialize_build_exit_iql_v3_m1 as v3_m1


ACTION = "BUILD_EXIT_IQL_V5_V12_M1"
DEFAULT_REPORTS_ROOT = v3_m1.DEFAULT_REPORTS_ROOT
DEFAULT_CANONICAL_FEATURES_PATH = v3_m1.DEFAULT_CANONICAL_FEATURES_PATH

# ── V12 design constants ─────────────────────────────────────────────────
K_HORIZONS = [12, 60, 240, 480, 1440]
N_K = len(K_HORIZONS)
K_PRIMARY = 240                      # 4h default (mid-range; user can override)
V12_NEVER_FIRE_PENALTY_BPS = -1000.0  # matches Phase 2 builder constant

# ── V12-specific state-feature lists (extends v3_m1 base) ────────────────
V12_V3_TRACKING_COLS = [
    "v3_should_exit_decision_v1",
    "v3_decision_confidence_v1",
    "v3_max_prob_in_trade_v1",
    "v3_consecutive_exits_v1",
    "v3_signal_acceleration_v1",
    "v3_total_exit_decisions_v1",
    "v3_max_consecutive_exits_v1",
]
V12_V10_SNAPSHOT_COLS = [
    "v10_p_long_at_entry_v1",
    "v10_p_short_at_entry_v1",
    "v10_path_quality_at_entry_v1",
    "v10_mfe_pred_at_entry_v1",
]

NUMERIC_STATE_COLS_PER_BAR = (
    list(v3_m1.NUMERIC_STATE_COLS_PER_BAR)
    + V12_V3_TRACKING_COLS
    + V12_V10_SNAPSHOT_COLS
)
NUMERIC_STATE_COLS_CANDIDATE = list(v3_m1.NUMERIC_STATE_COLS_CANDIDATE)
ONE_HOT_COLS = list(v3_m1.ONE_HOT_COLS)


# ── V12 reward variant ───────────────────────────────────────────────────


REWARD_VARIANTS_V12 = ["R_V12", "R_NET_REAL", "R_REGRET"]


def build_reward_matrix_v12(df: pd.DataFrame, *, variant: str) -> np.ndarray:
    """V12 reward matrix.

    R[n, action, K]:
      action 0 = HOLD,  action 1 = EXIT_NOW

    Variants:
      - R_V12: uses pre-computed `r_hold_K{N}_v1` + `r_exit_now_v1` from Phase 2
               builder (already includes capital-cost adjustment), with
               never-fire penalty applied to HOLD rows.
      - R_NET_REAL / R_REGRET: legacy variants computed on-the-fly from raw
               `hold_max_pnl_K{N}_v1` and `exit_now_reward_bps_v1`. Useful for
               A/B comparison vs V12 design. Falls through to v3_m1 builder
               (which honors our patched K_HORIZONS).
    """
    n = len(df)
    R = np.zeros((n, v2_train.N_ACTIONS_EXIT, N_K), dtype=np.float32)

    if variant == "R_V12":
        # Pre-computed columns from Phase 2 (capital-cost already applied)
        r_exit = pd.to_numeric(df["r_exit_now_v1"], errors="coerce").fillna(0.0).to_numpy().astype(np.float32)
        never_fire = (
            df.get("is_never_fire_v1", pd.Series([False] * n))
            .fillna(False).astype(bool).to_numpy()
        )
        for ki, K in enumerate(K_HORIZONS):
            col = f"r_hold_K{K}_v1"
            if col not in df.columns:
                raise KeyError(f"R_V12 requires precomputed column {col!r} (built in Phase 2 / v12_phase2_build_per_bar_dataset.py)")
            r_hold = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy().astype(np.float32)
            # Never-fire penalty: if a bar has is_never_fire_v1=True, HOLD reward
            # at every K is -1000. This makes "never commit" infeasible for IQL.
            r_hold = np.where(never_fire, np.float32(V12_NEVER_FIRE_PENALTY_BPS), r_hold)
            r_hold = np.clip(r_hold, -1500.0, 1000.0)
            r_exit_clip = np.clip(r_exit, -500.0, 500.0)
            R[:, v2_train.ACTION_HOLD_ID, ki] = r_hold
            R[:, v2_train.ACTION_EXIT_NOW_ID, ki] = r_exit_clip
        return R

    # Legacy variants — delegate to v3_m1 builder. v3_m1 reads K_HORIZONS from
    # its module-global, so we patch that to V12's set for the call duration.
    saved_v3_m1_K = v3_m1.K_HORIZONS
    saved_v3_m1_N = v3_m1.N_K
    v3_m1.K_HORIZONS = K_HORIZONS
    v3_m1.N_K = N_K
    try:
        R = v3_m1.build_reward_matrix(df, variant=variant)
    finally:
        v3_m1.K_HORIZONS = saved_v3_m1_K
        v3_m1.N_K = saved_v3_m1_N
    return R


def build_state_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """V12 state matrix — same logic as v3_m1, but with V12 NUMERIC list."""
    parts: list[pd.DataFrame] = []
    feature_names: list[str] = []
    nan_warnings: list[tuple[str, float]] = []
    for c in NUMERIC_STATE_COLS_PER_BAR + NUMERIC_STATE_COLS_CANDIDATE:
        if c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce")
        else:
            col = pd.Series(np.zeros(len(df)), index=df.index)
        nan_frac = float(col.isna().mean()) if len(col) > 0 else 0.0
        if nan_frac > 0.05:
            nan_warnings.append((c, nan_frac))
        parts.append(col.fillna(0.0).rename(c))
        feature_names.append(c)
    for c in ONE_HOT_COLS:
        if c in df.columns:
            dummies = pd.get_dummies(df[c].astype(str), prefix=c, dummy_na=False)
            parts.append(dummies)
            feature_names.extend(dummies.columns.tolist())
    if nan_warnings:
        print(f"[{ACTION}] feature NaN warnings (frac >5%):", flush=True)
        for c, frac in nan_warnings[:20]:
            print(f"    {c}: {frac:.2%}", flush=True)
        if len(nan_warnings) > 20:
            print(f"    ... +{len(nan_warnings) - 20} more", flush=True)
    X = pd.concat(parts, axis=1).astype(np.float32).to_numpy()
    return X, feature_names


def write_artifacts(
    per_bar_dir: Path,
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    canonical_path: Path = DEFAULT_CANONICAL_FEATURES_PATH,
    out_root: Path | None = None,
    sample_n_rows: int | None = None,
    variants_subset: list[str] | None = None,
    built_at_utc: str | None = None,
    k_primary_override: int | None = None,
    exit_reward_multiplier: float = 1.0,
) -> dict[str, Any]:
    timestamp = built_at_utc or v2_train._stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    per_week_dir = per_bar_dir / "per_week"
    if not per_week_dir.exists():
        return {
            "artifact_root": str(artifact_root),
            "summary": {
                "final_status_v1": "EXIT_IQL_V5_V12_M1_BLOCKED_BY_INPUT_LOCK_MISSING",
                "next_action_v1": "REPAIR_INPUT_BEFORE_FURTHER_WORK",
                "missing_input_v1": str(per_week_dir),
            },
        }

    print(f"[{ACTION}] loading per-bar V12 dataset from {per_week_dir} (lazy-join)", flush=True)
    df = v3_m1.load_per_bar_dataset_lazy_join(
        per_week_dir,
        reports_root=reports_root,
        canonical_path=canonical_path,
        sample_n_rows=sample_n_rows,
    )
    print(f"[{ACTION}] loaded rows={len(df):,} cols={len(df.columns)}", flush=True)

    # Sanity check: V12 reward columns must be present
    required_v12 = ["r_exit_now_v1", "is_never_fire_v1"] + [f"r_hold_K{K}_v1" for K in K_HORIZONS]
    missing = [c for c in required_v12 if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"[{ACTION}] V12 reward columns missing: {missing}. "
            f"Did you build with v12_phase2_build_per_bar_dataset.py?"
        )

    X, feature_names = build_state_matrix(df)
    print(f"[{ACTION}] state matrix: {X.shape}, features={len(feature_names)}", flush=True)

    variants = variants_subset or REWARD_VARIANTS_V12
    R_by_variant: dict[str, np.ndarray] = {}
    for variant in variants:
        R_by_variant[variant] = build_reward_matrix_v12(df, variant=variant)
        print(f"[{ACTION}] reward matrix {variant}: shape={R_by_variant[variant].shape} "
              f"mean={float(R_by_variant[variant].mean()):.3f} "
              f"std={float(R_by_variant[variant].std()):.3f}", flush=True)

    effective_k_primary = int(k_primary_override) if k_primary_override is not None else K_PRIMARY
    if effective_k_primary not in K_HORIZONS:
        raise ValueError(f"k_primary_override={effective_k_primary} not in K_HORIZONS={K_HORIZONS}")

    if exit_reward_multiplier != 1.0:
        for variant in variants:
            R_by_variant[variant][:, v2_train.ACTION_EXIT_NOW_ID, :] *= float(exit_reward_multiplier)
        print(f"[{ACTION}] applied exit_reward_multiplier={exit_reward_multiplier}", flush=True)

    # Patch v2_train's K-related globals so its helpers (oracle, folds, eval)
    # operate over V12 K's. Same pattern as v3_m1.
    saved_k_horizons = v2_train.K_HORIZONS
    saved_n_k = v2_train.N_K
    saved_k_primary = v2_train.K_PRIMARY
    v2_train.K_HORIZONS = K_HORIZONS
    v2_train.N_K = N_K
    v2_train.K_PRIMARY = effective_k_primary
    try:
        R_for_strat = R_by_variant.get("R_V12", next(iter(R_by_variant.values())))
        K_primary_idx = K_HORIZONS.index(effective_k_primary)
        oracle_action = v2_train.derive_oracle_action(R_for_strat, K_primary_idx)
        oracle_dist = {ai: int((oracle_action == ai).sum()) for ai in range(v2_train.N_ACTIONS_EXIT)}
        print(f"[{ACTION}] oracle action (R_V12, K={effective_k_primary}) dist: {oracle_dist}", flush=True)

        folds = v2_train.build_stratified_folds(n_rows=len(df), oracle_action=oracle_action)

        per_fold_results: list[dict[str, Any]] = []
        flat_evaluations: list[dict[str, Any]] = []
        for variant in variants:
            for fold in folds:
                r = v2_train.evaluate_one_fold(
                    fold, X, R_by_variant[variant], oracle_action,
                    variant=variant, artifact_root=artifact_root,
                )
                per_fold_results.append(r)
                flat_evaluations.extend(r.get("all_evaluations_v1", []))

        v2_train._write_rows(artifact_root / "per_fold_per_variant_evaluations_v1.csv", flat_evaluations)
        status, next_action, recommendation, headline = v2_train.determine_status(per_fold_results)
    finally:
        v2_train.K_HORIZONS = saved_k_horizons
        v2_train.N_K = saved_n_k
        v2_train.K_PRIMARY = saved_k_primary

    status = status.replace("EXIT_IQL_V2", "EXIT_IQL_V5_V12_M1")
    next_action = next_action.replace("EXIT_IQL_V2", "EXIT_IQL_V5_V12_M1")

    summary = {
        "layer_name": "EXIT_IQL_V5_V12_M1_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": v2_train._utc_now(),
        "cadence_v1": "M1",
        "v12_design_constants_v1": {
            "k_horizons": K_HORIZONS,
            "k_primary": effective_k_primary,
            "never_fire_penalty_bps": V12_NEVER_FIRE_PENALTY_BPS,
        },
        "v12_state_extensions_v1": {
            "v3_tracking_cols": V12_V3_TRACKING_COLS,
            "v10_snapshot_cols": V12_V10_SNAPSHOT_COLS,
        },
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "input_per_bar_dir_v1": str(per_bar_dir),
        "n_rows_v1": int(len(df)),
        "n_features_v1": int(X.shape[1]),
        "feature_names_v1": feature_names,
        "k_horizons_v1": K_HORIZONS,
        "k_primary_v1": effective_k_primary,
        "n_actions_v1": v2_train.N_ACTIONS_EXIT,
        "action_labels_v1": v2_train.ACTION_LABELS_EXIT,
        "reward_variants_v1": list(R_by_variant.keys()),
        "oracle_action_distribution_v1": oracle_dist,
        "n_folds_v1": v2_train.N_FOLDS,
        "seed_v1": v2_train.SEED_V1,
        "per_fold_summary_v1": [
            {"fold_id_v1": r["fold_id_v1"], "variant_v1": r["variant_v1"],
             "best_combo_v1": r["best_combo_v1"], "best_test_metric_v1": r["best_test_metric_v1"],
             "val_class_guard_status_v1": r["val_class_guard_status_v1"],
             "val_action_counts_v1": r["val_action_counts_v1"],
             "test_baselines_v1": r["test_baselines_v1"]}
            for r in per_fold_results
        ],
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
    }
    v2_train._write_json(artifact_root / "summary_v1.json", summary)
    v2_train._write_json(artifact_root / "status_v1.json", {
        "layer_name": "EXIT_IQL_V5_V12_M1_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status, "next_action_v1": next_action,
    })
    v2_train._write_json(artifact_root / "manifest_v1.json", {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "per_fold_csv": str(artifact_root / "per_fold_per_variant_evaluations_v1.csv"),
            "trained_models_dir": str(artifact_root / "trained_models_v1"),
        },
    })
    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Materialize {ACTION} (V12 Exit-IQL trainer).")
    parser.add_argument("--per-bar-dir", type=str, required=True,
                        help="Path to V12_PER_BAR_V3TRACKED_*_LOCK (Phase 4 output)")
    parser.add_argument("--reports-root", type=str, default=str(DEFAULT_REPORTS_ROOT))
    parser.add_argument("--canonical-features", type=str, default=str(DEFAULT_CANONICAL_FEATURES_PATH))
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--sample-n-rows", type=int, default=None)
    parser.add_argument("--variants", type=str, default=None,
                        help=f"Comma-separated subset of {REWARD_VARIANTS_V12}")
    parser.add_argument("--budget", type=str, default="fast",
                        choices=list(v2_train.BUDGET_PRESETS.keys()))
    parser.add_argument("--k-primary", type=int, default=None,
                        help=f"K_PRIMARY override. Allowed: {K_HORIZONS}. Default {K_PRIMARY}.")
    parser.add_argument("--exit-reward-multiplier", type=float, default=1.0)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()

    v2_train.BUDGET_ACTIVE = v2_train.BUDGET_PRESETS[args.budget]

    variants_subset = None
    if args.variants:
        variants_subset = [v.strip() for v in args.variants.split(",") if v.strip()]
        for v in variants_subset:
            if v not in REWARD_VARIANTS_V12:
                raise ValueError(f"variant {v!r} not in {REWARD_VARIANTS_V12}")

    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None

    result = write_artifacts(
        per_bar_dir=Path(args.per_bar_dir).expanduser().resolve(),
        reports_root=Path(args.reports_root).expanduser().resolve(),
        canonical_path=Path(args.canonical_features).expanduser().resolve(),
        out_root=out_root,
        sample_n_rows=args.sample_n_rows,
        variants_subset=variants_subset,
        built_at_utc=args.built_at_utc,
        k_primary_override=args.k_primary,
        exit_reward_multiplier=args.exit_reward_multiplier,
    )
    print(json.dumps(v2_train._jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
