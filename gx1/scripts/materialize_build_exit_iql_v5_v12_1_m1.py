#!/usr/bin/env python3
"""V12.1 Exit-IQL trainer — let winners run via MFE-floor reward.

V12 (R_V12) measured: 99.6% IQL active, but capture only 45% of MFE-peak.
Mean giveback +61.9 bps/trade — IQL exits before momentum exhausts.

V12.1 fix: shape HOLD reward to give IQL an OPTIMISTIC FLOOR when V10 forecasts
high MFE. Specifically:

    hold_K = max(hold_max_pnl_K{N}_v1, mfe_first_n_pred * MFE_FLOOR_WEIGHT)
             - CAPITAL_COST * bars_held
    exit_reward = current_unrealized_pnl_bps_v1                       # unchanged
    + giveback penalty: if current_pnl < cum_peak * GIVEBACK_FRAC,
        exit_reward -= (cum_peak - current_pnl) * GIVEBACK_PENALTY_FACTOR
    + trail-stop bonus: if cum_peak >= TRAIL_PEAK_MIN AND
        current_pnl >= cum_peak * TRAIL_CAPTURE_FRAC,
        exit_reward += cum_peak * TRAIL_BONUS_FACTOR

Compared to V12:
- CAPITAL_COST: 0.1 → 0.05 bps/bar (halved — less pressure to exit)
- HOLD floor: hold_max_pnl_K{N}  ➜  max(hold_max_pnl_K{N}, V10's MFE prediction)
- EXIT giveback penalty: discourage cutting in dip
- EXIT trail-stop bonus: encourage exit near peak (not before)

Reuses V12 dataset (no new Phase 2/3/4) — only R_V12_1 reward variant differs.

Run:
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 \\
        gx1/scripts/materialize_build_exit_iql_v5_v12_1_m1.py \\
        --per-bar-dir <V12_V3TRACKED_LOCK> \\
        --out-root <V12_1_TRAINED_LOCK> \\
        --budget medium --sample-n-rows 2000000
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
from gx1.scripts import materialize_build_exit_iql_v5_v12_m1 as v12_train


ACTION = "BUILD_EXIT_IQL_V5_V12_1_M1"
DEFAULT_REPORTS_ROOT = v3_m1.DEFAULT_REPORTS_ROOT
DEFAULT_CANONICAL_FEATURES_PATH = v3_m1.DEFAULT_CANONICAL_FEATURES_PATH

# ── V12.1 design constants ────────────────────────────────────────────────
K_HORIZONS = list(v12_train.K_HORIZONS)            # reuse V12: [12,60,240,480,1440]
N_K = len(K_HORIZONS)
K_PRIMARY = v12_train.K_PRIMARY                    # 240

# V12.1 reward shaping params (defaults — overridable via CLI)
DEFAULT_CAPITAL_COST_BPS = 0.05                    # V12 was 0.10 — halved to ease HOLD pressure
DEFAULT_MFE_FLOOR_WEIGHT = 0.5                     # use 50% of V10's MFE prediction as HOLD floor
DEFAULT_GIVEBACK_FRAC = 0.7                        # if current_pnl < cum_peak * 0.7 → giveback regime
DEFAULT_GIVEBACK_PENALTY = 0.5                     # penalize half of giveback bps in EXIT reward
DEFAULT_TRAIL_PEAK_MIN_BPS = 50.0                  # only apply trail bonus if peak ≥ 50 bps achieved
DEFAULT_TRAIL_CAPTURE_FRAC = 0.8                   # bonus when current_pnl ≥ 80% of peak
DEFAULT_TRAIL_BONUS_FRAC = 0.0                     # V12.1.1 NO_TRAIL cemented (grid winner) — disable trail-stop bonus
NEVER_FIRE_PENALTY_BPS = v12_train.V12_NEVER_FIRE_PENALTY_BPS

# ── V12.1 reuses V12 state-feature lists ──────────────────────────────────
NUMERIC_STATE_COLS_PER_BAR = list(v12_train.NUMERIC_STATE_COLS_PER_BAR)
NUMERIC_STATE_COLS_CANDIDATE = list(v12_train.NUMERIC_STATE_COLS_CANDIDATE)
ONE_HOT_COLS = list(v12_train.ONE_HOT_COLS)
build_state_matrix = v12_train.build_state_matrix


REWARD_VARIANTS_V12_1 = ["R_V12_1", "R_V12", "R_NET_REAL"]


def build_reward_matrix_v12_1(
    df: pd.DataFrame,
    *,
    variant: str,
    capital_cost_bps: float = DEFAULT_CAPITAL_COST_BPS,
    mfe_floor_weight: float = DEFAULT_MFE_FLOOR_WEIGHT,
    giveback_frac: float = DEFAULT_GIVEBACK_FRAC,
    giveback_penalty: float = DEFAULT_GIVEBACK_PENALTY,
    trail_peak_min_bps: float = DEFAULT_TRAIL_PEAK_MIN_BPS,
    trail_capture_frac: float = DEFAULT_TRAIL_CAPTURE_FRAC,
    trail_bonus_frac: float = DEFAULT_TRAIL_BONUS_FRAC,
) -> np.ndarray:
    """V12.1 reward matrix.

    Variants:
      - R_V12_1: V12.1 design — MFE floor + reduced capital cost + giveback penalty + trail-stop bonus
      - R_V12: legacy V12 (delegates)
      - R_NET_REAL: legacy v3_m1 baseline
    """
    n = len(df)
    R = np.zeros((n, v2_train.N_ACTIONS_EXIT, N_K), dtype=np.float32)

    if variant == "R_V12_1":
        # Required columns
        cur_pnl = pd.to_numeric(df["current_unrealized_pnl_bps_v1"], errors="coerce").fillna(0.0).to_numpy().astype(np.float32)
        cum_peak = pd.to_numeric(df["current_mfe_bps_v1"], errors="coerce").fillna(0.0).to_numpy().astype(np.float32)
        bars_held = pd.to_numeric(df["bars_in_trade_v1"], errors="coerce").fillna(0.0).to_numpy().astype(np.float32)
        mfe_pred = pd.to_numeric(df.get("mfe_first_n_pred", pd.Series(np.zeros(n))), errors="coerce").fillna(0.0).to_numpy().astype(np.float32)
        never_fire = (
            df.get("is_never_fire_v1", pd.Series([False] * n))
            .fillna(False).astype(bool).to_numpy()
        )

        # ── EXIT_NOW reward (K-independent, broadcast to all K) ──
        r_exit = cur_pnl.copy()
        # Giveback penalty: if current_pnl < cum_peak * giveback_frac → penalize
        giveback_bps = np.maximum(0.0, cum_peak - cur_pnl)
        giveback_mask = (cur_pnl < cum_peak * giveback_frac) & (cum_peak > 0)
        r_exit = np.where(giveback_mask, r_exit - giveback_bps * giveback_penalty, r_exit)
        # Trail-stop bonus: if peak ≥ min AND captured ≥ trail_capture_frac of peak
        trail_mask = (cum_peak >= trail_peak_min_bps) & (cur_pnl >= cum_peak * trail_capture_frac)
        r_exit = np.where(trail_mask, r_exit + cum_peak * trail_bonus_frac, r_exit)
        r_exit = np.clip(r_exit, -500.0, 700.0)

        # ── HOLD reward per K (K-dependent floor) ──
        for ki, K in enumerate(K_HORIZONS):
            col = f"hold_max_pnl_K{K}_v1"
            if col not in df.columns:
                raise KeyError(f"R_V12_1 requires {col!r} from Phase 2 builder")
            hold_max_K = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy().astype(np.float32)
            # MFE floor: use max(hold_max_K, mfe_pred * weight)
            hold_floor = np.maximum(hold_max_K, mfe_pred * mfe_floor_weight)
            r_hold = hold_floor - capital_cost_bps * bars_held
            # Never-fire penalty: HOLD = -1000 if at runaway-cap
            r_hold = np.where(never_fire, np.float32(NEVER_FIRE_PENALTY_BPS), r_hold)
            r_hold = np.clip(r_hold, -1500.0, 1000.0)

            R[:, v2_train.ACTION_HOLD_ID, ki] = r_hold
            R[:, v2_train.ACTION_EXIT_NOW_ID, ki] = r_exit
        return R

    # Legacy variants delegate to V12 trainer (which handles K_HORIZONS patching)
    return v12_train.build_reward_matrix_v12(df, variant=variant)


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
    capital_cost_bps: float = DEFAULT_CAPITAL_COST_BPS,
    mfe_floor_weight: float = DEFAULT_MFE_FLOOR_WEIGHT,
    giveback_frac: float = DEFAULT_GIVEBACK_FRAC,
    giveback_penalty: float = DEFAULT_GIVEBACK_PENALTY,
    trail_peak_min_bps: float = DEFAULT_TRAIL_PEAK_MIN_BPS,
    trail_capture_frac: float = DEFAULT_TRAIL_CAPTURE_FRAC,
    trail_bonus_frac: float = DEFAULT_TRAIL_BONUS_FRAC,
    year_filter: str | None = None,
) -> dict[str, Any]:
    timestamp = built_at_utc or v2_train._stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    per_week_dir = per_bar_dir / "per_week"
    if not per_week_dir.exists():
        return {
            "artifact_root": str(artifact_root),
            "summary": {"final_status_v1": "BLOCKED_INPUT_MISSING",
                        "missing_input_v1": str(per_week_dir)},
        }

    # Year-filter: pre-filter weekly parquets by filename year prefix.
    # filename = exit_per_bar_m1_TRUTH_MONFRI_WEEK_YYYYMMDD_YYYYMMDD.parquet
    year_filter_env = (year_filter or "").strip()
    if year_filter_env:
        all_parquets = sorted(per_week_dir.glob("exit_per_bar_m1_*.parquet"))
        # Filter where week's start-date year matches
        kept = [p for p in all_parquets if p.name.split("_WEEK_")[1][:4] == year_filter_env]
        print(f"[{ACTION}] year_filter={year_filter_env} → {len(kept)}/{len(all_parquets)} weekly parquets selected", flush=True)
        # Build temporary symlink dir to point lazy-join at filtered subset
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"v12_1_year_{year_filter_env}_"))
        for p in kept:
            (tmp_dir / p.name).symlink_to(p)
        load_dir = tmp_dir
    else:
        load_dir = per_week_dir

    print(f"[{ACTION}] loading V12 V3TRACKED dataset from {load_dir} (lazy-join)", flush=True)
    df = v3_m1.load_per_bar_dataset_lazy_join(
        load_dir, reports_root=reports_root, canonical_path=canonical_path,
        sample_n_rows=sample_n_rows,
    )
    print(f"[{ACTION}] loaded rows={len(df):,} cols={len(df.columns)}", flush=True)

    # Sanity: V12.1-required columns
    required = ["current_unrealized_pnl_bps_v1", "current_mfe_bps_v1",
                "bars_in_trade_v1", "mfe_first_n_pred"] + [f"hold_max_pnl_K{K}_v1" for K in K_HORIZONS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{ACTION}] missing required cols: {missing}")

    X, feature_names = build_state_matrix(df)
    print(f"[{ACTION}] state matrix: {X.shape}, features={len(feature_names)}", flush=True)

    variants = variants_subset or REWARD_VARIANTS_V12_1
    R_by_variant: dict[str, np.ndarray] = {}
    for variant in variants:
        R_by_variant[variant] = build_reward_matrix_v12_1(
            df, variant=variant,
            capital_cost_bps=capital_cost_bps,
            mfe_floor_weight=mfe_floor_weight,
            giveback_frac=giveback_frac,
            giveback_penalty=giveback_penalty,
            trail_peak_min_bps=trail_peak_min_bps,
            trail_capture_frac=trail_capture_frac,
            trail_bonus_frac=trail_bonus_frac,
        )
        R = R_by_variant[variant]
        print(f"[{ACTION}] reward {variant}: shape={R.shape} mean={float(R.mean()):.3f} std={float(R.std()):.3f}", flush=True)
        # Also: oracle dist preview
        if variant == "R_V12_1":
            R_K = R[:, :, K_HORIZONS.index(K_PRIMARY)]
            oracle_v12_1 = np.argmax(R_K, axis=1)
            dist = {0: int((oracle_v12_1 == 0).sum()), 1: int((oracle_v12_1 == 1).sum())}
            print(f"[{ACTION}]   R_V12_1 oracle (K={K_PRIMARY}): HOLD={dist[0]:,} ({100*dist[0]/len(R_K):.1f}%) "
                  f"EXIT={dist[1]:,} ({100*dist[1]/len(R_K):.1f}%)", flush=True)

    effective_k_primary = int(k_primary_override) if k_primary_override is not None else K_PRIMARY
    if effective_k_primary not in K_HORIZONS:
        raise ValueError(f"k_primary {effective_k_primary} not in {K_HORIZONS}")

    if exit_reward_multiplier != 1.0:
        for variant in variants:
            R_by_variant[variant][:, v2_train.ACTION_EXIT_NOW_ID, :] *= float(exit_reward_multiplier)

    saved_k_horizons = v2_train.K_HORIZONS
    saved_n_k = v2_train.N_K
    saved_k_primary = v2_train.K_PRIMARY
    v2_train.K_HORIZONS = K_HORIZONS
    v2_train.N_K = N_K
    v2_train.K_PRIMARY = effective_k_primary
    try:
        R_for_strat = R_by_variant.get("R_V12_1", next(iter(R_by_variant.values())))
        K_primary_idx = K_HORIZONS.index(effective_k_primary)
        oracle_action = v2_train.derive_oracle_action(R_for_strat, K_primary_idx)
        oracle_dist = {ai: int((oracle_action == ai).sum()) for ai in range(v2_train.N_ACTIONS_EXIT)}
        print(f"[{ACTION}] stratification oracle (R_V12_1, K={effective_k_primary}): {oracle_dist}", flush=True)

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

    status = status.replace("EXIT_IQL_V2", "EXIT_IQL_V5_V12_1_M1")
    next_action = next_action.replace("EXIT_IQL_V2", "EXIT_IQL_V5_V12_1_M1")

    summary = {
        "layer_name": "EXIT_IQL_V5_V12_1_M1_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": v2_train._utc_now(),
        "cadence_v1": "M1",
        "v12_1_design_constants_v1": {
            "k_horizons": K_HORIZONS,
            "k_primary": effective_k_primary,
            "capital_cost_bps": capital_cost_bps,
            "mfe_floor_weight": mfe_floor_weight,
            "giveback_frac": giveback_frac,
            "giveback_penalty": giveback_penalty,
            "trail_peak_min_bps": trail_peak_min_bps,
            "trail_capture_frac": trail_capture_frac,
            "trail_bonus_frac": trail_bonus_frac,
            "never_fire_penalty_bps": NEVER_FIRE_PENALTY_BPS,
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
    }
    v2_train._write_json(artifact_root / "summary_v1.json", summary)
    v2_train._write_json(artifact_root / "manifest_v1.json", {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "per_fold_csv": str(artifact_root / "per_fold_per_variant_evaluations_v1.csv"),
            "trained_models_dir": str(artifact_root / "trained_models_v1"),
        },
    })
    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    p = argparse.ArgumentParser(description=f"Materialize {ACTION} (V12.1 trainer with MFE-floor reward).")
    p.add_argument("--per-bar-dir", type=str, required=True)
    p.add_argument("--reports-root", type=str, default=str(DEFAULT_REPORTS_ROOT))
    p.add_argument("--canonical-features", type=str, default=str(DEFAULT_CANONICAL_FEATURES_PATH))
    p.add_argument("--out-root", type=str, default=None)
    p.add_argument("--sample-n-rows", type=int, default=None)
    p.add_argument("--variants", type=str, default=None)
    p.add_argument("--budget", type=str, default="fast", choices=list(v2_train.BUDGET_PRESETS.keys()))
    p.add_argument("--k-primary", type=int, default=None)
    p.add_argument("--exit-reward-multiplier", type=float, default=1.0)
    p.add_argument("--built-at-utc", type=str, default=None)
    p.add_argument("--capital-cost-bps", type=float, default=DEFAULT_CAPITAL_COST_BPS)
    p.add_argument("--mfe-floor-weight", type=float, default=DEFAULT_MFE_FLOOR_WEIGHT)
    p.add_argument("--giveback-frac", type=float, default=DEFAULT_GIVEBACK_FRAC)
    p.add_argument("--giveback-penalty", type=float, default=DEFAULT_GIVEBACK_PENALTY)
    p.add_argument("--trail-peak-min-bps", type=float, default=DEFAULT_TRAIL_PEAK_MIN_BPS)
    p.add_argument("--trail-capture-frac", type=float, default=DEFAULT_TRAIL_CAPTURE_FRAC)
    p.add_argument("--trail-bonus-frac", type=float, default=DEFAULT_TRAIL_BONUS_FRAC)
    p.add_argument("--year-filter", type=str, default=None,
                   help="Filter weekly parquets to weeks starting in this year (e.g., '2022')")
    args = p.parse_args()

    v2_train.BUDGET_ACTIVE = v2_train.BUDGET_PRESETS[args.budget]
    variants_subset = None
    if args.variants:
        variants_subset = [v.strip() for v in args.variants.split(",") if v.strip()]

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
        capital_cost_bps=args.capital_cost_bps,
        mfe_floor_weight=args.mfe_floor_weight,
        giveback_frac=args.giveback_frac,
        giveback_penalty=args.giveback_penalty,
        trail_peak_min_bps=args.trail_peak_min_bps,
        trail_capture_frac=args.trail_capture_frac,
        trail_bonus_frac=args.trail_bonus_frac,
        year_filter=args.year_filter,
    )
    print(json.dumps(v2_train._jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
