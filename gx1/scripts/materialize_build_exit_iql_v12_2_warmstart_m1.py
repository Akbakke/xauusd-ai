#!/usr/bin/env python3
"""V12.2 warm-start trainer — continue Exit-IQL v5 training from V12.1.1 checkpoint.

Loads V12.1.1 (NO_TRAIL) checkpoints, trains on a NEW 2M sample (different seed)
to broaden coverage cumulatively. Net-effect: Q-net sees ~3.9M unique states
across V12.1.1 (2M, seed=20260501) + V12.2 (2M, seed=20260601) combined,
≈ 6.2% of full 63.6M dataset.

Architecture:
- Reuses V12.1 build_state_matrix + build_reward_matrix_v12_1
- Reuses v2_train.build_stratified_folds for fold split
- INLINE IQL training loop (clone of iql_core.train_multi_head_entry_iql)
  with init from V12.1.1 fold checkpoints
- Saves V12.2 ckpts in same format as V12.1.1

Run:
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 \\
        gx1/scripts/materialize_build_exit_iql_v12_2_warmstart_m1.py \\
        --init-checkpoint-lock <V12_1_1_LOCK> \\
        --per-bar-dir <V12_V3TRACKED_LOCK> \\
        --out-root <V12_2_LOCK> \\
        --sample-n-rows 2000000 --warmstart-seed 20260601 \\
        --extra-k-iterations 20  # extra training rounds on warm Q+V
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import materialize_build_exit_iql_v2 as v2_train
from gx1.scripts import materialize_build_exit_iql_v3_m1 as v3_m1
from gx1.scripts import materialize_build_exit_iql_v5_v12_1_m1 as v12_1
from gx1.scripts import entry_iql_multi_head_gpu_core_v1 as iql_core


ACTION = "BUILD_EXIT_IQL_V12_2_WARMSTART_M1"
DEFAULT_REPORTS_ROOT = v3_m1.DEFAULT_REPORTS_ROOT
DEFAULT_CANONICAL_FEATURES_PATH = v3_m1.DEFAULT_CANONICAL_FEATURES_PATH
K_HORIZONS = list(v12_1.K_HORIZONS)
K_PRIMARY = v12_1.K_PRIMARY
N_K = len(K_HORIZONS)


def expectile_loss(diff: torch.Tensor, tau: float,
                   sample_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Reuse iql_core's expectile loss."""
    return iql_core.expectile_loss(diff, tau, sample_weights=sample_weights)


def warm_start_iql_train(
    X_np: np.ndarray, R_np: np.ndarray,
    *,
    k_horizons: list[int],
    n_actions: int,
    init_q_state_dict: dict,
    init_v_state_dict: dict,
    init_feature_means: np.ndarray,
    init_feature_stds: np.ndarray,
    hidden_dim: int, n_hidden: int, dropout: float,
    tau: float = 0.8,
    extra_k_iterations: int = 20,
    epochs_q: int = 1,
    epochs_v: int = 1,
    lr: float = 5e-4,        # halved from cold-start LR for fine-tuning
    weight_decay: float = 1e-5,
    batch_size: int = 4096,
    seed: int = 20260601,
    prefer_cuda: bool = True,
) -> tuple[nn.Module, nn.Module, np.ndarray, np.ndarray]:
    """Warm-start IQL training: continue from V12.1.1 weights on new sample.

    Skips iql_core's Phase 1 (Q-warmup MSE) since Q-net is already trained.
    Goes straight to alternating V/Q updates for `extra_k_iterations` rounds.
    Uses V12.1.1's normalization stats (consistent z-score across runs).
    """
    n, na, nk = R_np.shape
    assert na == n_actions and nk == len(k_horizons)

    torch.manual_seed(seed)
    device = iql_core._device(prefer_cuda)
    state_dim = X_np.shape[1]

    # Normalize using INIT means/stds (consistent w/ V12.1.1)
    X_norm = ((X_np - init_feature_means) / np.maximum(init_feature_stds, 1e-6)).astype(np.float32)
    X_norm = np.where(np.isnan(X_norm), 0.0, X_norm)
    R_clean = np.where(np.isnan(R_np), 0.0, R_np).astype(np.float32)

    X_t = torch.as_tensor(X_norm, dtype=torch.float32, device=device)
    R_t = torch.as_tensor(R_clean, dtype=torch.float32, device=device)
    R_flat = R_t.view(n, n_actions * nk)
    w_t = torch.ones(n, dtype=torch.float32, device=device)

    # Construct nets + load init weights (warm start)
    q_net = iql_core.MLP(state_dim, n_actions * nk,
                          hidden_dim=hidden_dim, n_hidden=n_hidden, dropout=dropout).to(device)
    v_net = iql_core.MLP(state_dim, nk,
                          hidden_dim=hidden_dim, n_hidden=n_hidden, dropout=dropout).to(device)
    q_net.load_state_dict(init_q_state_dict)
    v_net.load_state_dict(init_v_state_dict)
    q_opt = torch.optim.Adam(q_net.parameters(), lr=lr, weight_decay=weight_decay)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=lr, weight_decay=weight_decay)
    g = torch.Generator(device="cpu").manual_seed(seed)

    # Skip Phase 1 (Q-warmup) — Q-net already trained.
    # Run alternating V/Q updates for extra_k_iterations rounds.
    for k_iter in range(extra_k_iterations):
        # V update
        q_net.eval()
        with torch.no_grad():
            q_full = q_net(X_t).view(n, n_actions, nk)
            q_max_over_a = q_full.max(dim=1).values
        v_net.train()
        for _ in range(epochs_v):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                v_pred = v_net(X_t[idx])
                diff = q_max_over_a[idx] - v_pred
                loss = expectile_loss(diff, tau, sample_weights=w_t[idx])
                v_opt.zero_grad()
                loss.backward()
                v_opt.step()
        # Q refit
        v_net.eval()
        q_net.train()
        for _ in range(epochs_q):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                q_pred = q_net(X_t[idx])
                diff = q_pred - R_flat[idx]
                loss = (w_t[idx].unsqueeze(-1) * diff.pow(2)).mean()
                q_opt.zero_grad()
                loss.backward()
                q_opt.step()

    q_net.eval()
    v_net.eval()
    # Return updated nets + normalizer (unchanged — we kept V12.1.1's)
    return q_net, v_net, init_feature_means, init_feature_stds


def main() -> None:
    p = argparse.ArgumentParser(description=ACTION)
    p.add_argument("--init-checkpoint-lock", required=True,
                   help="Path to V12.1.1 LOCK with trained_models_v1/R_V12_1_FOLD_*.pt")
    p.add_argument("--per-bar-dir", required=True)
    p.add_argument("--out-root", default=None)
    p.add_argument("--sample-n-rows", type=int, default=2000000)
    p.add_argument("--warmstart-seed", type=int, default=20260601,
                   help="Seed for new sample selection — different from V12.1.1 (20260501)")
    p.add_argument("--extra-k-iterations", type=int, default=20,
                   help="Extra alternating V/Q rounds on warm-started weights")
    p.add_argument("--lr", type=float, default=5e-4,
                   help="Lower LR for fine-tuning (V12.1.1 used 1e-3)")
    # Reward params (NO_TRAIL config from V12.1 grid winner)
    p.add_argument("--capital-cost-bps", type=float, default=0.05)
    p.add_argument("--mfe-floor-weight", type=float, default=0.5)
    p.add_argument("--giveback-frac", type=float, default=0.7)
    p.add_argument("--giveback-penalty", type=float, default=0.5)
    p.add_argument("--trail-capture-frac", type=float, default=0.8)
    p.add_argument("--trail-bonus-frac", type=float, default=0.0)
    args = p.parse_args()

    init_lock = Path(args.init_checkpoint_lock).expanduser().resolve()
    per_bar_dir = Path(args.per_bar_dir).expanduser().resolve()
    timestamp = v2_train._stamp()
    out_root = (Path(args.out_root).expanduser().resolve()
                if args.out_root
                else DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "trained_models_v1").mkdir(exist_ok=True)

    print(f"[{ACTION}] init checkpoint: {init_lock}")
    print(f"[{ACTION}] out: {out_root}")
    print(f"[{ACTION}] sample={args.sample_n_rows:,} seed={args.warmstart_seed} extra_k_iter={args.extra_k_iterations}")

    # ── Load V12.1.1 fold checkpoints ─────────────────────────────────────
    init_ckpts = {}
    for fold_i in (1, 2, 3):
        ckpt_path = init_lock / "trained_models_v1" / f"R_V12_1_FOLD_{fold_i}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing init ckpt: {ckpt_path}")
        init_ckpts[fold_i] = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"[{ACTION}] loaded {len(init_ckpts)} init checkpoints")
    # Get arch params from first ckpt
    ckpt0 = init_ckpts[1]
    hidden_dim = int(ckpt0["hidden_dim"])
    n_hidden = int(ckpt0["n_hidden"])
    dropout = float(ckpt0["dropout"])
    n_actions = int(ckpt0["n_actions"])
    state_dim = int(ckpt0["state_dim"])
    print(f"[{ACTION}] arch: state_dim={state_dim} hidden={hidden_dim}x{n_hidden} dropout={dropout}")

    # ── Load NEW dataset (different seed than V12.1.1) ────────────────────
    print(f"[{ACTION}] loading V12 V3TRACKED dataset (sample={args.sample_n_rows:,}, seed={args.warmstart_seed})")
    df = v3_m1.load_per_bar_dataset_lazy_join(
        per_bar_dir / "per_week",
        reports_root=DEFAULT_REPORTS_ROOT,
        canonical_path=DEFAULT_CANONICAL_FEATURES_PATH,
        sample_n_rows=args.sample_n_rows,
        seed=args.warmstart_seed,
    )
    print(f"[{ACTION}] loaded rows={len(df):,}")

    # Build X + R (using V12.1 helpers, NO_TRAIL params from CLI)
    X, feature_names = v12_1.build_state_matrix(df)
    print(f"[{ACTION}] state matrix: {X.shape}")
    if X.shape[1] != state_dim:
        raise RuntimeError(f"State dim mismatch: ckpt={state_dim} vs new dataset={X.shape[1]}")

    R = v12_1.build_reward_matrix_v12_1(
        df, variant="R_V12_1",
        capital_cost_bps=args.capital_cost_bps,
        mfe_floor_weight=args.mfe_floor_weight,
        giveback_frac=args.giveback_frac,
        giveback_penalty=args.giveback_penalty,
        trail_peak_min_bps=v12_1.DEFAULT_TRAIL_PEAK_MIN_BPS,
        trail_capture_frac=args.trail_capture_frac,
        trail_bonus_frac=args.trail_bonus_frac,
    )
    print(f"[{ACTION}] reward matrix: shape={R.shape} mean={float(R.mean()):.3f}")

    # Stratified folds (using V12.1 helpers; patched K_HORIZONS for v2_train)
    saved_K = v2_train.K_HORIZONS
    saved_NK = v2_train.N_K
    saved_KP = v2_train.K_PRIMARY
    v2_train.K_HORIZONS = K_HORIZONS
    v2_train.N_K = N_K
    v2_train.K_PRIMARY = K_PRIMARY
    try:
        K_primary_idx = K_HORIZONS.index(K_PRIMARY)
        oracle_action = v2_train.derive_oracle_action(R, K_primary_idx)
        oracle_dist = {ai: int((oracle_action == ai).sum()) for ai in range(v2_train.N_ACTIONS_EXIT)}
        print(f"[{ACTION}] oracle dist (R_V12_1, K=240): {oracle_dist}")
        folds = v2_train.build_stratified_folds(n_rows=len(df), oracle_action=oracle_action)
    finally:
        v2_train.K_HORIZONS = saved_K
        v2_train.N_K = saved_NK
        v2_train.K_PRIMARY = saved_KP

    # ── Per-fold warm-start training ──────────────────────────────────────
    fold_results = []
    for fold_i, fold in enumerate(folds, start=1):
        ckpt = init_ckpts[fold_i]
        init_q = ckpt["q_state_dict"]
        init_v = ckpt["v_state_dict"]
        init_means = np.asarray(ckpt["feature_means"], dtype=np.float32)
        init_stds = np.asarray(ckpt["feature_stds"], dtype=np.float32)

        # v2_train.build_stratified_folds returns dicts with _v1-suffix keys
        train_idx = fold["train_idx_v1"]
        test_idx_local = fold["test_idx_v1"]
        X_train = X[train_idx]
        R_train = R[train_idx]
        print(f"\n[{ACTION}] FOLD_{fold_i} warm-start: train_rows={len(X_train):,}  extra_k={args.extra_k_iterations}")

        t0 = time.time()
        q_net, v_net, means, stds = warm_start_iql_train(
            X_train, R_train,
            k_horizons=K_HORIZONS, n_actions=n_actions,
            init_q_state_dict=init_q, init_v_state_dict=init_v,
            init_feature_means=init_means, init_feature_stds=init_stds,
            hidden_dim=hidden_dim, n_hidden=n_hidden, dropout=dropout,
            extra_k_iterations=args.extra_k_iterations,
            lr=args.lr,
            seed=args.warmstart_seed + fold_i,
        )
        elapsed = time.time() - t0
        print(f"[{ACTION}] FOLD_{fold_i} warm-start done in {elapsed:.0f}s")

        # Evaluate on test split (sanity)
        X_test = X[test_idx_local]
        R_test = R[test_idx_local]
        X_test_norm = (X_test - means) / np.maximum(stds, 1e-6)
        X_test_norm = np.where(np.isnan(X_test_norm), 0.0, X_test_norm).astype(np.float32)
        device = next(q_net.parameters()).device
        with torch.no_grad():
            q_full = q_net(torch.as_tensor(X_test_norm, device=device)).view(-1, n_actions, N_K)
            actions_pred = q_full.mean(dim=2).argmax(dim=1).cpu().numpy()
        # mean reward at K_primary
        mean_reward = float(R_test[np.arange(len(R_test)), actions_pred, K_primary_idx].mean())
        exit_frac = float((actions_pred == 1).mean())
        print(f"[{ACTION}] FOLD_{fold_i} test: mean_reward_bps={mean_reward:+.2f}  exit_frac={exit_frac*100:.1f}%")
        fold_results.append({
            "fold_id_v1": f"FOLD_{fold_i}",
            "best_test_metric_v1": {
                "mean_reward_bps_v1": mean_reward,
                "fractions_v1": {"frac_action_0_v1": 1 - exit_frac, "frac_action_1_v1": exit_frac},
                "counts_v1": {"n_action_0_v1": int((actions_pred == 0).sum()),
                               "n_action_1_v1": int((actions_pred == 1).sum()),
                               "n_total_v1": int(len(actions_pred))},
            },
            "variant_v1": "R_V12_1",
        })

        # Save warm-started checkpoint (compatible with ExitIQLV2Adapter.load)
        out_ckpt = out_root / "trained_models_v1" / f"R_V12_1_FOLD_{fold_i}.pt"
        torch.save({
            "q_state_dict": q_net.state_dict(),
            "v_state_dict": v_net.state_dict(),
            "feature_means": means.tolist(),
            "feature_stds": stds.tolist(),
            "k_horizons": K_HORIZONS,
            "n_actions": n_actions,
            "state_dim": state_dim,
            "variant": "R_V12_1", "fold_id": f"FOLD_{fold_i}",
            "hidden_dim": hidden_dim, "n_hidden": n_hidden, "dropout": dropout,
            "schema_v1": "MULTI_HEAD_EXIT_IQL_V2_CHECKPOINT",
            "action_labels": v2_train.ACTION_LABELS_EXIT,
            "warmstart_from_v1": str(init_lock),
            "warmstart_seed_v1": args.warmstart_seed,
            "warmstart_extra_k_iter_v1": args.extra_k_iterations,
        }, out_ckpt)
        print(f"[{ACTION}] saved {out_ckpt.name}")

    # Summary (compatible with ExitIQLV2Adapter.load — needs feature_names_v1)
    summary = {
        "action_v1": ACTION,
        "built_at_utc_v1": v2_train._utc_now(),
        "input_per_bar_dir_v1": str(per_bar_dir),
        "init_checkpoint_lock_v1": str(init_lock),
        "n_rows_v1": int(len(df)),
        "n_features_v1": int(X.shape[1]),
        "feature_names_v1": feature_names,
        "k_horizons_v1": K_HORIZONS,
        "k_primary_v1": K_PRIMARY,
        "n_actions_v1": n_actions,
        "action_labels_v1": v2_train.ACTION_LABELS_EXIT,
        "reward_variants_v1": ["R_V12_1"],
        "warmstart_seed_v1": args.warmstart_seed,
        "warmstart_extra_k_iter_v1": args.extra_k_iterations,
        "warmstart_lr_v1": args.lr,
        "v12_1_no_trail_params_v1": {
            "capital_cost_bps": args.capital_cost_bps,
            "mfe_floor_weight": args.mfe_floor_weight,
            "giveback_frac": args.giveback_frac,
            "giveback_penalty": args.giveback_penalty,
            "trail_capture_frac": args.trail_capture_frac,
            "trail_bonus_frac": args.trail_bonus_frac,
        },
        "per_fold_summary_v1": fold_results,
        "research_only_v1": True,
    }
    v2_train._write_json(out_root / "summary_v1.json", summary)
    print(f"\n[{ACTION}] DONE. Output: {out_root}")
    print(json.dumps({"out_root": str(out_root),
                       "n_ckpts": len(fold_results),
                       "mean_reward_test_avg": float(np.mean([f["best_test_metric_v1"]["mean_reward_bps_v1"] for f in fold_results]))},
                     indent=2))


if __name__ == "__main__":
    main()
