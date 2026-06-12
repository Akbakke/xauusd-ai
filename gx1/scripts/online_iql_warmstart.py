#!/usr/bin/env python3
"""Warm-start Entry-IQL update from a replay buffer of live journal events.

The cement Entry-IQL was trained cold (~6-8h) on 6 years of fwd-outcome
parquet. This script does the opposite: load cement weights as initialization,
continue training for a small number of epochs with a low learning rate on the
live replay buffer produced by build_online_replay_buffer.py.

Goal: regime-adapt the Q/V heads without catastrophic forgetting. Cement is the
strong prior; live data nudges weights toward whatever the new regime rewards.

Designed-not-forked:
  - Architecture/loss/training loop reused from
    entry_iql_multi_head_gpu_core_v1.train_multi_head_entry_iql — we don't
    re-implement IQL; we call it with a different lr/epochs/init.
  - Checkpoint schema matches entry_iql_v2_adapter.from_artifact_root() so live
    runtime can load the warm-started bundle with no code change.
  - Vedtak gate identical to materialize_build_entry_iql_v2 (--vedtak required).

DOES NOT auto-promote the new bundle. After this script writes the output:
  1. Phase 6 OOT-validate the warm-started bundle.
  2. If OOT >= cement, vedtak: edit PROJECT_STATE_artifacts.json manually to
     swap active_bundle, then restart paper_runner.

Usage:
  python -m gx1.scripts.online_iql_warmstart \
      --base-bundle /home/andre2/GX1_DATA/.../BUILD_ENTRY_IQL_V2_COSTFIX_HEADS_FAST_20260528T071646Z \
      --replay /home/andre2/GX1_DATA/reports/online_replay/replay_W2026_23.parquet \
      --variant R_WAIT_OPP_K96_LAM50 \
      --fold FOLD_1 \
      --out-dir /home/andre2/GX1_DATA/reports/online_iql/warmstart_W2026_23 \
      --vedtak weekly_warmstart_W2026_23
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gx1.scripts import entry_iql_multi_head_gpu_core_v1 as iql_core
from gx1.scripts.build_online_replay_buffer import LABEL_CONVENTION_CEMENT_M5
from gx1_guards.gates import GateError, require_retrain_vedtak

logging.basicConfig(level=logging.INFO, format="[warmstart] %(message)s")
LOG = logging.getLogger("online_iql_warmstart")


def load_cement(base_bundle: Path, variant: str, fold: str) -> dict:
    summary_path = base_bundle / "summary_v1.json"
    ckpt_path = base_bundle / "trained_models_v1" / f"{variant}_{fold}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"cement checkpoint missing: {ckpt_path}")
    summary = json.loads(summary_path.read_text())
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {"ckpt": ckpt, "summary": summary, "ckpt_path": ckpt_path}


def buffer_to_tensors(df: pd.DataFrame, k_horizons: list[int]) -> tuple[np.ndarray, np.ndarray]:
    state_cols = sorted(c for c in df.columns if c.startswith("s") and c[1:].isdigit())
    if not state_cols:
        raise RuntimeError("replay buffer missing state columns s000, s001, ...")
    X = df[state_cols].to_numpy(dtype=np.float32)

    n = len(df)
    n_actions = 3  # SKIP, LONG, SHORT
    n_k = len(k_horizons)
    R = np.zeros((n, n_actions, n_k), dtype=np.float32)
    for ki, k in enumerate(k_horizons):
        R[:, 0, ki] = df[f"r_skip_K{k}"].to_numpy(dtype=np.float32)
        R[:, 1, ki] = df[f"r_long_K{k}"].to_numpy(dtype=np.float32)
        R[:, 2, ki] = df[f"r_short_K{k}"].to_numpy(dtype=np.float32)
    return X, R


def warmstart_train(
    cement_ckpt: dict,
    X_new: np.ndarray,
    R_new: np.ndarray,
    *,
    lr: float,
    epochs_q: int,
    epochs_v: int,
    k_iterations: int,
    batch_size: int,
    seed: int,
    sample_weights: np.ndarray | None = None,
) -> dict:
    """Load cement weights into a fresh MultiHeadEntryIQLModel + continue training on (X_new, R_new).

    sample_weights: optional per-row loss weight aligned with X_new (anti-forgetting
    mix: live rows 1.0, cement-anchor rows --mix-weight). None = all-ones (unchanged).
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    state_dim = int(cement_ckpt["state_dim"])
    n_actions = int(cement_ckpt["n_actions"])
    k_horizons = list(cement_ckpt["k_horizons"])
    n_k = len(k_horizons)
    hidden_dim = int(cement_ckpt.get("hidden_dim", 128))
    n_hidden = int(cement_ckpt.get("n_hidden", 2))
    dropout = float(cement_ckpt.get("dropout", iql_core.DEFAULT_DROPOUT))

    if X_new.shape[1] != state_dim:
        raise RuntimeError(
            f"replay buffer state_dim={X_new.shape[1]} != cement state_dim={state_dim}. "
            f"buffer was built against a different bundle — check --base-bundle."
        )
    if R_new.shape != (X_new.shape[0], n_actions, n_k):
        raise RuntimeError(f"R_new shape {R_new.shape} mismatches cement ({n_actions}, {n_k})")

    q_net = iql_core.MLP(state_dim, n_actions * n_k, hidden_dim=hidden_dim,
                         n_hidden=n_hidden, dropout=dropout).to(device)
    v_net = iql_core.MLP(state_dim, n_k, hidden_dim=hidden_dim,
                         n_hidden=n_hidden, dropout=dropout).to(device)
    q_net.load_state_dict(cement_ckpt["q_state_dict"])
    v_net.load_state_dict(cement_ckpt["v_state_dict"])

    feature_means = np.asarray(cement_ckpt["feature_means"], dtype=np.float32)
    feature_stds = np.asarray(cement_ckpt["feature_stds"], dtype=np.float32)
    Z_CLAMP_SIGMA = 5.0
    X_norm = ((X_new - feature_means) / feature_stds).astype(np.float32)
    X_norm = np.clip(X_norm, -Z_CLAMP_SIGMA, Z_CLAMP_SIGMA)
    X_norm = np.where(np.isnan(X_norm), 0.0, X_norm)
    R_clean = np.where(np.isnan(R_new), 0.0, R_new).astype(np.float32)

    X_t = torch.as_tensor(X_norm, dtype=torch.float32, device=device)
    R_t = torch.as_tensor(R_clean, dtype=torch.float32, device=device)
    n = X_t.shape[0]
    R_flat = R_t.view(n, n_actions * n_k)
    # Per-row loss weight — already consumed by BOTH the Q-loss and the V
    # expectile loss below; the mix path just feeds non-uniform values in.
    if sample_weights is not None:
        if sample_weights.shape != (n,):
            raise RuntimeError(f"sample_weights shape {sample_weights.shape} != ({n},)")
        w_t = torch.as_tensor(np.asarray(sample_weights, dtype=np.float32), device=device)
    else:
        w_t = torch.ones(n, dtype=torch.float32, device=device)

    q_opt = torch.optim.Adam(q_net.parameters(), lr=lr, weight_decay=iql_core.DEFAULT_WEIGHT_DECAY)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=lr, weight_decay=iql_core.DEFAULT_WEIGHT_DECAY)
    g = torch.Generator(device="cpu").manual_seed(seed)

    LOG.info(f"warm-start training: n={n} rows, state_dim={state_dim}, "
             f"K={n_k} horizons, lr={lr}, epochs_q={epochs_q}, epochs_v={epochs_v}, k_iter={k_iterations}")

    # Q refit on new R (low LR, few epochs)
    q_net.train()
    for ep in range(epochs_q):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            q_pred = q_net(X_t[idx])
            loss = ((q_pred - R_flat[idx]).pow(2) * w_t[idx].unsqueeze(-1)).mean()
            q_opt.zero_grad()
            loss.backward()
            q_opt.step()

    # Alternating V/Q (truncated)
    for _ in range(k_iterations):
        q_net.eval()
        with torch.no_grad():
            q_full = q_net(X_t).view(n, n_actions, n_k)
            q_max_over_a = q_full.max(dim=1).values
        v_net.train()
        for _ in range(epochs_v):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                v_pred = v_net(X_t[idx])
                diff = q_max_over_a[idx] - v_pred
                loss = iql_core.expectile_loss(diff, iql_core.DEFAULT_TAU if hasattr(iql_core, "DEFAULT_TAU") else 0.8,
                                               sample_weights=w_t[idx])
                v_opt.zero_grad()
                loss.backward()
                v_opt.step()
        v_net.eval()
        q_net.train()
        for _ in range(epochs_q):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                q_pred = q_net(X_t[idx])
                loss = ((q_pred - R_flat[idx]).pow(2) * w_t[idx].unsqueeze(-1)).mean()
                q_opt.zero_grad()
                loss.backward()
                q_opt.step()

    q_net.eval()
    v_net.eval()
    return {
        "q_state_dict": q_net.state_dict(),
        "v_state_dict": v_net.state_dict(),
        "state_dim": state_dim,
        "n_actions": n_actions,
        "k_horizons": k_horizons,
        "hidden_dim": hidden_dim,
        "n_hidden": n_hidden,
        "dropout": dropout,
        "feature_means": feature_means.tolist(),
        "feature_stds": feature_stds.tolist(),
        "variant": cement_ckpt["variant"],
        "fold_id": cement_ckpt["fold_id"],
        "schema_v1": "MULTI_HEAD_ENTRY_IQL_V2_CHECKPOINT",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-bundle", type=Path, required=True,
                   help="Path to cement bundle (BUILD_ENTRY_IQL_V2_...)")
    p.add_argument("--replay", type=Path, required=True,
                   help="Replay buffer parquet from build_online_replay_buffer")
    # ANTI-FORGETTING mix (2026-06-12): the live buffer alone is hundreds of rows —
    # mixing a frozen cement-distribution sample anchors the Q-net while it adapts.
    p.add_argument("--mix-cement", type=Path, default=None,
                   help="Cement-distribution sample parquet (build_online_replay_buffer "
                        "--export-cement-sample) mixed in as anti-forgetting anchor")
    p.add_argument("--mix-weight", type=float, default=0.5,
                   help="Per-row loss weight for cement rows (live rows always 1.0)")
    p.add_argument("--variant", type=str, required=True,
                   help="Variant to warm-start (must match a cement checkpoint)")
    p.add_argument("--fold", type=str, default="FOLD_1", choices=["FOLD_1", "FOLD_2", "FOLD_3"])
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output bundle directory (will have summary_v1.json + trained_models_v1/)")
    p.add_argument("--lr", type=float, default=1e-5, help="Warm-start LR (cement used 3e-4)")
    p.add_argument("--epochs-q", type=int, default=2, help="Q-net epochs per phase (cement used 30+)")
    p.add_argument("--epochs-v", type=int, default=2)
    p.add_argument("--k-iterations", type=int, default=2, help="Alternating V/Q iterations")
    p.add_argument("--batch-size", type=int, default=4096,
                   help="IQL big-batch hard rule (vedtak 2026-06-10): tiny MLPs train "
                        "at >=4096, never 256/512 — clamped to buffer size if smaller.")
    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--vedtak", type=str, default=None,
                   help="Explicit user decision id. REQUIRED — fail-closed.")
    args = p.parse_args()

    try:
        require_retrain_vedtak(args.vedtak)
    except GateError as e:
        p.error(str(e))

    LOG.info(f"loading cement bundle: {args.base_bundle.name}")
    cement = load_cement(args.base_bundle, args.variant, args.fold)
    LOG.info(f"cement checkpoint: {cement['ckpt_path'].name}")

    LOG.info(f"loading replay buffer: {args.replay}")
    df = pd.read_parquet(args.replay)
    meta = json.loads(args.replay.with_suffix(".meta.json").read_text())
    if meta["variant"] != args.variant:
        raise RuntimeError(
            f"replay buffer was built for variant {meta['variant']!r} but warm-start "
            f"requested {args.variant!r}. Rebuild buffer with matching variant."
        )
    # 2026-06-12: k_horizons_m5 (cement M5-bar units) is REQUIRED. Metas with only
    # the legacy 'k_horizons' key carry the known 5x M1/M5 horizon bug — refuse.
    if "k_horizons_m5" not in meta:
        raise RuntimeError(
            "replay buffer meta lacks 'k_horizons_m5' — legacy buffer with the 5x "
            "M1/M5 horizon-unit bug. Rebuild it with build_online_replay_buffer."
        )
    k_horizons = meta["k_horizons_m5"]
    if list(k_horizons) != list(cement["ckpt"]["k_horizons"]):
        raise RuntimeError(
            f"buffer k_horizons {k_horizons} != cement ckpt k_horizons "
            f"{cement['ckpt']['k_horizons']} — rebuild buffer with --k-horizons matching the bundle."
        )
    # Reward-label env parity (2026-06-12): the refit MUST rebuild rewards under
    # the same flags the cement was trained with. Newer ckpts stamp reward_env_v1;
    # older ones (e.g. volbal) were fingerprint-verified posgate=1.
    cement_env = cement["ckpt"].get("reward_env_v1")
    buffer_env = meta.get("reward_env_v1") or {}
    if cement_env is not None:
        cpg = cement_env.get("GX1_REWARD_BADPATH_POSGATE", "0")
        bpg = str(buffer_env.get("GX1_REWARD_BADPATH_POSGATE", "?"))[:1]
        if cpg != bpg:
            raise RuntimeError(
                f"reward env mismatch: cement POSGATE={cpg} vs buffer POSGATE={bpg} — "
                f"rebuild the buffer with GX1_REWARD_BADPATH_POSGATE={cpg}."
            )
    else:
        LOG.warning("cement ckpt has no reward_env_v1 stamp (pre-2026-06-12 bundle) — "
                    "verify buffer posgate matches the bundle's training (volbal = posgate 1).")
    LOG.info(f"buffer rows: {len(df)}  variant: {meta['variant']}  k_horizons: {k_horizons}")
    if len(df) < 50:
        LOG.warning(f"only {len(df)} rows — warm-start needs ~50+ trades for stable update")

    X_new, R_new = buffer_to_tensors(df, k_horizons)
    LOG.info(f"X shape: {X_new.shape}  R shape: {R_new.shape}")

    # ANTI-FORGETTING mix: concat a frozen cement-distribution sample after the
    # live rows; per-row loss weight 1.0 (live) vs --mix-weight (cement) threads
    # into warmstart_train's existing w_t. All parity checks fail loud — a
    # mismatched sample would anchor the net to the WRONG distribution.
    sample_weights: np.ndarray | None = None
    n_live, n_cem = int(len(df)), 0
    if args.mix_cement is not None:
        # The cement anchor rows ARE the cement label definition — mixing them with
        # M1-grid live rewards trains one Q-net on two label conventions (measured
        # mean |Δ| 54 bps). Require the live buffer to be cement-grid, fail loud.
        live_conv = meta.get("label_convention")
        if live_conv != LABEL_CONVENTION_CEMENT_M5:
            raise RuntimeError(
                f"--mix-cement requires live buffer label_convention="
                f"{LABEL_CONVENTION_CEMENT_M5!r}, got {live_conv!r} — the live rewards "
                f"are not on the cement M5 label grid. Rebuild the buffer with "
                f"build_online_replay_buffer (re-anchored 2026-06-12)."
            )
        LOG.info(f"loading cement mix sample: {args.mix_cement}")
        df_cem = pd.read_parquet(args.mix_cement)
        cem_meta = json.loads(args.mix_cement.with_suffix(".meta.json").read_text())
        if cem_meta["variant"] != args.variant:
            raise RuntimeError(
                f"cement sample variant {cem_meta['variant']!r} != requested {args.variant!r} — "
                f"re-export with --export-cement-sample --variant {args.variant}."
            )
        if list(cem_meta["k_horizons_m5"]) != list(k_horizons):
            raise RuntimeError(
                f"cement sample k_horizons {cem_meta['k_horizons_m5']} != buffer/ckpt {k_horizons}."
            )
        cem_pg = str((cem_meta.get("reward_env_v1") or {}).get("GX1_REWARD_BADPATH_POSGATE", "?"))[:1]
        ref_pg = str((cement_env or buffer_env).get("GX1_REWARD_BADPATH_POSGATE", "?"))[:1]
        if cem_pg != ref_pg:
            raise RuntimeError(
                f"reward env mismatch: cement sample POSGATE={cem_pg} vs bundle/buffer POSGATE={ref_pg}."
            )
        # State-contract check: names AND ORDER vs the bundle summary — a
        # same-width reorder would pass the state_dim check silently.
        bundle_names = cement["summary"].get("feature_names_v1")
        sample_names = cem_meta.get("feature_names")
        if bundle_names is not None and sample_names is not None:
            if list(sample_names) != list(bundle_names):
                raise RuntimeError(
                    "cement sample feature_names do not match the bundle's "
                    "feature_names_v1 (names/order) — re-export against this bundle."
                )
        X_cem, R_cem = buffer_to_tensors(df_cem, k_horizons)
        if X_cem.shape[1] != X_new.shape[1]:
            raise RuntimeError(
                f"cement sample state_dim={X_cem.shape[1]} != live buffer state_dim={X_new.shape[1]}."
            )
        n_cem = int(X_cem.shape[0])
        X_new = np.concatenate([X_new, X_cem], axis=0)
        R_new = np.concatenate([R_new, R_cem], axis=0)
        sample_weights = np.concatenate([
            np.ones(n_live, dtype=np.float32),
            np.full(n_cem, float(args.mix_weight), dtype=np.float32),
        ])
        LOG.info(f"ANTI-FORGETTING MIX: {n_live} live rows (w=1.0) + {n_cem} cement rows "
                 f"(w={args.mix_weight}) = {n_live + n_cem} total — loss mass "
                 f"live={float(n_live):.1f} vs cement={n_cem * float(args.mix_weight):.1f}")

    new_ckpt = warmstart_train(
        cement["ckpt"], X_new, R_new,
        lr=args.lr, epochs_q=args.epochs_q, epochs_v=args.epochs_v,
        k_iterations=args.k_iterations, batch_size=args.batch_size, seed=args.seed,
        sample_weights=sample_weights,
    )

    # Persist in cement-compatible layout (stamp reward env for the NEXT refit)
    new_ckpt["reward_env_v1"] = cement_env or buffer_env
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "trained_models_v1").mkdir(exist_ok=True)
    ckpt_out = args.out_dir / "trained_models_v1" / f"{args.variant}_{args.fold}.pt"
    torch.save(new_ckpt, ckpt_out)
    LOG.info(f"wrote warm-started checkpoint → {ckpt_out}")

    # Re-emit summary with same feature_names (state_dim is unchanged)
    summary_out = dict(cement["summary"])
    summary_out["base_cement_bundle"] = str(args.base_bundle)
    summary_out["replay_buffer"] = str(args.replay)
    summary_out["warmstart_meta"] = {
        "vedtak": args.vedtak,
        "lr": args.lr,
        "epochs_q": args.epochs_q,
        "epochs_v": args.epochs_v,
        "k_iterations": args.k_iterations,
        "n_rows": int(len(df)),
        "n_journals": len(meta.get("journals", [])),
        "date_from": meta.get("date_from"),
        "date_to": meta.get("date_to"),
        # ANTI-FORGETTING mix composition (None/0 when --mix-cement unused)
        "mix_cement": str(args.mix_cement) if args.mix_cement else None,
        "mix_weight": float(args.mix_weight) if args.mix_cement else None,
        "n_live_rows": n_live,
        "n_cement_rows": n_cem,
    }
    (args.out_dir / "summary_v1.json").write_text(json.dumps(summary_out, indent=2, default=str))
    LOG.info(f"wrote summary → {args.out_dir / 'summary_v1.json'}")

    LOG.info("")
    LOG.info("=== NEXT STEPS (vedtak required, NOT automated) ===")
    LOG.info("1. Gate vs cement (parallel exit-chain replay + hardened posthoc eval):")
    LOG.info(f"   python -m gx1.scripts.v12.v12_phase6_joint_validation --gate ... (candidate {args.out_dir})")
    LOG.info(f"   python -m gx1.execution.v12_counterfactual_replay --mode stress ... (RR2 short-in-uptrend gate)")
    LOG.info("2. If gates PASS on the load-bearing metrics (bps/take, per-year floors, DD):")
    LOG.info(f"   manual vedtak → flip PROJECT_STATE_artifacts.json active.entry_iql.path = {args.out_dir}")
    LOG.info("3. bash scripts/stop_live_practice.sh && bash scripts/launch_live_practice.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
