"""Multi-head entry-IQL GPU core for fully-observable-counterfactual contextual bandit.

Mission
-------
Entry-IQL trains on candidates where forward bars give us the OBSERVED reward
for EVERY action (not just the one that was taken). The 5 actions are:

  SKIP, TAKE_LONG_NOW, TAKE_SHORT_NOW, WAIT_LONG, WAIT_SHORT

For each candidate state s, we observe a reward matrix R[a, K] of shape
(n_actions, n_K) — one reward per (action, K-horizon) cell. K_HORIZONS
encode different holding durations: sniper (K=12) through swing (K=192).

This is RICHER than standard IQL bandit setting (where only one r is observed
per row). With full counterfactuals, Q(s, a, K) can be trained as a multi-task
regression across all (a, K) cells from each sample.

Algorithm
---------
Q-net:  state → (n_actions, n_K)  output. Loss: MSE on observed R[a, K].
V-net:  state → (n_K,)  output. V(s, K) = expectile_tau of max_a Q(s, a, K).
Policy: argmax over a of aggregator_K(Q(s, a, K)) where aggregator is one of:
        mean, max, weighted_sum.

Architecture
------------
Shared MLP backbone → linear head producing (n_actions × n_K) outputs,
reshaped to (n_actions, n_K) at inference. State features are z-score
normalized by the trainer (not this module).

This module is algorithm-agnostic: the trainer feeds state, reward matrix,
and hyperparameters; the module returns a trained model with predict()
that accepts a state matrix and returns Q-values per (action, K).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


# --- Action contract (3-action: SKIP semantics = "re-evaluate next M5 bar") ---
# Updated 2026-05-02: removed WAIT_LONG/WAIT_SHORT. SKIP now means "this M5 bar
# is not good enough — re-evaluate at next M5 bar's candidate". The runtime
# calls the entry-IQL adapter again on every fresh V10/Skip-V2 candidate, so
# explicit delayed-entry actions are unnecessary.

ACTION_SKIP_ID = 0
ACTION_TAKE_LONG_NOW_ID = 1
ACTION_TAKE_SHORT_NOW_ID = 2
N_ACTIONS_V1 = 3
ACTION_LABELS_V1 = {
    ACTION_SKIP_ID: "SKIP",
    ACTION_TAKE_LONG_NOW_ID: "TAKE_LONG_NOW",
    ACTION_TAKE_SHORT_NOW_ID: "TAKE_SHORT_NOW",
}

DEFAULT_HIDDEN_DIM = 256
DEFAULT_N_HIDDEN = 3
DEFAULT_DROPOUT = 0.1
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_EPOCHS_Q = 80
DEFAULT_EPOCHS_V = 30
DEFAULT_BATCH_SIZE = 256
DEFAULT_K_VQ_ITERATIONS = 6


def _device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MLP(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        n_hidden: int = DEFAULT_N_HIDDEN,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def expectile_loss(
    diff: torch.Tensor, tau: float, sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    weight = torch.where(diff >= 0, tau, 1.0 - tau)
    if sample_weights is not None:
        weight = weight * sample_weights.unsqueeze(-1)  # broadcast across K
    return (weight * diff.pow(2)).mean()


def q_ranking_margin_loss(
    q_flat: torch.Tensor,       # (B, n_actions*nk) predicted Q
    r_flat: torch.Tensor,       # (B, n_actions*nk) target rewards (defines per-(b,K) best action)
    n_actions: int,
    nk: int,
    margin: float,
    sample_w: torch.Tensor,     # (B,)
) -> torch.Tensor:
    """EIQL-2: per-(state, K) margin-ranking regularizer for the Q-net.

    Pure MSE pins Q to the reward scale (bps) but is indifferent to whether the BEST
    action is cleanly separated from the rest — when two actions have near-equal reward,
    noise can flip the argmax that the live policy uses. This adds a small hinge that asks
    Q[best] to exceed Q[other] by `margin` bps for every other action, per K. It is ADDITIVE
    and small-weight so Q stays interpretable in bps for the cemented Phase-6 gates. The
    per-(b,K) "best" action is taken from the REWARD target (not the prediction), so it never
    chases the model's own mistakes.
    """
    B = q_flat.shape[0]
    q = q_flat.view(B, n_actions, nk)
    r = r_flat.view(B, n_actions, nk)
    a_star = r.argmax(dim=1, keepdim=True)              # (B,1,nk) best action by reward
    q_star = torch.gather(q, 1, a_star)                 # (B,1,nk)
    hinge = torch.relu(margin - (q_star - q))           # (B,n_actions,nk); a* slot -> relu(margin)
    mask = torch.ones_like(hinge)
    mask.scatter_(1, a_star, 0.0)                       # zero out the a* slot (gap is 0 there)
    per_bk = (hinge * mask).sum(dim=1) / float(max(1, n_actions - 1))   # (B, nk)
    return (sample_w.unsqueeze(-1) * per_bk).mean()


@dataclass
class MultiHeadEntryIQLModel:
    """Trained model. q_net outputs (n, n_actions * n_K) which is reshaped."""
    q_net: MLP
    v_net: MLP
    state_dim: int
    n_actions: int
    n_k: int
    k_horizons: list[int]
    device: torch.device
    feature_means: np.ndarray  # (state_dim,) — z-score normalization
    feature_stds: np.ndarray   # (state_dim,)
    z_clamp: float = 5.0       # ±sigma clamp at normalization (inference + training)

    def _normalize(self, X_np: np.ndarray) -> np.ndarray:
        # Clamp z-scores to ±z_clamp sigma. Training also clamps so model never
        # learns to extrapolate outside this range. Audit C2/C3 follow-up 2026-05-20:
        # live values outside p1-p99 of training distribution were producing
        # extreme z-scores (50-100σ) and Q-network exploded to +1184 bps adv
        # (5× training reward max).
        z = (X_np - self.feature_means) / self.feature_stds
        z = np.clip(z, -self.z_clamp, self.z_clamp)
        # NaN-fill after clip (audit 3 L-1 fix 2026-05-20): any NaN from divide-
        # by-zero std or upstream missing feature would cascade through the Q
        # network. Defensive replace with 0.0 (mean of normalized distribution).
        z = np.where(np.isnan(z), 0.0, z)
        return z.astype(np.float32)

    def predict_q(self, X_np: np.ndarray) -> np.ndarray:
        """Return (n, n_actions, n_K) Q-values."""
        self.q_net.eval()
        Xn = self._normalize(X_np)
        with torch.no_grad():
            x = torch.as_tensor(Xn, dtype=torch.float32, device=self.device)
            q = self.q_net(x)
            q = q.view(-1, self.n_actions, self.n_k)
            return q.cpu().numpy()

    def predict_v(self, X_np: np.ndarray) -> np.ndarray:
        """Return (n, n_K) state values."""
        self.v_net.eval()
        Xn = self._normalize(X_np)
        with torch.no_grad():
            x = torch.as_tensor(Xn, dtype=torch.float32, device=self.device)
            return self.v_net(x).cpu().numpy()


def train_multi_head_entry_iql(
    X_np: np.ndarray,             # (n, state_dim)
    R_np: np.ndarray,              # (n, n_actions, n_K) observed rewards (counterfactual)
    *,
    k_horizons: Sequence[int],
    n_actions: int = N_ACTIONS_V1,
    tau: float = 0.8,
    sample_weights: np.ndarray | None = None,
    k_iterations: int = DEFAULT_K_VQ_ITERATIONS,
    epochs_q: int = DEFAULT_EPOCHS_Q,
    epochs_v: int = DEFAULT_EPOCHS_V,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    n_hidden: int = DEFAULT_N_HIDDEN,
    dropout: float = DEFAULT_DROPOUT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 20260501,
    prefer_cuda: bool = True,
    q_ranking_weight: float = 0.0,    # EIQL-2: 0.0 = bit-parity (pure MSE); small (~0.05) in retrain
    q_ranking_margin: float = 5.0,    # margin in REWARD units (bps); only active when weight>0
    loss_mask: np.ndarray | None = None,  # EXIT-9: (n, n_actions*n_K) 0/1 mask; None = bit-parity
    init_q_state_dict: dict | None = None,  # WARM-START: load cement Q weights as init; None = cold (bit-parity)
    init_v_state_dict: dict | None = None,  # WARM-START: load cement V weights as init; None = cold (bit-parity)
) -> MultiHeadEntryIQLModel:
    """Train multi-head Q(s, a, K) on fully-observable counterfactual rewards.

    Q-net learns to predict R[s, a, K] for ALL actions and horizons via MSE.
    V-net learns expectile_tau of (max over a of Q(s, a, K)) per K.
    """
    if R_np.ndim != 3:
        raise ValueError(f"R_np must be 3-D (n, n_actions, n_K); got shape {R_np.shape}")
    n, na, nk = R_np.shape
    if na != n_actions:
        raise ValueError(f"R_np action dim {na} != n_actions {n_actions}")
    if nk != len(k_horizons):
        raise ValueError(f"R_np K dim {nk} != len(k_horizons) {len(k_horizons)}")

    torch.manual_seed(seed)
    device = _device(prefer_cuda)
    state_dim = X_np.shape[1]

    # Normalize state features (z-score on train data) — WINSORIZED.
    # Audit C2/C3 2026-05-19: a few rows with 10^6-magnitude outliers were
    # poisoning the mean/std (e.g. v10_mfe_pred_at_entry_v1 raw mean=15,617,
    # std=446k — features were dead post-normalization). Clip per-column to
    # the 1st/99th percentile before computing stats.
    X_for_stats = X_np.copy()
    lo = np.nanpercentile(X_for_stats, 1.0, axis=0)
    hi = np.nanpercentile(X_for_stats, 99.0, axis=0)
    X_clipped = np.clip(X_for_stats, lo, hi)
    feature_means = np.nanmean(X_clipped, axis=0).astype(np.float32)
    feature_stds = np.nanstd(X_clipped, axis=0).astype(np.float32) + 1e-6
    # Use UNCLIPPED inputs for training (model sees real distribution after
    # normalization with robust stats — outliers normalize to plausible z-scores).
    Z_CLAMP_SIGMA = 5.0   # match adapter clamp; training+inference must agree
    X_norm = ((X_np - feature_means) / feature_stds).astype(np.float32)
    X_norm = np.clip(X_norm, -Z_CLAMP_SIGMA, Z_CLAMP_SIGMA)
    # NaN-fill after normalization (defensive)
    X_norm = np.where(np.isnan(X_norm), 0.0, X_norm)
    # Sanity log: how many features have suspiciously huge raw mean/std post-clip
    huge = int(np.sum(np.abs(feature_means) > 1e4) + np.sum(feature_stds > 1e4))
    if huge > 0:
        print(f"[winsorize] WARN: {huge} feature stats still >1e4 after p1-p99 clip")

    # Replace NaN rewards with 0 (typically only in degenerate forward windows)
    R_clean = np.where(np.isnan(R_np), 0.0, R_np).astype(np.float32)

    X_t = torch.as_tensor(X_norm, dtype=torch.float32, device=device)
    R_t = torch.as_tensor(R_clean, dtype=torch.float32, device=device)
    R_flat = R_t.view(n, n_actions * nk)
    if sample_weights is not None:
        w_t = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)
    else:
        w_t = torch.ones(n, dtype=torch.float32, device=device)
    # EXIT-9: optional per-(row, action*K) 0/1 mask. When provided, the Q-MSE is a masked mean
    # (degenerate forced_terminal / K_eff-clipped HOLD cells contribute no gradient). None keeps
    # the exact `.mean()` -> bit-parity for entry + the cemented exit.
    M_t = None
    if loss_mask is not None:
        M_t = torch.as_tensor(np.asarray(loss_mask, dtype=np.float32), dtype=torch.float32, device=device)
        if tuple(M_t.shape) != (n, n_actions * nk):
            raise ValueError(f"loss_mask shape {tuple(M_t.shape)} != (n, n_actions*nk)=({n},{n_actions*nk})")

    q_net = MLP(state_dim, n_actions * nk, hidden_dim=hidden_dim, n_hidden=n_hidden, dropout=dropout).to(device)
    v_net = MLP(state_dim, nk, hidden_dim=hidden_dim, n_hidden=n_hidden, dropout=dropout).to(device)
    # WARM-START (2026-06-15): when an init state-dict is supplied, load the cement Q/V weights as the
    # starting point INSTEAD of random init, so the refit nudges the converged policy rather than
    # re-learning from scratch (mirrors online_iql_warmstart.warmstart_train, ONE-TRUTH trainer). Loaded
    # with strict=True so an arch mismatch (hidden/state_dim) fails LOUD. Default None = cold = bit-parity.
    if init_q_state_dict is not None:
        _miss_q = q_net.load_state_dict(init_q_state_dict, strict=True)
        print(f"[iql-warmstart] loaded init Q-net weights (missing={_miss_q.missing_keys}, "
              f"unexpected={_miss_q.unexpected_keys})", flush=True)
    if init_v_state_dict is not None:
        _miss_v = v_net.load_state_dict(init_v_state_dict, strict=True)
        print(f"[iql-warmstart] loaded init V-net weights (missing={_miss_v.missing_keys}, "
              f"unexpected={_miss_v.unexpected_keys})", flush=True)
    q_opt = torch.optim.Adam(q_net.parameters(), lr=lr, weight_decay=weight_decay)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=lr, weight_decay=weight_decay)

    g = torch.Generator(device="cpu").manual_seed(seed)

    def _q_mse_loss(diff: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # EXIT-9: masked-mean MSE when a loss_mask is supplied; else the exact `.mean()`.
        wd = w_t[idx].unsqueeze(-1)
        if M_t is None:
            return (wd * diff.pow(2)).mean()
        m = M_t[idx]
        return (wd * m * diff.pow(2)).sum() / (wd * m).sum().clamp_min(1.0)

    # Phase 1: Q warmup — multi-head MSE on all observed rewards
    q_net.train()
    for _ in range(epochs_q):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            q_pred = q_net(X_t[idx])
            diff = q_pred - R_flat[idx]
            loss = _q_mse_loss(diff, idx)
            if q_ranking_weight > 0.0:
                loss = loss + q_ranking_weight * q_ranking_margin_loss(
                    q_pred, R_flat[idx], n_actions, nk, q_ranking_margin, w_t[idx]
                )
            q_opt.zero_grad()
            loss.backward()
            q_opt.step()

    # Phase 2: alternating V/Q
    for _ in range(k_iterations):
        # V update: V(s, K) ← expectile_tau of max_a Q(s, a, K)
        q_net.eval()
        with torch.no_grad():
            q_full = q_net(X_t).view(n, n_actions, nk)
            q_max_over_a = q_full.max(dim=1).values  # (n, nk)
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

        # Q refit: regression on observed R
        v_net.eval()
        q_net.train()
        for _ in range(epochs_q):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                q_pred = q_net(X_t[idx])
                diff = q_pred - R_flat[idx]
                loss = _q_mse_loss(diff, idx)
                if q_ranking_weight > 0.0:
                    loss = loss + q_ranking_weight * q_ranking_margin_loss(
                        q_pred, R_flat[idx], n_actions, nk, q_ranking_margin, w_t[idx]
                    )
                q_opt.zero_grad()
                loss.backward()
                q_opt.step()

    q_net.eval()
    v_net.eval()
    return MultiHeadEntryIQLModel(
        q_net=q_net, v_net=v_net,
        state_dim=state_dim, n_actions=n_actions, n_k=nk, k_horizons=list(k_horizons),
        device=device,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )


def policy_argmax_action(
    X_np: np.ndarray,
    model: MultiHeadEntryIQLModel,
    *,
    aggregator: str = "mean",
    k_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy action: argmax over a of aggregator_K(Q(s, a, K)).

    aggregator: "mean" | "max" | "weighted" (requires k_weights of shape (n_K,)).
    Returns action ids of shape (n,).
    """
    q = model.predict_q(X_np)  # (n, n_actions, n_K)
    if aggregator == "mean":
        score = q.mean(axis=2)
    elif aggregator == "max":
        score = q.max(axis=2)
    elif aggregator == "weighted":
        if k_weights is None or k_weights.shape != (model.n_k,):
            raise ValueError("weighted aggregator requires k_weights of shape (n_K,)")
        score = (q * k_weights[None, None, :]).sum(axis=2)
    else:
        raise ValueError(f"unknown aggregator: {aggregator}")
    return score.argmax(axis=1)


def policy_action_probs(
    X_np: np.ndarray,
    model: MultiHeadEntryIQLModel,
    *,
    beta: float,
    aggregator: str = "mean",
    k_weights: np.ndarray | None = None,
) -> np.ndarray:
    """π(a | s) = softmax(β · score(s, a)) over n_actions."""
    q = model.predict_q(X_np)
    if aggregator == "mean":
        score = q.mean(axis=2)
    elif aggregator == "max":
        score = q.max(axis=2)
    elif aggregator == "weighted":
        score = (q * k_weights[None, None, :]).sum(axis=2)
    else:
        raise ValueError(f"unknown aggregator: {aggregator}")
    scaled = beta * score
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def info(model: MultiHeadEntryIQLModel) -> dict[str, Any]:
    return {
        "device_v1": str(model.device),
        "state_dim_v1": int(model.state_dim),
        "n_actions_v1": int(model.n_actions),
        "n_k_v1": int(model.n_k),
        "k_horizons_v1": list(model.k_horizons),
        "q_param_count_v1": sum(p.numel() for p in model.q_net.parameters()),
        "v_param_count_v1": sum(p.numel() for p in model.v_net.parameters()),
        "action_labels_v1": ACTION_LABELS_V1,
    }
