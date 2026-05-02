"""Distributional Q-Learning GPU core (QR-DQN, Dabney et al. 2018).

Replaces the point Q(s,a) estimate with a vector of N quantiles z_j(s,a)
representing the cumulative reward distribution. Loss is the quantile Huber:

  L_QR(theta) = sum_j sum_k rho_(tau_j) ( target_k - z_j(s,a) )

where rho_tau(u) = |tau - 1[u<0]| * Huber_kappa(u) and tau_j = (j + 0.5) / N
are the quantile midpoints.

For IQL integration:
  - V(s) is fit via expectile regression on the *expected* Q under the
    learned quantile distribution.
  - Bellman target for each quantile: r + gamma * V(s') * (1-done).
    (Single-target broadcast, since V is scalar; in standard QR-DQN
    you would broadcast across all target quantiles.)
  - Policy: sigmoid(beta * (E[Q(s,EXIT)] - E[Q(s,HOLD)])) — expected-value
    advantage, identical to point IQL when N=1.

This is "distributional in Q-only" — V remains scalar and is not itself
distributional. Sufficient for richer Q-function representation; full
distributional V is left for a future gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gx1.scripts.true_iql_gpu_core_v1 import (
    ACTION_EXIT_NOW_ID,
    ACTION_HOLD_ID,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_INNER_EPOCHS,
    DEFAULT_K_ITERATIONS,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    MLP,
    N_ACTIONS,
    _batches,
    _device,
    _to_sa_tensor,
    expectile_loss,
)

DEFAULT_N_QUANTILES = 51
DEFAULT_HUBER_KAPPA = 1.0


def _quantile_midpoints(n: int, device: torch.device) -> torch.Tensor:
    return (torch.arange(n, dtype=torch.float32, device=device) + 0.5) / n


def quantile_huber_loss(
    z_pred: torch.Tensor, target: torch.Tensor, kappa: float = DEFAULT_HUBER_KAPPA
) -> torch.Tensor:
    """Quantile Huber loss for QR-DQN.

    z_pred: (n, N_q) predicted quantiles.
    target: (n,) scalar target (broadcast across all quantiles).
    """
    n_q = z_pred.shape[1]
    taus = _quantile_midpoints(n_q, z_pred.device)  # (N_q,)
    u = target.unsqueeze(1) - z_pred  # (n, N_q)
    huber = F.huber_loss(z_pred, target.unsqueeze(1).expand_as(z_pred),
                         delta=kappa, reduction="none")  # (n, N_q)
    weight = torch.abs(taus.unsqueeze(0) - (u < 0).float())  # (n, N_q)
    return (weight * huber).mean()


def _expected_q_for_action(
    q_quantile_net: MLP, X: torch.Tensor, action_id: int, n_actions: int = N_ACTIONS
) -> torch.Tensor:
    """E[Q(s, a)] = mean over predicted quantiles. Returns (n,)."""
    n = X.shape[0]
    a_oh = torch.zeros((n, n_actions), dtype=X.dtype, device=X.device)
    a_oh[:, action_id] = 1.0
    sa = torch.cat([X, a_oh], dim=1)
    z = q_quantile_net(sa)  # (n, N_q)
    return z.mean(dim=1)


@dataclass
class DistributionalIQLGpuModel:
    v_net: MLP
    q_quantile_net: MLP
    state_dim: int
    n_quantiles: int
    device: torch.device

    def predict_v(self, X_np: np.ndarray) -> np.ndarray:
        self.v_net.eval()
        with torch.no_grad():
            x = torch.as_tensor(X_np, dtype=torch.float32, device=self.device)
            return self.v_net(x).squeeze(-1).cpu().numpy()

    def predict_q_quantiles(self, X_np: np.ndarray, action_id: int) -> np.ndarray:
        """Returns (n, n_quantiles)."""
        self.q_quantile_net.eval()
        with torch.no_grad():
            n = X_np.shape[0]
            x = torch.as_tensor(X_np, dtype=torch.float32, device=self.device)
            a_oh = torch.zeros((n, N_ACTIONS), dtype=torch.float32, device=self.device)
            a_oh[:, action_id] = 1.0
            sa = torch.cat([x, a_oh], dim=1)
            return self.q_quantile_net(sa).cpu().numpy()

    def predict_expected_q(self, X_np: np.ndarray, action_id: int) -> np.ndarray:
        return self.predict_q_quantiles(X_np, action_id).mean(axis=1)


def train_distributional_iql_gpu(
    X_train_np: np.ndarray,
    a_train_np: np.ndarray,
    r_train_np: np.ndarray,
    next_idx_train_np: np.ndarray,
    done_train_np: np.ndarray,
    *,
    tau: float,
    n_quantiles: int = DEFAULT_N_QUANTILES,
    huber_kappa: float = DEFAULT_HUBER_KAPPA,
    gamma: float = 0.99,
    k_iterations: int = DEFAULT_K_ITERATIONS,
    inner_epochs: int = DEFAULT_INNER_EPOCHS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    n_hidden: int = DEFAULT_HIDDEN_LAYERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 20260430,
    prefer_cuda: bool = True,
) -> DistributionalIQLGpuModel:
    """Train V (expectile, scalar) + Q (distributional, N quantiles) on GPU."""
    torch.manual_seed(seed)
    device = _device(prefer_cuda)
    state_dim = X_train_np.shape[1]
    n = X_train_np.shape[0]

    X = torch.as_tensor(X_train_np, dtype=torch.float32, device=device)
    a = torch.as_tensor(a_train_np, dtype=torch.long, device=device)
    r = torch.as_tensor(r_train_np, dtype=torch.float32, device=device)
    done = torch.as_tensor(done_train_np, dtype=torch.float32, device=device)
    next_idx = torch.as_tensor(next_idx_train_np, dtype=torch.long, device=device)
    valid_next = next_idx >= 0

    sa = _to_sa_tensor(X, a)

    v_net = MLP(state_dim, 1, hidden_dim=hidden_dim, n_hidden=n_hidden).to(device)
    q_quantile_net = MLP(
        state_dim + N_ACTIONS, n_quantiles,
        hidden_dim=hidden_dim, n_hidden=n_hidden,
    ).to(device)

    v_opt = torch.optim.Adam(v_net.parameters(), lr=lr, weight_decay=weight_decay)
    q_opt = torch.optim.Adam(q_quantile_net.parameters(), lr=lr, weight_decay=weight_decay)

    # Q warm-start: regress all quantiles toward r (degenerate distribution).
    for _ in range(inner_epochs):
        for batch in _batches(n, batch_size, seed):
            z = q_quantile_net(sa[batch])
            loss = quantile_huber_loss(z, r[batch], kappa=huber_kappa)
            q_opt.zero_grad()
            loss.backward()
            q_opt.step()

    for _ in range(k_iterations):
        # ---- Step 1: V via expectile regression on E[Q(s,a)] (Q frozen).
        with torch.no_grad():
            q_target_full = q_quantile_net(sa).mean(dim=1)
        for _ in range(inner_epochs):
            for batch in _batches(n, batch_size, seed):
                v_pred = v_net(X[batch]).squeeze(-1)
                diff = q_target_full[batch] - v_pred
                loss = expectile_loss(diff, tau)
                v_opt.zero_grad()
                loss.backward()
                v_opt.step()

        # ---- Step 2: Q quantile distribution via Bellman backup.
        with torch.no_grad():
            v_full = v_net(X).squeeze(-1)
            v_next = torch.zeros(n, dtype=torch.float32, device=device)
            v_next[valid_next] = v_full[next_idx[valid_next]]
            target_full = r + gamma * v_next * (1.0 - done)
        for _ in range(inner_epochs):
            for batch in _batches(n, batch_size, seed):
                z = q_quantile_net(sa[batch])
                loss = quantile_huber_loss(z, target_full[batch], kappa=huber_kappa)
                q_opt.zero_grad()
                loss.backward()
                q_opt.step()

    return DistributionalIQLGpuModel(
        v_net=v_net,
        q_quantile_net=q_quantile_net,
        state_dim=state_dim,
        n_quantiles=n_quantiles,
        device=device,
    )


def distributional_iql_policy_exit_prob_gpu(
    X_np: np.ndarray,
    model: DistributionalIQLGpuModel,
    *,
    beta: float,
    clip: float = 5.0,
) -> np.ndarray:
    """π(EXIT|s) = sigmoid(β · clip(E[Q(s,EXIT)] - E[Q(s,HOLD)], ±clip))."""
    q_exit = model.predict_expected_q(X_np, action_id=ACTION_EXIT_NOW_ID)
    q_hold = model.predict_expected_q(X_np, action_id=ACTION_HOLD_ID)
    adv = np.clip(q_exit - q_hold, -clip, clip)
    return 1.0 / (1.0 + np.exp(-beta * adv))


def info(model: DistributionalIQLGpuModel) -> dict[str, Any]:
    return {
        "device": str(model.device),
        "state_dim_v1": int(model.state_dim),
        "n_quantiles_v1": int(model.n_quantiles),
        "v_param_count_v1": sum(p.numel() for p in model.v_net.parameters()),
        "q_param_count_v1": sum(p.numel() for p in model.q_quantile_net.parameters()),
    }
