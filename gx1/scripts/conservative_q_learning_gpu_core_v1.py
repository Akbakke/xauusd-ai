"""Conservative Q-Learning GPU core (Kumar et al. 2020).

CQL extends IQL by adding an explicit out-of-distribution-action regularizer:

  L_total(Q) = L_Bellman(Q) + alpha * E_s [ logsumexp_a Q(s,a) - E_(s,a)~D Q(s,a) ]

The first term inside the brackets pulls *all* action Q-values down (via
log-sum-exp); the second term pulls dataset-action Q-values up. Net effect:
Q-values for actions NOT seen in the dataset are penalized, providing
conservative behavior on OOD inputs that pure IQL only handles implicitly.

This module imports the IQL GPU core for V network, expectile loss, and
shared helpers, and adds:
  - cql_regularizer: the OOD penalty term
  - train_iql_cql_gpu: training loop that mixes IQL Bellman + CQL term
  - true_iql_cql_policy_exit_prob_gpu: same policy as IQL (sigmoid of advantage)

Same 2-action HOLD/EXIT_NOW setup as IQL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

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

CQL_ALPHA_GRID: list[float] = [0.1, 0.5, 1.0]


def cql_regularizer(
    q_all_actions: torch.Tensor, q_data_action: torch.Tensor
) -> torch.Tensor:
    """L_CQL = E_s [logsumexp_a Q(s,a) - E_(s,a)~D Q(s,a)].

    q_all_actions: (n, n_actions) Q for every action at every state.
    q_data_action: (n,) Q at the action that was actually taken in data.
    """
    log_sum_exp = torch.logsumexp(q_all_actions, dim=1)
    return (log_sum_exp - q_data_action).mean()


def _q_for_all_actions(
    q_net: nn.Module, X: torch.Tensor, n_actions: int = N_ACTIONS
) -> torch.Tensor:
    """Returns (n, n_actions) Q-values."""
    n = X.shape[0]
    out = []
    for a_id in range(n_actions):
        a = torch.full((n, 1), a_id, dtype=torch.long, device=X.device)
        a_oh = torch.zeros((n, n_actions), dtype=X.dtype, device=X.device)
        a_oh.scatter_(1, a, 1.0)
        sa = torch.cat([X, a_oh], dim=1)
        out.append(q_net(sa).squeeze(-1))
    return torch.stack(out, dim=1)


@dataclass
class CQLGpuModel:
    v_net: MLP
    q_net: MLP
    state_dim: int
    device: torch.device
    cql_alpha: float

    def predict_v(self, X_np: np.ndarray) -> np.ndarray:
        self.v_net.eval()
        with torch.no_grad():
            x = torch.as_tensor(X_np, dtype=torch.float32, device=self.device)
            return self.v_net(x).squeeze(-1).cpu().numpy()

    def predict_q(self, X_np: np.ndarray, action_id: int) -> np.ndarray:
        self.q_net.eval()
        with torch.no_grad():
            n = X_np.shape[0]
            x = torch.as_tensor(X_np, dtype=torch.float32, device=self.device)
            a_oh = torch.zeros((n, N_ACTIONS), dtype=torch.float32, device=self.device)
            a_oh[:, action_id] = 1.0
            sa = torch.cat([x, a_oh], dim=1)
            return self.q_net(sa).squeeze(-1).cpu().numpy()


def train_iql_cql_gpu(
    X_train_np: np.ndarray,
    a_train_np: np.ndarray,
    r_train_np: np.ndarray,
    next_idx_train_np: np.ndarray,
    done_train_np: np.ndarray,
    *,
    tau: float,
    cql_alpha: float,
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
) -> CQLGpuModel:
    """Alternating expectile-V / Bellman-Q-with-CQL training on GPU.

    Outer K-loop:
      1. V_psi = argmin E[L_tau(Q(s,a) - V(s))]
      2. Q_theta = argmin E[(r + gamma*V(s')*(1-done) - Q(s,a))^2]
                          + alpha * E[logsumexp_a Q(s,a) - Q(s,a_data)]
    """
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
    q_net = MLP(state_dim + N_ACTIONS, 1, hidden_dim=hidden_dim, n_hidden=n_hidden).to(device)
    v_opt = torch.optim.Adam(v_net.parameters(), lr=lr, weight_decay=weight_decay)
    q_opt = torch.optim.Adam(q_net.parameters(), lr=lr, weight_decay=weight_decay)

    # Q warm-start with regression on r + CQL term (no Bellman yet).
    for _ in range(inner_epochs):
        for batch in _batches(n, batch_size, seed):
            q_data = q_net(sa[batch]).squeeze(-1)
            mse = (q_data - r[batch]).pow(2).mean()
            q_all = _q_for_all_actions(q_net, X[batch])
            cql = cql_regularizer(q_all, q_data)
            loss = mse + cql_alpha * cql
            q_opt.zero_grad()
            loss.backward()
            q_opt.step()

    for _ in range(k_iterations):
        # ---- Step 1: V via expectile regression on Q(s,a) (Q frozen).
        with torch.no_grad():
            q_target_full = q_net(sa).squeeze(-1)
        for _ in range(inner_epochs):
            for batch in _batches(n, batch_size, seed):
                v_pred = v_net(X[batch]).squeeze(-1)
                diff = q_target_full[batch] - v_pred
                loss = expectile_loss(diff, tau)
                v_opt.zero_grad()
                loss.backward()
                v_opt.step()

        # ---- Step 2: Q via Bellman backup + CQL regularizer.
        with torch.no_grad():
            v_full = v_net(X).squeeze(-1)
            v_next = torch.zeros(n, dtype=torch.float32, device=device)
            v_next[valid_next] = v_full[next_idx[valid_next]]
            target_full = r + gamma * v_next * (1.0 - done)
        for _ in range(inner_epochs):
            for batch in _batches(n, batch_size, seed):
                q_data = q_net(sa[batch]).squeeze(-1)
                mse = (q_data - target_full[batch]).pow(2).mean()
                q_all = _q_for_all_actions(q_net, X[batch])
                cql = cql_regularizer(q_all, q_data)
                loss = mse + cql_alpha * cql
                q_opt.zero_grad()
                loss.backward()
                q_opt.step()

    return CQLGpuModel(
        v_net=v_net, q_net=q_net, state_dim=state_dim,
        device=device, cql_alpha=cql_alpha,
    )


def true_iql_cql_policy_exit_prob_gpu(
    X_np: np.ndarray,
    model: CQLGpuModel,
    *,
    beta: float,
    clip: float = 5.0,
) -> np.ndarray:
    """π(EXIT|s) = sigmoid(β · clip(Q(s,EXIT) - Q(s,HOLD), ±clip)).

    Same policy form as IQL — CQL only changes Q learning, not policy
    extraction.
    """
    q_exit = model.predict_q(X_np, action_id=ACTION_EXIT_NOW_ID)
    q_hold = model.predict_q(X_np, action_id=ACTION_HOLD_ID)
    adv = np.clip(q_exit - q_hold, -clip, clip)
    return 1.0 / (1.0 + np.exp(-beta * adv))


def info(model: CQLGpuModel) -> dict[str, Any]:
    return {
        "device": str(model.device),
        "state_dim_v1": int(model.state_dim),
        "cql_alpha_v1": float(model.cql_alpha),
        "v_param_count_v1": sum(p.numel() for p in model.v_net.parameters()),
        "q_param_count_v1": sum(p.numel() for p in model.q_net.parameters()),
    }
