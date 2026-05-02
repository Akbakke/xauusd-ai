"""GPU True IQL core: NN-based V/Q with expectile regression.

API parallel to materialize_build_true_implicit_q_learning_v1's ridge-based
train_true_iql / true_iql_policy_exit_prob, but implemented with PyTorch MLPs
trained on CUDA. Same expectile loss for V, Bellman MSE for Q, sigmoid-of-
advantage policy. Designed so the existing per-fold evaluation loop can
substitute the GPU trainer with a one-line swap once integrated.

The module is a pure utility — it does not produce any artifacts on its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

ACTION_HOLD_ID = 0
ACTION_EXIT_NOW_ID = 1
N_ACTIONS = 2

DEFAULT_HIDDEN_DIM = 64
DEFAULT_HIDDEN_LAYERS = 2
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_INNER_EPOCHS = 50
DEFAULT_K_ITERATIONS = 10
DEFAULT_BATCH_SIZE = 256


def _device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        n_hidden: int = DEFAULT_HIDDEN_LAYERS,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """L_tau(u) = |tau - 1[u<0]| * u^2"""
    weight = torch.where(diff >= 0, tau, 1.0 - tau)
    return (weight * diff.pow(2)).mean()


@dataclass
class IQLGpuModel:
    v_net: MLP
    q_net: MLP
    state_dim: int
    device: torch.device

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


def _to_sa_tensor(
    X: torch.Tensor, a: torch.Tensor, n_actions: int = N_ACTIONS
) -> torch.Tensor:
    a_oh = torch.zeros((X.shape[0], n_actions), dtype=X.dtype, device=X.device)
    a_oh.scatter_(1, a.view(-1, 1), 1.0)
    return torch.cat([X, a_oh], dim=1)


def train_true_iql_gpu(
    X_train_np: np.ndarray,
    a_train_np: np.ndarray,
    r_train_np: np.ndarray,
    next_idx_train_np: np.ndarray,
    done_train_np: np.ndarray,
    *,
    tau: float,
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
) -> IQLGpuModel:
    """Alternating expectile-V / Bellman-Q training on GPU.

    Same outer K-loop as the ridge version:
      1. V_psi = argmin E[L_tau(Q(s,a) - V(s))]
      2. Q_theta = argmin E[(r + gamma * V(s') * (1 - done) - Q(s,a))^2]
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

    # Warm-start Q with regression on r, ignoring Bellman.
    for _ in range(inner_epochs):
        for batch in _batches(n, batch_size, seed):
            q_pred = q_net(sa[batch]).squeeze(-1)
            loss = (q_pred - r[batch]).pow(2).mean()
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

        # ---- Step 2: Q via Bellman backup using V(s') (V frozen).
        with torch.no_grad():
            v_full = v_net(X).squeeze(-1)
            v_next = torch.zeros(n, dtype=torch.float32, device=device)
            v_next[valid_next] = v_full[next_idx[valid_next]]
            target_full = r + gamma * v_next * (1.0 - done)
        for _ in range(inner_epochs):
            for batch in _batches(n, batch_size, seed):
                q_pred = q_net(sa[batch]).squeeze(-1)
                loss = (q_pred - target_full[batch]).pow(2).mean()
                q_opt.zero_grad()
                loss.backward()
                q_opt.step()

    return IQLGpuModel(v_net=v_net, q_net=q_net, state_dim=state_dim, device=device)


def _batches(n: int, batch_size: int, seed: int) -> list[torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    return [perm[i : i + batch_size] for i in range(0, n, batch_size)]


def true_iql_policy_exit_prob_gpu(
    X_np: np.ndarray,
    model: IQLGpuModel,
    *,
    beta: float,
    clip: float = 5.0,
) -> np.ndarray:
    """π(EXIT_NOW | s) = sigmoid(β · clip(Q(s,EXIT) - Q(s,HOLD), ±clip))."""
    q_exit = model.predict_q(X_np, action_id=ACTION_EXIT_NOW_ID)
    q_hold = model.predict_q(X_np, action_id=ACTION_HOLD_ID)
    adv = np.clip(q_exit - q_hold, -clip, clip)
    return 1.0 / (1.0 + np.exp(-beta * adv))


def info(model: IQLGpuModel) -> dict[str, Any]:
    return {
        "device": str(model.device),
        "state_dim_v1": int(model.state_dim),
        "v_param_count_v1": sum(p.numel() for p in model.v_net.parameters()),
        "q_param_count_v1": sum(p.numel() for p in model.q_net.parameters()),
    }
