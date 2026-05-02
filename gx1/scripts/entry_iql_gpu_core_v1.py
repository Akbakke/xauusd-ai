"""Entry-IQL GPU core: 3-action contextual bandit with expectile V + softmax-Q policy.

Mission
-------
Decide at each TRIGGER moment (when XGB+V10 says "this is a candidate") whether
the right action is:

  - SKIP        (don't take this trade — it's bad quality)
  - TAKE_NOW    (take it immediately at trigger price)
  - WAIT        (better to wait for confirmation; take later)

Unlike exit-IQL which is a multi-step MDP (per-bar HOLD/EXIT decisions during
a held trade), entry is a **one-shot contextual bandit**. There are no state
transitions — at trigger time you pick one of three actions and you learn the
outcome.

Algorithm
---------
For each (state, action_taken, reward) triple:
  - Train Q(s, a) — one head per action — via expectile regression on reward
  - Train V(s) — single state value — via expectile regression on Q(s, a_taken)
  - Policy: π(a | s) ∝ exp(β · Q(s, a))  (softmax over 3 actions)

Direction-aware state
---------------------
The state vector includes BOTH long-side and short-side XGB signals
(p_long, p_short, p_flat, margin). The IQL learns to recognize direction
conflicts (e.g., p_short > p_long → SKIP a long candidate) without needing
its own direction-decision action — direction is set upstream by V10.

This is the ORDENTLIG production-quality IQL, not the reduced-budget toy.
Default hyperparameters are tuned for entry-quality rather than fast-iteration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ACTION_SKIP_ID = 0
ACTION_TAKE_NOW_ID = 1
ACTION_WAIT_ID = 2
N_ACTIONS = 3
ACTION_LABELS = {ACTION_SKIP_ID: "SKIP", ACTION_TAKE_NOW_ID: "TAKE_NOW", ACTION_WAIT_ID: "WAIT"}

# Production-quality defaults — NOT reduced-budget.
DEFAULT_HIDDEN_DIM = 128
DEFAULT_N_HIDDEN = 2
DEFAULT_DROPOUT = 0.1
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 128
DEFAULT_K_VQ_ITERATIONS = 10  # alternate V and Q this many times


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
    diff: torch.Tensor, tau: float, sample_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """L_tau(u) = |tau - 1[u<0]| * u^2, optionally per-sample weighted."""
    weight = torch.where(diff >= 0, tau, 1.0 - tau)
    if sample_weights is not None:
        weight = weight * sample_weights
    return (weight * diff.pow(2)).mean()


@dataclass
class EntryIQLGpuModel:
    """Container with V-net (state value) and Q-net (per-action value).

    Q-net outputs (n_actions,) tensor — Q(s, a) for all actions in one forward.
    """
    v_net: MLP
    q_net: MLP
    state_dim: int
    n_actions: int
    device: torch.device

    def predict_v(self, X_np: np.ndarray) -> np.ndarray:
        self.v_net.eval()
        with torch.no_grad():
            x = torch.as_tensor(X_np, dtype=torch.float32, device=self.device)
            return self.v_net(x).squeeze(-1).cpu().numpy()

    def predict_q(self, X_np: np.ndarray) -> np.ndarray:
        """Returns (n, n_actions) Q-values for all actions."""
        self.q_net.eval()
        with torch.no_grad():
            x = torch.as_tensor(X_np, dtype=torch.float32, device=self.device)
            return self.q_net(x).cpu().numpy()


def train_entry_iql_gpu(
    X_train_np: np.ndarray,
    a_train_np: np.ndarray,
    r_train_np: np.ndarray,
    *,
    tau: float,
    sample_weights: np.ndarray | None = None,
    n_actions: int = N_ACTIONS,
    k_iterations: int = DEFAULT_K_VQ_ITERATIONS,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    n_hidden: int = DEFAULT_N_HIDDEN,
    dropout: float = DEFAULT_DROPOUT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 20260430,
    prefer_cuda: bool = True,
) -> EntryIQLGpuModel:
    """Train V/Q for the contextual-bandit entry decision via expectile + IQL pattern.

    No Bellman backup — entry is one-shot (no s'). Training loop:
      Repeat k_iterations:
        1. Q_theta = argmin sum_i (r_i - Q(s_i, a_i))^2  (per-action)
        2. V_psi  = argmin sum_i L_tau(Q(s_i, a_i_taken) - V(s_i))
    """
    torch.manual_seed(seed)
    device = _device(prefer_cuda)
    state_dim = X_train_np.shape[1]
    n = X_train_np.shape[0]

    X = torch.as_tensor(X_train_np, dtype=torch.float32, device=device)
    a = torch.as_tensor(a_train_np, dtype=torch.long, device=device)
    r = torch.as_tensor(r_train_np, dtype=torch.float32, device=device)
    if sample_weights is not None:
        w = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)
    else:
        w = torch.ones(n, dtype=torch.float32, device=device)

    v_net = MLP(state_dim, 1, hidden_dim=hidden_dim, n_hidden=n_hidden, dropout=dropout).to(device)
    q_net = MLP(state_dim, n_actions, hidden_dim=hidden_dim, n_hidden=n_hidden, dropout=dropout).to(device)

    v_opt = torch.optim.Adam(v_net.parameters(), lr=lr, weight_decay=weight_decay)
    q_opt = torch.optim.Adam(q_net.parameters(), lr=lr, weight_decay=weight_decay)

    g = torch.Generator(device="cpu").manual_seed(seed)

    def _q_for_taken_action(q_full: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return q_full.gather(1, actions.view(-1, 1)).squeeze(-1)

    # Q warmup: regress Q(s, a) → r using only the action that was taken.
    q_net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            q_full = q_net(X[idx])
            q_taken = _q_for_taken_action(q_full, a[idx])
            loss = (w[idx] * (q_taken - r[idx]).pow(2)).mean()
            q_opt.zero_grad()
            loss.backward()
            q_opt.step()

    for _ in range(k_iterations):
        # Step 1: V via expectile regression on Q(s, a_taken).
        q_net.eval()
        with torch.no_grad():
            q_full_eval = q_net(X)
            q_taken_full = _q_for_taken_action(q_full_eval, a)
        v_net.train()
        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                v_pred = v_net(X[idx]).squeeze(-1)
                diff = q_taken_full[idx] - v_pred
                loss = expectile_loss(diff, tau, sample_weights=w[idx])
                v_opt.zero_grad()
                loss.backward()
                v_opt.step()

        # Step 2: Q refit on r (no Bellman; entry is one-shot bandit).
        v_net.eval()
        q_net.train()
        for _ in range(epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                q_full = q_net(X[idx])
                q_taken = _q_for_taken_action(q_full, a[idx])
                loss = (w[idx] * (q_taken - r[idx]).pow(2)).mean()
                q_opt.zero_grad()
                loss.backward()
                q_opt.step()

    v_net.eval()
    q_net.eval()
    return EntryIQLGpuModel(
        v_net=v_net, q_net=q_net,
        state_dim=state_dim, n_actions=n_actions, device=device,
    )


def entry_iql_policy_action_probs(
    X_np: np.ndarray,
    model: EntryIQLGpuModel,
    *,
    beta: float,
) -> np.ndarray:
    """π(a | s) = softmax(β · Q(s, ·)) over n_actions. Returns (n, n_actions)."""
    q = model.predict_q(X_np)  # (n, n_actions)
    scaled = beta * q
    scaled = scaled - scaled.max(axis=1, keepdims=True)  # numerical stability
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def entry_iql_policy_argmax_action(
    X_np: np.ndarray, model: EntryIQLGpuModel,
) -> np.ndarray:
    """Greedy action: argmax_a Q(s, a). Returns (n,) of action ids."""
    q = model.predict_q(X_np)
    return q.argmax(axis=1)


def info(model: EntryIQLGpuModel) -> dict[str, Any]:
    return {
        "device": str(model.device),
        "state_dim_v1": int(model.state_dim),
        "n_actions_v1": int(model.n_actions),
        "v_param_count_v1": sum(p.numel() for p in model.v_net.parameters()),
        "q_param_count_v1": sum(p.numel() for p in model.q_net.parameters()),
        "action_labels_v1": ACTION_LABELS,
    }
