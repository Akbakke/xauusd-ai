"""Entry-IQL V2 runtime adapter — load trained checkpoint, infer per candidate.

Mission
-------
Wire the entry-IQL V2 (multi-head Q(s, a, K)) into the live runtime as a
"smart head" that supervises and overrides the entry transformer (V10).
This adapter is the bridge between offline training artifacts and online
candidate decisions.

Live-stack position:
    XGB → V10 (frozen) → Skip-V2 → [Entry-IQL adapter] → V3 (frozen) → Exit-IQL → ensemble

At each candidate moment, the adapter:
  1. Receives a candidate state dict (post-V10, post-Skip-V2)
  2. Builds a 26-dim state vector matching training feature order/encoding
  3. Runs multi-head Q(s, a, K) inference on the loaded model
  4. Aggregates Q across K-horizons via the chosen aggregator (mean/max/weighted)
  5. Returns an EntryRecommendation with: action_id, action_label, q_per_action,
     q_per_action_per_K, advantage_over_skip, confidence

The recommendation is RESEARCH-ONLY by default; the runtime decides whether
to follow it. Once the joint policy validation gate passes, this adapter
becomes the production override layer.

Schema
------
Checkpoint format `MULTI_HEAD_ENTRY_IQL_V2_CHECKPOINT` (saved by
`materialize_build_entry_iql_v2.py`):
  - q_state_dict, v_state_dict
  - feature_means, feature_stds  (z-score normalization)
  - k_horizons, n_actions, state_dim
  - hidden_dim, n_hidden, dropout
  - variant, fold_id

Companion `summary_v1.json` (in the same artifact_root) provides:
  - feature_names_v1  (ordered list of 26 feature column names)

The adapter requires BOTH the .pt and the summary to reconstruct correct
feature ordering; loading is fail-fast if either is missing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from gx1.scripts import entry_iql_multi_head_gpu_core_v1 as iql_core


VALID_AGGREGATORS = ("mean", "max", "weighted")
DEFAULT_AGGREGATOR = "mean"


@dataclass(frozen=True)
class EntryRecommendation:
    """Per-candidate IQL recommendation surface."""
    action_id_v1: int
    action_label_v1: str
    q_per_action_v1: np.ndarray              # (n_actions,) — aggregated Q
    q_per_action_per_k_v1: np.ndarray         # (n_actions, n_K) — full multi-head
    advantage_over_skip_v1: float            # q[chosen] - q[SKIP]
    advantage_over_realized_v1: float         # q[chosen] - q[best_take_now]
    confidence_softmax_v1: np.ndarray         # softmax(beta·q_per_action) — (n_actions,)
    aggregator_v1: str
    k_horizons_v1: list[int]
    variant_v1: str
    fold_id_v1: str
    feature_names_v1: list[str]
    state_v1: np.ndarray                      # raw (un-normalized) state vector


@dataclass
class EntryIQLV2Adapter:
    """Loaded IQL V2 model + feature contract, ready for inference."""
    model: iql_core.MultiHeadEntryIQLModel
    feature_names: list[str]
    variant: str
    fold_id: str
    aggregator: str
    beta: float
    k_weights: np.ndarray | None
    artifact_root: Path

    # ------- Loading -------

    @classmethod
    def load(
        cls,
        artifact_root: Path,
        *,
        variant: str = "R_NET_REAL",
        fold_id: str = "FOLD_1",
        aggregator: str = DEFAULT_AGGREGATOR,
        beta: float = 1.0,
        k_weights: Sequence[float] | None = None,
        prefer_cuda: bool = True,
    ) -> "EntryIQLV2Adapter":
        if aggregator not in VALID_AGGREGATORS:
            raise ValueError(f"aggregator {aggregator!r} not in {VALID_AGGREGATORS}")
        artifact_root = Path(artifact_root)
        summary_path = artifact_root / "summary_v1.json"
        ckpt_path = artifact_root / "trained_models_v1" / f"{variant}_{fold_id}.pt"
        if not summary_path.is_file():
            raise FileNotFoundError(f"summary missing: {summary_path}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")
        summary = json.loads(summary_path.read_text())
        feature_names = list(summary["feature_names_v1"])
        device = torch.device("cuda") if (prefer_cuda and torch.cuda.is_available()) else torch.device("cpu")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ckpt.get("schema_v1") not in (None, "MULTI_HEAD_ENTRY_IQL_V2_CHECKPOINT"):
            raise ValueError(f"unsupported checkpoint schema: {ckpt.get('schema_v1')}")
        state_dim = int(ckpt["state_dim"])
        n_actions = int(ckpt["n_actions"])
        k_horizons = list(ckpt["k_horizons"])
        n_k = len(k_horizons)
        # Reconstruct architecture. hidden_dim/n_hidden default to fast-budget
        # values for older checkpoints that didn't save them.
        hidden_dim = int(ckpt.get("hidden_dim", 128))
        n_hidden = int(ckpt.get("n_hidden", 2))
        dropout = float(ckpt.get("dropout", iql_core.DEFAULT_DROPOUT))
        if state_dim != len(feature_names):
            raise RuntimeError(
                f"state_dim mismatch: ckpt={state_dim} vs summary feature_names={len(feature_names)}"
            )
        q_net = iql_core.MLP(state_dim, n_actions * n_k,
                             hidden_dim=hidden_dim, n_hidden=n_hidden, dropout=dropout).to(device)
        v_net = iql_core.MLP(state_dim, n_k,
                             hidden_dim=hidden_dim, n_hidden=n_hidden, dropout=dropout).to(device)
        q_net.load_state_dict(ckpt["q_state_dict"])
        v_net.load_state_dict(ckpt["v_state_dict"])
        q_net.eval()
        v_net.eval()
        feature_means = np.asarray(ckpt["feature_means"], dtype=np.float32)
        feature_stds = np.asarray(ckpt["feature_stds"], dtype=np.float32)
        model = iql_core.MultiHeadEntryIQLModel(
            q_net=q_net, v_net=v_net,
            state_dim=state_dim, n_actions=n_actions, n_k=n_k,
            k_horizons=k_horizons, device=device,
            feature_means=feature_means, feature_stds=feature_stds,
        )
        kw = np.asarray(k_weights, dtype=np.float32) if k_weights is not None else None
        if aggregator == "weighted" and (kw is None or len(kw) != n_k):
            raise ValueError(f"weighted aggregator needs k_weights of length {n_k}")
        return cls(
            model=model,
            feature_names=feature_names,
            variant=str(ckpt.get("variant", variant)),
            fold_id=str(ckpt.get("fold_id", fold_id)),
            aggregator=aggregator,
            beta=float(beta),
            k_weights=kw,
            artifact_root=artifact_root,
        )

    # ------- State construction -------

    def build_state_vector(self, candidate: dict[str, Any]) -> np.ndarray:
        """Convert a runtime candidate dict to a 1-D state vector.

        Numeric features are looked up by name, missing → 0.
        One-hot features (those with `__` in the feature name) are activated
        if the underlying categorical column matches the suffix value.

        Returns un-normalized vector; caller (or predict_one) applies z-score.
        """
        v = np.zeros(len(self.feature_names), dtype=np.float32)
        for i, fname in enumerate(self.feature_names):
            if "__" in fname:
                cat_col, _, cat_val = fname.partition("__")
                runtime_val = candidate.get(cat_col)
                if runtime_val is not None and str(runtime_val) == cat_val:
                    v[i] = 1.0
                else:
                    v[i] = 0.0
            else:
                raw = candidate.get(fname)
                if raw is None:
                    v[i] = 0.0
                else:
                    try:
                        v[i] = float(raw)
                    except (TypeError, ValueError):
                        v[i] = 0.0
        return v

    # ------- Inference -------

    def predict(self, candidates: list[dict[str, Any]]) -> list[EntryRecommendation]:
        """Vectorized inference over a batch of runtime candidate dicts."""
        n = len(candidates)
        if n == 0:
            return []
        states = np.stack([self.build_state_vector(c) for c in candidates], axis=0)
        q_full = self.model.predict_q(states)  # (n, n_actions, n_K)
        # Aggregate across K
        if self.aggregator == "mean":
            q_agg = q_full.mean(axis=2)
        elif self.aggregator == "max":
            q_agg = q_full.max(axis=2)
        elif self.aggregator == "weighted":
            q_agg = (q_full * self.k_weights[None, None, :]).sum(axis=2)
        else:
            raise AssertionError(f"unhandled aggregator {self.aggregator}")

        actions = q_agg.argmax(axis=1)
        # Softmax confidence
        scaled = self.beta * q_agg
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        soft = np.exp(scaled)
        soft = soft / soft.sum(axis=1, keepdims=True)
        # Advantages
        skip_id = iql_core.ACTION_SKIP_ID
        out: list[EntryRecommendation] = []
        for i in range(n):
            a = int(actions[i])
            chosen_q = float(q_agg[i, a])
            adv_skip = chosen_q - float(q_agg[i, skip_id])
            # Take_now advantage: chosen vs the better of TAKE_LONG / TAKE_SHORT
            best_take_q = float(max(q_agg[i, iql_core.ACTION_TAKE_LONG_NOW_ID],
                                     q_agg[i, iql_core.ACTION_TAKE_SHORT_NOW_ID]))
            adv_take = chosen_q - best_take_q
            out.append(EntryRecommendation(
                action_id_v1=a,
                action_label_v1=iql_core.ACTION_LABELS_V1[a],
                q_per_action_v1=q_agg[i].copy(),
                q_per_action_per_k_v1=q_full[i].copy(),
                advantage_over_skip_v1=adv_skip,
                advantage_over_realized_v1=adv_take,
                confidence_softmax_v1=soft[i].copy(),
                aggregator_v1=self.aggregator,
                k_horizons_v1=list(self.model.k_horizons),
                variant_v1=self.variant,
                fold_id_v1=self.fold_id,
                feature_names_v1=list(self.feature_names),
                state_v1=states[i].copy(),
            ))
        return out

    def predict_one(self, candidate: dict[str, Any]) -> EntryRecommendation:
        return self.predict([candidate])[0]

    # ------- Diagnostic -------

    def info(self) -> dict[str, Any]:
        return {
            "artifact_root_v1": str(self.artifact_root),
            "variant_v1": self.variant,
            "fold_id_v1": self.fold_id,
            "aggregator_v1": self.aggregator,
            "beta_v1": self.beta,
            "feature_count_v1": len(self.feature_names),
            "feature_names_v1": list(self.feature_names),
            "k_horizons_v1": list(self.model.k_horizons),
            "n_actions_v1": int(self.model.n_actions),
            "device_v1": str(self.model.device),
            "schema_v1": "ENTRY_IQL_V2_RUNTIME_ADAPTER_V1",
        }
