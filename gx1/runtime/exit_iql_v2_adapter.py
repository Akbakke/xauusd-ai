"""Exit-IQL V2 runtime adapter — load trained checkpoint, infer per held-trade-bar.

Mission
-------
Wire the exit-IQL V2 (multi-head Q(s, a, K), 2-action HOLD/EXIT_NOW) into the
live runtime as the "smart RL head" supervising the exit transformer (V3).

Live-stack position:
    [model-native Entry opens trade] → V3 (frozen) → [Exit-IQL adapter] → ensemble exit

At each bar inside an active trade, the adapter:
  1. Receives a per-bar state dict (trade-state + carried candidate context)
  2. Builds a state vector matching training feature order/encoding
  3. Runs multi-head Q(s, HOLD/EXIT, K) inference
  4. Returns ExitRecommendation with action_id, action_label, advantage_over_hold,
     per-K Q-values, confidence

Schema
------
Checkpoint format `MULTI_HEAD_EXIT_IQL_V2_CHECKPOINT` (saved by
`materialize_build_exit_iql_v2.py`). Companion summary_v1.json provides
feature_names_v1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from gx1.scripts import exit_iql_multi_head_gpu_core_v1 as iql_core
from gx1_guards.artifacts import (
    exit_iql_ordered_feature_names_sha256,
    require_exit_iql_summary_contract,
)


VALID_AGGREGATORS = ("mean", "max", "weighted")
DEFAULT_AGGREGATOR = "mean"
ACTION_HOLD_ID = 0
ACTION_EXIT_NOW_ID = 1
ACTION_LABELS_EXIT = {ACTION_HOLD_ID: "HOLD", ACTION_EXIT_NOW_ID: "EXIT_NOW"}


def _checkpoint_float_vector(
    raw: object,
    *,
    field_name: str,
    state_dim: int,
) -> np.ndarray:
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().numpy()
    try:
        vector = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Exit-IQL checkpoint {field_name} is not a float vector"
        ) from exc
    if vector.shape != (state_dim,):
        raise RuntimeError(
            f"Exit-IQL checkpoint {field_name} shape={vector.shape}, "
            f"expected ({state_dim},)"
        )
    if not np.isfinite(vector).all():
        raise RuntimeError(
            f"Exit-IQL checkpoint {field_name} contains non-finite values"
        )
    return vector


def require_exit_iql_checkpoint_binding(
    ckpt: object,
    *,
    feature_names: list[str],
    feature_names_sha256: str,
    requested_variant: str,
    requested_fold_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Bind checkpoint weights and normalization to one exact feature order."""

    if not isinstance(ckpt, dict):
        raise RuntimeError("Exit-IQL checkpoint must be a mapping")
    if ckpt.get("schema_v1") != "MULTI_HEAD_EXIT_IQL_V2_CHECKPOINT":
        raise RuntimeError(
            f"unsupported checkpoint schema: {ckpt.get('schema_v1')!r}"
        )
    if ckpt.get("variant") != requested_variant:
        raise RuntimeError(
            "Exit-IQL checkpoint variant does not match the requested variant"
        )
    if ckpt.get("fold_id") != requested_fold_id:
        raise RuntimeError(
            "Exit-IQL checkpoint fold_id does not match the requested serving fold"
        )

    checkpoint_feature_names = ckpt.get("feature_names_v1")
    if not isinstance(checkpoint_feature_names, list):
        raise RuntimeError(
            "Exit-IQL checkpoint lacks ordered feature_names_v1"
        )
    if checkpoint_feature_names != feature_names:
        raise RuntimeError(
            "Exit-IQL checkpoint feature_names_v1 differs from summary order"
        )
    checkpoint_hash = ckpt.get("feature_names_sha256_v1")
    if (
        not isinstance(checkpoint_hash, str)
        or checkpoint_hash != checkpoint_hash.strip().lower()
        or len(checkpoint_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in checkpoint_hash)
    ):
        raise RuntimeError(
            "Exit-IQL checkpoint feature_names_sha256_v1 is not an exact SHA-256"
        )
    computed_hash = exit_iql_ordered_feature_names_sha256(
        checkpoint_feature_names
    )
    if checkpoint_hash != computed_hash:
        raise RuntimeError(
            "Exit-IQL checkpoint ordered feature_names_v1 SHA-256 mismatch"
        )
    if checkpoint_hash != feature_names_sha256:
        raise RuntimeError(
            "Exit-IQL checkpoint feature hash differs from summary feature hash"
        )

    state_dim = ckpt.get("state_dim")
    if type(state_dim) is not int or state_dim != len(feature_names):
        raise RuntimeError(
            "Exit-IQL checkpoint state_dim does not match bound feature_names_v1"
        )
    feature_means = _checkpoint_float_vector(
        ckpt.get("feature_means"),
        field_name="feature_means",
        state_dim=state_dim,
    )
    feature_stds = _checkpoint_float_vector(
        ckpt.get("feature_stds"),
        field_name="feature_stds",
        state_dim=state_dim,
    )
    if not bool((feature_stds > 0.0).all()):
        raise RuntimeError(
            "Exit-IQL checkpoint feature_stds must be strictly positive"
        )
    return feature_means, feature_stds


@dataclass(frozen=True)
class ExitRecommendation:
    action_id_v1: int
    action_label_v1: str
    q_per_action_v1: np.ndarray              # (2,) — aggregated Q
    q_per_action_per_k_v1: np.ndarray         # (2, n_K)
    advantage_exit_over_hold_v1: float       # q[EXIT] - q[HOLD]
    confidence_softmax_v1: np.ndarray         # (2,)
    aggregator_v1: str
    k_horizons_v1: list[int]
    variant_v1: str
    fold_id_v1: str
    feature_names_v1: list[str]
    state_v1: np.ndarray


@dataclass
class ExitIQLV2Adapter:
    model: iql_core.MultiHeadExitIQLModel
    feature_names: list[str]
    variant: str
    fold_id: str
    aggregator: str
    beta: float
    k_weights: np.ndarray | None
    artifact_root: Path
    exit_margin: float = 0.0  # V9 Issue 2: relax decision threshold (>0 fires more, <0 fires less)
    required_feature_names: frozenset = frozenset()

    @classmethod
    def load(
        cls, artifact_root: Path, *,
        fold_id: str,
        variant: str = "R_NET_REAL",
        aggregator: str = DEFAULT_AGGREGATOR, beta: float = 1.0,
        k_weights: Sequence[float] | None = None,
        prefer_cuda: bool = True,
        exit_margin: float = 0.0,
    ) -> "ExitIQLV2Adapter":
        if aggregator not in VALID_AGGREGATORS:
            raise ValueError(f"aggregator {aggregator!r} not in {VALID_AGGREGATORS}")
        if not isinstance(fold_id, str) or not fold_id or fold_id != fold_id.strip():
            raise ValueError("fold_id must be one explicit exact serving fold")
        artifact_root = Path(artifact_root)
        summary_path = artifact_root / "summary_v1.json"
        ckpt_path = artifact_root / "trained_models_v1" / f"{variant}_{fold_id}.pt"
        if not summary_path.is_file():
            raise FileNotFoundError(f"summary missing: {summary_path}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        feature_names, feature_names_sha256 = require_exit_iql_summary_contract(
            summary,
            context=f"Exit-IQL bundle {artifact_root}",
        )
        device = torch.device("cuda") if (prefer_cuda and torch.cuda.is_available()) else torch.device("cpu")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        feature_means, feature_stds = require_exit_iql_checkpoint_binding(
            ckpt,
            feature_names=feature_names,
            feature_names_sha256=feature_names_sha256,
            requested_variant=variant,
            requested_fold_id=fold_id,
        )
        state_dim = int(ckpt["state_dim"])
        n_actions = int(ckpt["n_actions"])
        if n_actions != 2:
            raise ValueError(f"exit-IQL must have 2 actions; got {n_actions}")
        k_horizons = list(ckpt["k_horizons"])
        n_k = len(k_horizons)
        hidden_dim = int(ckpt.get("hidden_dim", 128))
        n_hidden = int(ckpt.get("n_hidden", 2))
        dropout = float(ckpt.get("dropout", iql_core.DEFAULT_DROPOUT))
        q_net = iql_core.MLP(state_dim, n_actions * n_k,
                             hidden_dim=hidden_dim, n_hidden=n_hidden, dropout=dropout).to(device)
        v_net = iql_core.MLP(state_dim, n_k,
                             hidden_dim=hidden_dim, n_hidden=n_hidden, dropout=dropout).to(device)
        q_net.load_state_dict(ckpt["q_state_dict"])
        v_net.load_state_dict(ckpt["v_state_dict"])
        q_net.eval()
        v_net.eval()
        model = iql_core.MultiHeadExitIQLModel(
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
            aggregator=aggregator, beta=float(beta), k_weights=kw,
            artifact_root=artifact_root,
            exit_margin=float(exit_margin),
            required_feature_names=frozenset(feature_names),
        )

    def build_state_vector(self, bar_state: dict[str, Any]) -> np.ndarray:
        v = np.zeros(len(self.feature_names), dtype=np.float32)
        missing: list[str] = []
        categorical_columns: set[str] = set()
        matched_categorical_columns: set[str] = set()
        for i, fname in enumerate(self.feature_names):
            if "__" in fname:
                cat_col, _, cat_val = fname.partition("__")
                categorical_columns.add(cat_col)
                runtime_val = bar_state.get(cat_col)
                if runtime_val is None:
                    missing.append(fname)
                    continue
                if str(runtime_val) == cat_val:
                    v[i] = 1.0
                    matched_categorical_columns.add(cat_col)
            else:
                raw = bar_state.get(fname)
                if raw is None:
                    missing.append(fname)
                    continue
                try:
                    v[i] = float(raw)
                except (TypeError, ValueError):
                    missing.append(fname)
                    continue
                if not np.isfinite(v[i]):
                    missing.append(fname)
        for cat_col in categorical_columns - matched_categorical_columns:
            if bar_state.get(cat_col) is not None:
                missing.append(f"{cat_col}__<known-category-required>")
        if missing:
            unique_missing = list(dict.fromkeys(missing))
            raise RuntimeError(
                f"[FEATURE_COVERAGE_FATAL] {len(unique_missing)} model-bound "
                "feature(s) are missing, non-numeric, or non-finite; refusing "
                f"to manufacture state values: {unique_missing[:20]}"
                + (
                    f" (+{len(unique_missing)-20} more)"
                    if len(unique_missing) > 20
                    else ""
                )
            )
        return v

    def predict(self, bar_states: list[dict[str, Any]]) -> list[ExitRecommendation]:
        n = len(bar_states)
        if n == 0:
            return []
        states = np.stack([self.build_state_vector(s) for s in bar_states], axis=0)
        q_full = self.model.predict_q(states)  # (n, 2, n_K)
        if self.aggregator == "mean":
            q_agg = q_full.mean(axis=2)
        elif self.aggregator == "max":
            q_agg = q_full.max(axis=2)
        elif self.aggregator == "weighted":
            q_agg = (q_full * self.k_weights[None, None, :]).sum(axis=2)
        else:
            raise AssertionError(f"unhandled aggregator {self.aggregator}")

        # V9 Issue 2: threshold gate. Default exit_margin=0 = argmax (current).
        # exit_margin > 0 → fire EXIT_NOW more aggressively (Q_EXIT_NOW > Q_HOLD - margin).
        # exit_margin < 0 → require larger advantage for EXIT_NOW (more conservative).
        if self.exit_margin == 0.0:
            actions = q_agg.argmax(axis=1)
        else:
            adv_eh_all = q_agg[:, ACTION_EXIT_NOW_ID] - q_agg[:, ACTION_HOLD_ID]
            actions = (adv_eh_all > -self.exit_margin).astype(np.int64)
        scaled = self.beta * q_agg
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        soft = np.exp(scaled)
        soft = soft / soft.sum(axis=1, keepdims=True)
        out: list[ExitRecommendation] = []
        for i in range(n):
            a = int(actions[i])
            adv_eh = float(q_agg[i, ACTION_EXIT_NOW_ID] - q_agg[i, ACTION_HOLD_ID])
            out.append(ExitRecommendation(
                action_id_v1=a,
                action_label_v1=ACTION_LABELS_EXIT[a],
                q_per_action_v1=q_agg[i].copy(),
                q_per_action_per_k_v1=q_full[i].copy(),
                advantage_exit_over_hold_v1=adv_eh,
                confidence_softmax_v1=soft[i].copy(),
                aggregator_v1=self.aggregator,
                k_horizons_v1=list(self.model.k_horizons),
                variant_v1=self.variant, fold_id_v1=self.fold_id,
                feature_names_v1=list(self.feature_names),
                state_v1=states[i].copy(),
            ))
        return out

    def predict_one(self, bar_state: dict[str, Any]) -> ExitRecommendation:
        return self.predict([bar_state])[0]

    def info(self) -> dict[str, Any]:
        return {
            "artifact_root_v1": str(self.artifact_root),
            "variant_v1": self.variant, "fold_id_v1": self.fold_id,
            "aggregator_v1": self.aggregator, "beta_v1": self.beta,
            "feature_count_v1": len(self.feature_names),
            "feature_names_sha256_v1": exit_iql_ordered_feature_names_sha256(
                self.feature_names
            ),
            "k_horizons_v1": list(self.model.k_horizons),
            "n_actions_v1": int(self.model.n_actions),
            "device_v1": str(self.model.device),
            "schema_v1": "EXIT_IQL_V2_RUNTIME_ADAPTER_V1",
        }
