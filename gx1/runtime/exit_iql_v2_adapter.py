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


VALID_AGGREGATORS = ("mean", "max", "weighted")
DEFAULT_AGGREGATOR = "mean"
ACTION_HOLD_ID = 0
ACTION_EXIT_NOW_ID = 1
ACTION_LABELS_EXIT = {ACTION_HOLD_ID: "HOLD", ACTION_EXIT_NOW_ID: "EXIT_NOW"}

# Fail-closed feature-coverage guard derived from the active Exit checkpoint.
# Trainer computes feature_stds = nanstd + 1e-6, so constant-in-training features
# sit at ~1e-6; real-variance features are >= ~0.1. A real-variance feature
# silently 0-filled at serve is the 2026-05-19 LONG-bias skew. std >= threshold
# => REQUIRED at inference; derived per-bundle from the checkpoint (one truth).
STD_REQUIRED_THRESHOLD = 1e-3


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
    strict_failclosed: bool = True  # raise (not 0-fill) if a train-variance feature is missing
    warmup_grace_features: frozenset = frozenset()  # names demoted to warn-only (cold-start regime/MTF)
    required_feature_names: frozenset = frozenset()  # derived in load(): std >= STD_REQUIRED_THRESHOLD

    @classmethod
    def load(
        cls, artifact_root: Path, *,
        variant: str = "R_NET_REAL", fold_id: str = "FOLD_1",
        aggregator: str = DEFAULT_AGGREGATOR, beta: float = 1.0,
        k_weights: Sequence[float] | None = None,
        prefer_cuda: bool = True,
        exit_margin: float = 0.0,
        strict_failclosed: bool = True,
        warmup_grace_features: Sequence[str] | None = None,
    ) -> "ExitIQLV2Adapter":
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
        if ckpt.get("schema_v1") not in (None, "MULTI_HEAD_EXIT_IQL_V2_CHECKPOINT"):
            raise ValueError(f"unsupported checkpoint schema: {ckpt.get('schema_v1')}")
        state_dim = int(ckpt["state_dim"])
        n_actions = int(ckpt["n_actions"])
        if n_actions != 2:
            raise ValueError(f"exit-IQL must have 2 actions; got {n_actions}")
        k_horizons = list(ckpt["k_horizons"])
        n_k = len(k_horizons)
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
        model = iql_core.MultiHeadExitIQLModel(
            q_net=q_net, v_net=v_net,
            state_dim=state_dim, n_actions=n_actions, n_k=n_k,
            k_horizons=k_horizons, device=device,
            feature_means=feature_means, feature_stds=feature_stds,
        )
        kw = np.asarray(k_weights, dtype=np.float32) if k_weights is not None else None
        if aggregator == "weighted" and (kw is None or len(kw) != n_k):
            raise ValueError(f"weighted aggregator needs k_weights of length {n_k}")
        # Derive REQUIRED features from the checkpoint's training stds (one truth).
        required_feature_names = frozenset(
            fn for fn, sd in zip(feature_names, feature_stds.tolist())
            if float(sd) >= STD_REQUIRED_THRESHOLD
        )
        return cls(
            model=model,
            feature_names=feature_names,
            variant=str(ckpt.get("variant", variant)),
            fold_id=str(ckpt.get("fold_id", fold_id)),
            aggregator=aggregator, beta=float(beta), k_weights=kw,
            artifact_root=artifact_root,
            exit_margin=float(exit_margin),
            strict_failclosed=bool(strict_failclosed),
            warmup_grace_features=frozenset(warmup_grace_features or ()),
            required_feature_names=required_feature_names,
        )

    def build_state_vector(self, bar_state: dict[str, Any]) -> np.ndarray:
        v = np.zeros(len(self.feature_names), dtype=np.float32)
        missing: list[str] = []
        for i, fname in enumerate(self.feature_names):
            if "__" in fname:
                cat_col, _, cat_val = fname.partition("__")
                runtime_val = bar_state.get(cat_col)
                if runtime_val is None:
                    missing.append(fname)
                    continue
                if str(runtime_val) == cat_val:
                    v[i] = 1.0
            else:
                raw = bar_state.get(fname)
                if raw is None:
                    missing.append(fname)
                    continue
                try:
                    v[i] = float(raw)
                except (TypeError, ValueError):
                    v[i] = 0.0
        # Fail-closed coverage guard (2026-06-04). A REQUIRED feature (real training
        # variance, std >= STD_REQUIRED_THRESHOLD) missing at serve is the 2026-05-19
        # LONG-bias skew → refuse to 0-fill. Constants / one-hots / dead-zero debt
        # (std ~1e-6) stay warn-only. warmup_grace_features demotes named cold-start
        # features to warn-only.
        if missing:
            required_missing = [
                f for f in missing
                if f in self.required_feature_names and f not in self.warmup_grace_features
            ]
            if required_missing and self.strict_failclosed:
                raise RuntimeError(
                    f"[FEATURE_COVERAGE_FATAL] {len(required_missing)} required "
                    f"(train-variance) feature(s) missing from bar_state — refusing to "
                    f"0-fill (2026-05-19 skew guard): {required_missing[:20]}"
                    + (f" (+{len(required_missing)-20} more)" if len(required_missing) > 20 else "")
                    + " | pass strict_failclosed=False or add to warmup_grace_features to override."
                )
            if not getattr(self, "_missing_warned_set", None) == tuple(missing):
                import logging
                n_miss = len(missing)
                tot = len(self.feature_names)
                head = ", ".join(missing[:10])
                logging.getLogger("exit_iql_v2_adapter").warning(
                    f"[FEATURE_COVERAGE] {n_miss}/{tot} features missing from bar_state "
                    f"(0-fill; {len(required_missing)} required) — first 10: [{head}]"
                    + (f" (+{n_miss-10} more)" if n_miss > 10 else "")
                )
                self._missing_warned_set = tuple(missing)
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
            "k_horizons_v1": list(self.model.k_horizons),
            "n_actions_v1": int(self.model.n_actions),
            "device_v1": str(self.model.device),
            "schema_v1": "EXIT_IQL_V2_RUNTIME_ADAPTER_V1",
        }
