"""V12 Exit-Decider runtime adapter — wraps Exit-IQL v5 + V3 fail-safe override.

Mission
-------
At runtime, V12 design (handover protocol §5 Stage 4) calls for:

    if v3_v8_should_exit_prob > 0.95:
        decision = EXIT_NOW         # bypass IQL Q-values
    else:
        decision = argmax(IQL_Q[hold, exit])

This adapter encapsulates that override logic over a trained Exit-IQL v5
checkpoint, so the live runtime sees a single decide(bar_state) entry point.

Stack position:
    XGB → V10 → Entry-IQL v3 → [opens trade] → V3 v8 → ExitDeciderV12 → close

The bar_state dict at decision time MUST include:
  - all keys Exit-IQL v5's adapter expects (per `feature_names`)
  - "v3_v8_should_exit_prob": float in [0,1]   (computed upstream by V3 v8 inference per bar)

The override threshold is configurable (default 0.95). Setting it to None
(or 1.0) disables override → behavior identical to plain Exit-IQL v5 (V12_OFF).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gx1.runtime.exit_iql_v2_adapter import ExitIQLV2Adapter, ExitRecommendation


V3_OVERRIDE_DEFAULT_THRESHOLD = 0.95
EXIT_HOLD_ID = 0
EXIT_NOW_ID = 1


@dataclass(frozen=True)
class ExitDeciderV12Recommendation:
    """Per-bar V12 exit decision."""
    action_id_v1: int                  # 0=HOLD, 1=EXIT_NOW
    action_label_v1: str               # "HOLD" | "EXIT_NOW"
    decision_source_v1: str            # "V3_OVERRIDE" | "IQL_Q"
    v3_should_exit_prob_v1: float      # raw V3 v8 prob seen at this bar
    iql_recommendation_v1: ExitRecommendation
    override_threshold_v1: float


@dataclass
class ExitDeciderV12Adapter:
    """Wrap Exit-IQL v5 with V3 fail-safe override (V12 Stage 4)."""
    iql_adapter: ExitIQLV2Adapter
    v3_override_threshold: float | None = V3_OVERRIDE_DEFAULT_THRESHOLD
    v3_prob_field: str = "v3_v8_should_exit_prob"

    @classmethod
    def load(
        cls,
        artifact_root: Path,
        *,
        variant: str = "R_V12",
        fold_id: str = "FOLD_1",
        v3_override_threshold: float | None = V3_OVERRIDE_DEFAULT_THRESHOLD,
        prefer_cuda: bool = True,
    ) -> "ExitDeciderV12Adapter":
        iql = ExitIQLV2Adapter.load(
            artifact_root=Path(artifact_root),
            variant=variant, fold_id=fold_id,
            prefer_cuda=prefer_cuda,
        )
        thr = float(v3_override_threshold) if v3_override_threshold is not None else None
        return cls(iql_adapter=iql, v3_override_threshold=thr)

    def decide(self, bar_state: dict[str, Any]) -> ExitDeciderV12Recommendation:
        """Decide HOLD/EXIT_NOW for one bar. V3 fail-safe checked first."""
        v3_prob = float(bar_state.get(self.v3_prob_field, 0.0) or 0.0)
        iql_rec = self.iql_adapter.predict_one(bar_state)

        if (self.v3_override_threshold is not None
                and v3_prob > self.v3_override_threshold):
            return ExitDeciderV12Recommendation(
                action_id_v1=EXIT_NOW_ID,
                action_label_v1="EXIT_NOW",
                decision_source_v1="V3_OVERRIDE",
                v3_should_exit_prob_v1=v3_prob,
                iql_recommendation_v1=iql_rec,
                override_threshold_v1=self.v3_override_threshold,
            )
        return ExitDeciderV12Recommendation(
            action_id_v1=int(iql_rec.action_id_v1),
            action_label_v1=str(iql_rec.action_label_v1),
            decision_source_v1="IQL_Q",
            v3_should_exit_prob_v1=v3_prob,
            iql_recommendation_v1=iql_rec,
            override_threshold_v1=self.v3_override_threshold,
        )

    def decide_batch(self, bar_states: list[dict[str, Any]]) -> list[ExitDeciderV12Recommendation]:
        """Vectorized decide for a batch (e.g. all bars of one trade or many trades)."""
        n = len(bar_states)
        if n == 0:
            return []
        # Run IQL inference in batch for efficiency
        iql_recs = self.iql_adapter.predict(bar_states)
        out: list[ExitDeciderV12Recommendation] = []
        for s, r in zip(bar_states, iql_recs):
            v3_prob = float(s.get(self.v3_prob_field, 0.0) or 0.0)
            if (self.v3_override_threshold is not None
                    and v3_prob > self.v3_override_threshold):
                out.append(ExitDeciderV12Recommendation(
                    action_id_v1=EXIT_NOW_ID,
                    action_label_v1="EXIT_NOW",
                    decision_source_v1="V3_OVERRIDE",
                    v3_should_exit_prob_v1=v3_prob,
                    iql_recommendation_v1=r,
                    override_threshold_v1=self.v3_override_threshold,
                ))
            else:
                out.append(ExitDeciderV12Recommendation(
                    action_id_v1=int(r.action_id_v1),
                    action_label_v1=str(r.action_label_v1),
                    decision_source_v1="IQL_Q",
                    v3_should_exit_prob_v1=v3_prob,
                    iql_recommendation_v1=r,
                    override_threshold_v1=self.v3_override_threshold,
                ))
        return out

    def info(self) -> dict[str, Any]:
        return {
            "schema_v1": "EXIT_DECIDER_V12_RUNTIME_ADAPTER_V1",
            "iql_adapter_v1": self.iql_adapter.info(),
            "v3_override_threshold_v1": self.v3_override_threshold,
            "v3_prob_field_v1": self.v3_prob_field,
        }
