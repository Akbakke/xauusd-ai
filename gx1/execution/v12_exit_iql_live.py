#!/usr/bin/env python3
"""V12 Exit-IQL V12.1 live inference wrapper — the per-bar exit-decision
loop for open trades.

Wraps ExitDeciderV12Adapter (cemented V12.1.1 NO_TRAIL config) with a
state-builder that converts:
  - TradeState running stats
  - V10 entry-snapshot (frozen at trade open)
  - V3 v8 outputs at this bar (currently STUBBED — see scope note)
  - Augmented canonical_v3 features at this bar
  - One-hot side flags (long/short)

…into the 201-feature bar_state dict the adapter expects.

V12.1.1 NO_TRAIL config (per project_gx1_v12_1_results_2026q2):
  - variant: R_V12_1
  - fold_id: FOLD_1
  - v3_override_threshold: None  (V3 fail-safe DISABLED — its
    contribution was found to degrade mean PnL in Phase 6, so the
    cemented config drops it. This means the V3 v8 outputs in the
    state vector only affect IQL's *Q-learning* — they don't trigger
    overrides. Stubbing them to 0 mildly biases Q-values but does not
    cause V3-based override-exits.)

⚠️ V3 v8 stubbing — known scope deferral:
  Live V3 v8 inference would require running a 91-feature × 512-bar
  transformer forward pass per M1 minute per active trade. That's a
  separate piece of infrastructure (~300-500 lines + a ~GB bundle
  load). For sesjon 4 the V3 outputs default to 0; the 4 V3-tracking
  features in Exit-IQL state are thus zeroed. Effect: Q-values diverge
  somewhat from training distribution. Validate in shadow mode before
  trusting for live trades.

Usage:
    exit_iql = ExitIQLLiveInference.load_default()
    # On each new M1 bar while a trade is open:
    trade.update_bar(bid=bid, ask=ask, m1_close=m1_close)
    rec = exit_iql.decide_for_trade(
        trade,
        canonical_v3_row=augmented_cv3.iloc[-1],
        v3_v8_out=None,    # None → stubbed-to-zero
    )
    if rec.action_id_v1 == 1:  # EXIT_NOW
        ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.runtime.exit_decider_v12_adapter import (
    ExitDeciderV12Adapter,
    ExitDeciderV12Recommendation,
)
from gx1.execution.v12_trade_state import TradeState, DEFAULT_V3_FEATURES

LOG = logging.getLogger("v12_exit_iql_live")

DEFAULT_BUNDLE_DIR = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "BUILD_EXIT_IQL_PER_BAR_DATASET_V12_20260508T093249Z_LOCK_V3TRACKED_"
    "20260508T225545Z_LOCK_GRID_6YR_NO_TRAIL_20260509T123707Z_LOCK"
)
DEFAULT_VARIANT = "R_V12_1"
DEFAULT_FOLD = "FOLD_1"
DEFAULT_V3_OVERRIDE = None    # V12.1.1 NO_TRAIL: V3 fail-safe disabled

SESSION_ID_TO_LABEL = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}


@dataclass
class ExitIQLLiveInference:
    decider: ExitDeciderV12Adapter
    feature_names: list[str] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        bundle_dir: Path = DEFAULT_BUNDLE_DIR,
        variant: str = DEFAULT_VARIANT,
        fold_id: str = DEFAULT_FOLD,
        v3_override_threshold: float | None = DEFAULT_V3_OVERRIDE,
        prefer_cuda: bool = True,
    ) -> "ExitIQLLiveInference":
        decider = ExitDeciderV12Adapter.load(
            artifact_root=Path(bundle_dir),
            variant=variant, fold_id=fold_id,
            v3_override_threshold=v3_override_threshold,
            prefer_cuda=prefer_cuda,
        )
        feature_names = list(decider.iql_adapter.feature_names)
        LOG.info(f"Exit-IQL V12.1 loaded: {bundle_dir.name}  variant={variant}  "
                  f"v3_override={'disabled' if v3_override_threshold is None else v3_override_threshold}")
        LOG.info(f"  feature_names: {len(feature_names)}")
        return cls(decider=decider, feature_names=feature_names)

    @classmethod
    def load_default(cls) -> "ExitIQLLiveInference":
        return cls.load()

    # ── state construction ───────────────────────────────────────────

    def build_bar_state(
        self,
        trade: TradeState,
        canonical_v3_row: pd.Series,
        v3_v8_out: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Build the 201-feature bar_state dict for the Exit-IQL adapter.

        Combines:
          - trade-state running stats (13)
          - V10 entry-snapshot (10)
          - V3 v8 outputs at this bar (4) — stubbed to 0 if v3_v8_out=None
          - augmented canonical_v3 features at this bar (~170)
          - side one-hot (2)
          - categorical one-hots (4)
        """
        bar_state: dict[str, Any] = {}
        # Trade state
        bar_state.update(trade.build_trade_state_features())
        # V10 entry snapshot
        bar_state.update(trade.build_v10_entry_snapshot_features())
        # V3 v8 outputs (stubbed if not provided)
        v3_block = v3_v8_out if v3_v8_out is not None else DEFAULT_V3_FEATURES
        bar_state.update({k: float(v) for k, v in v3_block.items()})
        # Side one-hot
        bar_state.update(trade.build_side_one_hot())
        # current_atr_bps
        bar_state["current_atr_bps_v1"] = float(canonical_v3_row.get("atr_bps", 0.0) or 0.0)

        # m5_phase one-hots (from canonical_v3)
        for p in range(5):
            col = f"m5_phase_{p}"
            bar_state[f"m5_phase_{p}_v1"] = float(canonical_v3_row.get(col, 0.0) or 0.0)

        # Categorical one-hots (matches training labels)
        sid = int(canonical_v3_row.get("session_id", 0) or 0)
        sess_label = SESSION_ID_TO_LABEL.get(sid, "ASIA")
        bar_state[f"session_{sess_label}"] = 1.0
        # Set zeros for other sessions
        for s in ("ASIA", "EU", "OVERLAP", "US"):
            bar_state.setdefault(f"session_{s}", 0.0)
        # vol_regime / trend_regime / decision_reason — same placeholder convention as training
        bar_state["vol_regime_MEDIUM"] = 1.0
        bar_state["trend_regime_TREND_NEUTRAL"] = 1.0
        bar_state["decision_reason_v2_inference_batch"] = 1.0

        # All remaining canonical_v3 + augmented columns as plain features
        for col, val in canonical_v3_row.items():
            if col in ("time",):
                continue
            if col in bar_state:
                continue  # already populated
            try:
                v = float(val)
                if not np.isfinite(v):
                    v = 0.0
            except (TypeError, ValueError):
                continue
            bar_state[col] = v

        return bar_state

    # ── inference ────────────────────────────────────────────────────

    def decide_for_trade(
        self,
        trade: TradeState,
        canonical_v3_row: pd.Series,
        v3_v8_out: dict[str, float] | None = None,
    ) -> tuple[ExitDeciderV12Recommendation, dict[str, Any]]:
        """One-shot helper: build bar_state + run decider.

        Returns (recommendation, bar_state_dict). The bar_state is
        returned so the caller can log it to the trade-bar journal
        for offline distillation / V12.3 training.
        """
        bar_state = self.build_bar_state(trade, canonical_v3_row, v3_v8_out)
        rec = self.decider.decide(bar_state)
        return rec, bar_state
