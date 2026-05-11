#!/usr/bin/env python3
"""V12 Entry-IQL v2 live inference wrapper.

Wraps EntryIQLV2Adapter with a state-builder that converts upstream
(canonical_v3 augmented + XGB v5 + V10 v3) outputs into the
165-feature candidate dict the adapter expects.

The Entry-IQL state contract (from
gx1/scripts/materialize_build_entry_iql_v2.py:140-247) breaks into:

  16 candidate features (XGB + V10 outputs + time):
    weekday_utc, hour_utc, atr_bps,
    p_long, p_short, p_flat, p_hat, margin, uncertainty_score,
    tradable_prob, mfe_first_n_pred, path_quality_pred, bad_path_prob,
    direction_logit_long, direction_logit_short, direction_logit_flat

  ~70 chunk_0 features with `_chunk0_v1` suffix:
    canonical_v3 + augmented features at decision time
    (same values the live pipeline produces)

  ~78 canon features with `_canon_v1` suffix:
    canonical_v3 features — in production training these came from a
    separate parquet, but they're the same per-bar values, so live
    inference uses identical numbers under both suffixes.

  1 derived: entropy_v1 = Shannon entropy of V10 direction softmax

  4 one-hot categoricals (handled by adapter via "cat_col__cat_val"):
    session__ASIA, vol_regime__MEDIUM, trend_regime__TREND_NEUTRAL,
    decision_reason__v2_inference_batch
    (only these 4 levels were frequent enough in training to survive
     into the trained model's feature set; live sets the underlying
     string values and the adapter handles the look-up)

Output is an EntryRecommendation:
    action_id_v1 ∈ {0=SKIP, 1=TAKE_LONG_NOW, 2=TAKE_SHORT_NOW}
    q_per_action_v1                ← Q-trio (skip, long, short)
    advantage_over_skip_v1         ← Q[chosen] - Q[SKIP]
    confidence_softmax_v1          ← softmax(beta·Q)

The cemented V12.1 Q-advantage filter (min_advantage_bps=15.1 = P70
sweet-spot per V9 validation) is applied by default — actions with
Q-adv below threshold are forced back to SKIP.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.runtime.entry_iql_v2_adapter import EntryIQLV2Adapter, EntryRecommendation

LOG = logging.getLogger("v12_entry_iql_live")

DEFAULT_BUNDLE_DIR = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/BUILD_ENTRY_IQL_V2_20260506T195420Z_LOCK"
)
DEFAULT_VARIANT = "R_NET_REAL"
DEFAULT_FOLD = "FOLD_1"
DEFAULT_AGGREGATOR = "mean"
DEFAULT_MIN_ADVANTAGE_BPS = 15.1   # V9 P70 sweet spot
DEFAULT_BETA = 1.0                  # softmax temperature for confidence

# Categorical label conventions from inference_batch_v3 (matches training):
SESSION_ID_TO_LABEL = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}
DEFAULT_VOL_REGIME_LABEL = "MEDIUM"           # always — placeholder in training
DEFAULT_TREND_REGIME_LABEL = "TREND_NEUTRAL"  # always
DEFAULT_DECISION_REASON = "v2_inference_batch"


@dataclass
class EntryIQLLiveInference:
    adapter: EntryIQLV2Adapter
    feature_names: list[str] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        bundle_dir: Path = DEFAULT_BUNDLE_DIR,
        variant: str = DEFAULT_VARIANT,
        fold_id: str = DEFAULT_FOLD,
        aggregator: str = DEFAULT_AGGREGATOR,
        beta: float = DEFAULT_BETA,
        min_advantage_bps: float = DEFAULT_MIN_ADVANTAGE_BPS,
        prefer_cuda: bool = True,
    ) -> "EntryIQLLiveInference":
        adapter = EntryIQLV2Adapter.load(
            artifact_root=Path(bundle_dir),
            variant=variant, fold_id=fold_id,
            aggregator=aggregator, beta=beta,
            min_advantage_bps=min_advantage_bps,
            prefer_cuda=prefer_cuda,
        )
        LOG.info(f"Entry-IQL v2 loaded: {bundle_dir.name}  variant={variant}  "
                  f"fold={fold_id}  min_adv_bps={min_advantage_bps}")
        return cls(adapter=adapter, feature_names=list(adapter.feature_names))

    @classmethod
    def load_default(cls) -> "EntryIQLLiveInference":
        return cls.load()

    # ── state construction ───────────────────────────────────────────

    def build_candidate(
        self,
        augmented_cv3_row: pd.Series,
        xgb_out: dict[str, Any],
        v10_out: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the 165-feature candidate dict for the adapter.

        Args:
            augmented_cv3_row: pd.Series — one row from augment_canonical_v3()
                indexed by feature name (i.e. .iloc[-1] from the DataFrame).
            xgb_out: dict with keys p_long, p_short, p_flat (scalar floats for
                this specific bar — caller picks idx and unwraps).
            v10_out: dict from V10LiveInference.predict() — has direction_probs,
                direction_logits, tradable_prob, mfe_first_n, path_quality,
                bad_path_prob, etc.

        Returns a candidate dict with all 165 feature-name keys covered, plus
        the one-hot underlying string columns (session/vol_regime/trend_regime/
        decision_reason).
        """
        # Derived V10 stats: p_hat, margin, uncertainty_score, entropy_v1
        dp = v10_out["direction_probs"]
        p_long = float(dp[0]); p_short = float(dp[1]); p_flat = float(dp[2])
        p_hat = float(max(p_long, p_short, p_flat))
        sorted_dir = sorted([p_long, p_short, p_flat], reverse=True)
        margin = float(sorted_dir[0] - sorted_dir[1])
        uncertainty_score = float(1.0 - p_hat)
        entropy_v1 = float(sum(
            -p * np.log(p) for p in (p_long, p_short, p_flat) if p > 1e-12
        ))

        # Timestamp components (decision-bar timestamp)
        ts = augmented_cv3_row.name  # DatetimeIndex element
        weekday_utc = int(ts.dayofweek)
        hour_utc = int(ts.hour)

        # 16 candidate features
        candidate: dict[str, Any] = {
            "weekday_utc": weekday_utc,
            "hour_utc": hour_utc,
            "atr_bps": float(augmented_cv3_row.get("atr_bps", 0.0) or 0.0),
            "p_long": p_long,
            "p_short": p_short,
            "p_flat": p_flat,
            "p_hat": p_hat,
            "margin": margin,
            "uncertainty_score": uncertainty_score,
            "tradable_prob": float(v10_out["tradable_prob"]),
            "mfe_first_n_pred": float(v10_out["mfe_first_n"]),
            "path_quality_pred": float(v10_out["path_quality"]),
            "bad_path_prob": float(v10_out["bad_path_prob"]),
            "direction_logit_long": float(v10_out["direction_logits"][0]),
            "direction_logit_short": float(v10_out["direction_logits"][1]),
            "direction_logit_flat": float(v10_out["direction_logits"][2]),
            "entropy_v1": entropy_v1,
        }

        # Canonical_v3 + augment features under BOTH _chunk0_v1 and _canon_v1
        # suffixes (in production training the two came from separate sources but
        # they represent the same per-bar feature values).
        for col, val in augmented_cv3_row.items():
            if col in ("time",):
                continue
            try:
                v = float(val)
                if not np.isfinite(v):
                    v = 0.0
            except (TypeError, ValueError):
                continue
            candidate[f"{col}_chunk0_v1"] = v
            candidate[f"{col}_canon_v1"] = v

        # One-hot categoricals (adapter looks up cat_col → cat_val match)
        sid = int(augmented_cv3_row.get("session_id", 0) or 0)
        candidate["session"] = SESSION_ID_TO_LABEL.get(sid, "ASIA")
        candidate["vol_regime"] = DEFAULT_VOL_REGIME_LABEL
        candidate["trend_regime"] = DEFAULT_TREND_REGIME_LABEL
        candidate["decision_reason"] = DEFAULT_DECISION_REASON

        return candidate

    # ── inference ────────────────────────────────────────────────────

    def predict(self, candidate: dict[str, Any]) -> EntryRecommendation:
        return self.adapter.predict_one(candidate)

    def predict_from_pipeline(
        self,
        augmented_cv3_row: pd.Series,
        xgb_out: dict[str, Any],
        v10_out: dict[str, Any],
    ) -> tuple[EntryRecommendation, dict[str, Any]]:
        """One-shot helper: build candidate + run inference.

        Returns (recommendation, candidate_dict) — the candidate dict is
        returned so the caller can log it in the journal for offline replay.
        """
        candidate = self.build_candidate(augmented_cv3_row, xgb_out, v10_out)
        rec = self.predict(candidate)
        return rec, candidate
