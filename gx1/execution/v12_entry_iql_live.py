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

  ~78 canon features with `_canon_v1` suffix:
    canonical_v3 + augmented features at decision time. The `_chunk0_v1`
    mirror was dropped 2026-05-21 — chunk_0_data parquet was missing in
    training so every chunk0_v1 slot was zero-filled (feature-importance
    analysis confirmed gain=0/perm=0 across the board).

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

# 2026-05-21: PORTFOLIO + PLUS5 ENSEMBLE — 182-feature Entry-IQL (173 v3+ aux +
# 4 portfolio + 5 PLUS5 canonical features that were 0-filled in earlier runs).
# 5-seed ensemble for epistemic robustness. Individual seeds (mean_test_reward_bps):
#   1337: +96,137  1338: +95,396  1339: +94,969  1340: +94,607  1341: +95,449
# Mean +95,312 bps. Live canonical pipeline now computes the 5 PLUS5 features
# via v12_canonical_incremental._compute_plus5_features (matches training).
#
# 🚨 2026-05-24 WARNING: BUNDLES DELETED IN CLEANUP TIER-A
# All 5 PORTFOLIO_PLUS5_SEED133[7-9, 0-1] bundles were deleted in disk cleanup.
# A LIVE RUNTIME RESTART WILL FAIL until either:
#   (a) These exact bundles are restored, OR
#   (b) DEFAULT_BUNDLE_DIR + ENSEMBLE_BUNDLE_DIRS are updated to the new winning
#       bundle (likely R_WAIT_OPP_K96_LAM20 + dip/struct + M15/D1 hi/lo) once
#       trained and validated.
# Currently-running paper-runner process holds weights in RAM; do NOT restart
# until new bundles are deployed.
DEFAULT_BUNDLE_DIR = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "BUILD_ENTRY_IQL_V3PLUS_PORTFOLIO_PLUS5_SEED1337_20260521T111046Z"
)
ENSEMBLE_BUNDLE_DIRS = [
    Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/BUILD_ENTRY_IQL_V3PLUS_PORTFOLIO_PLUS5_SEED1337_20260521T111046Z"),
    Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/BUILD_ENTRY_IQL_V3PLUS_PORTFOLIO_PLUS5_SEED1338_20260521T111046Z"),
    Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/BUILD_ENTRY_IQL_V3PLUS_PORTFOLIO_PLUS5_SEED1339_20260521T111046Z"),
    Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/BUILD_ENTRY_IQL_V3PLUS_PORTFOLIO_PLUS5_SEED1340_20260521T111046Z"),
    Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/BUILD_ENTRY_IQL_V3PLUS_PORTFOLIO_PLUS5_SEED1341_20260521T111046Z"),
]
DEFAULT_VARIANT = "R_NET_REAL"
DEFAULT_FOLD = "FOLD_1"
DEFAULT_AGGREGATOR = "mean"
# V12.2: filter off (0.0) to match Phase 6 validation. V12.1.1 used 15.1 (V9 P70)
# but V12.2 Q-adv distribution shifted lower (p70=10.2) since more candidates pass.
DEFAULT_MIN_ADVANTAGE_BPS = 0.0    # 2026-05-21: replaced static threshold with
                                   # adaptive threshold in v12_paper_runner.py
                                   # (min_adv = ADAPTIVE_MIN_ADV_ATR_MULT * ATR_bps).
                                   # The static 30 bps was based on Q-spike data
                                   # before z-clamp fix; with clean Q-values the
                                   # adv-distribution is much tighter (5-15 bps in
                                   # low-vol, 20-50 bps in high-vol).
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
    # 2026-05-21 ensemble support: when len > 1, predict() averages Q across all.
    ensemble_adapters: list[EntryIQLV2Adapter] = field(default_factory=list)

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
        # 2026-05-21: default = 5-seed ensemble. Fall back to single bundle if
        # ensemble dirs unavailable (e.g., legacy systems or testing).
        try:
            for d in ENSEMBLE_BUNDLE_DIRS:
                if not d.exists():
                    raise FileNotFoundError(f"ensemble bundle missing: {d}")
            return cls.load_ensemble(ENSEMBLE_BUNDLE_DIRS)
        except Exception as exc:
            LOG.warning(f"ensemble load failed: {exc}; falling back to single bundle")
            return cls.load()

    @classmethod
    def load_ensemble(
        cls,
        bundle_dirs: list[Path],
        variant: str = DEFAULT_VARIANT,
        fold_id: str = DEFAULT_FOLD,
        aggregator: str = DEFAULT_AGGREGATOR,
        beta: float = DEFAULT_BETA,
        min_advantage_bps: float = DEFAULT_MIN_ADVANTAGE_BPS,
        prefer_cuda: bool = True,
    ) -> "EntryIQLLiveInference":
        """Load K Entry-IQL bundles for ensemble inference (#12 improvement).

        Each bundle was trained with different seed → slightly different Q.
        predict() runs each adapter and averages Q-values. Reduces single-model
        overfit and gives uncertainty estimate via Q-std across ensemble.
        """
        if not bundle_dirs:
            raise ValueError("load_ensemble: bundle_dirs cannot be empty")
        adapters = []
        for bd in bundle_dirs:
            adapters.append(EntryIQLV2Adapter.load(
                artifact_root=Path(bd),
                variant=variant, fold_id=fold_id,
                aggregator=aggregator, beta=beta,
                min_advantage_bps=min_advantage_bps,
                prefer_cuda=prefer_cuda,
            ))
        LOG.info(f"[ENSEMBLE] loaded {len(adapters)} Entry-IQL adapters")
        # feature_names should match across ensemble (same training data)
        first_features = adapters[0].feature_names
        for i, a in enumerate(adapters[1:], 1):
            if a.feature_names != first_features:
                raise ValueError(
                    f"Ensemble feature mismatch: bundle 0 has {len(first_features)} "
                    f"features, bundle {i} has {len(a.feature_names)}"
                )
        return cls(
            adapter=adapters[0],   # primary for build_candidate compatibility
            feature_names=first_features,
            ensemble_adapters=adapters,
        )

    # ── state construction ───────────────────────────────────────────

    def build_candidate(
        self,
        augmented_cv3_row: pd.Series,
        xgb_out: dict[str, Any],
        v10_out: dict[str, Any],
        portfolio_state: dict[str, float] | None = None,
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

        # 16 base candidate features + 5 v3+ aux head features
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
            # V10 v3+ aux head features — Entry-IQL trained with these (173 features).
            # Without them the model receives 0-fills for the 5 cols and collapses
            # to a constant action (observed 2026-05-19: all-LONG bias).
            # Defaults aligned with Exit-IQL build_v10_entry_snapshot_features
            # (all 0.0) — audit 3 H-3 fix 2026-05-20.
            "tf_agreement_pred": float(v10_out.get("tf_agreement_pred", 0.0)),
            "path_quality_log_var": float(v10_out.get("path_quality_log_var", 0.0)),
            "path_quality_std": float(v10_out.get("path_quality_std", 0.0)),
            "position_size_pred": float(v10_out.get("position_size_pred", 0.0)),
            "hold_horizon_pred": float(v10_out.get("hold_horizon_pred", 0.0)),
            # Portfolio state features (#1 2026-05-21): populated by caller
            # from open trades. Default 0 = no open trades (matches training
            # candidate at start of week with no prior entries).
            "portfolio_n_open_long_at_decision": float((portfolio_state or {}).get(
                "n_open_long", 0.0)),
            "portfolio_n_open_short_at_decision": float((portfolio_state or {}).get(
                "n_open_short", 0.0)),
            "portfolio_combined_pnl_bps_at_decision": float((portfolio_state or {}).get(
                "combined_pnl_bps", 0.0)),
            "portfolio_time_since_last_entry_min": float((portfolio_state or {}).get(
                "time_since_last_entry_min", 240.0)),
        }

        # Canonical_v3 + augment features under the _canon_v1 suffix only.
        # 2026-05-21: _chunk0_v1 mirror dropped — those state-vector slots were
        # constants in training (chunk_0_data parquet was missing) and the
        # corresponding NUMERIC_STATE_COLS_CHUNK0 list is now empty, so the
        # adapter will not look them up.
        for col, val in augmented_cv3_row.items():
            if col in ("time",):
                continue
            try:
                v = float(val)
                if not np.isfinite(v):
                    v = 0.0
            except (TypeError, ValueError):
                continue
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
        # 2026-05-21 ensemble: when multiple adapters loaded, average Q-values
        # across all and use combined Q for final decision. Single-adapter
        # codepath unchanged for backwards compat.
        if len(self.ensemble_adapters) <= 1:
            return self.adapter.predict_one(candidate)
        # Get full Q-per-K from each adapter, then average.
        # 2026-05-24 DEFENSIVE: each adapter builds its OWN raw state vector and
        # runs its OWN predict_q (which applies seed-specific feature_means/stds).
        # Same raw vector across adapters when feature_names match (validated in
        # load_ensemble), but explicit per-adapter call makes the contract
        # self-documenting and refactor-safe.
        q_full_list = []
        for a in self.ensemble_adapters:
            state_a = a.build_state_vector(candidate)[None, :]  # (1, n_features)
            q_full_list.append(a.model.predict_q(state_a))      # (1, n_actions, n_K)
        q_full_ens = np.mean(np.stack(q_full_list, axis=0), axis=0)
        # Use primary adapter's aggregator/beta/min_adv to convert Q → action
        # (all adapters share the same training config so settings are uniform).
        a0 = self.adapter
        if a0.aggregator == "mean":
            q_agg = q_full_ens.mean(axis=2)
        elif a0.aggregator == "max":
            q_agg = q_full_ens.max(axis=2)
        else:
            q_agg = (q_full_ens * a0.k_weights[None, None, :]).sum(axis=2)
        actions = q_agg.argmax(axis=1)
        a_id = int(actions[0])
        # Compute std across ensemble per action for uncertainty
        q_std_per_action = np.stack(q_full_list, axis=0).mean(axis=-1).std(axis=0)[0]
        # Build a recommendation using primary adapter for label/conf logic but
        # with averaged Q.
        from gx1.runtime.entry_iql_v2_adapter import EntryRecommendation, iql_core
        scaled = a0.beta * q_agg
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        soft = np.exp(scaled); soft = soft / soft.sum(axis=1, keepdims=True)
        skip_id = iql_core.ACTION_SKIP_ID
        chosen_q = float(q_agg[0, a_id])
        adv = chosen_q - float(q_agg[0, skip_id])
        # take_now advantage: chosen vs better of TAKE_LONG / TAKE_SHORT
        best_take_q = float(max(q_agg[0, iql_core.ACTION_TAKE_LONG_NOW_ID],
                                 q_agg[0, iql_core.ACTION_TAKE_SHORT_NOW_ID]))
        adv_take = chosen_q - best_take_q
        # min_advantage_bps filter applied by ensemble too
        if a0.min_advantage_bps > 0.0 and a_id != skip_id and adv < a0.min_advantage_bps:
            a_id = skip_id
        # Note: q_std_per_action available via q_std_per_action variable but
        # not added to EntryRecommendation dataclass to keep schema stable.
        # Could be logged separately when needed for uncertainty-aware sizing.
        return EntryRecommendation(
            action_id_v1=a_id,
            action_label_v1=iql_core.ACTION_LABELS_V1[a_id],
            q_per_action_v1=q_agg[0].copy(),
            q_per_action_per_k_v1=q_full_ens[0].copy(),
            advantage_over_skip_v1=adv,
            advantage_over_realized_v1=adv_take,
            confidence_softmax_v1=soft[0],
            aggregator_v1=a0.aggregator,
            k_horizons_v1=list(a0.model.k_horizons),
            variant_v1=a0.variant,
            fold_id_v1=a0.fold_id,
            feature_names_v1=list(a0.feature_names),
            state_v1=states[0],
        )

    def predict_from_pipeline(
        self,
        augmented_cv3_row: pd.Series,
        xgb_out: dict[str, Any],
        v10_out: dict[str, Any],
        portfolio_state: dict[str, float] | None = None,
    ) -> tuple[EntryRecommendation, dict[str, Any]]:
        """One-shot helper: build candidate + run inference.

        Returns (recommendation, candidate_dict) — the candidate dict is
        returned so the caller can log it in the journal for offline replay.
        """
        candidate = self.build_candidate(
            augmented_cv3_row, xgb_out, v10_out,
            portfolio_state=portfolio_state,
        )
        rec = self.predict(candidate)
        return rec, candidate
