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
import os
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
# 2026-05-29: contract-driven default — reads the ACTIVE entry_iql bundle from
# PROJECT_STATE_artifacts.json via gx1_guards.load_decision_entry. The retired
# 5-seed PORTFOLIO_PLUS5 ensemble is gone; current cement is a single bundle.
def _resolve_default_entry_iql_entry() -> dict:
    # FAIL-CLOSED (2026-06-03 audit): NO hardcoded fallback dict. A guard failure MUST
    # propagate — the swallowed guard would serve a stale bundle/variant silently on the
    # next cement repoint (train/serve skew). Mirrors the already-correct Exit-IQL resolver.
    from gx1_guards.artifacts import load_decision_entry
    return load_decision_entry("entry_iql")

_DEFAULT_ENTRY = _resolve_default_entry_iql_entry()
DEFAULT_BUNDLE_DIR = Path(_DEFAULT_ENTRY["path"])
DEFAULT_VARIANT = _DEFAULT_ENTRY.get("active_variant", "R_WAIT_OPP_K96_LAM50")
DEFAULT_FOLD = (_DEFAULT_ENTRY.get("active_folds") or ["FOLD_1"])[0]
DEFAULT_AGGREGATOR = "mean"
# Ensemble was retired with the PLUS5 cleanup. load_default() now uses a single
# bundle (the current cement). Keep an empty list so any callers expecting the
# name still find an iterable.
ENSEMBLE_BUNDLE_DIRS: list[Path] = []
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

# Conviction-gate (entry-selection #1, gate-validated 2026-06-10: ~2x total edge, win held 0.953,
# AND lower cap-3 DD than LAM50 −201 vs −734). When GX1_CONVICTION_GATE=1 the entry OPENS by raw
# advantage: TAKE if raw_adv = best-TAKE-Q − SKIP-Q (UN-clipped) >= GX1_CONVICTION_THR (default −34.2
# = the offline conviction20 20th-pctile), side = argmax(TAKE-Q) — overriding the IQL's conservative
# SKIP. OFF by default (= legacy LAM50, advantage>0 ~8% take). Reproduces the offline conviction20
# decisions on the same Q (test==serve). REQUIRES skip-ASIA (GX1_SKIP_ASIA) live to clear the per-year floor.
CONVICTION_GATE_ON = os.environ.get("GX1_CONVICTION_GATE", "0") == "1"
CONVICTION_THR = float(os.environ.get("GX1_CONVICTION_THR", "-34.2"))

# STEP-1 reversible serve-time overlays (2026-06-11, default OFF = no-op; the feature cols read 0.0 when the
# candidate lacks them, so live is byte-unchanged until BOTH the feature flag is built into the prebuilt AND
# the overlay flag is set). Calibrate thresholds via the OOT A/B gate (repaired-April kept).
RECLAIM_GATE_ON = os.environ.get("GX1_SMC_RECLAIM_GATE", "0") == "1"          # skip a TAKE into an un-reclaimed falling knife
RECLAIM_GATE_MIN = float(os.environ.get("GX1_SMC_RECLAIM_MIN", "0.5"))         # min opposite-side reclaim displacement to veto
ROUND_WALL_ON = os.environ.get("GX1_ROUND_NUMBER_WALL", "0") == "1"           # penalize low-conviction TAKE into adverse $-wall
ROUND_WALL_NEAR_ATR = float(os.environ.get("GX1_ROUND_WALL_NEAR_ATR", "0.25"))  # "near a wall" if |dist| < this (ATR)
ROUND_WALL_EXTRA = float(os.environ.get("GX1_ROUND_WALL_EXTRA", "20.0"))       # extra conviction required to take into a wall

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
        # 2026-06-03 audit: the 5-seed ensemble was retired with the PLUS5 cleanup —
        # ENSEMBLE_BUNDLE_DIRS is always [] so the old try/except path raised+logged a
        # false WARNING on EVERY startup (masking real load errors) before falling here.
        # Single-bundle is the cement; load it directly. (ENSEMBLE_BUNDLE_DIRS kept for a
        # future re-promotion via load_ensemble.)
        if ENSEMBLE_BUNDLE_DIRS:
            return cls.load_ensemble(ENSEMBLE_BUNDLE_DIRS)
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
        }

        # 2026-05-29: unpack the 5 new V10 head arrays into scalar features
        # matching the Entry-IQL state schema. COSTFIX V10 emits these as lists
        # in v10_out; Entry-IQL was trained on them named v10_dip_0..v10_dip_17,
        # v10_forecast_0..3, v10_timing_0..11, v10_tail_risk_0..5, v10_vol_forecast_0..2
        # (43 cols, one-truth mapping in
        # gx1/scripts/materialize_inference_batch_candidates_v3_v1.py).
        # Without this unpack the adapter silent-zero-fills 43 features → severe
        # train/serve skew. Each missing array stays as zeros (matches the
        # candidate-gen fallback path).
        _V10_NEW_HEAD_EMIT = (
            ("dip_pred_v1", "v10_dip", 18),
            ("forecast_pred_v1", "v10_forecast", 4),
            ("timing_pred_v1", "v10_timing", 12),
            ("tail_risk_pred_v1", "v10_tail_risk", 6),
            ("vol_forecast_pred_v1", "v10_vol_forecast", 3),
        )
        # 2026-06-02 fix (audit MEDIUM-#1): make every slot EXPLICIT.
        # Previously when an aux head was missing, slots were left ABSENT and
        # the adapter silent-0-fills them with a coverage warning (which fires
        # only ONCE per unique missing-set). With this change, every slot is
        # explicitly written either with the real value or an explicit 0.0 +
        # a marker — so missing-head events surface as a structured log line
        # per poll and Entry-IQL state is reproducible.
        missing_heads = []
        for src_key, out_prefix, width in _V10_NEW_HEAD_EMIT:
            arr = v10_out.get(src_key)
            values: list[float] = []
            if arr is None:
                missing_heads.append(src_key)
            else:
                try:
                    values = [float(x) for x in (arr if hasattr(arr, "__len__") else [arr])]
                except (TypeError, ValueError):
                    missing_heads.append(src_key)
                    values = []
            for i in range(width):
                v = values[i] if i < len(values) else 0.0
                candidate[f"{out_prefix}_{i}"] = v
        if missing_heads and not getattr(self, "_v10_head_warn_set", None) == tuple(missing_heads):
            LOG.warning(
                f"[V10_AUX_HEAD_MISSING] heads={missing_heads} not in v10_out — "
                f"explicit 0-fill applied. Train/serve skew risk if this persists. "
                f"Check V10 bundle loaded variants."
            )
            self._v10_head_warn_set = tuple(missing_heads)

        # Re-open the dict so further candidate_v3 augmentation below still works.
        candidate.update({
            # Portfolio state features (#1 2026-05-21): populated by caller
            # from open trades. Default 0 = no open trades.
            # TRAIN/SERVE PARITY NOTE (2026-06-03, user vedtak = Option A): this LIVE
            # path counts ACTUAL open positions (canonical = matches the name
            # portfolio_n_open_*_at_decision). The TRAINING column was built as
            # candidate-density (~12-16 mid-week), so the cement model saw a skewed
            # input. Fix is at the TRAINING side at next Entry-IQL retrain (recompute
            # to real-open-positions); live stays as-is. See materialize_build_
            # candidate_forward_outcome_dataset_v1.py PORTFOLIO SIMULATION block.
            "portfolio_n_open_long_at_decision": float((portfolio_state or {}).get(
                "n_open_long", 0.0)),
            "portfolio_n_open_short_at_decision": float((portfolio_state or {}).get(
                "n_open_short", 0.0)),
            "portfolio_combined_pnl_bps_at_decision": float((portfolio_state or {}).get(
                "combined_pnl_bps", 0.0)),
            "portfolio_time_since_last_entry_min": float((portfolio_state or {}).get(
                "time_since_last_entry_min", 240.0)),
        })

        # Canonical_v3 + augment features. 2026-05-29: emit BOTH the raw column
        # name AND the `_canon_v1`-suffixed variant. Entry-IQL training mixed
        # conventions — some features kept their raw name (e.g. dip_confirmed_*_v3,
        # m15_*_v2), others got `_canon_v1` appended (e.g. _v1_atr14_canon_v1).
        # The dual-emit makes the adapter look up the right one without us having
        # to know per-feature which convention it expects.
        for col, val in augmented_cv3_row.items():
            if col in ("time",):
                continue
            try:
                v = float(val)
                if not np.isfinite(v):
                    v = 0.0
            except (TypeError, ValueError):
                continue
            candidate[col] = v
            # Only emit the _canon_v1 alias if the raw name doesn't already end
            # with another versioned suffix that Entry-IQL trained on directly.
            if not (col.endswith("_v3") or col.endswith("_v2")
                    or col.endswith("_canon_v1") or col.endswith("_canon_v2")
                    or col.endswith("_canon_v3")):
                candidate[f"{col}_canon_v1"] = v

        # One-hot categoricals (adapter looks up cat_col → cat_val match).
        # 2026-05-29: ALSO emit the explicit one-hot binary cols
        # `{cat_col}__{cat_val}` since Entry-IQL state names them that way.
        sid = int(augmented_cv3_row.get("session_id", 0) or 0)
        session_label = SESSION_ID_TO_LABEL.get(sid, "ASIA")
        candidate["session"] = session_label
        for v in ("ASIA", "EU", "OVERLAP", "US"):
            candidate[f"session__{v}"] = 1.0 if session_label == v else 0.0

        # H6 (2026-06-03 gap-register): real regime when GX1_REGIME_V4=1 (MIRROR of
        # materialize_inference_batch_candidates_v3_v1._{TREND,VOL}_REGIME_ID_TO_LABEL — keep in
        # sync). Else cement placeholders (regime-blind). Emit all 4 vol categories (the build
        # ONE_HOT expects EXTREME too; was only emitting 3 here = a latent mismatch).
        if os.environ.get("GX1_REGIME_V4", "0") == "1":
            _tr_id = int(augmented_cv3_row.get("trend_regime_id", 1) or 1)
            _vr_id = int(augmented_cv3_row.get("vol_regime_id", 2) or 2)
            vol_label = {0: "LOW", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "EXTREME"}.get(_vr_id, "MEDIUM")
            trend_label = {0: "TREND_DOWN", 1: "TREND_NEUTRAL", 2: "TREND_UP"}.get(_tr_id, "TREND_NEUTRAL")
        else:
            vol_label = DEFAULT_VOL_REGIME_LABEL
            trend_label = DEFAULT_TREND_REGIME_LABEL
        candidate["vol_regime"] = vol_label
        for v in ("LOW", "MEDIUM", "HIGH", "EXTREME"):
            candidate[f"vol_regime__{v}"] = 1.0 if vol_label == v else 0.0

        candidate["trend_regime"] = trend_label
        for v in ("TREND_UP", "TREND_NEUTRAL", "TREND_DOWN"):
            candidate[f"trend_regime__{v}"] = 1.0 if trend_label == v else 0.0

        candidate["decision_reason"] = DEFAULT_DECISION_REASON
        candidate[f"decision_reason__{DEFAULT_DECISION_REASON}"] = 1.0

        # 2026-05-29: a few Entry-IQL training features are aliases of explicit
        # build_candidate scalars or aren't present in cv3 (legacy canonical_v1
        # outputs the v3 prebuilt no longer carries). Emit best-effort aliases
        # so the adapter resolves them instead of zero-filling.
        candidate.setdefault("margin", candidate.get("margin_top1_top2", 0.0))

        return candidate

    # ── inference ────────────────────────────────────────────────────

    def predict(self, candidate: dict[str, Any]) -> EntryRecommendation:
        # 2026-06-11 GATE-HOIST FIX: the conviction-gate + STEP-1 overlays used to live ONLY in the
        # ensemble (>=2 adapters) branch while the live single-bundle cement early-returned
        # adapter.predict_one() — GX1_CONVICTION_GATE=1 was a silent NO-OP live. Both paths now
        # compute Q and flow through ONE shared selection block (gate → min_adv → overlays).
        from gx1.runtime.entry_iql_v2_adapter import EntryRecommendation, iql_core
        a0 = self.adapter
        if len(self.ensemble_adapters) <= 1:
            state_a = a0.build_state_vector(candidate)[None, :]   # (1, n_features)
            q_full_ens = a0.model.predict_q(state_a)              # (1, n_actions, n_K)
        else:
            # 2026-05-21 ensemble: average Q-per-K across all adapters.
            # 2026-05-24 DEFENSIVE: each adapter builds its OWN raw state vector and
            # runs its OWN predict_q (which applies seed-specific feature_means/stds).
            # Same raw vector across adapters when feature_names match (validated in
            # load_ensemble), but explicit per-adapter call makes the contract
            # self-documenting and refactor-safe.
            q_full_list = []
            state_a = None
            for a in self.ensemble_adapters:
                state_a = a.build_state_vector(candidate)[None, :]  # (1, n_features)
                q_full_list.append(a.model.predict_q(state_a))      # (1, n_actions, n_K)
            q_full_ens = np.mean(np.stack(q_full_list, axis=0), axis=0)
        # Aggregate across K with the primary adapter's settings (uniform across an ensemble) —
        # identical math to EntryIQLV2Adapter.predict, so flags-OFF output == predict_one.
        if a0.aggregator == "mean":
            q_agg = q_full_ens.mean(axis=2)
        elif a0.aggregator == "max":
            q_agg = q_full_ens.max(axis=2)
        else:
            q_agg = (q_full_ens * a0.k_weights[None, None, :]).sum(axis=2)
        a_id = int(q_agg.argmax(axis=1)[0])
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
        # Selection logic: conviction-gate (open by raw advantage, gate-validated) OR legacy min_adv filter.
        if CONVICTION_GATE_ON:
            # TAKE if the best TAKE's UN-clipped advantage over SKIP clears the calibrated threshold;
            # side = the higher-Q TAKE direction. Overrides the IQL's conservative argmax SKIP.
            raw_adv = best_take_q - float(q_agg[0, skip_id])
            if raw_adv >= CONVICTION_THR:
                _l, _s = iql_core.ACTION_TAKE_LONG_NOW_ID, iql_core.ACTION_TAKE_SHORT_NOW_ID
                a_id = _l if q_agg[0, _l] >= q_agg[0, _s] else _s
            else:
                a_id = skip_id
        elif a0.min_advantage_bps > 0.0 and a_id != skip_id and adv < a0.min_advantage_bps:
            a_id = skip_id
        # STEP-1 reversible overlays (default-OFF): falling-knife skip + round-number wall, applied to
        # whatever TAKE was chosen. Flag ON + missing feature cols = LOUD once-per-process warning
        # (rule 9: never a silent no-op); the 0.0 default only covers the flags-OFF byte-identity case.
        if a_id != skip_id and (RECLAIM_GATE_ON or ROUND_WALL_ON):
            _L, _S = iql_core.ACTION_TAKE_LONG_NOW_ID, iql_core.ACTION_TAKE_SHORT_NOW_ID
            if RECLAIM_GATE_ON:
                if "smc_sweep_reclaim_down_displacement_atr" not in candidate and not getattr(self, "_reclaim_cols_warned", False):
                    LOG.warning("[STEP1_OVERLAY] GX1_SMC_RECLAIM_GATE=1 but smc_sweep_reclaim_* cols are MISSING "
                                "from the candidate — the falling-knife gate is a NO-OP. Rebuild the cv3 prebuilt "
                                "with GX1_SMC_SWEEP_RECLAIM=1 before trusting this overlay.")
                    self._reclaim_cols_warned = True
                _rec_up = float(candidate.get("smc_sweep_reclaim_up_displacement_atr", 0.0))    # bullish reclaim
                _rec_dn = float(candidate.get("smc_sweep_reclaim_down_displacement_atr", 0.0))  # bearish reclaim
                if a_id == _L and _rec_dn > RECLAIM_GATE_MIN and _rec_up <= 0.0:
                    a_id = skip_id   # long into an active bearish reclaim, no bullish = catching a falling knife
                elif a_id == _S and _rec_up > RECLAIM_GATE_MIN and _rec_dn <= 0.0:
                    a_id = skip_id
            if a_id != skip_id and ROUND_WALL_ON:
                if "dist_to_round_50_atr" not in candidate and not getattr(self, "_round_cols_warned", False):
                    LOG.warning("[STEP1_OVERLAY] GX1_ROUND_NUMBER_WALL=1 but dist_to_round_* cols are MISSING "
                                "from the candidate — the round-wall veto is a NO-OP. Build the round-number "
                                "features (GX1_ROUND_NUMBER=1) into the candidate path first.")
                    self._round_cols_warned = True
                _raw = best_take_q - float(q_agg[0, skip_id])
                # PER-GRID veto (2026-06-11 fix): a wall vetoes only when the SAME grid's level is both
                # NEAR and ADVERSE. The old min(|d50|,|d100|) + (d50<0 OR d100<0) conflated the grids and
                # systematically vetoed longs sitting just above an xx50 support (every $100 level is also
                # a $50 level, so a far-adverse $100 always co-occurred with a near-favorable $50).
                if _raw < (CONVICTION_THR + ROUND_WALL_EXTRA):
                    for _col in ("dist_to_round_50_atr", "dist_to_round_100_atr"):
                        _d = float(candidate.get(_col, 0.0))   # signed: <0 = price below the level
                        _adverse_g = (_d < 0.0) if a_id == _L else (_d > 0.0)
                        if _adverse_g and abs(_d) < ROUND_WALL_NEAR_ATR:
                            a_id = skip_id   # low-conviction TAKE entering into an adverse $-wall
                            break
        # Recompute advantages for the FINAL action (a_id may have changed by the gate/filter above).
        chosen_q = float(q_agg[0, a_id])
        adv = chosen_q - float(q_agg[0, skip_id])
        adv_take = chosen_q - best_take_q
        return EntryRecommendation(
            action_id_v1=a_id,
            action_label_v1=iql_core.ACTION_LABELS_V1[a_id],
            q_per_action_v1=q_agg[0].copy(),
            q_per_action_per_k_v1=q_full_ens[0].copy(),
            advantage_over_skip_v1=adv,
            advantage_over_realized_v1=adv_take,
            confidence_softmax_v1=soft[0].copy(),
            aggregator_v1=a0.aggregator,
            k_horizons_v1=list(a0.model.k_horizons),
            variant_v1=a0.variant,
            fold_id_v1=a0.fold_id,
            feature_names_v1=list(a0.feature_names),
            state_v1=(state_a[0].copy() if state_a is not None else np.zeros(0, dtype=np.float32)),
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
