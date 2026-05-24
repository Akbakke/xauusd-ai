#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────
# RUNBOOK: /home/andre2/GX1_DATA/V12_4_RUNBOOK.md  ← seksjon 1 (Strategy F-reglene)
# ─────────────────────────────────────────────────────────────────────
"""V12 Exit-IQL V12.1 live inference wrapper — the per-bar exit-decision
loop for open trades.

Wraps ExitDeciderV12Adapter (cemented V12.1.1 NO_TRAIL config) with a
state-builder that converts:
  - TradeState running stats
  - V10 entry-snapshot (frozen at trade open)
  - V3 v8 outputs at this bar (currently STUBBED — see scope note)
  - Augmented canonical_v3 features at this bar
  - One-hot side flags (long/short)

…into the bar_state dict the adapter expects. (Feature count varies by
trained bundle — the adapter looks up names from its own feature list.
2026-05-21 prune removed all ~55 chunk0_v1 mirror cols + 9 dead canon_v1
cols, so the next retrained bundle will see a ~140-feature contract.)

V12.2 cement config (per project_gx1_v12_2_cemented_2026q2):
  - variant: R_V12  (V12.2 retrain on V3 v9 multi-TF outputs)
  - fold_id: FOLD_1
  - v3_override_threshold: None  (V3 fail-safe DISABLED — Phase 6
    validated V12_OFF +73.64 bps beats V12_ON +70.60 bps at 0.95
    threshold. V3 outputs feed Q-learning state but never trigger
    overrides directly.)

V3 v9 outputs ARE produced live via V3LiveInference (multi-TF). The 4
V3-tracking features (should_exit_decision, decision_confidence,
max_prob_in_trade, consecutive_exits etc.) are computed in TradeState
from per-bar V3 v9 inference. Q-values match training distribution.

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

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.runtime.exit_iql_v2_adapter import (
    ACTION_HOLD_ID as EXIT_HOLD_ID_V2,
    ACTION_EXIT_NOW_ID as EXIT_NOW_ID_V2,
    ACTION_LABELS_EXIT as EXIT_LABELS_V2,
)
from gx1.runtime.exit_decider_v12_adapter import (
    EXIT_NOW_ID,
    ExitDeciderV12Adapter,
    ExitDeciderV12Recommendation,
)
from gx1.execution.v12_trade_state import TradeState, DEFAULT_V3_FEATURES


# ══════════════════════════════════════════════════════════════════════════
# V12.4 CEMENT — STRATEGY F (LOCKED, NO ABLATION)
# ══════════════════════════════════════════════════════════════════════════
# Cemented 2026-05-16. Backtest 84K cands × 277 weeks: +134 bps eqv (+185%
# over V12.2 baseline). OOS-validated split test confirmed robustness.
#
# THREE COMBINED RULES enforced unconditionally:
#   (1) PROFIT-LOCK:    MFE ≥ 15 AND drawdown ≥ 30% × MFE   → EXIT
#   (2) BREAK-EVEN-CUT: MFE ≥ 10 AND pnl < 30% × MFE        → EXIT
#   (3) STRONG-HOLD:    IQL Q_adv < -200                    → SUPPRESS 1+2
#
# Older policies (V12.1 / V12.2 baseline / V12.3 partial overlay / V13 native)
# are SUPERSEDED — DELIBERATELY HARDCODED so no env-var or CLI can revert.
# To run a deliberate ablation in research mode, git-checkout an older sha.
MFE_GIVEBACK_ENABLED = True            # V12.4 LOCKED
STRATEGY_F_ENABLED = True              # V12.4 LOCKED
MFE_GIVEBACK_PCT = 0.30                # tuned via 84K-cand backtest
# Audit + counterfactual replay 2026-05-21: live LONGs have +43 bps mean 4h
# terminal, but realized was -10 to -30 bps because profit_lock fired at
# MFE>=15 + 30% giveback, capping winners far below their 4h potential.
# Raised to 30 to let winners develop closer to their counterfactual edge.
MFE_GIVEBACK_MIN_MFE_BPS = 30.0
BREAKEVEN_RATIO = 0.30
BREAKEVEN_MIN_MFE = 10.0
STRONG_HOLD_QADV = -200.0              # tuned via param sweep — top decile of Q-adv
# V10 v3+ Target 4: hold-horizon-expired override.
# When V10 v3+ bundle is loaded, it predicts expected trade hold in bars.
# If trade exceeds K × prediction AND MFE never reached profit-lock floor,
# force EXIT_NOW with reason HOLD_HORIZON_EXPIRED — cuts stale grinders.
# Set K = 1.5 (50% margin over model's expectation).
HOLD_HORIZON_OVERRUN_MULT = 1.5
# Sentinel: V10 returns -1 when bundle has no hold_horizon head.
HOLD_HORIZON_INVALID_SENTINEL = 0.0
# Audit 3 H-1 fix 2026-05-20: V10 hold_horizon head outputs sigmoid → bars via
# round(p * 1440). Typical predictions are tiny (0.01-0.10 → 14-144 bars). At
# the low end, MULT*pred could fire from bar 4 — kills new trades before they
# can develop. Floor prevents this.
HOLD_HORIZON_MIN_FLOOR_BARS = 60

# Hard guard: refuse to load if user explicitly tries to disable via env.
# This is a NO-CONFIG-DRIFT enforcement — older policies were retired.
for _retired_env in (
    "GX1_MFE_GIVEBACK_ENABLED", "GX1_STRATEGY_F_ENABLED",
    "GX1_MFE_GIVEBACK_PCT", "GX1_MFE_GIVEBACK_MIN_MFE_BPS",
    "GX1_BREAKEVEN_RATIO", "GX1_BREAKEVEN_MIN_MFE", "GX1_STRONG_HOLD_QADV",
):
    if _retired_env in os.environ:
        raise RuntimeError(
            f"V12.4_HARD_LOCKED: env var {_retired_env!r} is no longer honored. "
            f"V12.4 policy is the only supported config. Remove the env var "
            f"or git-checkout an older revision for ablation runs."
        )

LOG = logging.getLogger("v12_exit_iql_live")

# V12.4-cement Exit-IQL bundle. The Exit-IQL TRAINING is identical to V12.2's
# (variant R_V12, FOLD_1). V12.4 differs from V12.2 ONLY in the post-IQL
# Strategy-F overlay above, NOT in the underlying model. R_V13_MFE_AWARE
# was tested but RETIRED (OOS-overfit).
# 2026-05-19: cement replaced with v3+ chain (Entry-IQL retrained on V10 v3+ aux
# heads + Exit-IQL retrained on v3+ Entry-IQL decisions). Phase 6 OOT (7,500 sub-
# sample): +70.92 bps/cand vs prior +73.64 (-2.72 in noise band; full self-consistency
# across all 4 components is the primary value).
# 2026-05-21: Z-CLAMP 8M sample retrain — winsorized stats + ±5σ z-score clamp.
# +49.80M bps test reward (vs winsorized 6M +37.61M = +32% improvement from
# larger sample + clamp-bounded gradients). Z-clamp prevents live Q-spikes from
# out-of-distribution features.
#
# 🚨 2026-05-24 WARNING: ALL EXIT-IQL BUNDLES DELETED IN CLEANUP
# Tier B + later kills removed every Exit-IQL bundle from disk, including this
# V12.4-approved one. A LIVE RESTART WILL FAIL load_default() with hard-lock
# RuntimeError on line ~227.
# To recover:
#   1. Retrain Exit-IQL on the new V3PLUS_FIXED_FULL per-bar dataset (with
#      M15/D1 dist features + SMC alive + ema_stack ffill fixes), then
#   2. Update V12_4_APPROVED_BUNDLE to the new bundle name, or
#   3. Replace hard-lock with dynamic "latest valid" lookup.
# Currently-running paper-runner process holds weights in RAM; do NOT restart.
V12_4_APPROVED_BUNDLE = (
    "BUILD_EXIT_IQL_PER_BAR_DATASET_V12_V3PLUS_FULL_20260519T012648Z_LOCK_"
    "V3TRACKED_20260519T022946Z_LOCK_ZCLAMP8M_TRAINED_20260520T190905Z_LOCK"
)
V12_4_APPROVED_VARIANT = "R_V12"   # V13's R_V13_MFE_AWARE was retired
V12_4_APPROVED_FOLD = "FOLD_1"

DEFAULT_BUNDLE_DIR = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/" + V12_4_APPROVED_BUNDLE
)
DEFAULT_VARIANT = V12_4_APPROVED_VARIANT
DEFAULT_FOLD = V12_4_APPROVED_FOLD
DEFAULT_V3_OVERRIDE = None         # V12_OFF — V3 override never helped, retired

SESSION_ID_TO_LABEL = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}


def _exit_rec_with_distilled_q(rec, v3_v8_out: dict[str, float]):
    """Rebuild ExitDeciderV12Recommendation using V3 distilled q_head.

    Replaces iql_recommendation_v1 with one built from v3_q_hold_v1/v3_q_exit_v1.
    If the original decision_source was IQL_Q, the outer action is re-derived
    from the distilled Q (argmax). A V3_OVERRIDE source is preserved — we only
    swap the underlying iql_rec for diagnostic visibility.
    """
    from gx1.runtime.exit_iql_v2_adapter import ExitRecommendation
    q_hold = float(v3_v8_out["v3_q_hold_v1"])
    q_exit = float(v3_v8_out["v3_q_exit_v1"])
    q_vec = np.asarray([q_hold, q_exit], dtype=np.float32)
    distilled_action = EXIT_NOW_ID_V2 if q_exit > q_hold else EXIT_HOLD_ID_V2
    inner = rec.iql_recommendation_v1
    # Build a new ExitRecommendation with distilled Q. Keep variant/feature_names
    # for traceability so journals show where Q came from.
    new_inner = dataclasses.replace(
        inner,
        action_id_v1=distilled_action,
        action_label_v1=EXIT_LABELS_V2[distilled_action],
        q_per_action_v1=q_vec.copy(),
        q_per_action_per_k_v1=np.broadcast_to(q_vec[:, None],
                                              (2, len(inner.k_horizons_v1))).copy(),
        advantage_exit_over_hold_v1=q_exit - q_hold,
        confidence_softmax_v1=_softmax2(q_vec),
        variant_v1=f"{inner.variant_v1}+DISTILLED_V3_QHEAD",
    )
    # If the outer decider chose IQL_Q, propagate the distilled action upward.
    # If it chose V3_OVERRIDE, leave outer untouched (override still wins).
    if rec.decision_source_v1 == "IQL_Q":
        return dataclasses.replace(
            rec,
            action_id_v1=distilled_action,
            action_label_v1=EXIT_LABELS_V2[distilled_action],
            decision_source_v1="DISTILLED_V3_QHEAD",
            iql_recommendation_v1=new_inner,
        )
    return dataclasses.replace(rec, iql_recommendation_v1=new_inner)


def _softmax2(q: np.ndarray) -> np.ndarray:
    s = q - q.max()
    e = np.exp(s)
    return (e / e.sum()).astype(np.float32)


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
        # V12.4 hard lockdown — refuse non-cement configs.
        if Path(bundle_dir).name != V12_4_APPROVED_BUNDLE:
            raise RuntimeError(
                f"V12.4_HARD_LOCKED: bundle_dir name {Path(bundle_dir).name!r} "
                f"does not match V12.4-approved {V12_4_APPROVED_BUNDLE!r}. "
                f"Retired bundles (V12.1, R_V13_MFE_AWARE, etc.) cannot be loaded."
            )
        if variant != V12_4_APPROVED_VARIANT:
            raise RuntimeError(
                f"V12.4_HARD_LOCKED: variant {variant!r} != approved {V12_4_APPROVED_VARIANT!r}. "
                f"R_V13_MFE_AWARE was retired due to OOS-overfit."
            )
        if fold_id != V12_4_APPROVED_FOLD:
            raise RuntimeError(
                f"V12.4_HARD_LOCKED: fold_id {fold_id!r} != approved {V12_4_APPROVED_FOLD!r}."
            )
        if v3_override_threshold is not None:
            raise RuntimeError(
                f"V12.4_HARD_LOCKED: v3_override_threshold must be None (V12_OFF). "
                f"V3 override was retired in V12.2 cement."
            )
        decider = ExitDeciderV12Adapter.load(
            artifact_root=Path(bundle_dir),
            variant=variant, fold_id=fold_id,
            v3_override_threshold=v3_override_threshold,
            prefer_cuda=prefer_cuda,
        )
        feature_names = list(decider.iql_adapter.feature_names)
        LOG.info(f"Exit-IQL V12.4 loaded: {bundle_dir.name}  variant={variant}  "
                 f"strategy_F=ON  v3_override=disabled")
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
        current_m1_atr_bps_override: float | None = None,
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
        # Candidate context (constant during trade) — Exit-IQL training expects
        # 12 unsuffixed candidate-context features (NUMERIC_STATE_COLS_CANDIDATE
        # in materialize_build_exit_iql_v2.py:122-134). These were silent 0-fills
        # in live before 2026-05-19.
        s = trade.v10_snapshot
        dp = s.get("direction_probs", [0.0, 0.0, 0.0])
        p_long_e = float(dp[0] if hasattr(dp, "__len__") else 0.0)
        p_short_e = float(dp[1] if hasattr(dp, "__len__") and len(dp) > 1 else 0.0)
        p_flat_e = float(dp[2] if hasattr(dp, "__len__") and len(dp) > 2 else 0.0)
        p_hat_e = max(p_long_e, p_short_e, p_flat_e)
        sorted_probs = sorted([p_long_e, p_short_e, p_flat_e], reverse=True)
        margin_e = sorted_probs[0] - sorted_probs[1]
        uncertainty_e = 1.0 - p_hat_e
        # atr_bps at entry — fallback to last_atr_bps if v10_snapshot lacks entry atr
        atr_bps_entry = float(s.get("atr_bps", trade.last_atr_bps or 0.0))
        bar_state.update({
            "weekday_utc": float(trade.entry_ts.dayofweek),
            "hour_utc": float(trade.entry_ts.hour),
            "atr_bps": atr_bps_entry,
            "p_long": p_long_e,
            "p_short": p_short_e,
            "p_flat": p_flat_e,
            "p_hat": p_hat_e,
            "margin": margin_e,
            "uncertainty_score": uncertainty_e,
            "tradable_prob": float(s.get("tradable_prob", 0.0)),
            "mfe_first_n_pred": float(s.get("mfe_first_n", 0.0)),
            "path_quality_pred": float(s.get("path_quality", 0.0)),
        })
        # V3 v8 outputs (stubbed if not provided)
        v3_block = v3_v8_out if v3_v8_out is not None else DEFAULT_V3_FEATURES
        bar_state.update({k: float(v) for k, v in v3_block.items()})
        # V3-tracking running stats — Exit-IQL training expects 7 features:
        # v3_should_exit_decision_v1, v3_decision_confidence_v1,
        # v3_max_prob_in_trade_v1, v3_consecutive_exits_v1,
        # v3_signal_acceleration_v1, v3_total_exit_decisions_v1,
        # v3_max_consecutive_exits_v1. Without this call they were 0-filled
        # at decision time (fixed 2026-05-19).
        bar_state.update(trade.build_v3_tracking_features())
        # Side one-hot
        bar_state.update(trade.build_side_one_hot())
        # current_atr_bps
        # current_atr_bps_v1 = per-M1-bar (ask_high-bid_low)/mid bps (training:
        # materialize_build_exit_iql_per_bar_dataset_v1.py:215-217). Falls back
        # to canonical M5 ATR14 only when M1 reader unavailable (init / disk error).
        if current_m1_atr_bps_override is not None and current_m1_atr_bps_override > 0:
            bar_state["current_atr_bps_v1"] = float(current_m1_atr_bps_override)
        else:
            bar_state["current_atr_bps_v1"] = float(canonical_v3_row.get("atr_bps", 0.0) or 0.0)

        # m5_phase one-hots (from canonical_v3). Audit H3 2026-05-19: log a
        # one-time warning if none of the 5 phases are set — indicates the
        # prebuilt was built without the m5_phase augmenter and Exit-IQL is
        # seeing a constant 0-vector for these features.
        phase_sum = 0.0
        for p in range(5):
            col = f"m5_phase_{p}"
            val = float(canonical_v3_row.get(col, 0.0) or 0.0)
            bar_state[f"m5_phase_{p}_v1"] = val
            phase_sum += val
        if phase_sum < 1e-6 and not getattr(self, "_m5_phase_warned", False):
            import logging as _l
            _l.getLogger("exit_iql_live").warning(
                "[M5_PHASE_MISSING] canonical_v3_row has no m5_phase_{0..4} cols — "
                "Exit-IQL bar_state's 5 phase one-hots are all 0. Verify "
                "prebuilt was built with the m5_phase augmenter."
            )
            self._m5_phase_warned = True

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

        # Canonical_v3 + augment features under the _canon_v1 suffix only.
        # 2026-05-21: _chunk0_v1 mirror dropped — chunk_0_data parquet was
        # missing in training so every chunk0_v1 slot was zero-filled. With
        # the chunk0 list now empty in materialize_build_exit_iql_v2.py the
        # adapter does not look up the suffix, so we stop populating it.
        for col, val in canonical_v3_row.items():
            if col in ("time",):
                continue
            try:
                v = float(val)
                if not np.isfinite(v):
                    v = 0.0
            except (TypeError, ValueError):
                continue
            canon_key = f"{col}_canon_v1"
            if canon_key not in bar_state:
                bar_state[canon_key] = v

        return bar_state

    # ── inference ────────────────────────────────────────────────────

    def decide_for_trade(
        self,
        trade: TradeState,
        canonical_v3_row: pd.Series,
        v3_v8_out: dict[str, float] | None = None,
        current_m1_atr_bps_override: float | None = None,
    ) -> tuple[ExitDeciderV12Recommendation, dict[str, Any]]:
        """One-shot helper: build bar_state + run decider.

        Returns (recommendation, bar_state_dict). The bar_state is
        returned so the caller can log it to the trade-bar journal
        for offline distillation / V12.3 training.

        When GX1_MFE_GIVEBACK_ENABLED=1, applies Strategy-C MFE-giveback
        override: if (cum_mfe ≥ min_mfe) AND (drawdown_from_peak ≥ pct × mfe),
        force EXIT_NOW regardless of Exit-IQL action. This is the V12.3
        candidate validated to give +52% PnL equivalent via shorter trades.
        """
        bar_state = self.build_bar_state(
            trade, canonical_v3_row, v3_v8_out,
            current_m1_atr_bps_override=current_m1_atr_bps_override,
        )
        rec = self.decider.decide(bar_state)
        # Phase 3b A/B: when GX1_USE_DISTILLED_EXIT=1 and the V3 bundle exposes
        # a distilled q_head, swap the underlying iql_recommendation_v1 with one
        # built from v3_q_hold_v1/v3_q_exit_v1. Strategy F overlays below still
        # apply on top of the swapped baseline.
        if (
            os.environ.get("GX1_USE_DISTILLED_EXIT", "0") == "1"
            and v3_v8_out
            and "v3_q_hold_v1" in v3_v8_out
            and "v3_q_exit_v1" in v3_v8_out
        ):
            rec = _exit_rec_with_distilled_q(rec, v3_v8_out)

        if MFE_GIVEBACK_ENABLED:
            mfe = float(trade.cum_mfe_bps or 0.0)
            pnl = float(trade.current_pnl_bps or 0.0)
            drawdown = max(0.0, mfe - pnl)

            # Rule 1: Profit-lock — MFE peak with significant giveback
            profit_lock = (mfe >= MFE_GIVEBACK_MIN_MFE_BPS
                           and drawdown >= MFE_GIVEBACK_PCT * mfe
                           and mfe > 0)

            # Rule 2 (V12.4): Break-even-cut — trade drifting back to zero from peak
            breakeven_cut = (STRATEGY_F_ENABLED
                              and mfe >= BREAKEVEN_MIN_MFE
                              and pnl < BREAKEVEN_RATIO * mfe)

            f_trigger = profit_lock or breakeven_cut

            # Rule 3 (V12.4): Strong-hold override — let lottery winners ride
            # if IQL is VERY confident HOLD (Q_adv very negative).
            iql_q_adv = float(rec.iql_recommendation_v1.advantage_exit_over_hold_v1 or 0.0)
            strong_hold = (STRATEGY_F_ENABLED
                            and iql_q_adv < STRONG_HOLD_QADV)

            # Rule 4 (V10 v3+ Target 4): hold-horizon-expired — cuts stale trades
            # when realized hold exceeds K × model's predicted hold AND no
            # significant MFE accumulated. Only active when bundle has the
            # hold_horizon head (predicts > 0; legacy bundles return -1).
            hold_horizon_pred_bars = float(
                (trade.v10_snapshot or {}).get("hold_horizon_bars_pred", -1.0)
            )
            # Apply min-floor to prevent rule firing from bar 3-4 on
            # low-confidence predictions (audit 3 H-1 fix 2026-05-20).
            hold_horizon_effective_bars = max(
                hold_horizon_pred_bars, float(HOLD_HORIZON_MIN_FLOOR_BARS)
            ) if hold_horizon_pred_bars > HOLD_HORIZON_INVALID_SENTINEL else hold_horizon_pred_bars
            hold_horizon_expired = (
                STRATEGY_F_ENABLED
                and hold_horizon_pred_bars > HOLD_HORIZON_INVALID_SENTINEL
                and int(trade.bars_in_trade or 0) > int(HOLD_HORIZON_OVERRUN_MULT * hold_horizon_effective_bars)
                and mfe < MFE_GIVEBACK_MIN_MFE_BPS  # only cut if trade never built real edge
            )
            if hold_horizon_expired and not strong_hold:
                rec = ExitDeciderV12Recommendation(
                    action_id_v1=EXIT_NOW_ID,
                    action_label_v1="EXIT_NOW",
                    decision_source_v1="HOLD_HORIZON_EXPIRED",
                    v3_should_exit_prob_v1=rec.v3_should_exit_prob_v1,
                    iql_recommendation_v1=rec.iql_recommendation_v1,
                    override_threshold_v1=HOLD_HORIZON_OVERRUN_MULT,
                )
                return rec, bar_state

            if f_trigger and not strong_hold:
                # F override — exit now
                reason = ("BREAKEVEN_CUT" if breakeven_cut and not profit_lock
                          else "MFE_GIVEBACK_OVERRIDE")
                rec = ExitDeciderV12Recommendation(
                    action_id_v1=EXIT_NOW_ID,
                    action_label_v1="EXIT_NOW",
                    decision_source_v1=reason,
                    v3_should_exit_prob_v1=rec.v3_should_exit_prob_v1,
                    iql_recommendation_v1=rec.iql_recommendation_v1,
                    override_threshold_v1=MFE_GIVEBACK_PCT,
                )
        return rec, bar_state
