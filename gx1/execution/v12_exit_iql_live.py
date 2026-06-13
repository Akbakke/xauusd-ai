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
  - V3 fail-safe override REMOVED 2026-06-13 (Phase-6 had validated V12_OFF +73.64 bps > V12_ON
    +70.60 at the 0.95 threshold; it then fired 0/977 on May/June and is now deleted from the chain).
    V3 v9 outputs still feed the Exit-IQL Q-learning STATE (the 4 V3-tracking features below) — they
    just never trigger a direct override. Exit = Exit-IQL argmax + the Strategy-F overlay.

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
# STRATEGY F overlay — ABLATABLE (option A, 2026-05-25)
# ══════════════════════════════════════════════════════════════════════════
# Cemented 2026-05-16 (84K cands × 277 weeks: +134 bps eqv, +185% over V12.2).
# Previously HARD-LOCKED (no env could disable it). As of 2026-05-25 the
# overlay is ABLATABLE so Phase 6 OOT can compare "IQL alone" vs "IQL +
# overlay" on real edge — the principled path toward a pure learned policy.
#
#   (1) PROFIT-LOCK:    MFE ≥ min_mfe AND drawdown ≥ pct × MFE → EXIT
#   (2) BREAK-EVEN-CUT: MFE ≥ be_min  AND pnl < be_ratio × MFE → EXIT
#   (3) STRONG-HOLD:    IQL Q_adv < qadv                       → SUPPRESS 1+2
#   (4) HOLD-HORIZON-EXPIRED: bars > K × V10 hold-pred, low MFE → EXIT
#
# ALL DEFAULTS = the cemented V12.4 values, so behaviour is BIT-IDENTICAL
# unless a GX1_* env var overrides it. Master switch GX1_STRATEGY_F_ENABLED=0
# disables the whole overlay (pure Exit-IQL policy). The MFE-protection
# OBJECTIVE now also lives in the Exit-IQL reward (R_NET_REAL giveback penalty
# + peak-retention bonus), so the IQL can LEARN the behaviour the overlay
# hard-codes — and Phase 6 decides whether the overlay still adds edge.


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


# Master switch — turn the entire overlay off for the "IQL alone" ablation.
STRATEGY_F_ENABLED = _env_bool("GX1_STRATEGY_F_ENABLED", True)
# Back-compat alias; the outer gate is the master switch.
MFE_GIVEBACK_ENABLED = STRATEGY_F_ENABLED
MFE_GIVEBACK_PCT = _env_float("GX1_MFE_GIVEBACK_PCT", 0.30)
MFE_GIVEBACK_MIN_MFE_BPS = _env_float("GX1_MFE_GIVEBACK_MIN_MFE_BPS", 30.0)
BREAKEVEN_RATIO = _env_float("GX1_BREAKEVEN_RATIO", 0.30)
BREAKEVEN_MIN_MFE = _env_float("GX1_BREAKEVEN_MIN_MFE", 10.0)
STRONG_HOLD_QADV = _env_float("GX1_STRONG_HOLD_QADV", -200.0)
# V10 v3+ Target 4: hold-horizon-expired override (cuts stale grinders).
HOLD_HORIZON_OVERRUN_MULT = _env_float("GX1_HOLD_HORIZON_OVERRUN_MULT", 1.5)
# Sentinel: V10 returns -1 when bundle has no hold_horizon head.
HOLD_HORIZON_INVALID_SENTINEL = 0.0
# Floor so low hold-horizon predictions can't fire the rule from bar 3-4.
HOLD_HORIZON_MIN_FLOOR_BARS = _env_float("GX1_HOLD_HORIZON_MIN_FLOOR_BARS", 60)

LOG = logging.getLogger("v12_exit_iql_live")
LOG.info(
    "[STRATEGY_F] enabled=%s giveback_pct=%.2f min_mfe=%.1f be_ratio=%.2f "
    "be_min=%.1f strong_hold_qadv=%.1f hold_horizon_mult=%.2f floor=%.0f",
    STRATEGY_F_ENABLED, MFE_GIVEBACK_PCT, MFE_GIVEBACK_MIN_MFE_BPS, BREAKEVEN_RATIO,
    BREAKEVEN_MIN_MFE, STRONG_HOLD_QADV, HOLD_HORIZON_OVERRUN_MULT, HOLD_HORIZON_MIN_FLOOR_BARS,
)


def strategy_f_decision(
    mfe_bps: float,
    pnl_bps: float,
    iql_q_adv: float,
    hold_horizon_pred_bars: float,
    bars_in_trade: int,
    *,
    enabled: bool = True,
) -> tuple[bool, str]:
    """ONE-TRUTH Strategy-F overlay decision (the 4-rule post-IQL exit override).

    Returns (force_exit_now, reason). reason ∈ {'', 'HOLD_HORIZON_EXPIRED', 'BREAKEVEN_CUT',
    'MFE_GIVEBACK_OVERRIDE'}. Shared by the LIVE exit (build_bar_state_and_decide) AND the Phase-6
    gate (v12_phase6_joint_validation.simulate_one_candidate) so the gate scores the +Strategy-F
    policy live actually runs (2026-06-13 vedtak L7A — the cement had been gated on pure Exit-IQL +
    v3-override, but live forces ~55% of exits through this overlay). `enabled` lets a caller request
    a pure-IQL replay arm without the global GX1_STRATEGY_F_ENABLED switch.
    """
    if not (enabled and STRATEGY_F_ENABLED):
        return False, ""
    mfe = float(mfe_bps or 0.0)
    pnl = float(pnl_bps or 0.0)
    drawdown = max(0.0, mfe - pnl)
    # Rule 1: profit-lock (MFE peak with significant giveback)
    profit_lock = (mfe >= MFE_GIVEBACK_MIN_MFE_BPS and drawdown >= MFE_GIVEBACK_PCT * mfe and mfe > 0)
    # Rule 2: break-even-cut (drifting back to zero from peak)
    breakeven_cut = (mfe >= BREAKEVEN_MIN_MFE and pnl < BREAKEVEN_RATIO * mfe)
    f_trigger = profit_lock or breakeven_cut
    # Rule 3: strong-hold override — let lottery winners ride if IQL is VERY confident HOLD.
    strong_hold = (float(iql_q_adv or 0.0) < STRONG_HOLD_QADV)
    # Rule 4: hold-horizon-expired — cut stale grinders that never built edge.
    hold_pred = float(hold_horizon_pred_bars if hold_horizon_pred_bars is not None else -1.0)
    hold_eff = (max(hold_pred, float(HOLD_HORIZON_MIN_FLOOR_BARS))
                if hold_pred > HOLD_HORIZON_INVALID_SENTINEL else hold_pred)
    hold_horizon_expired = (
        hold_pred > HOLD_HORIZON_INVALID_SENTINEL
        and int(bars_in_trade or 0) > int(HOLD_HORIZON_OVERRUN_MULT * hold_eff)
        and mfe < MFE_GIVEBACK_MIN_MFE_BPS
    )
    if hold_horizon_expired and not strong_hold:
        return True, "HOLD_HORIZON_EXPIRED"
    if f_trigger and not strong_hold:
        return True, ("BREAKEVEN_CUT" if breakeven_cut and not profit_lock else "MFE_GIVEBACK_OVERRIDE")
    return False, ""

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
# 2026-05-29: retired the V12.4 bundle-name hard-lock. Live wiring now reads
# the ACTIVE Exit-IQL bundle (path + variant + fold + aggregator) from
# PROJECT_STATE_artifacts.json via gx1_guards.load_decision_entry("exit_iql").
# One truth: cement the bundle by editing the contract, not by editing constants.
# (2026-06-10: removed the dead _V12_4_LEGACY_BUNDLE / DEFAULT_BUNDLE_DIR fallback constants —
# verified unreferenced; load() resolves exclusively via load_decision_entry("exit_iql").)
DEFAULT_VARIANT = "R_NET_REAL"
DEFAULT_FOLD = "FOLD_1"
DEFAULT_AGGREGATOR = "max"
# (V3 fail-safe override fully REMOVED 2026-06-13 — never helped, fired 0/977; exit = Exit-IQL argmax + Strategy-F.)

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
        bundle_dir: Path | None = None,
        variant: str | None = None,
        fold_id: str | None = None,
        aggregator: str | None = None,
        prefer_cuda: bool = True,
    ) -> "ExitIQLLiveInference":
        """Load the live Exit-IQL inference. By default, reads bundle path +
        variant + fold + aggregator from the ACTIVE entry in
        PROJECT_STATE_artifacts.json via gx1_guards. Explicit kwargs override
        contract values when set (used by tests / shadow runs)."""
        # Resolve config from contract unless caller passed explicit overrides.
        if bundle_dir is None or variant is None or fold_id is None or aggregator is None:
            try:
                from gx1_guards.artifacts import load_decision_entry
            except ImportError as _e:
                raise RuntimeError(
                    "Exit-IQL load(): gx1_guards.artifacts not importable. "
                    "Either fix the import or pass explicit bundle_dir/variant/fold_id/aggregator."
                ) from _e
            entry = load_decision_entry("exit_iql")
            bundle_dir = bundle_dir if bundle_dir is not None else Path(entry["path"])
            variant = variant if variant is not None else entry.get("active_variant", DEFAULT_VARIANT)
            fold_id = fold_id if fold_id is not None else entry.get("active_folds", [DEFAULT_FOLD])[0]
            aggregator = aggregator if aggregator is not None else entry.get("active_aggregator", DEFAULT_AGGREGATOR)
        # V3 fail-safe override REMOVED 2026-06-13 (was retired/disabled since the V12.2 cement,
        # fired 0/977 on May/June). The exit is the Exit-IQL argmax + the Strategy-F overlay below.
        decider = ExitDeciderV12Adapter.load(
            artifact_root=Path(bundle_dir),
            variant=variant, fold_id=fold_id,
            aggregator=aggregator,
            prefer_cuda=prefer_cuda,
        )
        feature_names = list(decider.iql_adapter.feature_names)
        LOG.info(f"Exit-IQL loaded: {Path(bundle_dir).name}  "
                 f"variant={variant}  fold={fold_id}  aggregator={aggregator}  "
                 f"v3_override=disabled")
        LOG.info(f"  feature_names: {len(feature_names)}")
        # 2026-06-02 fix (audit MEDIUM-#3): validate that Exit-IQL's expected
        # feature_names include the V3-block keys we'll feed at decision time.
        # Without this, a retrained bundle that drops/renames V3 features would
        # silently 0-fill them via DEFAULT_V3_FEATURES (used when v3_v8_out=None
        # or when V3 inference fails). The check below catches contract drift
        # at load rather than surfacing it as silent train/serve skew.
        feat_set = set(feature_names)
        v3_block_keys = set(DEFAULT_V3_FEATURES.keys())
        missing_v3 = v3_block_keys - feat_set
        if missing_v3:
            LOG.warning(
                f"[EXIT_IQL_V3_BLOCK_PARTIAL] adapter feature_names missing {len(missing_v3)} "
                f"of {len(v3_block_keys)} V3-block features: {sorted(missing_v3)[:5]}. "
                f"Bar-state will write them but adapter will ignore them — train/serve OK "
                f"as long as this is intentional. Verify bundle's training-time V3 schema."
            )
        v3_track_keys = {"v3_max_prob_in_trade_v1", "v3_consecutive_exits_v1",
                          "v3_total_exit_decisions_v1", "v3_signal_acceleration_v1"}
        missing_track = v3_track_keys - feat_set
        if missing_track == v3_track_keys:
            LOG.warning(
                f"[EXIT_IQL_V3_TRACKING_MISSING] none of the V3-tracking features "
                f"(v3_max_prob_in_trade_v1 etc.) are in adapter — bundle was trained "
                f"without V3 tracking. Check materialize_build_exit_iql_v3_m1 version."
            )
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
        now_minute: "pd.Timestamp | None" = None,  # EX1: live M1 bar ts -> m5_phase = minute%5
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
        # EX1 (2026-06-04 train==serve parity): the per-bar TRAINER encodes m5_phase = minute%5
        # (compute_m5_phase_index — XGB-refresh STALENESS). The old serve read canonical_v3_row['m5_phase_{p}']
        # = cv3's minute//12 HOUR-segment bucket (materialize_build_canonical_features_v1) — a DIFFERENT
        # formula, and both have variance so the fail-closed coverage guard never caught it. Compute minute%5
        # from the live M1 bar ts when available; fall back to the cv3 col only if now_minute is absent.
        if now_minute is not None:
            from gx1.exits.contracts.exit_io_v3_ctx36_m1l512_phase5 import compute_m5_phase_onehot
            _m5_oh = compute_m5_phase_onehot(now_minute)
            for p in range(5):
                bar_state[f"m5_phase_{p}_v1"] = float(_m5_oh[p])
            phase_sum = 1.0
        else:
            # 2026-06-11: now_minute absent → fall back to the cv3 minute//12 m5_phase cols, which use a
            # DIFFERENT formula than the trainer's minute%5 (train≠serve on the phase block). Keep the fallback
            # for live stability (don't halt a trade) but WARN LOUD once so it is visible, not silent.
            if not getattr(self, "_m5_phase_fallback_warned", False):
                import logging as _l
                _l.getLogger("exit_iql_live").warning(
                    "[M5_PHASE_FALLBACK] now_minute absent → using cv3 minute//12 m5_phase cols (WRONG formula "
                    "vs trainer minute%5; train!=serve on the phase block). Pass now_minute for every live exit decision."
                )
                self._m5_phase_fallback_warned = True
            for p in range(5):
                col = f"m5_phase_{p}"
                val = float(canonical_v3_row.get(col, 0.0) or 0.0)
                bar_state[f"m5_phase_{p}_v1"] = val
                phase_sum += val
        if phase_sum < 1e-6 and not getattr(self, "_m5_phase_warned", False):
            import logging as _l
            _l.getLogger("exit_iql_live").warning(
                "[M5_PHASE_MISSING] no now_minute and canonical_v3_row has no m5_phase_{0..4} cols — "
                "Exit-IQL bar_state's 5 phase one-hots are all 0. Pass now_minute (the live M1 bar ts)."
            )
            self._m5_phase_warned = True

        # Categorical one-hots (matches training labels)
        sid = int(canonical_v3_row.get("session_id", 0) or 0)
        sess_label = SESSION_ID_TO_LABEL.get(sid, "ASIA")
        bar_state[f"session_{sess_label}"] = 1.0
        # Set zeros for other sessions
        for s in ("ASIA", "EU", "OVERLAP", "US"):
            bar_state.setdefault(f"session_{s}", 0.0)
        # vol_regime / trend_regime — real per-bar regime when GX1_REGIME_V4=1 (gap-register H6,
        # exit side: Exit-IQL was regime-blind via these const placeholders). Mirror of the
        # Entry-IQL H6 mappings (materialize_inference_batch_candidates_v3_v1). Else cement
        # placeholder. Per-bar regime from canonical_v3_row (Exit ALWAYS M1, per-bar conditioning).
        if os.environ.get("GX1_REGIME_V4", "0") == "1":
            # NaN-safe coercion (2026-06-13 audit): the old `... or 1` / `... or 2`
            # truthiness-coerced a VALID id 0 to the default — trend_regime_id=0
            # (TREND_DOWN) became NEUTRAL and vol_regime_id=0 (LOW, ~20% of bars)
            # became MEDIUM, so TREND_DOWN + vol_regime_LOW could NEVER be emitted
            # live on the EXIT side (~20% of in-trade M1 bars got the WRONG regime
            # one-hot — a silent train≠serve skew). Same bug the entry side fixed
            # 2026-06-11; this is the exit-side twin the entry fix missed. Mirror it.
            _tr_raw = canonical_v3_row.get("trend_regime_id")
            _tr = int(_tr_raw) if _tr_raw is not None and np.isfinite(_tr_raw) else 1
            _vr_raw = canonical_v3_row.get("vol_regime_id")
            _vr = int(_vr_raw) if _vr_raw is not None and np.isfinite(_vr_raw) else 2
            _vl = {0: "LOW", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "EXTREME"}.get(_vr, "MEDIUM")
            _tl = {0: "TREND_DOWN", 1: "TREND_NEUTRAL", 2: "TREND_UP"}.get(_tr, "TREND_NEUTRAL")
            for _v in ("LOW", "MEDIUM", "HIGH", "EXTREME"):
                bar_state[f"vol_regime_{_v}"] = 1.0 if _vl == _v else 0.0
            for _v in ("TREND_UP", "TREND_NEUTRAL", "TREND_DOWN"):
                bar_state[f"trend_regime_{_v}"] = 1.0 if _tl == _v else 0.0
        else:
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
            # EX2 (2026-06-04): also emit the BARE name so an AUG64-trained Exit-IQL
            # (GX1_EXIT_AUGMENT_64=1) finds its 64 base-named feats (vol_z_20, dip_confirmed_m5_v3,
            # struct_*_v3, atr_ratio_*, ...). The serve cv3 already carries all 64 via the same
            # one-truth helpers (_augment_cv3_with_volume_features / _group_a_and_dip_struct). Safe
            # unconditionally: the adapter ignores bar_state keys not in the bundle's feature_names,
            # so AUG64-OFF (130-feat cement) bundles are unaffected. Guard prevents clobbering a
            # more-specific bare key (trade-state / candidate feats) set earlier.
            if col not in bar_state:
                bar_state[col] = v

        return bar_state

    # ── inference ────────────────────────────────────────────────────

    def decide_for_trade(
        self,
        trade: TradeState,
        canonical_v3_row: pd.Series,
        v3_v8_out: dict[str, float] | None = None,
        current_m1_atr_bps_override: float | None = None,
        now_minute: "pd.Timestamp | None" = None,  # EX1: live M1 bar ts -> m5_phase minute%5
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
            now_minute=now_minute,  # EX1: m5_phase = minute%5 of the live bar
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

        if STRATEGY_F_ENABLED:
            # ONE-TRUTH Strategy-F overlay (2026-06-13 L7A): the 4-rule decision is now in
            # strategy_f_decision so the Phase-6 gate scores the IDENTICAL +Strategy-F policy.
            iql_q_adv = float(rec.iql_recommendation_v1.advantage_exit_over_hold_v1 or 0.0)
            force_exit, reason = strategy_f_decision(
                mfe_bps=float(trade.cum_mfe_bps or 0.0),
                pnl_bps=float(trade.current_pnl_bps or 0.0),
                iql_q_adv=iql_q_adv,
                hold_horizon_pred_bars=float((trade.v10_snapshot or {}).get("hold_horizon_bars_pred", -1.0)),
                bars_in_trade=int(trade.bars_in_trade or 0),
            )
            if force_exit:
                rec = ExitDeciderV12Recommendation(
                    action_id_v1=EXIT_NOW_ID,
                    action_label_v1="EXIT_NOW",
                    decision_source_v1=reason,
                    v3_should_exit_prob_v1=rec.v3_should_exit_prob_v1,
                    iql_recommendation_v1=rec.iql_recommendation_v1,
                    override_threshold_v1=(HOLD_HORIZON_OVERRUN_MULT
                                           if reason == "HOLD_HORIZON_EXPIRED" else MFE_GIVEBACK_PCT),
                )
        return rec, bar_state
