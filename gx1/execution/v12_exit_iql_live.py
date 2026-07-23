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
  - exact V3 v8 outputs at this bar
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
    trade.update_bar(
        m1_bar_ts=closed_m1_bar_ts,
        bid=bid,
        ask=ask,
        m1_close=m1_close,
        bid_high=bid_high,
        bid_low=bid_low,
        ask_high=ask_high,
        ask_low=ask_low,
    )
    rec = exit_iql.decide_for_trade(
        trade,
        canonical_v3_row=augmented_cv3.iloc[-1],
        v3_v8_out=v3_v8_out,
        current_m1_atr_bps_override=exact_m1_atr_bps,
        now_minute=closed_m1_bar_time,
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
from gx1.execution.v12_trade_state import (
    TradeState,
    require_model_native_entry_snapshot,
)


REQUIRED_V3_STATE_FEATURES = (
    "v3_v8_should_exit_prob",
    "v3_v8_profit_protect_prob",
    "v3_v8_family_argmax",
    "v3_v8_family_logit_max",
)


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
# DEFERRAL HORIZON CAP (user vedtak EXIT_IQL_DEFERRAL_RELABEL_20260707, default 0 = OFF = exact
# pre-vedtak behavior): when > 0, a strong-hold deferral of a triggered Strategy-F fire may run at
# most this many M1 bars counted from the FIRST vetoed fire of the trade; past the cap the fire
# proceeds regardless of Q ("the model may only defer within the horizon cap"). Requires the caller
# to own a per-trade defer_state dict (live: TradeState attr; phase6 gate: per-candidate dict) —
# without one the cap is inert, never guessed.
STRATEGY_F_DEFER_CAP_BARS = _env_float("GX1_STRATEGY_F_DEFER_CAP_BARS", 0.0)
# V10 v3+ Target 4: hold-horizon-expired override (cuts stale grinders).
HOLD_HORIZON_OVERRUN_MULT = _env_float("GX1_HOLD_HORIZON_OVERRUN_MULT", 1.5)
# Sentinel: V10 returns -1 when bundle has no hold_horizon head.
HOLD_HORIZON_INVALID_SENTINEL = 0.0
# Floor so low hold-horizon predictions can't fire the rule from bar 3-4.
HOLD_HORIZON_MIN_FLOOR_BARS = _env_float("GX1_HOLD_HORIZON_MIN_FLOOR_BARS", 60)

# ── LET-WINNERS-RUN overlay (2026-06-18, default OFF) ─────────────────────────────────────────────
# The 06-18 self-diagnosis found held_too_short = the dominant live leak (508 bps): all 10 are
# CONTINUATION-MISSES — the Exit-IQL takes profit AT the in-trade MFE peak (giveback ~0) and the price
# keeps running (post-exit MFE median +27, up to +114). This is the IQL's OWN EXIT_NOW, not Strategy-F
# (which needs 30% giveback) and not strong-hold (which only blocks Strategy-F). LWR suppresses a PROFIT
# EXIT_NOW while the trade is in profit AND still NEAR its peak (giveback fraction < LWR_GIVEBACK_FRAC) so
# the winner rides until a REAL trailing giveback (Strategy-F 30%) or the hard-stop closes it. Never
# suppresses a losing exit (pnl < LWR_MIN_PNL_BPS) or a real giveback. ENV-GATED default-OFF (cement keeps
# train==serve); the OOT exit-replay gate tests it ON before any arming. ONE TRUTH (phase6 + live import it).
LET_WINNERS_RUN = _env_bool("GX1_EXIT_LET_WINNERS_RUN", False)
LWR_GIVEBACK_FRAC = _env_float("GX1_LWR_GIVEBACK_FRAC", 0.30)   # release (allow exit) once giveback >= this × MFE
LWR_MIN_PNL_BPS = _env_float("GX1_LWR_MIN_PNL_BPS", 15.0)       # only let genuine WINNERS run

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
    defer_state: dict | None = None,  # per-trade mutable dict for the defer-cap (None = cap inert)
) -> tuple[bool, str]:
    """ONE-TRUTH Strategy-F overlay decision (the 4-rule post-IQL exit override).

    Returns (force_exit_now, reason). reason ∈ {'', 'HOLD_HORIZON_EXPIRED', 'BREAKEVEN_CUT',
    'MFE_GIVEBACK_OVERRIDE'}. The live exit calls this one implementation; admitted
    replay must import it rather than copy the rules so it scores the +Strategy-F
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
    # DEFERRAL HORIZON CAP (vedtak EXIT_IQL_DEFERRAL_RELABEL_20260707, default OFF): a strong-hold
    # veto of a triggered fire is honoured for at most STRATEGY_F_DEFER_CAP_BARS M1 bars from the
    # trade's FIRST vetoed fire; past the cap the fire proceeds regardless of Q. Matches the
    # training-side relabel whose continuation horizons are all <= 240 M1 bars. Inert when the cap
    # is 0 (cement) or the caller passes no defer_state.
    if (defer_state is not None and STRATEGY_F_DEFER_CAP_BARS > 0
            and strong_hold and (f_trigger or hold_horizon_expired)):
        _first_veto = defer_state.setdefault("sf_first_veto_bar", int(bars_in_trade or 0))
        if int(bars_in_trade or 0) - int(_first_veto) >= STRATEGY_F_DEFER_CAP_BARS:
            strong_hold = False  # deferral budget exhausted -> release the veto
    if hold_horizon_expired and not strong_hold:
        return True, "HOLD_HORIZON_EXPIRED"
    if f_trigger and not strong_hold:
        return True, ("BREAKEVEN_CUT" if breakeven_cut and not profit_lock else "MFE_GIVEBACK_OVERRIDE")
    return False, ""


def let_winners_run_hold(current_pnl_bps: float, cum_mfe_bps: float, *, enabled: bool = True) -> bool:
    """ONE-TRUTH LET-WINNERS-RUN overlay (default-OFF; no-op unless GX1_EXIT_LET_WINNERS_RUN=1).

    Returns True = SUPPRESS this bar's profit-EXIT_NOW and HOLD, so a winning trade that the Exit-IQL
    would close AT its peak keeps running until it gives back >= LWR_GIVEBACK_FRAC of the peak (then the
    normal exit / Strategy-F / hard-stop closes it). Only fires when the trade is a genuine winner
    (pnl >= LWR_MIN_PNL_BPS) AND still near its peak (giveback fraction < LWR_GIVEBACK_FRAC). Never holds a
    loser or a real giveback. Shared by the LIVE exit (make_exit_decision) AND the Phase-6 gate so the gate
    scores the IDENTICAL policy. `enabled` lets a caller force-disable for a baseline arm."""
    if not (enabled and LET_WINNERS_RUN):
        return False
    pnl = float(current_pnl_bps or 0.0)
    mfe = float(cum_mfe_bps or 0.0)
    if pnl < LWR_MIN_PNL_BPS or mfe <= 0.0:
        return False
    giveback_frac = (mfe - pnl) / mfe
    return giveback_frac < LWR_GIVEBACK_FRAC


# ══════════════════════════════════════════════════════════════════════════
# EXIT OPERATING-POINT CONTRACT PIN (user vedtak EXIT_OPERATING_POINT_CONTRACT_PIN_20260707)
# ══════════════════════════════════════════════════════════════════════════
# Every overlay constant above is env-read with a code default that DIFFERS from
# the live policy on the load-bearing knobs (hard-stop 0-vs-80, LWR off-vs-on):
# before this pin they lived ONLY as launch_live_practice.sh exports, so any
# other entrypoint (gate replay, nightly netcap, a manual runner start) silently
# ran a DIFFERENT exit policy than live — gate evidence could be measured on a
# non-live policy. The contract's exit_iql.operating_point.live_env is the ONE
# truth for the effective exit-policy env:
#   - assert_exit_env_matches_contract(): fail-closed startup assert — runs on
#     every contract-resolved ExitIQLLiveInference.load() (the live path) and is
#     called by the launch_live_practice.sh launch-assert. Mismatch →
#     RuntimeError with a per-var diff.
#   - Gate/replay launchers pin their env FROM the contract via
#     scripts/gx1_exit_env_pin.sh (eval its `export` lines) — never hardcoded.
#   - Escape hatch GX1_EXIT_ENV_ASSERT=0 for EXPLICIT research replays only —
#     logs a WARNING, never silent.

_EXIT_ENV_ASSERT_VAR = "GX1_EXIT_ENV_ASSERT"


def _normalize_env_value(raw: object) -> str:
    """Canonicalize an env/contract value for comparison: '80'=='80.0',
    '0.30'=='0.3', 'true'=='1'=='on' (mirrors _env_bool's truthy set). None
    (unset) → '<unset>', which never equals a pinned value — a contract-named
    but UNexported contract variable is a mismatch."""
    if raw is None:
        return "<unset>"
    s = str(raw).strip()
    low = s.lower()
    if low in ("true", "yes", "on"):
        return "1"
    if low in ("false", "no", "off"):
        return "0"
    try:
        f = float(s)
    except (TypeError, ValueError):
        return low
    return str(int(f)) if f == int(f) else repr(f)


def exit_env_contract_diff(live_env: dict, environ=os.environ) -> list[str]:
    """Per-var diff of the EFFECTIVE env vs a contract live_env {var: value}
    dict. Empty list == match. ONE compare truth — shared by the live startup
    assert below AND the launch_live_practice.sh launch-assert."""
    diffs: list[str] = []
    for var in sorted(live_env):
        expected = live_env[var]
        effective = environ.get(var)
        if _normalize_env_value(effective) != _normalize_env_value(expected):
            eff_repr = "<unset>" if effective is None else repr(effective)
            diffs.append(f"{var}: contract={str(expected)!r} effective={eff_repr}")
    return diffs


def assert_exit_env_matches_contract(context: str, contract_entry: dict | None = None) -> None:
    """Fail-closed: the effective exit-policy env MUST equal the contract's
    exit_iql.operating_point.live_env — otherwise this process runs a different
    exit policy (hard-stop / LWR / Strategy-F / AUG64 / regime flags) than the
    pinned live operating point. RuntimeError carries the per-var diff.
    GX1_EXIT_ENV_ASSERT=0 = explicit research-replay escape hatch (WARNING)."""
    if os.environ.get(_EXIT_ENV_ASSERT_VAR, "1") == "0":
        LOG.warning(
            "[EXIT_ENV_ASSERT] DISABLED via %s=0 (%s) — explicit research-replay "
            "escape hatch; the effective exit policy is NOT verified against the "
            "contract live_env. Never produce live/gate evidence this way.",
            _EXIT_ENV_ASSERT_VAR, context,
        )
        return
    if contract_entry is None:
        from gx1_guards.artifacts import load_decision_entry
        contract_entry = load_decision_entry("exit_iql")
    live_env = (contract_entry.get("operating_point") or {}).get("live_env") or {}
    if not isinstance(live_env, dict) or not live_env:
        raise RuntimeError(
            f"[EXIT_ENV_ASSERT] ({context}) contract exit_iql.operating_point.live_env "
            "is missing/empty — the exit operating point is UNPINNED (fail-closed). "
            "Pin it via vedtak in PROJECT_STATE_artifacts.json, or set "
            f"{_EXIT_ENV_ASSERT_VAR}=0 for an explicit research replay."
        )
    diffs = exit_env_contract_diff(live_env)
    if diffs:
        raise RuntimeError(
            f"[EXIT_ENV_ASSERT] ({context}) effective exit-policy env does NOT match "
            "the contract exit_iql.operating_point.live_env — refusing to run a "
            "policy that differs from the pinned live operating point:\n  "
            + "\n  ".join(diffs)
            + "\nFix: export the contract values (launch_live_practice.sh, or "
            "`eval \"$(bash scripts/gx1_exit_env_pin.sh)\"` for gates/replays), or set "
            f"{_EXIT_ENV_ASSERT_VAR}=0 for an explicit research replay ONLY."
        )
    LOG.info(
        "[EXIT_ENV_ASSERT] OK (%s): %d exit-policy env vars match the contract live_env.",
        context, len(live_env),
    )


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

def _exit_rec_with_distilled_q(rec, v3_v8_out: dict[str, float]):
    """Rebuild ExitDeciderV12Recommendation using V3 distilled q_head.

    Replaces iql_recommendation_v1 with one built from v3_q_hold_v1/v3_q_exit_v1.
    If the original decision_source was IQL_Q, the outer action is re-derived
    from the distilled Q (argmax). A V3_OVERRIDE source is preserved — we only
    swap the underlying iql_rec for diagnostic visibility.
    """
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
            # Startup assert (vedtak EXIT_OPERATING_POINT_CONTRACT_PIN_20260707): a
            # contract-resolved load IS the live policy — the effective env must match
            # the pinned exit_iql.operating_point.live_env (hard-stop/LWR/Strategy-F/
            # AUG64/regime flags). Explicit-args loads (tests/shadow/research) skip
            # this; gate launchers pin the env via scripts/gx1_exit_env_pin.sh.
            assert_exit_env_matches_contract(
                context="ExitIQLLiveInference.load(contract-resolved)",
                contract_entry=entry,
            )
            bundle_dir = bundle_dir if bundle_dir is not None else Path(entry["path"])
            variant = (
                variant if variant is not None else entry["active_variant"]
            )
            fold_id = (
                fold_id if fold_id is not None else entry["serving_fold"]
            )
            aggregator = (
                aggregator
                if aggregator is not None
                else entry["active_aggregator"]
            )
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
        # The check below catches a retrained bundle that drops or renames the
        # required V3 state before it can become train/serve skew.
        feat_set = set(feature_names)
        v3_block_keys = set(REQUIRED_V3_STATE_FEATURES)
        missing_v3 = v3_block_keys - feat_set
        if missing_v3:
            raise RuntimeError(
                "[EXIT_IQL_V3_BLOCK_PARTIAL] adapter feature_names omit "
                f"{len(missing_v3)} of {len(v3_block_keys)} mandatory V3 features: "
                f"{sorted(missing_v3)}"
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
          - exact V3 v8 outputs at this bar (4 required features)
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
        s = require_model_native_entry_snapshot(trade.v10_snapshot)
        dp = s["direction_probs"]
        p_long_e = float(dp[0])
        p_short_e = float(dp[1])
        p_flat_e = float(dp[2])
        p_hat_e = max(p_long_e, p_short_e, p_flat_e)
        sorted_probs = sorted([p_long_e, p_short_e, p_flat_e], reverse=True)
        margin_e = sorted_probs[0] - sorted_probs[1]
        uncertainty_e = 1.0 - p_hat_e
        atr_bps_entry = float(s["atr_bps"])
        decision_ts = pd.Timestamp(s["decision_ts"])
        if decision_ts.tz is None or str(decision_ts.tz) != "UTC":
            raise RuntimeError("EXIT_IQL_ENTRY_DECISION_TIME_NOT_UTC")
        bar_state.update({
            "weekday_utc": float(decision_ts.dayofweek),
            "hour_utc": float(decision_ts.hour),
            "atr_bps": atr_bps_entry,
            "p_long": p_long_e,
            "p_short": p_short_e,
            "p_flat": p_flat_e,
            "p_hat": p_hat_e,
            "margin": margin_e,
            "uncertainty_score": uncertainty_e,
            "tradable_prob": float(s["tradable_prob"]),
            "mfe_first_n_pred": float(s["mfe_first_n"]),
            "path_quality_pred": float(s["path_quality"]),
        })
        # Exact V3 v8 state. The active pipeline validates the richer output
        # contract before this call; direct callers still fail closed instead
        # of manufacturing the old all-zero block.
        if not isinstance(v3_v8_out, dict):
            raise RuntimeError("EXIT_IQL_V3_STATE_MISSING")
        missing_v3 = [name for name in REQUIRED_V3_STATE_FEATURES if name not in v3_v8_out]
        if missing_v3:
            raise RuntimeError(f"EXIT_IQL_V3_STATE_FIELDS_MISSING: {missing_v3}")
        invalid_v3: list[str] = []
        for name in REQUIRED_V3_STATE_FEATURES:
            try:
                value = float(v3_v8_out[name])
            except (TypeError, ValueError):
                invalid_v3.append(name)
                continue
            if not np.isfinite(value):
                invalid_v3.append(name)
        if invalid_v3:
            raise RuntimeError(f"EXIT_IQL_V3_STATE_NONFINITE: {invalid_v3}")
        v3_block = v3_v8_out
        bar_state.update({k: float(v) for k, v in v3_block.items()})
        # Side one-hot
        bar_state.update(trade.build_side_one_hot())
        # current_atr_bps
        # current_atr_bps_v1 must be the exact per-M1-bar
        # (ask_high-bid_low)/mid bps used by training. M5 ATR substitution is
        # distribution-changing and therefore forbidden.
        try:
            current_atr_bps = float(current_m1_atr_bps_override)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("EXIT_IQL_M1_ATR_MISSING") from exc
        if not np.isfinite(current_atr_bps) or current_atr_bps <= 0.0:
            raise RuntimeError("EXIT_IQL_M1_ATR_INVALID")
        bar_state["current_atr_bps_v1"] = current_atr_bps

        # EX1 (2026-06-04 train==serve parity): the per-bar TRAINER encodes m5_phase = minute%5
        # (compute_m5_phase_index — XGB-refresh STALENESS). The old serve read canonical_v3_row['m5_phase_{p}']
        # = cv3's minute//12 HOUR-segment bucket (materialize_build_canonical_features_v1) — a DIFFERENT
        # formula, and both have variance. Exact M1 time is mandatory.
        if now_minute is None:
            raise RuntimeError("EXIT_IQL_M1_PHASE_TIME_MISSING")
        parsed_now = pd.Timestamp(now_minute)
        if parsed_now.tz is None or str(parsed_now.tz) != "UTC":
            raise RuntimeError("EXIT_IQL_M1_PHASE_TIME_NOT_UTC")
        from gx1.exits.contracts.exit_io_v3_ctx36_m1l512_phase5 import (
            compute_m5_phase_onehot,
        )

        m5_onehot = compute_m5_phase_onehot(parsed_now)
        for phase in range(5):
            bar_state[f"m5_phase_{phase}_v1"] = float(m5_onehot[phase])

        # Exit training carries these categorical coordinates from the
        # candidate row, so live must keep their exact entry-time values frozen
        # for the whole trade. Current-bar session/regime substitution is a
        # train/serve semantic mismatch.
        sess_label = str(s["session"])
        bar_state[f"session_{sess_label}"] = 1.0
        # Set zeros for other sessions
        for session_name in ("ASIA", "EU", "OVERLAP", "US"):
            bar_state.setdefault(f"session_{session_name}", 0.0)
        # The Exit training rows carry vol/trend labels from the Entry
        # candidate. Keep those exact frozen coordinates while current-bar
        # regime features remain available separately through canonical state.
        if os.environ.get("GX1_REGIME_V4") != "1":
            raise RuntimeError("EXIT_IQL_REGIME_V4_NOT_PINNED")
        vol_label = str(s["entry_vol_regime"])
        trend_label = str(s["entry_trend_regime"])
        for label in ("LOW", "MEDIUM", "HIGH", "EXTREME"):
            bar_state[f"vol_regime_{label}"] = 1.0 if vol_label == label else 0.0
        for label in ("TREND_UP", "TREND_NEUTRAL", "TREND_DOWN"):
            bar_state[f"trend_regime_{label}"] = (
                1.0 if trend_label == label else 0.0
            )
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
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v):
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

        required_features = tuple(getattr(self, "feature_names", ()))
        missing_features = [
            name for name in required_features if name not in bar_state
        ]
        invalid_features: list[str] = []
        for name in required_features:
            if name not in bar_state:
                continue
            try:
                value = float(bar_state[name])
            except (TypeError, ValueError):
                invalid_features.append(name)
                continue
            if not np.isfinite(value):
                invalid_features.append(name)
        if missing_features or invalid_features:
            raise RuntimeError(
                "EXIT_IQL_REQUIRED_FEATURE_CONTRACT_INVALID: "
                f"missing={missing_features[:20]} "
                f"nonfinite={invalid_features[:20]}"
            )

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
            # Defer-cap state (vedtak EXIT_IQL_DEFERRAL_RELABEL_20260707): per-trade dict hung on
            # the TradeState so the cap survives across bars. Inert unless
            # GX1_STRATEGY_F_DEFER_CAP_BARS > 0 (cement default 0).
            _defer_state = getattr(trade, "sf_defer_state_v1", None)
            if _defer_state is None:
                _defer_state = {}
                try:
                    setattr(trade, "sf_defer_state_v1", _defer_state)
                except Exception:  # noqa: BLE001 — a frozen/slotted TradeState degrades to inert cap
                    _defer_state = None
            force_exit, reason = strategy_f_decision(
                mfe_bps=float(trade.cum_mfe_bps or 0.0),
                pnl_bps=float(trade.current_pnl_bps or 0.0),
                iql_q_adv=iql_q_adv,
                hold_horizon_pred_bars=float((trade.v10_snapshot or {}).get("hold_horizon_bars_pred", -1.0)),
                bars_in_trade=int(trade.bars_in_trade or 0),
                defer_state=_defer_state,
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
