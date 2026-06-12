#!/usr/bin/env python3
"""V12 live pipeline orchestrator.

Single entry point for the live V12 stack:
    M1 → canonical_v3 (live) → ctx_augment → XGB v5 → V10 v3 → Entry-IQL v2
                                                          ↓ if TAKE
                                                       open TradeState
                                                          ↓ per-M1
                                                       Exit-IQL V12.1 → HOLD/EXIT

Encapsulates model loading (~300 ms one-time at startup) and provides
two main inference methods:

  .make_entry_decision(now_minute, bid, ask)
      No open trade → returns SKIP / TAKE_LONG_NOW / TAKE_SHORT_NOW
      with Q-values, V10 outputs, XGB outputs, and the full state
      snapshot (for journaling).

  .make_exit_decision(trade, now_minute, bid, ask, m1_close)
      Open trade → advances trade state, returns HOLD / EXIT_NOW
      with the full 201-feature bar_state (for journaling).

Used by v12_paper_runner.py to drive live trade decisions.

⚠️ Sesjon 3/4 known approximations (carried through):
  - 4 pre-prune chunk0 features that the Entry-IQL contract expects
    but canonical_v3 has dropped (handled by adapter zero-fill).
  - V3 v8 inference not yet wired → V3-tracking features in Exit-IQL
    state are 0. Affects Q-values somewhat; safe for shadow mode.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.execution.v12_state_from_prebuilt import PrebuiltStateLoader
from gx1.execution.v12_xgb_live import XGBLiveInference
from gx1.execution.v12_v10_live import V10LiveInference, SEQ_LEN as V10_SEQ_LEN
from gx1.execution.v12_entry_iql_live import EntryIQLLiveInference
from gx1.execution.v12_exit_iql_live import ExitIQLLiveInference
from gx1.execution.v12_v3_live import V3LiveInference
from gx1.execution.v12_trade_state import TradeState, SIDE_LONG, SIDE_SHORT

LOG = logging.getLogger("v12_pipeline")


# Phase 3a/3b A/B-switch: when set to "1", swap Entry-IQL Q-values with the
# V10 distilled q_head's q_per_action. Used by Phase 6 to A/B test distilled
# bundles vs the IQL teacher. The downstream gates (cluster1, min_adv,
# regime filter, portfolio guard) all still apply unchanged.
_USE_DISTILLED_ENTRY = os.environ.get("GX1_USE_DISTILLED_ENTRY", "0") == "1"

# IN-PROCESS SHADOW (2026-06-12, ladder wave): when GX1_SHADOW_BUNDLE_DIR points
# at a candidate Entry-IQL bundle, a SECOND adapter scores every poll's candidate
# through the same predict() (incl. the live conviction-gate/overlay env flags)
# and the shadow Q/action are journaled alongside the live decision. The live
# decision is NEVER affected — fail-safe: any shadow error logs once and disables.
# Same env var as the Track B daemon so one export shadows the candidate everywhere.
_SHADOW_BUNDLE_DIR = os.environ.get("GX1_SHADOW_BUNDLE_DIR", "").strip()
_SHADOW_VARIANT = os.environ.get("GX1_SHADOW_VARIANT", "").strip()  # default: contract active_variant
_SHADOW_FOLD = os.environ.get("GX1_SHADOW_FOLD", "").strip()        # default: contract first active fold


def _entry_rec_with_distilled_q(rec, v10_out, beta: float = 1.0):
    """Rebuild an EntryRecommendation using V10 q_head values instead of IQL Q.

    Only the Q-vector and derived fields (action, advantage, softmax) are
    replaced. variant/fold/feature_names/state stay as-is — they're only
    informational. Caller must verify v10_out has q_per_action_v1 before calling.
    """
    from gx1.runtime.entry_iql_v2_adapter import EntryRecommendation, iql_core
    q = np.asarray(v10_out["q_per_action_v1"], dtype=np.float32)
    a_id = int(np.argmax(q))
    q_skip = float(q[iql_core.ACTION_SKIP_ID])
    chosen_q = float(q[a_id])
    best_take = float(max(q[iql_core.ACTION_TAKE_LONG_NOW_ID],
                          q[iql_core.ACTION_TAKE_SHORT_NOW_ID]))
    adv_skip = chosen_q - q_skip
    adv_take = chosen_q - best_take
    scaled = beta * q
    scaled = scaled - scaled.max()
    soft = np.exp(scaled); soft = soft / soft.sum()
    return dataclasses.replace(
        rec,
        action_id_v1=a_id,
        action_label_v1=iql_core.ACTION_LABELS_V1[a_id],
        q_per_action_v1=q.copy(),
        q_per_action_per_k_v1=np.broadcast_to(q[:, None], (3, len(rec.k_horizons_v1))).copy(),
        advantage_over_skip_v1=adv_skip,
        advantage_over_realized_v1=adv_take,
        confidence_softmax_v1=soft.astype(np.float32),
    )

COLLECTOR_DIR = Path("/home/andre2/GX1_DATA/reports/v12_live_data")
CANONICAL_M1_DIR = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL")


# ── ID mapping (matches iql_core ACTION_*_ID) ─────────────────────────
ACTION_LABEL_BY_ID = {0: "SKIP", 1: "TAKE_LONG_NOW", 2: "TAKE_SHORT_NOW"}


@dataclass
class V12Pipeline:
    prebuilt_loader: PrebuiltStateLoader
    xgb: XGBLiveInference
    v10: V10LiveInference
    entry_iql: EntryIQLLiveInference
    exit_iql: ExitIQLLiveInference
    v3: V3LiveInference | None = None     # V3 v8 — used for exit decisions
    # Cache for the most recent augmented window + XGB bridge (refreshed per M5)
    _last_augmented_bucket: pd.Timestamp | None = None
    _last_augmented: pd.DataFrame | None = None
    _last_bridge: np.ndarray | None = None
    _last_xgb_p_long: np.ndarray | None = None
    _last_xgb_p_short: np.ndarray | None = None
    _last_xgb_p_flat: np.ndarray | None = None
    # V12.2 cluster-1 fix: per-side last-entry M5 bucket. Blocks repeat same-side
    # entries within the same M5 (5-min) window — addresses live "4 LONG in 7min"
    # cluster pattern observed 2026-05-12 where V10 fired 4× same signal in same bucket.
    _last_entry_m5_by_side: dict[str, pd.Timestamp] = field(default_factory=dict)
    # Per-M1-bar atr_bps cache for current_atr_bps_v1 in Exit-IQL bar_state.
    # Training computes this as (ask_high - bid_low)/mid * 1e4 per M1 bar — typical
    # value 3-7 bps. Live had been using canonical_v3 M5 ATR14 (10-50 bps), a 10x
    # distribution shift that destabilized Exit-IQL Q-values. C1 fix 2026-05-19.
    _last_m1_atr_bps: float = 0.0
    _last_m1_atr_minute: pd.Timestamp | None = None
    # V4 (R13): full intrabar OHLC of the latest CLOSED M1 bar (one source for the
    # V3 overlay's intrabar peak/trough/atr AND current_atr_bps_v1).
    _last_m1_bar: dict | None = None
    # In-process shadow (2026-06-12, ladder wave): candidate Entry-IQL scoring
    # every poll alongside the live adapter. None = shadow off / load failed.
    shadow_entry_iql: EntryIQLLiveInference | None = None
    _shadow_error_logged: bool = False

    @classmethod
    def _load_shadow_entry_iql(cls) -> "EntryIQLLiveInference | None":
        """Load the candidate shadow bundle from GX1_SHADOW_BUNDLE_DIR.

        FAIL-SAFE by design (NOT fail-closed): the shadow is observability, not
        decisioning — a broken candidate bundle must never block the live stack.
        Variant/fold default to the contract-resolved ACTIVE entry's values so a
        warm-started candidate (same ckpt naming) loads with zero extra config.
        """
        if not _SHADOW_BUNDLE_DIR:
            return None
        try:
            from gx1.execution.v12_entry_iql_live import (
                DEFAULT_VARIANT, DEFAULT_FOLD, DEFAULT_AGGREGATOR)
            variant = _SHADOW_VARIANT or DEFAULT_VARIANT
            fold = _SHADOW_FOLD or DEFAULT_FOLD
            shadow = EntryIQLLiveInference.load(
                bundle_dir=Path(_SHADOW_BUNDLE_DIR), variant=variant, fold_id=fold,
                aggregator=DEFAULT_AGGREGATOR,
            )
            LOG.info(f"SHADOW Entry-IQL loaded: {Path(_SHADOW_BUNDLE_DIR).name} "
                     f"variant={variant} fold={fold} — scores every poll, affects nothing")
            return shadow
        except Exception as exc:
            LOG.error(f"SHADOW Entry-IQL load FAILED ({exc}) — live unaffected, shadow disabled")
            return None

    @classmethod
    def load_default(cls) -> "V12Pipeline":
        t0 = time.perf_counter()
        loader = PrebuiltStateLoader()
        loader.load()
        xgb = XGBLiveInference.load_default()
        v10 = V10LiveInference.load_default()
        entry_iql = EntryIQLLiveInference.load_default()
        exit_iql = ExitIQLLiveInference.load_default()
        v3 = V3LiveInference.load_default()   # V3 v8 exit transformer
        shadow_entry_iql = cls._load_shadow_entry_iql()
        # V12.2: if any model needs multi-TF, build the per-bar feature tables once
        # on the loader so predict() calls can slice cheaply.
        needs_mtf = getattr(v10, "_enable_multi_tf", False) or getattr(v3, "_enable_multi_tf", False)
        if needs_mtf:
            LOG.info("V12.2: building multi-TF features on PrebuiltStateLoader (one-time)")
            loader.build_multi_tf_features()
        LOG.info(f"V12Pipeline loaded in {(time.perf_counter()-t0)*1000:.0f} ms")
        LOG.info(f"  prebuilt cutoff: {loader.cutoff_ts}  multi_tf_active={needs_mtf}")
        return cls(prebuilt_loader=loader, xgb=xgb, v10=v10,
                    entry_iql=entry_iql, exit_iql=exit_iql, v3=v3,
                    shadow_entry_iql=shadow_entry_iql)

    def _refresh_m1_bar(self, now_minute: pd.Timestamp) -> dict | None:
        """Latest CLOSED M1 bar's intrabar OHLC (bid/ask high/low/close) + per-bar
        atr_bps, from the OANDA collector parquet. ONE source for BOTH the V3
        overlay's intrabar peak/trough/atr (V4/R13) AND current_atr_bps_v1.

        Training (materialize_build_exit_iql_per_bar_dataset_v1.py compute_per_bar_signals)
        uses per-M1-bar bid/ask hi-lo (atr typical 3-7 bps; intrabar excursion for
        MFE/MAE). Live previously used canonical M5 ATR14 (10-50 bps) for atr and a
        close-only MFE/MAE approximation — both train/serve skews. Cached per minute
        to avoid disk thrash. Returns None (keeps prior cache) if no bar available.
        """
        cur_min = now_minute.replace(second=0, microsecond=0)
        if self._last_m1_atr_minute == cur_min and self._last_m1_bar is not None:
            return self._last_m1_bar
        try:
            from pathlib import Path
            day = cur_min.strftime("%Y%m%d")
            p = Path(f"/home/andre2/GX1_DATA/reports/v12_live_data/xauusd_m1_{day}.parquet")
            if not p.exists():
                return self._last_m1_bar
            df = pd.read_parquet(p, columns=[
                "time", "bid_high", "bid_low", "ask_high", "ask_low", "ask_close", "bid_close",
            ]).tail(2)
            if len(df) == 0:
                return self._last_m1_bar
            row = df.iloc[-1]
            ah = float(row["ask_high"]); bl = float(row["bid_low"])
            ac = float(row["ask_close"]); bc = float(row["bid_close"])
            mid = (ac + bc) / 2.0
            atr = (ah - bl) / max(mid, 1e-6) * 1e4 if mid > 0 else 0.0
            bar = {
                "bid_high": float(row["bid_high"]), "bid_low": bl,
                "ask_high": ah, "ask_low": float(row["ask_low"]),
                "bid_close": bc, "ask_close": ac, "atr_bps": float(atr),
            }
            self._last_m1_bar = bar
            self._last_m1_atr_bps = float(atr)
            self._last_m1_atr_minute = cur_min
            return bar
        except Exception as exc:
            LOG.warning(f"_refresh_m1_bar failed: {exc}; keeping prior bar")
            return self._last_m1_bar

    def _refresh_m1_atr_bps(self, now_minute: pd.Timestamp) -> float:
        """current_atr_bps_v1 = per-M1-bar (ask_high-bid_low)/mid*1e4. Thin accessor
        over the one-truth _refresh_m1_bar (same source, same per-minute cache)."""
        bar = self._refresh_m1_bar(now_minute)
        return float(bar["atr_bps"]) if bar else self._last_m1_atr_bps

    # ── shared canonical_v3 build (cached per M5 bucket) ───────────────

    def record_entry_for_cluster(self, side: str, decision_m5_ts: pd.Timestamp) -> None:
        """Caller (paper-runner) invokes this AFTER the trade is filled and the
        new TradeState is created. Records the M5-bucket so subsequent same-M5
        ticks are correctly cluster-blocked. Audit 3 C-3 fix 2026-05-20."""
        side_key = "long" if side in ("long", "TAKE_LONG_NOW") else "short"
        self._last_entry_m5_by_side[side_key] = pd.Timestamp(decision_m5_ts)

    def _refresh_canonical(self, now_minute: pd.Timestamp) -> bool:
        """Refresh augmented window + XGB bridge from disk prebuilt.

        If now_minute is past the prebuilt cutoff, automatically falls back to
        the cutoff bar (latest available CLOSED M5). The runner then makes a
        decision on the freshest data available, instead of skipping. The
        incremental updater keeps cutoff within ~5-15 min of real-time.

        Returns True if data available, False only if prebuilt is empty or
        history insufficient (early-history edge cases).
        """
        # Hot-reload prebuilts from disk if incremental updater extended them.
        # Invalidates cached window if cutoff advanced.
        if self.prebuilt_loader.refresh_if_changed():
            self._last_augmented_bucket = None
            self._last_augmented = None
        cutoff = self.prebuilt_loader.cutoff_ts
        # FAIL-CLOSED staleness cap (2026-06-03 audit): the clip below silently decides on
        # stale features if the canonical-incremental daemon stalls/dies (now>cutoff frozen).
        # Refuse (SKIP) when the prebuilt is older than the cap. Live-only by construction:
        # replay always has now<=cutoff so this never fires there.
        import os as _os
        _max_stale_min = float(_os.environ.get("GX1_MAX_PREBUILT_STALENESS_MIN", "30"))
        if cutoff is not None and now_minute > cutoff:
            _age_min = (now_minute - cutoff).total_seconds() / 60.0
            if _age_min > _max_stale_min:
                LOG.error(f"[PREBUILT_STALE] cutoff {cutoff} is {_age_min:.0f} min behind now "
                          f"{now_minute} (> {_max_stale_min} cap) — SKIP (canonical daemon stalled?)")
                return False
        # Clip to latest available M5 if past cutoff (within staleness cap)
        effective_ts = now_minute if now_minute <= cutoff else cutoff
        cur_bucket = effective_ts.floor("5min")
        if self._last_augmented_bucket == cur_bucket and self._last_augmented is not None:
            return True

        # Read 96-bar window directly from canonical_v3 + BASE28 prebuilts.
        # Identical values to what V12 cascade trainings saw — no live recompute.
        augmented = self.prebuilt_loader.get_window(effective_ts, n_bars=V10_SEQ_LEN)
        if augmented.empty:
            LOG.warning(f"prebuilt empty for {effective_ts} (cutoff={cutoff}) — system not ready")
            return False
        if len(augmented) < V10_SEQ_LEN:
            LOG.warning(f"only {len(augmented)} bars (need {V10_SEQ_LEN}) — early-history bar")
            return False

        # Run XGB on the entire 96-bar window (needed for V10 seq_x signal_bridge)
        xgb_out = self.xgb.predict(augmented)
        self._last_augmented_bucket = cur_bucket
        self._last_augmented = augmented
        self._last_bridge = xgb_out["signal_bridge_v1"]
        self._last_xgb_p_long = xgb_out["p_long"]
        self._last_xgb_p_short = xgb_out["p_short"]
        self._last_xgb_p_flat = xgb_out["p_flat"]
        return True

    # ── entry decision ────────────────────────────────────────────────

    def make_entry_decision(
        self,
        now_minute: pd.Timestamp,
        bid: float,
        ask: float,
        portfolio_state: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run the full XGB → V10 → Entry-IQL chain for the current bar.

        Returns dict with:
            action: SKIP / TAKE_LONG_NOW / TAKE_SHORT_NOW
            action_id: 0 / 1 / 2
            q_per_action: [Q_skip, Q_take_long, Q_take_short]
            advantage_over_skip: float
            advantage_over_skip_long: float
            advantage_over_skip_short: float
            xgb: {p_long, p_short, p_flat}
            v10: {direction_probs, tradable_prob, mfe_first_n, ...}
            decision_ts: ISO timestamp of the M5 bucket used
        """
        if not self._refresh_canonical(now_minute):
            return {"action": "SKIP", "error": "no_canonical_data",
                     "q_per_action": [0.0, 0.0, 0.0],
                     "advantage_over_skip": 0.0,
                     "advantage_over_skip_long": 0.0, "advantage_over_skip_short": 0.0}

        augmented = self._last_augmented
        bridge = self._last_bridge

        # V10 requires 96-bar history → make sure we have it
        if len(augmented) < V10_SEQ_LEN:
            return {"action": "SKIP", "error": f"insufficient_history_{len(augmented)}<{V10_SEQ_LEN}",
                     "q_per_action": [0.0, 0.0, 0.0],
                     "advantage_over_skip": 0.0,
                     "advantage_over_skip_long": 0.0, "advantage_over_skip_short": 0.0}

        # The decision-bar is the LATEST closed M5 bar in augmented
        end_idx = len(augmented) - 1
        # V12.2: fetch multi-TF windows if V10 bundle requires them (no-op for v3 bundle)
        mtf_windows = None
        if getattr(self.v10, "_enable_multi_tf", False):
            decision_ts = augmented.index[end_idx]
            mtf_windows = self.prebuilt_loader.get_multi_tf_windows(decision_ts)
            if not mtf_windows:
                LOG.warning("V10 multi-TF needed but PrebuiltStateLoader has no multi-TF features built — "
                             "build_multi_tf_features() must be called once after load()")
        v10_out = self.v10.predict(augmented, bridge, end_idx=end_idx, multi_tf_windows=mtf_windows)

        # Entry-IQL input
        row = augmented.iloc[end_idx]
        xgb_this = {
            "p_long": float(self._last_xgb_p_long[end_idx]),
            "p_short": float(self._last_xgb_p_short[end_idx]),
            "p_flat": float(self._last_xgb_p_flat[end_idx]),
        }
        rec, candidate = self.entry_iql.predict_from_pipeline(
            row, xgb_this, v10_out, portfolio_state=portfolio_state,
        )
        # Phase 3a A/B: replace IQL Q with distilled V10 q_head when env says so.
        # Has no effect if v10_out lacks q_per_action_v1 (non-distilled bundle).
        if _USE_DISTILLED_ENTRY and "q_per_action_v1" in v10_out:
            rec = _entry_rec_with_distilled_q(rec, v10_out, beta=self.entry_iql.adapter.beta)

        # IN-PROCESS SHADOW (2026-06-12): candidate bundle scores the SAME inputs
        # through the SAME predict() path (incl. conviction-gate/overlay env flags
        # → candidate-vs-active on the live operating point). Observability only —
        # any error logs ONCE and disables; the live decision is already final.
        shadow_rec = None
        if self.shadow_entry_iql is not None:
            try:
                shadow_rec, _ = self.shadow_entry_iql.predict_from_pipeline(
                    row, xgb_this, v10_out, portfolio_state=portfolio_state,
                )
            except Exception as exc:
                if not self._shadow_error_logged:
                    LOG.error(f"SHADOW predict failed ({exc}) — disabling shadow for this run")
                    self._shadow_error_logged = True
                self.shadow_entry_iql = None
                shadow_rec = None

        # V12.2 CLUSTER-1 RATE-LIMIT: block repeat entries within same M5 bucket.
        # Live V10/IQL fires multiple times per minute; without this, IQL takes 4×
        # same LONG signal in same 5-min window (Cluster-1 pattern observed
        # 2026-05-12). H5 audit fix 2026-05-19: also block OPPOSITE-side flapping
        # within the same M5 (e.g. LONG@16:00:30 then SHORT@16:01:15 would have
        # passed before — now blocked) to prevent same-bar contradiction trades.
        # 2026-05-30: PURE_PHASE6 originally bypassed CLUSTER1_RATE_LIMIT too,
        # because Phase 6 OOT didn't gate same-M5 re-entries (each OOT candidate
        # is its own decision point, already deduplicated).
        # 2026-06-02: REVERSED — CLUSTER1 is now ALWAYS ON in live regardless
        # of PURE_PHASE6. The OOT-replay justification was wrong because in live,
        # one M5 yields many M1 ticks and each ticks re-evaluates the same V10
        # snapshot → without CLUSTER1, NEW PQ_COND policy stacked 7 short trades
        # on the same M5 bar overnight 2026-06-02 (−1,348 USD on a 4500→4520
        # adverse move). CLUSTER1 is a sanity-floor for live, not a feature gate.
        # Override via GX1_CLUSTER1_DISABLE=1 only for explicit OOT-replay runs.
        import os as _os
        _cluster1_disabled = bool(int(_os.environ.get("GX1_CLUSTER1_DISABLE", "0")))
        action_label = rec.action_label_v1
        if (not _cluster1_disabled) and action_label in ("TAKE_LONG_NOW", "TAKE_SHORT_NOW"):
            side_key = "long" if action_label == "TAKE_LONG_NOW" else "short"
            decision_m5 = augmented.index[end_idx]  # already an M5-bar timestamp
            last_m5_same = self._last_entry_m5_by_side.get(side_key)
            last_m5_opp = self._last_entry_m5_by_side.get(
                "short" if side_key == "long" else "long"
            )
            block_reason = None
            if last_m5_same == decision_m5:
                block_reason = "CLUSTER1_SAME_SIDE_SAME_M5"
            elif last_m5_opp == decision_m5:
                block_reason = "CLUSTER1_OPPOSITE_SIDE_SAME_M5"
            if block_reason is not None:
                LOG.info(
                    f"[CLUSTER1_RATE_LIMIT] blocking {side_key} entry ({block_reason}) — "
                    f"last_same={last_m5_same} last_opp={last_m5_opp} current_m5={decision_m5}"
                )
                return {
                    "action": "SKIP",
                    "action_id": 0,
                    "q_per_action": [float(rec.q_per_action_v1[0]), float(rec.q_per_action_v1[1]), float(rec.q_per_action_v1[2])],
                    "q_skip": float(rec.q_per_action_v1[0]),
                    "q_take_long": float(rec.q_per_action_v1[1]),
                    "q_take_short": float(rec.q_per_action_v1[2]),
                    "advantage_over_skip": 0.0,
                    "advantage_over_skip_long": float(rec.q_per_action_v1[1] - rec.q_per_action_v1[0]),
                    "advantage_over_skip_short": float(rec.q_per_action_v1[2] - rec.q_per_action_v1[0]),
                    "blocked_reason": block_reason,
                    "decision_ts": str(decision_m5),
                    "stub": False,
                }
            # Cluster1-state advance moved to v12_paper_runner.py post-fill.
            # Audit 3 C-3 fix 2026-05-20: previously this wrote state regardless
            # of downstream gate/portfolio-guard rejection, causing false
            # cluster-blocks on subsequent ticks. Caller must call
            # `pipeline.record_entry_for_cluster(side, decision_m5)` AFTER the
            # trade is actually opened (not on TAKE_*_NOW recommendation alone).

        q = rec.q_per_action_v1
        # 2026-05-30: capture full 192-dim state vector + feature_names + per-K
        # raw Q for offline counterfactual variant replay (multi_variant_counterfactual.py)
        # and online-IQL prep. Adds ~2 KB per journal event. Safe to drop if
        # journal disk pressure becomes an issue.
        try:
            state_arr = rec.state_v1
            state_list = [float(v) for v in state_arr.tolist()] if state_arr is not None else None
        except Exception as exc:
            LOG.warning(f"[ONLINE_IQL_CAPTURE] state_v1 extract failed: {exc} — "
                        f"journal will have null state_v1 for this poll. If this "
                        f"persists, online-IQL replay buffer build will skip these "
                        f"events.")
            state_list = None
        try:
            q_per_k_arr = rec.q_per_action_per_k_v1
            q_per_k_list = q_per_k_arr.tolist() if q_per_k_arr is not None else None
        except Exception as exc:
            LOG.warning(f"[ONLINE_IQL_CAPTURE] q_per_action_per_k_v1 extract failed: {exc}")
            q_per_k_list = None
        out = {
            "action": rec.action_label_v1,
            "action_id": int(rec.action_id_v1),
            "q_per_action": [float(q[0]), float(q[1]), float(q[2])],
            "q_skip": float(q[0]),
            "q_take_long": float(q[1]),
            "q_take_short": float(q[2]),
            "advantage_over_skip": float(rec.advantage_over_skip_v1),
            "advantage_over_skip_long": float(q[1] - q[0]),
            "advantage_over_skip_short": float(q[2] - q[0]),
            "confidence_softmax": [float(p) for p in rec.confidence_softmax_v1],
            "xgb": xgb_this,
            "v10_path_quality_pred": float(v10_out["path_quality"]),
            "v10_mfe_pred_at_entry": float(v10_out["mfe_first_n"]),
            "v10_p_long": float(v10_out["direction_probs"][0]),
            "v10_p_short": float(v10_out["direction_probs"][1]),
            "v10_tradable_prob": float(v10_out["tradable_prob"]),
            "v10_bad_path_prob": float(v10_out["bad_path_prob"]),
            # V10 v3+ aux heads (only present when retrained with those flags).
            # Use .get() so older v_FIXED bundles don't break the dict.
            "v10_tf_agreement_pred": float(v10_out.get("tf_agreement_pred", -1.0)),
            "v10_path_quality_std": float(v10_out.get("path_quality_std", -1.0)),
            "v10_position_size_pred": float(v10_out.get("position_size_pred", -1.0)),
            "v10_hold_horizon_pred": float(v10_out.get("hold_horizon_pred", -1.0)),
            "v10_hold_horizon_bars_pred": int(v10_out.get("hold_horizon_bars_pred", -1)),
            "decision_ts": str(augmented.index[end_idx]),
            "_v10_snapshot": v10_out,   # for later TradeState.open()
            "stub": False,
            # Online-IQL prep payload (2026-05-30). state_v1 is the 192-dim raw
            # vector Entry-IQL saw at this poll. Schema constants (feature_names,
            # k_horizons, variant, fold, aggregator) DON'T change per poll — they
            # live in the bundle dir referenced by PROJECT_STATE_artifacts.json
            # so we don't pay disk for redundancy.
            "entry_iql_state_v1": state_list,
            "entry_iql_q_per_action_per_k_v1": q_per_k_list,
        }
        # IN-PROCESS SHADOW fields (only when a candidate is loaded — journals
        # stay byte-identical when GX1_SHADOW_BUNDLE_DIR is unset).
        if shadow_rec is not None:
            sq = shadow_rec.q_per_action_v1
            out["shadow_action"] = shadow_rec.action_label_v1
            out["shadow_q_per_action"] = [float(sq[0]), float(sq[1]), float(sq[2])]
            out["shadow_advantage_over_skip_long"] = float(sq[1] - sq[0])
            out["shadow_advantage_over_skip_short"] = float(sq[2] - sq[0])
            out["shadow_agrees_with_live"] = bool(shadow_rec.action_label_v1 == rec.action_label_v1)
            out["shadow_bundle"] = Path(_SHADOW_BUNDLE_DIR).name
        return out

    # ── exit decision ────────────────────────────────────────────────

    def make_exit_decision(
        self,
        trade: TradeState,
        now_minute: pd.Timestamp,
        bid: float,
        ask: float,
        m1_close: float,
    ) -> dict[str, Any]:
        """Run Exit-IQL V12.1 for one M1 bar on an open trade.

        Advances the trade's state (PnL/MFE/MAE/etc.), then queries
        Exit-IQL. Returns dict with HOLD / EXIT_NOW action.
        """
        # Advance bar state first. V4 (R13): thread the latest CLOSED M1 bar's
        # intrabar OHLC so the V3 overlay's peak/trough/atr use the REAL intrabar
        # range (one-truth with the train builder's compute_per_bar_signals),
        # NOT a close-only degrade. _refresh_m1_bar shares the per-minute cache
        # with the current_atr_bps_v1 path (same M1 bar source).
        _m1bar = self._refresh_m1_bar(now_minute)
        if _m1bar is not None:
            trade.update_bar(
                bid=bid, ask=ask, m1_close=m1_close,
                bid_high=_m1bar["bid_high"], bid_low=_m1bar["bid_low"],
                ask_high=_m1bar["ask_high"], ask_low=_m1bar["ask_low"],
            )
        else:
            trade.update_bar(bid=bid, ask=ask, m1_close=m1_close)

        if not self._refresh_canonical(now_minute):
            return {
                "action": "HOLD", "action_id": 0, "stub": False,
                "error": "no_canonical_data",
                "bars_in_trade": trade.bars_in_trade,
                "current_pnl_bps": trade.current_pnl_bps,
            }

        augmented = self._last_augmented
        # Use the M5 bucket that contains this M1 minute
        m5_bucket = now_minute.floor("5min")
        if m5_bucket not in augmented.index:
            # Use latest available bar as fallback
            cv3_row = augmented.iloc[-1]
        else:
            cv3_row = augmented.loc[m5_bucket]

        # last_atr_bps = latest M5 atr (journal/diagnostic + from_dict backfill only).
        # NOTE: the V3 overlay's atr_bps_now is NO LONGER sourced here — V4 records a
        # per-M1-bar atr in update_bar (intrabar (ask_high-bid_low)/mid), the one-truth
        # builder basis. This M5 value is kept for the per-bar journal field.
        trade.last_atr_bps = float(cv3_row.get("atr_bps", 0.0) or 0.0)

        # Run V3 v8 inference with trade-state overlay (B3 wire-up)
        v3_v8_out = None
        try:
            overlay = trade.build_v3_overlay() if trade.bars_in_trade > 0 else None
            # V12.2: fetch multi-TF windows if V3 bundle requires them
            v3_mtf_windows = None
            if getattr(self.v3, "_enable_multi_tf", False):
                v3_mtf_windows = self.prebuilt_loader.get_multi_tf_windows(pd.Timestamp(now_minute))
                if not v3_mtf_windows:
                    LOG.warning("V3 multi-TF needed but loader missing features — call build_multi_tf_features()")
            v3_v8_out = self.v3.predict(
                end_ts=pd.Timestamp(now_minute),
                base34_prebuilt=self.prebuilt_loader._base28,
                canonical_v3_window=augmented,
                xgb_inferer=self.xgb,
                trade_overlay=overlay,
                multi_tf_windows=v3_mtf_windows,
            )
            # Update trade with V3 output → maintains running stats for next bar
            trade.update_v3(v3_v8_out)
        except Exception as exc:
            LOG.warning(f"V3 v8 inference failed: {exc}; using zero fallback")
            v3_v8_out = None

        # C1 fix 2026-05-19: training uses per-M1-bar (ask_high - bid_low)/mid bps
        # (typical 3-7 bps), live had been using canonical M5 ATR14 (10-50 bps).
        # 10× distribution shift on a feature Exit-IQL depends on.
        m1_atr_bps = self._refresh_m1_atr_bps(now_minute)
        rec, bar_state = self.exit_iql.decide_for_trade(
            trade, cv3_row, v3_v8_out=v3_v8_out,
            current_m1_atr_bps_override=m1_atr_bps if m1_atr_bps > 0 else None,
            now_minute=now_minute,  # EX1: serve m5_phase = minute%5 of the live M1 bar (== trainer)
        )
        # Inject V3-tracking running stats into bar_state (overwriting any prior 0-fills)
        bar_state.update(trade.build_v3_tracking_features())
        # Surface raw Exit-IQL Q-values for diagnostics (was None before, made
        # debugging premature exits impossible). q_per_action_v1 = [q_hold, q_exit].
        iql_rec = rec.iql_recommendation_v1
        q_hold = float(iql_rec.q_per_action_v1[0])
        q_exit = float(iql_rec.q_per_action_v1[1])
        # Diagnostic: cast every bar_state value to a JSON-serializable scalar
        # so the runner can log the full 204-feature state to journal.
        bar_state_clean = {}
        for k, v in bar_state.items():
            try:
                bar_state_clean[k] = float(v)
            except (TypeError, ValueError):
                bar_state_clean[k] = str(v)
        return {
            "action": rec.action_label_v1,
            "action_id": int(rec.action_id_v1),
            "decision_source": rec.decision_source_v1,
            "v3_should_exit_prob": float(rec.v3_should_exit_prob_v1),
            "q_hold": q_hold,
            "q_exit": q_exit,
            "q_advantage": float(iql_rec.advantage_exit_over_hold_v1),
            "bar_state": bar_state_clean,
            "bars_in_trade": int(trade.bars_in_trade),
            "current_pnl_bps": float(trade.current_pnl_bps),
            "cum_mfe_bps": float(trade.cum_mfe_bps),
            "cum_mae_bps": float(trade.cum_mae_bps),
            "stub": False,
        }


# Backwards-compat module-level callable so existing paper-runner code
# that calls `make_v12_decision(features_snapshot)` can be lifted with
# minimal changes. The caller provides a singleton V12Pipeline.

_GLOBAL_PIPELINE: V12Pipeline | None = None


def get_global_pipeline() -> V12Pipeline:
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = V12Pipeline.load_default()
    return _GLOBAL_PIPELINE
