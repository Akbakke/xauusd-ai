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

import logging
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
        LOG.info(f"V12Pipeline loaded in {(time.perf_counter()-t0)*1000:.0f} ms")
        LOG.info(f"  prebuilt cutoff: {loader.cutoff_ts}")
        return cls(prebuilt_loader=loader, xgb=xgb, v10=v10,
                    entry_iql=entry_iql, exit_iql=exit_iql, v3=v3)

    # ── shared canonical_v3 build (cached per M5 bucket) ───────────────

    def _refresh_canonical(self, now_minute: pd.Timestamp) -> bool:
        """Refresh augmented window + XGB bridge from disk prebuilt if a new M5 bucket.
        Returns True if data available, False if past prebuilt cutoff."""
        cur_bucket = now_minute.floor("5min")
        if self._last_augmented_bucket == cur_bucket and self._last_augmented is not None:
            return True

        # Read 96-bar window directly from canonical_v3 + BASE28 prebuilts.
        # Identical values to what V12 cascade trainings saw — no live recompute.
        augmented = self.prebuilt_loader.get_window(now_minute, n_bars=V10_SEQ_LEN)
        if augmented.empty:
            LOG.warning(f"prebuilt empty/past cutoff for {now_minute} "
                         f"(loader cutoff: {self.prebuilt_loader.cutoff_ts})")
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
        v10_out = self.v10.predict(augmented, bridge, end_idx=end_idx)

        # Entry-IQL input
        row = augmented.iloc[end_idx]
        xgb_this = {
            "p_long": float(self._last_xgb_p_long[end_idx]),
            "p_short": float(self._last_xgb_p_short[end_idx]),
            "p_flat": float(self._last_xgb_p_flat[end_idx]),
        }
        rec, candidate = self.entry_iql.predict_from_pipeline(row, xgb_this, v10_out)

        q = rec.q_per_action_v1
        return {
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
            "decision_ts": str(augmented.index[end_idx]),
            "_v10_snapshot": v10_out,   # for later TradeState.open()
            "stub": False,
        }

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
        # Advance bar state first
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

        # Update trade's atr_bps from latest M5 bar (for V3 overlay's atr_bps_now)
        trade.last_atr_bps = float(cv3_row.get("atr_bps", 0.0) or 0.0)

        # Run V3 v8 inference with trade-state overlay (B3 wire-up)
        v3_v8_out = None
        try:
            overlay = trade.build_v3_overlay() if trade.bars_in_trade > 0 else None
            v3_v8_out = self.v3.predict(
                end_ts=pd.Timestamp(now_minute),
                base34_prebuilt=self.prebuilt_loader._base28,
                canonical_v3_window=augmented,
                xgb_inferer=self.xgb,
                trade_overlay=overlay,
            )
            # Update trade with V3 output → maintains running stats for next bar
            trade.update_v3(v3_v8_out)
        except Exception as exc:
            LOG.warning(f"V3 v8 inference failed: {exc}; using zero fallback")
            v3_v8_out = None

        rec, bar_state = self.exit_iql.decide_for_trade(
            trade, cv3_row, v3_v8_out=v3_v8_out,
        )
        # Inject V3-tracking running stats into bar_state (overwriting any prior 0-fills)
        bar_state.update(trade.build_v3_tracking_features())
        return {
            "action": rec.action_label_v1,
            "action_id": int(rec.action_id_v1),
            "decision_source": rec.decision_source_v1,
            "v3_should_exit_prob": float(rec.v3_should_exit_prob_v1),
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
