# ARCHIVED: ENTRY_ONLY diagnostic mode removed from runtime; non-canonical.
"""
Preserves the prior ENTRY_ONLY implementation from gx1.execution.oanda_demo_runner.GX1DemoRunner.

This mode logged hypothetical entries to CSV without opening trades or running exits.
It is no longer supported at runtime; kept here for reference (schema and behavior).
"""

from __future__ import annotations

import csv
import os
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def init_entry_only_mode(self, chunk_id: str) -> None:
    """Legacy setup: configure entry-only CSV path and buffers."""
    if chunk_id:
        entry_only_log_path = Path(f"gx1/live/entry_only_log_v9_test_chunk_{chunk_id}.csv")
    else:
        entry_only_log_path = Path("gx1/live/entry_only_log_v9_test.csv")
    entry_only_log_path.parent.mkdir(parents=True, exist_ok=True)
    self.entry_only_log_path = entry_only_log_path
    self._entry_only_log_buffer = []
    self._entry_only_log_buffer_size = 500
    log.info(
        "[ENTRY_ONLY] Entry-only logging enabled: %s (buffered, flush every %d entries)",
        self.entry_only_log_path,
        self._entry_only_log_buffer_size,
    )


def _log_entry_only_event_impl(
    self,
    timestamp: pd.Timestamp,
    side: str,
    price: float,
    prediction: Any,  # EntryPrediction-like object
    policy_state: Dict[str, Any],
) -> None:
    """Legacy: buffer a hypothetical entry event."""
    if getattr(self, "entry_only_log_path", None) is None:
        return

    trend_regime = policy_state.get("brain_trend_regime", "UNKNOWN")
    vol_regime = policy_state.get("brain_vol_regime", "UNKNOWN")
    session = policy_state.get("session", "UNKNOWN")

    row = {
        "run_id": getattr(self, "run_id", None),
        "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
        "side": side.lower(),
        "entry_price": f"{price:.3f}",
        "session": session,
        "trend_regime": trend_regime,
        "vol_regime": vol_regime,
        "p_long_entry": f"{getattr(prediction, 'prob_long', 0.0):.4f}" if hasattr(prediction, "prob_long") else "",
        "p_short_entry": f"{getattr(prediction, 'prob_short', 0.0):.4f}" if hasattr(prediction, "prob_short") else "",
        "margin_entry": f"{getattr(prediction, 'margin', 0.0):.4f}" if hasattr(prediction, "margin") else "",
        "p_hat_entry": f"{getattr(prediction, 'p_hat', 0.0):.4f}" if hasattr(prediction, "p_hat") else "",
    }

    if not hasattr(self, "_entry_only_log_buffer"):
        self._entry_only_log_buffer = []
        self._entry_only_log_buffer_size = 100

    self._entry_only_log_buffer.append(row)
    if len(self._entry_only_log_buffer) >= self._entry_only_log_buffer_size:
        _flush_entry_only_log_buffer_impl(self)


def _flush_entry_only_log_buffer_impl(self) -> None:
    """Legacy: flush buffered entry-only rows to CSV."""
    if not hasattr(self, "_entry_only_log_buffer") or len(self._entry_only_log_buffer) == 0:
        return
    if getattr(self, "entry_only_log_path", None) is None:
        return

    fieldnames = [
        "run_id",
        "timestamp",
        "side",
        "entry_price",
        "session",
        "trend_regime",
        "vol_regime",
        "p_long_entry",
        "p_short_entry",
        "margin_entry",
        "p_hat_entry",
        "p_long_v7",
        "p_short_v7",
        "margin_v7",
        "p_hat_v7",
    ]

    file_exists = self.entry_only_log_path.exists()
    with self.entry_only_log_path.open("a", newline="", encoding="utf-8", buffering=8192) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(self._entry_only_log_buffer)

    buffer_size = len(self._entry_only_log_buffer)
    self._entry_only_log_buffer = []
    if not hasattr(self, "_flush_count"):
        self._flush_count = 0
    self._flush_count += 1
    if os.getenv("GX1_ENTRY_ONLY_DEBUG", "0") == "1":
        log.debug("[ENTRY_ONLY] Flushed %d entries to CSV (total flushes: %d)", buffer_size, self._flush_count)


def entry_only_per_bar_logging_example(
    df: pd.DataFrame,
    i: int,
    ts: pd.Timestamp,
    trade,
    entry_only_log_path: Path,
) -> None:
    """
    Snapshot of the per-bar ENTRY_ONLY logging (extended schema) that was removed.
    Retained for schema reference and analytical reproducibility.
    """
    file_exists = entry_only_log_path.exists()
    fieldnames = [
        "run_id", "timestamp", "side", "entry_price",
        "session", "session_id", "trend_regime", "atr_regime", "vol_regime",
        "hour_of_day", "day_of_week",
        "p_long_entry", "p_short_entry", "margin_entry", "p_hat_entry",
        "body_pct", "wick_asym", "bar_range", "atr_bps",
        "htf_context_h1", "htf_context_h4",
        "mfe_5b", "mae_5b", "t_mfe", "t_mae",
        "next_vol_regime", "next_session",
    ]

    current_bar = df.iloc[-1]
    extra = trade.extra if hasattr(trade, "extra") and trade.extra else {}
    session = extra.get("session", current_bar.get("session", "UNKNOWN"))
    session_id = current_bar.get("session_id", current_bar.get("_v1_session_tag", "UNKNOWN"))
    atr_regime = extra.get("atr_regime", current_bar.get("atr_regime", "UNKNOWN"))
    vol_regime = extra.get("brain_vol_regime", atr_regime)
    trend_regime = extra.get("brain_trend_regime", current_bar.get("trend_regime_tf24h", "UNKNOWN"))
    hour_of_day = ts.hour
    day_of_week = ts.dayofweek
    body_pct = current_bar.get("body_pct", current_bar.get("_v1_body_tr", 0.0))
    wick_asym = current_bar.get("wick_asym", 0.0)
    bar_range = (
        (current_bar.get("high", 0) - current_bar.get("low", 0)) / current_bar.get("close", 1) * 10000
        if "high" in current_bar.index and "low" in current_bar.index
        else 0.0
    )
    htf_h1 = current_bar.get("_v1_int_slope_h1_us", 0.0)
    htf_h4 = current_bar.get("_v1_int_slope_h4_atr", 0.0)

    mfe_5b = mae_5b = t_mfe = t_mae = np.nan
    next_vol_regime = "UNKNOWN"
    next_session = "UNKNOWN"
    if i + 5 < len(df):
        future_bars = df.iloc[i + 1 : i + 6]
        if len(future_bars) > 0 and "high" in future_bars.columns and "low" in future_bars.columns:
            entry_price = float(current_bar["close"])
            highs = future_bars["high"].values
            lows = future_bars["low"].values
            mfe_5b = (np.max(highs) - entry_price) / entry_price * 10000.0
            mae_5b = (entry_price - np.min(lows)) / entry_price * 10000.0
            mfe_idx = np.argmax(highs)
            mae_idx = np.argmin(lows)
            t_mfe = float(mfe_idx + 1)
            t_mae = float(mae_idx + 1)
            if i + 1 < len(df):
                next_bar = df.iloc[i + 1]
                next_vol_regime = next_bar.get("atr_regime", next_bar.get("brain_vol_regime", "UNKNOWN"))
                next_session = next_bar.get("session", next_bar.get("_v1_session_tag", "UNKNOWN"))

    with open(entry_only_log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": getattr(trade, "run_id", None),
                "timestamp": ts.isoformat(),
                "side": trade.side,
                "entry_price": f"{trade.entry_price:.3f}",
                "session": session,
                "session_id": str(session_id),
                "trend_regime": trend_regime,
                "atr_regime": atr_regime,
                "vol_regime": vol_regime,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "p_long_entry": f"{trade.entry_prob_long:.4f}",
                "p_short_entry": f"{trade.entry_prob_short:.4f}",
                "margin_entry": f"{getattr(trade, 'margin', 0.0):.4f}" if hasattr(trade, "margin") else "",
                "p_hat_entry": f"{getattr(trade, 'p_hat', 0.0):.4f}" if hasattr(trade, "p_hat") else "",
                "body_pct": f"{body_pct:.4f}" if not np.isnan(body_pct) else "",
                "wick_asym": f"{wick_asym:.4f}" if not np.isnan(wick_asym) else "",
                "bar_range": f"{bar_range:.2f}" if not np.isnan(bar_range) else "",
                "atr_bps": f"{trade.atr_bps:.2f}",
                "htf_context_h1": f"{htf_h1:.4f}" if not np.isnan(htf_h1) else "",
                "htf_context_h4": f"{htf_h4:.4f}" if not np.isnan(htf_h4) else "",
                "mfe_5b": f"{mfe_5b:.2f}" if not np.isnan(mfe_5b) else "",
                "mae_5b": f"{mae_5b:.2f}" if not np.isnan(mae_5b) else "",
                "t_mfe": f"{t_mfe:.1f}" if not np.isnan(t_mfe) else "",
                "t_mae": f"{t_mae:.1f}" if not np.isnan(t_mae) else "",
                "next_vol_regime": next_vol_regime,
                "next_session": next_session,
            }
        )
