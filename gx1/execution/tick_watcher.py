"""
TickWatcher: live tick monitoring for TP/SL/BE triggers (callback-based).
"""

from __future__ import annotations

import json as jsonlib
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import requests

from gx1.utils.pnl import compute_pnl_bps

log = logging.getLogger(__name__)


@dataclass
class OpenPos:
    trade_id: str
    direction: str  # "LONG" or "SHORT"
    entry_px: float
    entry_bid: float
    entry_ask: float
    units: int
    entry_ts: datetime
    tp_bps: int
    sl_bps: int
    be_active: bool = False
    be_price: float | None = None  # if BE price is already set
    early_stop_bps: int = 0  # optional, e.g., 40 bps first 10-12 min


class TickWatcher:
    def __init__(self, *, host, stream_host, account_id, api_key, instrument, cfg, logger, close_cb, get_positions_cb):
        """
        close_cb(pos, reason, px, pnl_bps) -> should close position immediately
        get_positions_cb() -> List[OpenPos]
        """
        self.host = host.rstrip("/")
        self.stream_host = stream_host.rstrip("/")
        self.account_id = account_id
        self.api_key = api_key
        self.instrument = instrument
        self.cfg = cfg
        self.logger = logger
        self.close_cb = close_cb
        self.get_positions_cb = get_positions_cb
        self.update_be_callback = None  # Callback to update trade.extra when BE is activated

        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        self._last_check = datetime.now(timezone.utc)

    # ---------- public API ----------
    def start(self) -> None:
        if not self.cfg.get("enabled", False):
            self.logger.info("[TICK] disabled in policy")
            return
        if self._t and self._t.is_alive():
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._run_loop, name="TickWatcher", daemon=True)
        self._t.start()
        self.logger.info("[TICK] watcher started")

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=2.0)
        self.logger.info("[TICK] watcher stopped")

    # ---------- internals ----------
    def _run_loop(self) -> None:
        # loop: try stream first (using stream_host), fall back to snapshot-polling
        backoff = 0.2
        while not self._stop.is_set():
            try:
                if self.cfg.get("stream", True):
                    # Try stream first (using stream_host for pricing stream endpoint)
                    try:
                        self._stream_pricing()
                        # If stream works, it will loop indefinitely until _stop is set
                        # If stream fails (404), it will return and fall through to snapshot
                    except Exception as stream_error:
                        # Stream failed, fall back to snapshot polling
                        self.logger.warning(f"[TICK] Stream failed: {stream_error!r}, falling back to snapshot polling")
                        self.cfg["stream"] = False  # Disable stream mode, use snapshot instead
                
                # Use snapshot polling (always works with OANDA v3 API)
                if not self.cfg.get("stream", True):
                    self._poll_snapshot()
            except Exception as e:
                self.logger.warning(f"[TICK] loop error: {e!r}")
                time.sleep(min(backoff, float(self.cfg.get("max_backoff_s", 5))))
                backoff = min(backoff * 2.0, float(self.cfg.get("max_backoff_s", 5)))
            else:
                backoff = 0.2  # reset if normal return

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"}

    def _stream_pricing(self) -> None:
        # OANDA pricing stream endpoint (uses stream_host, not REST host)
        # URL: https://stream-fxpractice.oanda.com/v3/accounts/{accountID}/pricing/stream
        url = f"{self.stream_host}/v3/accounts/{self.account_id}/pricing/stream"
        params = {"instruments": self.instrument}
        with requests.get(url, headers=self._headers(), params=params, stream=True, timeout=30) as r:
            r.raise_for_status()
            self.logger.info("[TICK] Pricing stream connected (using stream_host)")
            for line in r.iter_lines(decode_unicode=True):
                if self._stop.is_set():
                    return
                if not line:
                    continue
                try:
                    data = jsonlib.loads(line)
                except jsonlib.JSONDecodeError:
                    continue
                if data.get("type") not in ("PRICE", "HEARTBEAT"):
                    continue
                if data.get("type") == "HEARTBEAT":
                    continue
                bids = data.get("bids") or []
                asks = data.get("asks") or []
                if not bids or not asks:
                    continue
                # use bid/ask for exits
                bid = float(bids[0]["price"])
                ask = float(asks[0]["price"])
                ts = self._iso_to_utc(data.get("time"))
                self._on_tick(ts, bid, ask)

    def _poll_snapshot(self) -> None:
        # fallback: snapshot every N ms
        url = f"{self.host}/v3/accounts/{self.account_id}/pricing"
        period = float(self.cfg.get("snapshot_ms", 400)) / 1000.0
        while not self._stop.is_set():
            try:
                r = requests.get(url, headers=self._headers(), params={"instruments": self.instrument}, timeout=10)
                r.raise_for_status()
                data = r.json()
                prices = data.get("prices") or []
                if not prices:
                    raise RuntimeError("No prices returned in snapshot")
                price = prices[0]
                bids = price.get("bids") or []
                asks = price.get("asks") or []
                if not bids or not asks:
                    raise RuntimeError("Missing bid/ask in snapshot")
                bid = float(bids[0]["price"])
                ask = float(asks[0]["price"])
                ts = self._iso_to_utc(price.get("time"))
                self._on_tick(ts, bid, ask)
            except Exception as e:
                self.logger.warning(f"[TICK] snapshot error: {e!r}")
            time.sleep(period)

    def _iso_to_utc(self, s: Optional[str]):
        if not s:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    def _on_tick(self, ts: datetime, bid: float, ask: float) -> None:
        positions = self.get_positions_cb()
        if not positions:
            return
        now = datetime.now(timezone.utc)
        delta = now - self._last_check
        min_time_between = float(self.cfg.get("min_time_between_checks_s", 0.5))
        if delta.total_seconds() < min_time_between:
            return
        self._last_check = now

        # Debug-only: track lateness
        if self.cfg.get("debug_timing", False):
            self.logger.debug("[TICK] tick_ts=%s now=%s delta_ms=%.1f bid=%.5f ask=%.5f", ts, now, delta.total_seconds() * 1000, bid, ask)

        for pos in positions:
            pnl_bps = self._pnl_bps(pos, bid, ask)
            px = ask if pos.direction.upper() == "LONG" else bid

            # Break-even activation (if enabled)
            be_cfg = self.cfg.get("be", {})
            if be_cfg.get("enabled", False) and not pos.be_active:
                be_activate_at_bps = int(be_cfg.get("activate_at_bps", 50))
                be_bias_price = float(be_cfg.get("bias_price", 0.3))
                if pnl_bps >= be_activate_at_bps:
                    pos.be_active = True
                    pos.be_price = self._compute_be_price(pos.direction, pos.entry_px, be_bias_price)
                    
                    # Sync BE status back to trade.extra (so it persists across cycles)
                    # Call callback to update trade.extra (thread-safe)
                    try:
                        if self.update_be_callback:
                            self.update_be_callback(pos.trade_id, be_active=True, be_price=pos.be_price)
                    except Exception as e:
                        self.logger.warning(f"[TICK] Failed to sync BE status to trade.extra: {e!r}")
            
            # Break-even check (if already active)
            if pos.be_active and pos.be_price is not None:
                # if we've fallen back to BE or below (for LONG) – or above for SHORT
                if (pos.direction == "LONG" and px <= pos.be_price) or (pos.direction == "SHORT" and px >= pos.be_price):
                    self._try_close(pos, "BE_TICK", px, pnl_bps)
                    continue

            # TP/SL
            if pnl_bps >= pos.tp_bps:
                self._try_close(pos, "TP_TICK", px, pnl_bps)
                continue
            if pnl_bps <= -pos.sl_bps:
                self._try_close(pos, "SL_TICK", px, pnl_bps)
                continue
            
            # Soft-stop: cut losing trades early (anytime, not time-based)
            soft_stop_bps = int(self.cfg.get("soft_stop_bps", 0))
            if soft_stop_bps > 0 and pnl_bps <= -soft_stop_bps:
                self._try_close(pos, "SOFT_STOP_TICK", px, pnl_bps)
                continue

    def _compute_be_price(self, direction: str, entry_px: float, bias_price: float) -> float:
        if direction == "LONG":
            return entry_px + (bias_price / 10000.0)
        return entry_px - (bias_price / 10000.0)

    def _pnl_bps(self, pos: OpenPos, bid_now: float, ask_now: float) -> float:
        side = "long" if pos.direction.upper() == "LONG" else "short"
        # Add defensive logging for first N calls
        if not hasattr(self, "_pnl_log_count"):
            self._pnl_log_count = 0
        if self._pnl_log_count < 5:
            log.debug(
                "[PNL] Using bid/ask PnL: entry_bid=%.5f entry_ask=%.5f exit_bid=%.5f exit_ask=%.5f side=%s",
                pos.entry_bid, pos.entry_ask, bid_now, ask_now, side
            )
            self._pnl_log_count += 1
        return compute_pnl_bps(pos.entry_bid, pos.entry_ask, bid_now, ask_now, side)

    def _try_close(self, pos: OpenPos, reason: str, px: float, pnl_bps: float):
        try:
            self.close_cb(pos, reason, px, pnl_bps)
        except Exception as e:
            self.logger.error(f"[TICK] close_cb error trade_id={pos.trade_id} reason={reason}: {e!r}")


if __name__ == "__main__":
    print("TickWatcher module (ARCHIVE MOVE) - no runtime entrypoint")
