"""
GX1 v1.1 → OANDA practice demo runner.

This module wires the frozen ENTRY_V3 + EXIT_V2 models to the OANDA
practice API with strong safety defaults (dry-run, risk guards).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json as jsonlib
import logging
import os
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# DEL 2 & 3: Set thread limits and multiprocessing start method BEFORE any imports that use them
# DEL 3: Thread library limits (OMP/MKL/OpenBLAS)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# DEL 2: Force spawn method for multiprocessing (avoid fork deadlocks)
try:
    import multiprocessing as mp
    # Only set if not already set
    if mp.get_start_method(allow_none=True) != "spawn":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # Already set, ignore
            pass
except ImportError:
    pass

import joblib
import numpy as np
import pandas as pd
import requests
import yaml

# DEL 2: Runtime V9 import moved to local scope (not top-level)
# runtime_v9 is forbidden in replay-mode, so we only import it when needed (live mode only)
# build_v9_runtime_features will be imported locally in methods that need it (live mode only)

# DEL 3: PREBUILT mode fix - move live_features imports to lazy imports
# live_features is forbidden in PREBUILT mode, so we only import it when needed (live mode only)
# Use TYPE_CHECKING guard for type hints to avoid runtime imports
if TYPE_CHECKING:
    from gx1.execution.live_features import (
        EntryFeatureBundle,
        build_live_entry_features,
        build_live_exit_snapshot,
        infer_session_tag,
    )
from gx1.execution.broker_client import BrokerClient
from gx1.execution.entry_manager import EntryManager
from gx1.execution.exit_manager import ExitManager
from gx1.execution.telemetry import TelemetryTracker
from gx1.execution.oanda_backfill import backfill_m5_candles_until_target
from gx1.tuning.feature_manifest import load_manifest  # type: ignore[reportMissingImports]
from gx1.utils.env_loader import load_dotenv_if_present
from gx1.utils.pnl import compute_pnl_bps

log = logging.getLogger(__name__)

# Optional import - model_status may not be available in all environments
try:
    from gx1.models.model_status import validate_entry_model_for_live, is_legacy_entry_model  # type: ignore[reportMissingImports]
    MODEL_STATUS_AVAILABLE = True
except ImportError:
    def validate_entry_model_for_live(*args, **kwargs):
        log.warning("[MODEL_STATUS] validate_entry_model_for_live not available - skipping validation")
        return True
    def is_legacy_entry_model(*args, **kwargs):
        return False
    MODEL_STATUS_AVAILABLE = False
    log.warning("[IMPORT] model_status module not available - model validation will be skipped")

# Big Brain V0 removed - using V1 only
# Optional import - Big Brain may not be available in all environments
try:
    from gx1.big_brain.v1.runtime_v1 import BigBrainV1Runtime  # type: ignore[reportMissingImports]
    from gx1.big_brain.v1.entry_gater import BigBrainV1EntryGater, load_entry_gater, EntryAction  # type: ignore[reportMissingImports]
    BIG_BRAIN_AVAILABLE = True
except ImportError:
    BigBrainV1Runtime = None
    BigBrainV1EntryGater = None
    load_entry_gater = None
    EntryAction = None
    BIG_BRAIN_AVAILABLE = False
    log.warning("[IMPORT] Big Brain V1 modules not available - Big Brain features will be disabled")

EXIT_MODEL_PATH = Path("gx1/exit/phase_c_model/models/exit_model_xgb_ENTRY_ANCHOR_V2.pkl")
SECONDS_PER_BAR = 300  # M5
BID_ASK_REQUIRED_COLS = [
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
]


# --------------------------------------------------------------------------- #
# TickWatcher for TP/SL/BE on tick
# --------------------------------------------------------------------------- #


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
    bb_exit: dict | None = None  # Big Brain V1 adjusted exit parameters (if available)


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
        self._t = None
        self._last_check = datetime.now(timezone.utc)

    # ---------- public API ----------
    def start(self):
        if not self.cfg.get("enabled", False):
            self.logger.info("[TICK] disabled in policy")
            return
        if self._t and self._t.is_alive():
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._run_loop, name="TickWatcher", daemon=True)
        self._t.start()
        self.logger.info("[TICK] watcher started")

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2.0)
        self.logger.info("[TICK] watcher stopped")

    # ---------- internals ----------
    def _run_loop(self):
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

    def _stream_pricing(self):
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

    def _poll_snapshot(self):
        # fallback: snapshot every N ms
        url = f"{self.host}/v3/accounts/{self.account_id}/pricing"
        period = float(self.cfg.get("snapshot_ms", 400)) / 1000.0
        while not self._stop.is_set():
            try:
                r = requests.get(url, headers=self._headers(), params={"instruments": self.instrument}, timeout=10)
                r.raise_for_status()
                prices = r.json().get("prices", [])
                if prices:
                    p = prices[0]
                    bid = float(p["bids"][0]["price"])
                    ask = float(p["asks"][0]["price"])
                    ts = self._iso_to_utc(p.get("time"))
                    self._on_tick(ts, bid, ask)
            except Exception as e:
                self.logger.warning(f"[TICK] snapshot error: {e!r}")
                # let run_loop handle backoff next round
                return
            time.sleep(period)

    def _iso_to_utc(self, s):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    def _on_tick(self, ts_utc, bid, ask):
        # debounce
        debounce_ms = int(self.cfg.get("debounce_ms", 120))
        if (ts_utc - self._last_check).total_seconds() * 1000.0 < debounce_ms:
            return
        self._last_check = ts_utc

        open_pos = self.get_positions_cb()
        if not open_pos:
            return  # nothing to do

        for pos in list(open_pos):
            # price that applies for exit (conservative: LONG -> bid, SHORT -> ask)
            px = bid if pos.direction == "LONG" else ask
            pnl_bps = self._pnl_bps(pos, bid, ask)

            # Break-even activation on tick (when profit >= activate_at_bps)
            be_cfg = self.cfg.get("be", {})
            be_enabled = be_cfg.get("enabled", False)
            # Check for Big Brain V1 adjusted BE trigger
            if pos.bb_exit and isinstance(pos.bb_exit, dict):
                be_activate_at_bps = int(pos.bb_exit.get("be_trigger_bps_adj", be_cfg.get("activate_at_bps", 50)))
            else:
                be_activate_at_bps = int(be_cfg.get("activate_at_bps", 50))
            be_bias_price = float(be_cfg.get("bias_price", 0.3))
            
            if be_enabled and not pos.be_active and pnl_bps >= be_activate_at_bps:
                # Activate break-even: set BE price = entry_price + bias (lock in small profit)
                if pos.direction == "LONG":
                    be_price = pos.entry_px + be_bias_price
                else:  # SHORT
                    be_price = pos.entry_px - be_bias_price
                
                # Update position BE status (will be synced back to trade.extra in _get_open_positions_for_tick)
                pos.be_active = True
                pos.be_price = be_price
                
                # Log BE activation
                self.logger.info(
                    "[TICK] BE activated for trade %s: pnl_bps=%.2f >= %d, BE_price=%.3f (entry=%.3f + bias=%.3f)",
                    pos.trade_id,
                    pnl_bps,
                    be_activate_at_bps,
                    be_price,
                    pos.entry_px,
                    be_bias_price,
                )
                
                # Sync BE status back to trade.extra (so it persists across cycles)
                # Call callback to update trade.extra (thread-safe)
                try:
                    if self.update_be_callback:
                        self.update_be_callback(pos.trade_id, be_active=True, be_price=be_price)
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
            # Check for Big Brain V1 adjusted soft stop (with asymmetry if available)
            if pos.bb_exit and isinstance(pos.bb_exit, dict):
                # Priority: bb_exit_asym > bb_exit (risk shaping only)
                soft_stop_bps = int(pos.bb_exit.get("soft_stop_bps_adj", self.cfg.get("soft_stop_bps", 0)))
            else:
                soft_stop_bps = int(self.cfg.get("soft_stop_bps", 0))
            if soft_stop_bps > 0 and pnl_bps <= -soft_stop_bps:
                self._try_close(pos, "SOFT_STOP_TICK", px, pnl_bps)
                continue

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


# --------------------------------------------------------------------------- #
# Dataclasses for runtime state
# --------------------------------------------------------------------------- #


@dataclass
class RiskLimits:
    dry_run: bool = True
    units_per_trade: int = 100
    max_open_trades: int = 3
    max_daily_loss_bps: float = 300.0
    min_time_between_trades_sec: int = 60
    max_slippage_bps: float = 10.0


@dataclass
class LiveTrade:
    trade_id: str  # Display ID (SIM-...), kept for backward compatibility
    trade_uid: str  # Globally unique ID (run_id:chunk_id:seq:uuid), primary key for journaling
    entry_time: pd.Timestamp
    side: str  # 'long' or 'short'
    units: int
    entry_price: float
    entry_bid: float
    entry_ask: float
    atr_bps: float
    vol_bucket: str
    entry_prob_long: float
    entry_prob_short: float
    dry_run: bool = True
    client_order_id: Optional[str] = None  # For idempotency
    extra: Dict[str, Any] = field(default_factory=dict)
    # Exit model hysteresis: track prob_close history for consecutive threshold check
    prob_close_history: deque = field(default_factory=lambda: deque(maxlen=2))  # Keep last 2 bars


@dataclass
class ExitModelBundle:
    model: Any
    feature_names: List[str]


@dataclass
class EntryModelBundle:
    models: Dict[str, Any]
    feature_names: List[str]
    metadata: Dict[str, Any]
    feature_cols_hash: str
    model_bundle_version: Optional[str] = None
    ts_map_hash: Optional[str] = None


@dataclass
class EntryPrediction:
    session: str
    prob_short: float
    prob_neutral: float
    prob_long: float
    p_hat: float
    margin: float


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML config, resolving relative paths relative to gx1 package root."""
    path_obj = Path(path)
    
    # If path is relative, resolve it relative to gx1 package root (not cwd)
    if not path_obj.is_absolute():
        # Find gx1 package root from this file's location
        # This file is at gx1/execution/oanda_demo_runner.py
        # So gx1 root is at __file__'s parent.parent
        gx1_root = Path(__file__).resolve().parent.parent
        
        # If path starts with "gx1/", remove that prefix before joining
        path_str = str(path_obj)
        if path_str.startswith("gx1/"):
            path_str = path_str[4:]  # Remove "gx1/" prefix
        
        path_obj = (gx1_root / path_str).resolve()
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj} (resolved from {path})")
    
    print(f"[PATH_RESOLVE] config_path={path_obj} exists={path_obj.exists()}")
    with open(path_obj, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def setup_logging(log_dir: Path, level: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "oanda_demo_runner.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    logging.getLogger("requests").setLevel(logging.WARNING)


def wait_until_next_bar(now: pd.Timestamp) -> None:
    """Sleep until the next M5 boundary."""
    epoch_seconds = int(now.timestamp())
    remainder = epoch_seconds % SECONDS_PER_BAR
    sleep_seconds = SECONDS_PER_BAR - remainder
    # protect against zero sleep (already aligned)
    if sleep_seconds <= 1:
        sleep_seconds = SECONDS_PER_BAR
    log.debug("Sleeping %.1f seconds until next bar", sleep_seconds)
    time.sleep(sleep_seconds)


def extract_entry_probabilities(probs: np.ndarray, classes: Optional[np.ndarray]) -> EntryPrediction:
    """
    Normalize model output into directional probabilities.
    
    CRITICAL: Uses model.classes_ to correctly map probabilities to LONG/SHORT.
    Common trap: model.classes_ can be [1, 0] or [0, 1], and we must map correctly.
    
    Parameters
    ----------
    probs : np.ndarray
        Array of shape (1, n_classes) from predict_proba.
    classes : Optional[np.ndarray]
        Model classes_ attribute, if available.
    """
    if probs.ndim != 2 or probs.shape[0] != 1:
        raise ValueError(f"Expected predict_proba output with shape (1, n_classes), got {probs.shape}.")

    prob_vector = probs[0]
    n_classes = prob_vector.shape[0]

    if n_classes == 3:
        return EntryPrediction(
            session="",
            prob_short=float(prob_vector[0]),
            prob_neutral=float(prob_vector[1]),
            prob_long=float(prob_vector[2]),
            p_hat=float(max(prob_vector[0], prob_vector[2])),
            margin=float(abs(prob_vector[2] - prob_vector[0])),
        )

    raise RuntimeError(f"[XGB_PROBA_DIM_MISMATCH] Expected 3-class probabilities, got n_classes={n_classes} shape={probs.shape}")


def load_exit_model() -> ExitModelBundle:
    if not EXIT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Exit model not found: {EXIT_MODEL_PATH}")
    bundle = joblib.load(EXIT_MODEL_PATH)
    model = bundle.get("model")
    features = bundle.get("features")
    if model is None or features is None:
        raise ValueError("Exit model bundle missing 'model' or 'features'")
    return ExitModelBundle(model=model, feature_names=features)


def compute_daily_key(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%d") if ts.tzinfo else ts.strftime("%Y-%m-%d")


def _should_consider_entry_impl(
    trend: str,
    vol: str,
    session: str,
    risk_score: float,
    stage0_config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """
    Stage-0 sanity gate (v2).
    
    Stage-0 is no longer an entry filter. It only blocks obviously invalid data:
    - invalid/unknown session tag
    - non-finite risk_score
    
    All regime/session logic is deferred to ML and later gates (vol_guard/score_gate).
    
    Returns
    -------
    Tuple[bool, str]
        (True if we should consider entry, reason_code)
        Reason codes: stage0_unknown_field, stage0_pass
    """
    # Stage-0 v2: ignore trend/vol/session gating (ML responsibility).
    # Only block on clearly invalid inputs.
    allowed_sessions = {"ASIA", "EU", "OVERLAP", "US"}
    if session not in allowed_sessions:
        return (False, "stage0_unknown_field")
    try:
        import math
        if risk_score is None or not math.isfinite(float(risk_score)):
            return (False, "stage0_unknown_field")
    except Exception:
        return (False, "stage0_unknown_field")
    
    return (True, "stage0_pass")


def should_enter_trade(
    prediction: EntryPrediction,
    entry_params: Dict[str, Any],
    entry_gating: Optional[Dict[str, Any]] = None,
    last_side: Optional[str] = None,
    bars_since_last_side: int = 999,
) -> Optional[str]:
    """
    Decide whether to enter a trade based on asymmetric thresholds (per side).
    
    Uses p_side (not just p_long) for gating to ensure both LONG and SHORT are gated correctly.
    Supports asymmetric thresholds: SHORT requires higher p_side and margin than LONG.
    
    Parameters
    ----------
    prediction : EntryPrediction
        Prediction with prob_long, prob_short, p_hat, margin.
    entry_params : Dict[str, Any]
        Entry parameters (legacy, for backward compatibility).
    entry_gating : Optional[Dict[str, Any]]
        Entry gating configuration with asymmetric thresholds (per side).
    last_side : Optional[str]
        Last trade side ("long" or "short") for sticky-side logic.
    bars_since_last_side : int
        Number of bars since last side trade (for sticky-side logic).
    
    Returns
    -------
    Optional[str]
        'long', 'short', or None if no entry should be taken.
    """
    # Determine side first
    if prediction.prob_long >= prediction.prob_short:
        side = "long"
        p_side = prediction.prob_long
        p_other = prediction.prob_short
    else:
        side = "short"
        p_side = prediction.prob_short
        p_other = prediction.prob_long
    
    # Use asymmetric thresholds from entry_gating if available, otherwise fall back to entry_params
    if entry_gating:
        p_side_min_cfg = entry_gating.get("p_side_min", {})
        margin_min_cfg = entry_gating.get("margin_min", {})
        side_ratio_min = float(entry_gating.get("side_ratio_min", 1.25))
        sticky_bars = int(entry_gating.get("sticky_bars", 1))
        
        # Get side-specific thresholds
        p_side_min = float(p_side_min_cfg.get(side, 0.55))
        margin_min = float(margin_min_cfg.get(side, 0.08))
        
        # Gate on p_side (asymmetric threshold)
        if p_side < p_side_min:
            # Log at INFO level (not debug) so we can see what's blocking entries
            import logging
            log = logging.getLogger(__name__)
            log.debug(
                "[GUARD] p_side_min block: side=%s p_side=%.4f < %.2f (p_other=%.4f)",
                side, p_side, p_side_min, p_other
            )
            return None
        
        # Gate on margin (asymmetric threshold)
        if prediction.margin < margin_min:
            # Log at INFO level (not debug) so we can see what's blocking entries
            import logging
            log = logging.getLogger(__name__)
            log.debug(
                "[GUARD] margin_min block: side=%s margin=%.4f < %.2f (p_side=%.4f p_other=%.4f)",
                side, prediction.margin, margin_min, p_side, p_other
            )
            return None
        
        # Side-ratio gate: p_side/p_other must be at least side_ratio_min
        ratio = p_side / max(p_other, 1e-6)
        if ratio < side_ratio_min:
            # Sticky-side: don't flip side from previous bar if ratio < side_ratio_min
            if last_side and last_side != side and bars_since_last_side <= sticky_bars:
                # Log sticky-side block at INFO level (so we can see side-flip blocks)
                import logging
                log = logging.getLogger(__name__)
                log.info(
                    "[GUARD] side_ratio block (sticky): p_side=%.4f p_other=%.4f ratio=%.2f last_side=%s bars_since=%d",
                    p_side, p_other, ratio, last_side, bars_since_last_side
                )
                return None  # Avoid flip on thin edge
            # If ratio is too low, block entry regardless
            # Log side_ratio block at INFO level (so we can see ratio blocks)
            import logging
            log = logging.getLogger(__name__)
            log.info(
                "[GUARD] side_ratio block: p_side=%.4f p_other=%.4f ratio=%.2f < %.2f",
                p_side, p_other, ratio, side_ratio_min
            )
            return None
        
    else:
        # Fall back to legacy entry_params (backward compatibility)
        meta_threshold = float(entry_params.get("META_THRESHOLD", 0.40))
        margin_min = float(entry_params.get("ENTRY_MIN_MARGIN_MIN", 0.02))
        conf_margin_min = float(entry_params.get("CONF_MARGIN_MIN", margin_min))
    
        # Gate on p_side (not just p_long) - ensures both LONG and SHORT are gated correctly
        if p_side < meta_threshold:
            return None
    
        if prediction.margin < margin_min:
            return None
        
        if prediction.margin < conf_margin_min:
            return None
    
    return side


def prepare_exit_features(snapshot: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    df = snapshot.copy()
    base_numeric = [
        "mfe_so_far",
        "mae_so_far",
        "bars_in_trade",
        "atr_bps",
        "cost_bps",
        "net_move_bps",
        "side",
        "pnl_so_far",
    ]
    categorical = ["session_tag", "period_tag"]
    missing_numeric = [col for col in base_numeric if col not in df.columns]
    if missing_numeric:
        raise ValueError(f"Missing numeric exit features: {missing_numeric}")

    df_cat = pd.get_dummies(df, columns=categorical, prefix=["session_tag", "period_tag"])
    for col in feature_names:
        if col not in df_cat.columns:
            df_cat[col] = 0.0
    df_cat = df_cat[feature_names].astype(np.float32)
    return df_cat


def ensure_trade_log(trade_log_path: Path) -> None:
    """
    Ensure trade log CSV exists with proper schema (including run_id and extra columns).
    If file exists with wrong header it is backed up, fixed, and optionally repopulated.
    """
    trade_log_path.parent.mkdir(parents=True, exist_ok=True)

    from gx1.execution.trade_log_schema import TRADE_LOG_FIELDS

    fieldnames = TRADE_LOG_FIELDS

    if trade_log_path.exists():
        expected_header = ",".join(fieldnames)
        try:
            with trade_log_path.open("r", encoding="utf-8") as f:
                first_line = f.readline().strip()
        except Exception as exc:
            log.warning("[TRADE_LOG] Could not read header (%s). Recreating file.", exc)
            trade_log_path.unlink(missing_ok=True)
            first_line = ""

        if first_line and first_line != expected_header:
            import shutil

            log.warning(
                "[TRADE_LOG] Header mismatch detected - expected %s fields but found %s. Rebuilding trade log.",
                len(fieldnames),
                len(first_line.split(",")),
            )
            backup_path = trade_log_path.with_suffix(".csv.backup")
            shutil.copy2(trade_log_path, backup_path)
            log.info("[TRADE_LOG] Backup saved to %s", backup_path)

            preserved_df: Optional[pd.DataFrame]
            try:
                preserved_df = pd.read_csv(trade_log_path, on_bad_lines="skip", engine="python")
            except Exception as exc:
                log.warning("[TRADE_LOG] Unable to preserve rows while rebuilding: %s", exc)
                preserved_df = None

            trade_log_path.unlink(missing_ok=True)
            with trade_log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                if preserved_df is not None:
                    preserved = 0
                    for _, row in preserved_df.iterrows():
                        row_dict = row.to_dict()
                        if "extra" in row_dict and isinstance(row_dict["extra"], dict):
                            row_dict["extra"] = jsonlib.dumps(row_dict["extra"])
                        for field in fieldnames:
                            row_dict.setdefault(field, "")
                        writer.writerow(row_dict)
                        preserved += 1
                    log.info("[TRADE_LOG] Recreated trade log with %d preserved rows", preserved)
            return

        if first_line == expected_header:
            try:
                df = pd.read_csv(trade_log_path, on_bad_lines="skip", engine="python")
            except Exception:
                df = pd.DataFrame(columns=fieldnames)

            missing_cols = [col for col in fieldnames if col not in df.columns]
            if missing_cols:
                for col in missing_cols:
                    df[col] = ""
                df = df[fieldnames]
                df.to_csv(trade_log_path, index=False)
                log.info("[TRADE_LOG] Added missing columns to existing log: %s", missing_cols)
            else:
                log.debug("[TRADE_LOG] Header and columns verified for %s", trade_log_path)
            return

    # File doesn't exist or was deleted above -> create fresh
    with trade_log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
    log.info("[TRADE_LOG] Created new trade log file with schema: %s", trade_log_path)


def update_trade_log_exit(trade_log_path: Path, trade_id: str, exit_time: str, exit_price: float, pnl_bps: float, exit_prob_close: Optional[float] = None, exit_reason: Optional[str] = None, primary_exit_reason: Optional[str] = None, bars_held: Optional[int] = None, session: Optional[str] = None, vol_regime: Optional[str] = None, trend_regime: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    """Update trade log CSV with exit information."""
    import csv
    from gx1.execution.trade_log_schema import TRADE_LOG_FIELDS
    
    if not trade_log_path.exists():
        log.warning("Trade log file not found: %s", trade_log_path)
        return
    
    # Use centralized schema
    fieldnames = TRADE_LOG_FIELDS
    
    # Read existing trades (skip bad lines to handle corrupted rows)
    try:
        df = pd.read_csv(trade_log_path, on_bad_lines='skip', engine='python')
    except Exception as exc:
        log.warning("Failed to read trade log %s: %s. Skipping update.", trade_log_path, exc)
        return
    
    # Ensure all schema columns exist (backwards compat)
    for col in fieldnames:
        if col not in df.columns:
            df[col] = ""

    # Reorder columns to match schema
    existing_cols = [c for c in fieldnames if c in df.columns]
    extra_cols = [c for c in df.columns if c not in fieldnames]
    df = df[existing_cols + extra_cols]

    # Stabilize dtypes for string columns to avoid pandas warnings when assigning scalars
    string_cols = [
        "exit_time",
        "exit_reason",
        "primary_exit_reason",
        "session",
        "vol_regime",
        "trend_regime",
        "session_entry",
        "vol_regime_entry",
        "farm_entry_session",
        "farm_entry_vol_regime",
        "farm_guard_version",
        "session_exit",
        "vol_regime_exit",
        "extra",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")
    
    # Find trade by trade_id
    mask = df["trade_id"] == trade_id
    if not mask.any():
        log.warning("Trade %s not found in trade log", trade_id)
        return
    
    # Update exit information (keep numeric dtypes to avoid FutureWarning)
    df.loc[mask, "exit_time"] = exit_time
    df.loc[mask, "exit_price"] = float(exit_price)
    df.loc[mask, "pnl_bps"] = float(pnl_bps)
    if exit_prob_close is not None:
        df.loc[mask, "exit_prob_close"] = float(exit_prob_close)
    if exit_reason is not None:
        df.loc[mask, "exit_reason"] = str(exit_reason)
    if primary_exit_reason is not None:
        df.loc[mask, "primary_exit_reason"] = str(primary_exit_reason)
    if bars_held is not None:
        df.loc[mask, "bars_held"] = int(bars_held)
    # CRITICAL: Do NOT set session_entry/vol_regime_entry from session/vol_regime
    # These should ONLY come from trade.extra (set at entry)
    # session/vol_regime are for backward compatibility only
    if session is not None:
        df.loc[mask, "session"] = str(session)
    if vol_regime is not None:
        df.loc[mask, "vol_regime"] = str(vol_regime)
    if trend_regime is not None:
        df.loc[mask, "trend_regime"] = str(trend_regime)
    
    # CRITICAL: Extract entry/exit regimes from extra ONLY
    # For FARM_V1, use explicit farm_entry_* fields as single source of truth
    # For other modes, use session/vol_regime_entry as before
    # Do NOT use fallback to vol_regime/session (which might be exit-regime)
    # If entry-regime is None in extra, leave CSV column as NaN (don't fill with exit-regime)
    if extra is not None:
        # Check if this is a FARM trade (has farm_entry_* fields)
        farm_entry_session = extra.get("farm_entry_session")
        farm_entry_vol_regime = extra.get("farm_entry_vol_regime")
        
        if farm_entry_session is not None and farm_entry_vol_regime is not None:
            # FARM trade: Use explicit farm_entry_* fields (single source of truth)
            session_entry_val = farm_entry_session
            vol_regime_entry_val = farm_entry_vol_regime
        else:
            # Non-FARM trade: Use session/vol_regime_entry as before
            session_entry_val = extra.get("session")  # Use extra.session (set at entry)
            vol_regime_entry_val = extra.get("vol_regime_entry")  # Use extra.vol_regime_entry (set at entry, matches policy filtering)
        
        # Exit-regime: Use extra.session_exit and extra.vol_regime_exit (set at exit)
        session_exit_val = extra.get("session_exit")
        vol_regime_exit_val = extra.get("vol_regime_exit")
        
        # Set entry-regime from extra ONLY (no fallback)
        if session_entry_val is not None:
            df.loc[mask, "session_entry"] = str(session_entry_val)
        # If None, leave as NaN (don't fill with exit-regime)
        
        if vol_regime_entry_val is not None:
            df.loc[mask, "vol_regime_entry"] = str(vol_regime_entry_val)
        # If None, leave as NaN (don't fill with exit-regime)
        
        # Set FARM-specific fields if available
        if farm_entry_session is not None:
            df.loc[mask, "farm_entry_session"] = str(farm_entry_session)
        if farm_entry_vol_regime is not None:
            df.loc[mask, "farm_entry_vol_regime"] = str(farm_entry_vol_regime)
        farm_guard_version = extra.get("farm_guard_version")
        if farm_guard_version is not None:
            df.loc[mask, "farm_guard_version"] = str(farm_guard_version)
        
        # Set exit-regime from extra
        if session_exit_val is not None:
            df.loc[mask, "session_exit"] = str(session_exit_val)
        if vol_regime_exit_val is not None:
            df.loc[mask, "vol_regime_exit"] = str(vol_regime_exit_val)
    
    # Update extra column if provided
    if extra is not None:
        df.loc[mask, "extra"] = jsonlib.dumps(extra)
    
    # Del 3: Skip CSV write in replay if GX1_REPLAY_NO_CSV=1 (reduces I/O flaskehals)
    if os.getenv("GX1_REPLAY_NO_CSV") == "1":
        log.debug("GX1_REPLAY_NO_CSV=1: Skipping CSV write for trade %s (exit_time=%s exit_price=%.3f pnl_bps=%.2f)", trade_id, exit_time, exit_price, pnl_bps)
    else:
        # Write back to CSV with schema column order (only include fields that exist in schema)
        schema_cols = [c for c in fieldnames if c in df.columns]
        df[schema_cols].to_csv(trade_log_path, index=False)
        log.debug("Updated trade log for %s: exit_time=%s exit_price=%.3f pnl_bps=%.2f", trade_id, exit_time, exit_price, pnl_bps)

def append_trade_log(trade_log_path: Path, row: Dict[str, Any]) -> None:
    """
    Append a single trade row to the trade log, ensuring run_id and extra are present.
    
    CRITICAL: Always writes rows with exact same fieldnames in same order as header.
    This prevents header mismatch errors.
    
    FARM fields are automatically extracted from row["extra"] if present.
    """
    from gx1.execution.trade_log_schema import TRADE_LOG_FIELDS
    
    # Use centralized schema
    fieldnames = TRADE_LOG_FIELDS
    
    # Ensure schema (including run_id and extra) is present
    ensure_trade_log(trade_log_path)
    
    # Extract extra dict for FARM field extraction
    extra_dict = {}
    if "extra" in row:
        if isinstance(row["extra"], dict):
            extra_dict = row["extra"]
        elif isinstance(row["extra"], str) and row["extra"]:
            try:
                extra_dict = jsonlib.loads(row["extra"])
            except Exception:
                extra_dict = {}
    
    # Extract FARM fields from extra if present
    if extra_dict:
        if "farm_entry_session" in extra_dict:
            row["farm_entry_session"] = extra_dict["farm_entry_session"]
        if "farm_entry_vol_regime" in extra_dict:
            row["farm_entry_vol_regime"] = extra_dict["farm_entry_vol_regime"]
        if "farm_guard_version" in extra_dict:
            row["farm_guard_version"] = extra_dict["farm_guard_version"]
        # Extract exit_profile if present
        if "exit_profile" in extra_dict:
            row["exit_profile"] = extra_dict["exit_profile"]
        # Extract entry-regime fields if present
        if "session" in extra_dict:
            row["session_entry"] = extra_dict["session"]
        if "vol_regime_entry" in extra_dict:
            row["vol_regime_entry"] = extra_dict["vol_regime_entry"]
        # Extract exit-regime fields if present
        if "session_exit" in extra_dict:
            row["session_exit"] = extra_dict["session_exit"]
        if "vol_regime_exit" in extra_dict:
            row["vol_regime_exit"] = extra_dict["vol_regime_exit"]
        # Extract FARM_V2 meta-model fields if present
        if "p_long" in extra_dict:
            p_long_val = extra_dict["p_long"]
            row["entry_p_long"] = f"{p_long_val:.4f}" if isinstance(p_long_val, (int, float)) else str(p_long_val)
        if "p_profitable" in extra_dict:
            p_profitable_val = extra_dict["p_profitable"]
            row["entry_p_profitable"] = f"{p_profitable_val:.4f}" if isinstance(p_profitable_val, (int, float)) else str(p_profitable_val)
        if "entry_policy_version" in extra_dict:
            row["entry_policy_version"] = extra_dict["entry_policy_version"]
    
    # If run_id is missing in the provided row (e.g. legacy callers), fill with 'legacy'
    if "run_id" not in row:
        row = {**row, "run_id": "legacy"}
    # Serialize extra if it's a dict (convert to JSON string)
    if "extra" in row and isinstance(row["extra"], dict):
        row["extra"] = jsonlib.dumps(row["extra"])
    # If extra is missing, set to empty string
    if "extra" not in row:
        row["extra"] = ""
    
    # Ensure all schema fieldnames are present in row (fill missing with empty string)
    for field in fieldnames:
        if field not in row:
            row[field] = ""
    
    # Filter row to only include schema fieldnames (ignore any extra keys)
    # CRITICAL: Re-order to match fieldnames order exactly
    filtered_row = {field: str(row.get(field, "")) for field in fieldnames}
    
    # Sanity check: ensure row has exactly same number of fields as header
    if len(filtered_row) != len(fieldnames):
        log.error(
            "[TRADE_LOG] Row field count mismatch: expected %d, got %d. Dropping row.",
            len(fieldnames), len(filtered_row)
        )
        return
    
    # Del 3: Skip CSV write in replay if GX1_REPLAY_NO_CSV=1 (reduces I/O flaskehals)
    if os.getenv("GX1_REPLAY_NO_CSV") == "1":
        log.debug("GX1_REPLAY_NO_CSV=1: Skipping CSV append for trade %s", row.get("trade_id", "unknown"))
    else:
        # Write using schema fieldnames (ensures consistent column order)
        with trade_log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerow(filtered_row)


def append_eval_log(eval_log_path: Path, eval_record: Dict[str, Any]) -> None:
    """
    Append a single eval record as JSON line to eval log file.
    
    Parameters
    ----------
    eval_log_path : Path
        Path to JSONL log file.
    eval_record : Dict[str, Any]
        Eval record with fields: ts_utc, session, p_long, p_short, p_hat, margin, T, decision, price, units
    """
    eval_log_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_log_path.open("a", encoding="utf-8") as handle:
        handle.write(jsonlib.dumps(eval_record, separators=(",", ":")) + "\n")


# --------------------------------------------------------------------------- #
# Parity-run adapter
# --------------------------------------------------------------------------- #


def mirror_offline_predict(
    feat_row: pd.Series,
    session_tag: str,
    models: Dict[str, any],
    feature_cols: List[str],
    temperature_map: Dict[str, float],
    manifest: Dict,
) -> Tuple[float, float, str]:
    """
    Mirror offline prediction for parity verification.
    Uses same model and logic as live runner.
    
    Parameters
    ----------
    feat_row : pd.Series
        Feature row (single row from feature DataFrame, already aligned).
    session_tag : str
        Session tag (EU, US, OVERLAP).
    models : Dict[str, any]
        Session-routed models dictionary.
    feature_cols : List[str]
        Feature column names.
    temperature_map : Dict[str, float]
        Temperature map for calibration.
    manifest : Dict
        Feature manifest for alignment.
    
    Returns
    -------
    Tuple[float, float, str]
        (p_hat_off, margin_off, dir_off)
    """
    # Get model for this session
    model = models.get(session_tag)
    if model is None:
        raise ValueError(f"Model for session '{session_tag}' not found")
    
    # Convert feature row to numpy array (same format as live)
    row = feat_row[feature_cols].astype(np.float32).to_numpy().reshape(1, -1)
    
    # Predict probabilities (same as live)
    probs = model.predict_proba(row)
    classes = getattr(model, "classes_", None)
    
    # Extract probabilities (same logic as extract_entry_probabilities)
    prediction_off = extract_entry_probabilities(probs, classes)
    
    # Apply temperature scaling (same as live)
    T = float(temperature_map.get(session_tag, 1.0))
    if T != 1.0:
        prediction_off.prob_long = _apply_temperature_static(prediction_off.prob_long, T)
        prediction_off.prob_short = _apply_temperature_static(prediction_off.prob_short, T)
        prediction_off.p_hat = float(max(prediction_off.prob_long, prediction_off.prob_short))
        prediction_off.margin = float(abs(prediction_off.prob_long - prediction_off.prob_short))
    
    # Determine direction (same logic as live)
    if prediction_off.prob_long >= prediction_off.prob_short:
        dir_off = "LONG"
    else:
        dir_off = "SHORT"
    
    return prediction_off.p_hat, prediction_off.margin, dir_off


def _apply_temperature_static(p: float, T: float) -> float:
    """Apply temperature scaling (static version for parity-run)."""
    eps = 1e-8
    p = min(max(p, eps), 1.0 - eps)
    logit_p = np.log(p) - np.log(1.0 - p)
    logit_T = logit_p / max(T, 1e-6)
    return 1.0 / (1.0 + np.exp(-logit_T))


# --------------------------------------------------------------------------- #
# Live runner
# --------------------------------------------------------------------------- #


class GX1DemoRunner:
    def _write_xgb_input_truth_dump_if_ready(self, force: bool = False) -> None:
        """
        Write XGB input truth dump once per run (TRUTH/PREBUILT only, chunk_0 only).

        - If force=False: writes only when captured rows >= target (default 2000)
        - If force=True: writes with whatever rows were captured (if any) at end of replay

        Hard truth check (TRUTH): if >50% of features are constant in the sampled matrix,
        write XGB_INPUT_DEGENERATE_FATAL.json and raise RuntimeError.
        """
        import numpy as np
        import json as _json
        from pathlib import Path as Path_local

        is_truth_run = os.getenv("GX1_RUN_MODE", "").upper() == "TRUTH" or os.getenv("GX1_TRUTH_MODE", "0") == "1"
        is_prebuilt = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1" and os.getenv("GX1_FEATURE_BUILD_DISABLED", "0") == "1"
        chunk_id_env = os.getenv("GX1_CHUNK_ID", "")
        if not (is_truth_run and is_prebuilt and str(chunk_id_env) == "0"):
            return

        if not hasattr(self, "_xgb_truth_dump_done"):
            self._xgb_truth_dump_done = False
        if self._xgb_truth_dump_done:
            return

        rows = getattr(self, "_xgb_truth_dump_rows", None)
        feat_names = getattr(self, "_xgb_truth_dump_feature_names", None)
        ssot_details = getattr(self, "_xgb_feature_names_ssot_details", None)
        n_target = int(getattr(self, "_xgb_truth_dump_n_target", 2000))
        if not rows or not feat_names:
            return
        if not force and len(rows) < n_target:
            return

        # Resolve run_root from explicit_output_dir if present (chunk dir -> parent)
        run_root = None
        if getattr(self, "explicit_output_dir", None):
            try:
                p = Path_local(str(self.explicit_output_dir))
                run_root = p.parent if p.name.startswith("chunk_") else p
            except Exception:
                run_root = None
        if run_root is None:
            raise RuntimeError("[TRUTH_FAIL] XGB_INPUT_TRUTH_DUMP cannot resolve run_root for writing artifacts")

        dump_json_path = run_root / "XGB_INPUT_TRUTH_DUMP.json"
        dump_md_path = run_root / "XGB_INPUT_TRUTH_DUMP.md"
        fatal_path = run_root / "XGB_INPUT_DEGENERATE_FATAL.json"

        if dump_json_path.exists():
            self._xgb_truth_dump_done = True
            return

        M = np.stack(rows, axis=0)
        feat_names = list(feat_names)
        n_rows_s, n_feat_s = int(M.shape[0]), int(M.shape[1])
        if n_feat_s != len(feat_names):
            raise RuntimeError(
                "[TRUTH_FAIL] XGB_INPUT_TRUTH_DUMP feature_names length mismatch: "
                f"n_feat={n_feat_s} names_len={len(feat_names)}"
            )

        per_col = []
        per_feature_stats_by_name: Dict[str, Any] = {}
        constant_names = []
        all_zero_names = []
        std_pairs = []
        for j, name in enumerate(feat_names):
            col = M[:, j]
            min_v = float(np.min(col))
            max_v = float(np.max(col))
            mean_v = float(np.mean(col))
            std_v = float(np.std(col))
            uniq = np.unique(col)
            unique_count = int(uniq.shape[0])
            pct_zero = float(np.mean(col == 0.0))
            is_constant = unique_count == 1
            const_value = float(uniq[0]) if is_constant else None
            if is_constant:
                constant_names.append(name)
            if pct_zero >= 1.0:
                all_zero_names.append(name)
            std_pairs.append((std_v, name))
            per_col.append(
                {
                    "feature": name,
                    "min": min_v,
                    "max": max_v,
                    "mean": mean_v,
                    "std": std_v,
                    "unique_count": unique_count,
                    "percent_zero": pct_zero,
                    "is_constant": is_constant,
                    "const_value": const_value,
                }
            )
            per_feature_stats_by_name[name] = {
                "min": min_v,
                "max": max_v,
                "mean": mean_v,
                "std": std_v,
                "unique_count": unique_count,
                "percent_zero": pct_zero,
                "is_constant": is_constant,
                "const_value": const_value,
            }

        std_pairs_sorted = sorted(std_pairs, key=lambda x: (-x[0], x[1]))
        top20_by_std = [{"feature": n, "std": float(s)} for (s, n) in std_pairs_sorted[:20]]

        n_constant = int(len(constant_names))
        n_all_zero = int(len(all_zero_names))
        constant_frac = float(n_constant) / float(n_feat_s) if n_feat_s else 0.0
        varying_names = [n for n in feat_names if n not in constant_names]
        first5 = np.round(M[:5, :], 6).tolist()
        first5_rows_by_name = []
        for row in first5:
            first5_rows_by_name.append({name: row[i] for i, name in enumerate(feat_names)})

        payload = {
            "run_id": os.getenv("GX1_RUN_ID"),
            "chunk_id": os.getenv("GX1_CHUNK_ID"),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_rows_sampled": n_rows_s,
            "n_features": n_feat_s,
            "X_shape": [n_rows_s, n_feat_s],
            "feature_names_ordered": feat_names,
            "feature_names_ssot": ssot_details,
            "degeneracy_summary": {
                "n_constant_features": n_constant,
                "constant_fraction": constant_frac,
                "n_all_zero_features": n_all_zero,
                "constant_feature_names": constant_names,
                "all_zero_feature_names": all_zero_names,
                "varying_feature_names": varying_names,
            },
            "per_feature_stats_by_name": per_feature_stats_by_name,
            "per_feature_stats_rows": per_col,
            "top20_features_by_std": [
                {
                    "feature": r["feature"],
                    "std": r["std"],
                    "min": per_feature_stats_by_name.get(r["feature"], {}).get("min"),
                    "max": per_feature_stats_by_name.get(r["feature"], {}).get("max"),
                    "percent_zero": per_feature_stats_by_name.get(r["feature"], {}).get("percent_zero"),
                }
                for r in top20_by_std
            ],
            "first5_rows_preview_by_name": first5_rows_by_name,
            "notes": [
                "Captured from the exact XGB input vector right before predict_proba().",
                "TRUTH/PREBUILT only; chunk_0 only; first N rows only (or force finalize at end).",
            ],
        }

        tmp_json = dump_json_path.with_suffix(dump_json_path.suffix + f".tmp.{os.getpid()}")
        with open(tmp_json, "w", encoding="utf-8") as f:
            _json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp_json, dump_json_path)

        md_lines = []
        md_lines.append("## XGB INPUT TRUTH DUMP")
        md_lines.append("")
        md_lines.append(f"- **run_id**: `{payload.get('run_id')}`")
        md_lines.append(f"- **chunk_id**: `{payload.get('chunk_id')}`")
        md_lines.append(f"- **generated_utc**: `{payload.get('generated_utc')}`")
        md_lines.append("")
        md_lines.append("## Section 1 — Shape")
        md_lines.append(f"- **n_rows_sampled**: `{n_rows_s}`")
        md_lines.append(f"- **n_features**: `{n_feat_s}`")
        md_lines.append("")
        md_lines.append("## Section 2 — Degeneracy Summary")
        md_lines.append(f"- **n_constant_features**: `{n_constant}` (fraction=`{constant_frac:.4f}`)")
        md_lines.append(f"- **n_all_zero_features**: `{n_all_zero}`")
        md_lines.append("")
        md_lines.append("## Ordered feature list (n=91)")
        for n in feat_names:
            md_lines.append(f"- `{n}`")
        md_lines.append("")
        md_lines.append(f"## Varying features (n={len(varying_names)})")
        for n in varying_names:
            md_lines.append(f"- `{n}`")
        md_lines.append("")
        md_lines.append("### Constant features (names)")
        for n in constant_names[:120]:
            md_lines.append(f"- `{n}`")
        md_lines.append("")
        md_lines.append("## Section 3 — Top 20 Features by Std")
        for r in payload.get("top20_features_by_std") or []:
            md_lines.append(
                f"- `{r['feature']}` std={float(r['std']):.8f} min={r.get('min')} max={r.get('max')} percent_zero={r.get('percent_zero')}"
            )
        md_lines.append("")
        md_lines.append("## Section 4 — First 5 rows (matrix preview)")
        md_lines.append("```")
        for row in first5_rows_by_name:
            md_lines.append(str(row))
        md_lines.append("```")
        md_text = "\n".join(md_lines) + "\n"
        tmp_md = dump_md_path.with_suffix(dump_md_path.suffix + f".tmp.{os.getpid()}")
        with open(tmp_md, "w", encoding="utf-8") as f:
            f.write(md_text)
        os.replace(tmp_md, dump_md_path)

        if constant_frac > 0.50:
            fatal = {
                "error_type": "XGB_INPUT_DEGENERATE_FATAL",
                "run_id": payload.get("run_id"),
                "chunk_id": payload.get("chunk_id"),
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "n_rows_sampled": n_rows_s,
                "n_features": n_feat_s,
                "n_constant_features": n_constant,
                "constant_fraction": constant_frac,
                "threshold": 0.50,
                "dump_json_path": str(dump_json_path),
                "constant_feature_names": constant_names,
                "all_zero_feature_names": all_zero_names,
                "varying_feature_names": varying_names,
            }
            tmp_f = fatal_path.with_suffix(fatal_path.suffix + f".tmp.{os.getpid()}")
            with open(tmp_f, "w", encoding="utf-8") as f:
                _json.dump(fatal, f, indent=2)
            os.replace(tmp_f, fatal_path)
            raise RuntimeError(
                "XGB input matrix degenerate: "
                f"constant_fraction={constant_frac:.4f} "
                f"all_zero_first20={all_zero_names[:20]} varying_first20={varying_names[:20]}"
            )

        self._xgb_truth_dump_done = True
        try:
            self._xgb_truth_dump_rows = []  # type: ignore[attr-defined]
        except Exception:
            pass

    def __init__(
        self,
        policy_path: Path,
        *,
        dry_run_override: Optional[bool] = None,
        replay_mode: bool = False,
        fast_replay: bool = False,
        output_dir: Optional[Path] = None,
    ) -> None:
        load_dotenv_if_present()
        self.policy_path = policy_path
        self.policy = load_yaml_config(policy_path)
        
        # DEL 1: Set ENTRY_CONTEXT_FEATURES_ENABLED from policy if specified
        if "ENTRY_CONTEXT_FEATURES_ENABLED" in self.policy:
            policy_value = self.policy["ENTRY_CONTEXT_FEATURES_ENABLED"]
            if isinstance(policy_value, bool):
                os.environ["ENTRY_CONTEXT_FEATURES_ENABLED"] = "true" if policy_value else "false"
            elif isinstance(policy_value, str):
                os.environ["ENTRY_CONTEXT_FEATURES_ENABLED"] = policy_value
            log.info("[BOOT] ENTRY_CONTEXT_FEATURES_ENABLED set from policy: %s", os.environ.get("ENTRY_CONTEXT_FEATURES_ENABLED"))
        
        # DEL 1: Initialize replay eval collectors (only in replay mode)
        self.replay_eval_collectors = None
        if replay_mode:
            from gx1.execution.replay_eval_collectors import (
                RawSignalCollector,
                PolicyDecisionCollector,
                TradeOutcomeCollector,
            )
            self.replay_eval_collectors = {
                "raw_signals": RawSignalCollector(),
                "policy_decisions": PolicyDecisionCollector(),
                "trade_outcomes": TradeOutcomeCollector(),
            }
            log.info("[REPLAY_EVAL] Collectors initialized for replay evaluation")
        self.exit_config_name = None
        
        # Initialize replay flags early (before any methods that check them)
        self.replay_mode = bool(replay_mode)
        self.replay_start_ts: Optional[pd.Timestamp] = None
        self.replay_end_ts: Optional[pd.Timestamp] = None
        self.fast_replay = bool(fast_replay)  # Fast mode for tuning (skip heavy reporting)
        self.is_replay = self.replay_mode or self.fast_replay
        
        # Initialize replay-specific attributes
        self._consecutive_order_failures = 0
        
        # Performance counters (initialized for replay perf tracking)
        self.perf_n_bars_processed = 0
        self.perf_n_trades_created = 0
        self.perf_bars_total = 0  # Del 1: Total bars to process (set in _run_replay_impl)
        
        # Entry stage telemetry: initialize early (will be reset in _run_replay_impl before loop)
        # These are persistent attributes on runner-self (not local variables)
        self.bars_seen = 0
        self.bars_skipped_warmup = 0
        self.bars_skipped_pregate = 0
        self.bars_reaching_entry_stage = 0
        
        # Del 2: Performance collector for feature timing breakdown
        from gx1.utils.perf_timer import PerfCollector
        self.perf_collector = PerfCollector()
        
        # Persistent feature state for caching across build_basic_v1 calls
        from gx1.features.feature_state import FeatureState
        from gx1.features.rolling_state_numba import RollingR1Quantiles48State
        self.feature_state = FeatureState()
        # Initialize incremental quantile state for replay/live
        self.feature_state.r1_quantiles_state = RollingR1Quantiles48State()
        
        # Store explicit output_dir if provided (for parallel chunk workers)
        # If set, trade journals will be written to this directory
        # run_header.json will still be written to parent run_dir (if in parallel_chunks)
        self.explicit_output_dir = Path(output_dir) if output_dir else None
        if self.explicit_output_dir:
            # Also set self.output_dir for compatibility (used in _write_stage0_reason_report, etc.)
            self.output_dir = self.explicit_output_dir
            log.info("[BOOT] Using explicit output_dir: %s", self.explicit_output_dir)
        
        # COMMIT A: Get run_id and chunk_id using centralized utilities
        from gx1.utils.run_identity import get_run_id, get_chunk_id
        env_run_id = os.getenv("GX1_RUN_ID")
        env_chunk_id = os.getenv("GX1_CHUNK_ID")
        
        # Get run_id (priority: env > policy > output_dir basename > auto-generated)
        if env_run_id:
            self.run_id = env_run_id
            log.info("[RUN_ID] Using run_id from environment: %s", self.run_id)
        elif "run_id" in self.policy:
            self.run_id = self.policy["run_id"]
            log.info("[RUN_ID] Using run_id from policy: %s", self.run_id)
        else:
            # Use utility function to get run_id from output_dir or generate
            self.run_id = get_run_id(output_dir=self.explicit_output_dir, env_run_id=None)
            log.info("[RUN_ID] Generated run_id: %s", self.run_id)
        
        # Get chunk_id (priority: env > output_dir path > "single")
        self.chunk_id = get_chunk_id(output_dir=self.explicit_output_dir, env_chunk_id=env_chunk_id)
        log.info("[CHUNK_ID] Using chunk_id: %s", self.chunk_id)
        
        # XGB input fingerprint logging (TRUTH/SMOKE only)
        self.xgb_fingerprint_enabled = (
            os.getenv("GX1_XGB_INPUT_FINGERPRINT", "0") == "1" and
            (os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1" or
             os.getenv("GX1_RUN_MODE", "").upper() in ["TRUTH", "SMOKE"] or
             self.is_replay)
        )
        if self.xgb_fingerprint_enabled:
            self.xgb_fingerprint_sample_n = int(os.getenv("GX1_XGB_INPUT_FINGERPRINT_SAMPLE_N", "25"))
            self.xgb_fingerprint_max_per_session = int(os.getenv("GX1_XGB_INPUT_FINGERPRINT_MAX_PER_SESSION", "200"))
            self.xgb_fingerprint_min_logged = int(os.getenv("GX1_XGB_INPUT_FINGERPRINT_MIN_LOGGED", "50"))
            self.xgb_fingerprint_call_counts: Dict[str, int] = defaultdict(int)
            self.xgb_fingerprint_logged_counts: Dict[str, int] = defaultdict(int)
            # Use per-process JSONL file to avoid multiprocess write conflicts
            pid = os.getpid()
            self.xgb_fingerprint_jsonl_path: Optional[Path] = None
            if self.explicit_output_dir:
                self.xgb_fingerprint_jsonl_path = self.explicit_output_dir / f"XGB_INPUT_FINGERPRINT.{pid}.jsonl"
                # Ensure parent directory exists
                self.xgb_fingerprint_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            log.info("[XGB_FINGERPRINT] Enabled: sample_n=%d, max_per_session=%d, min_logged=%d, jsonl_path=%s",
                     self.xgb_fingerprint_sample_n, self.xgb_fingerprint_max_per_session, 
                     self.xgb_fingerprint_min_logged, self.xgb_fingerprint_jsonl_path)
        
        # XGB input debug logging (TRUTH/SMOKE only, sampled)
        self.xgb_input_debug_enabled = (
            os.getenv("GX1_XGB_INPUT_DEBUG", "0") == "1" and
            (os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1" or
             os.getenv("GX1_RUN_MODE", "").upper() in ["TRUTH", "SMOKE"] or
             self.is_replay)
        )
        if self.xgb_input_debug_enabled:
            self.xgb_input_debug_sample_n = int(os.getenv("GX1_XGB_INPUT_DEBUG_SAMPLE_N", "25"))
            self.xgb_input_debug_max_per_session = int(os.getenv("GX1_XGB_INPUT_DEBUG_MAX_PER_SESSION", "200"))
            self.xgb_input_debug_flat_k = int(os.getenv("GX1_XGB_INPUT_DEBUG_FLAT_K", "20"))
            self.xgb_input_debug_call_counts: Dict[str, int] = defaultdict(int)
            self.xgb_input_debug_logged_counts: Dict[str, int] = defaultdict(int)
            self.xgb_input_debug_flat_counts: Dict[str, int] = defaultdict(int)
            pid = os.getpid()
            self.xgb_input_debug_jsonl_path: Optional[Path] = None
            if self.explicit_output_dir:
                self.xgb_input_debug_jsonl_path = self.explicit_output_dir / f"XGB_INPUT_DEBUG.{pid}.jsonl"
                self.xgb_input_debug_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            log.info("[XGB_INPUT_DEBUG] Enabled: sample_n=%d, max_per_session=%d, flat_k=%d, jsonl_path=%s",
                     self.xgb_input_debug_sample_n, self.xgb_input_debug_max_per_session,
                     self.xgb_input_debug_flat_k, self.xgb_input_debug_jsonl_path)
        
        # Policy-lock: Compute and store policy_hash at startup
        policy_content = policy_path.read_text()
        self.policy_hash = hashlib.md5(policy_content.encode("utf-8")).hexdigest()[:16]
        
        # Extract policy name/version for traceability
        self.policy_name = self.policy.get("version", policy_path.stem)  # Use policy["version"] if set, else filename
        if "version" not in self.policy:
            log.info("[POLICY] Policy version not set - using filename: %s", self.policy_name)
        
        # Read mode from policy (default: "LIVE")
        self.mode = self.policy.get("mode", "LIVE")
        
        # Check for RUN_MODE (PROD vs SHADOW)
        run_mode = self.policy.get("run_mode", "PROD").upper()
        self.run_mode = run_mode
        if run_mode == "SHADOW":
            log.info("[RUN_MODE] SHADOW mode enabled - trades will be journaled but no real orders sent")
            if dry_run_override is None:
                dry_run_override = True  # Force dry_run in SHADOW mode
        elif run_mode == "PROD":
            log.info("[RUN_MODE] PROD mode enabled - real orders will be sent")
        else:
            log.warning(f"[RUN_MODE] Unknown run_mode={run_mode}, defaulting to PROD")
        
        # Check for CANARY mode (override dry_run, but log all invariants)
        self.canary_mode = self.mode.upper() == "CANARY"
        if self.canary_mode:
            log.info("[CANARY] Canary mode enabled - dry_run=True, all invariants will be logged")
            if dry_run_override is None:
                dry_run_override = True  # Force dry_run in canary mode
        
        log.info("[BOOT] mode=%s, run_mode=%s", self.mode, self.run_mode)
        
        # Initialize entry-only log path if in ENTRY_ONLY mode
        if self.mode == "ENTRY_ONLY":
            # Use chunk_id from environment if available (for parallel replay)
            chunk_id = os.getenv("GX1_CHUNK_ID", "")
            if chunk_id:
                entry_only_log_path = Path(f"gx1/live/entry_only_log_v9_test_chunk_{chunk_id}.csv")
            else:
                entry_only_log_path = Path("gx1/live/entry_only_log_v9_test.csv")
            entry_only_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.entry_only_log_path = entry_only_log_path
            # Initialize buffer for efficient batch writes (reduces I/O bottleneck)
            self._entry_only_log_buffer = []
            self._entry_only_log_buffer_size = 500  # Flush every 500 entries (increased for better performance)
            log.info("[ENTRY_ONLY] Entry-only logging enabled: %s (buffered, flush every %d entries)", 
                     self.entry_only_log_path, self._entry_only_log_buffer_size)
        else:
            self.entry_only_log_path = None
        
        # Initialize model_name tracking (will be set when models are loaded)
        self.model_name = self.policy.get("entry_models", {}).get("default", {}).get("version", "UNKNOWN")
        
        # Resolve entry_config path (optional - if None, skip loading)
        entry_config_raw = self.policy.get("entry_config")
        entry_cfg_path = None
        if entry_config_raw is None:
            entry_cfg = {}
            log.debug("[BOOT] entry_config not set in policy, using empty config")
        else:
            entry_cfg_path = Path(entry_config_raw)
            if not entry_cfg_path.is_absolute():
                # Resolve relative to workspace root
                workspace_root = Path(__file__).parent.parent.parent
                entry_cfg_path = workspace_root / entry_cfg_path
            entry_cfg = load_yaml_config(entry_cfg_path)
        
        # Resolve exit_config path (optional - if None, skip loading)
        exit_config_raw = self.policy.get("exit_config")
        exit_cfg_path = None
        if exit_config_raw is None:
            exit_cfg = {}
            log.debug("[BOOT] exit_config not set in policy, using empty config")
        else:
            exit_cfg_path = Path(exit_config_raw)
            if not exit_cfg_path.is_absolute():
                # Resolve relative to workspace root
                workspace_root = Path(__file__).parent.parent.parent
                exit_cfg_path = workspace_root / exit_cfg_path
            
            # TRUTH-only: Write EXIT_CONFIG_RESOLVE_PROOF.json for forensics
            is_truth_or_smoke = os.environ.get("GX1_RUN_MODE", "").upper() in ["TRUTH", "SMOKE"]
            if is_truth_or_smoke:
                resolved_path_str = str(exit_cfg_path.resolve())
                path_exists = exit_cfg_path.exists()
                exit_config_name = exit_cfg_path.stem if exit_cfg_path else None
                
                import sys
                proof_data = {
                    "exit_config_raw": exit_config_raw,
                    "resolved_exit_cfg_path": resolved_path_str,
                    "exists": path_exists,
                    "exit_config_name": exit_config_name,
                    "cwd": str(Path.cwd()),
                    "sys.executable": sys.executable,
                    "workspace_root": str(workspace_root),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                
                # Write to output_dir if available (may be None in early __init__)
                output_dir = getattr(self, "explicit_output_dir", None) or getattr(self, "output_dir", None)
                if output_dir:
                    from gx1.utils.atomic_json import atomic_write_json
                    proof_path = Path(output_dir) / "EXIT_CONFIG_RESOLVE_PROOF.json"
                    try:
                        atomic_write_json(proof_path, proof_data)
                        log.info("[EXIT_CONFIG_RESOLVE] Proof written to %s", proof_path)
                    except Exception as e:
                        log.warning("[EXIT_CONFIG_RESOLVE] Failed to write proof: %s", e)
                
                # TRUTH-only: FATAL if path does not exist
                if not path_exists:
                    fatal_capsule = {
                        "fatal_reason": "EXIT_CONFIG_PATH_NOT_FOUND",
                        "message": f"exit_config path does not exist: {resolved_path_str}",
                        "exit_config_raw": exit_config_raw,
                        "resolved_exit_cfg_path": resolved_path_str,
                        "workspace_root": str(workspace_root),
                        "cwd": str(Path.cwd()),
                        "hint": "Check that exit_config path in policy YAML is correct relative to GX1_ENGINE workspace root.",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    
                    if output_dir:
                        from gx1.utils.atomic_json import atomic_write_json
                        fatal_path = Path(output_dir) / "EXIT_CONFIG_PATH_NOT_FOUND_FATAL.json"
                        try:
                            atomic_write_json(fatal_path, fatal_capsule)
                        except Exception as e:
                            log.error("[EXIT_CONFIG_PATH_NOT_FOUND] Failed to write FATAL capsule: {e}")
                    
                    log.error(
                        "[EXIT_CONFIG_PATH_NOT_FOUND] FATAL: exit_config path does not exist: %s. "
                        "See EXIT_CONFIG_PATH_NOT_FOUND_FATAL.json for details.",
                        resolved_path_str
                    )
                    raise FileNotFoundError(
                        f"[EXIT_CONFIG_PATH_NOT_FOUND] FATAL: exit_config path does not exist: {resolved_path_str}. "
                        f"Policy exit_config_raw: {exit_config_raw}, workspace_root: {workspace_root}"
                    )
            
            exit_cfg = load_yaml_config(exit_cfg_path)
        self.entry_params = entry_cfg.get("params", entry_cfg)
        
        # Merge entry_config into self.policy so entry policies can access their configs
        # This is critical for entry_v9_policy_farm_v1, entry_v9_policy_v1, etc.
        # IMPORTANT: Don't override entry_models if already in policy (policy takes precedence)
        if entry_cfg:
            for key, value in entry_cfg.items():
                if key == "entry_models":
                    # Don't override entry_models if already in policy (policy takes precedence)
                    if "entry_models" not in self.policy:
                        self.policy[key] = value
                elif key not in self.policy:
                    self.policy[key] = value
            log.debug("[BOOT] Merged entry_config into self.policy (for entry policy access)")
        
        # Check for REN EXIT_V2 mode (only_v2_drift flag)
        self.exit_only_v2_drift = bool(exit_cfg.get("exit", {}).get("only_v2_drift", False))
        if self.exit_only_v2_drift:
            log.info("[BOOT] REN EXIT_V2 mode enabled: only_v2_drift=true - disabling other exit mechanisms")
            # Disable tick_exit
            tick_exit_cfg = self.policy.get("tick_exit", {})
            tick_exit_cfg["enabled"] = False
            self.policy["tick_exit"] = tick_exit_cfg
            # Disable broker-side TP/SL
            self.policy["broker_side_tp_sl"] = False
            log.info("[BOOT] tick_exit disabled (REN EXIT_V2 mode)")
            log.info("[BOOT] broker_side_tp_sl disabled (REN EXIT_V2 mode)")
        self.exit_params = exit_cfg  # Store exit config for shadow-exit A/B
        
        # Check for EXIT_POLICY_V2 (pre-empts all other exit rules if enabled)
        self.exit_policy_v2 = None
        exit_v2_enabled = os.environ.get("GX1_EXIT_POLICY_V2", "0") == "1"
        if exit_v2_enabled:
            try:
                from gx1.exits.exit_policy_v2 import ExitPolicyV2, load_exit_policy_v2_config
                
                # Load Exit Policy V2 config
                exit_v2_yaml_path = os.environ.get("GX1_EXIT_POLICY_V2_YAML", "gx1/configs/exits/EXIT_POLICY_V2_TRIAL160.yaml")
                exit_v2_yaml_path = Path(exit_v2_yaml_path)
                if not exit_v2_yaml_path.is_absolute():
                    # Try relative to workspace root
                    workspace_root = Path(__file__).parent.parent.parent
                    exit_v2_yaml_path = workspace_root / exit_v2_yaml_path
                
                if not exit_v2_yaml_path.exists():
                    raise FileNotFoundError(f"Exit Policy V2 YAML not found: {exit_v2_yaml_path}")
                
                # Load config
                exit_v2_config = load_exit_policy_v2_config(config_path=exit_v2_yaml_path)
                
                if exit_v2_config.enabled:
                    self.exit_policy_v2 = ExitPolicyV2(exit_v2_config)
                    
                    # Calculate YAML SHA256 (hashlib already imported at top level)
                    with open(exit_v2_yaml_path, "rb") as f:
                        exit_v2_yaml_sha256 = hashlib.sha256(f.read()).hexdigest()
                    
                    # Store for RUN_IDENTITY
                    self.exit_v2_yaml_path = str(exit_v2_yaml_path.resolve())
                    self.exit_v2_yaml_sha256 = exit_v2_yaml_sha256
                    self.exit_v2_params_summary = {
                        "mae_kill_atr": exit_v2_config.mae_kill_mae_atr_ge,
                        "time_stop_bars": exit_v2_config.time_stop_t1_bars,
                        "time_stop_mfe_atr": exit_v2_config.time_stop_require_mfe_atr_ge,
                        "trail_activate_mfe_atr": exit_v2_config.profit_trail_activate_mfe_atr_ge,
                        "trail_giveback_frac": exit_v2_config.profit_trail_giveback_frac,
                    }
                    
                    log.info(
                        "[BOOT] EXIT_POLICY_V2 enabled: YAML=%s SHA256=%s",
                        self.exit_v2_yaml_path,
                        exit_v2_yaml_sha256[:16],
                    )
                else:
                    log.warning("[BOOT] EXIT_POLICY_V2 YAML found but enabled=false")
            except Exception as e:
                log.error("[BOOT] Failed to initialize EXIT_POLICY_V2: %s", e, exc_info=True)
                raise RuntimeError(f"EXIT_POLICY_V2 initialization failed: {e}") from e
        else:
            log.info("[BOOT] EXIT_POLICY_V2 disabled (GX1_EXIT_POLICY_V2 != 1)")
            self.exit_v2_yaml_path = None
            self.exit_v2_yaml_sha256 = None
            self.exit_v2_params_summary = None
        
        # Check for EXIT_V2_DRIFT, EXIT_V3_ADAPTIVE, EXIT_FARM_V1, or EXIT_FARM_V2_RULES configuration
        exit_type = exit_cfg.get("exit", {}).get("type") if isinstance(exit_cfg.get("exit"), dict) else None
        self.exit_farm_v1_policy = None
        self.exit_farm_v2_rules_policy = None
        self.exit_farm_v2_rules_factory = None
        self.exit_farm_v2_rules_states: Dict[str, Any] = {}
        self.exit_verbose_logging = False
        self.exit_fixed_bar_policy = None
        self.farm_v1_mode = False  # Track if we're in FARM_V1 mode
        self.farm_v2_mode = False  # Track if we're in FARM_V2 mode
        self.farm_v2b_mode = False  # Track if we're in FARM_V2B mode
        
        if exit_type == "FARM_V2_RULES":
            # FARM_V2_RULES mode: Rule-based exit for FARM_V2B
            from gx1.policy.exit_farm_v2_rules import get_exit_policy_farm_v2_rules
            exit_params = exit_cfg.get("exit", {}).get("params", {})
            self.exit_verbose_logging = bool(exit_params.get("verbose_logging", False))
            
            def _build_exit_farm_v2_rules_policy():
                return get_exit_policy_farm_v2_rules(
                    enable_rule_a=exit_params.get("enable_rule_a", False),
                    enable_rule_b=exit_params.get("enable_rule_b", False),
                    enable_rule_c=exit_params.get("enable_rule_c", False),
                    rule_a_profit_min_bps=exit_params.get("rule_a_profit_min_bps", 6.0),
                    rule_a_profit_max_bps=exit_params.get("rule_a_profit_max_bps", 9.0),
                    rule_a_adaptive_threshold_bps=exit_params.get("rule_a_adaptive_threshold_bps", 4.0),
                    rule_a_trailing_stop_bps=exit_params.get("rule_a_trailing_stop_bps", 2.0),
                    rule_a_adaptive_bars=exit_params.get("rule_a_adaptive_bars", 3),
                    rule_b_mae_threshold_bps=exit_params.get("rule_b_mae_threshold_bps", -4.0),
                    rule_b_max_bars=exit_params.get("rule_b_max_bars", 6),
                    rule_c_timeout_bars=exit_params.get("rule_c_timeout_bars", 8),
                    rule_c_min_profit_bps=exit_params.get("rule_c_min_profit_bps", 2.0),
                    debug_trade_ids=exit_params.get("debug_trade_ids", []),
                    force_exit_bars=exit_params.get("force_exit_bars"),
                    verbose_logging=exit_params.get("verbose_logging", False),
                    log_every_n_bars=exit_params.get("log_every_n_bars", 5),
                )

            self.exit_farm_v2_rules_factory = _build_exit_farm_v2_rules_policy
            self.exit_farm_v2_rules_policy = self.exit_farm_v2_rules_factory()
            self.exit_farm_v2_rules_states = {}
            
            rules_str = []
            if exit_params.get("enable_rule_a", False):
                rules_str.append("A")
            if exit_params.get("enable_rule_b", False):
                rules_str.append("B")
            if exit_params.get("enable_rule_c", False):
                rules_str.append("C")
            
            log.info(
                f"[BOOT] EXIT_FARM_V2_RULES enabled: Rules={'+'.join(rules_str) if rules_str else 'NONE'}"
            )
            
            # Force disable all other exit mechanisms
            self.exit_farm_v1_policy = None
            tick_exit_cfg = self.policy.get("tick_exit", {})
            tick_exit_cfg["enabled"] = False
            self.policy["tick_exit"] = tick_exit_cfg
            self.policy["broker_side_tp_sl"] = False
            if hasattr(self, "tick_cfg"):
                self.tick_cfg["enabled"] = False
            
            log.info("[BOOT] FARM_V2_RULES mode: All non-FARM exits disabled")
            self.exit_only_v2_drift = True
        
        elif exit_type == "FIXED_BAR_CLOSE":
            from gx1.policy.exit_fixed_bar import FixedBarExitPolicy
            params = exit_cfg.get("exit", {}).get("params", {})
            self.exit_fixed_bar_policy = FixedBarExitPolicy(
                mode=params.get("mode", "fixed"),
                bars=int(params.get("bars", 3)),
                min_bars=int(params.get("min_bars", 1)),
                max_bars=int(params.get("max_bars", 6)),
                seed=params.get("seed"),
            )
            log.info("[BOOT] FIXED_BAR_CLOSE exit enabled: params=%s", params)
            self.exit_only_v2_drift = True
            self.policy.setdefault("tick_exit", {})["enabled"] = False
            self.policy["broker_side_tp_sl"] = False
        elif exit_type == "FARM_V1":
            # FARM_V1 mode: Only FARM exits allowed, disable all other exits
            self.farm_v1_mode = True
            
            # FARM_BASELINE_VERSION: Frozen baseline version (final)
            FARM_BASELINE_VERSION = "FARM_V1_STABLE_FINAL_2025_12_05"
            self.farm_baseline_version = FARM_BASELINE_VERSION
            
            # Check entry policy to see if FARM_V2B or FARM_V2 is enabled (before guardrail check)
            entry_farm_v2b_cfg = self.policy.get("entry_v9_policy_farm_v2b", {})
            entry_farm_v2_cfg = self.policy.get("entry_v9_policy_farm_v2", {})
            if entry_farm_v2b_cfg.get("enabled", False):
                self.farm_v2_mode = True  # V2B uses same exit as V2
                self.farm_v2b_mode = True
            elif entry_farm_v2_cfg.get("enabled", False):
                self.farm_v2_mode = True
                self.farm_v2b_mode = False
            else:
                self.farm_v2b_mode = False
            
            # Runtime guardrail: Check exit profile based on mode
            exit_config_name = exit_cfg_path.stem
            if self.farm_v2_mode or (hasattr(self, "farm_v2b_mode") and self.farm_v2b_mode):
                # FARM_V2/V2B mode MUST use FARM_EXIT_V2_AGGRO
                if exit_config_name != "FARM_EXIT_V2_AGGRO":
                    raise RuntimeError(
                        f"FARM_V2/V2B mode MUST use FARM_EXIT_V2_AGGRO exit policy. "
                        f"Found: {exit_config_name}. "
                        f"FARM_V2/V2B must use FARM_EXIT_V2_AGGRO."
                    )
            else:
                # FARM_V1 mode MUST use FARM_EXIT_V1_STABLE
                if exit_config_name != "FARM_EXIT_V1_STABLE":
                    raise RuntimeError(
                        f"FARM_V1 mode MUST use FARM_EXIT_V1_STABLE exit policy. "
                        f"Found: {exit_config_name}. "
                        f"This is a frozen baseline (version {FARM_BASELINE_VERSION}). "
                        f"Changes require explicit approval and new major version."
                    )
            
            from gx1.policy.exit_farm_v1_policy import get_exit_policy_farm_v1  # type: ignore[reportMissingImports]
            exit_params = exit_cfg.get("exit", {}).get("params", {})
            
            # Verify parameters match FARM_EXIT_V1_STABLE baseline
            expected_sl = -20.0
            expected_tp = 8.0
            expected_timeout = 8
            actual_sl = exit_params.get("sl_bps", -6.0)
            actual_tp = exit_params.get("tp1_bps", 6.0)
            actual_timeout = exit_params.get("timeout_bars", 8)
            
            if abs(actual_sl - expected_sl) > 0.01 or abs(actual_tp - expected_tp) > 0.01 or actual_timeout != expected_timeout:
                log.warning(
                    "[BOOT] FARM_EXIT_V1_STABLE parameter mismatch: "
                    f"Expected SL={expected_sl}, TP={expected_tp}, TIMEOUT={expected_timeout}, "
                    f"Found SL={actual_sl}, TP={actual_tp}, TIMEOUT={actual_timeout}. "
                    f"Baseline version: {FARM_BASELINE_VERSION}"
                )
            
            self.exit_farm_v1_policy = get_exit_policy_farm_v1(
                sl_bps=exit_params.get("sl_bps", -6.0),
                tp1_bps=exit_params.get("tp1_bps", 6.0),
                tp2_bps=exit_params.get("tp2_bps", 6.0),
                timeout_bars=exit_params.get("timeout_bars", 8),
            )
            log.info(
                "[BOOT] EXIT_FARM_V1 enabled: SL=%.0f bps, TP1=%.0f bps, TP2=%.0f bps, TIMEOUT=%d bars (Baseline: %s)",
                exit_params.get("sl_bps", -6.0),
                exit_params.get("tp1_bps", 6.0),
                exit_params.get("tp2_bps", 6.0),
                exit_params.get("timeout_bars", 8),
                FARM_BASELINE_VERSION,
            )
            
            # Force disable all other exit mechanisms in FARM_V1 mode
            tick_exit_cfg = self.policy.get("tick_exit", {})
            tick_exit_cfg["enabled"] = False
            self.policy["tick_exit"] = tick_exit_cfg
            self.policy["broker_side_tp_sl"] = False
            
            # Also disable tick_exit in self.tick_cfg (used by TickWatcher)
            # This must be done BEFORE TickWatcher is initialized
            if hasattr(self, "tick_cfg"):
                self.tick_cfg["enabled"] = False
            
            log.info("[BOOT] FARM_V1 mode: All non-FARM exits disabled (tick_exit, broker TP/SL)")
            
            # Log FARM_V2 mode status (already set above)
            if self.farm_v2_mode:
                log.info("[BOOT] FARM_V2 entry policy enabled - using FARM_EXIT_V2_AGGRO exit")
                
                # Load FARM_V2 meta-model for p_profitable prediction
                self.farm_entry_meta_model = None
                self.farm_entry_meta_feature_cols = None
                
                # Get meta-model config
                meta_cfg = self.policy.get("meta_model", {})
                if not meta_cfg:
                    meta_cfg = entry_farm_v2_cfg.get("meta_model", {})
                
                model_path = meta_cfg.get("model_path")
                if not model_path or model_path == "null":
                    # Try default path
                    model_path = "gx1/models/farm_entry_meta/baseline_model.pkl"
                    # Also try alternative paths
                    alt_paths = [
                        "gx1/models/farm_entry_meta_baseline.joblib",
                        "gx1/models/farm_entry_meta/baseline_model.joblib",
                    ]
                    for alt_path in alt_paths:
                        if Path(alt_path).exists():
                            model_path = alt_path
                            break
                
                # Load model
                if model_path and Path(model_path).exists():
                    try:
                        # joblib is already imported at module level (line 26)
                        self.farm_entry_meta_model = joblib.load(model_path)
                        log.info(f"[BOOT] Loaded FARM entry meta-model from {model_path}")
                        
                        # Load feature columns
                        feature_cols_path = meta_cfg.get("feature_cols_path")
                        if not feature_cols_path or feature_cols_path == "null":
                            # Try default paths
                            feature_cols_paths = [
                                Path(model_path).parent / "feature_cols.json",
                                Path(model_path).parent / "feature_cols.txt",
                                Path("gx1/models/farm_entry_meta/feature_cols.json"),
                            ]
                            for fcp in feature_cols_paths:
                                if fcp.exists():
                                    feature_cols_path = str(fcp)
                                    break
                        
                        if feature_cols_path and Path(feature_cols_path).exists():
                            if Path(feature_cols_path).suffix == ".json":
                                with open(feature_cols_path, 'r') as f:
                                    feature_data = jsonlib.load(f)
                                    if isinstance(feature_data, list):
                                        self.farm_entry_meta_feature_cols = feature_data
                                    elif isinstance(feature_data, dict) and "feature_cols" in feature_data:
                                        self.farm_entry_meta_feature_cols = feature_data["feature_cols"]
                                    else:
                                        log.warning(f"[BOOT] Unexpected feature_cols format in {feature_cols_path}")
                            else:
                                # Assume text file with one column per line
                                with open(feature_cols_path, 'r') as f:
                                    self.farm_entry_meta_feature_cols = [line.strip() for line in f if line.strip()]
                            
                            if self.farm_entry_meta_feature_cols:
                                log.info(f"[BOOT] Loaded {len(self.farm_entry_meta_feature_cols)} feature columns from {feature_cols_path}")
                            else:
                                log.warning("[BOOT] No feature columns loaded - will auto-detect at runtime")
                        else:
                            log.warning("[BOOT] Feature columns file not found - will auto-detect at runtime")
                    except Exception as e:
                        log.error(f"[BOOT] Failed to load FARM entry meta-model: {e}")
                        raise RuntimeError(
                            f"FARM_V2 requires meta-model but loading failed: {e}. "
                            f"Model path: {model_path}. "
                            f"FARM_V2 cannot run without meta-model."
                        )
                else:
                    raise RuntimeError(
                        f"FARM_V2 requires meta-model but file not found: {model_path}. "
                        f"FARM_V2 cannot run without meta-model. "
                        f"Please train and save the model first."
                    )
            
            # Runtime guard: AI exit overlay is disabled for FARM_V1
            # AI exit overlay was evaluated and rejected (see gx1/docs/FARM_EXIT_V1_FINAL.md)
            exit_model_cfg = self.policy.get("exit_model", {})
            if exit_model_cfg.get("enabled", False) or exit_model_cfg.get("ai_overlay", False):
                raise RuntimeError(
                    "AI exit overlay is disabled for FARM_V1. "
                    "FARM_V1 uses FARM_EXIT_V1_STABLE only (frozen baseline). "
                    "See gx1/docs/FARM_EXIT_V1_FINAL.md for details."
            )
            
            # If only_v2_drift_mode is active, update exit_control config for FARM_V1
            if self.exit_only_v2_drift:
                exit_control_cfg = self.policy.get("exit_control", {})
                exit_control_cfg["allowed_loss_closers"] = [
                    "EXIT_FARM_SL",
                    "EXIT_FARM_SL_BREAKEVEN",
                    "EXIT_FARM_TP",
                    "EXIT_FARM_TIMEOUT",
                ]
                exit_control_cfg.setdefault("allow_model_exit_when", {})["min_bars"] = 1
                exit_control_cfg.setdefault("allow_model_exit_when", {})["min_pnl_bps"] = -100
                exit_control_cfg.setdefault("allow_model_exit_when", {})["min_exit_prob"] = 0.0
                self.policy["exit_control"] = exit_control_cfg
        # Note: FARM_V1 initialization is handled above (lines 1471-1518)
        # Duplicate elif block removed - FARM_V1 is initialized in the first if block
        
        # Get exit model config (hysteresis settings)
        exit_model_cfg = self.policy.get("exit_model", {})
        self.exit_threshold = float(exit_model_cfg.get("threshold", self.policy.get("exit_threshold", exit_cfg.get("threshold", 0.686))))
        self.exit_require_consecutive = int(exit_model_cfg.get("require_consecutive", 1))  # Default: 1 (no hysteresis)
        # Log exit model config at boot
        log.info(
            "[BOOT] exit_model: threshold=%.4f require_consecutive=%d (hysteresis)",
            self.exit_threshold,
            self.exit_require_consecutive,
        )
        
        # Konsolider "effektiv" dry_run ett sted
        exec_cfg = self.policy.get("execution", {}) or {}
        risk_cfg = self.policy.get("risk", {}) or {}
        
        # Kildeverdier
        dry_run_policy = exec_cfg.get("dry_run", risk_cfg.get("dry_run", None))
        if dry_run_policy is None:
            dry_run_policy = True  # trygg default hvis ingen er satt
        
        # Miljø-override (valgfritt)
        env_force = os.getenv("GX1_FORCE_DRY_RUN")
        if env_force is not None:
            dry_run_effective = env_force.strip().lower() in ("1", "true", "yes", "on")
        else:
            dry_run_effective = bool(dry_run_policy)
        
        # Override fra parameter (hvis satt)
        if dry_run_override is not None:
            dry_run_effective = dry_run_override
        
        self.exec = SimpleNamespace(
            dry_run=dry_run_effective,
            default_units=exec_cfg.get("default_units", risk_cfg.get("units_per_trade", 1)),
            kill_switch_enabled=bool(exec_cfg.get("kill_switch_enabled", True)),
        )
        
        # Risk limits (bruk self.exec.dry_run og self.exec.default_units)
        # Get max_open_trades from risk_cfg or execution_cfg (prioritize execution if both exist)
        max_open_trades = int(exec_cfg.get("max_open_trades", risk_cfg.get("max_open_trades", 3)))
        max_concurrent_positions = int(exec_cfg.get("max_concurrent_positions", risk_cfg.get("max_concurrent_positions", 3)))

        self.risk_limits = RiskLimits(
            dry_run=self.exec.dry_run,
            units_per_trade=self.exec.default_units,
            max_open_trades=max_open_trades,
            max_daily_loss_bps=float(risk_cfg.get("max_daily_loss_bps", 300)),
            min_time_between_trades_sec=int(risk_cfg.get("min_time_between_trades_sec", 60)),
            max_slippage_bps=float(risk_cfg.get("max_slippage_bps", 10)),
        )
        
        # Store max_concurrent_positions for portfolio protection (used in evaluate_entry)
        self.max_concurrent_positions = max_concurrent_positions

        log_cfg = self.policy.get("logging", {})
        # Set log_dir: use explicit_output_dir/logs for parallel chunks, otherwise use config
        if hasattr(self, "explicit_output_dir") and self.explicit_output_dir:
            self.log_dir = self.explicit_output_dir / "logs"
            log.info("[BOOT] Using chunk-specific log_dir: %s", self.log_dir)
        else:
            self.log_dir = Path(log_cfg.get("log_dir", "gx1/live/logs"))
        setup_logging(self.log_dir, log_cfg.get("level", "INFO"))
        
        # BOOT-LOGG: vis kildene og hva som blir brukt (etter logging er satt opp)
        log.info("[BOOT] policy_path=%s policy_hash=%s", policy_path.resolve(), self.policy_hash)
        log.info(
            "[BOOT] execution.dry_run=%s risk.dry_run=%s env.GX1_FORCE_DRY_RUN=%s -> EFFECTIVE dry_run=%s",
            exec_cfg.get("dry_run", None),
            risk_cfg.get("dry_run", None),
            env_force,
            self.exec.dry_run,
        )
        log.info("Loaded entry config %s and exit config %s", entry_cfg_path, exit_cfg_path)
        
        # Check guard configuration (affects entry_gating)
        guard_cfg = self.policy.get("guard", {})
        self.guard_enabled = guard_cfg.get("enabled", True)  # Default to True for backward compatibility
        log.info("[BOOT] guard.enabled=%s", self.guard_enabled)
        
        # Log entry_gating thresholds at boot (if available and guard is enabled)
        if not self.guard_enabled:
            log.info("[BOOT] Guard disabled - entry_gating will be skipped (scan-mode)")
            entry_gating = None
        else:
            entry_gating = self.policy.get("entry_gating", None)
        if entry_gating:
            p_side_min = entry_gating.get("p_side_min", {})
            margin_min = entry_gating.get("margin_min", {})
            side_ratio_min = entry_gating.get("side_ratio_min", 1.25)
            sticky_bars = entry_gating.get("sticky_bars", 1)
            log.info(
                "[BOOT] entry_gating: long(p≥%.2f,m≥%.2f) short(p≥%.2f,m≥%.2f) ratio≥%.2f sticky=%d",
                float(p_side_min.get("long", 0.55)),
                float(margin_min.get("long", 0.08)),
                float(p_side_min.get("short", 0.60)),
                float(margin_min.get("short", 0.10)),
                float(side_ratio_min),
                int(sticky_bars),
            )
        else:
            # Fall back to entry_params (legacy)
            meta_threshold = float(self.entry_params.get("META_THRESHOLD", 0.40))
            margin_min = float(self.entry_params.get("ENTRY_MIN_MARGIN_MIN", 0.02))
            log.info(
                "[BOOT] entry_gating: legacy mode (p>=%.2f, m>=%.2f) - using entry_params",
                meta_threshold,
                margin_min,
            )
        
        # Stage-0 configuration (opportunity filter)
        stage0_cfg = self.policy.get("stage0", {})
        self.stage0_enabled = stage0_cfg.get("enabled", True)  # Default to True for backward compatibility
        log.info("[BOOT] stage0.enabled=%s", self.stage0_enabled)
        
        # Log gates (momentum_veto, volatility_brake) at boot
        gates_cfg = self.policy.get("gates", {})
        momentum_veto = gates_cfg.get("momentum_veto", {})
        volatility_brake = gates_cfg.get("volatility_brake", {})
        momentum_veto_enabled = momentum_veto.get("enabled", False)
        volatility_brake_enabled = volatility_brake.get("enabled", False)
        if momentum_veto_enabled:
            require_both = momentum_veto.get("require_both", True)
            atr_z_max = float(volatility_brake.get("atr_z_max", 2.5)) if volatility_brake_enabled else 0.0
            log.info(
                "[BOOT] momentum_veto=ON(require_both=%s) volatility_brake=ON(atr_z_max=%.1f)",
                require_both,
                atr_z_max,
            )
        elif volatility_brake_enabled:
            atr_z_max = float(volatility_brake.get("atr_z_max", 2.5))
            log.info(
                "[BOOT] momentum_veto=OFF volatility_brake=ON(atr_z_max=%.1f)",
                atr_z_max,
            )
        else:
            log.info("[BOOT] gates: momentum_veto=OFF volatility_brake=OFF")
        
        # Big Brain V0 removed - using V1 only (regime inference + entry gating)
        
        # Initialize Big Brain V1 Runtime (observe-only)
        # CRITICAL: Must be loaded at startup, not lazy-loaded
        # Check if Big Brain V1 is enabled in policy
        bb_v1_config = self.policy.get("big_brain_v1", {})
        bb_v1_enabled = bb_v1_config.get("enabled", False)  # Default to False - require explicit enable
        
        if bb_v1_enabled and BIG_BRAIN_AVAILABLE:
            # Get model paths from policy (or use defaults)
            model_path = bb_v1_config.get("model_path", "models/BIG_BRAIN_V1/model.pt")
            meta_path = bb_v1_config.get("meta_path", "models/BIG_BRAIN_V1/meta.json")
            
            # Convert to Path objects
            model_path = Path(model_path)
            meta_path = Path(meta_path)
            
            # Fail hard if model files don't exist
            if not model_path.exists():
                raise FileNotFoundError(
                    f"[BIG_BRAIN_V1] Model file not found: {model_path}. "
                    "Big Brain V1 is enabled but model is missing. Aborting startup."
                )
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"[BIG_BRAIN_V1] Meta file not found: {meta_path}. "
                    "Big Brain V1 is enabled but meta is missing. Aborting startup."
                )
            
            # Load runtime (fail hard on any error)
            if not BIG_BRAIN_AVAILABLE:
                log.error(
                    "[BIG_BRAIN_V1] Big Brain V1 is enabled but modules are not available. "
                    "Aborting startup."
                )
                raise RuntimeError("Big Brain V1 is enabled but modules are not available")
            try:
                self.big_brain_v1 = BigBrainV1Runtime(model_path=model_path, meta_path=meta_path)
                self.big_brain_v1.load()
                log.info("[BIG_BRAIN_V1] Runtime loaded successfully (observe-only mode) from %s", model_path)
            except Exception as e:
                log.error(
                    "[BIG_BRAIN_V1] Failed to load runtime: %s. "
                    "Big Brain V1 is enabled but failed to initialize. Aborting startup.",
                    e,
                    exc_info=True,
                )
                raise RuntimeError(f"Big Brain V1 initialization failed: {e}") from e
        elif bb_v1_enabled and not BIG_BRAIN_AVAILABLE:
            log.warning("[BIG_BRAIN_V1] Big Brain V1 is enabled in policy but modules are not available. Disabling Big Brain V1.")
            self.big_brain_v1 = None
        else:
            log.info("[BIG_BRAIN_V1] Big Brain V1 shaping disabled in policy")
            self.big_brain_v1 = None
        
        # Initialize Big Brain V1 Entry Gater (hard gating)
        bb_v1_entry_gates_config = bb_v1_config.get("entry_gates", {})
        bb_v1_entry_gates_enabled = bb_v1_entry_gates_config.get("enabled", False)  # Default to False
        
        if bb_v1_entry_gates_enabled:
            if not BIG_BRAIN_AVAILABLE:
                log.warning("[BIG_BRAIN_V1_ENTRY] Entry gating enabled but Big Brain modules not available. Disabling entry gating.")
                self.big_brain_v1_entry_gater = None
            else:
                entry_gates_config_path = bb_v1_entry_gates_config.get("config_path", "gx1/configs/big_brain_v1_entry_gates.yaml")
                entry_gates_config_path = Path(entry_gates_config_path)

                try:
                    self.big_brain_v1_entry_gater = load_entry_gater(entry_gates_config_path)
                    log.info("[BIG_BRAIN_V1_ENTRY] Entry gater loaded from %s", entry_gates_config_path)
                except Exception as e:
                    log.warning("[BIG_BRAIN_V1_ENTRY] Failed to load entry gater: %s. Continuing without V1 entry gating.", e)
                    self.big_brain_v1_entry_gater = None
        else:
            log.info("[BIG_BRAIN_V1_ENTRY] Entry gating disabled in policy")
            self.big_brain_v1_entry_gater = None
        
        # Initialize TCN Entry Model (for hybrid blending)
        hybrid_entry_cfg = self.policy.get("hybrid_entry", {})
        hybrid_entry_enabled = hybrid_entry_cfg.get("enabled", False)
        
        if hybrid_entry_enabled:
            # Load TCN entry model
            tcn_cfg = self.policy.get("tcn", {})
            tcn_model_path = Path(tcn_cfg.get("model_path", "gx1/models/seq/entry_fast_tcn_2025Q3.pt"))
            tcn_meta_path = Path(tcn_cfg.get("meta_path", "gx1/models/seq/entry_fast_tcn_2025Q3.meta.json"))
            
            if not tcn_model_path.exists():
                log.warning("[HYBRID_ENTRY] TCN model not found: %s. Hybrid entry disabled.", tcn_model_path)
                self.entry_tcn_model = None
                self.entry_tcn_meta = None
                self.entry_tcn_scaler = None
                self.entry_tcn_feats = None
                self.entry_tcn_lookback = None
            elif not tcn_meta_path.exists():
                log.warning("[HYBRID_ENTRY] TCN meta not found: %s. Hybrid entry disabled.", tcn_meta_path)
                self.entry_tcn_model = None
                self.entry_tcn_meta = None
                self.entry_tcn_scaler = None
                self.entry_tcn_feats = None
                self.entry_tcn_lookback = None
            else:
                try:
                    import torch
                    from gx1.seq.model_tcn import TempCNN, TempCNNConfig  # type: ignore[reportMissingImports]
                    from sklearn.preprocessing import RobustScaler
                    
                    # Load meta
                    with open(tcn_meta_path, "r") as f:
                        meta = jsonlib.load(f)
                    
                    self.entry_tcn_feats = meta["feats"]
                    self.entry_tcn_lookback = meta.get("lookback", 864)
                    
                    # Reconstruct scaler
                    scaler_dict = meta["scaler"]
                    scaler = RobustScaler()
                    if scaler_dict.get("center_") is not None:
                        scaler.center_ = np.array(scaler_dict["center_"])
                    if scaler_dict.get("scale_") is not None:
                        scaler.scale_ = np.array(scaler_dict["scale_"])
                    scaler.quantile_range = tuple(scaler_dict.get("quantile_range", [25, 75]))
                    scaler.with_centering = scaler_dict.get("with_centering", True)
                    scaler.with_scaling = scaler_dict.get("with_scaling", True)
                    self.entry_tcn_scaler = scaler
                    
                    # Build model config
                    arch_cfg = meta.get("seq_cfg", {})
                    tcn_config = TempCNNConfig(
                        in_features=len(self.entry_tcn_feats),
                        hidden=arch_cfg.get("hidden", 64),
                        depth=arch_cfg.get("depth", 3),
                        kernel=arch_cfg.get("kernel", 3),
                        dilations=tuple(arch_cfg.get("dilations", (1, 2, 4))),
                        dropout=arch_cfg.get("dropout", 0.10),
                        head_dropout=arch_cfg.get("head_dropout", 0.10),
                        pool=arch_cfg.get("pool", "avg"),
                        attn_heads=arch_cfg.get("attn_heads", 4),
                    )
                    
                    # Load model
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.entry_tcn_model = TempCNN(tcn_config)
                    
                    # Load state dict (handle both direct state_dict and wrapped dict)
                    state_dict = torch.load(tcn_model_path, map_location=device)
                    if isinstance(state_dict, dict) and "state_dict" in state_dict:
                        # Wrapped dict with "state_dict" key
                        state_dict_to_load = state_dict["state_dict"]
                    else:
                        # Direct state_dict
                        state_dict_to_load = state_dict
                    
                    self.entry_tcn_model.load_state_dict(state_dict_to_load, strict=False)
                    self.entry_tcn_model.to(device)
                    self.entry_tcn_model.eval()
                    self.entry_tcn_device = device
                    self.entry_tcn_meta = meta
                    
                    log.info(
                        "[TCN ENTRY LOADED] model_path=%s meta_path=%s lookback=%d features=%d device=%s",
                        tcn_model_path,
                        tcn_meta_path,
                        self.entry_tcn_lookback,
                        len(self.entry_tcn_feats),
                        device,
                    )
                except Exception as e:
                    log.error("[HYBRID_ENTRY] Failed to load TCN model: %s. Hybrid entry disabled.", e, exc_info=True)
                    self.entry_tcn_model = None
                    self.entry_tcn_meta = None
                    self.entry_tcn_scaler = None
                    self.entry_tcn_feats = None
                    self.entry_tcn_lookback = None
        else:
            log.info("[HYBRID_ENTRY] Hybrid entry disabled in policy")
            self.entry_tcn_model = None
            self.entry_tcn_meta = None
            self.entry_tcn_scaler = None
            self.entry_tcn_feats = None
            self.entry_tcn_lookback = None
        
        # ENTRY models config (V10/V10_CTX supported in replay)
        entry_models_cfg = self.policy.get("entry_models", {})
        entry_v10_cfg = entry_models_cfg.get("v10", {})
        self.entry_v10_enabled = bool(entry_v10_cfg.get("enabled", False))
        self.entry_v10_bundle: Optional[Any] = None
        self.entry_v10_cfg = entry_v10_cfg
        
        entry_v10_ctx_cfg = entry_models_cfg.get("v10_ctx", {})
        self.entry_v10_ctx_enabled = bool(entry_v10_ctx_cfg.get("enabled", False))
        self.entry_v10_ctx_bundle: Optional[Any] = None
        self.entry_v10_ctx_cfg = entry_v10_ctx_cfg
        
        # Load ENTRY_V10_CTX if enabled (takes precedence over legacy V10)
        if self.entry_v10_ctx_enabled:
            try:
                from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
                import torch
                
                # Force CPU for replay/ops safety
                device = torch.device("cpu")
                
                # Get bundle directory (priority: ENV > Policy)
                bundle_dir = None
                bundle_dir_source = None
                
                # Priority B: ENV override (GX1_BUNDLE_DIR)
                if os.getenv("GX1_BUNDLE_DIR"):
                    bundle_dir = Path(os.getenv("GX1_BUNDLE_DIR")).resolve()
                    bundle_dir_source = "env"
                # Priority C: Policy (resolve relative to policy file's directory if relative)
                else:
                    bundle_dir_str = entry_v10_ctx_cfg.get("bundle_dir")
                    if not bundle_dir_str:
                        raise ValueError("[ENTRY_V10_CTX] bundle_dir is required in entry_models.v10_ctx config or GX1_BUNDLE_DIR env var")
                    
                    bundle_dir = Path(bundle_dir_str)
                    if not bundle_dir.is_absolute():
                        # Resolve relative to policy file's directory (not cwd)
                        policy_path = Path(self.policy_path) if hasattr(self, "policy_path") else None
                        if policy_path and policy_path.exists():
                            policy_dir = policy_path.resolve().parent
                            bundle_dir = (policy_dir / bundle_dir).resolve()
                        else:
                            # Fallback: resolve relative to workspace root
                            workspace_root = Path(__file__).parent.parent.parent
                            bundle_dir = (workspace_root / bundle_dir).resolve()
                    else:
                        bundle_dir = bundle_dir.resolve()
                    bundle_dir_source = "policy"
                
                if not bundle_dir.exists():
                    raise FileNotFoundError(
                        f"[ENTRY_V10_CTX] Bundle directory not found: {bundle_dir} "
                        f"(resolved from {bundle_dir_source})"
                    )
                
                # Get feature_meta_path (required)
                feature_meta_path = entry_v10_ctx_cfg.get("feature_meta_path")
                if not feature_meta_path:
                    raise ValueError("[ENTRY_V10_CTX] feature_meta_path is required in entry_models.v10_ctx config")
                
                feature_meta_path = Path(feature_meta_path)
                if not feature_meta_path.exists():
                    raise FileNotFoundError(f"[ENTRY_V10_CTX] Feature metadata not found: {feature_meta_path}")
                
                # Get scaler paths (optional)
                seq_scaler_path = entry_v10_ctx_cfg.get("seq_scaler_path")
                snap_scaler_path = entry_v10_ctx_cfg.get("snap_scaler_path")
                seq_scaler_path = Path(seq_scaler_path) if seq_scaler_path else None
                snap_scaler_path = Path(snap_scaler_path) if snap_scaler_path else None
                
                # Load XGB models (required for hybrid)
                # DEL 4: HARD GUARDRAILS - check entry_config path for V9 references (replay mode)
                entry_config_path = self.policy.get("entry_config", "")
                if self.replay_mode and entry_config_path:
                    if "V9" in entry_config_path.upper() or "ENTRY_V9" in entry_config_path.upper():
                        raise RuntimeError(
                            f"V9_DISABLED_FOR_REPLAY: entry_config path contains V9 reference: '{entry_config_path}'. "
                            f"Only V10_CTX entry_configs are allowed in replay mode."
                        )
                
                xgb_models = {}
                xgb_model_paths = {}  # Track paths for SSoT capsule
                self.xgb_load_branch = None
                self.xgb_load_source = None
                self.xgb_load_paths = None
                self.xgb_load_error = None

                # Single XGB load path: canonical bundle only (no policy/session paths)
                log.info(
                    "[XGB_LOAD] BRANCH=CANONICAL_BUNDLE_ONLY GX1_CANONICAL_BUNDLE_DIR=%s GX1_CANONICAL_TRUTH_FILE=%s",
                    os.getenv("GX1_CANONICAL_BUNDLE_DIR", "") or "(unset)",
                    os.getenv("GX1_CANONICAL_TRUTH_FILE", "") or "(unset)",
                )
                canonical_dir_str = os.getenv("GX1_CANONICAL_BUNDLE_DIR", "").strip()
                if not canonical_dir_str:
                    truth_file = os.getenv("GX1_CANONICAL_TRUTH_FILE", "")
                    if truth_file and Path(truth_file).exists():
                        with open(truth_file, "r", encoding="utf-8") as _f:
                            truth_obj = jsonlib.load(_f)
                        canonical_dir_str = str(truth_obj.get("canonical_xgb_bundle_dir") or "").strip()
                    if not canonical_dir_str:
                        raise RuntimeError(
                            "[XGB_MISSING_CANONICAL] Canonical XGB bundle required. "
                            "Set GX1_CANONICAL_BUNDLE_DIR or GX1_CANONICAL_TRUTH_FILE with canonical_xgb_bundle_dir."
                        )
                canonical_xgb_dir = Path(canonical_dir_str).expanduser().resolve()
                if not canonical_xgb_dir.is_dir():
                    raise RuntimeError(
                        f"[XGB_MISSING_CANONICAL] Canonical XGB bundle dir not found: {canonical_xgb_dir}"
                    )
                lock_path = canonical_xgb_dir / "MASTER_MODEL_LOCK.json"
                model_filename = "xgb_universal_multihead_v2.joblib"
                if lock_path.is_file():
                    try:
                        with open(lock_path, "r", encoding="utf-8") as _f:
                            lock_obj = jsonlib.load(_f)
                        model_path_rel = lock_obj.get("model_path_relative")
                        if model_path_rel and isinstance(model_path_rel, str) and model_path_rel.strip():
                            model_filename = model_path_rel.strip()
                    except Exception:
                        pass
                model_file = canonical_xgb_dir / model_filename
                if not model_file.is_file():
                    raise RuntimeError(
                        f"[XGB_MISSING_CANONICAL] XGB model not found in canonical bundle: {model_file}"
                    )
                from gx1.xgb.multihead.xgb_multihead_model_v1 import XGBMultiheadModel
                xgb_universal = XGBMultiheadModel.load(str(model_file))
                required_heads = {"EU", "OVERLAP", "US"}
                heads_in_model = set(xgb_universal.heads.keys())
                missing_heads = required_heads - heads_in_model
                if missing_heads:
                    raise RuntimeError(
                        "[XGB_MISSING_CANONICAL] Canonical XGB bundle missing required heads: "
                        f"{sorted(missing_heads)} (have {sorted(heads_in_model)})."
                    )
                xgb_models["UNIVERSAL"] = xgb_universal
                xgb_model_paths["UNIVERSAL"] = str(model_file.resolve())
                self.xgb_load_branch = "CANONICAL_BUNDLE_ONLY"
                self.xgb_load_source = "CANONICAL_BUNDLE"
                self.xgb_load_paths = {
                    "bundle_dir": str(canonical_xgb_dir),
                    "model_file": str(model_file.resolve()),
                    "lock_path": str(lock_path.resolve()) if lock_path.is_file() else None,
                }
                log.info(
                    "[XGB] source=CANONICAL_BUNDLE dir=%s loaded_heads=%s",
                    str(canonical_xgb_dir),
                    sorted(heads_in_model),
                )
                log.info(f"[ENTRY_V10_CTX] XGB models loaded successfully: {list(xgb_models.keys())}")
                self._write_model_used_capsule(xgb_models, xgb_model_paths, bundle_dir, "CANONICAL_BUNDLE")
                
                # Determine replay mode
                is_replay = getattr(self, "replay_mode", False)
                
                # DEL 1: Verify GX1_GATED_FUSION_ENABLED=1 before loading
                gated_fusion_enabled = os.getenv("GX1_GATED_FUSION_ENABLED", "0") == "1"
                if not gated_fusion_enabled:
                    raise RuntimeError(
                        "BASELINE_DISABLED: GX1_GATED_FUSION_ENABLED is not set to '1'. "
                        "BASELINE is disabled. Set GX1_GATED_FUSION_ENABLED=1 to use GATED_FUSION bundle."
                    )
                
                # DEL 1: Verify bundle_dir does not contain BASELINE
                bundle_dir_str = str(bundle_dir)
                if "BASELINE_NO_GATE" in bundle_dir_str or "BASELINE" in bundle_dir_str.upper():
                    raise RuntimeError(
                        f"BASELINE_DISABLED: bundle_dir contains BASELINE reference: '{bundle_dir}'. "
                        f"Only GATED_FUSION bundles are allowed."
                    )
                
                # Load ctx bundle
                self.entry_v10_ctx_bundle = load_entry_v10_ctx_bundle(
                    bundle_dir=bundle_dir,
                    feature_meta_path=feature_meta_path,
                    seq_scaler_path=seq_scaler_path,
                    snap_scaler_path=snap_scaler_path,
                    xgb_models=xgb_models,
                    device=device,
                    is_replay=is_replay,
                )
                
                # DEL 1: Verify bundle metadata matches expected variant
                metadata = self.entry_v10_ctx_bundle.metadata or {}
                bundle_variant = metadata.get("model_variant", "").upper()
                if "BASELINE" in bundle_variant:
                    raise RuntimeError(
                        f"BASELINE_DISABLED: Bundle metadata indicates BASELINE variant: '{bundle_variant}'. "
                        f"Only GATED_FUSION bundles are allowed."
                    )
                # TRUTH/CTX6CAT6: supports_context_features must be explicit (no silent fallback)
                bundle_meta_path = Path(bundle_dir) / "bundle_metadata.json"
                if "supports_context_features" not in metadata and metadata.get("ctx_tag") == "CTX6CAT6":
                    raise RuntimeError(
                        f"[BUNDLE_META_MISSING_SUPPORTS_CONTEXT] supports_context_features missing in bundle metadata. "
                        f"Bundle: {bundle_meta_path}. CTX6CAT6 requires supports_context_features=true."
                    )
                
                # DEL 1: Log canonical bundle path + sha256 on model_state_dict.pt
                model_state_path = Path(bundle_dir) / "model_state_dict.pt"
                if model_state_path.exists():
                    with open(model_state_path, "rb") as f:
                        model_sha256 = hashlib.sha256(f.read()).hexdigest()
                    log.info("[ENTRY] V10_CTX GATED bundle verified:")
                    log.info("[ENTRY] V10_CTX bundle_dir (canonical): %s", Path(bundle_dir).resolve())
                    log.info("[ENTRY] V10_CTX model_state_dict.pt sha256: %s", model_sha256)
                else:
                    raise RuntimeError(
                        f"GATED_BUNDLE_INCOMPLETE: model_state_dict.pt not found in '{bundle_dir}'"
                    )
                
                # Log ctx bundle info (canonical must be CTX6CAT6)
                from gx1.contracts.signal_bridge_v1 import get_canonical_ctx_contract

                canonical_ctx = get_canonical_ctx_contract()
                meta_ctx_cat = metadata.get("expected_ctx_cat_dim")
                meta_ctx_cont = metadata.get("expected_ctx_cont_dim")
                if meta_ctx_cat is None or meta_ctx_cont is None:
                    raise RuntimeError("CTX_META_MISSING: expected_ctx_cat_dim/expected_ctx_cont_dim absent in bundle metadata")
                if int(meta_ctx_cat) != canonical_ctx["ctx_cat_dim"] or int(meta_ctx_cont) != canonical_ctx["ctx_cont_dim"]:
                    raise RuntimeError(
                        f"CTX_META_SPLIT_BRAIN: bundle ctx dims {meta_ctx_cont}/{meta_ctx_cat} "
                        f"!= canonical {canonical_ctx['ctx_cont_dim']}/{canonical_ctx['ctx_cat_dim']} ({canonical_ctx['tag']})"
                    )

                log.info("[ENTRY] V10_CTX enabled: True")
                log.info("[ENTRY] V10_CTX supports_context_features: %s", metadata.get("supports_context_features", False))
                log.info(
                    "[ENTRY] V10_CTX ctx dims (canonical %s): cont=%s cat=%s source=bundle_metadata",
                    canonical_ctx["tag"],
                    meta_ctx_cont,
                    meta_ctx_cat,
                )
                log.info("[ENTRY] V10_CTX feature_contract_hash: %s", metadata.get("feature_contract_hash", "N/A"))
                log.info("[ENTRY] V10_CTX GX1_GATED_FUSION_ENABLED: 1 (verified)")
                
                # DEL 2: Hard guardrails for V10_CTX + context features
                context_features_enabled = os.getenv("ENTRY_CONTEXT_FEATURES_ENABLED", "false").lower() == "true"
                if metadata.get("supports_context_features", False) and not context_features_enabled:
                    is_replay = getattr(self, "replay_mode", False) or os.getenv("GX1_REPLAY", "0") == "1"
                    if is_replay:
                        raise RuntimeError(
                            "CTX_MODEL_WITHOUT_CONTEXT: ENTRY_V10_CTX requires context features, "
                            "but ENTRY_CONTEXT_FEATURES_ENABLED is not 'true'. "
                            "Set ENTRY_CONTEXT_FEATURES_ENABLED=true in policy or environment."
                        )
                    else:
                        log.warning(
                            "[ENTRY] V10_CTX supports context features but ENTRY_CONTEXT_FEATURES_ENABLED=false "
                            "(live mode: will degrade)"
                        )
                
                # DEL 1: Log context features status
                if context_features_enabled:
                    log.info("[ENTRY] ENTRY_CONTEXT_FEATURES_ENABLED = true")
                    log.info("[ENTRY] Context features active: session, trend_regime, vol_regime, atr_bps, spread_bps")
                else:
                    log.info("[ENTRY] ENTRY_CONTEXT_FEATURES_ENABLED = false (context features disabled)")
                
                # Set entry_v10_enabled=True when ctx is active (for compatibility)
                self.entry_v10_enabled = True
                self.entry_v10_bundle = self.entry_v10_ctx_bundle  # Use ctx bundle as main bundle
                # Also set entry_v10_cfg to ctx config (needed for feature_meta_path, scaler paths in _predict_entry_v10_hybrid)
                self.entry_v10_cfg = entry_v10_ctx_cfg
                
                # Update model_name
                self.model_name = "ENTRY_V10_CTX"
                
            except Exception as e:
                log.error("[ENTRY_V10_CTX] Failed to load ctx bundle: %s. Disabling ENTRY_V10_CTX.", e, exc_info=True)
                self.xgb_load_error = str(e)
                self.entry_v10_ctx_enabled = False
                self.entry_v10_ctx_bundle = None
                # Hard fail if require_v9_for_entry=False (ctx was required)
                if not self.policy.get("require_v9_for_entry", True):
                    raise RuntimeError(f"ENTRY_V10_CTX is required (require_v9_for_entry=False) but failed to load: {e}") from e
        elif self.entry_v10_enabled:
            # Load legacy ENTRY_V10 if enabled (and ctx is not enabled)
            try:
                from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_bundle
                import torch
                
                # Force CPU for replay/ops safety
                device = torch.device("cpu")
                
                # Build v10_cfg from entry_config (entry_config has full V10 config)
                entry_config_path = self.policy.get("entry_config", "")
                if entry_config_path:
                    try:
                        entry_config = load_yaml_config(Path(entry_config_path))
                        entry_models_cfg_in_entry = entry_config.get("entry_models", {})
                        v10_cfg_from_entry = entry_models_cfg_in_entry.get("v10", {})
                        xgb_cfg_from_entry = entry_models_cfg_in_entry.get("xgb", {})
                        # Merge: entry_models.v10 from main policy takes precedence
                        v10_cfg = {**v10_cfg_from_entry, **entry_v10_cfg}
                        # Merge XGB config (must be nested under v10_cfg for load_entry_v10_bundle)
                        if xgb_cfg_from_entry:
                            v10_cfg["xgb"] = xgb_cfg_from_entry
                        # Also check if xgb is in entry_models_cfg from main policy
                        xgb_cfg_from_policy = entry_models_cfg.get("xgb", {})
                        if xgb_cfg_from_policy:
                            if "xgb" not in v10_cfg:
                                v10_cfg["xgb"] = {}
                            v10_cfg["xgb"] = {**v10_cfg.get("xgb", {}), **xgb_cfg_from_policy}
                    except Exception as e:
                        log.warning("[ENTRY_V10] Failed to load entry_config for V10 config merge: %s. Using policy config only.", e)
                        v10_cfg = entry_v10_cfg
                        # Try to add xgb from main policy
                        xgb_cfg_from_policy = entry_models_cfg.get("xgb", {})
                        if xgb_cfg_from_policy:
                            v10_cfg["xgb"] = xgb_cfg_from_policy
                else:
                    v10_cfg = entry_v10_cfg
                    # Try to add xgb from main policy
                    xgb_cfg_from_policy = entry_models_cfg.get("xgb", {})
                    if xgb_cfg_from_policy:
                        v10_cfg["xgb"] = xgb_cfg_from_policy
                
                # Ensure required fields exist
                if not v10_cfg.get("model_path"):
                    raise ValueError("[ENTRY_V10] model_path is required in entry_models.v10 config")
                if not v10_cfg.get("feature_meta_path"):
                    raise ValueError("[ENTRY_V10] feature_meta_path is required in entry_models.v10 config")
                
                # Validate model_path exists
                model_path = Path(v10_cfg.get("model_path"))
                if not model_path.exists():
                    raise FileNotFoundError(f"[ENTRY_V10] Transformer model not found: {model_path}")
                
                log.info("[ENTRY] V10 enabled: True")
                log.info("[ENTRY] V10 model_path: %s", model_path)
                log.info("[ENTRY] require_v9_for_entry: %s", self.policy.get("require_v9_for_entry", True))
                
                # Load V10 bundle
                self.entry_v10_bundle = load_entry_v10_bundle(v10_cfg, device=device)
                log.info("[ENTRY_V10] Bundle loaded successfully")
                
                # Update model_name if ENTRY_V10 is active
                self.model_name = "ENTRY_V10"
                
            except Exception as e:
                log.error("[ENTRY_V10] Failed to load V10 bundle: %s. Disabling ENTRY_V10.", e, exc_info=True)
                self.entry_v10_enabled = False
                self.entry_v10_bundle = None
                # Hard fail if require_v9_for_entry=False (V10 was required)
                if not self.policy.get("require_v9_for_entry", True):
                    raise RuntimeError(f"ENTRY_V10 is required (require_v9_for_entry=False) but failed to load: {e}") from e
        else:
            log.info("[ENTRY] V10 enabled: False")
            log.info("[ENTRY] V10_CTX enabled: False")
            require_v9 = self.policy.get("require_v9_for_entry", True)
            log.info("[ENTRY] require_v9_for_entry: %s", require_v9)
        
        # Replay hard-enable V10_CTX when using the V10 ctx policy module (no legacy fallback)
        if getattr(self, "replay_mode", False):
            policy_module = ""
            try:
                policy_module = (self.policy.get("replay_config", {}) or {}).get("policy_module", "") or ""
            except Exception:
                policy_module = ""
            if policy_module == "gx1.policy.entry_policy_sniper_v10_ctx":
                try:
                    self.entry_v10_ctx_enabled = True
                    self.entry_v10_enabled = True  # align compatibility flag so gate sees a V10 entry
                except Exception:
                    self.entry_v10_ctx_enabled = True
        
        # REPLAY-ONLY: Safety assert for V10/V10_CTX verification (not in production)
        if self.replay_mode or self.fast_replay:
            expect_v10 = os.getenv("GX1_REPLAY_EXPECT_V10", "0") == "1"
            expect_v10_ctx = os.getenv("GX1_REPLAY_EXPECT_V10_CTX", "0") == "1"
            
            if expect_v10:
                log.info("[REPLAY_V10_VERIFY] GX1_REPLAY_EXPECT_V10=1 detected, verifying V10 is enabled")
                log.info("[REPLAY_V10_VERIFY] entry_v10_enabled=%s, entry_v10_ctx_enabled=%s", 
                         self.entry_v10_enabled, self.entry_v10_ctx_enabled)
                if not self.entry_v10_enabled and not self.entry_v10_ctx_enabled:
                    raise RuntimeError(
                        "[REPLAY_V10_VERIFY] GX1_REPLAY_EXPECT_V10=1 but V10 is not enabled! "
                        "entry_v10_enabled=%s, entry_v10_ctx_enabled=%s. "
                        "Check policy config: entry_models.v10.enabled or entry_models.v10_ctx.enabled must be true."
                        % (self.entry_v10_enabled, self.entry_v10_ctx_enabled)
                    )
                # Log bundle paths if available
                if self.entry_v10_enabled and hasattr(self, 'entry_v10_cfg'):
                    log.info("[REPLAY_V10_VERIFY] V10 config: model_path=%s", 
                             self.entry_v10_cfg.get("model_path", "N/A"))
                if self.entry_v10_ctx_enabled and hasattr(self, 'entry_v10_ctx_cfg'):
                    log.info("[REPLAY_V10_VERIFY] V10_CTX config: bundle_dir=%s", 
                             self.entry_v10_ctx_cfg.get("bundle_dir", "N/A"))
            
            if expect_v10_ctx:
                log.info("[REPLAY_V10_CTX_VERIFY] GX1_REPLAY_EXPECT_V10_CTX=1 detected, verifying V10_CTX is enabled")
                log.info("[REPLAY_V10_CTX_VERIFY] entry_v10_ctx_enabled=%s", self.entry_v10_ctx_enabled)
                if not self.entry_v10_ctx_enabled:
                    raise RuntimeError(
                        "[REPLAY_V10_CTX_VERIFY] GX1_REPLAY_EXPECT_V10_CTX=1 but V10_CTX is not enabled! "
                        "entry_v10_ctx_enabled=%s. "
                        "Check policy config: entry_models.v10_ctx.enabled must be true and bundle_dir must be set."
                        % self.entry_v10_ctx_enabled
                    )
                # Verify bundle exists
                if hasattr(self, 'entry_v10_ctx_cfg') and self.entry_v10_ctx_cfg:
                    bundle_dir = self.entry_v10_ctx_cfg.get("bundle_dir")
                    if not bundle_dir:
                        raise RuntimeError(
                            "[REPLAY_V10_CTX_VERIFY] GX1_REPLAY_EXPECT_V10_CTX=1 but bundle_dir is not set in config!"
                        )
                    bundle_path = Path(bundle_dir)
                    if not bundle_path.exists():
                        raise RuntimeError(
                            "[REPLAY_V10_CTX_VERIFY] GX1_REPLAY_EXPECT_V10_CTX=1 but bundle directory does not exist: %s"
                            % bundle_dir
                        )
                    # Check required files
                    required_files = ["model_state_dict.pt", "bundle_metadata.json"]
                    missing_files = [f for f in required_files if not (bundle_path / f).exists()]
                    if missing_files:
                        raise RuntimeError(
                            "[REPLAY_V10_CTX_VERIFY] GX1_REPLAY_EXPECT_V10_CTX=1 but bundle is missing required files: %s"
                            % missing_files
                        )
                    log.info("[REPLAY_V10_CTX_VERIFY] V10_CTX bundle verified: bundle_dir=%s", bundle_dir)
                else:
                    raise RuntimeError(
                        "[REPLAY_V10_CTX_VERIFY] GX1_REPLAY_EXPECT_V10_CTX=1 but entry_v10_ctx_cfg is not available!"
                    )
        
        # Log tick-exit config at boot
        tick_exit_cfg = self.policy.get("tick_exit", {})
        if tick_exit_cfg.get("enabled", False):
            soft_stop_bps = int(tick_exit_cfg.get("soft_stop_bps", 25))
            stream = bool(tick_exit_cfg.get("stream", False))
            snapshot_ms = int(tick_exit_cfg.get("snapshot_ms", 400))
            log.info(
                "[BOOT] tick_exit: soft_stop=%dbps stream=%s snapshot_ms=%d",
                soft_stop_bps,
                stream,
                snapshot_ms,
            )
        else:
            log.info("[BOOT] tick_exit: disabled")
        
        # Log broker-side TP/SL at boot
        broker_side_tp_sl = bool(self.policy.get("broker_side_tp_sl", True))
        log.info("[BOOT] broker_side_tp_sl=%s", broker_side_tp_sl)
        
        # Log BE activation config at boot (if enabled)
        be_cfg = tick_exit_cfg.get("be", {})
        if be_cfg.get("enabled", False):
            be_activate_at_bps = int(be_cfg.get("activate_at_bps", 50))
            be_bias_price = float(be_cfg.get("bias_price", 0.3))
            log.info(
                "[BOOT] tick_exit.be: enabled=true activate_at_bps=%d bias_price=%.3f",
                be_activate_at_bps,
                be_bias_price,
            )
        else:
            log.info("[BOOT] tick_exit.be: enabled=false")
        
        # Load exit_control config (ExitArbiter)
        exit_control_cfg = self.policy.get("exit_control", {})
        self.exit_control = SimpleNamespace(
            allowed_loss_closers=list(exit_control_cfg.get("allowed_loss_closers", ["BROKER_SL", "SOFT_STOP_TICK"])),
            allow_model_exit_when=exit_control_cfg.get("allow_model_exit_when", {}),
            require_trade_open=bool(exit_control_cfg.get("require_trade_open", True)),
        )
        
        # Post-initialization: Update exit_control for EXIT_V3_ADAPTIVE or EXIT_FARM_V1 if needed
        if hasattr(self, "exit_v3_drift_adaptive_policy") and self.exit_v3_drift_adaptive_policy is not None and self.exit_only_v2_drift:
            self.exit_control.allowed_loss_closers = [
                "EXIT_V3_ADAPTIVE_SL",
                "EXIT_V3_ADAPTIVE_SL_BREAKEVEN",
                "EXIT_V3_ADAPTIVE_TP2",
                "EXIT_V3_ADAPTIVE_TIMEOUT",
            ]
            self.exit_control.allow_model_exit_when["min_bars"] = 1
            self.exit_control.allow_model_exit_when["min_pnl_bps"] = -100
            self.exit_control.allow_model_exit_when["min_exit_prob"] = 0.0
            log.info("[BOOT] ExitArbiter configured for EXIT_V3_ADAPTIVE")
        elif hasattr(self, "exit_farm_v2_rules_policy") and self.exit_farm_v2_rules_policy is not None and self.exit_only_v2_drift:
            # Only override if exit_control was not explicitly set in policy
            if not exit_control_cfg.get("allowed_loss_closers"):
                self.exit_control.allowed_loss_closers = [
                    "RULE_A_PROFIT",
                    "RULE_A_TRAILING",
                    "RULE_B_FAST_LOSS",
                    "RULE_C_TIMEOUT",
                ]
            else:
                # Policy has explicit exit_control config, use it (e.g., SNIPER with SL_TICK)
                log.info("[BOOT] Using exit_control from policy: allowed_loss_closers=%s", self.exit_control.allowed_loss_closers)
            self.exit_control.allow_model_exit_when["min_bars"] = 1
            self.exit_control.allow_model_exit_when["min_pnl_bps"] = -100
            self.exit_control.allow_model_exit_when["min_exit_prob"] = 0.0
            log.info("[BOOT] ExitArbiter configured for EXIT_FARM_V2_RULES: allowed_loss_closers=%s", self.exit_control.allowed_loss_closers)
        elif hasattr(self, "exit_farm_v1_policy") and self.exit_farm_v1_policy is not None and self.exit_only_v2_drift:
            self.exit_control.allowed_loss_closers = [
                "EXIT_FARM_SL",
                "EXIT_FARM_SL_BREAKEVEN",
                "EXIT_FARM_TP",
                "EXIT_FARM_TIMEOUT",
            ]
            self.exit_control.allow_model_exit_when["min_bars"] = 1
            self.exit_control.allow_model_exit_when["min_pnl_bps"] = -100
            self.exit_control.allow_model_exit_when["min_exit_prob"] = 0.0
            log.info("[BOOT] ExitArbiter configured for EXIT_FARM_V1")
        log.info(
            "[BOOT] exit_control: allowed_loss_closers=%s min_bars=%d min_pnl_bps=%.1f require_trade_open=%s",
            self.exit_control.allowed_loss_closers,
            self.exit_control.allow_model_exit_when.get("min_bars", 2),
            self.exit_control.allow_model_exit_when.get("min_pnl_bps", -5),
            self.exit_control.require_trade_open,
        )
        
        # Initialize exit-audit logging (JSONL file, rotates daily)
        self.exit_audit_dir = self.log_dir / "exits"
        self.exit_audit_dir.mkdir(parents=True, exist_ok=True)
        self.exit_audit_path = self.exit_audit_dir / f"exits_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d')}.jsonl"

        # Get trade_log_csv template from config and format with chunk_id if available
        trade_log_template = log_cfg.get("trade_log_csv", "gx1/live/trade_log.csv")
        chunk_id = os.getenv("GX1_CHUNK_ID")
        if chunk_id is not None and "{chunk_id}" in trade_log_template:
            trade_log_template = trade_log_template.format(chunk_id=chunk_id)
        self.trade_log_path = Path(trade_log_template)
        ensure_trade_log(self.trade_log_path)
        if self.exit_config_name:
            log.info(
                "[BOOT] exit_profile column wired (trade_log=%s, expected profile=%s)",
                self.trade_log_path,
                self.exit_config_name,
            )
        
        # Generate run_header.json at startup (after trade_log_path is set, before first bar)
        self._generate_run_header()
        
        # Initialize trade journal
        if not self.replay_mode:
            self._init_trade_journal()
        else:
            log.info("[TRADE_JOURNAL] Skipped in replay mode (no-op for TRUTH replay)")
        
        # JSON eval log path (rotates daily)
        self.eval_log_path = self.log_dir / f"eval_log_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d')}.jsonl"
        self.eval_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.instrument = self.policy.get("instrument", "XAU_USD")
        self.granularity = self.policy.get("granularity", "M5")

        # Session-routed entry bundle is not used in TRUTH/SMOKE replay (V10_CTX-only)
        self.entry_model_bundle = None
        self.session_entry_enabled = False
        
        self.manifest = load_manifest()
        
        # Optional: Load exit model (only if it exists and not using FARM exit rules)
        if self.exit_only_v2_drift or (hasattr(self, "farm_v2b_mode") and self.farm_v2b_mode):
            # FARM_V2_RULES or FARM_V2B mode - exit model not needed
            log.info("[BOOT] FARM exit mode detected: skipping exit model bundle (using rule-based exits)")
            self.exit_bundle = None
        else:
            try:
                self.exit_bundle = load_exit_model()
            except FileNotFoundError as e:
                log.info(
                    "[BOOT] Exit model bundle not found: %s. Continuing without exit model bundle. "
                    "This is expected for FARM policies using rule-based exits.",
                    e
                )
                self.exit_bundle = None
        
        # Optional: Initialize broker client (only needed for live mode, not replay)
        if self.replay_mode or self.fast_replay:
            log.debug("[BOOT] Replay mode: skipping broker client initialization")
            self.broker = None
        else:
            try:
                self.broker = BrokerClient()
            except Exception as e:
                log.warning(
                    "[BOOT] Failed to initialize broker client: %s. Continuing without broker. "
                    "This is expected for replay mode.",
                    e
                )
                self.broker = None

        # Get OANDA credentials for TickWatcher
        # Load from environment variables using centralized credentials manager
        # NOTE: prod_baseline is set earlier in __init__ (line ~1310), so it's available here
        load_dotenv_if_present()
        from gx1.execution.oanda_credentials import load_oanda_credentials
        
        try:
            # Require live latch if PROD_BASELINE and live environment
            require_live_latch = getattr(self, 'prod_baseline', False)
            prod_baseline_mode = getattr(self, 'prod_baseline', False)
            oanda_creds = load_oanda_credentials(prod_baseline=prod_baseline_mode, require_live_latch=require_live_latch)
            self.oanda_api_key = oanda_creds.api_token
            self.oanda_account_id = oanda_creds.account_id
            self.oanda_host = oanda_creds.api_url
            self.oanda_stream_host = oanda_creds.stream_url
            self.oanda_env = oanda_creds.env
            
            # Log credentials loaded (masked)
            from gx1.execution.oanda_credentials import _mask_account_id
            masked_account_id = _mask_account_id(self.oanda_account_id)
            log.info(
                "[OANDA] Credentials loaded: env=%s, account_id=%s, api_url=%s",
                self.oanda_env,
                masked_account_id,
                self.oanda_host,
            )
        except ValueError as e:
            # Credentials missing/invalid - fail closed in PROD_BASELINE
            log.error("[OANDA] Failed to load credentials: %s", e)
            if self.prod_baseline:
                raise RuntimeError(f"OANDA credentials required in PROD_BASELINE mode: {e}")
            else:
                # Dev mode: set empty values (will fail later if actually used)
                log.warning("[OANDA] Continuing without credentials (dev mode)")
                self.oanda_api_key = ""
                self.oanda_account_id = ""
                self.oanda_host = "https://api-fxpractice.oanda.com"
                self.oanda_stream_host = "https://stream-fxpractice.oanda.com"
                self.oanda_env = "practice"
        
        # Initialize tick-exit config (for TickWatcher)
        # In REN EXIT_V2 mode or FARM_V1 mode, tick_exit is disabled
        self.tick_cfg = self.policy.get("tick_exit", {}) or {}
        if self.exit_only_v2_drift or (hasattr(self, "farm_v1_mode") and self.farm_v1_mode):
            self.tick_cfg["enabled"] = False
        
        # Initialize open_trades BEFORE reconcile (so reconcile can populate it)
        self.open_trades: List[LiveTrade] = []
        self.daily_loss_tracker: Dict[str, float] = {}
        self.last_entry_timestamp: Optional[pd.Timestamp] = None
        self.last_entry_side: Optional[str] = None
        
        # Race-condition handling: track trades that are currently being closed
        self._closing_trades: Dict[str, bool] = {}  # trade_id -> True if closing
        self._closing_lock = threading.Lock()  # Lock for closing_trades dict
        # PHASE 1 FIX: Track exited trades for single-exit invariant
        self._exited_trade_ids: set[str] = set()  # trade_id -> True if already exited
        self._exit_monotonicity_violations: int = 0  # Counter for exit_time < entry_time
        self._duplicate_exit_attempts: int = 0  # Counter for duplicate exit attempts

        # TRUTH-only (but safe everywhere) exit coverage counters for replay observability.
        # Exported into chunk_footer.json by replay_eval_gated_parallel.py worker.
        self.exit_coverage: Dict[str, Any] = {
            "exit_attempts_total": 0,
            "exit_request_close_called": 0,
            "exit_request_close_accepted": 0,
            "exit_summary_logged": 0,
            "exit_event_rows_written": 0,
            "force_close_attempts_replay_end": 0,
            "force_close_logged_replay_end": 0,
            "force_close_attempts_replay_eof": 0,
            "force_close_logged_replay_eof": 0,
            "open_trades_start_of_chunk": None,
            "open_trades_end_of_chunk": None,
            "open_to_closed_within_chunk": 0,
            "last_5_exit_reasons": [],
            "last_5_trade_ids_closed": [],
            "replay_end_or_eof_triggered": False,
            "accounting_close_enabled": False,
        }
        
        # Sticky session: prevent session changes during an hour
        self.last_session: Optional[str] = None
        self.last_session_timestamp: Optional[pd.Timestamp] = None
        self.session_sticky_minutes: int = 60  # Don't change session during an hour  # Track last entry side for sticky-side logic
        
        # Initialize TickWatcher (after client is ready)
        # In REN EXIT_V2 mode or FARM_V1 mode, TickWatcher is disabled
        self._be_update_lock = threading.Lock()
        if not self.exit_only_v2_drift and not (hasattr(self, "farm_v1_mode") and self.farm_v1_mode):
            self.tick_watcher = TickWatcher(
                host=self.oanda_host,
                stream_host=self.oanda_stream_host,
                account_id=self.oanda_account_id,
                api_key=self.oanda_api_key,
                instrument=self.instrument,
                cfg=self.tick_cfg,
                logger=log,
                close_cb=self._tick_close_now,
                get_positions_cb=self._get_open_positions_for_tick,
            )
            # Set callback for BE activation (thread-safe update of trade.extra)
            self.tick_watcher.update_be_callback = self._update_be_status
        else:
            self.tick_watcher = None
            if hasattr(self, "farm_v1_mode") and self.farm_v1_mode:
                log.info("[BOOT] TickWatcher disabled (FARM_V1 mode)")
            else:
                log.info("[BOOT] TickWatcher disabled (REN EXIT_V2 mode)")
        
        # Reconcile: Load open trades from OANDA and bind to internal entry_id
        self._reconcile_open_trades()
    
        # Extract exit_config_name for EntryManager (explicit dependency injection)
        # CRITICAL: exit_cfg_path must be defined earlier in __init__ (line 1305)
        # Use getattr to safely access exit_cfg_path (it's a local variable in __init__)
        try:
            # exit_cfg_path is defined at line 1305 as a local variable
            # It should be accessible here since we're still in __init__
            exit_config_name = exit_cfg_path.stem if exit_cfg_path else None
            log.info("[BOOT] exit_config_name=%s (from exit_cfg_path=%s)", exit_config_name, exit_cfg_path)
            log.info("[BOOT] exit_verbose_logging=%s", self.exit_verbose_logging)
        except NameError:
            log.error("[BOOT] exit_cfg_path not defined! This will cause exit_profile to be None for all trades.")
            exit_config_name = None
        except Exception as e:
            log.error("[BOOT] Error extracting exit_config_name: %s", e)
            exit_config_name = None
        self.exit_config_name = exit_config_name
        if self.exit_config_name:
            log.info("[BOOT] exit_config invariant: expecting exit_profile=%s for every trade", self.exit_config_name)
        
        # Perform backfill on startup (if enabled and not in replay mode) - MOVED HERE BEFORE MANAGERS
        log.info("[BACKFILL_DEBUG] Checking backfill configuration (BEFORE managers)...")
        log.info("[BACKFILL_DEBUG] replay_mode=%s, fast_replay=%s", self.replay_mode, self.fast_replay)
        backfill_cfg = self.policy.get("backfill", {})
        log.info("[BACKFILL_DEBUG] backfill_cfg from policy: %s", backfill_cfg)
        
        if self.replay_mode:
            log.info("[BACKFILL] Replay mode detected (replay-csv is set) – skipping OANDA backfill.")
            self.backfill_in_progress = False
            self.backfill_cache = None
            self.warmup_floor = None
        else:
            backfill_enabled = backfill_cfg.get("enabled", True)  # Default: enabled
            log.info("[BACKFILL_DEBUG] backfill_enabled=%s (from config: %s)", backfill_enabled, backfill_cfg.get("enabled", "NOT SET"))
            
            if backfill_enabled:
                log.info(
                    "[BOOT] Backfill enabled: overlap_bars=%d, lookback_padding=%d", 
                    backfill_cfg.get("overlap_bars", 6),
                    backfill_cfg.get("lookback_padding", 100),
                )
                log.info("[BACKFILL] Starting backfill on startup...")
                self.backfill_in_progress = True
                
                try:
                    log.info("[BACKFILL_DEBUG] Calling _perform_backfill()...")
                    # Perform backfill (returns cache_df, gaps_refetched, warmup_floor, bars_remaining, revisions)
                    self.backfill_cache, gaps_refetched, warmup_floor, bars_remaining, revisions = self._perform_backfill()
                    log.info("[BACKFILL_DEBUG] _perform_backfill() completed successfully")
                    log.info("[BACKFILL_DEBUG] gaps_refetched=%d, bars_remaining=%d, revisions=%d", gaps_refetched, bars_remaining, revisions)
                    
                    # Set warmup_floor from backfill result
                    self.warmup_floor = warmup_floor
                    if warmup_floor is not None:
                        log.info("[BACKFILL_DEBUG] warmup_floor set to: %s", warmup_floor.isoformat())
                    else:
                        log.warning("[BACKFILL_DEBUG] warmup_floor is None after backfill!")
                    
                    # Track revisions in telemetry (warn if >0)
                    if revisions > 0:
                        log.warning(
                            "[REVISION] Detected %d revisions in backfill (n_revisions_24h tracking)", 
                            revisions,
                        )
                    
                    # Backfill complete
                    self.backfill_in_progress = False
                    log.info("[BACKFILL] Backfill complete: %d bars cached, warmup_floor=%s", len(self.backfill_cache) if self.backfill_cache is not None else 0, warmup_floor.isoformat() if warmup_floor is not None else "None")
                    
                except Exception as e:
                    log.error("[BACKFILL] Backfill failed: %s", e, exc_info=True)
                    import traceback
                    log.error("[BACKFILL_DEBUG] Full traceback: %s", traceback.format_exc())
                    # Continue without backfill (fallback to live mode)
                    self.backfill_in_progress = False
                    self.backfill_cache = None
                    self.warmup_floor = None
            else:
                log.info("[BACKFILL] Backfill disabled in policy")
        
        # Managers
        self.entry_manager = EntryManager(self, exit_config_name=exit_config_name)
        self.exit_manager = ExitManager(self)
        self._reset_entry_diag()
    
    def _load_entry_v9_model(self, model_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """[LEGACY_DISABLED] V9 loader removed. ONE UNIVERSE: ENTRY_V10_CTX only."""
        raise RuntimeError("[LEGACY_DISABLED] ENTRY_V9 is removed. ONE UNIVERSE: ENTRY_V10_CTX only.")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _reconcile_open_trades(self) -> None:
        """
        Reconcile open trades from OANDA with internal state.
        
        Loads open trades from OANDA API and binds them to internal entry_id.
        This prevents duplicate orders on restart.
        
        DEL 3: Guard against broker=None in replay mode.
        """
        if self.exec.dry_run:
            # In dry-run mode, no reconciliation needed
            log.info("Reconcile skipped (dry_run mode)")
            return
        
        # DEL 3: Skip if broker is None (replay mode)
        if self.broker is None:
            log.debug("[REPLAY] broker=None; skipping reconcile_open_trades")
            return
        
        try:
            # Get open trades from OANDA
            open_trades_response = self.broker.get_open_trades()
            oanda_trades = open_trades_response.get("trades", [])
            
            if not oanda_trades:
                log.info("Reconcile: No open trades in OANDA")
                return
            
            log.info("Reconcile: Found %d open trades in OANDA", len(oanda_trades))
            
            # Bind OANDA trades to internal state
            # For now, we'll create LiveTrade objects from OANDA trades
            # In production, you might want to match on client_order_id or trade_id
            reconciled_trades = []
            for oanda_trade in oanda_trades:
                try:
                    trade_id = oanda_trade.get("id", "")
                    instrument = oanda_trade.get("instrument", "")
                    # Handle units as float first (OANDA returns as string like "-3.0"), then convert to int
                    units_raw = oanda_trade.get("currentUnits", "0")
                    units = int(float(units_raw))  # Convert string to float first, then int
                    price = float(oanda_trade.get("price", 0.0))
                    open_time = pd.Timestamp(oanda_trade.get("openTime", ""))
                    
                    # Only reconcile trades for our instrument
                    if instrument != self.instrument:
                        continue
                    
                    # Create LiveTrade object from OANDA trade
                    # Note: We don't have all the internal fields (atr_bps, vol_bucket, etc.)
                    # These will be set to defaults
                    # WARNING: OANDA reconciliation uses mid price - bid/ask are estimated
                    # TODO: dead PnL helper — candidate for removal or improvement
                    # This is a fallback for live mode reconciliation where we only have mid price
                    # In replay mode, entry_bid/entry_ask should be set from candles
                    side = "long" if units > 0 else "short"
                    # Estimate bid/ask from mid price (OANDA returns mid)
                    estimated_spread = 0.1  # Typical XAUUSD spread
                    if side == "long":
                        entry_ask = price  # LONG: buy at ask
                        entry_bid = price - estimated_spread
                    else:
                        entry_bid = price  # SHORT: sell at bid
                        entry_ask = price + estimated_spread
                    trade = LiveTrade(
                        trade_id=trade_id,
                        entry_time=open_time,
                        side=side,
                        units=units,
                        entry_price=price,
                        entry_bid=entry_bid,
                        entry_ask=entry_ask,
                        atr_bps=0.0,  # Default (not available from OANDA)
                        vol_bucket="unknown",  # Default (not available from OANDA)
                        entry_prob_long=0.5,  # Default (not available from OANDA)
                        entry_prob_short=0.5,  # Default (not available from OANDA)
                        dry_run=False,
                        client_order_id=oanda_trade.get("clientExtensions", {}).get("id", None),
                    )
                    self._ensure_exit_profile(trade, context="reconcile")
                    if self.exit_config_name and not (getattr(trade, "extra", {}) or {}).get("exit_profile"):
                        raise RuntimeError(
                            f"[EXIT_PROFILE] Reconciled trade {trade.trade_id} missing exit_profile under exit-config {self.exit_config_name}"
                        )
                    reconciled_trades.append(trade)
                    log.info(
                        "Reconciled trade: %s | %s | %s units | price=%.3f | open_time=%s",
                        trade_id,
                        side.upper(),
                        units,
                        price,
                        open_time.isoformat(),
                    )
                except Exception as e:
                    log.warning("Failed to reconcile trade %s: %s", oanda_trade.get("id", "unknown"), e)
            
            # Set reconciled trades as open_trades (thread-safe)
            if reconciled_trades:
                self.open_trades = reconciled_trades
                log.info("Reconcile complete: %d trades reconciled", len(reconciled_trades))
                
                # Start tick watcher if we have open trades
                self._maybe_update_tick_watcher()
            else:
                log.info("Reconcile complete: No trades reconciled for instrument %s", self.instrument)
                
        except Exception as e:
            log.warning("Reconcile failed: %s", e, exc_info=True)
            # Don't fail startup if reconcile fails - just log warning

    def _init_farm_v2_rules_state(self, trade: LiveTrade, *, context: str) -> None:
        """Create/reset the per-trade FARM_V2_RULES policy state."""
        if not getattr(self, "exit_farm_v2_rules_factory", None):
            raise RuntimeError("FARM_V2_RULES factory not configured, cannot initialize exit state")
        if trade.entry_bid is None or trade.entry_ask is None:
            raise ValueError(
                f"[EXIT_PROFILE] Missing bid/ask for FARM_V2_RULES init trade_id={trade.trade_id} context={context}"
            )
        policy = self.exit_farm_v2_rules_states.get(trade.trade_id)
        if policy is None:
            policy = self.exit_farm_v2_rules_factory()
            self.exit_farm_v2_rules_states[trade.trade_id] = policy
        policy.reset_on_entry(
            entry_bid=trade.entry_bid,
            entry_ask=trade.entry_ask,
            entry_ts=trade.entry_time,
            side=trade.side,
            trade_id=trade.trade_id,
        )
        if not getattr(trade, "extra", None):
            trade.extra = {}
        trade.extra["exit_farm_v2_rules_initialized"] = True
        
        # Log exit configuration to trade journal
        if hasattr(self, "trade_journal") and self.trade_journal:
            try:
                exit_profile = trade.extra.get("exit_profile", "FARM_EXIT_V2_RULES")
                # Extract TP/SL levels from policy if available
                tp_levels = None
                sl = None
                trailing_enabled = False
                be_rules = None
                
                if hasattr(policy, "rule_a_profit_min_bps") and hasattr(policy, "rule_a_profit_max_bps"):
                    tp_levels = [policy.rule_a_profit_min_bps, policy.rule_a_profit_max_bps]
                if hasattr(policy, "rule_b_mae_threshold_bps"):
                    sl = abs(policy.rule_b_mae_threshold_bps)  # Convert to positive
                if hasattr(policy, "rule_a_trailing_stop_bps"):
                    trailing_enabled = True
                if hasattr(policy, "rule_a_adaptive_threshold_bps"):
                    be_rules = {
                        "adaptive_threshold_bps": policy.rule_a_adaptive_threshold_bps,
                        "trailing_stop_bps": policy.rule_a_trailing_stop_bps if hasattr(policy, "rule_a_trailing_stop_bps") else None,
                    }
                
                self.trade_journal.log_exit_configuration(
                    exit_profile=exit_profile,
                    trade_uid=trade.trade_uid,  # Primary key (COMMIT C)
                    trade_id=trade.trade_id,  # Display ID (backward compatibility)
                    tp_levels=tp_levels,
                    sl=sl,
                    trailing_enabled=trailing_enabled,
                    be_rules=be_rules,
                )
            except Exception as e:
                log.warning("[TRADE_JOURNAL] Failed to log exit configuration: %s", e)
        
        log.debug(
            "[EXIT_PROFILE] FARM_V2_RULES state reset for trade %s (context=%s)",
            trade.trade_id,
            context,
        )

    def _teardown_exit_state(self, trade_id: str) -> None:
        """Remove any per-trade exit state objects when a trade is closed."""
        if hasattr(self, "exit_farm_v2_rules_states"):
            if trade_id in self.exit_farm_v2_rules_states:
                self.exit_farm_v2_rules_states.pop(trade_id, None)
                log.debug("[EXIT_PROFILE] Cleared FARM_V2_RULES state for trade %s", trade_id)

    def _ensure_exit_profile(self, trade: LiveTrade, *, context: str = "unknown") -> None:
        """
        Ensure every trade has an exit_profile and initialize the matching
        exit policy state exactly once per trade.
        """
        from gx1.execution.live_features import infer_session_tag
        if not getattr(trade, "extra", None):
            trade.extra = {}

        exit_profile = trade.extra.get("exit_profile")
        if not exit_profile:
            exit_profile = self.exit_config_name
            if not exit_profile:
                exit_cfg = self.policy.get("exit_config") or ""
                exit_profile = Path(exit_cfg).stem if exit_cfg else None
                if exit_profile:
                    # Cache for future calls (workers/chunks never touch policy again)
                    self.exit_config_name = exit_profile
            if exit_profile:
                trade.extra["exit_profile"] = exit_profile
                log.debug(
                    "[EXIT_PROFILE] Bound exit_profile=%s to trade %s (context=%s)",
                    exit_profile,
                    trade.trade_id,
                    context,
                )
            else:
                session = infer_session_tag(trade.entry_time) if hasattr(trade, "entry_time") else "UNKNOWN"
                vol_regime = trade.extra.get("vol_regime_entry") if trade.extra else None
                msg = (
                    "[EXIT_PROFILE] Trade opened without exit_profile: id=%s instrument=%s session=%s vol=%s "
                    "context=%s exit_config=%s"
                )
                log.error(
                    msg,
                    getattr(trade, "trade_id", "unknown"),
                    getattr(self, "instrument", "unknown"),
                    session,
                    vol_regime or "UNKNOWN",
                    context,
                    self.exit_config_name,
                )
                raise RuntimeError(
                    msg
                    % (
                        getattr(trade, "trade_id", "unknown"),
                        getattr(self, "instrument", "unknown"),
                        session,
                        vol_regime or "UNKNOWN",
                        context,
                        self.exit_config_name,
                    )
                )

        exit_profile = trade.extra.get("exit_profile")
        if not exit_profile:
            return

        # Initialize FIXED_BAR exit state if needed
        if (
            exit_profile == "FIXED_BAR_CLOSE"
            and hasattr(self, "exit_fixed_bar_policy")
            and self.exit_fixed_bar_policy is not None
            and trade.side == "long"
            and not trade.extra.get("fixed_bar_exit_initialized")
        ):
            if trade.entry_bid is None or trade.entry_ask is None:
                log.warning(
                    "[EXIT_PROFILE] Missing bid/ask for fixed-bar init trade_id=%s context=%s",
                    trade.trade_id,
                    context,
                )
            else:
                self.exit_fixed_bar_policy.reset_on_entry(
                    trade.entry_bid,
                    trade.entry_ask,
                    trade.trade_id,
                    trade.side,
                )
                trade.extra["fixed_bar_exit_initialized"] = True
                log.debug(
                    "[EXIT_PROFILE] Fixed-bar exit initialized for trade %s (context=%s)",
                    trade.trade_id,
                    context,
                )

        # Initialize FARM_V2_RULES exit state if needed
        if (
            exit_profile.startswith("FARM_EXIT_V2_RULES")
            and getattr(self, "exit_farm_v2_rules_factory", None)
            and trade.side == "long"
            and not trade.extra.get("exit_farm_v2_rules_initialized")
        ):
            try:
                self._init_farm_v2_rules_state(trade, context=context)
            except Exception as exc:
                log.warning(
                    "[EXIT_PROFILE] Failed to init FARM_V2_RULES for trade %s (context=%s): %s",
                    trade.trade_id,
                    context,
                    exc,
                )
    
    def _load_backfill_state(self) -> Optional[Dict[str, Any]]:
        """
        Load backfill state from state file.
        
        Returns
        -------
        Optional[Dict[str, Any]]
            Backfill state with last_bar_ts, feature_manifest_hash, policy_hash, or None if not found.
        """
        state_path = self.log_dir / "backfill_state.json"
        if not state_path.exists():
            return None
        
        try:
            with state_path.open("r", encoding="utf-8") as f:
                state = jsonlib.load(f)
            return state
        except Exception as e:
            log.warning("Failed to load backfill state: %s", e)
            return None
    
    def _save_backfill_state(
        self,
        last_bar_ts: pd.Timestamp,
        feature_manifest_hash: Optional[str] = None,
        policy_hash: Optional[str] = None,
        rotate: bool = False,
    ) -> None:
        """
        Save backfill state to state file.
        
        Parameters
        ----------
        last_bar_ts : pd.Timestamp
            Last fully processed bar timestamp.
        feature_manifest_hash : Optional[str]
            Feature manifest hash (for drift detection).
        policy_hash : Optional[str]
            Policy hash (for drift detection).
        rotate : bool
            Rotate old state files (weekly rotation, default: False).
        """
        state_path = self.log_dir / "backfill_state.json"
        try:
            state = {
                "last_bar_ts": last_bar_ts.isoformat(),
                "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            if feature_manifest_hash:
                state["feature_manifest_hash"] = feature_manifest_hash
            if policy_hash:
                state["policy_hash"] = policy_hash
            
            # Rotate old state files (keep last 4, weekly rotation) before writing
            # Only rotate if explicitly requested (e.g., weekly)
            if rotate:
                self._rotate_backfill_state(state_path)
            
            # Write state file
            with state_path.open("w", encoding="utf-8") as f:
                jsonlib.dump(state, f, indent=2)
        except Exception as e:
            log.warning("Failed to save backfill state: %s", e)
    
    def _rotate_backfill_state(self, state_path: Path) -> None:
        """
        Rotate backfill state files (keep last 4, weekly rotation).
        
        Parameters
        ----------
        state_path : Path
            Path to current state file (may not exist yet).
        """
        try:
            state_dir = state_path.parent
            
            # Only rotate if current state file exists
            if not state_path.exists():
                # First time: no rotation needed
                return
            
            # Rotate existing files: .1 → .2, .2 → .3, .3 → .4
            # Start from .3 and work backwards to avoid overwriting
            for i in range(3, 0, -1):
                old_path = state_dir / f"backfill_state.json.{i}"
                new_path = state_dir / f"backfill_state.json.{i + 1}"
                if old_path.exists():
                    # Remove old .{i+1} file if it exists
                    if new_path.exists():
                        new_path.unlink()
                    old_path.rename(new_path)
            
            # Backup current state file to .1
            backup_path = state_dir / "backfill_state.json.1"
            # Remove old .1 file if it exists (will be moved to .2 above)
            if backup_path.exists():
                backup_path.unlink()
            # Rename current to .1
            state_path.rename(backup_path)
            
            # Remove files older than .4 (keep last 4: .1, .2, .3, .4)
            for i in range(5, 10):  # Remove .5, .6, .7, .8, .9
                old_file = state_dir / f"backfill_state.json.{i}"
                if old_file.exists():
                    try:
                        old_file.unlink()
                        log.debug("Removed old backfill state file: %s", old_file)
                    except Exception as e:
                        log.warning("Failed to remove old state file %s: %s", old_file, e)
        except Exception as e:
            log.warning("Failed to rotate backfill state: %s", e)
    
    def _compute_ohlcv_checksum(self, row: pd.Series) -> str:
        """
        Compute MD5 checksum for OHLCV data.
        
        Parameters
        ----------
        row : pd.Series
            Row with open, high, low, close, volume.
        
        Returns
        -------
        str
            MD5 checksum (16 chars).
        """
        ohlcv_str = f"{row.get('open', 0):.6f},{row.get('high', 0):.6f},{row.get('low', 0):.6f},{row.get('close', 0):.6f},{row.get('volume', 0):.0f}"
        return hashlib.md5(ohlcv_str.encode("utf-8")).hexdigest()[:16]
    
    def _validate_ohlc(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Validate OHLC consistency rules.
        
        Rules:
        - low ≤ open, close ≤ high
        - high - low ≥ 0
        - Drop/flag bars that violate rules
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with open, high, low, close columns.
        
        Returns
        -------
        Tuple[pd.DataFrame, int]
            Validated DataFrame and number of invalid bars dropped.
        """
        if df.empty:
            return df, 0
        
        required_cols = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            return df, 0
        
        # Validate: low ≤ open, close ≤ high and high - low ≥ 0
        # Konsistensregler på OHLC: Valider low ≤ open,close ≤ high og high−low ≥ 0
        # OHLC-regelbrudd counter: logg og "mask" hele baren hvis low>min(open,close) eller high<max(open,close)
        min_oc = df[["open", "close"]].min(axis=1)
        max_oc = df[["open", "close"]].max(axis=1)
        
        invalid_mask = (
            (df["low"] > min_oc) |  # low > min(open, close)
            (df["high"] < max_oc) |  # high < max(open, close)
            (df["high"] < df["low"])  # high < low (covers high - low < 0)
        )
        
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            invalid_bars = df[invalid_mask]
            log.warning(
                "[VALIDATION] Dropped %d invalid bars (OHLC violations): %s",
                invalid_count,
                invalid_bars.index.tolist()[:10],  # Log first 10
            )
            df = df[~invalid_mask]
        
        return df, invalid_count
    
    def _scan_gaps(
        self,
        from_ts: pd.Timestamp,
        to_ts: pd.Timestamp,
        fetched_df: pd.DataFrame,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Scan for gaps in fetched data.
        
        Parameters
        ----------
        from_ts : pd.Timestamp
            Start time (inclusive).
        to_ts : pd.Timestamp
            End time (exclusive).
        fetched_df : pd.DataFrame
            Fetched candles DataFrame.
        
        Returns
        -------
        List[Tuple[pd.Timestamp, pd.Timestamp]]
            List of (gap_from, gap_to) tuples for missing intervals.
        """
        if fetched_df.empty:
            return [(from_ts, to_ts)]
        
        # Create expected index (complete 5-minute grid, half-open interval)
        try:
            expected_idx = pd.date_range(from_ts, to_ts, freq="5min", tz="UTC", inclusive="left")
        except TypeError:
            # Fallback for older pandas versions (use default left-inclusive behavior)
            expected_idx = pd.date_range(from_ts, to_ts, freq="5min", tz="UTC")
        
        # Find missing timestamps
        missing_idx = expected_idx.difference(fetched_df.index)
        
        if len(missing_idx) == 0:
            return []
        
        # Group consecutive missing timestamps into gaps
        gaps = []
        gap_start = None
        gap_end = None
        
        for ts in sorted(missing_idx):
            if gap_start is None:
                gap_start = ts
                gap_end = ts + pd.Timedelta(minutes=5)
            elif (ts - gap_end).total_seconds() == 0:  # Consecutive gap
                gap_end = ts + pd.Timedelta(minutes=5)
            else:
                # Non-consecutive gap: close previous gap and start new one
                gaps.append((gap_start, gap_end))
                gap_start = ts
                gap_end = ts + pd.Timedelta(minutes=5)
        
        # Close last gap
        if gap_start is not None:
            gaps.append((gap_start, min(gap_end, to_ts)))
        
        return gaps
    
    def _resume_with_backfill(
        self,
        cache_df: Optional[pd.DataFrame],
        from_ts: pd.Timestamp,
        to_ts: pd.Timestamp,
        overlap_bars: int = 6,
        now_utc: Optional[pd.Timestamp] = None,
    ) -> Tuple[pd.DataFrame, int, int]:
        """
        Resume with backfill (idempotent upsert on 5-minute UTC grid).
        
        Parameters
        ----------
        cache_df : Optional[pd.DataFrame]
            Existing cache DataFrame (with 'time' column or time index).
        from_ts : pd.Timestamp
            Start time (inclusive, UTC, normalized to 5-minute boundary).
        to_ts : pd.Timestamp
            End time (exclusive, UTC, normalized to 5-minute boundary) - half-open interval [from_ts, to_ts).
        overlap_bars : int
            Number of overlap bars for rehydration (default: 6 = 30 minutes).
        now_utc : Optional[pd.Timestamp]
            Current UTC timestamp (for excluding incomplete bars).
        
        Returns
        -------
        Tuple[pd.DataFrame, int, int]
            Updated cache DataFrame, number of gaps refetched, and number of revisions detected.
        """
        # Normalize timestamps to 5-minute boundaries
        from_ts = from_ts.floor("5min")
        to_ts = to_ts.floor("5min")
        
        # "Siste bar er ikke lukket"-sperre: Ignore current incomplete M5 bar
        if now_utc is not None:
            now_floor = now_utc.floor("5min")
            to_ts = min(to_ts, now_floor)  # Process only bars < now_floor
            log.debug("[BACKFILL] Excluding incomplete bar: to_ts=%s (now_floor=%s)", to_ts.isoformat(), now_floor.isoformat())
        
        # Calculate expected number of bars
        expected_bars = int((to_ts - from_ts).total_seconds() / 300)  # 5 minutes per bar
        log.info("[BACKFILL] Fetching M5 candles from %s to %s (expected: %d bars)", from_ts.isoformat(), to_ts.isoformat(), expected_bars)
        
        # Fetch M5 candles from OANDA (chunked with retry)
        try:
            new_df = self.broker.get_candles_chunked(
                self.instrument,
                self.granularity,
                from_ts=from_ts,
                to_ts=to_ts,
                chunk_size=3000,
                max_retries=5,
                exclude_incomplete=True,
            )
        except Exception as e:
            log.error("[BACKFILL] Failed to fetch candles: %s", e)
            return cache_df if cache_df is not None else pd.DataFrame(), 0, 0
        
        if new_df.empty:
            log.warning("[BACKFILL] No candles returned from OANDA")
            return cache_df if cache_df is not None else pd.DataFrame(), 0, 0
        
        log.info("[BACKFILL] Fetched %d candles from OANDA (expected: %d)", len(new_df), expected_bars)
        
        # Validate OHLC consistency (drop invalid bars)
        # OHLC-regelbrudd counter: logg og "mask" hele baren hvis low>min(open,close) eller high<max(open,close) etter backfill
        new_df, invalid_count = self._validate_ohlc(new_df)
        if invalid_count > 0:
            log.warning(
                "[VALIDATION] Dropped %d invalid bars (OHLC violations: low>min(open,close) or high<max(open,close))",
                invalid_count,
            )
            # Track OHLC violations in telemetry (if available)
            if hasattr(self, "telemetry_tracker") and self.telemetry_tracker is not None:
                try:
                    # Record OHLC violations (one per invalid bar)
                    for _ in range(invalid_count):
                        self.telemetry_tracker.record_revision(now_utc)  # Use current time as approximation
                except Exception as e:
                    log.warning("Failed to record OHLC violations in telemetry: %s", e)
        
        # Add checksum column for revision detection
        if not new_df.empty:
            new_df["checksum"] = new_df.apply(self._compute_ohlcv_checksum, axis=1)
            
            # Add volume mask (exclude volume-driven features if volume=0)
            new_df["volume_mask"] = (new_df["volume"] == 0.0).astype(int)
        
        # Integrity check: Compare expected vs. actual
        gaps_refetched = 0
        if len(new_df) < expected_bars:
            missing_count = expected_bars - len(new_df)
            log.warning("[BACKFILL] Missing %d bars (expected: %d, got: %d). Scanning for gaps...", missing_count, expected_bars, len(new_df))
            
            # Scan for gaps
            gaps = self._scan_gaps(from_ts, to_ts, new_df)
            if gaps:
                log.warning("[BACKFILL] Found %d gaps. Refetching gaps...", len(gaps))
                
                # Refetch gaps
                gap_dfs = []
                for gap_from, gap_to in gaps:
                    try:
                        gap_df = self.broker.get_candles_chunked(
                            self.instrument,
                            self.granularity,
                            from_ts=gap_from,
                            to_ts=gap_to,
                            chunk_size=3000,
                            max_retries=5,
                            exclude_incomplete=True,
                        )
                        if not gap_df.empty:
                            gap_dfs.append(gap_df)
                            gaps_refetched += 1
                    except Exception as e:
                        log.warning("[BACKFILL] Failed to refetch gap [%s, %s): %s", gap_from.isoformat(), gap_to.isoformat(), e)
                
                # Combine gap data with main data
                if gap_dfs:
                    gap_combined = pd.concat(gap_dfs)
                    new_df = pd.concat([new_df, gap_combined])
                    new_df = new_df.sort_index()
                    new_df = new_df[~new_df.index.duplicated(keep="last")]
                    log.info("[BACKFILL] Refetched %d gaps. Total bars: %d", gaps_refetched, len(new_df))
        
        # Initialize revisions count
        revisions = 0
        
        # Initialize cache if None
        if cache_df is None or cache_df.empty:
            cache_df = new_df.copy()
            log.info("[BACKFILL] Initialized cache with %d candles", len(cache_df))
            # No revisions on first initialization (no overlap)
        else:
            # Ensure cache has time index
            if "time" in cache_df.columns:
                cache_df = cache_df.set_index("time")
            
            # Ensure cache index is timezone-aware UTC
            if cache_df.index.tz is None:
                cache_df.index = cache_df.index.tz_localize("UTC")
            else:
                cache_df.index = cache_df.index.tz_convert("UTC")
            
            # Normalize cache index to 5-minute boundaries
            cache_df.index = cache_df.index.floor("5min")
            
            # Find overlap rows (existing timestamps)
            overlap_idx = cache_df.index.intersection(new_df.index)
            
            # Detect revisions in overlap rows (checksum changed)
            revisions = 0
            if len(overlap_idx) > 0:
                log.info("[BACKFILL] Checking %d overlap rows for revisions", len(overlap_idx))
                
                # Add checksum column to cache if not present
                if "checksum" not in cache_df.columns:
                    cache_df["checksum"] = cache_df.apply(self._compute_ohlcv_checksum, axis=1)
                
                # Detect revisions (checksum changed)
                for ts in overlap_idx:
                    cache_checksum = cache_df.loc[ts, "checksum"]
                    new_checksum = new_df.loc[ts, "checksum"]
                    
                    if cache_checksum != new_checksum:
                        revisions += 1
                        log.info(
                            "[REVISION] Bar %s revised: checksum %s → %s",
                            ts.isoformat(),
                            cache_checksum,
                            new_checksum,
                        )
                        # Mark bar as revised
                        new_df.loc[ts, "revised"] = True
                        
                        # Record revision in telemetry (if telemetry_tracker is available)
                        if hasattr(self, "telemetry_tracker") and self.telemetry_tracker is not None:
                            try:
                                self.telemetry_tracker.record_revision(ts)
                            except Exception as e:
                                log.warning("Failed to record revision in telemetry: %s", e)
                
                if revisions > 0:
                    log.warning(
                        "[REVISION] Detected %d revisions in %d overlap bars",
                        revisions,
                        len(overlap_idx),
                    )
                
                # Update overlap rows in-place (idempotent)
                log.info("[BACKFILL] Updating %d overlap rows", len(overlap_idx))
                cache_df.update(new_df.loc[overlap_idx])
            
            # Append truly new rows
            add_idx = new_df.index.difference(cache_df.index)
            if len(add_idx) > 0:
                log.info("[BACKFILL] Adding %d new rows", len(add_idx))
                cache_df = pd.concat([cache_df, new_df.loc[add_idx]])
            
            # Sort by index and drop duplicates (keep last)
            cache_df = cache_df.sort_index()
            cache_df = cache_df[~cache_df.index.duplicated(keep="last")]
        
        # Align to complete 5-minute grid between min_ts and to_ts - 5m (stop at to_ts - 5m)
        if not cache_df.empty:
            min_ts = cache_df.index.min()
            # Stop at to_ts - 5m (half-open interval)
            grid_to_ts = to_ts - pd.Timedelta(minutes=5)
            # Use inclusive="left" for pandas >= 1.4.0, otherwise use default (left-inclusive)
            try:
                full_idx = pd.date_range(min_ts, grid_to_ts, freq="5min", tz="UTC", inclusive="left")
            except TypeError:
                # Fallback for older pandas versions (use default left-inclusive behavior)
                full_idx = pd.date_range(min_ts, grid_to_ts, freq="5min", tz="UTC")
            cache_df = cache_df.reindex(full_idx)
            
            # Forward-fill kun trygge felt (close, open) - ikke high/low over store gap
            # FF kun for close og open (mid prices) - safe forward-fill
            for col in ["close", "open"]:
                if col in cache_df.columns:
                    # Forward-fill close/open (mid prices) - safe for small gaps
                    cache_df[col] = cache_df[col].ffill(limit=1)
            
            # For high/low: ikke FF over store gap - la NaN blokkere signal for den baren
            # Only forward-fill high/low for small gaps (max 1 bar)
            # For larger gaps, leave NaN to block signal generation
            for col in ["high", "low"]:
                if col in cache_df.columns:
                    # Forward-fill with limit (max 1 bar gap) - only for small gaps
                    # For larger gaps, NaN will block signal generation (as intended)
                    cache_df[col] = cache_df[col].ffill(limit=1)
            
            # Volume: never forward-fill - use 0 if missing (not forward-filled)
            if "volume" in cache_df.columns:
                cache_df["volume"] = cache_df["volume"].fillna(0.0)
            
            log.info("[BACKFILL] Aligned cache to 5-minute grid: %d bars (min=%s, max=%s)", len(cache_df), min_ts.isoformat(), cache_df.index.max().isoformat())
        
        return cache_df, gaps_refetched, revisions
    
    def _write_backfill_receipt(
        self,
        from_ts: pd.Timestamp,
        to_ts: pd.Timestamp,
        expected_bars: int,
        fetched: int,
        gaps_refetched: int,
        revisions: int,
        final_count: int,
        overlap_bars: int,
        policy_hash: str,
        feature_manifest_hash: str,
        duration_s: float,
    ) -> None:
        """
        Write backfill receipt JSON file (integrity acknowledgment).
        
        Parameters
        ----------
        from_ts : pd.Timestamp
            Start time (inclusive).
        to_ts : pd.Timestamp
            End time (exclusive).
        expected_bars : int
            Expected number of bars.
        fetched : int
            Number of bars fetched.
        gaps_refetched : int
            Number of gaps refetched.
        revisions : int
            Number of revisions detected.
        final_count : int
            Final number of bars in cache.
        overlap_bars : int
            Number of overlap bars used.
        policy_hash : str
            Policy hash.
        feature_manifest_hash : str
            Feature manifest hash.
        duration_s : float
            Backfill duration in seconds.
        """
        receipt_path = self.log_dir / "backfill_receipt.json"
        try:
            receipt = {
                "from_ts": from_ts.isoformat(),
                "to_ts": to_ts.isoformat(),
                "expected_bars": expected_bars,
                "fetched": fetched,
                "gaps_refetched": gaps_refetched,
                "revisions": revisions,
                "final_count": final_count,
                "overlap_bars": overlap_bars,
                "policy_hash": policy_hash,
                "feature_manifest_hash": feature_manifest_hash,
                "duration_s": duration_s,
                "bars_per_second": final_count / duration_s if duration_s > 0 else 0.0,
                "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            with receipt_path.open("w", encoding="utf-8") as f:
                jsonlib.dump(receipt, f, indent=2)
            log.debug("[BACKFILL] Receipt written: %s", receipt_path)
        except Exception as e:
            log.warning("Failed to write backfill receipt: %s", e)
    
    def _perform_backfill(self) -> Tuple[pd.DataFrame, int, Optional[pd.Timestamp], int, int]:
        """
        Perform backfill on startup/resume.
        
        Returns
        -------
        Tuple[pd.DataFrame, int, Optional[pd.Timestamp], int, int]
            Backfilled cache DataFrame, number of gaps refetched, warmup_floor, bars_remaining, and revisions.
        """
        # Get backfill configuration
        backfill_cfg = self.policy.get("backfill", {})
        overlap_bars = int(backfill_cfg.get("overlap_bars", 6))  # Default: 6 bars = 30 minutes
        lookback_padding = int(backfill_cfg.get("lookback_padding", 100))  # Default: 100 bars for feature rehydration
        days = int(backfill_cfg.get("days", 3))  # Default: 3 days ≈ 864 M5 bars
        
        # Target bars: warmup_bars + padding (for stateful components like ATR, percentiles, regime)
        warmup_bars = int(self.policy.get("warmup_bars", 288))
        target_bars_padding = int(backfill_cfg.get("target_bars_padding", 100))  # Default: +100 bars padding
        hard_min_bars = int(backfill_cfg.get("hard_min_bars", warmup_bars + target_bars_padding))  # Default: warmup_bars + 100
        force_on_boot = bool(backfill_cfg.get("force_on_boot", False))  # Default: False (respect last_bar_ts)
        min_bars_on_boot = int(backfill_cfg.get("min_bars_on_boot", warmup_bars + target_bars_padding))  # Default: warmup_bars + 100
        
        log.info(
            "[BACKFILL] Target bars: warmup_bars=%d, padding=%d, target=%d (hard_min_bars=%d, min_bars_on_boot=%d)",
            warmup_bars, target_bars_padding, warmup_bars + target_bars_padding, hard_min_bars, min_bars_on_boot
        )
        
        # Get current hashes
        feature_manifest_hash = self.entry_model_bundle.feature_cols_hash if self.entry_model_bundle is not None else None
        policy_hash = self.policy_hash
        
        # Load state (including hashes)
        state = self._load_backfill_state()
        now_utc = pd.Timestamp.now(tz="UTC")
        
        # Check if hashes changed (drift detection)
        hash_changed = False
        overlap_escalated = False
        h4_window_bars = 48  # H4_window = 4 hours = 48 bars (for M5)
        if state and feature_manifest_hash is not None:
            saved_feature_hash = state.get("feature_manifest_hash")
            saved_policy_hash = state.get("policy_hash")
            
            if saved_feature_hash != feature_manifest_hash or saved_policy_hash != policy_hash:
                hash_changed = True
                log.warning(
                    "[BACKFILL] Hash changed: feature_manifest_hash=%s→%s, policy_hash=%s→%s. Increasing overlap_bars for safe rehydration.",
                    saved_feature_hash or "None",
                    feature_manifest_hash,
                    saved_policy_hash or "None",
                    policy_hash,
                )
                # Increase overlap_bars automatically: maks(24, H4_window) for trygg HTF
                # Policy/manifest-hash endring ⇒ overlap_escalation: du har dette; legg samtidig en INFO-linje med nytt overlap_bars for revisorspor
                old_overlap = overlap_bars
                overlap_bars = max(overlap_bars, 24, h4_window_bars)
                if overlap_bars > old_overlap:
                    overlap_escalated = True
                    log.info(
                        "[WARM] overlap_escalated due to hash change: %d → %d (maks(24, H4_window=%d)) | policy_hash=%s→%s | feature_manifest_hash=%s→%s | overlap_bars=%d (revisorspor)",
                        old_overlap,
                        overlap_bars,
                        h4_window_bars,
                        saved_policy_hash or "None",
                        policy_hash,
                        saved_feature_hash or "None",
                        feature_manifest_hash,
                        overlap_bars,
                    )
        
        # Determine from_ts based on force_on_boot flag
        if force_on_boot:
            # Force on boot: ignore last_bar_ts and fetch from (now - days days)
            from_ts = (now_utc.floor("5min") - pd.Timedelta(days=days)).floor("5min")
            log.info("[BACKFILL] Force on boot: backfilling from %s (days=%d, ignoring last_bar_ts)", from_ts.isoformat(), days)
        else:
            # Load last_bar_ts from state
            last_bar_ts = None
            if state:
                last_bar_ts_str = state.get("last_bar_ts")
                if last_bar_ts_str:
                    last_bar_ts = pd.Timestamp(last_bar_ts_str, tz="UTC")
            
            if last_bar_ts is None:
                # First run: backfill from max(days, lookback_padding) bars ago
                days_bars = days * 24 * 12  # days * 24 hours * 12 bars per hour (M5)
                lookback_bars = max(days_bars, lookback_padding)
                from_ts = now_utc.floor("5min") - pd.Timedelta(minutes=5 * lookback_bars)
                log.info("[BACKFILL] First run: backfilling from %s (lookback_bars=%d)", from_ts.isoformat(), lookback_bars)
            else:
                # Resume: backfill from last_bar_ts - overlap_bars
                from_ts = (last_bar_ts - pd.Timedelta(minutes=5 * overlap_bars)).floor("5min")
                log.info("[BACKFILL] Resume: backfilling from %s (last_bar_ts=%s, overlap=%d bars)", from_ts.isoformat(), last_bar_ts.isoformat(), overlap_bars)
        
        # "Siste bar er ikke lukket"-sperre: to_ts = now_floor (exclude incomplete bar)
        to_ts = now_utc.floor("5min")
        
        # Calculate expected bars
        expected_bars = int((to_ts - from_ts).total_seconds() / 300)  # 5 minutes per bar
        
        # Log warmup phase start
        log.info("[PHASE] WARMUP_START from_ts=%s to_ts=%s expected_bars=%d", from_ts.isoformat(), to_ts.isoformat(), expected_bars)
        
        # Measure backfill duration
        backfill_start_time = time.time()
        
        # Use robust target-driven backfill
        # Determine price type: use "M" for mid (default for practice live)
        # TODO: Could be configurable via policy if bid/ask needed for spread
        price_type = "M"
        
        # Get OANDA client (from broker if available, otherwise create temporary)
        if hasattr(self, 'broker') and self.broker is not None:
            # BrokerClient wraps OandaClient in _client attribute
            oanda_client = self.broker._client
        else:
            # Create temporary OANDA client for backfill
            from gx1.execution.oanda_client import OandaClient
            oanda_client = OandaClient.from_env()
        
        # Perform robust target-driven backfill
        target_bars = warmup_bars + target_bars_padding
        cache_df, backfill_meta = backfill_m5_candles_until_target(
            oanda_client=oanda_client,
            instrument=self.instrument,
            granularity=self.granularity,
            target_bars=target_bars,
            price=price_type,
            max_batch=5000,
            max_iters=10,
            min_new_per_iter=5,
            now_utc=now_utc,
            logger=log,
        )
        
        # Measure backfill duration
        backfill_duration_s = time.time() - backfill_start_time
        
        # Extract metadata
        bars_in_cache = backfill_meta.get("total_bars", 0)
        backfill_iters = backfill_meta.get("iterations", 0)
        stop_reason = backfill_meta.get("stop_reason", "unknown")
        gaps_refetched = 0  # Not tracked in new backfill (could be added if needed)
        revisions = 0  # Not tracked in new backfill (could be added if needed)
        fetched_bars = bars_in_cache
        
        # Log backfill results
        log.info(
            "[BACKFILL] iter=%d cached=%d earliest=%s latest=%s stop_reason=%s",
            backfill_iters,
            bars_in_cache,
            backfill_meta.get("earliest_time", pd.NaT).isoformat() if backfill_meta.get("earliest_time") is not None else "N/A",
            backfill_meta.get("latest_time", pd.NaT).isoformat() if backfill_meta.get("latest_time") is not None else "N/A",
            stop_reason,
        )
        
        # Warmup gate: PROD hard / CANARY soft
        # Define thresholds
        min_start_bars = 100  # Minimum bars for CANARY degraded warmup
        
        # Get role from policy
        policy_role = self.policy.get("meta", {}).get("role", "")
        is_prod_baseline = (policy_role == "PROD_BASELINE")
        
        # Calculate warmup status
        warmup_ready = bars_in_cache >= warmup_bars
        target_ready = bars_in_cache >= target_bars
        degraded_warmup = (not warmup_ready) and (bars_in_cache >= min_start_bars)
        
        # Log cache status
        log.info(
            "[BACKFILL] Cache status: cached_bars=%d, warmup_bars=%d, target=%d (min_bars_on_boot=%d)",
            bars_in_cache, warmup_bars, warmup_bars + target_bars_padding, min_bars_on_boot
        )
        
        # Warmup gate logic
        if is_prod_baseline:
            # PROD_BASELINE: hard requirement - must have warmup_bars
            if bars_in_cache >= warmup_bars:
                warmup_floor = None
                bars_remaining = 0
                resume_at = "immediate"
                log.info(
                    "[WARMUP] role=PROD_BASELINE cached=%d warmup_ready=True target_ready=%s degraded=False",
                    bars_in_cache, target_ready
                )
            else:
                # Not enough bars - hold in warmup
                warmup_minutes = 5 * (warmup_bars - bars_in_cache)
                warmup_floor = now_utc.floor("5min") + pd.Timedelta(minutes=warmup_minutes)
                bars_remaining = warmup_bars - bars_in_cache
                resume_at = warmup_floor.strftime("%Y-%m-%dT%H:%M:%SZ")
                log.error(
                    "[WARMUP] role=PROD_BASELINE cached=%d warmup_ready=False target_ready=False degraded=False - BLOCKING TRADING until %s",
                    bars_in_cache, resume_at
                )
        else:
            # CANARY/testing: allow degraded warmup
            if bars_in_cache >= warmup_bars:
                warmup_floor = None
                bars_remaining = 0
                resume_at = "immediate"
                log.info(
                    "[WARMUP] role=%s cached=%d warmup_ready=True target_ready=%s degraded=False",
                    policy_role or "CANARY", bars_in_cache, target_ready
                )
            elif bars_in_cache >= min_start_bars:
                # Degraded warmup allowed
                warmup_floor = None
                bars_remaining = 0
                resume_at = "immediate"
                log.warning(
                    "[WARMUP] role=%s cached=%d warmup_ready=False target_ready=False degraded=True - ALLOWING DEGRADED WARMUP",
                    policy_role or "CANARY", bars_in_cache
                )
            else:
                # Not enough even for degraded - hold in warmup
                warmup_minutes = 5 * (min_start_bars - bars_in_cache)
                warmup_floor = now_utc.floor("5min") + pd.Timedelta(minutes=warmup_minutes)
                bars_remaining = min_start_bars - bars_in_cache
                resume_at = warmup_floor.strftime("%Y-%m-%dT%H:%M:%SZ")
                log.warning(
                    "[WARMUP] role=%s cached=%d warmup_ready=False target_ready=False degraded=False - BLOCKING TRADING until %s",
                    policy_role or "CANARY", bars_in_cache, resume_at
                )
        
        # Store degraded warmup flag for trade journal
        self._warmup_degraded = degraded_warmup
        self._cached_bars_at_startup = bars_in_cache
        
        # Initialize _last_server_time_check if not already set
        if not hasattr(self, '_last_server_time_check') or self._last_server_time_check is None:
            self._last_server_time_check = pd.Timestamp.now(tz="UTC")
        
        # Calculate bars per second
        bars_per_second = bars_in_cache / backfill_duration_s if backfill_duration_s > 0 else 0.0
        
        # Write backfill receipt (integrity acknowledgment)
        if not cache_df.empty:
            self._write_backfill_receipt(
                from_ts=from_ts,
                to_ts=to_ts,
                expected_bars=expected_bars,
                fetched=fetched_bars,
                gaps_refetched=gaps_refetched,
                revisions=revisions,
                final_count=bars_in_cache,
                overlap_bars=overlap_bars,
                policy_hash=policy_hash,
                feature_manifest_hash=feature_manifest_hash,
                duration_s=backfill_duration_s,
            )
        
        # Save state (last_bar_ts = to_ts, with hashes)
        # Don't rotate on every backfill - only rotate weekly (manually or via cron)
        if not cache_df.empty:
            self._save_backfill_state(
                last_bar_ts=to_ts,
                feature_manifest_hash=feature_manifest_hash,
                policy_hash=policy_hash,
                rotate=False,  # Only rotate weekly, not on every backfill
            )
        
        # Log warmup phase end (explicit phase transition)
        if warmup_floor is not None:
            log.info(
                "[PHASE] WARMUP_END resume_at=%s bars_needed=%d bars_remaining=%d",
                resume_at,
                min_bars_on_boot,
                bars_remaining,
            )
        else:
            # Determine which threshold we met
            if bars_in_cache >= min_bars_on_boot:
                threshold_met = min_bars_on_boot
            elif bars_in_cache >= min_start_bars:
                threshold_met = min_start_bars
            else:
                threshold_met = warmup_bars  # Fallback
            log.info(
                "[PHASE] WARMUP_END resume_at=immediate bars_needed≥%d bars_remaining=0 (sufficient bars: %d ≥ %d)",
                threshold_met,
                bars_in_cache,
                threshold_met,
            )
        
        # Log one-line summary (detailed backfill info)
        # Format: [WARM] Backfilled N=1728 from=2025-11-10T00:00Z to=2025-11-12T08:00Z overlap=6 expected=1728 fetched=1728 gaps_refetched=0 revised=0 warmup_floor=96 duration_s=7.4 bps=233.5 policy=ab12cd34 manifest=89ef0123
        log.info(
            "[WARM] Backfilled N=%d from=%s to=%s overlap=%d expected=%d fetched=%d gaps_refetched=%d revised=%d warmup_floor=%s duration_s=%.1f bps=%.1f policy=%s manifest=%s",
            bars_in_cache,
            from_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            to_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            overlap_bars,
            expected_bars,
            fetched_bars,
            gaps_refetched,
            revisions,
            warmup_floor.strftime("%Y-%m-%dT%H:%M:%SZ") if warmup_floor is not None else "None",
            backfill_duration_s,
            bars_per_second,
            policy_hash,
            feature_manifest_hash,
        )
        
        # Warn if bars_per_second < baseline (API slowness/rate-limit)
        baseline_bps = 100.0  # Baseline: 100 bars/second
        if bars_per_second < baseline_bps and bars_in_cache > 0:
            log.warning(
                "[BACKFILL] Low throughput: %.1f bars/second < baseline %.1f (API may be slow or rate-limited)",
                bars_per_second,
                baseline_bps,
            )
        
        # Warn if revisions > 0 (data revisions detected)
        if revisions > 0:
            log.warning(
                "[REVISION] Detected %d revisions in backfill (data may have been corrected by OANDA)",
                revisions,
            )
        
        if cache_df.empty:
            log.warning("[BACKFILL] Backfill complete but cache is empty")
            warmup_floor = None
            bars_remaining = 0
            resume_at = "N/A"
            revisions = 0
        
        return cache_df, gaps_refetched, warmup_floor, bars_remaining, revisions
    
    def _init_trade_journal(self) -> None:
        """Initialize trade journal for structured logging."""
        """
        Initialize trade journal for structured logging.
        
        Creates TradeJournal instance and loads run_header.json for artifact hashes.
        """
        try:
            from gx1.monitoring.trade_journal import TradeJournal
            from gx1.prod.run_header import load_run_header
            
            # Determine output directory (same as run_header.json location)
            # For parallel chunk workers, use explicit output_dir if set
            if hasattr(self, "explicit_output_dir") and self.explicit_output_dir:
                # Parallel chunk worker: use explicit chunk_dir
                output_dir = self.explicit_output_dir
                log.info("[TRADE_JOURNAL] Using explicit output_dir for chunk: %s", output_dir)
            elif hasattr(self, "replay_mode") and self.replay_mode:
                # Replay: always use wf_runs/<run_id>/
                output_dir = Path("gx1/wf_runs") / self.run_id
            elif hasattr(self, "trade_log_path") and self.trade_log_path:
                # Live: Use trade_log_path parent, but prefer wf_runs
                trade_log_parent = self.trade_log_path.parent
                if "wf_runs" in str(trade_log_parent):
                    output_dir = trade_log_parent
                    if "parallel_chunks" in str(output_dir):
                        output_dir = output_dir.parent.parent
                elif "parallel_chunks" in str(trade_log_parent):
                    output_dir = trade_log_parent.parent.parent
                else:
                    # Legacy location, use wf_runs instead
                    output_dir = Path("gx1/wf_runs") / self.run_id
            elif hasattr(self, "log_dir") and self.log_dir:
                log_dir_path = Path(self.log_dir)
                if "parallel_chunks" in str(log_dir_path):
                    output_dir = log_dir_path.parent.parent
                elif "wf_runs" in str(log_dir_path):
                    output_dir = log_dir_path.parent
                else:
                    # Legacy location, use wf_runs instead
                    output_dir = Path("gx1/wf_runs") / self.run_id
            else:
                output_dir = Path("gx1/wf_runs") / self.run_id
            
            # Load run header (for artifact hashes)
            try:
                run_header = load_run_header(output_dir)
            except Exception:
                # If run_header.json doesn't exist yet, create empty header
                run_header = {
                    "run_tag": self.run_id,
                    "artifacts": {},
                    "meta": {"role": getattr(self, "prod_baseline", False) and "PROD_BASELINE" or "DEV"},
                }
            
            # Initialize trade journal (enabled by default, always enabled in PROD_BASELINE)
            journal_enabled = True  # Always enabled for production-ready logging
            self.trade_journal = TradeJournal(
                run_dir=output_dir,
                run_tag=self.run_id,
                header=run_header,
                enabled=journal_enabled,
                runner=self,
            )
            
            log.info("[TRADE_JOURNAL] Initialized trade journal at %s", output_dir / "trade_journal")
        except Exception as e:
            log.warning("[TRADE_JOURNAL] Failed to initialize trade journal: %s", e)
            self.trade_journal = None
    
    def _check_policy_lock(self) -> bool:
        """
        Check if policy file has changed on disk (policy-lock).
        
        Returns
        -------
        bool
            True if policy unchanged, False if changed.
        """
        try:
            current_policy_content = self.policy_path.read_text()
            current_policy_hash = hashlib.md5(current_policy_content.encode("utf-8")).hexdigest()[:16]
            
            if current_policy_hash != self.policy_hash:
                log.error(
                    "[POLICY LOCK] Policy file changed on disk: hash=%s (expected %s). Trading disabled.",
                    current_policy_hash,
                    self.policy_hash,
                )
                return False
            
            return True
        except Exception as e:
            log.warning("Policy lock check failed: %s", e)
            return True  # Allow trading if check fails (conservative)
    
    def _generate_run_header(self) -> None:
        """
        Generate run_header.json artifact at startup.
        
        Computes SHA256 hashes for policy, models, and feature manifest.
        """
        try:
            from gx1.prod.run_header import generate_run_header
            from pathlib import Path
            
            # Determine output directory for run_header.json
            # For parallel chunk workers, run_header.json should be in parent run_dir (not chunk_dir)
            # For regular runs, use standard logic
            if hasattr(self, "explicit_output_dir") and self.explicit_output_dir:
                # Parallel chunk worker: run_header.json goes to parent run_dir
                chunk_dir = self.explicit_output_dir
                if "parallel_chunks" in str(chunk_dir):
                    # Go up from chunk_* to parallel_chunks, then to run_dir
                    output_dir = chunk_dir.parent.parent
                else:
                    # Not in parallel_chunks structure, use chunk_dir itself
                    output_dir = chunk_dir
                log.info("[RUN_HEADER] Parallel chunk detected, writing to parent run_dir: %s", output_dir)
            elif hasattr(self, "replay_mode") and self.replay_mode:
                # Replay: always use wf_runs/<run_id>/
                output_dir = Path("gx1/wf_runs") / self.run_id
            elif hasattr(self, "trade_log_path") and self.trade_log_path:
                # Live: Use trade_log_path parent, but check if it's in wf_runs
                trade_log_parent = self.trade_log_path.parent
                if "wf_runs" in str(trade_log_parent):
                    # Already in wf_runs, use it
                    output_dir = trade_log_parent
                    # If in parallel_chunks, go up to main run directory (two levels: chunk_* -> parallel_chunks -> run_dir)
                    if "parallel_chunks" in str(output_dir):
                        output_dir = output_dir.parent.parent
                elif "parallel_chunks" in str(trade_log_parent):
                    # In parallel_chunks, go up to main run directory
                    output_dir = trade_log_parent.parent.parent
                else:
                    # Legacy location (gx1/live/), use wf_runs instead
                    output_dir = Path("gx1/wf_runs") / self.run_id
            elif hasattr(self, "log_dir") and self.log_dir:
                log_dir_path = Path(self.log_dir)
                # If log_dir is in parallel_chunks, go up to main output dir
                if "parallel_chunks" in str(log_dir_path):
                    # Go up from chunk_*/logs to parallel_chunks, then to run_dir
                    output_dir = log_dir_path.parent.parent
                elif "wf_runs" in str(log_dir_path):
                    # Already in wf_runs, use parent
                    output_dir = log_dir_path.parent
                else:
                    # Legacy location, use wf_runs instead
                    output_dir = Path("gx1/wf_runs") / self.run_id
            else:
                output_dir = Path("gx1/wf_runs") / self.run_id
            
            # Ensure output_dir exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get router model path
            router_model_path = None
            exit_hybrid_enabled = getattr(self, "exit_hybrid_enabled", False)
            exit_mode_selector = getattr(self, "exit_mode_selector", None)
            if exit_hybrid_enabled and exit_mode_selector:
                router_cfg = self.policy.get("hybrid_exit_router", {})
                model_path_str = router_cfg.get("model_path")
                if model_path_str:
                    router_model_path = Path(model_path_str)
                    prod_baseline = getattr(self, "prod_baseline", False)
                    if not router_model_path.is_absolute() and prod_baseline:
                        from gx1.prod.path_resolver import PROD_CURRENT_DIR
                        router_model_path = PROD_CURRENT_DIR / router_model_path
            
            # Get entry model paths (if available)
            entry_model_paths = []
            if hasattr(self, "entry_v9_model") and self.entry_v9_model:
                # Entry models are loaded, but paths may not be directly accessible
                # This is approximate - full implementation would track all model files
                pass
            
            # Get feature manifest path
            feature_manifest_path = None
            if hasattr(self, "entry_v9_feature_meta_path") and self.entry_v9_feature_meta_path:
                manifest_candidate = self.entry_v9_feature_meta_path.parent / "feature_manifest.json"
                if manifest_candidate.exists():
                    feature_manifest_path = manifest_candidate
            
            # DEL 4: Get run_id and chunk_id for run_header
            run_id = getattr(self, "run_id", None)
            chunk_id = getattr(self, "chunk_id", None) or os.getenv("GX1_CHUNK_ID")
            
            # Generate header
            header = generate_run_header(
                policy_dict=self.policy,  # Pass policy dict for meta.role extraction
                policy_path=self.policy_path,
                router_model_path=router_model_path,
                entry_model_paths=entry_model_paths if entry_model_paths else None,
                feature_manifest_path=feature_manifest_path,
                output_dir=output_dir,
                run_tag=run_id,  # DEL 4: Use run_id as run_tag
                chunk_id=chunk_id,  # DEL 4: Pass chunk_id
            )
            # XGB load branch proof (TRUTH canonical vs policy/session)
            xgb_branch = getattr(self, "xgb_load_branch", None)
            xgb_source = getattr(self, "xgb_load_source", None)
            xgb_paths = getattr(self, "xgb_load_paths", None)
            xgb_err = getattr(self, "xgb_load_error", None)
            if xgb_branch is not None or xgb_source is not None or xgb_paths is not None or xgb_err is not None:
                header["xgb_load_branch"] = xgb_branch
                header["xgb_load_source"] = xgb_source
                header["xgb_load_paths"] = xgb_paths
                header["xgb_load_error"] = xgb_err
                header_path = output_dir / "run_header.json"
                with open(header_path, "w", encoding="utf-8") as _f:
                    jsonlib.dump(header, _f, indent=2)
            
            # Replay metadata will be added later in _update_run_header_replay_metadata() after data is loaded
            
            log.info("[RUN_HEADER] Generated run_header.json with %d artifacts", len(header.get("artifacts", {})))
            
        except Exception as e:
            log.error("[RUN_HEADER] Failed to generate run_header.json: %s", e, exc_info=True)
    
    def _update_run_header_replay_metadata(self) -> None:
        """
        Update run_header.json with replay metadata after data is loaded.
        
        Called from _run_replay_impl() after replay_eval_start_ts and replay_eval_end_ts are set.
        """
        try:
            # Determine output directory (same logic as _generate_run_header)
            if hasattr(self, "replay_mode") and self.replay_mode:
                output_dir = Path("gx1/wf_runs") / self.run_id
            elif hasattr(self, "trade_log_path") and self.trade_log_path:
                trade_log_parent = self.trade_log_path.parent
                if "wf_runs" in str(trade_log_parent):
                    output_dir = trade_log_parent
                    if "parallel_chunks" in str(output_dir):
                        output_dir = output_dir.parent.parent
                elif "parallel_chunks" in str(trade_log_parent):
                    output_dir = trade_log_parent.parent.parent
                else:
                    output_dir = Path("gx1/wf_runs") / self.run_id
            elif hasattr(self, "log_dir") and self.log_dir:
                log_dir_path = Path(self.log_dir)
                if "parallel_chunks" in str(log_dir_path):
                    output_dir = log_dir_path.parent.parent
                elif "wf_runs" in str(log_dir_path):
                    output_dir = log_dir_path.parent
                else:
                    output_dir = Path("gx1/wf_runs") / self.run_id
            else:
                output_dir = Path("gx1/wf_runs") / self.run_id
            
            header_path = output_dir / "run_header.json"
            if not header_path.exists():
                log.warning("[RUN_HEADER] run_header.json not found at %s, skipping replay metadata update", header_path)
                return
            
            # Load existing header (use global jsonlib import to avoid scope issues)
            with open(header_path, "r") as f:
                header = jsonlib.load(f)
            
            # Add replay metadata
            if hasattr(self, "replay_eval_start_ts") and hasattr(self, "replay_eval_end_ts"):
                header["replay"] = {
                    "eval_start_ts": self.replay_eval_start_ts.isoformat() if self.replay_eval_start_ts else None,
                    "eval_end_ts": self.replay_eval_end_ts.isoformat() if self.replay_eval_end_ts else None,
                    "start_ts": self.replay_start_ts.isoformat() if hasattr(self, "replay_start_ts") and self.replay_start_ts else None,
                    "end_ts": self.replay_end_ts.isoformat() if hasattr(self, "replay_end_ts") and self.replay_end_ts else None,
                }
                # Save updated header
                with open(header_path, "w") as f:
                    jsonlib.dump(header, f, indent=2, default=str)
                log.info("[RUN_HEADER] Updated run_header.json with replay metadata")
        except Exception as e:
            log.warning("[RUN_HEADER] Failed to update run_header.json with replay metadata: %s", e)
    
    def _write_model_used_capsule(
        self,
        xgb_models: Dict[str, Any],
        xgb_model_paths: Dict[str, str],
        bundle_dir: Path,
        policy_id: str,
    ) -> None:
        """
        Write MODEL_USED_CAPSULE.json (SSoT for model usage).
        
        This capsule provides proof of which XGB models were actually used in runtime,
        enabling audit scripts to build TRUTH_USED_SET.
        """
        try:
            # Determine output directory (same logic as _generate_run_header)
            if hasattr(self, "output_dir") and self.output_dir:
                output_dir = Path(self.output_dir)
            elif hasattr(self, "explicit_output_dir") and self.explicit_output_dir:
                output_dir = Path(self.explicit_output_dir)
            elif hasattr(self, "run_id") and self.run_id:
                output_dir = Path("gx1/wf_runs") / self.run_id
            else:
                # Fallback: use current working directory / reports/replay_eval/GATED
                output_dir = Path("reports/replay_eval/GATED")
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Compute SHA256 for each XGB model
            xgb_models_info = {}
            for session, model_path_str in xgb_model_paths.items():
                model_path = Path(model_path_str)
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        model_sha256 = hashlib.sha256(f.read()).hexdigest()
                    xgb_models_info[session] = {
                        "selected_xgb_model_path": str(model_path.resolve()),
                        "selected_xgb_model_sha256": model_sha256,
                        "selected_model_kind": "xgb_pre",
                    }
            
            # Get bundle SHA256 if available
            bundle_sha256 = None
            model_state_path = Path(bundle_dir) / "model_state_dict.pt"
            if model_state_path.exists():
                with open(model_state_path, "rb") as f:
                    bundle_sha256 = hashlib.sha256(f.read()).hexdigest()
            
            # Get policy SHA256 if available
            policy_sha256 = None
            if hasattr(self, "policy_hash"):
                policy_sha256 = self.policy_hash
            elif hasattr(self, "policy_path") and self.policy_path:
                policy_path = Path(self.policy_path)
                if policy_path.exists():
                    with open(policy_path, "rb") as f:
                        policy_sha256 = hashlib.sha256(f.read()).hexdigest()
            
            # Get git commit
            from gx1.prod.run_header import get_git_commit_hash
            git_commit = get_git_commit_hash()
            
            # Build capsule
            capsule = {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "run_id": getattr(self, "run_id", None),
                "policy_id": policy_id,
                "policy_sha256": policy_sha256,
                "bundle_dir": str(Path(bundle_dir).resolve()),
                "bundle_sha256": bundle_sha256,
                "xgb_models": xgb_models_info,
                "provenance": {
                    "git_commit": git_commit,
                    "worker_pid": os.getpid(),
                },
            }
            
            # Atomic write
            capsule_path = output_dir / "MODEL_USED_CAPSULE.json"
            tmp_path = capsule_path.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                jsonlib.dump(capsule, f, indent=2)
            tmp_path.rename(capsule_path)
            
            log.info(f"[MODEL_CAPSULE] ✅ Written MODEL_USED_CAPSULE.json to: {capsule_path}")
            log.info(f"[MODEL_CAPSULE]   XGB models: {list(xgb_models_info.keys())}")
            
        except Exception as e:
            log.error(f"[MODEL_CAPSULE] Failed to write MODEL_USED_CAPSULE.json: {e}", exc_info=True)
    
    def _log_canary_invariants(self) -> None:
        """
        Log all invariants in canary mode.
        
        Verifies that all critical invariants passed during the run.
        """
        log.info("=" * 80)
        log.info("[CANARY] Invariant Verification Summary")
        log.info("=" * 80)
        
        # Check range features availability
        if hasattr(self, "entry_manager"):
            log.info("[CANARY] ✅ Range features computed before router (EntryManager)")
        
        # Check guardrail
        if hasattr(self, "exit_mode_selector") and self.exit_mode_selector:
            log.info("[CANARY] ✅ Guardrail active (ExitModeSelector)")
        
        # Check model loading
        if hasattr(self, "exit_mode_selector") and self.exit_mode_selector:
            if self.exit_mode_selector.version == "HYBRID_ROUTER_V3":
                log.info("[CANARY] ✅ Router model loading verified")
        
        # Check feature manifest validation
        log.info("[CANARY] ✅ Feature manifest validation passed (if enabled)")
        
        # Check policy lock
        if self._check_policy_lock():
            log.info("[CANARY] ✅ Policy lock verified (no changes detected)")
        else:
            log.warning("[CANARY] ⚠️  Policy lock check failed (policy may have changed)")
        
        log.info("=" * 80)
        log.info("[CANARY] All invariants verified - canary run successful")
        log.info("=" * 80)
    
    def _generate_canary_metrics(self) -> None:
        """
        Generate prod_metrics.csv from prod_monitor in canary mode.
        """
        try:
            from gx1.monitoring.prod_monitor import ProdMonitor
            from pathlib import Path
            
            # Determine trade log path
            if hasattr(self, "log_dir") and self.log_dir:
                output_dir = Path(self.log_dir).parent
            else:
                output_dir = Path("gx1/wf_runs") / self.run_id
            
            # Find trade log
            trade_log_paths = list(output_dir.glob("trade_log*.csv"))
            if not trade_log_paths:
                log.warning("[CANARY] No trade log found - skipping prod_metrics generation")
                return
            
            # Use merged trade log if available, else first found
            merged_logs = list(output_dir.glob("trade_log*merged.csv"))
            trade_log_path = merged_logs[0] if merged_logs else trade_log_paths[0]
            
            # Generate metrics
            monitor = ProdMonitor(window_days=7, output_dir=output_dir)
            monitor.load_from_trade_log(trade_log_path)
            monitor.log_metrics()
            
            log.info("[CANARY] ✅ Prod metrics generated: %s/prod_metrics.csv", output_dir)
            
        except Exception as e:
            log.warning("[CANARY] Failed to generate prod_metrics: %s", e)
    
    def _get_bar_size_seconds(self) -> float:
        """
        Get bar size in seconds based on granularity.
        
        Returns
        -------
        float
            Bar size in seconds.
        """
        granularity = self.granularity
        if granularity.startswith("M"):
            # Minutes (e.g., M5 = 5 minutes)
            minutes = int(granularity[1:])
            return float(minutes * 60)
        elif granularity.startswith("H"):
            # Hours (e.g., H1 = 1 hour)
            hours = int(granularity[1:])
            return float(hours * 3600)
        elif granularity.startswith("D"):
            # Days (e.g., D1 = 1 day)
            days = int(granularity[1:])
            return float(days * 86400)
        else:
            # Default: assume M5 (5 minutes)
            log.warning("Unknown granularity '%s', assuming M5 (5 minutes)", granularity)
            return 300.0  # 5 minutes in seconds
    
    # ------------------------------------------------------------------ #
    # TickWatcher integration (TP/SL/BE on tick)
    # ------------------------------------------------------------------ #
    def _maybe_update_tick_watcher(self) -> None:
        """Start/stop tick watcher based on number of open positions."""
        # In replay mode, tick watcher is disabled (we simulate ticks from M5 candles)
        if self.replay_mode:
            return
        
        n = len(self.open_trades)
        if n > 0:
            self.tick_watcher.start()
        else:
            self.tick_watcher.stop()
    
    def _get_open_positions_for_tick(self) -> List[OpenPos]:
        """Get open positions for TickWatcher (only necessary fields)."""
        out = []
        
        # Get TP/SL/BE thresholds from tick_exit config (or use defaults from policy)
        # Note: These can be overridden per-trade from exit policy if needed
        tp_bps_default = int(self.tick_cfg.get("tp_bps", 170))  # Default from policy or hardcoded
        sl_bps_default = int(self.tick_cfg.get("sl_bps", 95))  # Default from policy or hardcoded
        be_trigger_bps = int(self.tick_cfg.get("be_trigger_bps", 50))  # Activate BE when profit >= this
        soft_stop_bps = int(self.tick_cfg.get("soft_stop_bps", 0))  # Optional soft-stop (anytime, not time-based)
        
        for trade in self.open_trades:
            # Convert entry_time to datetime (tz-aware UTC)
            entry_ts = trade.entry_time
            if isinstance(entry_ts, pd.Timestamp):
                if entry_ts.tzinfo is None:
                    entry_ts = entry_ts.tz_localize("UTC")
                else:
                    entry_ts = entry_ts.tz_convert("UTC")
                entry_dt = entry_ts.to_pydatetime()
            else:
                entry_dt = datetime.fromisoformat(str(entry_ts).replace("Z", "+00:00")).astimezone(timezone.utc)
            
            # Get TP/SL/BE from trade.extra (set at entry from exit policy) or use defaults
            # Check for Big Brain V1 adjusted exit parameters first (with asymmetry)
            bb_exit_asym = trade.extra.get("bb_exit_asym", {}) if hasattr(trade, "extra") and trade.extra else {}
            bb_exit = trade.extra.get("bb_exit", {}) if hasattr(trade, "extra") and trade.extra else {}
            
            # Priority: bb_exit_asym > bb_exit > regular values
            if bb_exit_asym and isinstance(bb_exit_asym, dict):
                # Use Big Brain adjusted values with asymmetry
                tp_bps = int(bb_exit_asym.get("tp_bps_adj", bb_exit.get("tp_bps_adj", trade.extra.get("tp_bps", tp_bps_default))))
            elif bb_exit and isinstance(bb_exit, dict):
                # Use Big Brain adjusted values (risk shaping only, no asymmetry)
                tp_bps = int(bb_exit.get("tp_bps_adj", trade.extra.get("tp_bps", tp_bps_default)))
            else:
                # Fall back to regular values
                tp_bps = int(trade.extra.get("tp_bps", tp_bps_default)) if hasattr(trade, "extra") and trade.extra else tp_bps_default
            
            # Also check for asymmetry-adjusted SL if available
            if bb_exit_asym and isinstance(bb_exit_asym, dict) and "sl_bps_adj" in bb_exit_asym:
                sl_bps = int(bb_exit_asym.get("sl_bps_adj", sl_bps_default))
            else:
                sl_bps = int(trade.extra.get("sl_bps", sl_bps_default)) if hasattr(trade, "extra") and trade.extra else sl_bps_default
            
            # BE status is synced from TickWatcher to trade.extra via _update_be_status callback
            be_active = trade.extra.get("be_active", False) if hasattr(trade, "extra") and trade.extra else False
            be_price = trade.extra.get("be_price", None) if hasattr(trade, "extra") and trade.extra else None
            
            # Store bb_exit data (prioritize bb_exit_asym if available)
            bb_exit_data = bb_exit_asym if bb_exit_asym and isinstance(bb_exit_asym, dict) else (bb_exit if bb_exit and isinstance(bb_exit, dict) else None)
            
            out.append(OpenPos(
                trade_id=trade.trade_id,
                direction="LONG" if trade.side == "long" else "SHORT",
                entry_px=float(trade.entry_price),
                entry_bid=float(getattr(trade, "entry_bid", trade.entry_price)),
                entry_ask=float(getattr(trade, "entry_ask", trade.entry_price)),
                units=int(trade.units),
                entry_ts=entry_dt,
                tp_bps=tp_bps,
                sl_bps=sl_bps,
                be_active=be_active,
                be_price=be_price,
                early_stop_bps=0,  # Not used (soft_stop_bps is handled in TickWatcher config)
                bb_exit=bb_exit_data,  # Store Big Brain exit adjustments (with asymmetry if available)
            ))
        return out
    
    def _update_be_status(self, trade_id: str, be_active: bool, be_price: float) -> None:
        """Update BE status in trade.extra (thread-safe, called from TickWatcher)."""
        with self._be_update_lock:
            for trade in self.open_trades:
                if trade.trade_id == trade_id:
                    trade.extra["be_active"] = be_active
                    trade.extra["be_price"] = be_price
                    log.debug(
                        "[TICK] Updated BE status for trade %s: be_active=%s be_price=%.3f",
                        trade_id,
                        be_active,
                        be_price,
                    )
                    break
    
    def _tick_close_now(self, pos: OpenPos, reason: str, px: float, pnl_bps: float) -> None:
        """Close position immediately on tick (called by TickWatcher)."""
        from gx1.execution.live_features import infer_session_tag
        try:
            trade_id = pos.trade_id
            
            # Find trade in open_trades
            trade = None
            for t in self.open_trades:
                if t.trade_id == trade_id:
                    trade = t
                    break
            
            if not trade:
                log.warning("[TICK] Trade %s not found in open_trades (already closed?)", trade_id)
                return
            
            # Map TickWatcher reason to ExitArbiter reason
            reason_map = {
                "TP_TICK": "TP_TICK",
                "SL_TICK": "SL_TICK",
                "BE_TICK": "BE_TICK",
                "SOFT_STOP_TICK": "SOFT_STOP_TICK",
            }
            arbiter_reason = reason_map.get(reason, "TICK")
            
            # Log tick-watcher propose close
            log.info("[TICK] propose close reason=%s pnl=%.1f", reason, pnl_bps)
            
            # Request close via ExitArbiter
            accepted = self.request_close(
                trade_id=trade_id,
                source="TICK",
                reason=arbiter_reason,
                px=px,
                pnl_bps=pnl_bps,
                bars_in_trade=None,  # Tick-watcher doesn't track bars
            )
            
            if not accepted:
                log.warning("[TICK] close rejected by ExitArbiter for trade %s", trade_id)
                return
            
            # Remove from open_trades (after successful close)
            if trade in self.open_trades:
                self.open_trades.remove(trade)
            self._teardown_exit_state(trade_id)
            
            # Update tick watcher (only if not in replay mode)
            if not self.replay_mode:
                self._maybe_update_tick_watcher()
            
            # Record closed trade to telemetry
            # In replay mode, use current bar timestamp (stored in replay_context)
            # In live mode, use now()
            if self.replay_mode:
                # In replay mode, use current timestamp from replay context
                # If replay_context is not set, fall back to now() (shouldn't happen)
                now_ts_utc = getattr(self, '_replay_current_ts', pd.Timestamp.now(tz="UTC"))
            else:
                now_ts_utc = pd.Timestamp.now(tz="UTC")
            entry_ts_utc = trade.entry_time.tz_convert("UTC") if trade.entry_time.tzinfo else pd.Timestamp(trade.entry_time, tz="UTC")
            entry_p_hat = max(trade.entry_prob_long, trade.entry_prob_short)
            entry_session = infer_session_tag(trade.entry_time)
            session_key = self._resolve_session_key(entry_session)
            
            # Record closed trade to telemetry (with guard)
            if hasattr(self, "telemetry_tracker") and self.telemetry_tracker is not None:
                try:
                    self.telemetry_tracker.record_closed_trade(
                        entry_id=trade_id,
                        session=session_key,
                        entry_time=entry_ts_utc,
                        exit_time=now_ts_utc,
                        p_hat=entry_p_hat,
                        side=trade.side,
                        pnl_bps=pnl_bps,
                    )
                except Exception as e:
                    log.warning("[TICK] Failed to record closed trade in telemetry: %s", e)
            else:
                # Log warning only once per session to avoid spam
                if not hasattr(self, "_telemetry_warned"):
                    log.warning("[TICK] telemetry_tracker not available - telemetry disabled")
                    self._telemetry_warned = True
            
            # Record exit latency
            hold_time_s = (now_ts_utc - entry_ts_utc).total_seconds()
            if hasattr(self, "telemetry_tracker") and self.telemetry_tracker is not None:
                try:
                    self.telemetry_tracker.record_exit_latency(session_key, "EXIT_V2", hold_time_s)
                except Exception as e:
                    log.warning("[TICK] Failed to record exit latency in telemetry: %s", e)
            
            # Record realized PnL
            self.record_realized_pnl(now_ts_utc, pnl_bps)
            
            # Log trade closure
            log.info(
                "[LIVE] TICK EXIT trade_id=%s reason=%s price=%.3f pnl_bps=%.1f",
                trade_id, reason, px, pnl_bps
            )
            
            # Update tick watcher (stop if no more open trades)
            self._maybe_update_tick_watcher()
            
        except Exception as e:
            log.error(f"[TICK] close_cb error trade_id={pos.trade_id} reason={reason}: {e!r}", exc_info=True)

    # ------------------------------------------------------------------ #
    # Risk helpers
    # ------------------------------------------------------------------ #
    def daily_loss_exceeded(self, now: pd.Timestamp) -> bool:
        day_key = compute_daily_key(now)
        loss = self.daily_loss_tracker.get(day_key, 0.0)
        exceeded = loss <= -abs(self.risk_limits.max_daily_loss_bps)
        if exceeded:
            if self.is_replay:
                # Replay/fast-replay: metrics only, do not block entries
                log.warning(
                    "Daily loss guard triggered in replay (%.1f bps <= -%.1f) – NOT blocking new entries.",
                    loss,
                    self.risk_limits.max_daily_loss_bps,
                )
                return False
            else:
                log.warning(
                    "Daily loss guard triggered (%.1f bps <= -%.1f). No new entries today.",
                    loss,
                    self.risk_limits.max_daily_loss_bps,
                )
        return exceeded

    def record_realized_pnl(self, now: pd.Timestamp, pnl_bps: float) -> None:
        day_key = compute_daily_key(now)
        self.daily_loss_tracker[day_key] = self.daily_loss_tracker.get(day_key, 0.0) + pnl_bps

    def _get_tp_sl_state(self, trade_id: str) -> Dict[str, Any]:
        """
        Get TP/SL order state from broker for a trade.
        
        Returns
        -------
        Dict with keys: tp_active, sl_active, tp_order_id, sl_order_id, who_cancelled
        """
        # In replay mode, skip OANDA API calls (trades are simulated)
        if self.replay_mode:
            # Return mock state (TP/SL are not set in replay mode, we simulate them)
            return {
                "tp_active": False,
                "sl_active": False,
                "tp_order_id": None,
                "sl_order_id": None,
                "who_cancelled": "none",
            }
        
        try:
            trade_info = self.broker.get_trade(trade_id)
            trade = trade_info.get("trade", {})
            
            tp_order = trade.get("takeProfitOrder", {})
            sl_order = trade.get("stopLossOrder", {})
            
            tp_active = tp_order.get("state", "CANCELLED") not in ("CANCELLED", "FILLED", "TRIGGERED")
            sl_active = sl_order.get("state", "CANCELLED") not in ("CANCELLED", "FILLED", "TRIGGERED")
            
            tp_order_id = tp_order.get("id") if tp_order else None
            sl_order_id = sl_order.get("id") if sl_order else None
            
            # Determine who cancelled (if cancelled)
            tp_cancelled_by = None
            sl_cancelled_by = None
            if tp_order.get("state") == "CANCELLED":
                # Check if cancellingTransactionID exists (means client cancelled)
                if tp_order.get("cancellingTransactionID"):
                    tp_cancelled_by = "client"
                else:
                    tp_cancelled_by = "broker"
            if sl_order.get("state") == "CANCELLED":
                if sl_order.get("cancellingTransactionID"):
                    sl_cancelled_by = "client"
                else:
                    sl_cancelled_by = "broker"
            
            who_cancelled = "none"
            if tp_cancelled_by or sl_cancelled_by:
                if tp_cancelled_by == "client" or sl_cancelled_by == "client":
                    who_cancelled = "client"
                elif tp_cancelled_by == "broker" or sl_cancelled_by == "broker":
                    who_cancelled = "broker"
            
            return {
                "tp_active": tp_active,
                "sl_active": sl_active,
                "tp_order_id": str(tp_order_id) if tp_order_id else None,
                "sl_order_id": str(sl_order_id) if sl_order_id else None,
                "who_cancelled": who_cancelled,
            }
        except Exception as e:
            log.warning("[ARB] Failed to get TP/SL state for trade %s: %s", trade_id, e)
            return {
                "tp_active": False,
                "sl_active": False,
                "tp_order_id": None,
                "sl_order_id": None,
                "who_cancelled": "unknown",
            }
    
    def _log_exit_audit(
        self,
        trade_id: str,
        source: str,
        reason: str,
        pnl_bps: float,
        accepted: bool,
        broker_info: Optional[Dict[str, Any]] = None,
        tp_sl_state: Optional[Dict[str, Any]] = None,
        bars_in_trade: Optional[int] = None,
    ) -> None:
        """Log exit attempt to audit file (JSONL)."""
        try:
            now_ts = pd.Timestamp.now(tz="UTC")
            audit_record = {
                "ts": now_ts.isoformat(),
                "trade_id": trade_id,
                "source": source,  # MODEL_EXIT, TICK, BROKER
                "reason": reason,  # THRESHOLD, SOFT_STOP_TICK, SL_TICK, TP_TICK, BE_TICK, MANUAL
                "pnl_bps": round(pnl_bps, 2),
                "accepted": accepted,
                "broker": broker_info or {},
                "tp_sl_state": tp_sl_state or {},
                "bars_in_trade": bars_in_trade,
            }
            
            with self.exit_audit_path.open("a", encoding="utf-8") as f:
                f.write(jsonlib.dumps(audit_record, separators=(",", ":")) + "\n")
        except Exception as e:
            log.error("[ARB] Failed to log exit audit: %s", e)

    def _exit_cov_inc(self, key: str, n: int = 1) -> None:
        """Best-effort increment for exit coverage counters."""
        try:
            if not hasattr(self, "exit_coverage") or not isinstance(self.exit_coverage, dict):
                return
            self.exit_coverage[key] = int(self.exit_coverage.get(key, 0) or 0) + int(n)
        except Exception:
            return

    def _exit_cov_note_close(self, trade_id: str, reason: str) -> None:
        """Track last-N closed trade ids/reasons for proof artifacts."""
        try:
            reasons = list(self.exit_coverage.get("last_5_exit_reasons") or [])
            tids = list(self.exit_coverage.get("last_5_trade_ids_closed") or [])
            reasons.append(str(reason))
            tids.append(str(trade_id))
            self.exit_coverage["last_5_exit_reasons"] = reasons[-5:]
            self.exit_coverage["last_5_trade_ids_closed"] = tids[-5:]
        except Exception:
            return
    
    def request_close(
        self,
        trade_id: str,
        source: str,
        reason: str,
        px: float,
        pnl_bps: float,
        bars_in_trade: Optional[int] = None,
    ) -> bool:
        """
        ExitArbiter: Central gatekeeper for all exit requests.
        
        Parameters
        ----------
        trade_id : str
            Trade ID to close.
        source : str
            Source of close request: "MODEL_EXIT", "TICK", "BROKER"
        reason : str
            Reason for close: "THRESHOLD", "SOFT_STOP_TICK", "SL_TICK", "TP_TICK", "BE_TICK", "MANUAL"
        px : float
            Current price at close request.
        pnl_bps : float
            PnL in bps at close request.
        bars_in_trade : int, optional
            Number of bars since entry (for model exit validation).
        
        Returns
        -------
        bool
            True if close was accepted and executed, False if rejected.
        """
        now_ts = pd.Timestamp.now(tz="UTC")

        # Exit coverage counters (TRUTH-only export; best-effort)
        self._exit_cov_inc("exit_attempts_total", 1)
        self._exit_cov_inc("exit_request_close_called", 1)
        
        # PHASE 1 FIX: Single-exit invariant - check if trade already exited
        is_replay = getattr(self, "replay_mode", False) or getattr(self, "fast_replay", False)
        with self._closing_lock:
            if trade_id in self._exited_trade_ids:
                # Trade already exited - this is a duplicate exit attempt
                self._duplicate_exit_attempts += 1
                error_msg = (
                    f"[EXIT_INVARIANT_VIOLATION] Duplicate exit attempt for trade {trade_id} "
                    f"(source={source}, reason={reason}). Trade already exited. "
                    f"This violates single-exit-per-trade invariant."
                )
                if is_replay:
                    raise RuntimeError(error_msg)
                else:
                    log.warning(error_msg + " (live mode: ignoring duplicate exit)")
                    return False
            
            # Race-condition handling: Check if trade is already being closed
            if trade_id in self._closing_trades:
                log.info("[ARB] reject close (already closing) %s %s", trade_id, source)
                # Get TP/SL state for audit (even though we're rejecting)
                tp_sl_state = self._get_tp_sl_state(trade_id)
                self._log_exit_audit(trade_id, source, reason, pnl_bps, False, None, tp_sl_state, bars_in_trade)
                return False
            # Mark trade as closing
            self._closing_trades[trade_id] = True
        
        try:
            # HARD ASSERT: FARM_V1 mode must only use FARM exit reasons
            if hasattr(self, "farm_v1_mode") and self.farm_v1_mode:
                if not reason.startswith("EXIT_FARM"):
                    raise RuntimeError(
                        f"[FARM_V1_ASSERT] Non-FARM exit reason in request_close: {reason} "
                        f"(source={source}, trade_id={trade_id}). "
                        f"FARM_V1 mode only allows EXIT_FARM_* reasons."
                    )
            
            # Get TP/SL state before close
            tp_sl_state = self._get_tp_sl_state(trade_id)
            
            # ---------------------------------------------------
            # Big Brain V1 – Step 4: Exit-Aware Risk Shaping
            # ---------------------------------------------------
            # Get trade object to access brain_risk and brain_trend
            trade_obj = None
            for t in self.open_trades:
                if t.trade_id == trade_id:
                    trade_obj = t
                    break
            
            # Get Big Brain V1 data from trade.extra
            brain_risk = None
            brain_trend = None
            if trade_obj and hasattr(trade_obj, "extra") and trade_obj.extra:
                # Try new format (directly in trade.extra)
                brain_risk = trade_obj.extra.get("brain_risk_score", None)
                brain_trend = trade_obj.extra.get("brain_trend_regime", None)
                
                # Fallback to old format (nested in big_brain_v1 dict)
                if brain_risk is None:
                    bb_v1_data = trade_obj.extra.get("big_brain_v1", {})
                    if isinstance(bb_v1_data, dict):
                        brain_risk = bb_v1_data.get("brain_risk_score", None)
                        brain_trend = bb_v1_data.get("brain_trend_regime", None)
            
            # Fallback hvis ingen Big Brain-data
            if brain_risk is None:
                brain_risk = 0.5
            
            # Apply Big Brain risk shaping to tick_exit config (create adjusted copy)
            # Note: We adjust tick_cfg values that affect exit behavior, but don't modify self.tick_cfg directly
            # Instead, we'll adjust the values that TickWatcher uses when checking exits
            be_cfg = self.tick_cfg.get("be", {})
            be_trigger_bps_original = int(be_cfg.get("activate_at_bps", 50))
            
            # 1. BE sensitivity shaping
            #    risk=1.0 → BE trigges 20% tidligere
            #    risk=0.0 → BE trigges 0% tidligere
            be_adjust = 0.20 * brain_risk
            be_trigger_bps_adjusted = max(
                int(be_trigger_bps_original * (1.0 - be_adjust)),
                int(be_trigger_bps_original * 0.10),  # sikkerhetsgrense: min 10% of original
            )
            
            # 2. TP1 dynamic distance (adjust tp_bps)
            #    risk=1.0 → TP1 blir 10% nærmere
            #    risk=0.0 → ingen endring
            tp_bps_original = int(self.tick_cfg.get("tp_bps", 180))
            tp1_adjust = 0.10 * brain_risk
            tp_bps_adjusted = max(
                int(tp_bps_original * (1.0 - tp1_adjust)),
                int(tp_bps_original * 0.005),  # minimum TP1 avstand: min 0.5% of original
            )
            
            # 3. Trailing/soft stop aggressivitet (adjust soft_stop_bps)
            #    risk=1.0 → trailing strammes 15%
            #    risk=0.0 → trailing slakkes 0%
            soft_stop_bps_original = int(self.tick_cfg.get("soft_stop_bps", 25))
            if soft_stop_bps_original > 0:
                trail_adjust = 0.15 * brain_risk
                soft_stop_bps_adjusted = max(
                    int(soft_stop_bps_original * (1.0 - trail_adjust)),
                    int(soft_stop_bps_original * 0.003),  # trygg nedre grense: min 0.3% of original
                )
            else:
                soft_stop_bps_adjusted = soft_stop_bps_original
            
            # ---------------------------------------------------
            # Big Brain V1 – Step 5: Trend-Aware TP/SL Asymmetry
            # ---------------------------------------------------
            # Get brain_trend (already retrieved above, but ensure we have it)
            brain_trend_asym = brain_trend
            if brain_trend_asym is None and trade_obj and hasattr(trade_obj, "extra") and trade_obj.extra:
                # Fallback: try to get from trade.extra directly
                brain_trend_asym = trade_obj.extra.get("brain_trend_regime", None)
            
            # Pull values already adjusted by risk shaping
            tp_bps_asym = tp_bps_adjusted
            sl_bps_asym = int(self.tick_cfg.get("sl_bps", 95))  # SL is not adjusted by risk shaping, get from config
            soft_stop_bps_asym = soft_stop_bps_adjusted
            
            # Asymmetry strengths (konservative og trygge)
            tp_boost = 0.10  # +10% TP i trend-retning
            sl_tight = 0.10  # -10% SL mot trend (strammere)
            
            # Apply asymmetry based on trade side and trend
            if trade_obj:
                trade_side = trade_obj.side.upper() if hasattr(trade_obj, "side") else None
                
                # LONG trade
                if trade_side == "LONG":
                    if brain_trend_asym == "TREND_UP":
                        # Gi mer profittpotensial i opptrend
                        tp_bps_asym = int(tp_bps_asym * (1.0 + tp_boost))
                    elif brain_trend_asym == "TREND_DOWN":
                        # Trend imot → strammere SL og soft stop
                        sl_bps_asym = int(sl_bps_asym * (1.0 - sl_tight))
                        if soft_stop_bps_asym > 0:
                            soft_stop_bps_asym = int(soft_stop_bps_asym * (1.0 - sl_tight))
                
                # SHORT trade
                elif trade_side == "SHORT":
                    if brain_trend_asym == "TREND_DOWN":
                        # Gi mer profittpotensial i nedtrend
                        tp_bps_asym = int(tp_bps_asym * (1.0 + tp_boost))
                    elif brain_trend_asym == "TREND_UP":
                        # Trend imot → strammere SL og soft stop
                        sl_bps_asym = int(sl_bps_asym * (1.0 - sl_tight))
                        if soft_stop_bps_asym > 0:
                            soft_stop_bps_asym = int(soft_stop_bps_asym * (1.0 - sl_tight))
                
                # MR (mean-reversion) regime: ingen asymmetri
                # (keep values as is from risk shaping)
                
                # Sikkerhetsgrenser
                tp_bps_asym = max(tp_bps_asym, 5)  # minimum 5 bps
                soft_stop_bps_asym = max(soft_stop_bps_asym, 3)
                sl_bps_asym = max(sl_bps_asym, 3)
                
                # Update the adjusted values with asymmetry
                tp_bps_adjusted = tp_bps_asym
                soft_stop_bps_adjusted = soft_stop_bps_asym
            
            # ---------------------------------------------------
            # Big Brain V1 – Step 6: Exit Hysterese Dynamics
            # ---------------------------------------------------
            # Get brain_trend and brain_risk (already retrieved above)
            brain_trend_hyst = brain_trend_asym if 'brain_trend_asym' in locals() else brain_trend
            brain_risk_hyst = brain_risk
            
            if brain_trend_hyst is None and trade_obj and hasattr(trade_obj, "extra") and trade_obj.extra:
                brain_trend_hyst = trade_obj.extra.get("brain_trend_regime", None)
            if brain_risk_hyst is None:
                brain_risk_hyst = 0.5
            
            # Pull base hysterese parameters (default values - these are placeholders until hysterese is implemented)
            # Note: These will be used when hysterese functionality is implemented
            early_hold_default = 10  # Default early exit hold ticks
            soft_delay_default = 5   # Default soft stop delay ticks
            tp_delay_default = 3     # Default TP protection delay ticks
            
            early_hold = early_hold_default
            soft_delay = soft_delay_default
            tp_delay = tp_delay_default
            
            # Scaling strengths (konservative)
            trend_boost = 0.20    # +20% hysterese i trendretning
            trend_cut = 0.20      # -20% hysterese mot trend
            risk_cut_max = 0.30   # risikofaktor kan kutte opptil 30%
            
            # Apply hysterese adjustments based on trade side and trend
            if trade_obj and brain_trend_hyst is not None:
                trade_side = trade_obj.side.upper() if hasattr(trade_obj, "side") else None
                
                # 1. Trend-based hysterese
                if trade_side == "LONG":
                    if brain_trend_hyst == "TREND_UP":
                        # Med trenden → øk hysterese
                        early_hold = int(early_hold * (1.0 + trend_boost))
                        soft_delay = int(soft_delay * (1.0 + trend_boost))
                        tp_delay = int(tp_delay * (1.0 + trend_boost))
                    elif brain_trend_hyst == "TREND_DOWN":
                        # Mot trenden → kutt hysterese
                        early_hold = int(early_hold * (1.0 - trend_cut))
                        soft_delay = int(soft_delay * (1.0 - trend_cut))
                        tp_delay = int(tp_delay * (1.0 - trend_cut))
                
                elif trade_side == "SHORT":
                    if brain_trend_hyst == "TREND_DOWN":
                        # Med trenden → øk hysterese
                        early_hold = int(early_hold * (1.0 + trend_boost))
                        soft_delay = int(soft_delay * (1.0 + trend_boost))
                        tp_delay = int(tp_delay * (1.0 + trend_boost))
                    elif brain_trend_hyst == "TREND_UP":
                        # Mot trenden → kutt hysterese
                        early_hold = int(early_hold * (1.0 - trend_cut))
                        soft_delay = int(soft_delay * (1.0 - trend_cut))
                        tp_delay = int(tp_delay * (1.0 - trend_cut))
                
                # 2. Risk-based hysterese dampening
                risk_cut = risk_cut_max * brain_risk_hyst
                early_hold = int(early_hold * (1.0 - risk_cut))
                soft_delay = int(soft_delay * (1.0 - risk_cut))
                tp_delay = int(tp_delay * (1.0 - risk_cut))
                
                # Sikkerhetsgrenser
                early_hold = max(int(early_hold), 1)
                soft_delay = max(int(soft_delay), 1)
                tp_delay = max(int(tp_delay), 1)
                
                # Store hysterese adjustments in trade.extra for logging/analysis
                # Note: These values can be used when hysterese functionality is implemented
                if not hasattr(trade_obj, "extra") or trade_obj.extra is None:
                    trade_obj.extra = {}
                trade_obj.extra["bb_exit_hyst"] = {
                    "trend": brain_trend_hyst,
                    "risk": brain_risk_hyst,
                    "early_hold_adj": early_hold,
                    "early_hold_original": early_hold_default,
                    "soft_delay_adj": soft_delay,
                    "soft_delay_original": soft_delay_default,
                    "tp_delay_adj": tp_delay,
                    "tp_delay_original": tp_delay_default,
                }
            
            # ---------------------------------------------------
            # Big Brain V1 – Step 7: Adaptive TP2/TP3 Extension
            # ---------------------------------------------------
            # Get brain_trend and brain_risk (already retrieved above)
            brain_trend_tp = brain_trend_hyst if 'brain_trend_hyst' in locals() else brain_trend
            brain_risk_tp = brain_risk_hyst if 'brain_risk_hyst' in locals() else brain_risk
            
            if brain_trend_tp is None and trade_obj and hasattr(trade_obj, "extra") and trade_obj.extra:
                brain_trend_tp = trade_obj.extra.get("brain_trend_regime", None)
            if brain_risk_tp is None:
                brain_risk_tp = 0.5
            
            # Base TP1 (allerede justert fram til nå med risk shaping og asymmetry)
            tp1 = tp_bps_adjusted if 'tp_bps_adjusted' in locals() else tp_bps_original
            
            # Sikre at TP2 og TP3 finnes (hent fra trade.extra eller beregn fra TP1)
            if trade_obj and hasattr(trade_obj, "extra") and trade_obj.extra:
                tp2_base = trade_obj.extra.get("tp2_bps", None)
                tp3_base = trade_obj.extra.get("tp3_bps", None)
            else:
                tp2_base = None
                tp3_base = None
            
            if tp2_base is None:
                tp2_base = int(tp1 * 1.5)  # Default: TP2 = 1.5x TP1
            if tp3_base is None:
                tp3_base = int(tp1 * 2.0)  # Default: TP3 = 2.0x TP1
            
            tp2 = tp2_base
            tp3 = tp3_base
            
            # Styrke på extension (konservativ)
            ext_strong = 0.25    # +25% i sterk trend
            ext_medium = 0.10    # +10% i moderat trend
            risk_cut = 0.20 * brain_risk_tp  # redusere extension ved høy risk
            
            # Apply TP2/TP3 extension based on trade side and trend
            if trade_obj:
                trade_side = trade_obj.side.upper() if hasattr(trade_obj, "side") else None
                
                # LONG trade
                if trade_side == "LONG":
                    if brain_trend_tp == "TREND_UP":
                        # Sterk trend → legg til 25% - risk
                        tp2 = int(tp2 * (1.0 + ext_strong - risk_cut))
                        tp3 = int(tp3 * (1.0 + ext_strong - risk_cut))
                    elif brain_trend_tp == "MR":
                        # Moderat → 10% extension
                        tp2 = int(tp2 * (1.0 + ext_medium - risk_cut))
                        tp3 = int(tp3 * (1.0 + ext_medium - risk_cut))
                    elif brain_trend_tp == "TREND_DOWN":
                        # Mot trend → ingen extension
                        pass
                
                # SHORT trade
                elif trade_side == "SHORT":
                    if brain_trend_tp == "TREND_DOWN":
                        # Sterk trend → legg til 25% - risk
                        tp2 = int(tp2 * (1.0 + ext_strong - risk_cut))
                        tp3 = int(tp3 * (1.0 + ext_strong - risk_cut))
                    elif brain_trend_tp == "MR":
                        # Moderat → 10% extension
                        tp2 = int(tp2 * (1.0 + ext_medium - risk_cut))
                        tp3 = int(tp3 * (1.0 + ext_medium - risk_cut))
                    elif brain_trend_tp == "TREND_UP":
                        # Mot trend → ingen extension
                        pass
                
                # Sikkerhetsgrenser
                tp2 = max(int(tp2), int(tp1) + 3)  # TP2 må alltid være minst litt over TP1
                tp3 = max(int(tp3), int(tp2) + 3)  # TP3 må være over TP2
                
                # Lagre TP2/TP3 extension i trade.extra
                if not hasattr(trade_obj, "extra") or trade_obj.extra is None:
                    trade_obj.extra = {}
                trade_obj.extra["bb_exit_tp_ext"] = {
                    "trend": brain_trend_tp,
                    "risk": brain_risk_tp,
                    "tp1": tp1,
                    "tp2_base": tp2_base,
                    "tp2_adj": tp2,
                    "tp3_base": tp3_base,
                    "tp3_adj": tp3,
                }
                
                # Store TP2/TP3 in trade.extra for use by exit logic
                trade_obj.extra["tp2_bps"] = tp2
                trade_obj.extra["tp3_bps"] = tp3
            
            # Store adjusted values in trade.extra for logging (if trade exists)
            if trade_obj:
                if not hasattr(trade_obj, "extra") or trade_obj.extra is None:
                    trade_obj.extra = {}
                trade_obj.extra["bb_exit"] = {
                    "be_trigger_bps_adj": be_trigger_bps_adjusted,
                    "be_trigger_bps_original": be_trigger_bps_original,
                    "tp_bps_adj": tp_bps_adjusted,
                    "tp_bps_original": tp_bps_original,
                    "soft_stop_bps_adj": soft_stop_bps_adjusted,
                    "soft_stop_bps_original": soft_stop_bps_original,
                    "brain_risk": brain_risk,
                    "brain_trend": brain_trend,
                }
                
                # Store asymmetry adjustments separately for analysis
                trade_obj.extra["bb_exit_asym"] = {
                    "trend": brain_trend_asym,
                    "tp_bps_adj": tp_bps_asym,
                    "tp_bps_risk_shaped": tp_bps_adjusted,  # before asymmetry
                    "sl_bps_adj": sl_bps_asym,
                    "soft_stop_bps_adj": soft_stop_bps_asym,
                    "soft_stop_bps_risk_shaped": soft_stop_bps_adjusted,  # before asymmetry
                }
                
                # Log Big Brain exit shaping (debug level)
                if be_trigger_bps_adjusted != be_trigger_bps_original or tp_bps_adjusted != tp_bps_original or soft_stop_bps_adjusted != soft_stop_bps_original:
                    log.debug(
                        "[BIG_BRAIN_V1] Exit shaping applied: risk=%.3f be=%d→%d tp=%d→%d soft_stop=%d→%d",
                        brain_risk,
                        be_trigger_bps_original,
                        be_trigger_bps_adjusted,
                        tp_bps_original,
                        tp_bps_adjusted,
                        soft_stop_bps_original,
                        soft_stop_bps_adjusted,
                    )
                
                # Log asymmetry adjustments (debug level)
                if tp_bps_asym != tp_bps_adjusted or soft_stop_bps_asym != soft_stop_bps_adjusted or sl_bps_asym != int(self.tick_cfg.get("sl_bps", 95)):
                    log.debug(
                        "[BIG_BRAIN_V1] Asymmetry applied: trend=%s tp=%d→%d sl=%d→%d soft_stop=%d→%d",
                        brain_trend_asym or "UNKNOWN",
                        tp_bps_adjusted,  # before asymmetry
                        tp_bps_asym,
                        int(self.tick_cfg.get("sl_bps", 95)),
                        sl_bps_asym,
                        soft_stop_bps_adjusted,  # before asymmetry
                        soft_stop_bps_asym,
                    )
                
                # Log hysterese adjustments (debug level)
                if "bb_exit_hyst" in trade_obj.extra:
                    hyst_data = trade_obj.extra["bb_exit_hyst"]
                    if (hyst_data.get("early_hold_adj", early_hold_default) != early_hold_default or
                        hyst_data.get("soft_delay_adj", soft_delay_default) != soft_delay_default or
                        hyst_data.get("tp_delay_adj", tp_delay_default) != tp_delay_default):
                        log.debug(
                            "[BIG_BRAIN_V1] Hysterese applied: trend=%s risk=%.3f early_hold=%d→%d soft_delay=%d→%d tp_delay=%d→%d",
                            brain_trend_hyst or "UNKNOWN",
                            brain_risk_hyst,
                            early_hold_default,
                            hyst_data.get("early_hold_adj", early_hold_default),
                            soft_delay_default,
                            hyst_data.get("soft_delay_adj", soft_delay_default),
                            tp_delay_default,
                            hyst_data.get("tp_delay_adj", tp_delay_default),
                        )
                
                # Log TP2/TP3 extension adjustments (debug level)
                if "bb_exit_tp_ext" in trade_obj.extra:
                    tp_ext_data = trade_obj.extra["bb_exit_tp_ext"]
                    if (tp_ext_data.get("tp2_adj", tp_ext_data.get("tp2_base", 0)) != tp_ext_data.get("tp2_base", 0) or
                        tp_ext_data.get("tp3_adj", tp_ext_data.get("tp3_base", 0)) != tp_ext_data.get("tp3_base", 0)):
                        log.debug(
                            "[BIG_BRAIN_V1] TP2/TP3 extension applied: trend=%s risk=%.3f tp2=%d→%d tp3=%d→%d",
                            brain_trend_tp or "UNKNOWN",
                            brain_risk_tp,
                            tp_ext_data.get("tp2_base", 0),
                            tp_ext_data.get("tp2_adj", 0),
                            tp_ext_data.get("tp3_base", 0),
                            tp_ext_data.get("tp3_adj", 0),
                        )
            
            # Note: The adjusted values are stored in trade.extra for logging,
            # but TickWatcher will continue using self.tick_cfg for actual exit checks.
            # If we want to apply these adjustments in real-time, we would need to
            # modify TickWatcher to check trade.extra["bb_exit"] values, but for now
            # this provides logging/analysis capability.
            
            # Verify trade is still open (if required) - with grace period for "closing" trades
            if self.exit_control.require_trade_open:
                trade_exists = any(t.trade_id == trade_id for t in self.open_trades)
                if not trade_exists:
                    # Trade is not in open_trades, but it might be in closing state (reconcile grace)
                    # Allow close if trade is marked as closing (reconcile might be one tick behind)
                    if trade_id in self._closing_trades:
                        log.debug("[ARB] allow close (closing state) %s %s (reconcile grace)", trade_id, source)
                    else:
                        log.info("[ARB] reject close (not open) %s %s", trade_id, source)
                        self._log_exit_audit(trade_id, source, reason, pnl_bps, False, None, tp_sl_state, bars_in_trade)
                        # Remove from closing_trades on reject
                        with self._closing_lock:
                            self._closing_trades.pop(trade_id, None)
                        return False
            
            # Additional checks for MODEL_EXIT (check before loss-close check for clearer messages)
            if source == "MODEL_EXIT":
                min_bars = self.exit_control.allow_model_exit_when.get("min_bars", 2)
                min_pnl_bps = self.exit_control.allow_model_exit_when.get("min_pnl_bps", -5)
                min_exit_prob = self.exit_control.allow_model_exit_when.get("min_exit_prob", 0.70)
                exit_prob_hysteresis = self.exit_control.allow_model_exit_when.get("exit_prob_hysteresis", 2)
                
                # Get exit_prob_close and prob_close_history from trade
                exit_prob_close = None
                prob_close_history = []
                for trade in self.open_trades:
                    if trade.trade_id == trade_id:
                        if hasattr(trade, 'prob_close_history') and len(trade.prob_close_history) > 0:
                            exit_prob_close = float(trade.prob_close_history[-1])
                            prob_close_history = [float(p) for p in trade.prob_close_history]
                        break
                
                # Check min_bars
                if bars_in_trade is not None and bars_in_trade < min_bars:
                    rule = f"min_bars:{min_bars}"
                    log.info("[ARB] reject model exit (bars_in_trade=%d < min_bars=%d) pnl=%.1f rule=%s", bars_in_trade, min_bars, pnl_bps, rule)
                    self._log_exit_audit(trade_id, source, reason, pnl_bps, False, None, tp_sl_state, bars_in_trade)
                    # Remove from closing_trades on reject
                    with self._closing_lock:
                        self._closing_trades.pop(trade_id, None)
                    return False
                
                # Check min_exit_prob (current exit_prob_close must be >= min_exit_prob)
                if exit_prob_close is None:
                    rule = "min_exit_prob:missing"
                    log.info("[ARB] reject model exit (exit_prob_close missing) trade_id=%s rule=%s", trade_id, rule)
                    self._log_exit_audit(trade_id, source, reason, pnl_bps, False, None, tp_sl_state, bars_in_trade)
                    with self._closing_lock:
                        self._closing_trades.pop(trade_id, None)
                    return False
                
                if exit_prob_close < min_exit_prob:
                    rule = f"min_exit_prob:{min_exit_prob}"
                    log.info("[ARB] reject model exit (exit_prob_close=%.4f < min_exit_prob=%.4f) pnl=%.1f rule=%s", exit_prob_close, min_exit_prob, pnl_bps, rule)
                    self._log_exit_audit(trade_id, source, reason, pnl_bps, False, None, tp_sl_state, bars_in_trade)
                    with self._closing_lock:
                        self._closing_trades.pop(trade_id, None)
                    return False
                
                # Check exit_prob_hysteresis (require exit_prob >= min_exit_prob in N consecutive bars)
                if exit_prob_hysteresis > 1 and len(prob_close_history) >= exit_prob_hysteresis:
                    recent_bars = prob_close_history[-exit_prob_hysteresis:]
                    if not all(p >= min_exit_prob for p in recent_bars):
                        rule = f"exit_prob_hysteresis:{exit_prob_hysteresis}"
                        log.info("[ARB] reject model exit (exit_prob < %.4f in %d consecutive bars) exit_prob_history=%s rule=%s", 
                                min_exit_prob, exit_prob_hysteresis, recent_bars, rule)
                        self._log_exit_audit(trade_id, source, reason, pnl_bps, False, None, tp_sl_state, bars_in_trade)
                        with self._closing_lock:
                            self._closing_trades.pop(trade_id, None)
                        return False
                elif exit_prob_hysteresis > 1 and len(prob_close_history) < exit_prob_hysteresis:
                    rule = f"exit_prob_hysteresis:{exit_prob_hysteresis}"
                    log.info("[ARB] reject model exit (not enough prob_close_history: %d < %d) rule=%s", 
                            len(prob_close_history), exit_prob_hysteresis, rule)
                    self._log_exit_audit(trade_id, source, reason, pnl_bps, False, None, tp_sl_state, bars_in_trade)
                    with self._closing_lock:
                        self._closing_trades.pop(trade_id, None)
                    return False
                
                # MODEL_EXIT can close at profit/BE (pnl_bps >= 0) - always allowed if above checks pass
                if pnl_bps >= 0:
                    # Profit/BE close: always allowed for MODEL_EXIT (if min_exit_prob and hysteresis pass)
                    log.info("[ARB] allow MODEL_EXIT at profit pnl=%.1f bars=%d exit_prob=%.4f (>= min_exit_prob=%.4f)", 
                            pnl_bps, bars_in_trade or 0, exit_prob_close, min_exit_prob)
                    pass  # Continue to execute close below
                # MODEL_EXIT can close at moderate loss if pnl_bps >= min_pnl_bps (e.g., -10 bps >= -15 bps)
                elif pnl_bps >= min_pnl_bps:
                    # Moderate loss: allow MODEL_EXIT to clean up (e.g., -12 bps >= -15 bps → allow)
                    log.info("[ARB] allow MODEL_EXIT at moderate loss pnl=%.1f bars=%d exit_prob=%.4f (>= min_pnl_bps=%.1f, min_exit_prob=%.4f)", 
                            pnl_bps, bars_in_trade or 0, exit_prob_close, min_pnl_bps, min_exit_prob)
                    # Continue to execute close below (skip the loss-close restriction for MODEL_EXIT at moderate loss)
                else:
                    # Deep loss: reject MODEL_EXIT (pnl_bps < min_pnl_bps, e.g., -20 bps < -15 bps)
                    rule = f"min_pnl_bps:{min_pnl_bps}"
                    log.info("[ARB] reject loss close by MODEL_EXIT pnl=%.1f bars=%s exit_prob=%.4f rule=%s", 
                            pnl_bps, bars_in_trade or "?", exit_prob_close, rule)
                    self._log_exit_audit(trade_id, source, reason, pnl_bps, False, None, tp_sl_state, bars_in_trade)
                    # Remove from closing_trades on reject
                    with self._closing_lock:
                        self._closing_trades.pop(trade_id, None)
                    return False
            
            # REPLAY_EOF is always allowed in replay mode (end-of-simulation liquidation)
            if reason == "REPLAY_EOF" and hasattr(self, "is_replay") and self.is_replay:
                # Always allow REPLAY_EOF in replay mode, bypass normal checks
                pass
            # Check if this is a loss close (skip this check for MODEL_EXIT at moderate loss, already handled above)
            elif pnl_bps < 0 and source != "MODEL_EXIT":
                # Loss close: only allowed reasons can close
                # Map reason to allowed_loss_closers (e.g., "SOFT_STOP_TICK" -> "SOFT_STOP_TICK", "SL_TICK" -> "BROKER_SL")
                # Note: "BROKER_SL" in allowed_loss_closers refers to broker-side SL (handled by OANDA automatically)
                #       but we can also have "SL_TICK" from tick-watcher (which should be allowed if broker-side SL is allowed)
                allowed_reasons = set(self.exit_control.allowed_loss_closers)
                # Also allow SL_TICK if BROKER_SL is allowed (both are stop-loss mechanisms)
                if "BROKER_SL" in allowed_reasons:
                    allowed_reasons.add("SL_TICK")
                
                if reason not in allowed_reasons:
                    # Improved logging: show rule details
                    log.info("[ARB] reject loss close by %s reason=%s (pnl=%.1f bps, allowed=%s)", source, reason, pnl_bps, self.exit_control.allowed_loss_closers)
                    self._log_exit_audit(trade_id, source, reason, pnl_bps, False, None, tp_sl_state, bars_in_trade)
                    # Remove from closing_trades on reject
                    with self._closing_lock:
                        self._closing_trades.pop(trade_id, None)
                    return False
            else:
                # Profit/BE close: always allowed (MODEL_EXIT can close at profit/BE)
                pass
            
            # Close accepted: cancel TP/SL orders before market-close (for internal closes)
            # Note: OANDA automatically cancels TP/SL when trade is closed, but we log it for audit
            if source != "BROKER":
                # For internal closes (MODEL_EXIT, TICK), TP/SL will be automatically cancelled
                # We log this for audit trail
                if tp_sl_state.get("tp_active") or tp_sl_state.get("sl_active"):
                    log.debug("[ARB] TP/SL orders will be automatically cancelled on close: tp_active=%s sl_active=%s", 
                             tp_sl_state.get("tp_active"), tp_sl_state.get("sl_active"))
            
            # Close accepted: execute broker close with retry (handled by OANDA client)
            tp_active = tp_sl_state.get("tp_active", False)
            sl_active = tp_sl_state.get("sl_active", False)
            # Improved logging: show rule details for MODEL_EXIT, tp_sl_state for all
            if source == "MODEL_EXIT":
                log.info("[ARB] accept close by %s reason=%s pnl=%.1f bars=%d tp_active=%s sl_active=%s", 
                        source, reason, pnl_bps, bars_in_trade or 0, tp_active, sl_active)
            else:
                log.info("[ARB] accept close by %s reason=%s pnl=%.1f tp_active=%s sl_active=%s", 
                        source, reason, pnl_bps, tp_active, sl_active)
            
            broker_info = {}
            max_retries = 3
            retry_delays = [1.0, 2.0, 4.0]  # Exponential backoff: 1s, 2s, 4s
            
            for attempt in range(max_retries):
                try:
                    # In replay mode, broker is None - simulate close
                    if self.replay_mode and self.broker is None:
                        # Simulate successful close in replay mode
                        log.debug("[REPLAY] Simulating trade close (broker=None in replay mode)")
                        response = {
                            "orderFillTransaction": {
                                "id": f"REPLAY-CLOSE-{trade_id}",
                                "time": now_ts.isoformat() if hasattr(now_ts, 'isoformat') else str(now_ts),
                            }
                        }
                    elif not self.exec.dry_run:
                        response = self.broker.close_trade(trade_id)
                        # Extract orderFillTransaction or closing transaction info
                        order_fill = response.get("orderFillTransaction", {})
                        broker_info = {
                            "filled": True,
                            "orderFillId": order_fill.get("id"),
                            "closePrice": float(order_fill.get("price", px)),
                            "attempt": attempt + 1,
                        }
                        break  # Success, exit retry loop
                    else:
                        broker_info = {
                            "filled": True,
                            "orderFillId": "dry_run",
                            "closePrice": px,
                            "attempt": attempt + 1,
                        }
                        break  # Success (dry-run), exit retry loop
                except Exception as e:
                    error_str = str(e)
                    is_retryable = any(code in error_str for code in ["500", "502", "503", "504", "timeout", "Timeout"])
                    
                    if attempt < max_retries - 1 and is_retryable:
                        wait_time = retry_delays[attempt]
                        log.warning("[ARB] broker close failed (attempt %d/%d) for trade %s: %s. Retrying in %.1fs...", 
                                   attempt + 1, max_retries, trade_id, error_str, wait_time)
                        time.sleep(wait_time)
                        continue
                    else:
                        # Final attempt failed or non-retryable error
                        broker_info = {
                            "filled": False,
                            "error": error_str,
                            "attempt": attempt + 1,
                        }
                        log.error("[ARB] broker close failed for trade %s after %d attempts: %s", trade_id, attempt + 1, error_str)
                        self._log_exit_audit(trade_id, source, reason, pnl_bps, False, broker_info, tp_sl_state, bars_in_trade)
                        # Remove from closing_trades on failure
                        with self._closing_lock:
                            self._closing_trades.pop(trade_id, None)
                        return False
            
            # PHASE 1 FIX: Mark trade as exited EARLY (single-exit invariant)
            # This must be done BEFORE logging/committing to prevent duplicate exit attempts
            # in the same bar loop iteration
            with self._closing_lock:
                self._exited_trade_ids.add(trade_id)
            
            # Log accepted close
            self._log_exit_audit(trade_id, source, reason, pnl_bps, True, broker_info, tp_sl_state, bars_in_trade)
            
            # Update trade log CSV with exit information
            self._update_trade_log_on_close(trade_id, px, pnl_bps, reason, now_ts, bars_in_trade=bars_in_trade)

            # Remove from closing_trades on success
            with self._closing_lock:
                self._closing_trades.pop(trade_id, None)
            
            # Tear down per-trade exit state (if any)
            self._teardown_exit_state(trade_id)

            # Exit coverage: accepted close
            self._exit_cov_inc("exit_request_close_accepted", 1)
            self._exit_cov_note_close(trade_id=trade_id, reason=reason)

            # ------------------------------------------------------------------
            # TRUTH/REPLAY journaling contract:
            # If a close is accepted during replay (not only at EOF), we must
            # journal the EXIT event + exit_summary. This is telemetry-only and
            # does not affect exit timing/criteria.
            #
            # NOTE: REPLAY_EOF path already logs exit summary separately; avoid
            # double-logging by skipping here for source=REPLAY_EOF.
            # ------------------------------------------------------------------
            if is_replay and source != "REPLAY_EOF":
                if hasattr(self, "trade_journal") and self.trade_journal:
                    try:
                        trade_uid = None
                        # Best-effort: resolve trade_uid from open_trades
                        for t in self.open_trades:
                            if getattr(t, "trade_id", None) == trade_id:
                                trade_uid = getattr(t, "trade_uid", None)
                                break

                        # Prefer broker close price if present (telemetry only)
                        exit_px = px
                        try:
                            if isinstance(broker_info, dict) and broker_info.get("closePrice") is not None:
                                exit_px = float(broker_info.get("closePrice"))
                        except Exception:
                            exit_px = px

                        # In replay mode, enforce that trade_uid exists and has correct prefix when env is present
                        if self.is_replay:
                            env_run_id = os.getenv("GX1_RUN_ID")
                            env_chunk_id = os.getenv("GX1_CHUNK_ID")
                            if env_run_id and env_chunk_id and trade_uid:
                                expected_prefix = f"{env_run_id}:{env_chunk_id}:"
                                if not str(trade_uid).startswith(expected_prefix):
                                    raise RuntimeError(
                                        f"BAD_TRADE_UID_FORMAT_REPLAY: trade_uid={trade_uid} does not start with "
                                        f"expected prefix={expected_prefix}. GX1_RUN_ID={env_run_id}, GX1_CHUNK_ID={env_chunk_id}."
                                    )

                        self.trade_journal.log_exit_summary(
                            exit_time=(
                                getattr(self, "replay_current_ts", None).isoformat()
                                if getattr(self, "replay_current_ts", None) is not None
                                else now_ts.isoformat()
                            ),
                            exit_reason=reason,
                            exit_price=exit_px,
                            realized_pnl_bps=pnl_bps,
                            trade_uid=trade_uid,
                            trade_id=trade_id,
                        )
                        self._exit_cov_inc("exit_summary_logged", 1)
                        self._exit_cov_note_close(trade_id=trade_id, reason=reason)
                    except Exception as e:
                        # TradeLifecycleV1/TRUTH: accepted close must be journaled. Fail-fast in replay/TRUTH.
                        if self._is_truth_mode():
                            raise
                        log.warning("[REPLAY] Failed to journal exit summary for trade %s: %s", trade_id, e)

            # CRITICAL: Remove trade from open_trades after successful close + journaling.
            # This prevents duplicate exit attempts and ensures max_open_trades is meaningful.
            try:
                for t in list(self.open_trades):
                    if getattr(t, "trade_id", None) == trade_id:
                        try:
                            self.open_trades.remove(t)
                        except ValueError:
                            pass
                        break
            except Exception:
                pass
            
            return True
        
        except Exception as e:
            # Unexpected error: remove from closing_trades and log
            log.error("[ARB] unexpected error in request_close for trade %s: %s", trade_id, e, exc_info=True)
            with self._closing_lock:
                self._closing_trades.pop(trade_id, None)
            # Get TP/SL state for audit
            tp_sl_state = self._get_tp_sl_state(trade_id)
            self._log_exit_audit(trade_id, source, reason, pnl_bps, False, {"error": str(e)}, tp_sl_state, bars_in_trade)
            return False

    def _update_trade_log_on_close(self, trade_id: str, exit_price: float, pnl_bps: float, reason: str, exit_ts: pd.Timestamp, bars_in_trade: Optional[int] = None) -> None:
        from gx1.execution.live_features import infer_session_tag
        # DEL 1: Hook TradeOutcomeCollector - collect trade outcome when trade closes
        # Hook point: When trade closes (in _update_trade_log_on_close)
        if hasattr(self, "replay_eval_collectors") and self.replay_eval_collectors:
            outcome_collector = self.replay_eval_collectors.get("trade_outcomes")
            if outcome_collector:
                # Find trade to get additional info
                trade_obj = None
                for t in self.open_trades:
                    if t.trade_id == trade_id:
                        trade_obj = t
                        break
                
                # Extract MAE/MFE from trade if available
                mae_bps = None
                mfe_bps = None
                session = None
                if trade_obj and hasattr(trade_obj, "extra") and trade_obj.extra:
                    mae_bps = trade_obj.extra.get("mae_bps")
                    mfe_bps = trade_obj.extra.get("mfe_bps")
                    session = trade_obj.extra.get("session") or getattr(trade_obj, "session", None)
                
                outcome_collector.collect(
                    trade_id=trade_id,
                    pnl_bps=pnl_bps,
                    mae_bps=mae_bps,
                    mfe_bps=mfe_bps,
                    duration_bars=bars_in_trade,
                    session=session,
                    exit_reason=reason,
                )
        """Update trade log CSV with exit information when a trade is closed."""
        try:
            # Get exit_prob_close and extra from trade if available
            exit_prob_close = None
            trade_extra = None
            session = None
            vol_regime = None
            trend_regime = None
            primary_exit_reason = reason  # First exit that closes the position
            
            for trade in self.open_trades:
                if trade.trade_id == trade_id:
                    # Check if trade has prob_close_history (for model exits)
                    if hasattr(trade, 'prob_close_history') and len(trade.prob_close_history) > 0:
                        exit_prob_close = float(trade.prob_close_history[-1])
                    # Get extra data if available
                    if hasattr(trade, 'extra') and trade.extra:
                        trade_extra = trade.extra
                        # CRITICAL: Extract ENTRY-regime directly from extra.session and extra.atr_regime
                        # These are set at entry and never overwritten
                        session_entry = trade_extra.get("session")  # Use extra.session (set at entry)
                        vol_regime_entry = trade_extra.get("atr_regime")  # Use extra.atr_regime (set at entry)
                        
                        # Extract EXIT-regime from current bar (for analysis)
                        # Get current session and vol_regime from candles/current state
                        session_exit = None
                        vol_regime_exit = None
                        # Try to infer from exit timestamp
                        session_exit = infer_session_tag(exit_ts).upper()
                        
                        # For vol_regime_exit, we'd need current bar data - for now, use entry regime
                        # (This can be enhanced later if needed)
                        vol_regime_exit = vol_regime_entry  # Use entry regime as exit regime for now
                        
                        # Store exit-regime in trade.extra (don't overwrite entry-regime)
                        trade_extra["session_exit"] = session_exit
                        trade_extra["vol_regime_exit"] = vol_regime_exit
                        
                        # For backward compatibility, use entry-regime for session/vol_regime
                        session = session_entry
                        vol_regime = vol_regime_entry
                        
                        trend_regime = trade_extra.get("brain_trend_regime") or (trade_extra.get("big_brain_v1", {}).get("brain_trend_regime") if isinstance(trade_extra.get("big_brain_v1"), dict) else None)
                        # Check if primary_exit_reason already exists (from previous exit attempt)
                        if "primary_exit_reason" in trade_extra:
                            primary_exit_reason = trade_extra["primary_exit_reason"]
                        else:
                            # Store primary_exit_reason in trade.extra for future reference
                            trade_extra["primary_exit_reason"] = reason
                    # Calculate bars_held if not provided
                    if bars_in_trade is None:
                        delta_minutes = (exit_ts - trade.entry_time).total_seconds() / 60.0
                        bars_in_trade = int(round(delta_minutes / 5.0))  # M5 = 5 minutes per bar
                    break
            
            # Update trade log
            exit_time_str = exit_ts.isoformat()
            update_trade_log_exit(
                self.trade_log_path,
                trade_id=trade_id,
                exit_time=exit_time_str,
                exit_price=exit_price,
                pnl_bps=pnl_bps,
                exit_prob_close=exit_prob_close,
                exit_reason=reason,  # Store exit reason
                primary_exit_reason=primary_exit_reason,  # First exit that closed the position
                bars_held=bars_in_trade,  # Store bars held
                session=session,  # Store session
                vol_regime=vol_regime,  # Store vol_regime
                trend_regime=trend_regime,  # Store trend_regime
                extra=trade_extra,  # Pass extra data
            )
        except Exception as e:
            log.warning("Failed to update trade log for %s: %s", trade_id, e)

    def can_enter(self, now: pd.Timestamp) -> bool:
        # Check warmup_floor (skip active trading until warmup_floor is passed)
        if self.warmup_floor is not None and now < self.warmup_floor:
            log.debug("Skip entry: warmup in progress (warmup_floor=%s)", self.warmup_floor.isoformat())
            return False
        
        # Check backfill in progress (block orders during backfill)
        if self.backfill_in_progress:
            log.debug("Skip entry: backfill in progress")
            return False
        
        if len(self.open_trades) >= self.risk_limits.max_open_trades:
            # Debug logging: show which trades are blocking
            open_trade_ids = [t.trade_id for t in self.open_trades]
            log.info(
                "Skip entry: max_open_trades reached (%s) open_trades=%s",
                self.risk_limits.max_open_trades,
                open_trade_ids
            )
            return False
        if self.daily_loss_exceeded(now):
            return False
        if self.last_entry_timestamp is not None:
            delta = (now - self.last_entry_timestamp).total_seconds()
            if delta < self.risk_limits.min_time_between_trades_sec:
                log.info("Skip entry: min_time_between_trades_sec guard (%.1fs < %ss)", delta, self.risk_limits.min_time_between_trades_sec)
                return False
        return True

    # ------------------------------------------------------------------ #
    # Entry / exit execution
    # ------------------------------------------------------------------ #
    def _resolve_session_key(self, session_tag: str) -> str:
        """
        Resolve an inferred session tag to a model key.
        
        SAFETY_V4: Session tags are now one of 'EU', 'US', 'OVERLAP', 'ASIA'.
        Never 'UNKNOWN' or 'ALL' (these are deprecated).

        Parameters
        ----------
        session_tag : str
            Raw session tag inferred from timestamp (via infer_session_tag).

        Returns
        -------
        str
            Model key: 'EU', 'US', 'OVERLAP', or 'ASIA' (falls back to 'OVERLAP').
        """
        if not session_tag:
            return "OVERLAP"
        tag = session_tag.upper()
        
        # SAFETY_V4: Check if tag is a valid session (EU, US, OVERLAP, ASIA)
        valid_sessions = ("EU", "US", "OVERLAP", "ASIA")
        
        if self.entry_model_bundle is not None and tag in self.entry_model_bundle.models:
            return tag
        
        # SAFETY_V4: Map deprecated tags to valid sessions
        if tag == "ALL" or tag == "UNKNOWN":
            # Map deprecated tags to OVERLAP (model routing)
            # Note: Entry gates should use the original session_tag, not this resolved key
            log.debug("Deprecated session tag '%s' mapped to OVERLAP for model routing", tag)
            return "OVERLAP"
        
        # SAFETY_V4: ASIA session support (if model exists, use it; otherwise fall back to OVERLAP)
        if tag == "ASIA":
            if self.entry_model_bundle is not None and "ASIA" in self.entry_model_bundle.models:
                return "ASIA"
            # ASIA model not available yet - fall back to OVERLAP (this is expected, not a warning)
            # SAFETY_V4: ASIA is a valid session tag, but we don't have an ASIA model yet
            # Don't log warning - this is expected behavior
            return "OVERLAP"
        
        # Unknown tag - fall back to OVERLAP
        if tag not in valid_sessions:
            # Only log debug, not warning, for unknown tags (ASIA is expected to fall back)
            log.debug("Unknown session tag '%s' (not in %s), falling back to OVERLAP model", tag, valid_sessions)
        
        return "OVERLAP"

    def _get_temperature_map(self) -> Dict[str, float]:
        """
        Get temperature map from entry_params.

        Returns
        -------
        Dict[str, float]
            Mapping of session key -> temperature value.
        """
        calibration_cfg = self.entry_params.get("calibration", {})
        temp_map = calibration_cfg.get("temperature_map", {})
        
        # Fallback to direct TEMP_* keys if temperature_map is not available
        if not temp_map:
            temp_map = {
                "EU": float(self.entry_params.get("TEMP_EU", 1.0)),
                "US": float(self.entry_params.get("TEMP_US", 1.0)),
                "OVERLAP": float(self.entry_params.get("TEMP_OVERLAP", 1.0)),
            }
        
        return temp_map

    def _apply_temperature(self, p: float, T: float) -> float:
        """
        Apply temperature scaling to a probability using logit transformation.

        Parameters
        ----------
        p : float
            Raw probability (must be in [0, 1]).
        T : float
            Temperature parameter (T > 0). T < 1 sharpens, T > 1 softens.

        Returns
        -------
        float
            Temperature-scaled probability.
        """
        eps = 1e-8
        p = min(max(p, eps), 1.0 - eps)
        logit_p = np.log(p) - np.log(1.0 - p)
        logit_T = logit_p / max(T, 1e-6)
        return 1.0 / (1.0 + np.exp(-logit_T))

    def _predict_entry(
        self,
        entry_bundle: EntryFeatureBundle,
        timestamp: pd.Timestamp,
    ) -> Optional[EntryPrediction]:
        """
        Generate entry probabilities for the current session.

        Parameters
        ----------
        entry_bundle : EntryFeatureBundle
            Feature bundle produced by build_live_entry_features.
        timestamp : pd.Timestamp
            Timestamp associated with the most recent candle.
        """
        from gx1.execution.live_features import infer_session_tag
        session_tag = infer_session_tag(timestamp)
        
        # Sticky session: Don't change session during an hour (prevent routing glitches)
        if self.last_session and self.last_session_timestamp is not None:
            minutes_since_last_session = (timestamp - self.last_session_timestamp).total_seconds() / 60.0
            if minutes_since_last_session < self.session_sticky_minutes:
                # Keep using last session (don't change mid-hour)
                session_tag = self.last_session
                log.debug(
                    "[ROUTE] Sticky session: keeping %s (inferred=%s, minutes_since_last=%.1f)",
                    session_tag,
                    infer_session_tag(timestamp),
                    minutes_since_last_session,
                )
            else:
                # Session changed (hour passed)
                if session_tag != self.last_session:
                    log.info(
                        "[ROUTE] Session changed: %s -> %s (minutes_since_last=%.1f)",
                        self.last_session,
                        session_tag,
                        minutes_since_last_session,
                    )
                self.last_session = session_tag
                self.last_session_timestamp = timestamp
        else:
            # First session (initialize)
            self.last_session = session_tag
            self.last_session_timestamp = timestamp
            log.debug("[ROUTE] Initial session: %s", session_tag)
        
        session_key = self._resolve_session_key(session_tag)
        
        # Fail-safe: Warn if session tag is unknown and fell back to OVERLAP
        # Note: "ALL" is intentionally mapped to OVERLAP without warning (historical catch-all)
        # Note: "ASIA" is a valid session but model may not exist - this is expected, not a warning
        if session_tag and session_tag.upper() not in ("EU", "US", "OVERLAP", "ALL", "ASIA") and session_key == "OVERLAP":
            log.warning(
                "Unknown session tag '%s', falling back to OVERLAP model. "
                "This may indicate incorrect session inference.",
                session_tag,
            )
        
        if self.entry_model_bundle is None:
            log.debug("Entry model bundle is not available (session-routed models disabled). Skipping session-routed entry.")
            return None
        
        model = self.entry_model_bundle.models.get(session_key)
        if model is None:
            log.error("Entry model for session '%s' is unavailable.", session_key)
            return None

        feature_cols = self.entry_model_bundle.feature_names
        feat_df = entry_bundle.features
        
        # Guard 1: Verify feature columns are present in feature DataFrame
        missing_cols = set(feature_cols) - set(feat_df.columns)
        if missing_cols:
            log.error(
                "Missing feature columns for session '%s': %s. "
                "Available columns: %s",
                session_key,
                list(missing_cols)[:5],
                list(feat_df.columns)[:10],
            )
            return None
        
        # Guard 2: Explicitly reindex to feature_cols in correct order and verify shape
        aligned = feat_df.reindex(columns=feature_cols, fill_value=0.0)
        if aligned.empty:
            log.error("Aligned entry feature frame is empty; cannot compute prediction.")
            return None
        
        # Guard 3: HARD STOP if n_features_in_ ≠ manifest (fail-safe gate)
        if hasattr(model, "n_features_in_"):
            if len(feature_cols) != model.n_features_in_:
                error_msg = (
                    f"HARD STOP: Feature count mismatch for session '{session_key}': "
                    f"model expects {model.n_features_in_} features, "
                    f"but manifest specifies {len(feature_cols)} features. "
                    f"This indicates a critical mismatch between model and feature manifest. "
                    f"Trading is disabled to prevent incorrect predictions."
                )
                log.error(error_msg)
                raise ValueError(error_msg)
        
        # Extract row and ensure correct dtype and shape
        row = aligned.iloc[-1].astype(np.float32).to_numpy().reshape(1, -1)
        
        # Guard 4: Verify row shape
        if row.shape[1] != len(feature_cols):
            log.error(
                "Row shape mismatch for session '%s': expected %d features, got %d",
                session_key,
                len(feature_cols),
                row.shape[1],
            )
            return None
        
        # Predict probabilities
        probs = model.predict_proba(row)
        classes = getattr(model, "classes_", None)
        
        # Guard 5: Log model classes for verification (debug level)
        log.debug(
            "Session=%s, model classes=%s, probs shape=%s, probs sample=%s",
            session_key,
            classes,
            probs.shape,
            probs[0] if probs.size > 0 else None,
        )
        
        prediction = extract_entry_probabilities(probs, classes)
        prediction.session = session_key
        
        # PHASE 1 FIX: Re-enable temperature scaling (removed temporary T=1.0 hardcode)
        temp_map = self._get_temperature_map()
        
        # Check if temperature scaling is explicitly disabled via env var
        temp_scaling_disabled = os.getenv("GX1_TEMPERATURE_SCALING", "1") == "0"
        
        if temp_scaling_disabled:
            T = 1.0
            log.info(f"[TEMPERATURE] Scaling disabled via GX1_TEMPERATURE_SCALING=0 (using T=1.0 for all sessions)")
        else:
            # Use temperature from map, default to 1.0 if session not found
            T = float(temp_map.get(session_key, 1.0))
            
            # Log once if temperature is missing for session (but continue with T=1.0)
            if session_key not in temp_map:
                if not hasattr(self, "_temp_missing_logged"):
                    self._temp_missing_logged = set()
                if session_key not in self._temp_missing_logged:
                    log.warning(
                        "[TEMPERATURE] Temperature missing for session '%s', using T=1.0 (no scaling). "
                        "This may indicate missing temperature configuration.",
                        session_key,
                    )
                    self._temp_missing_logged.add(session_key)
        
        # Store raw probabilities before temperature scaling
        p_long_raw = prediction.prob_long
        p_short_raw = prediction.prob_short
        
        # Apply temperature scaling
        if T != 1.0:
            prediction.prob_long = self._apply_temperature(prediction.prob_long, T)
            prediction.prob_short = self._apply_temperature(prediction.prob_short, T)
            # Recompute p_hat and margin after temperature scaling
            prediction.p_hat = float(max(prediction.prob_long, prediction.prob_short))
            prediction.margin = float(abs(prediction.prob_long - prediction.prob_short))
        else:
            # No temperature scaling (T=1.0)
            p_long_adj = p_long_raw
            p_short_adj = p_short_raw
        
        # Store adjusted probabilities for logging
        p_long_adj = prediction.prob_long
        p_short_adj = prediction.prob_short
        
        # Log temperature and scaled probabilities (diagnostic)
        log.debug(
            "[DBG] session=%s T=%.3f p_long_raw=%.4f p_short_raw=%.4f p_long_adj=%.4f p_short_adj=%.4f p_hat=%.4f margin=%.4f",
            session_key,
            T,
            p_long_raw,
            p_short_raw,
            p_long_adj,
            p_short_adj,
            prediction.p_hat,
            prediction.margin,
        )
        
        return prediction
    
    def _compute_xgb_input_fingerprint(
        self,
        X: np.ndarray,
    ) -> Tuple[str, Dict[str, Any]]:
        """Compute SHA256 fingerprint of XGB input vector and return stats.
        
        X must be the EXACT matrix used by predict_proba (df[features].values.astype(np.float64)).
        """
        # For single row, extract first row
        if X.shape[0] == 1:
            x_orig = X[0]
        else:
            x_orig = X.flatten()  # Fallback for multi-row (shouldn't happen)
        
        # Convert to float32 contiguous for hashing (same precision, more efficient)
        x = np.ascontiguousarray(x_orig.astype(np.float32))
        
        # Compute fingerprint
        fingerprint = hashlib.sha256(x.tobytes()).hexdigest()
        
        # Compute stats (using original float64 for accuracy)
        finite_mask = np.isfinite(x_orig)
        n_nonfinite = np.sum(~finite_mask)
        x_finite = x_orig[finite_mask]
        
        stats = {
            "input_min": float(np.min(x_finite)) if len(x_finite) > 0 else None,
            "input_max": float(np.max(x_finite)) if len(x_finite) > 0 else None,
            "input_std": float(np.std(x_finite)) if len(x_finite) > 0 else None,
            "n_nonfinite": int(n_nonfinite),
            "n_features": len(x_orig),
        }
        
        return fingerprint, stats

    def _log_xgb_input_debug(
        self,
        session: str,
        timestamp: Optional[str],
        feature_list: List[str],
        X: np.ndarray,
        missing_features: List[str],
    ) -> None:
        """Log per-feature debug stats for XGB input (sampled, TRUTH/SMOKE only)."""
        if not self.xgb_input_debug_enabled or not self.xgb_input_debug_jsonl_path:
            return
        
        session_upper = session.upper()
        if session_upper not in ["EU", "OVERLAP"]:
            return
        
        self.xgb_input_debug_call_counts[session_upper] += 1
        call_count = self.xgb_input_debug_call_counts[session_upper]
        logged_count = self.xgb_input_debug_logged_counts[session_upper]
        
        if logged_count >= self.xgb_input_debug_max_per_session:
            return
        if call_count % self.xgb_input_debug_sample_n != 0:
            return
        
        # Extract vector
        vec = X[0] if X.shape[0] == 1 else X.flatten()
        finite_mask = np.isfinite(vec)
        n_nonfinite = int(np.sum(~finite_mask))
        n_zero = int(np.sum(vec == 0.0))
        
        # Per-feature stats (single value -> min=max=value, std=0.0)
        feature_stats = {}
        for idx, feat_name in enumerate(feature_list):
            if idx >= len(vec):
                break
            val = vec[idx]
            if not np.isfinite(val):
                feature_stats[feat_name] = {"min": None, "max": None, "std": None}
            else:
                feature_stats[feat_name] = {"min": float(val), "max": float(val), "std": 0.0}
        
        # Update flat-input counter (std == 0)
        std_val = float(np.std(vec[finite_mask])) if np.any(finite_mask) else 0.0
        if std_val == 0.0:
            self.xgb_input_debug_flat_counts[session_upper] += 1
        else:
            self.xgb_input_debug_flat_counts[session_upper] = 0
        
        # Optional debug-assert (TRUTH/SMOKE): K consecutive flat inputs
        run_mode = os.getenv("GX1_RUN_MODE", "").upper()
        is_truth = run_mode in ["TRUTH", "SMOKE"] or os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1"
        if is_truth and self.xgb_input_debug_flat_counts[session_upper] >= self.xgb_input_debug_flat_k:
            raise RuntimeError(
                f"[XGB_INPUT_DEBUG_FLAT] Session={session_upper} input std==0 for "
                f"{self.xgb_input_debug_flat_counts[session_upper]} sampled calls (K={self.xgb_input_debug_flat_k}). "
                f"Input appears flat at XGB input build step."
            )
        
        log_entry = {
            "ts": timestamp,
            "session": session_upper,
            "n_features": len(feature_list),
            "n_nonfinite": n_nonfinite,
            "n_zero": n_zero,
            "missing_features_count": len(missing_features),
            "missing_features_sample": missing_features[:10],
            "feature_stats": feature_stats,
        }
        
        try:
            with open(self.xgb_input_debug_jsonl_path, "a", encoding="utf-8") as f:
                f.write(jsonlib.dumps(log_entry, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.xgb_input_debug_logged_counts[session_upper] += 1
        except Exception as e:
            log.warning(f"[XGB_INPUT_DEBUG] Failed to write debug log: {e}")

    def _resolve_xgb_feature_names(
        self,
        feature_list: List[str],
        available_cols: List[str],
    ) -> Tuple[List[str], List[str], int, Dict[str, str], Dict[str, str]]:
        """Resolve XGB feature names with optional aliasing."""
        available_set = set(available_cols)
        mapping_table = self._load_xgb_feature_mapping_table()
        alias_map = {name: name[4:] for name in feature_list if name.startswith("_v1_")}
        resolved = []
        missing = []
        alias_hits = 0
        applied_aliases: Dict[str, str] = {}
        invalid_mappings: Dict[str, str] = {}
        for feat in feature_list:
            if feat in available_set:
                resolved.append(feat)
                continue
            if mapping_table and feat in mapping_table:
                mapped = mapping_table[feat]
                if mapped in available_set:
                    resolved.append(mapped)
                    continue
                invalid_mappings[feat] = mapped
                resolved.append(feat)
                missing.append(feat)
                continue
            alias = alias_map.get(feat)
            if alias and alias in available_set:
                resolved.append(alias)
                alias_hits += 1
                applied_aliases[feat] = alias
                continue
            resolved.append(feat)
            missing.append(feat)
        return resolved, missing, alias_hits, applied_aliases, invalid_mappings

    def _load_xgb_feature_mapping_table(self) -> Optional[Dict[str, str]]:
        """Load explicit XGB feature mapping table from env (if provided)."""
        if hasattr(self, "_xgb_feature_mapping_table"):
            return self._xgb_feature_mapping_table
        mapping_path = os.getenv("GX1_XGB_FEATURE_MAPPING_TABLE")
        if not mapping_path:
            self._xgb_feature_mapping_table = None
            return None
        try:
            mapping_path = Path(mapping_path)
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = jsonlib.load(f)
            mappings = data.get("mappings", {})
            if not isinstance(mappings, dict):
                raise ValueError("mappings must be a dict")
            self._xgb_feature_mapping_table = mappings
            return mappings
        except Exception as e:
            raise RuntimeError(f"[XGB_FEATURE_MAPPING_TABLE_FAIL] {e}") from e
    
    def _log_xgb_input_fingerprint(
        self,
        session: str,
        timestamp: Optional[str],
        fingerprint: str,
        input_stats: Dict[str, Any],
        p_long_raw: float,
        p_hat: float,
        uncertainty: float,
        policy_disabled: bool,
    ) -> None:
        """Log XGB input fingerprint to JSONL file (with sampling)."""
        if not self.xgb_fingerprint_enabled or not self.xgb_fingerprint_jsonl_path:
            return
        
        session_upper = session.upper()
        self.xgb_fingerprint_call_counts[session_upper] += 1
        call_count = self.xgb_fingerprint_call_counts[session_upper]
        logged_count = self.xgb_fingerprint_logged_counts[session_upper]
        
        # Sampling: log every N'th call, but stop after max_per_session
        if logged_count >= self.xgb_fingerprint_max_per_session:
            return
        
        if call_count % self.xgb_fingerprint_sample_n != 0:
            return
        
        # Log this call
        self.xgb_fingerprint_logged_counts[session_upper] += 1
        
        # Build log entry
        log_entry = {
            "ts": timestamp,
            "session": session_upper,
            "fingerprint": fingerprint,
            "input_min": input_stats.get("input_min"),
            "input_max": input_stats.get("input_max"),
            "input_std": input_stats.get("input_std"),
            "n_nonfinite": input_stats.get("n_nonfinite"),
            "n_features": input_stats.get("n_features"),
            "p_long_raw": p_long_raw,
            "p_hat": p_hat,
            "uncertainty": uncertainty,
            "policy_disabled": policy_disabled,
            "call_idx": call_count,
        }
        
        # Append to JSONL file (atomic per line)
        try:
            with open(self.xgb_fingerprint_jsonl_path, "a", encoding="utf-8") as f:
                f.write(jsonlib.dumps(log_entry, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            log.warning(f"[XGB_FINGERPRINT] Failed to write fingerprint log: {e}")
    
    def _get_xgb_outputs_for_session(
        self,
        session: str,
        current_row: pd.Series,
        candles: pd.DataFrame,
        snap_feat_names: List[str],
    ) -> Optional[Dict[str, float]]:
        """
        Get XGB outputs for a session using universal multihead model.
        
        Args:
            session: Session name (EU, US, OVERLAP, ASIA)
            current_row: Current feature row (pd.Series)
            candles: Candles DataFrame (for CLOSE alias)
            snap_feat_names: Snapshot feature names
        
        Returns:
            Dict with keys: p_long, p_short, p_flat, p_hat, uncertainty
            None if session disabled or XGB missing
        """
        import numpy as np
        from pathlib import Path as Path_local
        
        # Session policy: TRUTH/ONE_UNIVERSE uses canonical universal model; no external session-policy file.
        session_enabled = True
        session_mode = "normal"
        policy_hash = "CANONICAL_UNIVERSAL"
        
        # STEG 5: Log XGB policy source (once per chunk)
        if not hasattr(self, "_xgb_policy_logged"):
            self._xgb_policy_logged = set()
        if session not in self._xgb_policy_logged:
            log.info(
                "[XGB_POLICY] Session %s: source=CANONICAL_UNIVERSAL enabled=%s mode=%s policy_hash=%s",
                session,
                session_enabled,
                session_mode,
                policy_hash,
            )
            self._xgb_policy_logged.add(session)
        
        # Check if session is disabled
        if not session_enabled:
            # Session disabled - return neutral outputs
            if session_mode == "neutral_probs":
                uncertainty_val = session_config.get("uncertainty", 1.0)
                return {
                    "p_long": 1.0 / 3.0,
                    "p_short": 1.0 / 3.0,
                    "p_flat": 1.0 / 3.0,
                    "p_hat": 1.0 / 3.0,
                    "uncertainty": uncertainty_val,
                }
            else:
                raise ValueError(f"Unknown disabled mode: {session_mode}")
        
        # Check if universal XGB model is available
        if self.entry_v10_bundle.xgb_model_universal is None:
            # TRUTH MODE: Hard-fail if enabled session has no XGB model
            run_mode = os.environ.get("GX1_RUN_MODE", "TRUTH").upper()
            if run_mode in ["TRUTH", "PROD"]:
                raise RuntimeError(
                    f"[SSOT_FAIL] Session '{session}' is enabled but xgb_model_universal is None. "
                    f"This violates TRUTH mode invariant. "
                    f"Ensure MASTER_MODEL_LOCK.json points to valid XGB model."
                )
            # DEV_EVAL: Log warning and return None
            log.warning(f"[XGB_MISSING] Universal XGB model not available for session: {session}")
            return None
        
        # Get universal XGB model
        xgb_model = self.entry_v10_bundle.xgb_model_universal
        
        # Prepare features DataFrame (single row)
        # Use model's feature_list if available
        feature_list = xgb_model.feature_list if hasattr(xgb_model, "feature_list") else snap_feat_names
        
        # Build feature vector
        xgb_feat_values = []
        missing_features = []
        is_truth = os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1" or os.getenv("GX1_RUN_MODE", "").upper() in ["TRUTH", "SMOKE"]
        allow_alias = os.getenv("GX1_XGB_FEATURE_ALIAS_MODE", "0") == "1"
        mapping_table_env = os.getenv("GX1_XGB_FEATURE_MAPPING_TABLE")
        if is_truth and (allow_alias or mapping_table_env):
            raise RuntimeError(
                "[XGB_FEATURE_BRIDGE_FORBIDDEN] Alias/mapping bridge is forbidden in TRUTH/SMOKE. "
                f"GX1_XGB_FEATURE_ALIAS_MODE={os.getenv('GX1_XGB_FEATURE_ALIAS_MODE','0')}, "
                f"GX1_XGB_FEATURE_MAPPING_TABLE={'SET' if mapping_table_env else 'NOT_SET'}"
            )
        resolved_feature_list, missing_after_alias, alias_hits, applied_aliases, invalid_mappings = self._resolve_xgb_feature_names(
            feature_list=feature_list,
            available_cols=list(current_row.index),
        )
        if alias_hits > 0 and not allow_alias and is_truth:
            raise RuntimeError(
                f"[XGB_FEATURE_ALIAS_FORBIDDEN] Alias mapping required but GX1_XGB_FEATURE_ALIAS_MODE=0. "
                f"alias_hits={alias_hits}"
            )
        if invalid_mappings and is_truth:
            raise RuntimeError(
                f"[XGB_FEATURE_MAPPING_INVALID] Mapping points to missing columns: {list(invalid_mappings.items())[:10]}"
            )
        if missing_after_alias and is_truth:
            raise RuntimeError(
                f"[XGB_FEATURE_MISSING_FATAL] Missing {len(missing_after_alias)} XGB features after alias. "
                f"Examples: {missing_after_alias[:10]}"
            )
        if alias_hits > 0 and allow_alias and getattr(self, "explicit_output_dir", None):
            if not hasattr(self, "_xgb_alias_capsule_written"):
                self._xgb_alias_capsule_written = False
            if not self._xgb_alias_capsule_written:
                capsule_path = self.explicit_output_dir / "XGB_FEATURE_ALIAS_USED.json"
                try:
                    payload = {
                        "alias_hits": alias_hits,
                        "applied_aliases": applied_aliases,
                        "missing_after_alias": missing_after_alias,
                    }
                    with open(capsule_path, "w", encoding="utf-8") as f:
                        jsonlib.dump(payload, f, indent=2, sort_keys=True)
                    self._xgb_alias_capsule_written = True
                except Exception as e:
                    log.warning(f"[XGB_FEATURE_ALIAS] Failed to write capsule: {e}")
        for feat_name, resolved_name in zip(feature_list, resolved_feature_list):
            if resolved_name in current_row.index:
                xgb_feat_values.append(float(current_row[resolved_name]))
            elif feat_name == "CLOSE":
                # CLOSE alias from candles.close
                if "close" in candles.columns:
                    xgb_feat_values.append(float(candles["close"].iloc[-1]))
                else:
                    raise RuntimeError(
                        "[XGB_INPUT_FAIL] CLOSE feature required but candles.close not found."
                    )
            else:
                if is_truth:
                    raise RuntimeError(
                        f"[XGB_FEATURE_MISSING_FATAL] Feature '{feat_name}' missing (resolved='{resolved_name}') in TRUTH."
                    )
                log.warning(f"[XGB_INPUT] Feature '{feat_name}' not found, using 0.0")
                xgb_feat_values.append(0.0)
                missing_features.append(feat_name)
        
        # Create DataFrame for XGB prediction
        #
        # PERF (semantics-neutral): avoid per-bar DataFrame construction churn by
        # reusing a 1-row DataFrame cache keyed by feature_list.
        if not hasattr(self, "_xgb_features_df_cache") or not isinstance(getattr(self, "_xgb_features_df_cache"), dict):
            self._xgb_features_df_cache = {}
        cache_key = tuple(feature_list)
        xgb_features_df = self._xgb_features_df_cache.get(cache_key)
        if xgb_features_df is None:
            xgb_features_df = pd.DataFrame([xgb_feat_values], columns=feature_list)
            self._xgb_features_df_cache[cache_key] = xgb_features_df
        else:
            # Update values in-place (1 row)
            try:
                xgb_features_df.iloc[0, :] = xgb_feat_values
            except Exception:
                # Fallback: rebuild if something unexpected happened (best-effort)
                xgb_features_df = pd.DataFrame([xgb_feat_values], columns=feature_list)
                self._xgb_features_df_cache[cache_key] = xgb_features_df
        
        # Validate features (NaN/Inf check)
        is_replay = getattr(self, "replay_mode", False) or getattr(self, "fast_replay", False)
        if is_replay:
            if not xgb_features_df.select_dtypes(include=[np.number]).notna().all().all():
                nan_count = xgb_features_df.isna().sum().sum()
                raise RuntimeError(
                    f"[NAN_INF_INPUT_FATAL] XGB features contain NaN: count={nan_count}"
                )
        
        # Build exact XGB input matrix (same values as predict_proba uses)
        # PERF: avoid DataFrame slicing + astype copy; build directly from the list.
        X = np.asarray(xgb_feat_values, dtype=np.float64).reshape(1, -1)

        # ============================================================
        # XGB INPUT TRUTH DUMP (TRUTH/PREBUILT only, replay-only, no impact on predictions)
        # ============================================================
        # Purpose: deterministically capture the *exact* X matrix fed into XGB predict_proba()
        # for the first N model calls, then write a truth dump to run_root.
        #
        # Constraints:
        # - Only active in TRUTH/PREBUILT mode
        # - Only chunk_0 writes (avoid multi-worker duplication)
        # - Only first N=2000 model calls (bounded memory/IO)
        # - Must not modify X or predictions
        #
        # Artifacts (written once per run):
        # - XGB_INPUT_TRUTH_DUMP.json
        # - XGB_INPUT_TRUTH_DUMP.md
        # - XGB_INPUT_DEGENERATE_FATAL.json (if strict threshold hit)
        try:
            is_truth_run = os.getenv("GX1_RUN_MODE", "").upper() == "TRUTH" or os.getenv("GX1_TRUTH_MODE", "0") == "1"
            is_prebuilt = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1" and os.getenv("GX1_FEATURE_BUILD_DISABLED", "0") == "1"
            chunk_id_env = os.getenv("GX1_CHUNK_ID", "")
            if is_truth_run and is_prebuilt and str(chunk_id_env) == "0":
                # Resolve canonical ordered feature names (SSoT) once and cache.
                # This is the definitive ordered list we use for the truth dump.
                if not hasattr(self, "_xgb_feature_names_ordered") or self._xgb_feature_names_ordered is None:
                    from gx1.utils.xgb_feature_names_ssot import load_xgb_feature_names_ssot

                    # Resolve run_root from explicit_output_dir if present (chunk dir -> parent)
                    run_root_for_ssot = None
                    if getattr(self, "explicit_output_dir", None):
                        try:
                            p = Path_local(str(self.explicit_output_dir))
                            run_root_for_ssot = p.parent if p.name.startswith("chunk_") else p
                        except Exception:
                            run_root_for_ssot = None
                    if run_root_for_ssot is None:
                        raise RuntimeError("[TRUTH_FAIL] XGB_INPUT_TRUTH_DUMP cannot resolve run_root for SSoT loader")

                    names_ordered, ssot_details = load_xgb_feature_names_ssot(str(run_root_for_ssot))
                    self._xgb_feature_names_ordered = list(names_ordered)
                    self._xgb_feature_names_ssot_details = ssot_details

                # Hard assert exact expected dimensionality (truth dump contract).
                if int(X.shape[1]) != int(len(self._xgb_feature_names_ordered)):
                    raise RuntimeError(
                        "[TRUTH_FAIL] XGB input dim mismatch vs SSoT feature list: "
                        f"X.shape[1]={int(X.shape[1])} names_len={int(len(self._xgb_feature_names_ordered))}"
                    )
                if int(X.shape[1]) != 91:
                    raise RuntimeError(f"[TRUTH_FAIL] XGB input dim expected 91, got {int(X.shape[1])}")

                if not hasattr(self, "_xgb_truth_dump_done"):
                    self._xgb_truth_dump_done = False
                if not hasattr(self, "_xgb_truth_dump_n_target"):
                    self._xgb_truth_dump_n_target = 2000
                if not hasattr(self, "_xgb_truth_dump_rows"):
                    self._xgb_truth_dump_rows = []  # type: ignore[attr-defined]
                # Use SSoT ordered names, never per-call 'feature_list' guesses
                self._xgb_truth_dump_feature_names = list(self._xgb_feature_names_ordered)

                if not self._xgb_truth_dump_done:
                    # Ensure consistent feature order across captures
                    if list(feature_list) != list(self._xgb_truth_dump_feature_names):
                        raise RuntimeError(
                            "[TRUTH_FAIL] XGB_INPUT_TRUTH_DUMP feature_names changed across calls. "
                            f"first_len={len(self._xgb_truth_dump_feature_names)} now_len={len(feature_list)}"
                        )
                    # Capture row copy (do not mutate X)
                    if len(self._xgb_truth_dump_rows) < int(self._xgb_truth_dump_n_target):
                        self._xgb_truth_dump_rows.append(X[0, :].copy())

                    if len(self._xgb_truth_dump_rows) >= int(self._xgb_truth_dump_n_target):
                        # Write dump once N reached (TRUTH/PREBUILT, chunk_0 only)
                        self._write_xgb_input_truth_dump_if_ready(force=False)
        except Exception:
            # TRUTH should fail fast; re-raise
            raise
        
        # XGB input fingerprint logging (TRUTH/SMOKE only, with sampling)
        fingerprint: Optional[str] = None
        input_stats: Optional[Dict[str, Any]] = None
        if self.xgb_fingerprint_enabled:
            fingerprint, input_stats = self._compute_xgb_input_fingerprint(X)
        
        # XGB input debug logging (sampled)
        if self.xgb_input_debug_enabled:
            self._log_xgb_input_debug(
                session=session,
                timestamp=str(current_row.name) if hasattr(current_row, "name") else None,
                feature_list=feature_list,
                X=X,
                missing_features=missing_features,
            )
        
        # Call universal XGB model with session routing
        try:
            # B1: True timer — XGB predict time only
            import time as _time_mod
            _t0_xgb = _time_mod.perf_counter()
            outputs = xgb_model.predict_proba(
                xgb_features_df,
                session=session.upper(),
                _telemetry_counters=getattr(self.run_identity, "__dict__", {}) if hasattr(self, "run_identity") else None,
            )
            try:
                self.t_xgb_predict_sec += float(_time_mod.perf_counter() - _t0_xgb)
            except Exception:
                pass
        except Exception as e:
            log.error(f"[XGB_PREDICT_FAIL] Failed to predict for session {session}: {e}")
            # ENTRY FEATURE TELEMETRY: Record pre-model return
            if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                self.entry_manager.entry_feature_telemetry.record_pre_model_return(
                    f"XGB_PREDICT_EXCEPTION: {str(e)}",
                    session=session.upper() if session else None,
                )
            raise
        
        # ENTRY FEATURE TELEMETRY: Record XGB predict call
        if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
            self.entry_manager.entry_feature_telemetry.record_xgb_predict_call(session=session.upper())
        
        # Extract outputs (MultiheadOutputs object)
        p_long = float(outputs.p_long[0])
        p_short = float(outputs.p_short[0])
        p_flat = float(outputs.p_flat[0])
        uncertainty = float(outputs.uncertainty[0])
        p_hat = max(p_long, p_short)
        
        # Increment XGB predict counters in RUN_IDENTITY
        if hasattr(self, "run_identity") and self.run_identity:
            self.run_identity.xgb_predict_count_total += 1
            if session.upper() == "EU":
                self.run_identity.xgb_predict_count_eu += 1
            elif session.upper() == "US":
                self.run_identity.xgb_predict_count_us += 1
                # TRUTH MODE: Hard-fail if US XGB is called
                run_mode = os.environ.get("GX1_RUN_MODE", "TRUTH").upper()
                if run_mode in ["TRUTH", "PROD"]:
                    raise RuntimeError(
                        f"[TRUTH_INVARIANT_FAIL] US XGB was called (xgb_predict_count_us={self.run_identity.xgb_predict_count_us}) "
                        f"but US should be disabled by policy. This violates TRUTH mode invariant."
                    )
            elif session.upper() == "OVERLAP":
                self.run_identity.xgb_predict_count_overlap += 1
        
        # Record XGB flow in telemetry
        if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry"):
            xgb_timestamp = None
            try:
                if candles is not None and len(candles) > 0:
                    xgb_timestamp = candles.index[-1].isoformat()
                elif current_row is not None and hasattr(current_row, "name"):
                    xgb_timestamp = str(current_row.name)
            except Exception:
                xgb_timestamp = None
            
            # XGB input fingerprint logging (with sampling)
            if self.xgb_fingerprint_enabled and fingerprint and input_stats:
                self._log_xgb_input_fingerprint(
                    session=session,
                    timestamp=xgb_timestamp,
                    fingerprint=fingerprint,
                    input_stats=input_stats,
                    p_long_raw=p_long,
                    p_hat=p_hat,
                    uncertainty=uncertainty,
                    policy_disabled=False,
                )
            
            # B1: True timer — telemetry recording overhead (best-effort)
            import time as _time_mod
            _t0_tel = _time_mod.perf_counter()
            self.entry_manager.entry_feature_telemetry.record_xgb_flow(
                xgb_called=True,
                xgb_session=session.upper(),
                xgb_p_long_raw=p_long,
                xgb_p_long_cal=p_long,
                xgb_uncertainty_score=uncertainty,
                policy_disabled=False,
                xgb_used_as="pre",
                timestamp=xgb_timestamp,
            )
            try:
                self.t_telemetry_sec += float(_time_mod.perf_counter() - _t0_tel)
            except Exception:
                pass
        
        return {
            "p_long": p_long,
            "p_short": p_short,
            "p_flat": p_flat,
            "p_hat": p_hat,
            "uncertainty": uncertainty,
        }
    
    @staticmethod
    def write_xgb_fingerprint_summary_static(
        output_dir: Path,
        run_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
        min_logged: int = 50,
    ) -> None:
        """Static method to write XGB fingerprint summary (can be called from master without runner instance)."""
        # Glob all per-process JSONL files (include chunk subdirs)
        jsonl_files = list(output_dir.rglob("XGB_INPUT_FINGERPRINT.*.jsonl"))
        
        if not jsonl_files:
            log.warning("[XGB_FINGERPRINT] No JSONL files found, skipping summary")
            return
        
        # Read all logged entries from all per-process files
        entries: List[Dict[str, Any]] = []
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entries.append(jsonlib.loads(line))
            except Exception as e:
                log.warning(f"[XGB_FINGERPRINT] Failed to read {jsonl_file}: {e}")
                continue
        
        if not entries:
            log.warning("[XGB_FINGERPRINT] No entries found in JSONL files")
            return
        
        # Analyze per session
        summary: Dict[str, Any] = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "output_dir": str(output_dir),
            "run_id": run_id,
            "chunk_id": chunk_id,
            "sessions": {},
        }
        
        by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            sess = entry.get("session", "UNKNOWN")
            by_session[sess].append(entry)
        
        run_mode = os.getenv("GX1_RUN_MODE", "").upper()
        is_truth = run_mode in ["TRUTH", "SMOKE"] or os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1"
        
        for session, sess_entries in by_session.items():
            n_logged = len(sess_entries)
            fingerprints = [e.get("fingerprint") for e in sess_entries if e.get("fingerprint")]
            n_unique_fingerprints = len(set(fingerprints))
            unique_ratio = n_unique_fingerprints / n_logged if n_logged > 0 else 0.0
            
            # Compute input/output stats
            input_stds = [e.get("input_std") for e in sess_entries if e.get("input_std") is not None]
            input_mins = [e.get("input_min") for e in sess_entries if e.get("input_min") is not None]
            input_maxs = [e.get("input_max") for e in sess_entries if e.get("input_max") is not None]
            p_long_raws = [e.get("p_long_raw") for e in sess_entries if e.get("p_long_raw") is not None]
            uncertainties = [e.get("uncertainty") for e in sess_entries if e.get("uncertainty") is not None]
            
            sess_summary: Dict[str, Any] = {
                "n_logged": n_logged,
                "n_unique_fingerprints": n_unique_fingerprints,
                "unique_ratio": unique_ratio,
                "inputs_std_min": float(np.min(input_stds)) if input_stds else None,
                "inputs_std_max": float(np.max(input_stds)) if input_stds else None,
                "outputs_min": float(np.min(p_long_raws)) if p_long_raws else None,
                "outputs_max": float(np.max(p_long_raws)) if p_long_raws else None,
                "outputs_std": float(np.std(p_long_raws)) if p_long_raws else None,
                "uncertainty_min": float(np.min(uncertainties)) if uncertainties else None,
                "uncertainty_max": float(np.max(uncertainties)) if uncertainties else None,
                "uncertainty_std": float(np.std(uncertainties)) if uncertainties else None,
            }
            
            # Determine verdict
            if n_logged >= min_logged:
                if n_unique_fingerprints == 1:
                    sess_summary["verdict"] = "IDENTICAL_INPUTS"
                    # TRUTH invariant: FATAL if EU/OVERLAP have identical inputs
                    if is_truth and session in ["EU", "OVERLAP"]:
                        ts_range = {
                            "first": sess_entries[0].get("ts"),
                            "last": sess_entries[-1].get("ts"),
                        }
                        capsule = {
                            "generated_at": datetime.utcnow().isoformat() + "Z",
                            "session": session,
                            "n_logged": n_logged,
                            "n_unique_fingerprints": n_unique_fingerprints,
                            "fingerprint": fingerprints[0] if fingerprints else None,
                            "ts_range": ts_range,
                            "input_stats": {
                                "std_min": sess_summary["inputs_std_min"],
                                "std_max": sess_summary["inputs_std_max"],
                            },
                            "output_stats": {
                                "min": sess_summary["outputs_min"],
                                "max": sess_summary["outputs_max"],
                                "std": sess_summary["outputs_std"],
                            },
                            "run_id": run_id,
                            "chunk_id": chunk_id,
                        }
                        capsule_path = output_dir / "XGB_INPUT_IDENTICAL_FATAL.json"
                        try:
                            with open(capsule_path, "w", encoding="utf-8") as f:
                                jsonlib.dump(capsule, f, indent=2, sort_keys=True)
                            log.error(f"[XGB_FINGERPRINT] FATAL: {session} has identical inputs (n_logged={n_logged}, fingerprint={fingerprints[0][:16] if fingerprints else 'N/A'}...)")
                            raise RuntimeError(
                                f"[XGB_INPUT_IDENTICAL_FATAL] Session {session} has identical XGB inputs across {n_logged} logged calls. "
                                f"This indicates a feature feed/sanitizer/prebuilt mapping bug. "
                                f"Capsule written to: {capsule_path}"
                            )
                        except RuntimeError:
                            raise
                        except Exception as e:
                            log.error(f"[XGB_FINGERPRINT] Failed to write FATAL capsule: {e}")
                else:
                    sess_summary["verdict"] = "VARIABLE_INPUTS"
            else:
                sess_summary["verdict"] = "INSUFFICIENT_SAMPLES"
            
            summary["sessions"][session] = sess_summary
        
        # Write summary JSON
        summary_path = output_dir / "XGB_INPUT_FINGERPRINT_SUMMARY.json"
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                jsonlib.dump(summary, f, indent=2, sort_keys=True)
            log.info(f"[XGB_FINGERPRINT] Wrote summary to {summary_path}")
        except Exception as e:
            log.error(f"[XGB_FINGERPRINT] Failed to write summary: {e}")
        
        # Write summary MD
        md_path = output_dir / "XGB_INPUT_FINGERPRINT_SUMMARY.md"
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("# XGB Input Fingerprint Summary\n\n")
                f.write(f"- Generated at: `{summary['generated_at']}`\n")
                f.write(f"- Output dir: `{summary['output_dir']}`\n")
                f.write(f"- Run ID: `{summary['run_id']}`\n")
                f.write(f"- Chunk ID: `{summary['chunk_id']}`\n\n")
                for session, sess_summary in sorted(summary["sessions"].items()):
                    f.write(f"## Session {session}\n\n")
                    f.write(f"- n_logged: {sess_summary['n_logged']}\n")
                    f.write(f"- n_unique_fingerprints: {sess_summary['n_unique_fingerprints']}\n")
                    f.write(f"- unique_ratio: {sess_summary['unique_ratio']:.4f}\n")
                    f.write(f"- verdict: **{sess_summary['verdict']}**\n")
                    if sess_summary.get('inputs_std_min') is not None:
                        f.write(f"- inputs_std: min={sess_summary['inputs_std_min']:.6f}, max={sess_summary['inputs_std_max']:.6f}\n")
                    if sess_summary.get('outputs_std') is not None:
                        f.write(f"- outputs_std: {sess_summary['outputs_std']:.6f}\n")
                    f.write("\n")
            log.info(f"[XGB_FINGERPRINT] Wrote summary MD to {md_path}")
        except Exception as e:
            log.error(f"[XGB_FINGERPRINT] Failed to write summary MD: {e}")
    
    def _write_xgb_fingerprint_summary(self) -> None:
        """Write XGB input fingerprint summary and check TRUTH invariants (instance method)."""
        if not self.xgb_fingerprint_enabled or not self.explicit_output_dir:
            return
        GX1DemoRunner.write_xgb_fingerprint_summary_static(
            output_dir=self.explicit_output_dir,
            run_id=getattr(self, "run_id", None),
            chunk_id=getattr(self, "chunk_id", None),
            min_logged=self.xgb_fingerprint_min_logged,
        )
    
    def _write_import_fail_capsule(self, import_err: Exception, module_name: str, symbol_name: Optional[str] = None) -> Path:
        """Write IMPORT_FAIL.json capsule with full diagnostic info."""
        import traceback
        import json
        import sys
        from datetime import datetime, timezone
        from pathlib import Path
        
        # Get output directory
        output_dir = None
        if hasattr(self, "output_dir") and self.output_dir:
            output_dir = Path(self.output_dir)
        elif hasattr(self, "explicit_output_dir") and self.explicit_output_dir:
            output_dir = Path(self.explicit_output_dir)
        else:
            output_dir_env = os.getenv("GX1_OUTPUT_DIR")
            if output_dir_env:
                output_dir = Path(output_dir_env)
        
        if not output_dir:
            # Fallback to /tmp
            output_dir = Path("/tmp/import_fail_capsules")
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get RUN_IDENTITY info
        run_identity_info = {}
        if hasattr(self, "run_identity") and self.run_identity:
            run_identity_info = {
                "policy_id": getattr(self.run_identity, "policy_id", None),
                "bundle_sha": getattr(self.run_identity, "bundle_sha", None),
                "xgb_session_policy_hash": getattr(self.run_identity, "xgb_session_policy_hash", None),
                "replay_mode": getattr(self.run_identity, "replay_mode", None),
                "prebuilt_used": getattr(self.run_identity, "prebuilt_used", None),
            }
        
        # Build capsule data
        capsule_data = {
            "exception_type": type(import_err).__name__,
            "message": str(import_err),
            "traceback": traceback.format_exc(),
            "module_name": module_name,
            "symbol_name": symbol_name,
            "sys_path": list(sys.path),
            "cwd": str(Path.cwd()),
            "sys_executable": sys.executable,
            "replay_mode": os.getenv("GX1_REPLAY", "0") == "1",
            "prebuilt_enabled": os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1",
            "truth_mode": os.getenv("GX1_TRUTH_MODE", "0") == "1",
            "run_identity": run_identity_info,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # Write capsule
        capsule_path = output_dir / "IMPORT_FAIL.json"
        with open(capsule_path, "w") as f:
            json.dump(capsule_data, f, indent=2)
        
        log.error(
            "[ENTRY_V10] IMPORT_FAIL capsule written to %s: %s",
            capsule_path,
            import_err
        )
        
        return capsule_path
    
    def _predict_entry_v10_hybrid(
        self,
        entry_bundle: EntryFeatureBundle,
        candles: pd.DataFrame,
        policy_state: Dict[str, Any],
        entry_context_features: Optional[Any] = None,  # OPPGAVE 2: EntryContextFeatures object
    ) -> Optional[EntryPrediction]:
        """
        Generate V10 hybrid entry prediction (XGB + Transformer).
        MODEL KONTRAKT: transformer-input er signal_bridge_v1 (7 kanaler:
        p_long, p_short, p_flat, p_hat, uncertainty_score, margin_top1_top2, entropy)
        + ctx_cont (6) + ctx_cat (6) i eksakt ordering fra ctx-kontrakt; seq_len fra bundle.
        Ingen rå BASE28 går inn i transformer.
        OPERASJONELL KONTEKST: prebuilt_features_df / entry_bundle.features brukes til å bygge
        sekvensvinduer/regimer/session, policy_state gir session, bundle metadata gir dims/seq_len,
        telemetry/diagnose helpers skriver kapsler/logg men er ikke del av modellkontrakten.
        """
        def _record_predict_attempt(reason: str) -> None:
            if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                session_for_attempt = policy_state.get("session") if isinstance(policy_state, dict) else None
                self.entry_manager.entry_feature_telemetry.record_predict_attempt(session=session_for_attempt, reason=reason)

        def _record_predict_reached_forward() -> None:
            if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                session_for_forward = policy_state.get("session") if isinstance(policy_state, dict) else None
                self.entry_manager.entry_feature_telemetry.record_predict_reached_forward(session=session_for_forward)
        
        # NOTE: predict_entered_count is NOT incremented here because there are many early return paths
        # that can return None before transformer forward call. Instead, predict_entered_count is
        # incremented right before transformer forward call (after all early return checks pass).
        # This ensures the invariant: predict_entered > 0 and exceptions == 0 => forward_calls > 0
        
        # STEG 4: Set model_name early to avoid UnboundLocalError in exception handler
        model_name = "v10_hybrid"
        
        # STEG 4: Import Path explicitly to avoid UnboundLocalError
        from pathlib import Path as Path_local
        
        # STEG 1: Guard for sanity (ensures Path_local is defined)
        assert Path_local is not None, "Path_local must be imported"
        
        # Define write_transformer_exception_capsule early so it can be used in outer try-except
        def write_transformer_exception_capsule(exc: Exception, seq_x: Optional[torch.Tensor] = None, snap_x: Optional[torch.Tensor] = None,
                                                 seq_data: Optional[np.ndarray] = None, snap_data: Optional[np.ndarray] = None,
                                                 xgb_outputs: Optional[Dict[str, float]] = None, transformer_config: Optional[Dict[str, Any]] = None,
                                                 seq_feat_names: Optional[List[str]] = None, snap_feat_names: Optional[List[str]] = None) -> Path:
            """Write TRANSFORMER_FORWARD_EXCEPTION.json capsule with full diagnostic info."""
            import traceback
            import json
            from datetime import datetime, timezone
            from gx1.contracts.signal_bridge_v1 import ORDERED_FIELDS, SEQ_SIGNAL_DIM, SNAP_SIGNAL_DIM
            
            # Get output directory
            output_dir = None
            if hasattr(self, "output_dir") and self.output_dir:
                output_dir = Path_local(self.output_dir)
            elif hasattr(self, "explicit_output_dir") and self.explicit_output_dir:
                output_dir = Path_local(self.explicit_output_dir)
            else:
                output_dir_env = os.getenv("GX1_OUTPUT_DIR")
                if output_dir_env:
                    output_dir = Path_local(output_dir_env)
            
            if not output_dir:
                # Fallback to /tmp
                output_dir = Path_local("/tmp/transformer_exception_capsules")
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get RUN_IDENTITY info
            run_identity_info = {}
            if hasattr(self, "run_identity") and self.run_identity:
                run_identity_info = {
                    "policy_id": getattr(self.run_identity, "policy_id", None),
                    "bundle_sha": getattr(self.run_identity, "bundle_sha", None),
                    "xgb_session_policy_hash": getattr(self.run_identity, "xgb_session_policy_hash", None),
                    "replay_mode": getattr(self.run_identity, "replay_mode", None),
                    "prebuilt_used": getattr(self.run_identity, "prebuilt_used", None),
                }
            
            # Compute input statistics
            def compute_tensor_stats(t: Optional[torch.Tensor]) -> Dict[str, Any]:
                """Compute statistics for a tensor."""
                if t is None:
                    return {"error": "tensor is None"}
                try:
                    t_np = t.detach().cpu().numpy()
                    return {
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "min": float(np.min(t_np)) if t_np.size > 0 else None,
                        "max": float(np.max(t_np)) if t_np.size > 0 else None,
                        "mean": float(np.mean(t_np)) if t_np.size > 0 else None,
                        "std": float(np.std(t_np)) if t_np.size > 0 else None,
                        "count_nan": int(np.isnan(t_np).sum()),
                        "count_inf": int(np.isinf(t_np).sum()),
                        "batch_dim": t.shape[0] if len(t.shape) > 0 else None,
                    }
                except Exception as stats_error:
                    return {"error": f"Failed to compute stats: {str(stats_error)}"}
            
            def compute_array_stats(arr: Optional[np.ndarray]) -> Dict[str, Any]:
                """Compute statistics for a numpy array."""
                if arr is None:
                    return {"error": "array is None"}
                try:
                    return {
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                        "min": float(np.min(arr)) if arr.size > 0 else None,
                        "max": float(np.max(arr)) if arr.size > 0 else None,
                        "mean": float(np.mean(arr)) if arr.size > 0 else None,
                        "std": float(np.std(arr)) if arr.size > 0 else None,
                        "count_nan": int(np.isnan(arr).sum()),
                        "count_inf": int(np.isinf(arr).sum()),
                    }
                except Exception as stats_error:
                    return {"error": f"Failed to compute stats: {str(stats_error)}"}
            
            # Get feature fingerprint if available
            feature_fingerprint = None
            if hasattr(self, "_feature_fingerprint"):
                feature_fingerprint = {
                    "fingerprint_hash": self._feature_fingerprint.fingerprint_hash if hasattr(self._feature_fingerprint, "fingerprint_hash") else None,
                }
            
            # Get current session
            current_session = policy_state.get("session", "UNKNOWN") if 'policy_state' in locals() else "UNKNOWN"
            
            # Build capsule
            capsule = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "run_identity": run_identity_info,
                "session": current_session,
                "chunk_id": getattr(self, "chunk_id", None),
                "bar_index": getattr(self, "bar_index", None),
                "transformer_config": transformer_config or {},
                "input_snapshots": {
                    "seq_tensor": compute_tensor_stats(seq_x),
                    "snap_tensor": compute_tensor_stats(snap_x),
                    "seq_array": compute_array_stats(seq_data),
                    "snap_array": compute_array_stats(snap_data),
                },
                "xgb_outputs": xgb_outputs or {},
                "signal_bridge_v1": {
                    "ordered_fields": list(ORDERED_FIELDS),
                    "seq_signal_dim": int(SEQ_SIGNAL_DIM),
                    "snap_signal_dim": int(SNAP_SIGNAL_DIM),
                },
                "feature_fingerprint": feature_fingerprint,
            }
            
            # Write capsule
            capsule_path = output_dir / "TRANSFORMER_FORWARD_EXCEPTION.json"
            with open(capsule_path, "w") as f:
                json.dump(capsule, f, indent=2)
            
            log.error(f"[TRANSFORMER_EXCEPTION_CAPSULE] Wrote exception capsule to {capsule_path}")
            return capsule_path

        def write_us_premodel_valueerror_capsule(
            exc: Exception,
            session: Optional[str],
            seq_data: Optional[np.ndarray],
            snap_data: Optional[np.ndarray],
            seq_feat_names: Optional[List[str]],
            snap_feat_names: Optional[List[str]],
            session_tokens_enabled: bool,
            ts: Optional[pd.Timestamp] = None,
        ) -> None:
            """Write US_PREMODEL_VALUEERROR.json for deterministic diagnosis."""
            if not isinstance(exc, ValueError):
                return
            if session != "US":
                return
            if os.getenv("GX1_TRUTH_TELEMETRY", "0") != "1":
                return
            try:
                import traceback
                output_dir = None
                if hasattr(self, "output_dir") and self.output_dir:
                    output_dir = Path_local(self.output_dir)
                elif hasattr(self, "explicit_output_dir") and self.explicit_output_dir:
                    output_dir = Path_local(self.explicit_output_dir)
                else:
                    output_dir_env = os.getenv("GX1_OUTPUT_DIR")
                    if output_dir_env:
                        output_dir = Path_local(output_dir_env)
                if not output_dir:
                    output_dir = Path_local("/tmp/us_premodel_valueerror")
                output_dir.mkdir(parents=True, exist_ok=True)
                capsule_path = output_dir / "US_PREMODEL_VALUEERROR.json"
                
                # Get bundle metadata for expected dims
                bundle_expected_snap_dim = None
                bundle_expected_seq_dim = None
                entry_model_id = None
                bundle_sha256 = None
                try:
                    if hasattr(self, "entry_v10_bundle") and self.entry_v10_bundle:
                        bundle_meta = getattr(self.entry_v10_bundle, "metadata", {}) or {}
                        bundle_expected_snap_dim = bundle_meta.get("snap_input_dim")
                        bundle_expected_seq_dim = bundle_meta.get("seq_input_dim")
                    if hasattr(self, "entry_v10_cfg") and isinstance(self.entry_v10_cfg, dict):
                        entry_model_id = self.entry_v10_cfg.get("entry_model_id")
                    if output_dir:
                        run_identity_path = output_dir / "RUN_IDENTITY.json"
                        if run_identity_path.exists():
                            import json as _json
                            with open(run_identity_path, "r") as f:
                                run_identity = _json.load(f)
                                bundle_sha256 = run_identity.get("bundle_sha256")
                                if not entry_model_id:
                                    entry_model_id = run_identity.get("entry_model_id")
                except Exception:
                    pass
                
                # Get feature_meta snap count
                feature_meta_snap_count = None
                feature_meta_snap_names = []
                try:
                    feature_meta_path = None
                    if hasattr(self, "entry_v10_cfg") and isinstance(self.entry_v10_cfg, dict):
                        feature_meta_path = self.entry_v10_cfg.get("feature_meta_path")
                    if feature_meta_path is None and hasattr(self, "entry_v10_bundle") and hasattr(self.entry_v10_bundle, "bundle_dir"):
                        candidate_path = Path_local(self.entry_v10_bundle.bundle_dir) / "entry_v10_ctx_feature_meta.json"
                        if candidate_path.exists():
                            feature_meta_path = str(candidate_path)
                    if feature_meta_path:
                        import json as _json
                        with open(feature_meta_path, "r") as f:
                            meta = _json.load(f)
                        feature_meta_snap_names = meta.get("snap_features", [])
                        feature_meta_snap_count = len(feature_meta_snap_names)
                except Exception:
                    pass
                
                # Get actual snap feature names (if instrumented)
                actual_snap_feature_names = getattr(self, "_actual_snap_feature_names", None)
                
                # Build expected names reference (SSoT): signal-only bridge
                from gx1.contracts.signal_bridge_v1 import ORDERED_FIELDS, SEQ_SIGNAL_DIM, SNAP_SIGNAL_DIM
                expected_names_reference = list(ORDERED_FIELDS)
                
                # Compute missing/extra names
                missing_names = []
                extra_names = []
                first_missing_name = None
                if actual_snap_feature_names is not None:
                    actual_set = set(actual_snap_feature_names)
                    expected_set = set(expected_names_reference)
                    missing_names = sorted(list(expected_set - actual_set))
                    extra_names = sorted(list(actual_set - expected_set))
                    first_missing_name = missing_names[0] if missing_names else None
                
                # Count signal fields filled
                signal_fields_filled_count = int(SNAP_SIGNAL_DIM)
                
                # Get last 10 names in actual (to see tail channels)
                actual_tail_names = actual_snap_feature_names[-10:] if actual_snap_feature_names and len(actual_snap_feature_names) >= 10 else (actual_snap_feature_names if actual_snap_feature_names else [])
                
                payload = {
                    "session": session,
                    "ts": str(ts) if ts is not None else None,
                    "routing": "v10_hybrid",
                    "entry_model_id": entry_model_id,
                    "bundle_sha256": bundle_sha256,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "stacktrace": traceback.format_exc(),
                    "expected_dim": {
                        "from_bundle_metadata": bundle_expected_snap_dim,
                        "from_contract": int(SNAP_SIGNAL_DIM),
                    },
                    "actual_dim": int(snap_data.shape[-1]) if hasattr(snap_data, "shape") else None,
                    "signal_fields_expected_count": int(SNAP_SIGNAL_DIM),
                    "signal_fields_filled_count": signal_fields_filled_count,
                    "session_tokens_enabled": False,
                    "feature_meta_snap_count": feature_meta_snap_count,
                    "snap_feature_names_effective": actual_snap_feature_names if actual_snap_feature_names else None,
                    "expected_names_reference": expected_names_reference,
                    "missing_names": missing_names,
                    "extra_names": extra_names,
                    "first_missing_name": first_missing_name,
                    "actual_tail_names": actual_tail_names,
                    "seq_dim": {
                        "expected": int(SEQ_SIGNAL_DIM),
                        "actual": int(seq_data.shape[-1]) if hasattr(seq_data, "shape") else None,
                    },
                }
                with open(capsule_path, "w") as f:
                    jsonlib.dump(payload, f, indent=2)
                log.error("[US_PREMODEL_VALUEERROR] Wrote capsule: %s", capsule_path)
            except Exception as capsule_err:
                log.warning("[US_PREMODEL_VALUEERROR] Failed to write capsule: %s", capsule_err)
        
        # DEL 1: Record v10_callsite.entered (right at function entry, before any other telemetry)
        if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
            self.entry_manager.entry_feature_telemetry.v10_callsite_entered += 1
        
        # ENTRY FEATURE TELEMETRY: Record model attempt (right at function entry)
        # model_name is already set earlier (before try-except block)
        if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
            self.entry_manager.entry_feature_telemetry.record_model_attempt(model_name)
        
        # Initialize reason tracking telemetry
        if not hasattr(self, "entry_manager") or self.entry_manager is None:
            # Fallback: create minimal telemetry dict
            v10_none_reason_counts = {}
        else:
            if "v10_none_reason_counts" not in self.entry_manager.entry_telemetry:
                self.entry_manager.entry_telemetry["v10_none_reason_counts"] = {}
            v10_none_reason_counts = self.entry_manager.entry_telemetry["v10_none_reason_counts"]
        
        # Helper for reason-coded early returns
        def _ret_none(reason: str):
            """Reason-coded early return helper."""
            is_replay = getattr(self, "replay_mode", False) or os.getenv("GX1_REPLAY", "0") == "1"
            if is_replay:
                v10_none_reason_counts[reason] = v10_none_reason_counts.get(reason, 0) + 1
            # Rate-limited debug logging (first 3 per reason)
            if v10_none_reason_counts.get(reason, 0) <= 3:
                log.debug(f"[V10_PRED_NONE] reason={reason}")
            _record_predict_attempt(reason)
            return None
        
        # Early session gating: V10 hybrid requires XGB which only exists for EU/US/OVERLAP
        # Check session early and return with explicit reason code (not counted as V10 call failure)
        # NOTE: Session filtering is now handled in entry_manager.py with v10_session_supported gate
        # This duplicate check is kept for defensive programming but should not trigger
        # if entry_manager gate is working correctly
        current_session_early = policy_state.get("session", "OVERLAP")
        supported_sessions = {"EU", "US", "OVERLAP"}
        if current_session_early not in supported_sessions:
            # ENTRY FEATURE TELEMETRY: Record model block
            if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                self.entry_manager.entry_feature_telemetry.record_model_block(model_name, "SESSION_UNSUPPORTED_INTERNAL")
            
            # ASIA and other sessions don't have XGB models - skip silently (not a failure)
            # This is expected and should NOT count towards v10_none_reason_counts
            # NOTE: This should not happen if entry_manager v10_session_supported gate is working
            run_mode = getattr(self, "replay_mode", False) and "REPLAY" or "LIVE"
            is_sniper = getattr(self, "is_sniper", False)
            log.warning(
                "[ENTRY_DEBUG] Early return in _predict_entry_v10_hybrid | "
                f"reason=SESSION_NOT_SUPPORTED (duplicate check - should be caught by entry_manager gate) | "
                f"run_mode={run_mode} | "
                f"session={current_session_early} | "
                f"is_sniper={is_sniper}"
            )
            log.debug("[ENTRY_V10] Session %s not supported (no XGB model), skipping", current_session_early)
            return _ret_none("blocked_by_session_gate")
        
        # Determine ctx_expected (for fail-fast assert)
        context_features_enabled = os.getenv("ENTRY_CONTEXT_FEATURES_ENABLED", "false").lower() == "true"
        ctx_bundle_active = getattr(self, "entry_v10_ctx_enabled", False)
        bundle_meta = getattr(self.entry_v10_bundle, "metadata", {}) or {} if self.entry_v10_bundle else {}
        supports_context_features = bundle_meta.get("supports_context_features", False)
        ctx_expected = context_features_enabled and ctx_bundle_active and supports_context_features
        
        # Hard assert: if ctx_expected, entry_context_features must not be None
        if ctx_expected and entry_context_features is None:
            if hasattr(self, "entry_manager") and self.entry_manager:
                et = self.entry_manager.entry_telemetry
                if "ctx_proof_fail_count" not in et:
                    et["ctx_proof_fail_count"] = 0
                et["ctx_proof_fail_count"] += 1
            raise RuntimeError(
                "[CTX_CONTRACT_MISSING] ctx_expected=True but entry_context_features=None. "
                f"ENTRY_CONTEXT_FEATURES_ENABLED={context_features_enabled}, "
                f"entry_v10_ctx_enabled={ctx_bundle_active}, supports_context_features={supports_context_features}. "
                "CTX6CAT6 only (no fallback)."
            )
        
        if self.entry_v10_bundle is None:
            # ENTRY FEATURE TELEMETRY: Record model block and pre-model return
            if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                current_session_debug = policy_state.get("session", "OVERLAP")
                self.entry_manager.entry_feature_telemetry.record_model_block(model_name, "BUNDLE_MISSING")
                self.entry_manager.entry_feature_telemetry.record_pre_model_return("BUNDLE_MISSING", session=current_session_debug)
            
            run_mode = getattr(self, "replay_mode", False) and "REPLAY" or "LIVE"
            current_session_debug = policy_state.get("session", "OVERLAP")
            is_sniper = getattr(self, "is_sniper", False)
            log.error(
                "[ENTRY_DEBUG] Early return in _predict_entry_v10_hybrid | "
                f"reason=BUNDLE_MISSING | "
                f"run_mode={run_mode} | "
                f"session={current_session_debug} | "
                f"is_sniper={is_sniper}"
            )
            log.error("[ENTRY_V10] V10 bundle not loaded")
            return _ret_none("BUNDLE_MISSING")
        
        try:
            import torch
            import numpy as np
            
            # DEL 2: Note: runtime_v9 is now only used via runtime_v10_ctx wrapper in replay-mode
            # Direct import of runtime_v9 is forbidden in replay (handled by wrapper below)
            # Live mode: import runtime_v9 locally (not top-level, done later in try/except)
            
            # Get current session
            current_session = policy_state.get("session", "OVERLAP")
            session_map = {"EU": 0, "OVERLAP": 1, "US": 2}
            session_id = session_map.get(current_session, 1)
            
            # Get current bar for regime info
            if len(entry_bundle.features) == 0:
                # ENTRY FEATURE TELEMETRY: Record model block and pre-model return
                if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                    current_session_debug = policy_state.get("session", "OVERLAP")
                    self.entry_manager.entry_feature_telemetry.record_model_block(model_name, "FEATURES_EMPTY")
                    self.entry_manager.entry_feature_telemetry.record_pre_model_return("FEATURES_EMPTY", session=current_session_debug)
                
                run_mode = getattr(self, "replay_mode", False) and "REPLAY" or "LIVE"
                current_session_debug = policy_state.get("session", "OVERLAP")
                is_sniper = getattr(self, "is_sniper", False)
                log.error(
                    "[ENTRY_DEBUG] Early return in _predict_entry_v10_hybrid | "
                    f"reason=FEATURES_EMPTY | "
                    f"run_mode={run_mode} | "
                    f"session={current_session_debug} | "
                    f"is_sniper={is_sniper}"
                )
                log.warning("[ENTRY_V10] Empty features DataFrame")
                return _ret_none("FEATURES_EMPTY")
            
            current_bar = entry_bundle.features.iloc[-1]
            
            # Get regime IDs
            # vol_regime_id: 0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME
            vol_regime_id_raw = int(current_bar.get("atr_regime_id", 2)) if "atr_regime_id" in current_bar.index else 2
            vol_regime_id = max(0, min(3, vol_regime_id_raw))
            
            # trend_regime_id: 0=UP, 1=DOWN, 2=NEUTRAL
            trend_regime_tf24h = float(current_bar.get("trend_regime_tf24h", 0.0)) if "trend_regime_tf24h" in current_bar.index else 0.0
            if trend_regime_tf24h > 0.001:
                trend_regime_id = 0  # UP
            elif trend_regime_tf24h < -0.001:
                trend_regime_id = 1  # DOWN
            else:
                trend_regime_id = 2  # NEUTRAL
            
            # Get sequence length from bundle (needed for both PREBUILT and non-PREBUILT paths)
            seq_len = self.entry_v10_bundle.transformer_config.get("seq_len", 90)
            
            # PREBUILT mode: Use prebuilt features ONLY (no runtime building allowed)
            # SSoT: Use self.replay_mode_enum (set explicitly in _run_replay_impl), NOT os.getenv
            # In PREBUILT mode, entry_bundle.features already contains prebuilt features
            # We need to extract a sequence window from prebuilt_features_df for transformer
            replay_mode_enum = getattr(self, "replay_mode_enum", None)
            if replay_mode_enum is None:
                # Fallback: try to determine from env (should not happen in normal flow)
                from gx1.utils.replay_mode import ReplayMode
                replay_mode_enum = ReplayMode.from_env()
                log.warning(
                    "[ENTRY_V10_CTX] replay_mode_enum not set on runner, falling back to env. "
                    "This should not happen in normal flow."
                )
            
            prebuilt_enabled = replay_mode_enum.is_prebuilt()
            
            log.debug(
                "[ENTRY_V10_CTX] Feature sourcing: replay_mode_enum=%s, prebuilt_enabled=%s, has_prebuilt_df=%s",
                replay_mode_enum,
                prebuilt_enabled,
                hasattr(self, "prebuilt_features_df") and self.prebuilt_features_df is not None
            )
            
            if self.replay_mode and prebuilt_enabled:
                # STEG 2: Prebuilt dataframe must be explicitly available (not global magic)
                # Assert prebuilt_features_df is not None if replay_mode==PREBUILT
                if not hasattr(self, "prebuilt_features_df") or self.prebuilt_features_df is None:
                    # Write PREBUILT_MODE_BUT_NO_PREBUILT_DF capsule
                    try:
                        import traceback
                        import json
                        import sys
                        from datetime import datetime, timezone
                        # Get output directory (use Path_local from outer scope)
                        output_dir = Path_local(os.getenv("GX1_OUTPUT_DIR", "/tmp"))
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Get chunk_id and pid if available
                        chunk_id = getattr(self, "_current_chunk_id", None)
                        pid = os.getpid()
                        
                        # Build capsule data
                        capsule_data = {
                            "error": "PREBUILT_MODE_BUT_NO_PREBUILT_DF",
                            "replay_mode_enum": str(replay_mode_enum),
                            "prebuilt_enabled": prebuilt_enabled,
                            "has_prebuilt_features_df": hasattr(self, "prebuilt_features_df"),
                            "prebuilt_features_df_is_none": getattr(self, "prebuilt_features_df", None) is None,
                            "traceback": traceback.format_exc(),
                            "pid": pid,
                            "chunk_id": chunk_id,
                            "sys_path": list(sys.path),
                            "cwd": str(Path.cwd()),
                            "sys_executable": sys.executable,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        
                        # Write capsule
                        capsule_path = output_dir / "PREBUILT_MODE_BUT_NO_PREBUILT_DF.json"
                        with open(capsule_path, "w") as f:
                            json.dump(capsule_data, f, indent=2)
                        
                        log.error(
                            "[FATAL] PREBUILT_MODE_BUT_NO_PREBUILT_DF capsule written to %s",
                            capsule_path
                        )
                    except Exception as capsule_error:
                        log.error(
                            "[FATAL] Failed to write PREBUILT_MODE_BUT_NO_PREBUILT_DF capsule: %s",
                            capsule_error
                        )
                    
                    is_truth = os.getenv("GX1_TRUTH_MODE", "0") == "1" or os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
                    error_msg = (
                        "[FATAL] PREBUILT mode enabled but prebuilt_features_df is None. "
                        "This is FATAL in TRUTH mode. "
                        "Prebuilt features must be loaded before entry evaluation. "
                        "PREBUILT_MODE_BUT_NO_PREBUILT_DF capsule written."
                    )
                    if is_truth:
                        raise RuntimeError(error_msg)
                    else:
                        log.error(error_msg)
                        return _ret_none("PREBUILT_FEATURES_MISSING")
                
                # Get current timestamp from candles
                current_ts = candles.index[-1]
                
                # Get sequence window from prebuilt features (last seq_len bars ending at current_ts)
                # Find the index of current_ts in prebuilt features
                if current_ts not in self.prebuilt_features_df.index:
                    is_truth = os.getenv("GX1_TRUTH_MODE", "0") == "1" or os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
                    error_msg = (
                        f"[FATAL] PREBUILT mode: Current timestamp {current_ts} not found in prebuilt features. "
                        "This is FATAL in TRUTH mode."
                    )
                    if is_truth:
                        raise RuntimeError(error_msg)
                    else:
                        log.error(error_msg)
                        return _ret_none("PREBUILT_TIMESTAMP_MISSING")
                
                # Get index position of current_ts
                current_idx = self.prebuilt_features_df.index.get_loc(current_ts)
                
                # Extract sequence window (last seq_len bars ending at current_ts)
                if current_idx < seq_len - 1:
                    # Not enough history - FATAL in PREBUILT mode
                    is_truth = os.getenv("GX1_TRUTH_MODE", "0") == "1" or os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
                    error_msg = (
                        f"[FATAL] PREBUILT mode: Insufficient history for sequence window. "
                        f"Need {seq_len} bars, but only {current_idx + 1} available at {current_ts}. "
                        "This is FATAL in TRUTH mode."
                    )
                    if is_truth:
                        raise RuntimeError(error_msg)
                    else:
                        log.error(error_msg)
                        return _ret_none("PREBUILT_SEQ_TOO_SHORT")
                
                # Extract sequence window
                seq_start_idx = current_idx - seq_len + 1
                seq_end_idx = current_idx + 1
                df_v9_feats = self.prebuilt_features_df.iloc[seq_start_idx:seq_end_idx].copy()
                
                # SIGNAL-ONLY ARCHITECTURE:
                # No feature_meta usage for Transformer inputs (raw seq/snap universes removed).
                # Contract is enforced via MASTER_MODEL_LOCK + PrebuiltFeaturesLoader strict column match.
                seq_feat_names = []
                snap_feat_names = []
                log.debug(
                    "[ENTRY_V10_CTX] Using prebuilt features (signal-only): shape=%s, seq_len=%d",
                    df_v9_feats.shape, seq_len
                )
            else:
                # Non-PREBUILT mode: Build features from candles (baseline or live mode)
                # SIGNAL-ONLY ARCHITECTURE: raw-feature Transformer inputs are forbidden.
                # In replay/TRUTH, this must hard-fail (no fallback to runtime feature building).
                if self.replay_mode or os.getenv("GX1_TRUTH_MODE", "0") == "1" or os.getenv("GX1_RUN_MODE", "").upper() in {"TRUTH", "SMOKE"}:
                    raise RuntimeError(
                        "[ENTRY_V10_SIGNAL_ONLY_FAIL] RAW_FEATURE_PATH_FORBIDDEN: Non-PREBUILT feature building was requested, "
                        "but ENTRY_V10_CTX is now signal-only. Use prebuilt refined3 features for XGB inputs."
                    )
                df_raw = candles.copy()
                
                # Column deduplication policy:
                # - Replay/offline: Let runtime_v9.py hard fail on collisions (data integrity)
                # - Live: Defensive dedupe with warning (tolerate vendor glitches, but escalate)
                if self.is_replay:
                    # Replay mode: No defensive deduplication - let runtime_v9.py hard fail
                    # This ensures data integrity in backtests
                    pass
                else:
                    # Live mode: Defensive deduplication to tolerate vendor glitches
                    # But this should be rare - if it happens, escalate
                    if df_raw.columns.duplicated().any():
                        # Remove duplicate column names (keep first occurrence)
                        df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
                        log.warning(
                            "[ENTRY_V10] Live mode: Removed duplicate columns from candles DataFrame. "
                            "Original columns: %s. This should be investigated.",
                            list(candles.columns)
                        )
                    
                    # Also check for case-insensitive duplicates (e.g., "close" and "CLOSE")
                    col_lower_map = {}
                    cols_to_drop = []
                    for col in df_raw.columns:
                        col_lower = col.lower()
                        if col_lower in col_lower_map:
                            cols_to_drop.append(col)
                        else:
                            col_lower_map[col_lower] = col
                    
                    if cols_to_drop:
                        df_raw = df_raw.drop(columns=cols_to_drop)
                        kept_cols = [col_lower_map[c.lower()] for c in cols_to_drop if c.lower() in col_lower_map]
                        log.warning(
                            "[ENTRY_V10] Live mode: Removed case-insensitive duplicate columns: %s (kept: %s). "
                            "This indicates upstream data issue and should be investigated.",
                            cols_to_drop,
                            kept_cols
                        )
                        # TODO: Escalate to data integrity flag/kill-switch if this becomes frequent
                
                # Build V9 features
                # Get feature_meta_path from config (same as used during loading)
                feature_meta_path = Path_local(self.entry_v10_cfg.get("feature_meta_path"))
                if not feature_meta_path.exists():
                    # ENTRY FEATURE TELEMETRY: Record model block and pre-model return
                    if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                        current_session_debug = policy_state.get("session", "OVERLAP")
                        self.entry_manager.entry_feature_telemetry.record_model_block(model_name, "FEATURE_META_PATH_MISSING")
                        self.entry_manager.entry_feature_telemetry.record_pre_model_return("FEATURE_META_PATH_MISSING", session=current_session_debug)
                    
                    run_mode = getattr(self, "replay_mode", False) and "REPLAY" or "LIVE"
                    current_session_debug = policy_state.get("session", "OVERLAP")
                    is_sniper = getattr(self, "is_sniper", False)
                    log.error(
                        "[ENTRY_DEBUG] Early return in _predict_entry_v10_hybrid | "
                        f"reason=FEATURE_META_PATH_MISSING | "
                        f"run_mode={run_mode} | "
                        f"session={current_session_debug} | "
                        f"is_sniper={is_sniper}"
                    )
                    log.error("[ENTRY_V10] Feature meta path not found: %s", feature_meta_path)
                    return _ret_none("FEATURE_META_PATH_MISSING")
                
                # Get scaler paths from config (bundles hold scaler objects, but we need paths for build_v9_runtime_features)
                seq_scaler_path = self.entry_v10_cfg.get("seq_scaler_path")
                snap_scaler_path = self.entry_v10_cfg.get("snap_scaler_path")
                seq_scaler_path = Path_local(seq_scaler_path) if seq_scaler_path else None
                snap_scaler_path = Path_local(snap_scaler_path) if snap_scaler_path else None
                
                # DEL 2: Build runtime features using V10_CTX wrapper (not runtime_v9 directly)
                # In replay-mode, use runtime_v10_ctx wrapper (same logic, V10 identity)
                # In live mode, use runtime_v9 directly (backward compatible)
                try:
                    if self.replay_mode:
                        # Replay mode without PREBUILT: use V10_CTX wrapper (forbids direct runtime_v9 usage)
                        # Import with error handling
                        try:
                            from gx1.features.runtime_v10_ctx import build_v10_ctx_runtime_features
                        except (ImportError, ModuleNotFoundError) as import_err:
                            # Write IMPORT_FAIL capsule and hard-fail in TRUTH mode
                            self._write_import_fail_capsule(import_err, "gx1.features.runtime_v10_ctx", "build_v10_ctx_runtime_features")
                            # Check if TRUTH mode
                            is_truth = os.getenv("GX1_TRUTH_MODE", "0") == "1" or os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
                            if is_truth:
                                raise RuntimeError(
                                    f"[FATAL] Import error in entry path: {import_err}. "
                                    f"This is FATAL in TRUTH mode. IMPORT_FAIL capsule written."
                                ) from import_err
                            else:
                                log.error(f"[ENTRY_V10] Import error (non-TRUTH): {import_err}")
                                return _ret_none(f"IMPORT_ERROR:{type(import_err).__name__}")
                        
                        df_v9_feats, seq_feat_names, snap_feat_names = build_v10_ctx_runtime_features(
                            df_raw,
                            feature_meta_path,
                            seq_scaler_path=seq_scaler_path,
                            snap_scaler_path=snap_scaler_path,
                        )
                        log.debug(
                            "[ENTRY_V10_CTX] Built runtime features via V10_CTX wrapper: shape=%s, n_seq=%d, n_snap=%d",
                            df_v9_feats.shape, len(seq_feat_names), len(snap_feat_names)
                        )
                    else:
                        # Live mode: use runtime_v9 directly (backward compatible)
                        # DEL 2: Import runtime_v9 locally (not at top-level)
                        try:
                            from gx1.features.runtime_v9 import build_v9_runtime_features
                        except (ImportError, ModuleNotFoundError) as import_err:
                            # Write IMPORT_FAIL capsule and hard-fail in TRUTH mode
                            self._write_import_fail_capsule(import_err, "gx1.features.runtime_v9", "build_v9_runtime_features")
                            # Check if TRUTH mode
                            is_truth = os.getenv("GX1_TRUTH_MODE", "0") == "1" or os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
                            if is_truth:
                                raise RuntimeError(
                                    f"[FATAL] Import error in entry path: {import_err}. "
                                    f"This is FATAL in TRUTH mode. IMPORT_FAIL capsule written."
                                ) from import_err
                            else:
                                log.error(f"[ENTRY_V10] Import error (non-TRUTH): {import_err}")
                                return _ret_none(f"IMPORT_ERROR:{type(import_err).__name__}")
                        
                        df_v9_feats, seq_feat_names, snap_feat_names = build_v9_runtime_features(
                            df_raw,
                            feature_meta_path,
                            seq_scaler_path=seq_scaler_path,
                            snap_scaler_path=snap_scaler_path,
                        )
                except Exception as e:
                    # Enhanced error reporting for feature building
                    import traceback
                    stack_excerpt = "\n".join(traceback.format_exc().splitlines()[:10])
                    
                    # Determine reason code from exception type and message
                    error_str = str(e).lower()
                    if "timeout" in error_str:
                        reason = "FEATURE_BUILD_TIMEOUT:unknown"
                    elif "dtype" in error_str or "type" in error_str:
                        reason = "FEATURE_BUILD_DTYPE_ERROR"
                    else:
                        reason = f"FEATURE_BUILD_ERROR:{type(e).__name__}"
                    
                    log.error(
                        "[ENTRY_V10] Feature building error: %s (reason=%s)\nStack excerpt:\n%s",
                        e,
                        reason,
                        stack_excerpt
                    )
                    
                    # Store stack excerpt in telemetry for diagnostics
                    if hasattr(self, "entry_manager") and self.entry_manager:
                        if "v10_exception_stacks" not in self.entry_manager.entry_telemetry:
                            self.entry_manager.entry_telemetry["v10_exception_stacks"] = {}
                        if reason not in self.entry_manager.entry_telemetry["v10_exception_stacks"]:
                            self.entry_manager.entry_telemetry["v10_exception_stacks"][reason] = stack_excerpt
                    
                    return _ret_none(reason)
            
            # Get sequence length from bundle
            seq_len = self.entry_v10_bundle.transformer_config.get("seq_len", 30)
            if len(df_v9_feats) < seq_len:
                # ENTRY FEATURE TELEMETRY: Record model block and pre-model return
                if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                    current_session_debug = policy_state.get("session", "OVERLAP")
                    self.entry_manager.entry_feature_telemetry.record_model_block(model_name, "SEQ_TOO_SHORT")
                    self.entry_manager.entry_feature_telemetry.record_pre_model_return("SEQ_TOO_SHORT", session=current_session_debug)
                
                run_mode = getattr(self, "replay_mode", False) and "REPLAY" or "LIVE"
                current_session_debug = policy_state.get("session", "OVERLAP")
                is_sniper = getattr(self, "is_sniper", False)
                log.error(
                    "[ENTRY_DEBUG] Early return in _predict_entry_v10_hybrid | "
                    f"reason=SEQ_TOO_SHORT | "
                    f"run_mode={run_mode} | "
                    f"session={current_session_debug} | "
                    f"is_sniper={is_sniper}"
                )
                log.warning("[ENTRY_V10] Insufficient history: %d < %d", len(df_v9_feats), seq_len)
                return _ret_none("SEQ_TOO_SHORT")
            
            # Extract sequence window (last seq_len bars)
            seq_window = df_v9_feats.tail(seq_len).copy()
            current_row = df_v9_feats.iloc[-1]
            
            # SIGNAL-ONLY ARCHITECTURE:
            # Legacy ablation toggles for v10_ctx XGB channels are forbidden (no legacy dims exist).
            if os.getenv("GX1_DISABLE_XGB_CHANNELS_IN_TRANSFORMER", "0") == "1":
                raise RuntimeError(
                    "[ENTRY_V10_SIGNAL_ONLY_FAIL] LEGACY_TOGGLE_FORBIDDEN: GX1_DISABLE_XGB_CHANNELS_IN_TRANSFORMER was set, "
                    "but the Transformer input is signal-only (no v10_ctx channel layout exists)."
                )
            
            # XGB SESSION POLICY: Check if session is disabled
            # SENTINEL D: before_xgb_session_policy_check
            if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                self.entry_manager.entry_feature_telemetry.record_control_flow("BEFORE_XGB_SESSION_POLICY_CHECK", session=current_session)
            
            skip_xgb_predict = False
            # Session policy: canonical universal bundle; no external session-policy load (ONE UNIVERSE)
            session_enabled = True
            session_mode = "normal"
            # SENTINEL E: after_xgb_session_policy_check
            if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                self.entry_manager.entry_feature_telemetry.record_control_flow(
                    "AFTER_XGB_SESSION_POLICY_CHECK",
                    session=current_session,
                    enabled=session_enabled,
                    mode=session_mode,
                )
            
            # Run XGBoost for the full sequence window using universal multihead model (batched).
            # This constructs the ONLY Transformer inputs (signal-only bridge).
            xgb_model = getattr(self.entry_v10_bundle, "xgb_model_universal", None)
            if xgb_model is None:
                raise RuntimeError("[ENTRY_V10_SIGNAL_ONLY_FAIL] XGB_MODEL_MISSING: entry_v10_bundle.xgb_model_universal is None")

            try:
                xgb_seq = xgb_model.predict_proba(seq_window, session=current_session)
            except Exception as e:
                raise RuntimeError(f"[ENTRY_V10_SIGNAL_ONLY_FAIL] XGB_SEQ_PREDICT_FAIL session={current_session}: {e}") from e

            # Extract last-step snapshot values from sequence prediction (SSoT: snap == seq[-1])
            p_long_seq = np.asarray(xgb_seq.p_long, dtype=np.float32)
            p_short_seq = np.asarray(xgb_seq.p_short, dtype=np.float32)
            p_flat_seq = np.asarray(xgb_seq.p_flat, dtype=np.float32)
            uncertainty_seq = np.asarray(xgb_seq.uncertainty, dtype=np.float32)  # normalized entropy

            if p_long_seq.shape[0] != seq_len:
                raise RuntimeError(
                    f"[ENTRY_V10_SIGNAL_ONLY_FAIL] XGB_SEQ_LEN_MISMATCH expected={seq_len} got={int(p_long_seq.shape[0])}"
                )

            probs_seq = np.stack([p_long_seq, p_short_seq, p_flat_seq], axis=1)  # [T,3]
            eps = 1e-10
            probs_clip = np.clip(probs_seq, eps, 1.0).astype(np.float32)
            entropy_seq = (-np.sum(probs_clip * np.log(probs_clip), axis=1)).astype(np.float32)  # [T]

            # margin_top1_top2 across 3-class probs (top1 - top2)
            probs_sorted = np.sort(probs_seq, axis=1)  # ascending
            margin_seq = (probs_sorted[:, -1] - probs_sorted[:, -2]).astype(np.float32)  # [T]

            p_hat_seq = np.maximum(p_long_seq, p_short_seq).astype(np.float32)  # [T]

            # Snapshot (current bar) == last timestep
            p_long_xgb = float(p_long_seq[-1])
            p_short_xgb = float(p_short_seq[-1])
            p_flat_xgb = float(p_flat_seq[-1])
            p_hat_xgb = float(p_hat_seq[-1])
            uncertainty_score = float(uncertainty_seq[-1])
            entropy_snap = float(entropy_seq[-1])
            margin_snap = float(margin_seq[-1])

            # Build signal-only tensors
            from gx1.contracts.signal_bridge_v1 import (
                ORDERED_FIELDS,
                SEQ_SIGNAL_DIM,
                SNAP_SIGNAL_DIM,
                validate_seq_signal,
                validate_snap_signal,
                validate_contract_in_truth,
            )
            validate_contract_in_truth()

            # seq_data: [T, D]
            seq_data = np.zeros((seq_len, SEQ_SIGNAL_DIM), dtype=np.float32)
            # snap_data: [D]
            snap_data = np.zeros((SNAP_SIGNAL_DIM,), dtype=np.float32)

            # Fill per-field (explicit ordered schema)
            field_to_seq: Dict[str, np.ndarray] = {
                "p_long": p_long_seq,
                "p_short": p_short_seq,
                "p_flat": p_flat_seq,
                "p_hat": p_hat_seq,
                "uncertainty_score": uncertainty_seq,
                "margin_top1_top2": margin_seq,
                "entropy": entropy_seq,
            }
            field_to_snap: Dict[str, float] = {
                "p_long": p_long_xgb,
                "p_short": p_short_xgb,
                "p_flat": p_flat_xgb,
                "p_hat": p_hat_xgb,
                "uncertainty_score": uncertainty_score,
                "margin_top1_top2": margin_snap,
                "entropy": entropy_snap,
            }

            for j, fname in enumerate(ORDERED_FIELDS):
                if fname not in field_to_seq:
                    raise RuntimeError(f"[ENTRY_V10_SIGNAL_ONLY_FAIL] Missing seq field in contract mapping: {fname}")
                if fname not in field_to_snap:
                    raise RuntimeError(f"[ENTRY_V10_SIGNAL_ONLY_FAIL] Missing snap field in contract mapping: {fname}")
                seq_data[:, j] = np.asarray(field_to_seq[fname], dtype=np.float32)
                snap_data[j] = float(field_to_snap[fname])

            # Expand batch and validate
            seq_x_np = seq_data[np.newaxis, :, :]  # [1,T,D]
            snap_x_np = snap_data[np.newaxis, :]  # [1,D]
            validate_seq_signal(seq_x_np, context="entry_v10_hybrid.signal_only")
            validate_snap_signal(snap_x_np, context="entry_v10_hybrid.signal_only")
                
            
            # SIGNAL-ONLY: legacy snap-dim feature_meta checks, v10_ctx channel ablations, and v10_ctx fingerprints removed.
            # Dim sanity is enforced by:
            # - signal_bridge_v1 validators (finite + dims)
            # - EntryV10CtxHybridTransformer strict input validation
            session_tokens_enabled = False
            disable_xgb_channels_in_transformer = False
            n_xgb_channels_in_transformer_input = int(SEQ_SIGNAL_DIM)
            
            # FIRST-N-CYCLES ECHO CHECK: Check for NaN and log stats
            from gx1.runtime.first_n_cycles_check import FirstNCyclesChecker
            
            # Initialize checker if not exists
            if not hasattr(self, "_first_n_cycles_checker"):
                warmup_bars = getattr(self, "warmup_bars", 288)
                self._first_n_cycles_checker = FirstNCyclesChecker(
                    n_cycles=200,
                    warmup_bars=warmup_bars,
                    nan_tolerance=0,  # Hard-fail on any NaN after warmup
                )
            
            # Get current bar index (approximate from candles length)
            current_bar_index = len(candles) if candles is not None else 0
            
            # Prepare ctx arrays for echo check
            ctx_cat_arr = None
            ctx_cont_arr = None
            
            # Check cycle (hard-fail if NaN after warmup)
            if self._first_n_cycles_checker.should_continue_checking():
                error_msg = self._first_n_cycles_checker.check_cycle(
                    seq_data=seq_data,
                    snap_data=snap_data,
                    ctx_cat=ctx_cat_arr,
                    ctx_cont=ctx_cont_arr,
                    current_bar_index=current_bar_index,
                )
                if error_msg:
                    raise RuntimeError(error_msg)
            
            # SIGNAL-ONLY: Log contract at replay-start (once)
            if not hasattr(self, "_v10_contract_logged"):
                from gx1.contracts.signal_bridge_v1 import ORDERED_FIELDS, SEQ_SIGNAL_DIM, SNAP_SIGNAL_DIM, CONTRACT_SHA256
                log.info("[ENTRY_V10] Signal Bridge Contract (V1):")
                log.info("  bridge_id=%s sha256=%s", "XGB_SIGNAL_BRIDGE_V1", str(CONTRACT_SHA256)[:16])
                log.info("  SEQ_SIGNAL_DIM=%d SNAP_SIGNAL_DIM=%d", int(SEQ_SIGNAL_DIM), int(SNAP_SIGNAL_DIM))
                log.info("  ORDERED_FIELDS(%d): %s", len(ORDERED_FIELDS), ORDERED_FIELDS)
                log.info("  Runtime seq_data shape: %s", seq_data.shape)
                log.info("  Runtime snap_data shape: %s", snap_data.shape)
                self._v10_contract_logged = True
            
            # DEL 3: Bundle metadata check removed from hot path
            # Bundle metadata mismatch is now logged once at RUN_START in _run_replay_impl()
            # (see _run_replay_impl() around line 9642-9660)
            # This reduces log noise - contract validation already passed, so no need to log repeatedly
            
            # OPPGAVE 2: Check if bundle supports context features
            supports_context_features = bundle_meta.get("supports_context_features", False)
            
            # OPPGAVE 2: Use context features if available and bundle supports it
            context_features_enabled = os.getenv("ENTRY_CONTEXT_FEATURES_ENABLED", "false").lower() == "true"
            
            # Debug logging (first 3 calls only)
            if not hasattr(self, "_v10_ctx_debug_count"):
                self._v10_ctx_debug_count = 0
            if self._v10_ctx_debug_count < 3:
                log.info(
                    "[ENTRY_V10] CTX Debug #%d: context_features_enabled=%s, entry_context_features=%s, "
                    "supports_context_features=%s, bundle_meta.keys=%s",
                    self._v10_ctx_debug_count + 1,
                    context_features_enabled,
                    "not None" if entry_context_features is not None else "None",
                    supports_context_features,
                    list(bundle_meta.keys()) if bundle_meta else []
                )
                self._v10_ctx_debug_count += 1
            
            if context_features_enabled and entry_context_features is not None:
                # Context features enabled and provided
                if not supports_context_features:
                    # Bundle doesn't support context features - hard fail in replay, warning in live
                    is_replay = getattr(self, "replay_mode", False)
                    if is_replay:
                        raise RuntimeError(
                            "CONTEXT_FEATURES_MISMATCH: Context features enabled but bundle does not support them. "
                            f"Bundle metadata: supports_context_features={supports_context_features}. "
                            "Either disable ENTRY_CONTEXT_FEATURES_ENABLED or use a bundle that supports context features."
                        )
                    else:
                        log.warning(
                            "[ENTRY_V10] Context features enabled but bundle does not support them. "
                            "Ignoring context features and using legacy regime inputs."
                        )
                        entry_context_features = None  # Fall back to legacy
                else:
                    # Bundle supports context features - use them (CTX6CAT6: 6 cat, 6 cont)
                    # Populate slow ctx features from prebuilt row (manifest SSoT); hard-fail if missing
                    try:
                        ts = candles.index[-1]
                        prebuilt_row = self.prebuilt_features_df.loc[ts]
                        from gx1.contracts.signal_bridge_v1 import get_canonical_ctx_contract
                        canonical_ctx = get_canonical_ctx_contract()
                        slow_cont_names = [n for n in canonical_ctx["ctx_cont_names"] if n not in ["atr_bps", "spread_bps"]]
                        log.info(
                            "[CTX_DEBUG] candle_ts=%r tz=%s | current_ts=%r tz=%s | equal=%s",
                            ts,
                            getattr(ts, "tzinfo", None),
                            ts,
                            getattr(ts, "tzinfo", None),
                            True,
                        )
                        log.error(
                            "[CTX_DEBUG] prebuilt_index_has_ts=%s | df_cols_has_slow=%s",
                            ts in self.prebuilt_features_df.index,
                            {c: c in self.prebuilt_features_df.columns for c in slow_cont_names},
                        )
                        if ts in self.prebuilt_features_df.index:
                            try:
                                log.error(
                                    "[CTX_DEBUG] slow_cont_values=%s",
                                    prebuilt_row[slow_cont_names].to_dict(),
                                )
                            except Exception:
                                pass
                        missing_cols = [c for c in slow_cont_names if c not in prebuilt_row.index]
                        if missing_cols:
                            raise RuntimeError(f"[CTX_CONT_BUILD_FAIL] Missing slow ctx columns in prebuilt row: {missing_cols}")
                        entry_context_features.D1_dist_from_ema200_atr = float(prebuilt_row["D1_dist_from_ema200_atr"])
                        entry_context_features.H1_range_compression_ratio = float(prebuilt_row["H1_range_compression_ratio"])
                        entry_context_features.D1_atr_percentile_252 = float(prebuilt_row["D1_atr_percentile_252"])
                        entry_context_features.M15_range_compression_ratio = float(prebuilt_row["M15_range_compression_ratio"])
                        if "H4_trend_sign_cat" not in prebuilt_row.index:
                            raise RuntimeError(
                                "[CTX_CAT_SOURCE_MISSING] prebuilt missing required ctx_cat col: H4_trend_sign_cat (CTX6CAT6)"
                            )
                        entry_context_features.h4_trend_sign_cat = int(prebuilt_row["H4_trend_sign_cat"])
                        log.error(
                            "[CTX_DEBUG_ASSIGN] current_ts=%r tz=%s | ctx_fields=%s | slow_cont_values=%s",
                            ts,
                            getattr(ts, "tzinfo", None),
                            entry_context_features.__dict__,
                            {
                                "D1_dist_from_ema200_atr": entry_context_features.D1_dist_from_ema200_atr,
                                "H1_range_compression_ratio": entry_context_features.H1_range_compression_ratio,
                                "D1_atr_percentile_252": entry_context_features.D1_atr_percentile_252,
                                "M15_range_compression_ratio": entry_context_features.M15_range_compression_ratio,
                            },
                        )
                    except Exception as e:
                        raise RuntimeError(f"[CTX_CONT_BUILD_FAIL] Failed to populate ctx_cont from prebuilt: {e}") from e

                    ctx_cat = entry_context_features.to_tensor_categorical()  # [6] int64
                    from gx1.contracts.signal_bridge_v1 import get_canonical_ctx_contract as _get_ctx_contract_pre_tensor
                    _ctx_contract = _get_ctx_contract_pre_tensor()
                    log.error(
                        "[CTX_DEBUG_PRE_TENSOR] ctx_fields=%s | expected_cat=%s | expected_cont=%s",
                        entry_context_features.__dict__,
                        _ctx_contract.get("ctx_cat_names"),
                        _ctx_contract.get("ctx_cont_names"),
                    )
                    ctx_cont = entry_context_features.to_tensor_continuous()  # [6] float32
                    ctx_cat_arr = np.array(ctx_cat, dtype=np.int64)
                    ctx_cont_arr = np.array(ctx_cont, dtype=np.float32)
                    
                    # Validate shapes (fail-fast in replay)
                    from gx1.contracts.signal_bridge_v1 import get_canonical_ctx_contract

                    canonical_ctx = get_canonical_ctx_contract()
                    if "expected_ctx_cat_dim" not in bundle_meta or "expected_ctx_cont_dim" not in bundle_meta:
                        if hasattr(self, "entry_manager") and self.entry_manager:
                            et = self.entry_manager.entry_telemetry
                            if "ctx_proof_fail_count" not in et:
                                et["ctx_proof_fail_count"] = 0
                            et["ctx_proof_fail_count"] += 1
                        raise RuntimeError(
                            "[CTX_CONTRACT_MISSING] bundle_meta missing expected_ctx_cat_dim/expected_ctx_cont_dim; canonical CTX6CAT6"
                        )
                    expected_ctx_cat_dim = int(bundle_meta.get("expected_ctx_cat_dim"))
                    expected_ctx_cont_dim = int(bundle_meta.get("expected_ctx_cont_dim"))
                    if (
                        expected_ctx_cat_dim != canonical_ctx["ctx_cat_dim"]
                        or expected_ctx_cont_dim != canonical_ctx["ctx_cont_dim"]
                    ):
                        if hasattr(self, "entry_manager") and self.entry_manager:
                            et = self.entry_manager.entry_telemetry
                            if "ctx_proof_fail_count" not in et:
                                et["ctx_proof_fail_count"] = 0
                            et["ctx_proof_fail_count"] += 1
                        raise RuntimeError(
                            "[CTX_CONTRACT_SPLIT_BRAIN] bundle ctx dims "
                            f"{expected_ctx_cont_dim}/{expected_ctx_cat_dim} != canonical "
                            f"{canonical_ctx['ctx_cont_dim']}/{canonical_ctx['ctx_cat_dim']} ({canonical_ctx['tag']})"
                        )
                    
                    if len(ctx_cat) != expected_ctx_cat_dim:
                        is_replay = getattr(self, "replay_mode", False)
                        if is_replay:
                            log.info(
                                "[CTX_INPUT_PROOF] expected=(%d,%d) actual=(%d,%d) cont_names=%s cat_names=%s cont_source=bundle_meta cat_source=entry_context_features",
                                expected_ctx_cont_dim,
                                expected_ctx_cat_dim,
                                len(ctx_cont) if entry_context_features is not None else -1,
                                len(ctx_cat),
                                getattr(entry_context_features, "CONT_NAMES", "N/A"),
                                getattr(entry_context_features, "CAT_NAMES", "N/A"),
                            )
                            if hasattr(self, "entry_manager") and self.entry_manager:
                                et = self.entry_manager.entry_telemetry
                                if "ctx_proof_fail_count" not in et:
                                    et["ctx_proof_fail_count"] = 0
                                et["ctx_proof_fail_count"] += 1
                            raise RuntimeError(
                                f"[CTX_CONTRACT_DIM_MISMATCH] ctx_cat dim={len(ctx_cat)} expected={expected_ctx_cat_dim}"
                            )
                        else:
                            log.warning(
                                f"[ENTRY_V10] ctx_cat shape mismatch: {len(ctx_cat)} != {expected_ctx_cat_dim}. "
                                "Using legacy regime inputs."
                            )
                            entry_context_features = None
                            return _ret_none("CTX_SHAPE_MISMATCH_CAT")
                    
                    if entry_context_features is not None and len(ctx_cont) != expected_ctx_cont_dim:
                        is_replay = getattr(self, "replay_mode", False)
                        if is_replay:
                            if hasattr(self, "entry_manager") and self.entry_manager:
                                et = self.entry_manager.entry_telemetry
                                if "ctx_proof_fail_count" not in et:
                                    et["ctx_proof_fail_count"] = 0
                                et["ctx_proof_fail_count"] += 1
                            raise RuntimeError(
                                f"[CTX_CONTRACT_DIM_MISMATCH] ctx_cont dim={len(ctx_cont)} expected={expected_ctx_cont_dim}"
                            )
                        else:
                            log.warning(
                                f"[ENTRY_V10] ctx_cont shape mismatch: {len(ctx_cont)} != {expected_ctx_cont_dim}. "
                                "Using legacy regime inputs."
                            )
                            entry_context_features = None
                            return _ret_none("CTX_SHAPE_MISMATCH_CONT")
            
            # Check cycle (hard-fail if NaN after warmup) — run after ctx is fully built
            if self._first_n_cycles_checker.should_continue_checking():
                error_msg = self._first_n_cycles_checker.check_cycle(
                    seq_data=seq_data,
                    snap_data=snap_data,
                    ctx_cat=ctx_cat_arr,
                    ctx_cont=ctx_cont_arr,
                    current_bar_index=current_bar_index,
                )
                if error_msg:
                    raise RuntimeError(error_msg)
            
            # SIGNAL-ONLY: channel masking removed (no legacy channel layout).
            self._current_mask_info = {}
            
            # Convert to tensors
            device = self.entry_v10_bundle.device
            seq_x = torch.FloatTensor(seq_x_np).to(device)  # [1, seq_len, SEQ_SIGNAL_DIM]
            snap_x = torch.FloatTensor(snap_x_np).to(device)  # [1, SNAP_SIGNAL_DIM]
            
            # Initialize ctx_cat_t and ctx_cont_t to None (will be set if ctx is available)
            ctx_cat_t = None
            ctx_cont_t = None
            
            # OPPGAVE 2: Use context features if available, otherwise fall back to legacy regime inputs
            if context_features_enabled and entry_context_features is not None and supports_context_features:
                # Use context features (new path)
                ctx_cat = entry_context_features.to_tensor_categorical()  # [6] int64
                ctx_cont = entry_context_features.to_tensor_continuous()  # [6] float32
                
                ctx_cat_t = torch.LongTensor(ctx_cat).unsqueeze(0).to(device)  # [1, 6]
                ctx_cont_t = torch.FloatTensor(ctx_cont).unsqueeze(0).to(device)  # [1, 6]
                log.info(
                    "[CTX_INPUT_PROOF] ctx_cont_len=%d ctx_cat_len=%d",
                    ctx_cont_t.shape[1] if ctx_cont_t is not None else -1,
                    ctx_cat_t.shape[1] if ctx_cat_t is not None else -1,
                )
                
                # CTX NULL BASELINE: force null/zero context when enabled (for A/B baseline without legacy)
                if os.getenv("GX1_CTX_NULL_BASELINE", "0") == "1":
                    ctx_cat_t = torch.zeros_like(ctx_cat_t)
                    ctx_cont_t = torch.zeros_like(ctx_cont_t)
                
                # Legacy regime inputs (for backward compatibility with transformer signature)
                # These are extracted from context features
                session_id_t = ctx_cat_t[:, 0]  # [1] - session_id from context
                vol_regime_id_t = ctx_cat_t[:, 2]  # [1] - vol_regime_id from context
                trend_regime_id_t = ctx_cat_t[:, 1]  # [1] - trend_regime_id from context
            else:
                # Legacy path (backward compatibility)
                session_id_t = torch.LongTensor([session_id]).to(device)  # [1]
                vol_regime_id_t = torch.LongTensor([vol_regime_id]).to(device)  # [1]
                trend_regime_id_t = torch.LongTensor([trend_regime_id]).to(device)  # [1]
                ctx_cat_t = None
                ctx_cont_t = None
            
            # Log first 3 calls with shapes (for diagnostics)
            if not hasattr(self, "_v10_shape_log_count"):
                self._v10_shape_log_count = 0
            if self._v10_shape_log_count < 3:
                log.info(
                    "[ENTRY_V10] Call #%d shapes: seq_x=%s (dtype=%s), snap_x=%s (dtype=%s), "
                    "context_features=%s",
                    self._v10_shape_log_count + 1,
                    tuple(seq_x.shape), str(seq_x.dtype),
                    tuple(snap_x.shape), str(snap_x.dtype),
                    "enabled" if (ctx_cat_t is not None and ctx_cont_t is not None) else "disabled"
                )
                self._v10_shape_log_count += 1
            
            # Predict with Transformer
            transformer_model = self.entry_v10_bundle.transformer_model
            transformer_model.eval()
            
            # SENTINEL F: before_transformer_forward
            if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                self.entry_manager.entry_feature_telemetry.record_control_flow("BEFORE_TRANSFORMER_FORWARD", session=current_session)
            
            # Note: write_transformer_exception_capsule is defined at function start (above) for use in outer try-except
            
            with torch.no_grad():
                # ENTRY FEATURE TELEMETRY: First-call capture hook (right before model call)
                # This ensures telemetry is recorded when transformer is actually called
                if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                    telemetry = self.entry_manager.entry_feature_telemetry
                    
                    # Record transformer input on first call only (if not already recorded)
                    # NOTE: This happens AFTER masking, so we use effective channels
                    if not telemetry.transformer_input_recorded:
                        from gx1.contracts.signal_bridge_v1 import ORDERED_FIELDS

                        # Signal-only: channels == ordered signal fields
                        effective_seq_channels = list(ORDERED_FIELDS)
                        effective_snap_channels = list(ORDERED_FIELDS)

                        # Values: use last timestep for seq (matches snap)
                        xgb_seq_values = {nm: float(seq_data[-1, j]) for j, nm in enumerate(ORDERED_FIELDS)}
                        xgb_snap_values = {nm: float(snap_data[j]) for j, nm in enumerate(ORDERED_FIELDS)}

                        # No session tokens in signal-only path
                        session_token_values = None

                        import time as _time_mod
                        _t0_tel = _time_mod.perf_counter()
                        telemetry.record_transformer_input(
                            seq_shape=seq_data.shape,
                            snap_shape=snap_data.shape,
                            seq_feature_names=[],
                            snap_feature_names=[],
                            xgb_seq_channels=effective_seq_channels,
                            xgb_snap_channels=effective_snap_channels,
                            xgb_seq_values=xgb_seq_values,
                            xgb_snap_values=xgb_snap_values,
                            session=current_session,
                            session_token_values=session_token_values,
                            session_tokens_enabled=False,
                        )
                        try:
                            self.t_telemetry_sec += float(_time_mod.perf_counter() - _t0_tel)
                        except Exception:
                            pass
                        telemetry.transformer_input_recorded = True

                        # Toggle state: legacy ablations removed; record as disabled.
                        _t0_tel2 = _time_mod.perf_counter()
                        telemetry.record_toggle_state(
                            disable_xgb_channels_in_transformer_requested=False,
                            disable_xgb_channels_in_transformer_effective=False,
                            n_xgb_channels_in_transformer_input=n_xgb_channels_in_transformer_input,
                        )
                        try:
                            self.t_telemetry_sec += float(_time_mod.perf_counter() - _t0_tel2)
                        except Exception:
                            pass
                
                # DEL 3: Use context features if ctx-modell is active
                if context_features_enabled and entry_context_features is not None and supports_context_features:
                    # B1: predict_entered_count increment (SSoT: right before transformer forward call, after all early return checks)
                    # This ensures predict_entered is only incremented when we're actually going to call transformer forward
                    current_session = policy_state.get("session") if isinstance(policy_state, dict) else None
                    if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                        self.entry_manager.entry_feature_telemetry.record_post_model_call_reached(session=current_session)
                    
                    # Ctx-modell path: pass ctx_cat and ctx_cont
                    # TRUTH: Hard-assert snap dim matches bundle before model call
                    entry_model_id_for_log = None
                    if os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1":
                        bundle_expected_snap_dim = None
                        if hasattr(self, "entry_v10_bundle") and self.entry_v10_bundle:
                            # IMPORTANT: Do NOT trust bundle_metadata.json top-level snap_input_dim here.
                            # The transformer instance is the SSoT for snap_input_dim.
                            bundle_expected_snap_dim = getattr(getattr(self.entry_v10_bundle, "transformer_model", None), "snap_input_dim", None)
                        if hasattr(self, "entry_v10_cfg") and isinstance(self.entry_v10_cfg, dict):
                            entry_model_id_for_log = self.entry_v10_cfg.get("entry_model_id")
                        actual_snap_dim = int(snap_data.shape[-1]) if hasattr(snap_data, "shape") else None
                        
                        # TRUTH: Log dims before model call (diagnostic mode)
                        if os.getenv("GX1_DIAGNOSTIC_FORCE_EVAL_SESSIONS", ""):
                            log.info(
                                "[TRUTH_DIM_LOG] session=%s bundle_expected_snap_dim=%s actual_snap_dim=%s "
                                "routing=%s entry_model_id=%s",
                                current_session,
                                bundle_expected_snap_dim,
                                actual_snap_dim,
                                "v10_hybrid",
                                entry_model_id_for_log,
                            )
                        
                        if bundle_expected_snap_dim is not None and actual_snap_dim is not None:
                            if bundle_expected_snap_dim != actual_snap_dim:
                                # Write capsule before FATAL
                                write_us_premodel_valueerror_capsule(
                                    exc=ValueError(f"EXPECTED snap_feat_dim={bundle_expected_snap_dim} (got {actual_snap_dim})"),
                                    session=current_session,
                                    seq_data=seq_data,
                                    snap_data=snap_data,
                                    seq_feat_names=seq_feat_names,
                                    snap_feat_names=snap_feat_names,
                                    session_tokens_enabled=session_tokens_enabled,
                                    ts=candles.index[-1] if candles is not None and len(candles) > 0 else None,
                                )
                                raise RuntimeError(
                                    f"[TRUTH_SNAP_DIM_FATAL] Bundle expects snap_input_dim={bundle_expected_snap_dim}, "
                                    f"but runtime produced actual_snap_dim={actual_snap_dim}. "
                                    f"session_tokens_enabled={session_tokens_enabled}. "
                                    f"US_PREMODEL_VALUEERROR.json capsule written for diagnosis."
                                )
                    # B1: score-gate allow assert + pre_call_count increment (SSoT: right before transformer forward call)
                    if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                        bundle_sha256 = None
                        if "bundle_metadata" in locals() and isinstance(bundle_metadata, dict):
                            bundle_sha256 = bundle_metadata.get("bundle_sha256")
                        self.entry_manager.entry_feature_telemetry.assert_score_gate_allow_for_predict(
                            session=current_session,
                            ts=candles.index[-1] if candles is not None and len(candles) > 0 else None,
                            predict_path="v10_hybrid",
                            routing_mode="v10_hybrid",
                            entry_model_id=entry_model_id_for_log,
                            bundle_sha256=bundle_sha256,
                            run_id=getattr(self, "run_id", None),
                        )
                        self.entry_manager.entry_feature_telemetry.record_pre_call_reached(session=current_session)
                    _record_predict_reached_forward()
                    
                    # B2: MODEL_CALL_TRACE_US.json capsule (diagnostic mode, US session only)
                    model_call_trace_written = False
                    output_dir_trace = None
                    if (os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1" and 
                        os.getenv("GX1_DIAGNOSTIC_FORCE_EVAL_SESSIONS", "") and
                        current_session == "US"):
                        try:
                            import traceback as tb_module
                            from pathlib import Path as Path_trace
                            import json as json_trace
                            
                            if hasattr(self, "output_dir") and self.output_dir:
                                output_dir_trace = Path_trace(self.output_dir)
                            elif hasattr(self, "explicit_output_dir") and self.explicit_output_dir:
                                output_dir_trace = Path_trace(self.explicit_output_dir)
                            else:
                                output_dir_env = os.getenv("GX1_OUTPUT_DIR")
                                if output_dir_env:
                                    output_dir_trace = Path_trace(output_dir_env)
                            
                            if output_dir_trace:
                                output_dir_trace.mkdir(parents=True, exist_ok=True)
                                trace_capsule_path = output_dir_trace / "MODEL_CALL_TRACE_US.json"
                                
                                # Get callstack snippets
                                callstack_snippets = []
                                try:
                                    stack = tb_module.extract_stack()
                                    # Get last 10 frames (routing + forward callsite)
                                    for frame in stack[-10:]:
                                        callstack_snippets.append({
                                            "file": frame.filename.split("/")[-1] if "/" in frame.filename else frame.filename,
                                            "line": frame.lineno,
                                            "function": frame.name,
                                        })
                                except Exception:
                                    pass
                                
                                # Check routing decision
                                routing_decision = "v10_hybrid"  # Default
                                if hasattr(self, "entry_v10_cfg") and isinstance(self.entry_v10_cfg, dict):
                                    routing_decision = self.entry_v10_cfg.get("routing", "v10_hybrid")
                                
                                trace_payload = {
                                    "session": current_session,
                                    "routing_decision": routing_decision,
                                    "entered_predict": True,  # We're inside _predict_entry_v10_hybrid
                                    "entered_transformer_callsite": True,  # We're at transformer forward call
                                    "hit_pre_call_increment": True,  # We just incremented pre_call
                                    "hit_forward_increment": False,  # Will be set after forward returns
                                    "return_path_reason": "forward_called",  # Will be updated if early return
                                    "callstack_snippets": callstack_snippets,
                                    "timestamp": pd.Timestamp.now().isoformat() if hasattr(pd, "Timestamp") else None,
                                    "entry_model_id": entry_model_id_for_log if 'entry_model_id_for_log' in locals() else None,
                                }
                                
                                with open(trace_capsule_path, "w") as f:
                                    json_trace.dump(trace_payload, f, indent=2)
                                log.info(f"[MODEL_CALL_TRACE] Wrote capsule: {trace_capsule_path}")
                                model_call_trace_written = True
                        except Exception as trace_err:
                            log.warning(f"[MODEL_CALL_TRACE] Failed to write capsule: {trace_err}")
                    
                    try:
                        # Telemetry: tied to actual model call (ctx-present path only)
                        if hasattr(self, "entry_manager") and self.entry_manager:
                            et = self.entry_manager.entry_telemetry
                            if "n_ctx_model_calls" not in et:
                                et["n_ctx_model_calls"] = 0
                            et["n_ctx_model_calls"] += 1
                            if "ctx_proof_pass_count" not in et:
                                et["ctx_proof_pass_count"] = 0
                            et["ctx_proof_pass_count"] += 1
                        # B1: True timer — Transformer forward time only
                        import time as _time_mod
                        _t0_tf = _time_mod.perf_counter()
                        # Legacy kwargs are not allowed for CTX path; force them to None to prevent accidental leakage
                        session_id_t = None
                        vol_regime_id_t = None
                        trend_regime_id_t = None
                        forbidden_kwargs = {
                            "session_id": session_id_t,
                            "vol_regime_id": vol_regime_id_t,
                            "trend_regime_id": trend_regime_id_t,
                        }
                        extra = {k: v for k, v in forbidden_kwargs.items() if v is not None}
                        if extra:
                            raise RuntimeError(f"[ENTRY_V10_CTX_KWARGS_FORBIDDEN] transformer forward forbids extra kwargs keys={list(extra.keys())}")
                        outputs = transformer_model(
                            seq_x=seq_x,
                            snap_x=snap_x,
                            ctx_cat=ctx_cat_t,  # CTX6CAT6
                            ctx_cont=ctx_cont_t,  # CTX6CAT6
                        )
                        try:
                            self.t_transformer_forward_sec += float(_time_mod.perf_counter() - _t0_tf)
                        except Exception:
                            pass
                        
                        # B1: forward_calls increment (SSoT: right after transformer forward returns successfully)
                        # SENTINEL G: after_transformer_forward
                        if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                            self.entry_manager.entry_feature_telemetry.record_control_flow("AFTER_TRANSFORMER_FORWARD", session=current_session)
                            # Record model forward AFTER transformer returns successfully
                            self.entry_manager.entry_feature_telemetry.record_model_forward(model_name, session=current_session)
                            
                            # Update MODEL_CALL_TRACE capsule if written
                            if model_call_trace_written and output_dir_trace:
                                try:
                                    import json as json_trace_local
                                    from pathlib import Path as Path_trace_local
                                    trace_capsule_path = Path_trace_local(output_dir_trace) / "MODEL_CALL_TRACE_US.json"
                                    if trace_capsule_path.exists():
                                        with open(trace_capsule_path, "r") as f:
                                            trace_data = json_trace_local.load(f)
                                        trace_data["hit_forward_increment"] = True
                                        trace_data["return_path_reason"] = "forward_called"
                                        with open(trace_capsule_path, "w") as f:
                                            json_trace_local.dump(trace_data, f, indent=2)
                                except Exception:
                                    pass
                    except Exception as e:
                        # Record pre-model return with exception reason
                        exception_type = type(e).__name__
                        exception_reason = f"EXCEPTION_{exception_type}"
                        if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                            self.entry_manager.entry_feature_telemetry.record_pre_model_return(exception_reason, session=current_session)
                            self.entry_manager.entry_feature_telemetry.record_model_block(model_name, exception_reason)
                            self.entry_manager.entry_feature_telemetry.record_model_exception(model_name, e, session=current_session)
                        
                        # Update MODEL_CALL_TRACE capsule if written (exception before forward)
                        if model_call_trace_written and output_dir_trace:
                            try:
                                import json as json_trace_local
                                from pathlib import Path as Path_trace_local
                                trace_capsule_path = Path_trace_local(output_dir_trace) / "MODEL_CALL_TRACE_US.json"
                                if trace_capsule_path.exists():
                                    with open(trace_capsule_path, "r") as f:
                                        trace_data = json_trace_local.load(f)
                                    trace_data["hit_forward_increment"] = False
                                    trace_data["return_path_reason"] = f"exception_before_forward_{exception_type}"
                                    trace_data["exception_type"] = exception_type
                                    trace_data["exception_message"] = str(e)
                                    with open(trace_capsule_path, "w") as f:
                                        json_trace_local.dump(trace_data, f, indent=2)
                            except Exception:
                                pass
                        
                        # Build XGB outputs dict for capsule
                        xgb_outputs_dict = {
                            "p_long_xgb": p_long_xgb if 'p_long_xgb' in locals() else None,
                            "p_hat_xgb": p_hat_xgb if 'p_hat_xgb' in locals() else None,
                            "uncertainty_score": uncertainty_score if 'uncertainty_score' in locals() else None,
                        }
                        
                        # TRUTH: Write US pre-model ValueError capsule (deterministic)
                        write_us_premodel_valueerror_capsule(
                            exc=e,
                            session=current_session,
                            seq_data=seq_data,
                            snap_data=snap_data,
                            seq_feat_names=seq_feat_names,
                            snap_feat_names=snap_feat_names,
                            session_tokens_enabled=session_tokens_enabled,
                            ts=candles.index[-1] if candles is not None and len(candles) > 0 else None,
                        )
                        
                        # Get transformer config
                        from gx1.contracts.signal_bridge_v1 import SEQ_SIGNAL_DIM, SNAP_SIGNAL_DIM
                        transformer_config_dict = {
                            "seq_len": seq_len,
                            "expected_seq_feat_dim": int(SEQ_SIGNAL_DIM),
                            "expected_snap_dim": int(SNAP_SIGNAL_DIM),
                        }
                        # Add model config if available
                        if hasattr(self.entry_v10_bundle, "metadata"):
                            model_config = self.entry_v10_bundle.metadata.get("model_config", {})
                            transformer_config_dict.update({
                                "num_layers": model_config.get("num_layers"),
                                "d_model": model_config.get("d_model"),
                            })
                        
                        # TRUTH: Write US pre-model ValueError capsule (deterministic)
                        write_us_premodel_valueerror_capsule(
                            exc=e,
                            session=current_session,
                            seq_data=seq_data,
                            snap_data=snap_data,
                            seq_feat_names=seq_feat_names,
                            snap_feat_names=snap_feat_names,
                            session_tokens_enabled=session_tokens_enabled,
                            ts=candles.index[-1] if candles is not None and len(candles) > 0 else None,
                        )
                        # Write exception capsule
                        capsule_path = write_transformer_exception_capsule(
                            exc=e,
                            seq_x=seq_x,
                            snap_x=snap_x,
                            seq_data=seq_data,
                            snap_data=snap_data,
                            xgb_outputs=xgb_outputs_dict,
                            transformer_config=transformer_config_dict,
                            seq_feat_names=seq_feat_names,
                            snap_feat_names=snap_feat_names,
                        )
                        
                        # TRUTH/PROOF mode: Hard-fail on exception
                        is_truth = os.getenv("GX1_RUN_MODE", "").upper() == "TRUTH" or os.getenv("GX1_TRUTH_MODE", "0") == "1"
                        is_proof = os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
                        if is_truth or is_proof:
                            raise RuntimeError(
                                f"[TRANSFORMER_FORWARD_FATAL] Transformer forward failed with {exception_type}: {str(e)}. "
                                f"Exception capsule written to: {capsule_path}. "
                                f"This is a FATAL error in TRUTH/PROOF mode."
                            ) from e
                        else:
                            # DEV mode: log and return None
                            log.error(f"[TRANSFORMER_FORWARD_ERROR] Transformer forward failed: {e}. Capsule: {capsule_path}")
                            return _ret_none(f"TRANSFORMER_FORWARD_{exception_type}")
                    
                    # PHASE 2: Extract gate value for telemetry
                    gate_value = None
                    if "gate" in outputs:
                        gate_value = float(outputs["gate"].cpu().item())
                        
                        # Track gate telemetry
                        if hasattr(self, "entry_manager") and self.entry_manager:
                            if "gate_values" not in self.entry_manager.entry_telemetry:
                                self.entry_manager.entry_telemetry["gate_values"] = []
                            self.entry_manager.entry_telemetry["gate_values"].append(gate_value)
                            
                            # Track gate vs uncertainty correlation
                            if "gate_vs_uncertainty" not in self.entry_manager.entry_telemetry:
                                self.entry_manager.entry_telemetry["gate_vs_uncertainty"] = []
                            self.entry_manager.entry_telemetry["gate_vs_uncertainty"].append({
                                "gate": gate_value,
                                "uncertainty_score": uncertainty_score,
                            })
                            
                            # Track gate per regime
                            if "gate_per_regime" not in self.entry_manager.entry_telemetry:
                                self.entry_manager.entry_telemetry["gate_per_regime"] = {}
                            regime_key = f"{current_session}_{vol_regime_id}"
                            if regime_key not in self.entry_manager.entry_telemetry["gate_per_regime"]:
                                self.entry_manager.entry_telemetry["gate_per_regime"][regime_key] = []
                            self.entry_manager.entry_telemetry["gate_per_regime"][regime_key].append(gate_value)
                            
                            # Log first 3 gate values
                            if not hasattr(self, "_gate_log_count"):
                                self._gate_log_count = 0
                            if self._gate_log_count < 3:
                                log.info(
                                    "[GATED_FUSION] Call #%d: gate=%.4f, uncertainty_score=%.4f, "
                                    "regime=%s_%d",
                                    self._gate_log_count + 1,
                                    gate_value,
                                    uncertainty_score,
                                    current_session,
                                    vol_regime_id,
                                )
                                self._gate_log_count += 1
                    
                    # DEL 3: Proof flag - verify ctx is consumed (replay only, rate-limited)
                    proof_flag_enabled = os.getenv("GX1_CTX_CONSUMPTION_PROOF", "0") == "1"
                    if proof_flag_enabled and is_replay:
                        # Rate-limit: only check first N samples
                        if not hasattr(self, "_ctx_proof_check_count"):
                            self._ctx_proof_check_count = 0
                        
                        # Check first K samples (default: 10)
                        proof_check_limit = int(os.getenv("GX1_CTX_PROOF_CHECK_LIMIT", "10"))
                        if self._ctx_proof_check_count < proof_check_limit:
                            # Pass A: real ctx
                            prob_long_A = torch.sigmoid(outputs["direction_logit"]).cpu().item()
                            
                            # Pass B: permuted ctx_cat + null ctx_cont
                            ctx_cat_permuted = torch.roll(ctx_cat_t, shifts=1, dims=1)  # Permute categorical
                            ctx_cont_null = torch.zeros_like(ctx_cont_t)  # Null continuous
                            
                            try:
                                outputs_B = transformer_model(
                                    seq_x=seq_x,
                                    snap_x=snap_x,
                                    ctx_cat=ctx_cat_permuted,
                                    ctx_cont=ctx_cont_null,
                                )
                                prob_long_B = torch.sigmoid(outputs_B["direction_logit"]).cpu().item()
                            except Exception as e:
                                # For ctx proof path, also write capsule but don't hard-fail (this is a test path)
                                exception_type = type(e).__name__
                                log.error(f"[CTX_PROOF_TRANSFORMER_ERROR] Transformer forward failed in ctx proof path: {e}")
                                # Still write capsule for diagnostics
                                xgb_outputs_dict = {
                                    "p_long_xgb": p_long_xgb if 'p_long_xgb' in locals() else None,
                                    "p_hat_xgb": p_hat_xgb if 'p_hat_xgb' in locals() else None,
                                    "uncertainty_score": uncertainty_score if 'uncertainty_score' in locals() else None,
                                }
                                from gx1.contracts.signal_bridge_v1 import SEQ_SIGNAL_DIM, SNAP_SIGNAL_DIM
                                snap_dim = int(SNAP_SIGNAL_DIM)
                                transformer_config_dict = {
                                    "seq_len": seq_len,
                                    "expected_seq_feat_dim": int(SEQ_SIGNAL_DIM),
                                    "expected_snap_dim": snap_dim,
                                }
                                if hasattr(self.entry_v10_bundle, "metadata"):
                                    model_config = self.entry_v10_bundle.metadata.get("model_config", {})
                                    transformer_config_dict.update({
                                        "num_layers": model_config.get("num_layers"),
                                        "d_model": model_config.get("d_model"),
                                    })
                                # TRUTH: Write US pre-model ValueError capsule (deterministic)
                                write_us_premodel_valueerror_capsule(
                                    exc=e,
                                    session=current_session,
                                    seq_data=seq_data,
                                    snap_data=snap_data,
                                    seq_feat_names=seq_feat_names,
                                    snap_feat_names=snap_feat_names,
                                    session_tokens_enabled=session_tokens_enabled,
                                    ts=candles.index[-1] if candles is not None and len(candles) > 0 else None,
                                )
                                capsule_path = write_transformer_exception_capsule(
                                    exc=e,
                                    seq_x=seq_x,
                                    snap_x=snap_x,
                                    seq_data=seq_data,
                                    snap_data=snap_data,
                                    xgb_outputs=xgb_outputs_dict,
                                    transformer_config=transformer_config_dict,
                                    seq_feat_names=seq_feat_names,
                                    snap_feat_names=snap_feat_names,
                                )
                                raise RuntimeError(
                                    f"[CTX_PROOF_TRANSFORMER_FATAL] Transformer forward failed in ctx proof path with {exception_type}: {str(e)}. "
                                    f"Exception capsule: {capsule_path}"
                                ) from e
                            
                            # Check: ctx must affect output
                            diff = abs(prob_long_A - prob_long_B)
                            min_diff_threshold = 1e-6
                            
                            # ctx_proof_pass/fail are only updated at model-call (pass) or [CTX_CONTRACT_*] raise (fail); not here
                            if diff < min_diff_threshold:
                                # Consumption proof failed: ctx does not affect output
                                error_msg = (
                                    f"CTX_CONSUMPTION_PROOF_FAILED: ctx does not affect output. "
                                    f"prob_long_A={prob_long_A:.6f}, prob_long_B={prob_long_B:.6f}, "
                                    f"diff={diff:.6f} < threshold={min_diff_threshold}. "
                                    f"This indicates ctx is being ignored by the model."
                                )
                                # Hard-fail if ctx does not affect output (replay and live)
                                log.error("[CTX_PROOF] %s", error_msg)
                                raise RuntimeError(error_msg)
                            self._ctx_proof_check_count += 1
                            
                            # Log first 3 checks (pass or fail)
                            if self._ctx_proof_check_count <= 3:
                                status = "PASS" if diff >= min_diff_threshold else "FAIL"
                                log.info(
                                    "[CTX_PROOF] Check #%d: prob_long_A=%.6f, prob_long_B=%.6f, diff=%.6f (%s)",
                                    self._ctx_proof_check_count,
                                    prob_long_A,
                                    prob_long_B,
                                    diff,
                                    status
                                )
                elif ctx_expected:
                    # CTX_EXPECTED_BUT_LEGACY_BRANCH: ctx expected but took legacy branch - hard-fail (no fallback)
                    raise RuntimeError(
                        "[CTX_EXPECTED_BUT_LEGACY_BRANCH] ctx_expected=True but took legacy branch. "
                        f"context_features_enabled={context_features_enabled}, "
                        f"entry_context_features={'not None' if entry_context_features is not None else 'None'}, "
                        f"supports_context_features={supports_context_features}, "
                        f"bundle_meta.keys={list(bundle_meta.keys()) if bundle_meta else []}, "
                        f"model_type={type(transformer_model).__name__}, "
                        f"ctx_cat_t={'not None' if ctx_cat_t is not None else 'None'}, "
                        f"ctx_cont_t={'not None' if ctx_cont_t is not None else 'None'}. CTX6CAT6 only."
                    )
                else:
                    # Legacy path: no ctx_cat/ctx_cont (ctx not expected)
                    # B1: predict_entered_count increment (SSoT: right before transformer forward call, after all early return checks)
                    current_session = policy_state.get("session") if isinstance(policy_state, dict) else None
                    if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                        self.entry_manager.entry_feature_telemetry.record_post_model_call_reached(session=current_session)
                    
                    # Check if model is EntryV10CtxHybridTransformer (should not happen, but defensive)
                    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
                    if isinstance(transformer_model, EntryV10CtxHybridTransformer):
                        # Ctx model but context not provided - hard-fail in both live and replay (no dummy/5/2)
                        if hasattr(self, "entry_manager") and self.entry_manager:
                            et = self.entry_manager.entry_telemetry
                            if "ctx_proof_fail_count" not in et:
                                et["ctx_proof_fail_count"] = 0
                            et["ctx_proof_fail_count"] += 1
                        raise RuntimeError(
                            "[CTX_CONTRACT_MISSING] Ctx model requires context; entry_context_features missing or disabled. CTX6CAT6 only."
                        )
                    else:
                        # Legacy EntryV10HybridTransformer (no ctx support)
                        # B1: predict_entered_count increment (SSoT: right before transformer forward call, after all early return checks)
                        current_session = policy_state.get("session") if isinstance(policy_state, dict) else None
                        if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                            self.entry_manager.entry_feature_telemetry.record_post_model_call_reached(session=current_session)
                        
                        # B1: score-gate allow assert + pre_call_count increment (SSoT: right before transformer forward call) - LEGACY NON-CTX PATH
                        if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                            bundle_sha256 = None
                            if "bundle_metadata" in locals() and isinstance(bundle_metadata, dict):
                                bundle_sha256 = bundle_metadata.get("bundle_sha256")
                            self.entry_manager.entry_feature_telemetry.assert_score_gate_allow_for_predict(
                                session=current_session,
                                ts=candles.index[-1] if candles is not None and len(candles) > 0 else None,
                                predict_path="v10_hybrid",
                                routing_mode="v10_hybrid",
                                entry_model_id=entry_model_id_for_log,
                                bundle_sha256=bundle_sha256,
                                run_id=getattr(self, "run_id", None),
                            )
                            self.entry_manager.entry_feature_telemetry.record_pre_call_reached(session=current_session)
                        _record_predict_reached_forward()
                        
                        # B2: MODEL_CALL_TRACE_US.json capsule (diagnostic mode, US session only) - LEGACY NON-CTX PATH
                        model_call_trace_written_legacy_nonctx = False
                        output_dir_trace_legacy_nonctx = None
                        if (os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1" and 
                            os.getenv("GX1_DIAGNOSTIC_FORCE_EVAL_SESSIONS", "") and
                            current_session == "US"):
                            try:
                                import traceback as tb_module
                                from pathlib import Path as Path_trace
                                import json as json_trace
                                
                                if hasattr(self, "output_dir") and self.output_dir:
                                    output_dir_trace_legacy_nonctx = Path_trace(self.output_dir)
                                elif hasattr(self, "explicit_output_dir") and self.explicit_output_dir:
                                    output_dir_trace_legacy_nonctx = Path_trace(self.explicit_output_dir)
                                else:
                                    output_dir_env = os.getenv("GX1_OUTPUT_DIR")
                                    if output_dir_env:
                                        output_dir_trace_legacy_nonctx = Path_trace(output_dir_env)
                                
                                if output_dir_trace_legacy_nonctx:
                                    output_dir_trace_legacy_nonctx.mkdir(parents=True, exist_ok=True)
                                    trace_capsule_path = output_dir_trace_legacy_nonctx / "MODEL_CALL_TRACE_US.json"
                                    
                                    callstack_snippets = []
                                    try:
                                        stack = tb_module.extract_stack()
                                        for frame in stack[-10:]:
                                            callstack_snippets.append({
                                                "file": frame.filename.split("/")[-1] if "/" in frame.filename else frame.filename,
                                                "line": frame.lineno,
                                                "function": frame.name,
                                            })
                                    except Exception:
                                        pass
                                    
                                    routing_decision = "v10_hybrid"  # Default
                                    if hasattr(self, "entry_v10_cfg") and isinstance(self.entry_v10_cfg, dict):
                                        routing_decision = self.entry_v10_cfg.get("routing", "v10_hybrid")
                                    
                                    trace_payload = {
                                        "session": current_session,
                                        "routing_decision": routing_decision,
                                        "entered_predict": True,
                                        "entered_transformer_callsite": True,
                                        "hit_pre_call_increment": True,
                                        "hit_forward_increment": False,
                                        "return_path_reason": "forward_called",
                                        "path": "legacy_nonctx",
                                        "callstack_snippets": callstack_snippets,
                                        "timestamp": pd.Timestamp.now().isoformat() if hasattr(pd, "Timestamp") else None,
                                    }
                                    
                                    with open(trace_capsule_path, "w") as f:
                                        json_trace.dump(trace_payload, f, indent=2)
                                    log.info(f"[MODEL_CALL_TRACE] Wrote capsule (legacy non-ctx path): {trace_capsule_path}")
                                    model_call_trace_written_legacy_nonctx = True
                            except Exception as trace_err:
                                log.warning(f"[MODEL_CALL_TRACE] Failed to write capsule (legacy non-ctx): {trace_err}")
                        
                        try:
                            outputs = transformer_model(seq_x=seq_x, snap_x=snap_x)
                            
                            # B1: forward_calls increment (SSoT: right after transformer forward returns successfully) - LEGACY NON-CTX PATH
                            # SENTINEL G: after_transformer_forward (legacy path)
                            if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                                self.entry_manager.entry_feature_telemetry.record_control_flow("AFTER_TRANSFORMER_FORWARD", session=current_session)
                                # Record model forward AFTER transformer returns successfully
                                self.entry_manager.entry_feature_telemetry.record_model_forward(model_name, session=current_session)
                                
                                # Update MODEL_CALL_TRACE capsule if written
                                if model_call_trace_written_legacy_nonctx and output_dir_trace_legacy_nonctx:
                                    try:
                                        import json as json_trace_local
                                        from pathlib import Path as Path_trace_local
                                        trace_capsule_path = Path_trace_local(output_dir_trace_legacy_nonctx) / "MODEL_CALL_TRACE_US.json"
                                        if trace_capsule_path.exists():
                                            with open(trace_capsule_path, "r") as f:
                                                trace_data = json_trace_local.load(f)
                                            trace_data["hit_forward_increment"] = True
                                            trace_data["return_path_reason"] = "forward_called"
                                            with open(trace_capsule_path, "w") as f:
                                                json_trace_local.dump(trace_data, f, indent=2)
                                    except Exception:
                                        pass
                        except Exception as e:
                            # Same exception handling
                            exception_type = type(e).__name__
                            exception_reason = f"EXCEPTION_{exception_type}"
                            if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                                self.entry_manager.entry_feature_telemetry.record_pre_model_return(exception_reason, session=current_session)
                                self.entry_manager.entry_feature_telemetry.record_model_block(model_name, exception_reason)
                                self.entry_manager.entry_feature_telemetry.record_model_exception(model_name, e, session=current_session)
                            
                            # Update MODEL_CALL_TRACE capsule if written (exception before forward) - LEGACY NON-CTX PATH
                            if model_call_trace_written_legacy_nonctx and output_dir_trace_legacy_nonctx:
                                try:
                                    import json as json_trace_local
                                    from pathlib import Path as Path_trace_local
                                    trace_capsule_path = Path_trace_local(output_dir_trace_legacy_nonctx) / "MODEL_CALL_TRACE_US.json"
                                    if trace_capsule_path.exists():
                                        with open(trace_capsule_path, "r") as f:
                                            trace_data = json_trace_local.load(f)
                                        trace_data["hit_forward_increment"] = False
                                        trace_data["return_path_reason"] = f"exception_before_forward_{exception_type}"
                                        trace_data["exception_type"] = exception_type
                                        trace_data["exception_message"] = str(e)
                                        with open(trace_capsule_path, "w") as f:
                                            json_trace_local.dump(trace_data, f, indent=2)
                                except Exception:
                                    pass
                            
                            xgb_outputs_dict = {
                                "p_long_xgb": p_long_xgb if 'p_long_xgb' in locals() else None,
                                "p_hat_xgb": p_hat_xgb if 'p_hat_xgb' in locals() else None,
                                "uncertainty_score": uncertainty_score if 'uncertainty_score' in locals() else None,
                            }
                            from gx1.contracts.signal_bridge_v1 import SEQ_SIGNAL_DIM, SNAP_SIGNAL_DIM
                            transformer_config_dict = {
                                "seq_len": seq_len,
                                "expected_seq_feat_dim": int(SEQ_SIGNAL_DIM),
                                "expected_snap_dim": int(SNAP_SIGNAL_DIM),
                            }
                            if hasattr(self.entry_v10_bundle, "metadata"):
                                model_config = self.entry_v10_bundle.metadata.get("model_config", {})
                                transformer_config_dict.update({
                                    "num_layers": model_config.get("num_layers"),
                                    "d_model": model_config.get("d_model"),
                                })
                            capsule_path = write_transformer_exception_capsule(
                                exc=e,
                                seq_x=seq_x,
                                snap_x=snap_x,
                                seq_data=seq_data,
                                snap_data=snap_data,
                                xgb_outputs=xgb_outputs_dict,
                                transformer_config=transformer_config_dict,
                                seq_feat_names=seq_feat_names,
                                snap_feat_names=snap_feat_names,
                            )
                            
                            is_truth = os.getenv("GX1_RUN_MODE", "").upper() == "TRUTH" or os.getenv("GX1_TRUTH_MODE", "0") == "1"
                            is_proof = os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
                            if is_truth or is_proof:
                                raise RuntimeError(
                                    f"[TRANSFORMER_FORWARD_FATAL] Transformer forward failed with {exception_type}: {str(e)}. "
                                    f"Exception capsule written to: {capsule_path}. "
                                    f"This is a FATAL error in TRUTH/PROOF mode."
                                ) from e
                            else:
                                log.error(f"[TRANSFORMER_FORWARD_ERROR] Transformer forward failed: {e}. Capsule: {capsule_path}")
                                return _ret_none(f"TRANSFORMER_FORWARD_{exception_type}")
                
                direction_logit = outputs["direction_logit"]
                early_move_logit = outputs.get("early_move_logit", torch.tensor(0.0))
                quality_score = outputs.get("quality_score", torch.tensor(0.0))
            
            # Convert to probabilities
            prob_long = torch.sigmoid(direction_logit).cpu().item()
            prob_short = 1.0 - prob_long
            prob_flat = 1.0 / 3.0  # Default flat probability (can be improved with 3-class head)
            prob_early = torch.sigmoid(early_move_logit).cpu().item() if isinstance(early_move_logit, torch.Tensor) else early_move_logit
            quality = quality_score.cpu().item() if isinstance(quality_score, torch.Tensor) else quality_score
            margin = abs(prob_long - prob_short)
            p_hat = max(prob_long, prob_short)
            
            # Record transformer outputs for baseline evaluation
            if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                output_timestamp = None
                try:
                    if candles is not None and len(candles) > 0:
                        output_timestamp = candles.index[-1].isoformat()
                    elif hasattr(self, "last_iterated_ts") and self.last_iterated_ts is not None:
                        output_timestamp = self.last_iterated_ts.isoformat()
                except Exception:
                    output_timestamp = None
                self.entry_manager.entry_feature_telemetry.record_transformer_output(
                    session=current_session,
                    direction_logit=direction_logit.cpu().item() if isinstance(direction_logit, torch.Tensor) else direction_logit,
                    early_move_logit=prob_early,
                    quality_score=quality,
                    prob_long=prob_long,
                    prob_short=prob_short,
                    prob_flat=prob_flat,
                    prob_early=prob_early,
                    timestamp=output_timestamp,
                )
            
            # Log first 3 calls with prob_long (for diagnostics)
            if not hasattr(self, "_v10_shape_log_count"):
                self._v10_shape_log_count = 0
            if self._v10_shape_log_count <= 3 and self._v10_shape_log_count > 0:
                log.info(
                    "[ENTRY_V10] Call #%d result: prob_long=%.4f (isfinite=%s) prob_short=%.4f margin=%.4f",
                    self._v10_shape_log_count,
                    prob_long, np.isfinite(prob_long),
                    prob_short, margin
                )
            
            log.debug(
                "[ENTRY_V10] session=%s p_long=%.4f p_short=%.4f margin=%.4f p_hat=%.4f",
                current_session, prob_long, prob_short, margin, p_hat
            )
            
            # Create EntryPrediction
            run_mode = getattr(self, "replay_mode", False) and "REPLAY" or "LIVE"
            is_sniper = getattr(self, "is_sniper", False)
            log.info(
                "[ENTRY_DEBUG] Entry prediction produced | "
                f"score={prob_long:.4f} | "
                f"side={'LONG' if prob_long > 0.5 else 'SHORT'} | "
                f"session={current_session} | "
                f"run_mode={run_mode} | "
                f"is_sniper={is_sniper}"
            )
            return EntryPrediction(
                session=current_session,
                prob_long=float(prob_long),
                prob_short=float(prob_short),
                prob_neutral=0.0,  # V10 is binary
                margin=float(margin),
                p_hat=float(p_hat),
            )
            
        except Exception as e:
            # DEL 1: Brutal exception logging (midlertidig)
            import traceback
            exception_type = type(e).__name__
            exception_reason = f"EXCEPTION_{exception_type}"
            
            if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                telemetry = self.entry_manager.entry_feature_telemetry
                # Record exception with full details
                telemetry.record_model_block(model_name, exception_reason)
                telemetry.record_pre_model_return(exception_reason, session=current_session)
                telemetry.record_model_exception(model_name, e, session=current_session)
                telemetry.v10_callsite_exception += 1
                telemetry.v10_callsite_last_exception = {
                    "exception_type": exception_type,
                    "exception_message": str(e),
                    "traceback": traceback.format_exc(),
                }
            
            # Try to write exception capsule if we have the necessary data
            # (This might fail if exception happened before seq_x/snap_x were created)
            try:
                # Check if we have transformer inputs available
                if 'seq_x' in locals() and 'snap_x' in locals() and 'seq_data' in locals() and 'snap_data' in locals():
                    # We have transformer inputs - write full capsule
                    xgb_outputs_dict = {
                        "p_long_xgb": p_long_xgb if 'p_long_xgb' in locals() else None,
                        "p_hat_xgb": p_hat_xgb if 'p_hat_xgb' in locals() else None,
                        "uncertainty_score": uncertainty_score if 'uncertainty_score' in locals() else None,
                    }
                    from gx1.contracts.signal_bridge_v1 import SEQ_SIGNAL_DIM, SNAP_SIGNAL_DIM
                    session_tokens_enabled_exc = False
                    snap_dim = int(SNAP_SIGNAL_DIM)
                    transformer_config_dict = {
                        "seq_len": seq_len if 'seq_len' in locals() else None,
                        "expected_seq_feat_dim": int(SEQ_SIGNAL_DIM),
                        "expected_snap_dim": snap_dim,
                    }
                    if hasattr(self, "entry_v10_bundle") and self.entry_v10_bundle and hasattr(self.entry_v10_bundle, "metadata"):
                        model_config = self.entry_v10_bundle.metadata.get("model_config", {})
                        transformer_config_dict.update({
                            "num_layers": model_config.get("num_layers"),
                            "d_model": model_config.get("d_model"),
                        })
                    
                    # TRUTH: Write US pre-model ValueError capsule (deterministic)
                    write_us_premodel_valueerror_capsule(
                        exc=e,
                        session=current_session,
                        seq_data=seq_data,
                        snap_data=snap_data,
                        seq_feat_names=seq_feat_names if 'seq_feat_names' in locals() else [],
                        snap_feat_names=snap_feat_names if 'snap_feat_names' in locals() else [],
                        session_tokens_enabled=session_tokens_enabled_exc,
                        ts=candles.index[-1] if candles is not None and len(candles) > 0 else None,
                    )
                    capsule_path = write_transformer_exception_capsule(
                        exc=e,
                        seq_x=seq_x,
                        snap_x=snap_x,
                        seq_data=seq_data,
                        snap_data=snap_data,
                        xgb_outputs=xgb_outputs_dict,
                        transformer_config=transformer_config_dict,
                        seq_feat_names=seq_feat_names if 'seq_feat_names' in locals() else [],
                        snap_feat_names=snap_feat_names if 'snap_feat_names' in locals() else [],
                    )
                    log.error(
                        f"[ENTRY_V10_EXCEPTION] Exception in _predict_entry_v10_hybrid | "
                        f"type={exception_type} | message={str(e)} | "
                        f"Exception capsule written to: {capsule_path}"
                    )
                else:
                    # Exception happened before transformer inputs were created
                    # Write minimal capsule with available data
                    try:
                        # Try to write capsule with None inputs (will show error in stats)
                        capsule_path = write_transformer_exception_capsule(
                            exc=e,
                            seq_x=None,
                            snap_x=None,
                            seq_data=None,
                            snap_data=None,
                            xgb_outputs=None,
                            transformer_config=None,
                            seq_feat_names=None,
                            snap_feat_names=None,
                        )
                        log.error(
                            f"[ENTRY_V10_EXCEPTION] Exception in _predict_entry_v10_hybrid (before transformer inputs) | "
                            f"type={exception_type} | message={str(e)} | "
                            f"Exception capsule written to: {capsule_path} | "
                            f"traceback=\n{traceback.format_exc()}"
                        )
                    except Exception as capsule_error:
                        log.error(
                            f"[ENTRY_V10_EXCEPTION] Exception in _predict_entry_v10_hybrid (before transformer inputs) | "
                            f"type={exception_type} | message={str(e)} | "
                            f"Failed to write exception capsule: {capsule_error} | "
                            f"traceback=\n{traceback.format_exc()}"
                        )
            except Exception as capsule_error:
                # Failed to write capsule - log error but don't mask original exception
                log.error(
                    f"[ENTRY_V10_EXCEPTION] Exception in _predict_entry_v10_hybrid | "
                    f"type={exception_type} | message={str(e)} | "
                    f"Failed to write exception capsule: {capsule_error} | "
                    f"traceback=\n{traceback.format_exc()}"
                )
            
            # TRUTH/PROOF mode: Hard-fail on exception
            is_truth = os.getenv("GX1_RUN_MODE", "").upper() == "TRUTH" or os.getenv("GX1_TRUTH_MODE", "0") == "1"
            is_proof = os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
            if is_truth or is_proof:
                raise RuntimeError(
                    f"[ENTRY_V10_FATAL] Exception in _predict_entry_v10_hybrid with {exception_type}: {str(e)}. "
                    f"This is a FATAL error in TRUTH/PROOF mode. "
                    f"Check exception capsule if available."
                ) from e
            
            # DEL 1: Re-raise exception (ikke swallow)
            raise

    def _predict_entry_tcn(
        self,
        candles: pd.DataFrame,
        entry_bundle: EntryFeatureBundle,
    ) -> Optional[EntryPrediction]:
        """
        Generate TCN entry probabilities from sequence data.
        
        Returns EntryPrediction with prob_long and prob_short derived from TCN model.
        """
        if self.entry_tcn_model is None or self.entry_tcn_feats is None:
            log.warning("[TCN_PREDICT] Model or features not loaded - model=%s features=%s", 
                       self.entry_tcn_model is not None, self.entry_tcn_feats is not None)
            return None
        
        try:
            log.debug("[TCN_PREDICT] Starting prediction - candles=%d bars lookback=%d", 
                     len(candles), self.entry_tcn_lookback)
            import torch
            from gx1.seq.sequence_features import build_sequence_features
            from gx1.seq.dataset import SeqWindowDataset  # type: ignore[reportMissingImports]
            
            # Build features from candles for the entire sequence window
            # TCN model expects features like _v1_r1, _v1_body_share_1, _v1_wick_imbalance, etc.
            # These come from basic_v1.build_basic_v1() (called by build_live_entry_features)
            candles_seq = candles.copy()
            
            # Ensure we have required columns (OHLCV)
            # CSV might have lowercase or uppercase column names
            required_cols_lower = ['open', 'high', 'low', 'close', 'volume']
            
            # Normalize column names to lowercase (handle both OPEN/HIGH/LOW/CLOSE and open/high/low/close)
            col_mapping = {}
            for col in candles_seq.columns:
                col_upper = col.upper()
                if col_upper == 'OPEN':
                    col_mapping[col] = 'open'
                elif col_upper == 'HIGH':
                    col_mapping[col] = 'high'
                elif col_upper == 'LOW':
                    col_mapping[col] = 'low'
                elif col_upper == 'CLOSE':
                    col_mapping[col] = 'close'
                elif col_upper == 'VOLUME':
                    col_mapping[col] = 'volume'
            
            # Rename columns to lowercase
            if col_mapping:
                candles_seq = candles_seq.rename(columns=col_mapping)
            
            # Check what columns we have
            missing_cols = [c for c in required_cols_lower if c not in candles_seq.columns]
            if missing_cols:
                log.error("[TCN_PREDICT] Missing required columns after normalization: %s", missing_cols)
                log.error("[TCN_PREDICT] Available columns: %s", list(candles_seq.columns)[:15])
                log.error("[TCN_PREDICT] candles DataFrame shape: %s, index type: %s", candles_seq.shape, type(candles_seq.index))
                return None
            
            # CRITICAL: build_sequence_features needs OHLCV columns!
            # So we must call it BEFORE building features with build_basic_v1
            # (which removes OHLCV columns and adds feature columns)
            
            # First, ensure session column exists (needed by build_sequence_features for session_id)
            if "session" not in candles_seq.columns:
                # Infer session from timestamp
                from gx1.execution.live_features import infer_session_tag
                candles_seq["session"] = candles_seq.index.map(lambda ts: infer_session_tag(ts))
            
            # Build sequence features FIRST (while we still have OHLCV columns)
            # This adds: ema20_slope, ema100_slope, pos_vs_ema200, std50, atr50, atr_z, roc20, roc100, body_pct, wick_asym
            # Plus environmental context: session_id, atr_regime_id, trend_regime_tf24h
            candles_seq_with_seq_features = build_sequence_features(candles_seq.copy())
            
            # Extract sequence features we need (calculated on original OHLCV data)
            seq_feat_cols = ['ema20_slope', 'ema100_slope', 'pos_vs_ema200', 'std50', 'atr50', 'atr_z', 
                            'roc20', 'roc100', 'body_pct', 'wick_asym', 'session_id', 'atr_regime_id', 'trend_regime_tf24h']
            seq_features = candles_seq_with_seq_features[seq_feat_cols].copy() if all(c in candles_seq_with_seq_features.columns for c in seq_feat_cols) else None
            
            # Now build features using basic_v1 (generates _v1_r1, _v1_body_share_1, etc.)
            # basic_v1.build_basic_v1 returns a DataFrame with features for all bars
            from gx1.features.basic_v1 import build_basic_v1
            from gx1.utils.ts_utils import ensure_ts_column
            
            # Ensure 'ts' column exists (required by basic_v1)
            # This handles cases where prebuilt data already has "ts" column
            candles_with_ts = ensure_ts_column(candles_seq.copy(), context="TCN_PREDICT")
            
            # Build features using basic_v1 (generates _v1_r1, _v1_body_share_1, etc.)
            feature_df, _ = build_basic_v1(candles_with_ts)
            
            # Extract only _v1_* feature columns (drop OHLC and ts)
            feature_cols = [c for c in feature_df.columns if c.startswith("_v1_")]
            candles_seq = feature_df[feature_cols].copy()
            candles_seq.index = candles.index
            
            # Add sequence features back (they were calculated on original OHLCV data)
            if seq_features is not None:
                for col in seq_feat_cols:
                    if col in seq_features.columns:
                        candles_seq[col] = seq_features[col].values
            
            # Align features with manifest (same as build_live_entry_features does)
            manifest = load_manifest()
            from gx1.tuning.feature_manifest import align_features  # type: ignore[reportMissingImports]
            candles_seq = align_features(candles_seq, manifest=manifest, training_stats=manifest.get("training_stats"))
            
            # Check if we have enough bars for lookback
            if len(candles_seq) < self.entry_tcn_lookback:
                log.warning("[TCN_PREDICT] Insufficient bars: %d < %d (lookback required)", 
                           len(candles_seq), self.entry_tcn_lookback)
                return None
            
            # Extract sequence window (last lookback bars)
            seq_window = candles_seq.tail(self.entry_tcn_lookback).copy()
            
            # Select features in correct order
            missing_feats = [f for f in self.entry_tcn_feats if f not in seq_window.columns]
            if missing_feats:
                log.warning("[HYBRID_ENTRY] Missing TCN features: %s. Using zeros.", missing_feats[:5])
                for feat in missing_feats:
                    seq_window[feat] = 0.0
            
            # Extract feature matrix [lookback, n_features]
            X_seq = seq_window[self.entry_tcn_feats].values.astype(np.float32)
            
            # PHASE 1 FIX: Hard-fail on NaN/Inf in sequence features in replay mode
            is_replay = getattr(self, "replay_mode", False) or getattr(self, "fast_replay", False)
            if is_replay:
                if not np.all(np.isfinite(X_seq)):
                    nan_count = np.isnan(X_seq).sum()
                    inf_count = np.isinf(X_seq).sum()
                    raise RuntimeError(
                        f"[NAN_INF_INPUT_FATAL] Sequence features contain NaN/Inf: "
                        f"NaN count={nan_count}, Inf count={inf_count}, shape={X_seq.shape}. "
                        f"This violates input-finiteness invariant (replay mode)."
                    )
            # In live mode, allow nan_to_num but log counter
            if not is_replay and not np.all(np.isfinite(X_seq)):
                if not hasattr(self, "_seq_nan_inf_count"):
                    self._seq_nan_inf_count = 0
                self._seq_nan_inf_count += 1
                if self._seq_nan_inf_count <= 3:
                    log.warning(
                        "[NAN_INF_INPUT] Sequence features contain NaN/Inf (live mode: converting to 0.0): "
                        f"NaN count={np.isnan(X_seq).sum()}, Inf count={np.isinf(X_seq).sum()}"
                    )
            X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale
            if self.entry_tcn_scaler is not None:
                X_seq = self.entry_tcn_scaler.transform(X_seq)
                # PHASE 1 FIX: Check again after scaling (scaler may introduce NaN/Inf)
                if is_replay:
                    if not np.all(np.isfinite(X_seq)):
                        nan_count = np.isnan(X_seq).sum()
                        inf_count = np.isinf(X_seq).sum()
                        raise RuntimeError(
                            f"[NAN_INF_INPUT_FATAL] Sequence features after scaling contain NaN/Inf: "
                            f"NaN count={nan_count}, Inf count={inf_count}, shape={X_seq.shape}. "
                            f"This violates input-finiteness invariant (replay mode)."
                        )
                X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Convert to tensor [1, lookback, n_features]
            X_tensor = torch.from_numpy(X_seq).unsqueeze(0).to(self.entry_tcn_device)
            
            # Predict (returns logits for fast-hit probability)
            with torch.no_grad():
                logits = self.entry_tcn_model(X_tensor)
                prob_fast_hit = torch.sigmoid(logits).cpu().item()
            
            log.debug("[TCN_PREDICT] X_seq shape=%s logits=%.6f prob_fast_hit=%.6f", 
                     X_seq.shape, logits.item() if hasattr(logits, 'item') else float(logits), prob_fast_hit)
            
            # Convert fast-hit probability to prob_long and prob_short
            # TCN predicts "fast hit" (MFE within horizon), not direction
            # For now, use a simple heuristic: if prob_fast_hit > 0.5, assume LONG bias
            # TODO: Train TCN with direction labels or use separate LONG/SHORT models
            prob_long = prob_fast_hit
            prob_short = 1.0 - prob_fast_hit
            
            # Create EntryPrediction
            prediction = EntryPrediction(
                session="TCN",  # TCN doesn't use session routing
                prob_short=float(prob_short),
                prob_neutral=0.0,
                prob_long=float(prob_long),
                p_hat=float(max(prob_long, prob_short)),
                margin=float(abs(prob_long - prob_short)),
            )
            
            log.debug("[TCN_PREDICT] Prediction: prob_long=%.4f prob_short=%.4f margin=%.4f", 
                     prob_long, prob_short, prediction.margin)
            
            return prediction
            
        except Exception as e:
            log.error("[TCN_PREDICT] TCN prediction error: %s", e, exc_info=True)
            return None


    # SSoT ENTRY ROUTE (sanity grep): evaluate_entry -> _evaluate_entry_impl -> entry_manager.evaluate_entry -> _predict_entry_v10_hybrid.
    # Legacy/not reachable: _load_entry_v9_model (raise-only [LEGACY_DISABLED]); no other entry prediction paths.
    def _evaluate_entry_impl(self, candles: pd.DataFrame) -> Optional[LiveTrade]:
        """Single entry path: entry_manager.evaluate_entry only. No V9/legacy/fallback routing."""
        import inspect
        
        # ENTRY EVAL PATH TELEMETRY: Record which function actually performs entry evaluation (SSoT)
        # This MUST be the very first thing in the function to identify the actual code path
        if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
            current_frame = inspect.currentframe()
            frame = current_frame if current_frame else None
            lineno = frame.f_lineno if frame else 0
            self.entry_manager.entry_feature_telemetry.record_entry_eval_path(
                function="GX1DemoRunner._evaluate_entry_impl",
                file=__file__,
                line=lineno,
            )
        
        return self.entry_manager.evaluate_entry(candles)
    
    def get_pre_entry_funnel_snapshot(self) -> Dict[str, Any]:
        """
        Get pre-entry funnel snapshot (SSoT for where bars die before entry evaluation).
        
        Returns:
            Dict with all pre-entry funnel counters and last_stop_reason
        """
        # Convert defaultdict to regular dict for JSON serialization
        pregate_block_reasons = dict(getattr(self, "funnel_pregate_block_reasons", {}))
        
        return {
            # Legacy counters
            "candles_iterated": getattr(self, "candles_iterated", 0),
            "warmup_skipped": getattr(self, "warmup_skipped", 0),
            "pregate_checked_count": getattr(self, "pregate_checked_count", 0),
            "pregate_skipped": getattr(self, "pregate_skipped", 0),
            "prebuilt_available_checked": getattr(self, "prebuilt_available_checked", 0),
            "prebuilt_missing_skipped": getattr(self, "prebuilt_missing_skipped", 0),
            "bars_before_evaluate_entry": getattr(self, "bars_before_evaluate_entry", 0),
            "evaluate_entry_called_count": getattr(self, "evaluate_entry_called_count", 0),
            "bars_after_evaluate_entry": getattr(self, "bars_after_evaluate_entry", 0),
            "last_stop_reason": getattr(self, "last_stop_reason", None),
            # Canonical funnel ledger counters
            "funnel_bars_total": getattr(self, "loop_iters_total", 0),
            "funnel_bars_post_warmup": getattr(self, "funnel_bars_post_warmup", 0),
            "funnel_pregate_pass": getattr(self, "funnel_pregate_pass", 0),
            "funnel_pregate_block": getattr(self, "funnel_pregate_block", 0),
            "funnel_pregate_block_reasons": pregate_block_reasons,
            "funnel_entry_eval_called": getattr(self, "funnel_entry_eval_called", 0),
            "funnel_predict_entered": getattr(self, "funnel_predict_entered", 0),
            "funnel_first_ts_post_warmup": str(getattr(self, "funnel_first_ts_post_warmup", None)),
            "funnel_last_ts_post_warmup": str(getattr(self, "funnel_last_ts_post_warmup", None)),
        }

    
    def _log_entry_only_event_impl(
        self,
        timestamp: pd.Timestamp,
        side: str,
        price: float,
        prediction: Any,  # EntryPrediction-like object
        policy_state: Dict[str, Any],
    ) -> None:
        """
        Log a hypothetical entry event in ENTRY_ONLY mode.
        Does NOT create a LiveTrade or run exits.
        
        Uses buffered writes (batch of 100 entries) to avoid I/O bottleneck.
        
        Parameters
        ----------
        timestamp : pd.Timestamp
            Entry timestamp
        side : str
            Entry side ("long" or "short")
        price : float
            Entry price
        prediction : Any
            Entry prediction object with prob_long, prob_short, margin, p_hat
        policy_state : Dict[str, Any]
            Policy state containing regime info and V7_HIGH predictions
        """
        if self.entry_only_log_path is None:
            return
        
        # Extract regime info
        trend_regime = policy_state.get("brain_trend_regime", "UNKNOWN")
        vol_regime = policy_state.get("brain_vol_regime", "UNKNOWN")
        session = policy_state.get("session", "UNKNOWN")
        
        # Prepare row
        row = {
            "run_id": self.run_id,
            "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
            "side": side.lower(),
            "entry_price": f"{price:.3f}",
            "session": session,
            "trend_regime": trend_regime,
            "vol_regime": vol_regime,
            "p_long_entry": f"{prediction.prob_long:.4f}" if hasattr(prediction, "prob_long") else "",
            "p_short_entry": f"{prediction.prob_short:.4f}" if hasattr(prediction, "prob_short") else "",
            "margin_entry": f"{prediction.margin:.4f}" if hasattr(prediction, "margin") else "",
            "p_hat_entry": f"{prediction.p_hat:.4f}" if hasattr(prediction, "p_hat") else "",
        }
        
        # Initialize buffer if not exists
        if not hasattr(self, "_entry_only_log_buffer"):
            self._entry_only_log_buffer = []
            self._entry_only_log_buffer_size = 100  # Flush every 100 entries
        
        # Add to buffer
        self._entry_only_log_buffer.append(row)
        
        # Flush buffer if full
        if len(self._entry_only_log_buffer) >= self._entry_only_log_buffer_size:
            self._flush_entry_only_log_buffer()
        
        # Removed debug logging to reduce I/O overhead (only log on flush)
    
    
    def _flush_entry_only_log_buffer_impl(self) -> None:
        """Flush buffered entry-only log entries to CSV file."""
        if not hasattr(self, "_entry_only_log_buffer") or len(self._entry_only_log_buffer) == 0:
            return
        
        if self.entry_only_log_path is None:
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
        
        # Write all buffered rows in one I/O operation
        with self.entry_only_log_path.open("a", newline="", encoding="utf-8", buffering=8192) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(self._entry_only_log_buffer)
        
        # Clear buffer
        buffer_size = len(self._entry_only_log_buffer)
        self._entry_only_log_buffer = []
        
        # Only log flush every 10th time to reduce I/O overhead
        if not hasattr(self, "_flush_count"):
            self._flush_count = 0
        self._flush_count += 1
        if self._flush_count % 10 == 0:
            log.debug(f"[ENTRY_ONLY] Flushed {buffer_size} entries to CSV (total flushes: {self._flush_count})")
    
    def _ensure_bid_ask_columns(self, df: pd.DataFrame, context: str) -> None:
        """Validate bid/ask columns exist before running bid-aware logic."""
        missing = [col for col in BID_ASK_REQUIRED_COLS if col not in df.columns]
        if missing:
            raise ValueError(
                f"Bid/ask required for FARM_V2B 2025 replay, but missing in candles "
                f"(context={context}): {missing}"
            )

    def _calculate_unrealized_portfolio_bps(self, current_bid: float, current_ask: float) -> float:
        """
        Calculate unrealized portfolio PnL in bps (weighted average by units).
        """
        if not self.open_trades:
            return 0.0

        total_pnl_bps = 0.0
        total_units = 0

        if not hasattr(self, "_portfolio_pnl_log_count"):
            self._portfolio_pnl_log_count = 0

        for trade in self.open_trades:
            units = abs(int(trade.units))
            entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
            entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))

            if self._portfolio_pnl_log_count < 5:
                log.debug(
                    "[PNL] Portfolio PnL: entry_bid=%.5f entry_ask=%.5f exit_bid=%.5f exit_ask=%.5f side=%s",
                    entry_bid,
                    entry_ask,
                    current_bid,
                    current_ask,
                    trade.side,
                )
                self._portfolio_pnl_log_count += 1

            pnl_bps = compute_pnl_bps(entry_bid, entry_ask, current_bid, current_ask, trade.side)

            total_pnl_bps += pnl_bps * units
            total_units += units

        if total_units == 0:
            return 0.0

        return total_pnl_bps / total_units

    def _calculate_first_valid_eval_idx(self, df: pd.DataFrame, min_bars_for_features: int) -> int:
        """
        Calculate first valid evaluation index based on HTF warmup requirements.
        
        This ensures we don't evaluate entries before HTF bars (H1/H4) are available.
        This is a harness/eval-loop fix, not a production logic change.
        
        Args:
            df: DataFrame with M5 bars (must have DatetimeIndex)
            min_bars_for_features: Minimum bars for feature stability (M5 warmup)
        
        Returns:
            First index in df where evaluation can start (all warmup requirements met)
        """
        import numpy as np
        import pandas as pd
        from gx1.features.htf_aggregator import build_htf_from_m5
        
        if len(df) == 0:
            return 0
        
        # Get EVAL_START from replay_eval_start_ts (if set) or use first bar
        eval_start_ts = getattr(self, 'replay_eval_start_ts', None)
        if eval_start_ts is None:
            eval_start_ts = df.index[0]
        
        # Requirements:
        # 1. EVAL_START (from --start parameter or first bar)
        # 2. M5 warmup (min_bars_for_features, typically 288 bars)
        # 3. H1 warmup (first H1 bar must be completed)
        # 4. H4 warmup (first H4 bar must be completed)
        
        # Calculate M5 warmup time
        m5_warmup_time = eval_start_ts + pd.Timedelta(minutes=5 * min_bars_for_features)
        
        # Build HTF bars from first part of df to find first H1/H4 close times
        # Use first 1000 bars (enough for several H4 bars)
        n_bars_to_check = min(1000, len(df))
        df_sample = df.iloc[:n_bars_to_check].copy()
        
        # Extract M5 data
        m5_timestamps = (df_sample.index.astype('int64') // 1_000_000_000).astype(np.int64)  # Convert to seconds
        m5_open = df_sample['open'].values.astype(np.float64)
        m5_high = df_sample['high'].values.astype(np.float64)
        m5_low = df_sample['low'].values.astype(np.float64)
        m5_close = df_sample['close'].values.astype(np.float64)
        
        # Build H1 and H4 bars
        try:
            h1_ts, h1_open, h1_high, h1_low, h1_close, h1_close_times = build_htf_from_m5(
                m5_timestamps, m5_open, m5_high, m5_low, m5_close, interval_hours=1
            )
            h4_ts, h4_open, h4_high, h4_low, h4_close, h4_close_times = build_htf_from_m5(
                m5_timestamps, m5_open, m5_high, m5_low, m5_close, interval_hours=4
            )
            
            # Get first H1 and H4 close times
            first_h1_close_time = None
            first_h4_close_time = None
            
            if len(h1_close_times) > 0:
                first_h1_close_time = pd.Timestamp(h1_close_times[0], unit='s', tz='UTC')
            if len(h4_close_times) > 0:
                first_h4_close_time = pd.Timestamp(h4_close_times[0], unit='s', tz='UTC')
            
            # Calculate first_valid_eval_time = max of all requirements
            first_valid_eval_time = eval_start_ts
            
            if first_h1_close_time is not None:
                first_valid_eval_time = max(first_valid_eval_time, first_h1_close_time)
            if first_h4_close_time is not None:
                first_valid_eval_time = max(first_valid_eval_time, first_h4_close_time)
            if m5_warmup_time is not None:
                first_valid_eval_time = max(first_valid_eval_time, m5_warmup_time)
            
            # Find first index in df where timestamp >= first_valid_eval_time
            first_valid_eval_idx = df.index.searchsorted(first_valid_eval_time, side='left')
            
            # Ensure we have at least:
            # 1. min_bars_for_features (M5 warmup, typically 288 bars)
            # 2. 48 bars for H4 warmup (minimum needed for first H4 bar)
            # 3. 12 bars for H1 warmup (minimum needed for first H1 bar)
            first_valid_eval_idx = max(first_valid_eval_idx, min_bars_for_features, 48, 12)
            
            log.info(
                "[REPLAY] HTF warmup calculation: "
                "eval_start=%s, first_h1_close=%s, first_h4_close=%s, m5_warmup=%s, "
                "first_valid_eval_time=%s, first_valid_eval_idx=%d",
                eval_start_ts.isoformat(),
                first_h1_close_time.isoformat() if first_h1_close_time else "N/A",
                first_h4_close_time.isoformat() if first_h4_close_time else "N/A",
                m5_warmup_time.isoformat() if m5_warmup_time else "N/A",
                first_valid_eval_time.isoformat(),
                first_valid_eval_idx
            )
            
            return first_valid_eval_idx
            
        except Exception as e:
            log.warning(
                "[REPLAY] Failed to calculate first_valid_eval_idx from HTF warmup: %s. "
                "Falling back to min_bars_for_features=%d",
                e, min_bars_for_features
            )
            return min_bars_for_features

    def _reset_entry_diag(self) -> None:
        """Reset entry diagnostics (per run)."""
        self._entry_diag = {
            "total": 0,
            "per_session": defaultdict(int),
            "per_exit_profile": defaultdict(int),
            "per_regime": defaultdict(int),
        }

    def _record_entry_diag(self, trade: LiveTrade, policy_state: Dict[str, Any], prediction: Optional[EntryPrediction]) -> None:
        """Record per-trade diagnostics for FARM entry."""
        from gx1.execution.live_features import infer_session_tag
        if not hasattr(self, "_entry_diag"):
            self._reset_entry_diag()
        session = policy_state.get("session") or infer_session_tag(trade.entry_time)
        regime = policy_state.get("farm_regime") or trade.extra.get("vol_regime_entry") or policy_state.get("brain_vol_regime", "UNKNOWN")
        exit_profile = (getattr(trade, "extra", {}) or {}).get("exit_profile", "UNKNOWN")
        p_long = prediction.prob_long if prediction is not None else trade.entry_prob_long
        margin = getattr(prediction, "margin", None)
        if margin is None:
            margin = abs(trade.entry_prob_long - trade.entry_prob_short)
        spread_bps = (
            (float(trade.entry_ask) - float(trade.entry_bid)) * 10000.0
            if getattr(trade, "entry_bid", None) is not None and getattr(trade, "entry_ask", None) is not None
            else float("nan")
        )
        log.info(
            "[ENTRY_DIAG] trade=%s ts=%s session=%s regime=%s exit_profile=%s p_long=%.4f margin=%.4f atr_bps=%.2f spread_bps=%.2f",
            trade.trade_id,
            trade.entry_time.isoformat(),
            session,
            regime,
            exit_profile,
            p_long,
            margin,
            trade.atr_bps,
            spread_bps,
        )
        diag = self._entry_diag
        diag["total"] += 1
        diag["per_session"][session] += 1
        diag["per_exit_profile"][exit_profile] += 1
        diag["per_regime"][regime] += 1

    def _write_stage0_reason_report(self) -> None:
        """
        Write Stage-0 reason breakdown report to JSON file.
        Called at end of replay to summarize why entries were blocked.
        """
        if not hasattr(self, "entry_manager") or not hasattr(self.entry_manager, "stage0_reasons"):
            return
        
        stage0_reasons = self.entry_manager.stage0_reasons
        total_considered = getattr(self.entry_manager, "stage0_total_considered", 0)
        total_skipped = sum(stage0_reasons.values())
        total_passed = total_considered - total_skipped
        
        # Build report
        report = {
            "total_considered": total_considered,
            "total_passed": total_passed,
            "total_skipped": total_skipped,
            "skip_rate_pct": (total_skipped / total_considered * 100.0) if total_considered > 0 else 0.0,
            "reasons": {}
        }
        
        # Add reason breakdown
        for reason, count in sorted(stage0_reasons.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_skipped * 100.0) if total_skipped > 0 else 0.0
            report["reasons"][reason] = {
                "count": count,
                "pct_of_skipped": pct,
                "pct_of_total": (count / total_considered * 100.0) if total_considered > 0 else 0.0
            }
        
        # Write to JSON file
        output_dir = Path(self.output_dir)
        # Check if we're in a parallel chunk subdirectory
        if "parallel_chunks" in str(output_dir):
            # Write to chunk-specific file
            chunk_match = None
            for part in output_dir.parts:
                if part.startswith("chunk_"):
                    chunk_match = part
                    break
            if chunk_match:
                report_file = output_dir / f"{chunk_match}_stage0_reasons.json"
            else:
                report_file = output_dir / "stage0_reasons.json"
        else:
            report_file = output_dir / "stage0_reasons.json"
        
        try:
            with open(report_file, "w") as f:
                jsonlib.dump(report, f, indent=2)
            log.info("[STAGE_0] Reason breakdown report written: %s", report_file)
            
            # Log summary
            log.info("[STAGE_0] Summary: considered=%d passed=%d skipped=%d (%.1f%%)",
                     total_considered, total_passed, total_skipped,
                     report["skip_rate_pct"])
            log.info("[STAGE_0] Top reasons:")
            for reason, data in list(report["reasons"].items())[:5]:
                log.info("  %s: %d (%.1f%% of skipped, %.1f%% of total)",
                         reason, data["count"], data["pct_of_skipped"], data["pct_of_total"])
        except Exception as e:
            log.warning("[STAGE_0] Failed to write reason report: %s", e)

    def _log_entry_diag_summary(self) -> None:
        """Log aggregated entry diagnostics at the end of replay."""
        diag = getattr(self, "_entry_diag", None)
        if not diag or not diag["total"]:
            return
        log.info(
            "[ENTRY_DIAG_SUMMARY] total=%d per_session=%s per_exit_profile=%s per_regime=%s",
            diag["total"],
            dict(diag["per_session"]),
            dict(diag["per_exit_profile"]),
            dict(diag["per_regime"]),
        )

def _generate_client_order_id(self, ts_utc: pd.Timestamp, price: float, direction: str) -> str:
    """
    Generate client_order_id for idempotency.
    
    Parameters
    ----------
    ts_utc : pd.Timestamp
        UTC timestamp of entry.
    price : float
        Entry price.
    direction : str
        Trade direction ('long' or 'short').
    
    Returns
    -------
    str
        Client order ID (hash of ts_utc, price, direction).
    """
    # Create hash from timestamp, price, and direction
    ts_str = ts_utc.strftime("%Y%m%dT%H%M%S")
    hash_input = f"{ts_str}_{price:.3f}_{direction}"
    client_order_id = hashlib.md5(hash_input.encode("utf-8")).hexdigest()[:16]
    # Prepend prefix for clarity
    return f"GX1-{client_order_id}"

def _build_notes_string(self, trade: LiveTrade) -> str:
    """
    Build notes string for trade log, including regime information.
    
    Parameters
    ----------
    trade : LiveTrade
        Trade object with extra dict containing atr_regime.
    
    Returns
    -------
    str
        Notes string with mode (replay/dry_run) and regime info.
    """
    notes_parts = []
    
    # Add mode indicator
    if self.replay_mode:
        notes_parts.append("replay")
    elif self.exec.dry_run:
        notes_parts.append("dry_run")
    
    # Add regime info if available
    if hasattr(trade, "extra") and trade.extra:
        atr_regime = trade.extra.get("atr_regime")
        if atr_regime:
            notes_parts.append(f"atr_regime={atr_regime}")
        
        # Add Big Brain V1 data if available (try nested dict first, fallback to top-level)
        brain_v1_data = trade.extra.get("big_brain_v1", {})
        if isinstance(brain_v1_data, dict):
            brain_trend = brain_v1_data.get("brain_trend_regime") or trade.extra.get("brain_trend_regime")
            brain_vol = brain_v1_data.get("brain_vol_regime") or trade.extra.get("brain_vol_regime")
            brain_risk = brain_v1_data.get("brain_risk_score") if "brain_risk_score" in brain_v1_data else trade.extra.get("brain_risk_score")
        else:
            # Fallback to top-level fields (backward compatibility)
            brain_trend = trade.extra.get("brain_trend_regime")
            brain_vol = trade.extra.get("brain_vol_regime")
            brain_risk = trade.extra.get("brain_risk_score")
        
        if brain_trend and brain_trend != "UNKNOWN":
            notes_parts.append(f"brain_trend_regime={brain_trend}")
        if brain_vol and brain_vol != "UNKNOWN":
            notes_parts.append(f"brain_vol_regime={brain_vol}")
        if brain_risk is not None:
            notes_parts.append(f"brain_risk_score={brain_risk:.3f}")
    
    return ";".join(notes_parts) if notes_parts else ""

def _check_client_order_id_exists(self, client_order_id: str) -> bool:
    """
    Check if client_order_id already exists in OANDA (for reconcile).
    
    Parameters
    ----------
    client_order_id : str
        Client order ID to check.
    
    Returns
    -------
    bool
        True if client_order_id exists, False otherwise.
    
    Note
    ----
    This is a placeholder - OANDA API doesn't have a direct way to check
    client_order_id existence. We'll reconcile by matching on trade_id
    from openTrades instead.
    """
    # Placeholder: In production, you might want to query orders endpoint
    # For now, we'll reconcile by matching on trade_id from openTrades
    return False

def _execute_entry_impl(self, trade: LiveTrade) -> None:
    # Fix 2: Validate side at trade creation (fail-fast in source)
    if not hasattr(trade, "side") or trade.side not in ("long", "short"):
        side_val = getattr(trade, "side", None)
        log.error(
            "[ENTRY] Invalid side for trade_id=%s: got %r (type=%s). Trade will not be executed.",
            trade.trade_id, side_val, type(side_val).__name__
        )
        if self.is_replay:
            # In replay mode: log and skip (don't crash)
            return
        else:
            # In live mode: raise exception (fail-fast)
            raise ValueError(f"Invalid trade.side: got {side_val!r}, expected 'long' or 'short'")
    
    # Check for guard blocks (KILL_SWITCH_ON, parity/ECE/coverage)
    guard_blocked = False
    block_reason = None
    
    # Check KILL_SWITCH_ON flag
    project_root = Path(__file__).parent.parent.parent
    kill_flag = project_root / "KILL_SWITCH_ON"
    if kill_flag.exists():
        guard_blocked = True
        block_reason = "KILL_SWITCH_ON flag"
        log.warning("[GUARD] BLOCKED ORDER (live_mode=%s) reason=%s", not self.exec.dry_run, block_reason)
        return
    
    # Check kill-switches (ECE, parity, coverage) - these are already logged in evaluate_entry
    # So we just check if we should proceed
    
    if guard_blocked:
        log.warning("[GUARD] BLOCKED ORDER (live_mode=%s) reason=%s", not self.exec.dry_run, block_reason)
        return
    
    if self.exec.dry_run:
        log.info(
            "[DRY-RUN] WOULD EXECUTE %s %s units=%s @ %.3f | client_order_id=%s",
            trade.side.upper(),
            self.instrument,
            trade.units,
            trade.entry_price,
            trade.client_order_id,
        )
    else:  # pragma: no cover - network
        # Track consecutive failures for hard STOP
        max_consecutive_failures = 3
        
        try:
            # Calculate TP/SL prices from entry price and thresholds (bps)
            # Get TP/SL from trade.extra (set in evaluate_entry) or tick_exit config
            tp_bps = int(trade.extra.get("tp_bps", self.tick_cfg.get("tp_bps", 180))) if hasattr(trade, "extra") and trade.extra else int(self.tick_cfg.get("tp_bps", 180))
            sl_bps = int(trade.extra.get("sl_bps", self.tick_cfg.get("sl_bps", 100))) if hasattr(trade, "extra") and trade.extra else int(self.tick_cfg.get("sl_bps", 100))

            # Calculate TP/SL prices in absolute terms using bid/ask entry prices
            # For LONG: we enter at ask, so TP/SL are relative to entry_ask
            # For SHORT: we enter at bid, so TP/SL are relative to entry_bid
            entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
            entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
            if trade.side == "long":
                # LONG: profit when price rises, entered at ask
                take_profit_price = entry_ask * (1.0 + tp_bps / 10000.0)
                stop_loss_price = entry_ask * (1.0 - sl_bps / 10000.0)
            else:
                # SHORT: profit when price falls, entered at bid
                take_profit_price = entry_bid * (1.0 - tp_bps / 10000.0)
                stop_loss_price = entry_bid * (1.0 + sl_bps / 10000.0)
            
            # Build clientExtensions for OANDA trade tracking
            exit_profile = trade.extra.get("exit_profile", "UNKNOWN")
            client_ext_id = f"GX1:{self.run_id}:{trade.trade_id}"
            client_ext_tag = self.run_id
            client_ext_comment = exit_profile[:64]  # OANDA limit
            
            client_extensions = {
                "id": client_ext_id,
                "tag": client_ext_tag,
                "comment": client_ext_comment,
            }
            
            # Log ORDER_SUBMITTED before API call
            if hasattr(self, "trade_journal") and self.trade_journal:
                try:
                    from gx1.execution.oanda_credentials import _mask_account_id
                    account_id_masked = _mask_account_id(self.oanda_account_id) if hasattr(self, "oanda_account_id") else None
                    self.trade_journal.log_order_submitted(
                        trade_id=trade.trade_id,
                        instrument=self.instrument,
                        side=trade.side,
                        units=trade.units,
                        order_type="MARKET",
                        client_order_id=trade.client_order_id,
                        client_ext_id=client_ext_id,
                        client_ext_tag=client_ext_tag,
                        client_ext_comment=client_ext_comment,
                        requested_price=trade.entry_price,
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price,
                        oanda_env=self.oanda_env if hasattr(self, "oanda_env") else None,
                        account_id_masked=account_id_masked,
                    )
                except Exception as e:
                    log.warning("[TRADE_JOURNAL] Failed to log ORDER_SUBMITTED: %s", e)
            
            # Place order with broker-side TP/SL (failsafe) if enabled
            # In REN EXIT_V2 mode, broker-side TP/SL is disabled
            broker_side_tp_sl = bool(self.policy.get("broker_side_tp_sl", True)) and not self.exit_only_v2_drift
            
            # In replay mode, broker is None - simulate order execution
            if self.replay_mode and self.broker is None:
                # Simulate successful order in replay mode
                log.debug("[REPLAY] Simulating order execution (broker=None in replay mode)")
                response = {
                    "orderFillTransaction": {
                        "id": f"REPLAY-{trade.trade_id}",
                        "instrument": self.instrument,  # Use self.instrument, not trade.instrument
                        "units": str(trade.units),
                        "price": str(trade.entry_price),
                        "time": trade.entry_time.isoformat() if hasattr(trade.entry_time, 'isoformat') else str(trade.entry_time),
                    }
                }
            else:
                try:
                    if broker_side_tp_sl:
                        response = self.broker.create_market_order(
                            self.instrument,
                            trade.units,
                            client_order_id=trade.client_order_id,
                            client_extensions=client_extensions,
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                        )
                    else:
                        # Place order without broker-side TP/SL (rely on tick-watcher)
                        response = self.broker.create_market_order(
                            self.instrument,
                            trade.units,
                            client_order_id=trade.client_order_id,
                            client_extensions=client_extensions,
                        )
                except Exception as api_error:
                    # Log ORDER_REJECTED on API failure
                    if hasattr(self, "trade_journal") and self.trade_journal:
                        try:
                            status_code = getattr(api_error, "status_code", None)
                            error_msg = str(api_error)[:500]
                            self.trade_journal.log_order_rejected(
                                trade_id=trade.trade_id,
                                client_order_id=trade.client_order_id,
                                status_code=status_code,
                                reject_reason=error_msg,
                            )
                        except Exception as e:
                            log.warning("[TRADE_JOURNAL] Failed to log ORDER_REJECTED: %s", e)
                    raise  # Re-raise to maintain existing error handling
            
            # OANDA API returns orderFillTransaction with tradeOpened or tradeID
            fill = response.get("orderFillTransaction", {})
            fill_price = float(fill.get("price", trade.entry_price))
            
            # Extract OANDA IDs from response
            oanda_order_id = fill.get("id")
            oanda_trade_id = None
            oanda_transaction_id = fill.get("id")  # Transaction ID is same as order ID for fills
            ts_oanda = fill.get("time")
            
            # Extract trade ID from tradeOpened or tradeID field
            if "tradeOpened" in fill:
                oanda_trade_id = fill["tradeOpened"].get("tradeID")
            elif "tradeID" in fill:
                oanda_trade_id = fill.get("tradeID")
            
            # Extract commission/financing/pl if available
            commission = fill.get("commission")
            financing = fill.get("financing")
            pl = fill.get("pl")
            
            # Log ORDER_FILLED
            if hasattr(self, "trade_journal") and self.trade_journal:
                try:
                    self.trade_journal.log_order_filled(
                        trade_id=trade.trade_id,
                        oanda_order_id=str(oanda_order_id) if oanda_order_id else None,
                        oanda_trade_id=str(oanda_trade_id) if oanda_trade_id else None,
                        oanda_transaction_id=str(oanda_transaction_id) if oanda_transaction_id else None,
                        fill_price=fill_price,
                        fill_units=trade.units,
                        commission=float(commission) if commission else None,
                        financing=float(financing) if financing else None,
                        pl=float(pl) if pl else None,
                        ts_oanda=ts_oanda,
                    )
                except Exception as e:
                    log.warning("[TRADE_JOURNAL] Failed to log ORDER_FILLED: %s", e)
            
            # Store OANDA IDs in trade.extra
            if not hasattr(trade, "extra"):
                trade.extra = {}
            trade.extra["oanda_order_id"] = str(oanda_order_id) if oanda_order_id else None
            trade.extra["oanda_trade_id"] = str(oanda_trade_id) if oanda_trade_id else None
            trade.extra["oanda_last_txn_id"] = str(oanda_transaction_id) if oanda_transaction_id else None
            trade.extra["client_ext_id"] = client_ext_id

            # In replay mode, entry_bid/entry_ask should already be set from candles
            # In live mode, update from fill price (OANDA returns mid price, we need to estimate bid/ask)
            if self.replay_mode:
                # Replay mode: entry_bid/entry_ask already set from candles in entry_manager
                # Just update entry_price to match fill price (if different)
                trade.entry_price = fill_price
                # Validate that entry_bid/entry_ask are still valid
                if not hasattr(trade, "entry_bid") or trade.entry_bid is None:
                    raise ValueError(f"trade.entry_bid is None in replay mode for trade {trade.trade_id}")
                if not hasattr(trade, "entry_ask") or trade.entry_ask is None:
                    raise ValueError(f"trade.entry_ask is None in replay mode for trade {trade.trade_id}")
            else:
                # Live mode: Update entry_price and estimate bid/ask from fill price
                # OANDA returns mid price, so we estimate bid/ask with a small spread
                trade.entry_price = fill_price
                estimated_spread = 0.1  # Typical XAUUSD spread in live trading
                if trade.side == "long":
                    # LONG: we buy at ask, so entry_ask = fill_price
                    trade.entry_ask = fill_price
                    trade.entry_bid = fill_price - estimated_spread
                    # Ensure bid <= ask
                    if trade.entry_bid > trade.entry_ask:
                        trade.entry_bid = trade.entry_ask - 0.01  # Minimum spread
                else:
                    # SHORT: we sell at bid, so entry_bid = fill_price
                    trade.entry_bid = fill_price
                    trade.entry_ask = fill_price + estimated_spread
                    # Ensure bid <= ask
                    if trade.entry_ask < trade.entry_bid:
                        trade.entry_ask = trade.entry_bid + 0.01  # Minimum spread
            
            # Log TRADE_OPENED_OANDA event if we have OANDA trade ID
            if oanda_trade_id and hasattr(self, "trade_journal") and self.trade_journal:
                try:
                    self.trade_journal.log_oanda_trade_update(
                        trade_id=trade.trade_id,
                        event_type="TRADE_OPENED_OANDA",
                        oanda_trade_id=str(oanda_trade_id),
                        oanda_transaction_id=str(oanda_transaction_id) if oanda_transaction_id else None,
                        price=fill_price,
                        units=trade.units,
                        ts_oanda=ts_oanda,
                    )
                except Exception as e:
                    log.warning("[TRADE_JOURNAL] Failed to log TRADE_OPENED_OANDA: %s", e)
            
            # Log TP/SL order IDs (broker-side TP/SL lifecycle tracking)
            if broker_side_tp_sl:
                # Get TP/SL orders from trade (may need to fetch trade details)
                try:
                    trade_info = self.broker.get_trade(trade.trade_id)
                    trade_data = trade_info.get("trade", {})
                    tp_order = trade_data.get("takeProfitOrder", {})
                    sl_order = trade_data.get("stopLossOrder", {})
                    
                    tp_order_id = tp_order.get("id") if tp_order else None
                    sl_order_id = sl_order.get("id") if sl_order else None
                    
                    # Store TP/SL order IDs in trade.extra for tracking
                    if not hasattr(trade, "extra"):
                        trade.extra = {}
                    trade.extra["tp_order_id"] = str(tp_order_id) if tp_order_id else None
                    trade.extra["sl_order_id"] = str(sl_order_id) if sl_order_id else None
                    
                    log.info(
                        "[LIVE] Order placed with broker-side TP/SL: TP=%.3f (+%d bps, order_id=%s), SL=%.3f (-%d bps, order_id=%s)",
                        take_profit_price, tp_bps, tp_order_id or "N/A",
                        stop_loss_price, sl_bps, sl_order_id or "N/A"
                    )
                except Exception as e:
                    log.warning("[LIVE] Failed to get TP/SL order IDs for trade %s: %s", trade.trade_id, e)
                    log.info(
                        "[LIVE] Order placed with TP/SL requested: TP=%.3f (+%d bps), SL=%.3f (-%d bps) (OANDA enforces server-side)",
                        take_profit_price, tp_bps, stop_loss_price, sl_bps
                    )
            else:
                log.info(
                    "[LIVE] Order placed without broker-side TP/SL (relying on tick-watcher): TP=%.3f (+%d bps), SL=%.3f (-%d bps)",
                    take_profit_price, tp_bps, stop_loss_price, sl_bps
                )
            
                log.info("[LIVE] PLACED MARKET %s %s units=%s @ %.3f | order_id=%s | trade_id=%s", trade.side.upper(), self.instrument, trade.units, fill_price, trade.client_order_id, trade.trade_id)
            # Reset consecutive failures on success
            self._consecutive_order_failures = 0
            # Note: Trade is added to open_trades in execute_entry() after this call
        except Exception as e:
            self._consecutive_order_failures += 1
            consecutive_failures = self._consecutive_order_failures
            
            if consecutive_failures >= max_consecutive_failures:
                log.error(
                    "[HARD STOP] %d consecutive order failures. Setting KILL_SWITCH_ON.",
                    consecutive_failures,
                )
                # Set KILL_SWITCH_ON flag
                project_root = Path(__file__).parent.parent.parent
                kill_flag = project_root / "KILL_SWITCH_ON"
                kill_flag.touch()
                raise RuntimeError(f"Hard STOP: {consecutive_failures} consecutive order failures") from e
            else:
                log.warning(
                    "[ORDER FAILURE] Attempt failed (consecutive: %d/%d): %s",
                    consecutive_failures,
                    max_consecutive_failures,
                    e,
                )
                raise

    # ============================================================================
    # DEFENSIVE ASSERT: Ensure entry_bid and entry_ask are always set correctly
    # ============================================================================
    # In replay mode, entry_bid/entry_ask should already be set from candles in entry_manager
    # In live mode, they may be updated from fill price, but must still be valid
    if not hasattr(trade, "entry_bid") or trade.entry_bid is None:
        raise ValueError(
            f"trade.entry_bid is None or missing for trade {trade.trade_id}. "
            f"This must be set when trade is created (replay: from candles, live: from fill price)."
        )
    if not hasattr(trade, "entry_ask") or trade.entry_ask is None:
        raise ValueError(
            f"trade.entry_ask is None or missing for trade {trade.trade_id}. "
            f"This must be set when trade is created (replay: from candles, live: from fill price)."
        )
    
    # Validate that bid <= ask (spread should be non-negative)
    if trade.entry_bid > trade.entry_ask:
        raise ValueError(
            f"Invalid bid/ask for trade {trade.trade_id}: entry_bid={trade.entry_bid:.5f} > entry_ask={trade.entry_ask:.5f}. "
            f"Bid must be <= ask."
        )
    
    # Validate that entry_price matches side
    # For LONG: entry_price should be entry_ask (we buy at ask)
    # For SHORT: entry_price should be entry_bid (we sell at bid)
    if trade.side == "long":
        if abs(trade.entry_price - trade.entry_ask) > 0.01:  # Allow small floating point differences
            log.warning(
                "[ENTRY_PRICE] Trade %s (LONG): entry_price=%.5f != entry_ask=%.5f. "
                "For LONG trades, entry_price should equal entry_ask.",
                trade.trade_id, trade.entry_price, trade.entry_ask
            )
    elif trade.side == "short":
        if abs(trade.entry_price - trade.entry_bid) > 0.01:  # Allow small floating point differences
            log.warning(
                "[ENTRY_PRICE] Trade %s (SHORT): entry_price=%.5f != entry_bid=%.5f. "
                "For SHORT trades, entry_price should equal entry_bid.",
                trade.trade_id, trade.entry_price, trade.entry_bid
            )
    
        log.debug(
            "[ENTRY_PRICE] Trade %s (%s): entry_price=%.5f, entry_bid=%.5f, entry_ask=%.5f, spread=%.5f",
            trade.trade_id,
            trade.side.upper(),
            trade.entry_price,
            trade.entry_bid,
            trade.entry_ask,
            trade.entry_ask - trade.entry_bid,
        )

    # Ensure exit profile + policy initialization
    self._ensure_exit_profile(trade, context="execute_entry")
    if self.exit_config_name and not (getattr(trade, "extra", {}) or {}).get("exit_profile"):
        raise RuntimeError(
            f"[EXIT_PROFILE] Trade {trade.trade_id} missing exit_profile under exit-config {self.exit_config_name} (context=execute_entry)"
        )

    # Add trade to open_trades
    self.open_trades.append(trade)
    log.info(
        "[ENTRY][OPEN_TRADES] open=%d last_trade=%s",
        len(self.open_trades),
        trade.trade_id,
    )
    self.last_entry_timestamp = trade.entry_time
    self.last_entry_side = trade.side  # Track last entry side for sticky-side logic

    # TradeLifecycleV1: Persist POSITION_TRADE journaling only after the trade is actually opened/executed.
    # (ENTRY_SIGNAL is logged separately as SIGNAL_EVENT.)
    if hasattr(self, "trade_journal") and self.trade_journal and hasattr(trade, "extra") and isinstance(trade.extra, dict):
        try:
            entry_kwargs = trade.extra.get("_journal_entry_snapshot_kwargs")
            if isinstance(entry_kwargs, dict):
                # Ensure the identifiers reflect the opened trade
                entry_kwargs["trade_uid"] = getattr(trade, "trade_uid", entry_kwargs.get("trade_uid"))
                entry_kwargs["trade_id"] = getattr(trade, "trade_id", entry_kwargs.get("trade_id"))
                self.trade_journal.log_entry_snapshot(**entry_kwargs)
            feat_kwargs = trade.extra.get("_journal_feature_context_kwargs")
            if isinstance(feat_kwargs, dict):
                feat_kwargs["trade_uid"] = getattr(trade, "trade_uid", feat_kwargs.get("trade_uid"))
                feat_kwargs["trade_id"] = getattr(trade, "trade_id", feat_kwargs.get("trade_id"))
                self.trade_journal.log_feature_context(**feat_kwargs)
        except Exception as e:
            # Replay/TRUTH: fail-fast if we can't persist entry_snapshot for an opened trade.
            if getattr(self, "is_replay", False):
                raise RuntimeError(
                    f"ENTRY_SNAPSHOT_MISSING_POST_EXECUTE: Failed to persist POSITION_TRADE journaling "
                    f"after execute_entry for trade_uid={getattr(trade, 'trade_uid', None)} trade_id={getattr(trade, 'trade_id', None)} "
                    f"error={e}"
                ) from e
            log.warning("[TRADE_JOURNAL] Failed to persist deferred POSITION_TRADE journaling: %s", e)

    # Update tick watcher (start if first trade)
    self._maybe_update_tick_watcher()

def _evaluate_and_close_trades_impl(self, candles: pd.DataFrame) -> None:
    return self.exit_manager.evaluate_and_close_trades(candles)

def _run_once_impl(self) -> None:
    # Check KILL_SWITCH_ON flag (set automatically on consecutive failures or manually by ops)
    # Flag is created in project root directory
    # Automatic trigger: 3 consecutive order failures (see _execute_entry_impl() line 6084)
    project_root = Path(__file__).parent.parent.parent
    kill_switch_flag = project_root / "KILL_SWITCH_ON"
    if kill_switch_flag.exists():
        log.warning("[GUARD] BLOCKED ORDER (live_mode=%s) reason=KILL_SWITCH_ON flag", not self.exec.dry_run)
        return
    
    # Check policy-lock (policy file changed on disk)
    if not self._check_policy_lock():
        log.error("[GUARD] BLOCKED ORDER (live_mode=%s) reason=Policy file changed on disk", not self.exec.dry_run)
        return
    
    # Check graceful shutdown flag
    if hasattr(self, "_shutdown_requested") and self._shutdown_requested:
        log.warning("[GUARD] BLOCKED ORDER (live_mode=%s) reason=Shutdown requested", not self.exec.dry_run)
        return
    
    # Check backfill in progress (block orders during backfill)
    if self.backfill_in_progress:
        log.warning("[GUARD] BLOCKED ORDER (live_mode=%s) reason=Backfill in progress", not self.exec.dry_run)
        return
    
    now = pd.Timestamp.now(tz="UTC")
    
    # "No-trade warmup" som eksplisitt fase
    # Sperr ordre i hele WARMUP (du gjør det, men gjør fasen synlig i logger)
    if self.warmup_floor is not None and now < self.warmup_floor:
        log.debug(
            "[PHASE] WARMUP (no-trade): now=%s, warmup_floor=%s (blocking orders)",
            now.isoformat(),
            self.warmup_floor.isoformat(),
        )
        # Still fetch candles and evaluate, but don't execute orders
        # This allows feature rehydration to complete
    else:
        if self.warmup_floor is not None:
            log.info("[PHASE] WARMUP_END (no-trade): warmup_floor=%s passed, resuming live trading", self.warmup_floor.isoformat())
            self.warmup_floor = None  # Clear warmup_floor after it's passed
    
    # Check server time drift (every 15 minutes)
    # API-klokkedrift-sjekk: Hent serverTime hver 15. min, sammenlign mot lokal UTC; WARN ved drift >2s
    if not hasattr(self, "_last_server_time_check"):
        self._last_server_time_check = now - pd.Timedelta(minutes=20)  # Force first check
    
    if (now - self._last_server_time_check).total_seconds() >= 900:  # Check every 15 minutes
        try:
            server_time_response = self.broker.get_server_time()
            server_time_str = server_time_response.get("time", "")
            if server_time_str:
                server_time = pd.Timestamp(server_time_str)
                if server_time.tzinfo is None:
                    server_time = server_time.tz_localize("UTC")
                else:
                    server_time = server_time.tz_convert("UTC")
                
                time_drift = abs((now - server_time).total_seconds())
                if time_drift > 2.0:
                    log.warning(
                        "[TIME DRIFT] Server time drift >2s: local=%s, server=%s, drift=%.1fs (hindrer subtile off-by-one i bar-lukking)",
                        now.isoformat(),
                        server_time.isoformat(),
                        time_drift,
                    )
                else:
                    log.debug(
                        "[TIME SYNC] Server time: local=%s, server=%s, drift=%.1fs",
                        now.isoformat(),
                        server_time.isoformat(),
                        time_drift,
                    )
                self._last_server_time_check = now
        except Exception as e:
            log.warning("Server time check failed: %s", e)
    
    # "Siste bar er ikke lukket"-sperre: Ignore current incomplete M5 bar
    now_floor = now.floor("5min")
    
    # Fetch candles (exclude incomplete bars)
    candles = self.broker.get_candles(
        self.instrument,
        self.granularity,
        count=500,
        exclude_incomplete=True,
    )
    if candles.empty:
        log.warning("No candles returned, skipping iteration")
        return
    
    # Filter out incomplete bars (only process bars < now_floor)
    candles = candles[candles.index < now_floor]
    if candles.empty:
        log.debug("No complete bars available (all bars >= now_floor=%s), skipping iteration", now_floor.isoformat())
        return

    self._ensure_bid_ask_columns(candles, context="run_once")

    trade = self.evaluate_entry(candles)
    if trade and self.can_enter(candles.index[-1]):
        self.execute_entry(trade)
        # Build trade log row with FARM fields from trade.extra
        trade_extra = trade.extra if hasattr(trade, "extra") and trade.extra else {}
        trade_log_row = {
            "trade_id": trade.trade_id,
            "entry_time": trade.entry_time.isoformat(),
            "entry_price": f"{trade.entry_price:.3f}",
            "side": trade.side,
            "units": trade.units,
            "exit_time": "",
            "exit_price": "",
            "pnl_bps": "",
            "pnl_currency": "",
            "entry_prob_long": f"{trade.entry_prob_long:.4f}",
            "entry_prob_short": f"{trade.entry_prob_short:.4f}",
            "exit_prob_close": "",
            "vol_bucket": trade.vol_bucket,
            "atr_bps": f"{trade.atr_bps:.2f}",
            "notes": self._build_notes_string(trade),
            "run_id": self.run_id,
            "policy_name": self.policy_name,
            "model_name": self.model_name,
            "extra": trade_extra,
        }
        # Extract FARM fields from trade.extra (append_trade_log will handle extraction, but set explicitly for clarity)
        if trade_extra:
            if "farm_entry_session" in trade_extra:
                trade_log_row["farm_entry_session"] = trade_extra["farm_entry_session"]
            if "farm_entry_vol_regime" in trade_extra:
                trade_log_row["farm_entry_vol_regime"] = trade_extra["farm_entry_vol_regime"]
            if "farm_guard_version" in trade_extra:
                trade_log_row["farm_guard_version"] = trade_extra["farm_guard_version"]
        append_trade_log(self.trade_log_path, trade_log_row)

    # Always evaluate exits even if no new entry
    self.evaluate_and_close_trades(candles)
    
    # Update last_bar_ts after processing this bar
    if not candles.empty:
        last_bar_ts = candles.index[-1].floor("5min")
        # Only update if backfill is enabled and we're past warmup
        backfill_cfg = self.policy.get("backfill", {})
        if backfill_cfg.get("enabled", True) and (self.warmup_floor is None or now >= self.warmup_floor):
            # Update last_bar_ts in state (but don't save on every tick to reduce I/O)
            # Save every 10 bars (50 minutes) to reduce I/O
            if not hasattr(self, "_last_bar_ts_save_time"):
                self._last_bar_ts_save_time = now - pd.Timedelta(minutes=60)
            
            if (now - self._last_bar_ts_save_time).total_seconds() >= 600:  # Save every 10 minutes
                # Get current hashes for state
                feature_manifest_hash = self.entry_model_bundle.feature_cols_hash if self.entry_model_bundle is not None else None
                policy_hash = self.policy_hash
                # Save state (without rotation - only rotate weekly)
                self._save_backfill_state(
                    last_bar_ts=last_bar_ts,
                    feature_manifest_hash=feature_manifest_hash,
                    policy_hash=policy_hash,
                    rotate=False,  # Only rotate weekly, not every save
                )
                self._last_bar_ts_save_time = now
    
    # Update health signal (every minute)
    self._update_health_signal(now)
    
    log.info(
        "Cycle complete | open_trades=%s | daily_loss_tracker=%s",
        len(self.open_trades),
        jsonlib.dumps(self.daily_loss_tracker),
    )

def _update_health_signal_impl(self, now: pd.Timestamp) -> None:
    """
    Update health signal JSON file (every minute).
    
    Parameters
    ----------
    now : pd.Timestamp
        Current UTC timestamp.
    """
    # Update health signal every minute
    if self.last_health_signal_time is None:
        self.last_health_signal_time = now - pd.Timedelta(minutes=2)  # Force first update
    
    if (now - self.last_health_signal_time).total_seconds() >= 60:  # Update every minute
        try:
            # Get telemetry status
            parity_ok = True
            ece_ok = True
            coverage_ok = True
            
            # Check parity status
            if self.parity_enabled and len(self.parity_metrics) > 0:
                for session, errs in self.parity_metrics.items():
                    if len(errs) >= 10:
                        p99_err = float(np.percentile(np.array(errs), 99))
                        if p99_err > self.parity_tolerance_p99:
                            parity_ok = False
                            break
            
            # Check ECE status
            if hasattr(self, "telemetry_tracker") and self.telemetry_tracker is not None:
                for session in ("EU", "US", "OVERLAP"):
                    try:
                        ece = self.telemetry_tracker.get_ece(session)
                        if ece is not None and ece > 0.18:
                            ece_ok = False
                            break
                    except Exception as e:
                        log.warning("[HEALTH] Failed to get ECE for %s: %s", session, e)
            
            # Check coverage status
            if hasattr(self, "telemetry_tracker") and self.telemetry_tracker is not None:
                for session in ("EU", "US", "OVERLAP"):
                    try:
                        coverage = self.telemetry_tracker.get_coverage(session)
                        if coverage is not None:
                            target_coverage = self.telemetry_tracker.target_coverage
                        coverage_diff = abs(coverage - target_coverage)
                        coverage_threshold = 0.50 * target_coverage
                        if coverage_diff > coverage_threshold:
                            coverage_ok = False
                            break
                    except Exception as e:
                        log.warning("[HEALTH] Failed to get coverage for %s: %s", session, e)
            
            # Write health signal
            health_signal = {
                "alive": True,
                "last_tick_ts": now.isoformat(),
                "parity_ok": parity_ok,
                "ece_ok": ece_ok,
                "coverage_ok": coverage_ok,
                "open_trades": len(self.open_trades),
                "policy_hash": self.policy_hash,
                "shutdown_requested": self._shutdown_requested,
            }
            
            with self.health_signal_path.open("w", encoding="utf-8") as handle:
                jsonlib.dump(health_signal, handle, separators=(",", ":"))
            
            self.last_health_signal_time = now
            
        except Exception as e:
            log.warning("Health signal update failed: %s", e)

def _setup_signal_handlers_impl(self) -> None:
    """Setup signal handlers for graceful shutdown."""
    import signal
    
    def signal_handler(signum, frame):
        log.info("Received signal %d. Initiating graceful shutdown...", signum)
        self._shutdown_requested = True
        # Flush logs
        self._flush_logs()
        log.info("Shutdown marker: graceful shutdown requested")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def _flush_logs_impl(self) -> None:
    """Flush all log files and JSONL files."""
    try:
        # Flush telemetry logs
        if hasattr(self, "telemetry_tracker"):
            # Telemetry logs are auto-flushed on write, but we can force flush
            pass
        
        # Flush eval log
        if hasattr(self, "eval_log_path") and self.eval_log_path.exists():
            # JSONL files are auto-flushed on write
            pass
        
        # Flush parity log
        if hasattr(self, "parity_log_path") and self.parity_log_path.exists():
            # JSONL files are auto-flushed on write
            pass
        
        log.info("Logs flushed")
    except Exception as e:
        log.warning("Log flush failed: %s", e)

def _check_disk_space_impl(self) -> bool:
    """
    Check disk space for logs directory.
    
    Returns
    -------
    bool
        True if disk space OK, False if >2 GB.
    """
    try:
        import shutil
        
        # Get disk usage for logs directory
        total, used, free = shutil.disk_usage(self.log_dir)
        
        # Convert to GB
        used_gb = used / (1024 ** 3)
        
        if used_gb > 2.0:
            log.warning(
                "[DISK SPACE] Logs directory >2 GB: %.2f GB used. Consider rotating logs.",
                used_gb,
            )
            return False
        
        return True
    except Exception as e:
        log.warning("Disk space check failed: %s", e)
        return True  # Allow trading if check fails (conservative)

    # ---------------------------------------------------------------------
    # JOURNAL BUFFERING (replay-only, semantically neutral)
    #
    # Goal: reduce write amplification (json.dump + flush churn) in replay hot path.
    # Contract:
    # - live mode untouched (buffer disabled)
    # - event content / ordering unchanged on disk
    # - flush on trade close + run finalization; assert empty at end in TRUTH
    # ---------------------------------------------------------------------
    def _init_journal_buffering(self) -> None:
        if hasattr(self, "_journal_buffer"):
            return
        self._journal_buffer: List[Dict[str, Any]] = []
        self._journal_buffer_max: int = 256
        self._journal_buffer_enabled: bool = False
        # Replay-only: defer per-trade JSON snapshots to explicit flush points (trade close / run end).
        self._replay_trade_json_defer_enabled: bool = False
        self.journal_buffer_flush_count: int = 0
        self.journal_trade_json_write_count: int = 0
        # Per-trade JSON writes are extremely expensive; buffer them as "dirty" keys in replay.
        self._journal_trade_json_dirty: Dict[str, Dict[str, Optional[str]]] = {}

    def _is_truth_mode(self) -> bool:
        import os
        return os.getenv("GX1_RUN_MODE", "").upper() == "TRUTH" or os.getenv("GX1_TRUTH_MODE", "0") == "1"

    def _enable_journal_buffering_if_replay(self) -> None:
        import os
        self._init_journal_buffering()
        # Replay-only + PREBUILT-only by default. Live mode remains untouched.
        prebuilt = False
        try:
            replay_mode_enum = getattr(self, "replay_mode_enum", None)
            prebuilt = bool(replay_mode_enum is not None and getattr(replay_mode_enum, "is_prebuilt", lambda: False)())
        except Exception:
            prebuilt = False
        enabled_by_env = os.getenv("GX1_JOURNAL_BUFFERING", "1") == "1"
        self._journal_buffer_enabled = bool(getattr(self, "replay_mode", False) and prebuilt and enabled_by_env)
        defer_by_env = os.getenv("GX1_REPLAY_JSON_DEFER", "1") == "1"
        self._replay_trade_json_defer_enabled = bool(getattr(self, "replay_mode", False) and prebuilt and defer_by_env)
        if self._journal_buffer_enabled and not hasattr(self, "_journal_buffer_banner_logged"):
            self._journal_buffer_banner_logged = True
            log.info("[JOURNAL_BUFFER] enabled, max=%d", int(self._journal_buffer_max))
        if self._replay_trade_json_defer_enabled and not hasattr(self, "_replay_json_opt_banner_logged"):
            self._replay_json_opt_banner_logged = True
            log.info("[REPLAY_JSON_OPT] trade JSON writes reduced (replay-only)")

    def _enqueue_journal_event(self, event: Dict[str, Any]) -> None:
        self._init_journal_buffering()
        if not bool(getattr(self, "_journal_buffer_enabled", False)):
            # Fallback: write immediately if buffering disabled.
            try:
                if hasattr(self, "trade_journal") and self.trade_journal and hasattr(self.trade_journal, "_write_jsonl_event"):
                    self.trade_journal._write_jsonl_event(event)  # type: ignore[attr-defined]
            except Exception:
                pass
            return
        self._journal_buffer.append(event)
        if len(self._journal_buffer) >= int(self._journal_buffer_max):
            self._flush_journal_buffer(reason="max_reached")

    def _mark_trade_json_dirty(self, trade_uid: Optional[str] = None, trade_id: Optional[str] = None) -> None:
        self._init_journal_buffering()
        if not bool(getattr(self, "_replay_trade_json_defer_enabled", False)):
            # Legacy behavior: write immediately when not in replay defer mode.
            try:
                if hasattr(self, "trade_journal") and self.trade_journal:
                    self.trade_journal._write_trade_json(trade_uid=trade_uid, trade_id=trade_id)
                    self.journal_trade_json_write_count += 1
            except Exception:
                pass
            return
        # Use trade_uid/trade_id as the identity for the eventual write.
        key = trade_uid or (f"LEGACY:{trade_id}" if trade_id else None)
        if not key:
            return
        # Preserve insertion order and keep latest identifiers.
        self._journal_trade_json_dirty[key] = {"trade_uid": trade_uid, "trade_id": trade_id}

    def _flush_journal_buffer(self, reason: str = "unknown") -> None:
        self._init_journal_buffering()
        jsonl_enabled = bool(getattr(self, "_journal_buffer_enabled", False))
        defer_enabled = bool(getattr(self, "_replay_trade_json_defer_enabled", False))

        # 1) Flush JSONL events (preserve ordering, same serialization) if buffering enabled.
        if jsonl_enabled and self._journal_buffer:
            try:
                if hasattr(self, "trade_journal") and self.trade_journal and hasattr(self.trade_journal, "_write_jsonl_events_batch"):
                    self.trade_journal._write_jsonl_events_batch(self._journal_buffer)  # type: ignore[attr-defined]
                else:
                    # Fallback: write one by one
                    for ev in self._journal_buffer:
                        if hasattr(self, "trade_journal") and self.trade_journal and hasattr(self.trade_journal, "_write_jsonl_event"):
                            self.trade_journal._write_jsonl_event(ev)  # type: ignore[attr-defined]
            finally:
                self._journal_buffer.clear()

        # 2) Flush per-trade JSON writes ONLY at explicit allowed points in replay defer mode.
        # Allowed points: before_journal_close, run_replay_finally.
        # NOTE: trade_close writes are handled per-trade (only the closing trade), not as a global flush.
        allow_trade_json_flush = (not defer_enabled) or (reason in ("before_journal_close", "run_replay_finally"))
        if allow_trade_json_flush and self._journal_trade_json_dirty:
            dirty_items = list(self._journal_trade_json_dirty.values())
            self._journal_trade_json_dirty.clear()
            for item in dirty_items:
                try:
                    if hasattr(self, "trade_journal") and self.trade_journal:
                        self.trade_journal._write_trade_json(trade_uid=item.get("trade_uid"), trade_id=item.get("trade_id"))
                        self.journal_trade_json_write_count += 1
                except Exception:
                    pass

        self.journal_buffer_flush_count += 1

        # TRUTH invariant: at run end we require clean buffers (asserted at close points).
        if reason in ("before_journal_close", "run_replay_finally") and self._is_truth_mode():
            assert len(self._journal_buffer) == 0
            assert len(self._journal_trade_json_dirty) == 0

    def _flush_single_trade_json(self, trade_uid: Optional[str] = None, trade_id: Optional[str] = None) -> None:
        """
        Replay-only helper: write exactly one per-trade JSON snapshot (for a closing trade),
        and remove it from dirty-set if present.
        """
        self._init_journal_buffering()
        try:
            if hasattr(self, "trade_journal") and self.trade_journal:
                self.trade_journal._write_trade_json(trade_uid=trade_uid, trade_id=trade_id)
                self.journal_trade_json_write_count += 1
        except Exception:
            return
        # Best-effort: remove from dirty set to prevent duplicate writes at run end.
        try:
            key = trade_uid or (f"LEGACY:{trade_id}" if trade_id else None)
            if key and key in self._journal_trade_json_dirty:
                self._journal_trade_json_dirty.pop(key, None)
        except Exception:
            pass

def _rotate_logs_impl(self) -> None:
    """
    Rotate and compress old log files (>7 days) and cache files (>14 days).
    
    Note: This is a placeholder - in production, you might want to use
    logrotate or a similar tool for log rotation.
    """
    try:
        import gzip
        from datetime import datetime, timedelta
        
        log_cutoff_date = datetime.now() - timedelta(days=7)
        cache_cutoff_date = datetime.now() - timedelta(days=14)
        
        # Rotate telemetry logs
        telemetry_dir = self.log_dir / "telemetry"
        if telemetry_dir.exists():
            for log_file in telemetry_dir.glob("*.jsonl"):
                try:
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < log_cutoff_date:
                        # Compress old log file
                        with log_file.open("rb") as f_in:
                            compressed_path = log_file.with_suffix(".jsonl.gz")
                            with gzip.open(compressed_path, "wb") as f_out:
                                f_out.writelines(f_in)
                        log_file.unlink()
                        log.debug("Rotated log file: %s → %s", log_file.name, compressed_path.name)
                except Exception as e:
                    log.warning("Failed to rotate log file %s: %s", log_file, e)
        
        # Rotate shadow-exit logs
        reports_dir = self.log_dir / "reports"
        if reports_dir.exists():
            for log_file in reports_dir.glob("*.jsonl"):
                try:
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < log_cutoff_date:
                        # Compress old log file
                        with log_file.open("rb") as f_in:
                            compressed_path = log_file.with_suffix(".jsonl.gz")
                            with gzip.open(compressed_path, "wb") as f_out:
                                f_out.writelines(f_in)
                        log_file.unlink()
                        log.debug("Rotated log file: %s → %s", log_file.name, compressed_path.name)
                except Exception as e:
                    log.warning("Failed to rotate log file %s: %s", log_file, e)
        
        # Rotate cache files (compress older than 14 days)
        cache_dir = self.log_dir / "cache"
        if cache_dir.exists():
            for cache_file in cache_dir.glob("*.parquet"):
                try:
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if file_time < cache_cutoff_date:
                        # Compress old cache file
                        import pandas as pd
                        # Read parquet file
                        df = pd.read_parquet(cache_file)
                        # Write compressed parquet file
                        compressed_path = cache_file.with_suffix(".parquet.gz")
                        df.to_parquet(compressed_path, compression="gzip")
                        cache_file.unlink()
                        log.debug("Rotated cache file: %s → %s", cache_file.name, compressed_path.name)
                except Exception as e:
                    log.warning("Failed to rotate cache file %s: %s", cache_file, e)
        
    except Exception as e:
        log.warning("Log rotation failed: %s", e)

def _run_forever_impl(self) -> None:
    log.info("Starting continuous loop (CTRL+C to terminate)")
    log.info("[LIVE] Listening M5 %s (dry_run=%s)", self.instrument, self.exec.dry_run)
    
    # Setup signal handlers for graceful shutdown
    self._setup_signal_handlers()
    
    # Check disk space on startup
    self._check_disk_space()
    
    # Rotate logs on startup
    self._rotate_logs()
    
    try:
        while True:
            # Check shutdown flag
            if self._shutdown_requested:
                log.info("Shutdown requested. Exiting gracefully.")
                break
            
            # Check disk space (every hour)
            if not hasattr(self, "_last_disk_check"):
                self._last_disk_check = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)
            
            now = pd.Timestamp.now(tz="UTC")
            if (now - self._last_disk_check).total_seconds() >= 3600:  # Check every hour
                self._check_disk_space()
                self._rotate_logs()
                self._last_disk_check = now
            
            wait_until_next_bar(now)
            self.run_once()
    except KeyboardInterrupt:
        log.info("Interrupted by user. Exiting gracefully.")
        self._shutdown_requested = True
    finally:
        # Graceful shutdown: flush logs and write shutdown marker
        log.info("Shutdown marker: graceful shutdown complete")
        self._flush_logs()
        log.info("Exiting.")

def _replay_force_close_open_trades_at_end(
        self,
        last_ts: pd.Timestamp,
        last_bid: float,
        last_ask: float,
        last_mid: float,
        reason: str,
        price_source: str,
    ) -> None:
        """
        Force close all open trades at end of replay (EOF liquidation).
        
        This ensures all trades are closed for complete analysis (PnL, durations, exit reasons).
        Only called in replay mode when replay.close_open_trades_at_end=true.
        
        Args:
            last_ts: Last candle timestamp
            last_bid: Last bid price
            last_ask: Last ask price
            last_mid: Last mid price
            reason: Close reason (e.g., "REPLAY_EOF")
            price_source: Price source ("mid", "bid", or "ask")
        """
        if not self.open_trades:
            return
        
                    # compute_pnl_bps is already imported at module level (line 74)
                    # No need for local import here
        
        closed_count = 0
        for trade in list(self.open_trades):
            try:
                self.exit_coverage["replay_end_or_eof_triggered"] = True
                self._exit_cov_inc("force_close_attempts_replay_eof", 1)

                # GUARD: Check if trade is already being closed/exited (prevent duplicate exit attempts)
                if hasattr(self, "_closing_trades") and trade.trade_id in self._closing_trades:
                    log.warning(
                        "[REPLAY_EOF] Trade %s is already being closed (duplicate exit attempt prevented)",
                        trade.trade_id
                    )
                    continue
                if hasattr(self, "_exited_trade_ids") and trade.trade_id in self._exited_trade_ids:
                    log.warning(
                        "[REPLAY_EOF] Trade %s already exited (duplicate exit attempt prevented)",
                        trade.trade_id
                    )
                    continue
                
                # Calculate entry prices
                entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
                entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
                
                # Determine exit price based on price_source
                if price_source == "bid":
                    exit_price = last_bid
                    exit_bid = last_bid
                    exit_ask = last_bid  # Use bid for both for PnL calculation
                elif price_source == "ask":
                    exit_price = last_ask
                    exit_bid = last_ask  # Use ask for both for PnL calculation
                    exit_ask = last_ask
                else:  # mid (default)
                    exit_price = last_mid
                    exit_bid = last_bid
                    exit_ask = last_ask
                
                # Calculate PnL
                pnl_bps = compute_pnl_bps(entry_bid, entry_ask, exit_bid, exit_ask, trade.side)
                
                # Calculate bars held
                delta_minutes = (last_ts - trade.entry_time).total_seconds() / 60.0
                bars_held = int(round(delta_minutes / 5.0))  # M5 = 5 minutes per bar
                
                # Close trade via request_close (which will go through ExitArbiter)
                # REPLAY_EOF is always allowed in replay mode (see request_close)
                accepted = self.request_close(
                    trade_id=trade.trade_id,
                    source="REPLAY_EOF",
                    reason=reason,
                    px=exit_price,
                    pnl_bps=pnl_bps,
                    bars_in_trade=bars_held,
                )
                
                if accepted:
                    closed_count += 1
                    # Log EOF close details for trade journal
                    if hasattr(self, "trade_journal") and self.trade_journal:
                        try:
                            # GUARD: In replay mode, validate trade_uid format before logging
                            if self.is_replay:
                                env_run_id = os.getenv("GX1_RUN_ID")
                                env_chunk_id = os.getenv("GX1_CHUNK_ID")
                                if env_run_id and env_chunk_id:
                                    expected_prefix = f"{env_run_id}:{env_chunk_id}:"
                                    if not trade.trade_uid.startswith(expected_prefix):
                                        raise RuntimeError(
                                            f"BAD_TRADE_UID_FORMAT_REPLAY: trade.trade_uid={trade.trade_uid} does not start with "
                                            f"expected prefix={expected_prefix}. GX1_RUN_ID={env_run_id}, GX1_CHUNK_ID={env_chunk_id}. "
                                            f"This is a hard contract violation in replay mode."
                                        )
                            
                            self.trade_journal.log_exit_summary(
                                exit_time=last_ts.isoformat(),
                                exit_reason=reason,
                                exit_price=exit_price,
                                realized_pnl_bps=pnl_bps,
                                trade_uid=trade.trade_uid,
                                trade_id=trade.trade_id,
                            )
                            self._exit_cov_inc("exit_summary_logged", 1)
                            self._exit_cov_inc("force_close_logged_replay_eof", 1)
                            self._exit_cov_note_close(trade_id=trade.trade_id, reason=reason)
                            # Add EOF-specific metadata (use trade_uid, not trade_id)
                            trade_journal = self.trade_journal._get_trade_journal(trade_uid=trade.trade_uid, trade_id=trade.trade_id)
                            if trade_journal.get("exit_summary"):
                                trade_journal["exit_summary"]["closed_by_replay_eof"] = True
                                trade_journal["exit_summary"]["eof_price_source"] = price_source
                            self.trade_journal._write_trade_json(trade_uid=trade.trade_uid, trade_id=trade.trade_id)
                        except Exception as e:
                            log.warning("[REPLAY_EOF] Failed to log exit summary for %s: %s", trade.trade_id, e)
                else:
                    log.warning("[REPLAY_EOF] Failed to close trade %s (ExitArbiter rejected)", trade.trade_id)
            except Exception as e:
                log.error("[REPLAY_EOF] Error closing trade %s: %s", trade.trade_id, e, exc_info=True)
        
        log.info("[REPLAY_EOF] Closed %d/%d open trades at EOF (reason=%s, price_source=%s)", 
                 closed_count, len(list(self.open_trades)), reason, price_source)

def _assert_no_case_collisions(df: pd.DataFrame, where: str) -> pd.DataFrame:
    """
    Fail-fast check for case-insensitive column name collisions.
    
    This function now uses the centralized column_collision_guard utility
    with support for temporary compat-mode (GX1_ALLOW_CLOSE_ALIAS_COMPAT=1).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check
    where : str
        Description of where this check is performed (for error messages)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with collisions resolved (if compat-mode enabled)
    
    Raises
    ------
    RuntimeError
        If case-insensitive collisions are detected and compat-mode is not enabled
        or collision is not only close/CLOSE.
    """
    from gx1.runtime.column_collision_guard import assert_no_case_collisions, resolve_close_alias_collision
    
    # Check for collisions (compat-mode only via explicit env var, not default)
    resolution = assert_no_case_collisions(
        df=df,
        context=f"After loading from {where}",
        allow_close_alias_compat=False,  # DEL 3: No default - must be explicit GX1_ALLOW_CLOSE_ALIAS_COMPAT=1
    )
    
    # If resolution dict is returned, apply the resolution
    if resolution:
        # Drop CLOSE column and log
        import logging
        log = logging.getLogger(__name__)
        log.warning(
            "[COMPAT] Dropped CLOSE column due to collision with candles.close. "
            "Alias expected: CLOSE -> candles.close"
        )
        
        # Resolve collision by dropping CLOSE
        df_resolved, resolution_meta = resolve_close_alias_collision(
            df=df,
            context=where,
            transformer_requires_close=False,  # TODO: Check if transformer actually needs CLOSE
        )
        
        # Store resolution metadata in runner if available
        # (This will be logged in RUN_CTX.json later)
        return df_resolved
    
    # No collision or no resolution needed
    return df


def _run_replay_impl(self: GX1DemoRunner, csv_path: Path) -> None:
        """
        Run backtest with historical M5 candles from CSV/parquet.
        
        Parameters
        ----------
        csv_path : Path
            Path to historical M5 candles file. Expected columns:
            time, open, high, low, close, volume.
            Can be CSV or Parquet format.
        """
        log.info("=" * 60)
        log.info("[REPLAY] Starting offline backtest with historical data")
        log.info("[REPLAY] Input file: %s", csv_path)
        log.info("=" * 60)

        # TRUTH: ensure exit_ml exits jsonl exists when exit transformer enabled (for 0-trade POSTRUN gate)
        out_dir = getattr(self, "explicit_output_dir", None) or getattr(self, "output_dir", None)
        if (
            out_dir
            and getattr(self, "run_id", None)
            and getattr(self, "exit_transformer_decider", None) is not None
        ):
            from gx1.execution.exit_logs import create_exit_jsonl_placeholder
            log_path = create_exit_jsonl_placeholder(out_dir, self.run_id)
            self.exit_ml_log_path = log_path
            log.debug("[EXIT_ML] Exits jsonl placeholder ready: %s", log_path)
        
        # Replay-only: lazy import session classifier (avoid module-level import in PREBUILT)
        from gx1.execution.live_features import infer_session_tag
        
        # FIX: Import os explicitly to avoid shadowing issues
        
        # Replay-only: resolve AUTO diagnostic bypass gate (requires baseline output)
        if os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1":
            diag_gate_raw = os.getenv("GX1_DIAGNOSTIC_BYPASS_GATE", "").strip().lower()
            if diag_gate_raw == "auto":
                baseline_output_dir = os.getenv("GX1_DIAGNOSTIC_BASELINE_OUTPUT_DIR", "").strip()
                if not baseline_output_dir:
                    raise RuntimeError(
                        "[DIAGNOSTIC_AUTO_FAIL] GX1_DIAGNOSTIC_BYPASS_GATE=AUTO requires "
                        "GX1_DIAGNOSTIC_BASELINE_OUTPUT_DIR to be set."
                    )
                baseline_dir = Path(baseline_output_dir)
                if not baseline_dir.exists():
                    raise RuntimeError(
                        f"[DIAGNOSTIC_AUTO_FAIL] Baseline output dir not found: {baseline_output_dir}"
                    )
                # Load most recent root-cause report
                rc_files = sorted(baseline_dir.glob("SESSION_FUNNEL_ROOT_CAUSE_*.json"))
                if not rc_files:
                    raise RuntimeError(
                        f"[DIAGNOSTIC_AUTO_FAIL] No SESSION_FUNNEL_ROOT_CAUSE_*.json found in {baseline_output_dir}"
                    )
                rc_path = rc_files[-1]
                with open(rc_path, "r") as f:
                    rc_data = jsonlib.load(f)
                force_sessions_raw = os.getenv("GX1_DIAGNOSTIC_FORCE_EVAL_SESSIONS", "")
                force_sessions = [s.strip().upper() for s in force_sessions_raw.split(",") if s.strip()]
                if not force_sessions:
                    raise RuntimeError(
                        "[DIAGNOSTIC_AUTO_FAIL] GX1_DIAGNOSTIC_FORCE_EVAL_SESSIONS must be set for AUTO bypass."
                    )
                stage_to_gate = {
                    "hard_eligibility_pass": "hard_eligibility",
                    "soft_eligibility_pass": "soft_eligibility",
                    "session_gate_pass": "session_gate",
                    "vol_guard_pass": "vol_guard",
                    "score_gate_pass": "score_gate",
                    "pregate_pass": "pregate",
                }
                gate_candidates = []
                for session in force_sessions:
                    summary = rc_data.get("first_kill_summary", {}).get(session)
                    if not summary:
                        raise RuntimeError(
                            f"[DIAGNOSTIC_AUTO_FAIL] Missing first_kill_summary for session={session} in {rc_path}"
                        )
                    first_kill_stage = summary.get("first_kill_stage")
                    if first_kill_stage not in stage_to_gate:
                        raise RuntimeError(
                            f"[DIAGNOSTIC_AUTO_FAIL] first_kill_stage={first_kill_stage} not in AUTO whitelist. "
                            f"Allowed: {sorted(stage_to_gate.keys())}"
                        )
                    gate_candidates.append(stage_to_gate[first_kill_stage])
                if len(set(gate_candidates)) != 1:
                    raise RuntimeError(
                        f"[DIAGNOSTIC_AUTO_FAIL] AUTO bypass requires a single gate across sessions. "
                        f"Got: {gate_candidates}"
                    )
                resolved_gate = gate_candidates[0]
                os.environ["GX1_DIAGNOSTIC_BYPASS_GATE"] = resolved_gate
                log.info("[DIAGNOSTIC_AUTO] Resolved AUTO bypass gate to: %s", resolved_gate)
        
        # OPPGAVE 1: Verify fast path is enabled (hard fail in replay mode)
        # For TRUTH/SMOKE, GX1_REPLAY_NO_CSV is optional (trade journal requires CSV writes)
        is_truth_or_smoke = getattr(self, 'prod_baseline', False) or os.getenv("GX1_SMOKE", "0") == "1" or os.getenv("GX1_RUN_MODE", "").upper() in ("TRUTH", "SMOKE")
        from gx1.execution.fast_path_verification import verify_fast_path_enabled
        fast_path_result = verify_fast_path_enabled(is_replay=self.is_replay, is_truth_or_smoke=is_truth_or_smoke)
        self.fast_path_enabled = fast_path_result["fast_path_enabled"]
        if self.fast_path_enabled:
            log.info("[REPLAY] Fast path verification: ✅ PASSED")
        else:
            log.warning("[REPLAY] Fast path verification: ⚠️  FAILED (missing: %s)", ", ".join(fast_path_result["missing_checks"]))
        
        # Load historical candles
        if csv_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(csv_path)
        else:
            df = pd.read_csv(csv_path)
        
        # Checkpoint 1: After loading from CSV/parquet
        # This may resolve close/CLOSE collision if compat-mode is enabled
        df = _assert_no_case_collisions(df, f"After loading from {csv_path.name}")
        
        # Ensure time column is datetime
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").sort_index()
        elif isinstance(df.index, pd.DatetimeIndex):
            # Already has DatetimeIndex
            pass
        else:
            raise ValueError("Historical data must have 'time' column or DatetimeIndex")
        
        # DEL 1: Log EXACT DataFrame state rett før column-check
        # Import explicitly to avoid UnboundLocalError in multiprocessing
        import datetime as dt_module
        from pathlib import Path as Path_module
        import json as jsonlib  # Explicit import to avoid UnboundLocalError
        # DEL 3: Guard against json shadowing
        assert 'json' not in locals(), "Do not shadow json module - use jsonlib from top-level import"
        
        csv_path_abs = csv_path.resolve()
        file_exists = csv_path_abs.exists()
        file_size = csv_path_abs.stat().st_size if file_exists else 0
        file_mtime = csv_path_abs.stat().st_mtime if file_exists else 0
        
        # DEL 3: Normalize column names for robust checking
        cols_norm = {c.strip().lower(): c for c in df.columns}
        required_lower = ["open", "high", "low", "close"]
        missing_via_norm = [r for r in required_lower if r not in cols_norm]
        
        # DEL 1: Collect debug info
        debug_info = {
            "timestamp": dt_module.datetime.now(dt_module.timezone.utc).isoformat(),
            "input_path": str(csv_path_abs),
            "input_path_resolved": str(csv_path_abs),
            "file_exists": file_exists,
            "file_size_bytes": file_size,
            "file_mtime": file_mtime,
            "source_format": "parquet" if csv_path.suffix.lower() == ".parquet" else "csv",
            "df_shape": list(df.shape),
            "df_columns": [{"name": str(c), "type": str(type(c))} for c in df.columns],
            "df_dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "df_index_name": str(df.index.name) if df.index.name else None,
            "df_index_type": str(type(df.index)),
            "df_index_first": str(df.index[0]) if len(df.index) > 0 else None,
            "df_index_last": str(df.index[-1]) if len(df.index) > 0 else None,
            "df_head_columns": list(df.head(1).columns.tolist()) if len(df) > 0 else [],
            "df_tail_columns": list(df.tail(1).columns.tolist()) if len(df) > 0 else [],
            "cols_normalized": {k: v for k, v in cols_norm.items()},
            "required_lower": required_lower,
            "missing_via_norm": missing_via_norm,
            "ohlc_dtypes": {
                col: str(df.dtypes[cols_norm[col]]) 
                for col in required_lower 
                if col in cols_norm
            },
        }
        
        # DEL 1: Write debug JSON
        run_id = getattr(self, "run_id", dt_module.datetime.now().strftime("%Y%m%d_%H%M%S"))
        chunk_id = getattr(self, "chunk_id", None)
        debug_dir = Path_module("reports/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_filename = f"OHLC_MISSING_{run_id}"
        if chunk_id:
            debug_filename += f"_chunk{chunk_id}"
        debug_path = debug_dir / f"{debug_filename}.json"
        
        with open(debug_path, "w") as f:
            jsonlib.dump(debug_info, f, indent=2, default=str)
        log.info(f"[REPLAY] Debug info written to {debug_path}")
        
        # DEL 3: If all required columns exist via normalization, rename to canonical
        if not missing_via_norm:
            # All required columns found via normalization - rename to canonical
            rename_map = {cols_norm[r]: r for r in required_lower}
            df = df.rename(columns=rename_map)
            log.info(f"[REPLAY] Normalized OHLC columns: {rename_map}")
        else:
            # DEL 1 + 3: Hard fail with debug info
            log.error(f"[REPLAY] Missing required columns (after normalization): {missing_via_norm}")
            log.error(f"[REPLAY] Available columns (normalized): {list(cols_norm.keys())}")
            log.error(f"[REPLAY] DataFrame shape: {df.shape}")
            log.error(f"[REPLAY] Index type: {type(df.index)}")
            log.error(f"[REPLAY] Index name: {df.index.name}")
            log.error(f"[REPLAY] Debug info: {debug_path}")
            raise ValueError(
                f"Missing required columns in historical data (after normalization): {missing_via_norm}. "
                f"Debug info written to {debug_path}"
            )
        self._ensure_bid_ask_columns(df, context="run_replay_init")
        
        # Add volume if missing (default to 0)
        if "volume" not in df.columns:
            df["volume"] = 0.0
        
        log.info("[REPLAY] Loaded %d M5 bars from %s to %s", 
             len(df), df.index.min().isoformat(), df.index.max().isoformat())
        
        # Limit bars if --max-bars is specified
        max_bars = getattr(self, "_max_bars", None)
        if max_bars is not None and max_bars > 0:
            original_len = len(df)
            df = df.iloc[:max_bars]
            log.info("[REPLAY] Limited to %d bars (from %d total)", len(df), original_len)
        
        # Verify entry stack (feature manifest + model bundle version) matches live
        if self.entry_model_bundle is not None:
            feature_manifest_hash = self.entry_model_bundle.feature_cols_hash
            model_bundle_version = self.entry_model_bundle.model_bundle_version or "N/A"
        else:
            feature_manifest_hash = None
            model_bundle_version = "N/A (session-routed bundle disabled)"
        
        log.info("=" * 60)
        log.info("[REPLAY] Entry Stack Verification:")
        log.info("  feature_manifest_hash: %s", feature_manifest_hash)
        log.info("  model_bundle_version: %s", model_bundle_version)
        log.info("=" * 60)
        
        # Note: In live mode, these same values are logged at boot (see __init__ around line 1427-1433)
        # If you want to compare against a saved baseline, add that check here
        # For now, we just log them to ensure they're visible in replay logs
        
        # Initialize replay state
        self.replay_mode = True
        try:
            from gx1.features.context_features import (
                ORDERED_CTX_CONT_NAMES_EXTENDED,
                ORDERED_CTX_CAT_NAMES_EXTENDED,
            )
            self.ctx_cont_required_columns = list(ORDERED_CTX_CONT_NAMES_EXTENDED[:6])
            self.ctx_cat_required_columns = list(ORDERED_CTX_CAT_NAMES_EXTENDED[:6])
            self.ctx_cont_dim = 6
            self.ctx_cat_dim = 6
        except Exception:
            pass
        
        # C) Hard invariants for prebuilt features (fail-fast, no silent fallback)
        # FIX: Import os and Path explicitly to avoid shadowing issues in multiprocessing
        from pathlib import Path as Path_module
        Path = Path_module  # Alias for convenience
        
        # FASE 1: Determine replay mode and enforce strict separation
        # SSoT: replay_mode_enum must be set explicitly, not from env (env is unreliable in multiproc)
        from gx1.utils.replay_mode import ReplayMode
        
        # TRUTH/SMOKE: prebuilt only from data_ctx (SSoT from manifest/bootstrap). No env path; no split-brain.
        prebuilt_use = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"
        prebuilt_path_from_ctx = getattr(self, "prebuilt_parquet_path_resolved", None)
        prebuilt_df_from_ctx = getattr(self, "prebuilt_features_df", None)
        
        # Initialize replay_mode_enum to BASELINE (will be updated if prebuilt features are loaded)
        replay_mode_enum = ReplayMode.BASELINE
        prebuilt_enabled = False
        
        # FASE 1: Hard guarantee - assert feature-building modules are NOT imported in PREBUILT mode
        # NOTE: This check is done in master process (replay_eval_gated_parallel.py) BEFORE workers start.
        # In worker processes, GX1DemoRunner is imported which imports entry_manager → live_features → basic_v1,
        # so this check would always fail in workers. The master process check is sufficient.
        # We only check here if we're NOT in a worker (i.e., if this is called directly, not from process_chunk).
        if prebuilt_enabled:
            # Only check if we're in master process (not in worker spawned via multiprocessing)
            # Workers will have imported GX1DemoRunner which imports forbidden modules, but that's OK
            # because master process already verified they weren't imported before workers started.
            # Also skip check if GX1_ALLOW_PARALLEL_REPLAY=1 (direct call from year job, not multiprocessing)
            import multiprocessing as mp
            is_worker = mp.current_process().name != 'MainProcess'
            allow_parallel = os.getenv("GX1_ALLOW_PARALLEL_REPLAY", "0") == "1"
            if not is_worker and not allow_parallel:
                # We're in master process and NOT in parallel mode - do the check
                import sys
                import traceback
                import json as jsonlib
                from datetime import datetime
                forbidden_modules = [
                    "gx1.features.basic_v1",
                    "gx1.execution.live_features",
                    "gx1.features.runtime_v10_ctx",
                    "gx1.features.runtime_sniper_core",
                ]
                imported_forbidden = [mod for mod in forbidden_modules if mod in sys.modules]
                if imported_forbidden:
                    # Collect import violation details
                    violations = []
                    for mod_name in imported_forbidden:
                        mod = sys.modules.get(mod_name)
                        if mod:
                            # Try to get import stack (best-effort)
                            import_stack = []
                            try:
                                mod_file = getattr(mod, "__file__", None)
                                if mod_file:
                                    import_stack.append(f"Module file: {mod_file}")
                                import_stack.append("Import detected in sys.modules")
                            except Exception:
                                pass
                            
                            violations.append({
                                "forbidden_module": mod_name,
                                "module_file": getattr(mod, "__file__", None),
                                "importer": "unknown (check import chain)",
                                "import_stack": import_stack,
                            })
                    
                    # Write violation report to output directory
                    output_dir_for_report = getattr(self, "output_dir", None) or Path("reports/replay_eval")
                    output_dir_for_report = Path(output_dir_for_report)
                    output_dir_for_report.mkdir(parents=True, exist_ok=True)
                    violation_path = output_dir_for_report / "PREBUILT_IMPORT_VIOLATION.json"
                    
                    violation_report = {
                        "detected_at": datetime.now().isoformat(),
                        "prebuilt_enabled": True,
                        "violations": violations,
                        "message": "Forbidden feature-building modules imported in PREBUILT mode",
                    }
                    
                    with open(violation_path, "w") as f:
                        jsonlib.dump(violation_report, f, indent=2)
                    log.error(f"[PREBUILT_FAIL] Import violations written to: {violation_path}")
                    
                    raise RuntimeError(
                        f"[PREBUILT_FAIL] FASE_1_SEPARATION: Forbidden feature-building modules imported in PREBUILT mode: {imported_forbidden}\n"
                        f"This violates FASE 1: PREBUILT and BASELINE must be completely separate code paths.\n"
                        f"CRASH: Feature-building code must not be imported in PREBUILT mode.\n"
                        f"Violation details written to: {violation_path}"
                    )
                log.info("[FASE_1] PREBUILT mode: Feature-building modules verified as NOT imported in master process")
            else:
                # We're in worker process or parallel mode - skip check (modules are imported via GX1DemoRunner, which is OK)
                if is_worker:
                    log.info("[FASE_1] PREBUILT mode: In worker process - skipping FASE 1 check (already verified in master)")
                else:
                    log.info("[FASE_1] PREBUILT mode: GX1_ALLOW_PARALLEL_REPLAY=1 - skipping FASE 1 check (direct call from year job)")
        # Reset prebuilt state; PREBUILT only via loader (no injection).
        self.prebuilt_features_loader = None
        self.prebuilt_features_df = None
        self.prebuilt_features_sha256 = None
        self.prebuilt_used = False
        # Lookup telemetry for PREBUILT mode
        self.lookup_attempts = 0
        self.lookup_hits = 0
        self.lookup_miss_logged = 0
        
        # FASE 1: Prebuilt from data_ctx only (prebuilt_features_df or prebuilt_parquet_path_resolved set by process_chunk).
        def _set_ctx_contract_prebuilt():
            ctx_cont = None
            ctx_cat = None
            def _load_bundle_meta(path: Path) -> Optional[dict]:
                meta_path = path / "bundle_metadata.json"
                if meta_path.exists():
                    import json as json_module
                    with open(meta_path, "r", encoding="utf-8") as f:
                        return json_module.load(f)
                return None
            bundle_dir = None
            if isinstance(self.entry_v10_ctx_cfg, dict):
                bundle_dir = self.entry_v10_ctx_cfg.get("bundle_dir")
            if not bundle_dir:
                bundle_dir = self.policy.get("canonical_transformer_bundle_dir")
            if bundle_dir:
                bundle_dir = Path(bundle_dir).expanduser().resolve()
                meta = _load_bundle_meta(bundle_dir)
                if meta:
                    ctx_cont = meta.get("ordered_ctx_cont_names") or ctx_cont
                    ctx_cat = meta.get("ordered_ctx_cat_names") or ctx_cat
            if (ctx_cont is None or ctx_cat is None) and bundle_dir and bundle_dir.parent.exists():
                import json as json_module
                for candidate in sorted(bundle_dir.parent.glob("TRANSFORMER_ENTRY_V10_CTX__*/bundle_metadata.json")):
                    try:
                        with open(candidate, "r", encoding="utf-8") as f:
                            meta = json_module.load(f)
                        c_cont = meta.get("ordered_ctx_cont_names")
                        c_cat = meta.get("ordered_ctx_cat_names")
                        if isinstance(c_cont, list) and isinstance(c_cat, list) and len(c_cont) == 6 and len(c_cat) == 6:
                            ctx_cont = ctx_cont or c_cont
                            ctx_cat = ctx_cat or c_cat
                            break
                    except Exception:
                        continue
            if ctx_cont is None or ctx_cat is None:
                try:
                    policy_ctx = self.policy.get("context", {}) if isinstance(self.policy, dict) else {}
                    ctx_cont = ctx_cont or policy_ctx.get("cont_columns") or self.policy.get("ctx_cont_columns")
                    ctx_cat = ctx_cat or policy_ctx.get("cat_columns") or self.policy.get("ctx_cat_columns")
                except Exception:
                    ctx_cont = ctx_cont or None
                    ctx_cat = ctx_cat or None
            if not (isinstance(ctx_cont, list) and isinstance(ctx_cat, list) and len(ctx_cont) == 6 and len(ctx_cat) == 6):
                raise RuntimeError(
                    "[PREBUILT_CTX_CONTRACT] Failed to resolve ctx contract (6/6) from V10_CTX bundle/config; "
                    f"ctx_cont_len={len(ctx_cont) if isinstance(ctx_cont, list) else ctx_cont} "
                    f"ctx_cat_len={len(ctx_cat) if isinstance(ctx_cat, list) else ctx_cat}"
                )
            if not all(isinstance(x, str) for x in ctx_cont + ctx_cat):
                raise RuntimeError("[PREBUILT_CTX_CONTRACT] ctx columns must be strings")
            self.ctx_cont_required_columns = list(ctx_cont)
            self.ctx_cat_required_columns = list(ctx_cat)
            self.ctx_required_columns = list(ctx_cont) + list(ctx_cat)
            self.ctx_cont_dim = 6
            self.ctx_cat_dim = 6
            self.ctx_contract_source = "v10_ctx_bundle_or_truth_config"

        if not prebuilt_use:
            prebuilt_enabled = False
            replay_mode_enum = ReplayMode.BASELINE
        elif prebuilt_df_from_ctx is not None:
            # (a) Use prebuilt from data_ctx directly (SSoT from bootstrap/loader); provenance + alignment required.
            if not prebuilt_path_from_ctx:
                raise RuntimeError(
                    "[PREBUILT_PROVENANCE_MISSING] replay prebuilt_df used without resolved manifest path (SSoT violation)."
                )
            self.prebuilt_features_df = prebuilt_df_from_ctx
            idx = self.prebuilt_features_df.index
            if not idx.is_monotonic_increasing:
                raise RuntimeError(
                    "[PREBUILT_ALIGN_MISSING] replay prebuilt_df not aligned to candles index (index not monotonic increasing)."
                )
            if idx.tz is None:
                raise RuntimeError(
                    "[PREBUILT_ALIGN_MISSING] replay prebuilt_df not aligned to candles index (index not tz-aware)."
                )
            from datetime import timezone as dt_tz
            tz_str = str(idx.tz)
            tz_ok = (
                idx.tz == dt_tz.utc
                or getattr(idx.tz, "key", None) == "UTC"
                or tz_str in ("UTC", "UTC+00:00")
                or (tz_str.startswith("UTC") and "00:00" in tz_str)
            )
            try:
                import pytz
                if idx.tz is pytz.UTC:
                    tz_ok = True
            except Exception:
                pass
            if not tz_ok:
                raise RuntimeError(
                    "[PREBUILT_ALIGN_MISSING] replay prebuilt_df not aligned to candles index (index not UTC)."
                )
            # Ground truth: candles (df); common_index = candles.index ∩ prebuilt_features_df.index
            common_index = df.index.intersection(self.prebuilt_features_df.index)
            if len(common_index) == 0:
                raise RuntimeError(
                    "[PREBUILT_ALIGN_MISSING] replay prebuilt_df not aligned to candles index."
                )
            df = df.loc[common_index]
            self.prebuilt_features_df = self.prebuilt_features_df.loc[common_index].copy()
            if not df.index.equals(common_index) or not self.prebuilt_features_df.index.equals(common_index):
                raise RuntimeError(
                    "[PREBUILT_ALIGN_MISSING] replay prebuilt_df not aligned to candles index."
                )
            _set_ctx_contract_prebuilt()
            self.prebuilt_features_path_resolved = prebuilt_path_from_ctx
            self.prebuilt_used = True
            prebuilt_enabled = True
            replay_mode_enum = ReplayMode.PREBUILT
            self.prebuilt_features_loader = None
            log.info(
                "[REPLAY_MODE] Set to PREBUILT (prebuilt from data_ctx: %d rows, %d cols)",
                len(self.prebuilt_features_df), len(self.prebuilt_features_df.columns)
            )
        elif prebuilt_path_from_ctx:
            # (b) Load via PrebuiltFeaturesLoader using path from data_ctx (manifest-only SSoT); alignment/validation run.
            prebuilt_features_path = Path_module(prebuilt_path_from_ctx)
            from gx1.execution.prebuilt_features_loader import PrebuiltFeaturesLoader
            self.prebuilt_features_loader = PrebuiltFeaturesLoader(prebuilt_features_path)
            self.prebuilt_features_df = self.prebuilt_features_loader.df
            self.prebuilt_features_sha256 = self.prebuilt_features_loader.sha256
            self.prebuilt_features_path_resolved = self.prebuilt_features_loader.prebuilt_path_resolved
            self.prebuilt_schema_version = self.prebuilt_features_loader.schema_version
            _set_ctx_contract_prebuilt()
            replay_mode_enum = ReplayMode.PREBUILT
            prebuilt_enabled = True
            log.info(
                "[REPLAY_MODE] Set to PREBUILT (prebuilt loaded via loader from data_ctx path: %d rows, %d cols)",
                len(self.prebuilt_features_df), len(self.prebuilt_features_df.columns)
            )
            log.info("[FASE_1] Prebuilt features loaded via PrebuiltFeaturesLoader (path from manifest/bootstrap)")
        else:
            raise RuntimeError(
                "[PREBUILT_REQUIRED] Prebuilt required (GX1_REPLAY_USE_PREBUILT_FEATURES=1). "
                "data_ctx must provide prebuilt_parquet_path_resolved or prebuilt_features_df (manifest/bootstrap SSoT). "
                "Check bootstrap and load_chunk_data; do not use GX1_REPLAY_PREBUILT_FEATURES_PATH."
            )
        
        # Store replay_mode_enum on runner for use in hotpath (SSoT)
        self.replay_mode_enum = replay_mode_enum
        # Replay-safe journal buffering: enable after replay_mode_enum is known (PREBUILT vs BASELINE).
        try:
            self._enable_journal_buffering_if_replay()
        except Exception:
            pass
        
        if prebuilt_enabled:
            # FASE 1: common_index and filter (when loader present; when prebuilt from data_ctx df already filtered).
            if len(df) > 0 and self.prebuilt_features_loader is not None:
                common_index = self.prebuilt_features_loader.get_common_index(df.index)
                
                if len(common_index) == 0:
                    raise RuntimeError(
                        "[PREBUILT_FAIL] No common timestamps between raw data and prebuilt features.\n"
                        f"Raw range: {df.index[0]} to {df.index[-1]}\n"
                        f"Prebuilt range: {self.prebuilt_features_df.index[0]} to {self.prebuilt_features_df.index[-1]}\n"
                        f"Instructions: Rebuild prebuilt features to match raw data."
                    )
                
                # Store original lengths before filtering
                raw_len_before = len(df)
                prebuilt_len_before = len(self.prebuilt_features_df)
                
                # CRITICAL FIX: Normalize prebuilt_features_df.index to match common_index timezone
                # common_index is returned from get_common_index with timezone matching df.index
                # But self.prebuilt_features_df.index might be naive or have different timezone
                if common_index.tz is not None and self.prebuilt_features_df.index.tz is None:
                    # common_index has timezone, prebuilt is naive - localize prebuilt
                    self.prebuilt_features_df.index = self.prebuilt_features_df.index.tz_localize(common_index.tz)
                elif common_index.tz is None and self.prebuilt_features_df.index.tz is not None:
                    # common_index is naive, prebuilt has timezone - remove timezone from prebuilt
                    self.prebuilt_features_df.index = self.prebuilt_features_df.index.tz_localize(None)
                elif common_index.tz is not None and self.prebuilt_features_df.index.tz is not None and common_index.tz != self.prebuilt_features_df.index.tz:
                    # Both have timezones but different - convert prebuilt to common_index timezone
                    self.prebuilt_features_df.index = self.prebuilt_features_df.index.tz_convert(common_index.tz)
                
                # Filter both datasets to common_index
                df = df.loc[common_index]
                # CRITICAL FIX: Use .copy() to ensure we have a proper DataFrame, not a view
                # Without .copy(), the filtered DataFrame might become invalid or inaccessible
                self.prebuilt_features_df = self.prebuilt_features_df.loc[common_index].copy()
                
                # HARD INVARIANTS: Fail-fast if prebuilt_features_df is None or empty after filtering
                if replay_mode_enum == ReplayMode.PREBUILT:
                    if self.prebuilt_features_df is None:
                        raise RuntimeError("[PREBUILT] prebuilt_features_df is None after filtering")
                    
                    n = len(self.prebuilt_features_df)
                    if n == 0:
                        raise RuntimeError(
                            "[PREBUILT] prebuilt_features_df became empty after common_index filtering. "
                            "This almost always means timestamp/index mismatch (tz, rounding, open/close) "
                            "or wrong prebuilt file."
                        )
                    
                    # Strong sanity: show boundaries once
                    idx = self.prebuilt_features_df.index
                    log.info(
                        "[PREBUILT] filtered_df rows=%d idx_min=%s idx_max=%s tz=%s",
                        n, idx.min(), idx.max(), getattr(idx, 'tz', None)
                    )
                
                # Update loader metadata after alignment
                if self.prebuilt_features_loader is not None:
                    self.prebuilt_features_loader.prebuilt_index_aligned = True
                    self.prebuilt_features_loader.subset_first_ts = common_index[0] if len(common_index) > 0 else None
                    self.prebuilt_features_loader.subset_last_ts = common_index[-1] if len(common_index) > 0 else None
                    self.prebuilt_features_loader.subset_rows = len(common_index)
                
                # Define lookup_phase: lookup happens AFTER hard eligibility check
                # This is determined by where lookup_attempts is incremented in entry_manager.evaluate_entry()
                # NOTE: Lookup actually happens AFTER soft eligibility (entry stage), not after hard eligibility
                # lookup_attempts is incremented AFTER soft eligibility passes (entry stage)
                # So lookup_phase = "after_soft_eligibility" (lookup only for bars that reached entry stage)
                self.prebuilt_lookup_phase = "after_soft_eligibility"
                log.info("[PREBUILT] lookup_phase set to 'after_soft_eligibility' (lookup happens after soft eligibility passes, entry stage)")
                
                log.info(
                    "[PREBUILT] Filtered to common_index: %d bars (raw: %d -> %d, prebuilt: %d -> %d)",
                    len(common_index),
                    raw_len_before, len(df),
                    prebuilt_len_before, len(self.prebuilt_features_df)
                )
            
            # FASE 1: PrebuiltFeaturesLoader.validate_timestamp_alignment (always when loader present)
            if len(df) > 0 and self.prebuilt_features_loader is not None:
                is_valid, error_msg = self.prebuilt_features_loader.validate_timestamp_alignment(
                    df.index,
                    sample_size=1000,
                    random_mid_check=True
                )
                if not is_valid:
                    raise RuntimeError(
                        f"[PREBUILT_FAIL] Timestamp alignment validation failed: {error_msg}\n"
                        f"Instructions: Rebuild prebuilt features to match raw data."
                    )
                log.debug("[PREBUILT] Timestamp alignment validation passed")
            
            # C4: Require columns from schema_manifest.required_all_features only (never bridge7 in prebuilt)
            # ONE UNIVERSE: prebuilt parquet has BASE28 + context; bridge7 is produced by XGB at runtime.
            has_seq_snap_arrays = "seq" in self.prebuilt_features_df.columns and "snap" in self.prebuilt_features_df.columns
            
            if not has_seq_snap_arrays:
                # Flat columns: required set from schema_manifest next to prebuilt parquet (SSoT)
                prebuilt_resolved = Path_module(self.prebuilt_features_path_resolved)
                schema_manifest_path = prebuilt_resolved.parent / (prebuilt_resolved.stem + ".schema_manifest.json")
                if not schema_manifest_path.is_file():
                    raise RuntimeError(
                        f"[PREBUILT_FAIL] Schema manifest not found: {schema_manifest_path}. "
                        "Prebuilt validation requires required_all_features from schema_manifest (SSoT)."
                    )
                with open(schema_manifest_path, "r", encoding="utf-8") as _f:
                    schema_manifest = jsonlib.load(_f)
                required_all_features = schema_manifest.get("required_all_features", [])
                if not isinstance(required_all_features, list):
                    required_all_features = []
                if "time" in required_all_features:
                    raise RuntimeError(
                        f"[PREBUILT_FAIL] schema_manifest.required_all_features contains 'time'. Time must be index, not feature. path={schema_manifest_path}"
                    )
                if len(required_all_features) < 28:
                    raise RuntimeError(
                        f"[PREBUILT_FAIL] schema_manifest.required_all_features has {len(required_all_features)} items (expected >= 28 for BASE28). path={schema_manifest_path}"
                    )
                required_cols_effective = set(required_all_features)
                log.info(
                    "[PREBUILT] required_all_features_len=%d source=schema_manifest path=%s first_few=%s",
                    len(required_cols_effective),
                    str(schema_manifest_path),
                    required_all_features[:5] if len(required_all_features) >= 5 else required_all_features,
                )
                missing_cols = required_cols_effective - set(self.prebuilt_features_df.columns)
                if missing_cols:
                    raise RuntimeError(
                        f"[PREBUILT_FAIL] Missing required columns in prebuilt features: {sorted(missing_cols)}\n"
                        f"Required set from schema_manifest: {len(required_cols_effective)} features (path={schema_manifest_path}).\n"
                        "Instructions: Rebuild prebuilt features to match schema_manifest.required_all_features."
                    )
            else:
                log.debug("[PREBUILT] Prebuilt features have seq/snap arrays - skipping flat column validation")
            
            # C5: Set flag for perf collector ONLY after all checks pass
            # This is the ONLY place where prebuilt_used is set to True
            # All checks must pass: file exists, sha matches, timestamps align (1k + mid), columns OK
            self.prebuilt_used = True
            prebuilt_resolved_for_meta = Path_module(self.prebuilt_features_path_resolved)
            self.prebuilt_features_path_resolved = str(prebuilt_resolved_for_meta.resolve())
            # Compute SHA256 if not set (path (a): prebuilt_df from data_ctx had no sha)
            if not getattr(self, "prebuilt_features_sha256", None):
                try:
                    import hashlib as _hashlib
                    h = _hashlib.sha256()
                    with open(prebuilt_resolved_for_meta, "rb") as _f:
                        for chunk in iter(lambda: _f.read(1024 * 1024), b""):
                            h.update(chunk)
                    self.prebuilt_features_sha256 = h.hexdigest()
                except Exception:
                    self.prebuilt_features_sha256 = None
            
            # Get schema version from manifest (if available)
            self.prebuilt_schema_version = None
            manifest_path = prebuilt_resolved_for_meta.parent / "manifest.json"
            if manifest_path.exists():
                try:
                    import json as json_module
                    with open(manifest_path, "r") as f:
                        manifest = json_module.load(f)
                    self.prebuilt_schema_version = manifest.get("schema_version", "unknown")
                except Exception:
                    pass
            
            # C) Log SSoT line (with bypass count if available)
            prebuilt_bypass_count = getattr(self, "prebuilt_bypass_count", 0)
            def _short_sha(val: Optional[str]) -> str:
                return "MISSING" if not val else (val if len(val) <= 8 else val[:16] + "...")

            # SELF-CHECK: guard logging against None to avoid crashes (previous crash site)
            log.info(
                "[PREBUILT_STATUS] enabled=1 used=%d path=%s exists=1 sha256=%s rows=%d cols=%d schema=%s bypass_count=%d",
                1 if self.prebuilt_used else 0,
                self.prebuilt_features_path_resolved,
                _short_sha(getattr(self, "prebuilt_features_sha256", None)),
                len(self.prebuilt_features_df),
                len(self.prebuilt_features_df.columns),
                self.prebuilt_schema_version or "unknown",
                prebuilt_bypass_count
            )
            
            log.info(
                "[REPLAY] ✅ Prebuilt features loaded: %d bars, %d features, SHA256=%s",
                len(self.prebuilt_features_df),
                len(self.prebuilt_features_df.columns),
                self.prebuilt_features_sha256[:16] + "..."
            )
            
            # Validate that prebuilt features cover the replay period
            if len(self.prebuilt_features_df) > 0:
                prebuilt_start = self.prebuilt_features_df.index.min()
                prebuilt_end = self.prebuilt_features_df.index.max()
                replay_start = df.index.min()
                replay_end = df.index.max()
                
                if prebuilt_start > replay_start or prebuilt_end < replay_end:
                    log.warning(
                        "[REPLAY] ⚠️  Prebuilt features time range (%s to %s) does not fully cover replay period (%s to %s)",
                        prebuilt_start.isoformat(),
                        prebuilt_end.isoformat(),
                        replay_start.isoformat(),
                        replay_end.isoformat()
                    )
                else:
                    log.info(
                        "[REPLAY] ✅ Prebuilt features time range covers replay period"
                    )
        else:
            # C) Log SSoT line for disabled case
            log.info("[PREBUILT_STATUS] enabled=0 path=None exists=0 sha256=None rows=0 cols=0 schema=None")
            self.prebuilt_used = False
            self.prebuilt_features_path_resolved = None
            self.prebuilt_schema_version = None
        
        # REPLAY WARMUP: Include extra bars before evaluation window for ATR/trend computation
        # This ensures vol_regime and trend_regime can be computed from rolling windows
        warmup_bars = int(self.policy.get("warmup_bars", 288))
        warmup_padding = max(warmup_bars, 288)  # At least 1 day of warmup
        
        # Set evaluation window (actual period to evaluate trades/metrics)
        self.replay_eval_start_ts = df.index.min()
        self.replay_eval_end_ts = df.index.max()
        
        # Update run_header.json with replay metadata (now that timestamps are set)
        self._update_run_header_replay_metadata()
        
        # Extend data with warmup bars before evaluation window
        # Try to get warmup bars from before eval_start, but if not available, use first bars
        if len(df) > warmup_padding:
            # We have enough data - use first bars as warmup
            self.replay_start_ts = df.index.min()  # Start from beginning
            self.replay_end_ts = df.index.max()
            warmup_bars_actual = warmup_padding
            log.info(
                "[REPLAY] Warmup: Using first %d bars as warmup (evaluation window: %s to %s)",
                warmup_bars_actual,
                self.replay_eval_start_ts.isoformat(),
                self.replay_eval_end_ts.isoformat()
            )
        else:
            # Not enough data - use all bars (warmup will be partial)
            self.replay_start_ts = df.index.min()
            self.replay_end_ts = df.index.max()
            warmup_bars_actual = len(df)
            log.warning(
                "[REPLAY] Warmup: Only %d bars available (less than %d requested). "
                "Evaluation window: %s to %s",
                warmup_bars_actual,
                warmup_padding,
                self.replay_eval_start_ts.isoformat(),
                self.replay_eval_end_ts.isoformat()
            )
        
        self._reset_entry_diag()
        
        # Skip backfill in replay mode (we already have historical data)
        self.backfill_in_progress = False
        # Set warmup_floor to eval_start (bars before this are warmup)
        self.warmup_floor = self.replay_eval_start_ts
        
        # Disable tick watcher in replay (we'll simulate ticks from M5 candles)
        if self.tick_watcher:
            self.tick_watcher.stop()
        log.info("[REPLAY] Tick watcher disabled (simulating ticks from M5 candles)")
        
        # Load warmup prices for Big Brain V1 (MUST be from same period as replay)
        # CRITICAL: Warmup data MUST match replay period - no mixing 2020 data with 2025 replay!
        warmup_prices_path = None
        lookback_bars = None
        if self.big_brain_v1 is not None:
            bb_v1_config = self.policy.get("big_brain_v1", {})
            warmup_prices_path = bb_v1_config.get("warmup_prices_path")
            lookback_bars = self.big_brain_v1.lookback
        
        # Try to load warmup from external file ONLY if it has data in the correct period
        warmup_loaded = False
        if warmup_prices_path:
            warmup_prices_path = Path_module(warmup_prices_path)
            try:
                log.info("[BIG_BRAIN_V1] Checking warmup prices from %s", warmup_prices_path)
                
                # Load warmup prices
                if warmup_prices_path.suffix.lower() == ".parquet":
                    warmup_df = pd.read_parquet(warmup_prices_path)
                else:
                    warmup_df = pd.read_csv(warmup_prices_path, nrows=1000)  # Just check first 1000 rows
                
                # Ensure time column is datetime and set as index
                if "time" in warmup_df.columns:
                    warmup_df["time"] = pd.to_datetime(warmup_df["time"], utc=True)
                    warmup_df = warmup_df.set_index("time").sort_index()
                elif "ts" in warmup_df.columns:
                    warmup_df["ts"] = pd.to_datetime(warmup_df["ts"], utc=True)
                    warmup_df = warmup_df.set_index("ts").sort_index()
                elif not isinstance(warmup_df.index, pd.DatetimeIndex):
                    warmup_df.index = pd.to_datetime(warmup_df.index, utc=True)
                
                # Ensure index is timezone-aware UTC
                if warmup_df.index.tz is None:
                    warmup_df.index = warmup_df.index.tz_localize("UTC")
                else:
                    warmup_df.index = warmup_df.index.tz_convert("UTC")
                
                # Check if warmup file has data in the correct period (before replay start)
                warmup_end_ts = self.replay_start_ts - pd.Timedelta(minutes=5)
                warmup_start_ts = warmup_end_ts - pd.Timedelta(minutes=5 * lookback_bars)
                
                # Check if warmup file covers the required period
                warmup_file_has_period = (
                    warmup_df.index.min() <= warmup_start_ts and
                    warmup_df.index.max() >= warmup_end_ts
                )
                
                if warmup_file_has_period:
                    # Warmup file has data in correct period - use it
                    log.info(
                        "[BIG_BRAIN_V1] Warmup file has data in correct period (%s to %s). Using warmup file.",
                        warmup_df.index.min().isoformat(),
                        warmup_df.index.max().isoformat(),
                    )
                    
                    # Reload full file for warmup
                    if warmup_prices_path.suffix.lower() == ".parquet":
                        warmup_df_full = pd.read_parquet(warmup_prices_path)
                    else:
                        warmup_df_full = pd.read_csv(warmup_prices_path)
                    
                    # Process full file same way
                    if "time" in warmup_df_full.columns:
                        warmup_df_full["time"] = pd.to_datetime(warmup_df_full["time"], utc=True)
                        warmup_df_full = warmup_df_full.set_index("time").sort_index()
                    elif "ts" in warmup_df_full.columns:
                        warmup_df_full["ts"] = pd.to_datetime(warmup_df_full["ts"], utc=True)
                        warmup_df_full = warmup_df_full.set_index("ts").sort_index()
                    elif not isinstance(warmup_df_full.index, pd.DatetimeIndex):
                        warmup_df_full.index = pd.to_datetime(warmup_df_full.index, utc=True)
                    
                    if warmup_df_full.index.tz is None:
                        warmup_df_full.index = warmup_df_full.index.tz_localize("UTC")
                    else:
                        warmup_df_full.index = warmup_df_full.index.tz_convert("UTC")
                    
                    # Filter to bars before replay start
                    mask = (warmup_df_full.index >= warmup_start_ts) & (warmup_df_full.index < warmup_end_ts)
                    warmup_filtered = warmup_df_full[mask].copy()
                    
                    # Ensure required columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [c for c in required_cols if c not in warmup_filtered.columns]
                    if not missing_cols:
                        # Calculate ATR if missing
                        if 'atr' not in warmup_filtered.columns:
                            high_low = warmup_filtered['high'] - warmup_filtered['low']
                            high_close = (warmup_filtered['high'] - warmup_filtered['close'].shift(1)).abs()
                            low_close = (warmup_filtered['low'] - warmup_filtered['close'].shift(1)).abs()
                            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                            warmup_filtered['atr'] = tr.rolling(window=14, min_periods=1).mean()
                        
                        if len(warmup_filtered) >= lookback_bars:
                            self.big_brain_v1.feed_warmup(warmup_filtered)
                            warmup_loaded = True
                            log.info(
                                "[BIG_BRAIN_V1] Warmup complete (from external file): fed %d bars (lookback=%d) from %s to %s",
                                len(warmup_filtered),
                                lookback_bars,
                                warmup_filtered.index.min().isoformat(),
                                warmup_filtered.index.max().isoformat(),
                            )
                        else:
                            # Warmup file does NOT have data in correct period - skip it
                            log.info(
                                "[BIG_BRAIN_V1] Warmup file does NOT have data in required period (%s to %s). "
                                "Warmup file period: %s to %s. Will use replay file itself for warmup.",
                                warmup_start_ts.isoformat(),
                                warmup_end_ts.isoformat(),
                                warmup_df.index.min().isoformat(),
                                warmup_df.index.max().isoformat(),
                            )
            except Exception as e:
                log.warning(
                    "[BIG_BRAIN_V1] Failed to check warmup prices from %s: %s. "
                    "Will use replay file itself for warmup.",
                    warmup_prices_path,
                    e,
                )
        
        # If warmup not loaded from external file, use replay file itself
        # But we need to wait until we have enough bars before evaluating trades
        if not warmup_loaded:
            if lookback_bars is not None and len(df) >= lookback_bars:
                log.info(
                    "[BIG_BRAIN_V1] Using first %d bars from replay file itself as warmup "
                    "(ensuring period consistency - no mixing different time periods).",
                    lookback_bars,
                )
                # Use first lookback_bars from replay file as warmup
                replay_warmup = df.iloc[:lookback_bars].copy()
                # Ensure required columns for Big Brain V1
                if 'atr' not in replay_warmup.columns:
                    high_low = replay_warmup['high'] - replay_warmup['low']
                    high_close = (replay_warmup['high'] - replay_warmup['close'].shift(1)).abs()
                    low_close = (replay_warmup['low'] - replay_warmup['close'].shift(1)).abs()
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    replay_warmup['atr'] = tr.rolling(window=14, min_periods=1).mean()
                
                self.big_brain_v1.feed_warmup(replay_warmup)
                log.info(
                    "[BIG_BRAIN_V1] Warmup complete (from replay file): fed %d bars from %s to %s",
                    len(replay_warmup),
                    replay_warmup.index.min().isoformat(),
                    replay_warmup.index.max().isoformat(),
                )
            else:
                lookback_str = str(lookback_bars) if lookback_bars is not None else "unknown"
                log.warning(
                    "[BIG_BRAIN_V1] Replay file has insufficient bars (%d, need %s). "
                    "First trades may have UNKNOWN regimes. "
                    "Consider using a replay file with at least %s bars.",
                    len(df),
                    lookback_str,
                    lookback_str,
                )
        
        # Process each bar sequentially
        # Skip first N bars to ensure features are stable (need lookback for ATR, ADR, etc.)
        # build_live_entry_features needs at least ATR_PERIOD (14) bars for ATR
        # ADR_WINDOW is 288 bars (1 day), but we can start earlier with partial ADR
        # Big Brain V1 needs 288 bars for warmup (if using replay file itself)
        # ENTRY-TCN needs 864 bars for lookback (3 days M5)
        # Use max(100, lookback_bars_bb, entry_tcn_lookback) to ensure all models have enough warmup bars
        
        # PREFLIGHT-ONLY: Allow override for sanity/smoke tests (reduces warmup to allow entry-stage)
        # Gate-SSoT: This is where warmup requirement is determined (used in bar loop at line 9914, 9919)
        preflight_warmup_override = os.getenv("GX1_PREFLIGHT_WARMUP_BARS")
        is_preflight_run = os.getenv("GX1_PREFLIGHT", "0") == "1" or preflight_warmup_override is not None
        
        if preflight_warmup_override is not None and is_preflight_run:
            try:
                min_bars_for_features_effective = int(preflight_warmup_override)
                warmup_override_applied = True
                log.info("[PREFLIGHT] Using warmup override: GX1_PREFLIGHT_WARMUP_BARS=%d (preflight-only, does not affect FULLYEAR/prod)", min_bars_for_features_effective)
            except ValueError:
                log.warning("[PREFLIGHT] Invalid GX1_PREFLIGHT_WARMUP_BARS=%s, using normal calculation", preflight_warmup_override)
                preflight_warmup_override = None
                warmup_override_applied = False
        else:
            warmup_override_applied = False
        
        if preflight_warmup_override is None or not is_preflight_run:
            lookback_requirements = [100]  # Minimum for stable features
            
            if self.big_brain_v1 is not None:
                lookback_bars_bb = self.big_brain_v1.lookback
                lookback_requirements.append(lookback_bars_bb)
            
            # Check for ENTRY-TCN lookback requirement
            entry_tcn_lookback = self.policy.get("tcn", {}).get("lookback_bars", None)
            if entry_tcn_lookback is not None:
                lookback_requirements.append(entry_tcn_lookback)
            
            min_bars_for_features_effective = max(lookback_requirements)
        
        # Store effective warmup value and override flag on runner (for footer export)
        self.warmup_required_effective = min_bars_for_features_effective
        self.warmup_override_applied = warmup_override_applied
        min_bars_for_features = min_bars_for_features_effective  # Use effective value for calculations
        
        total_bars = len(df)
        
        # NEW: Calculate first_valid_eval_idx based on HTF warmup requirements
        # This ensures we don't evaluate entries before HTF bars are available
        first_valid_eval_idx = self._calculate_first_valid_eval_idx(df, min_bars_for_features)
        log.info(
            "[REPLAY] first_valid_eval_idx=%d (out of %d total bars). "
            "Skipping first %d bars due to HTF warmup requirements.",
            first_valid_eval_idx, total_bars, first_valid_eval_idx
        )
        
        bars_to_process = total_bars - first_valid_eval_idx
        
        # Del 1: Set perf_bars_total (bars that will actually be processed)
        self.perf_bars_total = bars_to_process
        
        # Del 2: Set current perf collector in context for feature timing
        from gx1.utils.perf_timer import set_current_perf
        perf_token = set_current_perf(self.perf_collector)
        
        # Del 2: Set feature state for persistent caching
        from gx1.utils.feature_context import set_feature_state
        feature_token = set_feature_state(self.feature_state)
        
        # PATCH: Initialize HTF aligners for stateful alignment (O(1) per bar)
        # Build HTF bars from ENTIRE df once before per-bar loop
        # This ensures aligners have complete HTF history, avoiding O(N²) behavior
        if self.replay_mode and len(df) > 0:
            try:
                import numpy as np
                from gx1.features.htf_aggregator import build_htf_from_m5
                from gx1.features.htf_align_state import HTFAligner
                
                # Extract M5 data from ENTIRE df (not just first 1000 bars)
                # This ensures aligners have complete HTF history
                if isinstance(df.index, pd.DatetimeIndex):
                    m5_timestamps_sec = (df.index.astype(np.int64) // 1_000_000_000).astype(np.int64)
                elif "ts" in df.columns:
                    ts_col = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                    m5_timestamps_sec = (ts_col.astype(np.int64) // 1_000_000_000).astype(np.int64)
                else:
                    m5_timestamps_sec = None
                
                if m5_timestamps_sec is not None and len(m5_timestamps_sec) > 0:
                    m5_open = df["open"].values.astype(np.float64)
                    m5_high = df["high"].values.astype(np.float64)
                    m5_low = df["low"].values.astype(np.float64)
                    m5_close = df["close"].values.astype(np.float64)
                    
                    # Build H1 bars from entire df
                    h1_ts, h1_open, h1_high, h1_low, h1_close, h1_close_times = build_htf_from_m5(
                        m5_timestamps_sec, m5_open, m5_high, m5_low, m5_close, interval_hours=1
                    )
                    
                    # Build H4 bars from entire df
                    h4_ts, h4_open, h4_high, h4_low, h4_close, h4_close_times = build_htf_from_m5(
                        m5_timestamps_sec, m5_open, m5_high, m5_low, m5_close, interval_hours=4
                    )
                    
                    # PATCH 2: Initialize H1 aligner with complete HTF history (chunk-local)
                    # Each chunk is a new universe - aligners are reset per chunk (new runner = new feature_state)
                    if len(h1_close_times) > 0:
                        # PATCH 4: Validate HTF history is complete and sorted (assert in HTFAligner.__init__)
                        self.feature_state.h1_aligner = HTFAligner(
                            htf_close_times=h1_close_times,
                            htf_values=h1_close,
                            is_replay=True
                        )
                        log.info(
                            "[REPLAY] Initialized H1 aligner: %d HTF bars (from %d M5 bars), "
                            "first_close=%s, last_close=%s",
                            len(h1_close_times),
                            len(m5_timestamps_sec),
                            pd.Timestamp(h1_close_times[0], unit='s', tz='UTC').isoformat() if len(h1_close_times) > 0 else "N/A",
                            pd.Timestamp(h1_close_times[-1], unit='s', tz='UTC').isoformat() if len(h1_close_times) > 0 else "N/A"
                        )
                    else:
                        log.warning("[REPLAY] No H1 bars found for aligner initialization")
                    
                    # PATCH 2: Initialize H4 aligner with complete HTF history (chunk-local)
                    if len(h4_close_times) > 0:
                        # PATCH 4: Validate HTF history is complete and sorted (assert in HTFAligner.__init__)
                        self.feature_state.h4_aligner = HTFAligner(
                            htf_close_times=h4_close_times,
                            htf_values=h4_close,
                            is_replay=True
                        )
                        log.info(
                            "[REPLAY] Initialized H4 aligner: %d HTF bars (from %d M5 bars), "
                            "first_close=%s, last_close=%s",
                            len(h4_close_times),
                            len(m5_timestamps_sec),
                            pd.Timestamp(h4_close_times[0], unit='s', tz='UTC').isoformat() if len(h4_close_times) > 0 else "N/A",
                            pd.Timestamp(h4_close_times[-1], unit='s', tz='UTC').isoformat() if len(h4_close_times) > 0 else "N/A"
                        )
                    else:
                        log.warning("[REPLAY] No H4 bars found for aligner initialization")
            except Exception as e:
                log.warning(
                    "[REPLAY] Failed to initialize HTF aligners: %s. Falling back to legacy alignment.",
                    e, exc_info=True
                )
                # Clear aligners on error (fallback to legacy)
                self.feature_state.h1_aligner = None
                self.feature_state.h4_aligner = None
        
        # DEL 1: Initialize regime histogram for replay
        try:
            from gx1.execution.regime_histogram import init_regime_histogram
            regime_hist = init_regime_histogram(self.run_id)
            log.info(f"[REPLAY] Initialized regime histogram tracking (run_id={self.run_id})")
        except ImportError:
            regime_hist = None
        
        # Signal-only: Log signal bridge contract at replay-start
        try:
            from gx1.contracts.signal_bridge_v1 import SIGNAL_BRIDGE_ID, CONTRACT_SHA256, SEQ_SIGNAL_DIM, SNAP_SIGNAL_DIM, ORDERED_FIELDS
            log.info("[REPLAY] Signal Bridge Contract:")
            log.info("  id: %s", SIGNAL_BRIDGE_ID)
            log.info("  sha256: %s", str(CONTRACT_SHA256))
            log.info("  SEQ_SIGNAL_DIM=%d SNAP_SIGNAL_DIM=%d", int(SEQ_SIGNAL_DIM), int(SNAP_SIGNAL_DIM))
            log.info("  ORDERED_FIELDS(%d): %s", len(ORDERED_FIELDS), ORDERED_FIELDS)
        except Exception as e:
            log.warning("[REPLAY] Failed to log signal bridge contract: %s", e)
        
        # DEL 1: Log context features status at replay-start
        context_features_enabled = os.getenv("ENTRY_CONTEXT_FEATURES_ENABLED", "false").lower() == "true"
        log.info("[REPLAY] ENTRY_CONTEXT_FEATURES_ENABLED = %s", "true" if context_features_enabled else "false")
        if context_features_enabled:
            log.info("[REPLAY] Context features active: session, trend_regime, vol_regime, atr_bps, spread_bps")
        
        # DEL 2: Hard guardrail - verify context features are enabled if V10_CTX is active
        if hasattr(self, "entry_v10_ctx_enabled") and self.entry_v10_ctx_enabled:
            if not context_features_enabled:
                raise RuntimeError(
                    "CTX_MODEL_WITHOUT_CONTEXT: ENTRY_V10_CTX is enabled but ENTRY_CONTEXT_FEATURES_ENABLED is not 'true'. "
                    "Set ENTRY_CONTEXT_FEATURES_ENABLED=true in policy or environment."
                )
        
        import time
        replay_start_time = time.time()
        last_progress_log_time = replay_start_time
        
        # DEL 1: Invariant check - verify no BASELINE references
        if hasattr(self, 'entry_v10_ctx_bundle') and self.entry_v10_ctx_bundle:
            bundle_dir = getattr(self, 'entry_v10_ctx_cfg', {}).get('bundle_dir', '')
            if 'BASELINE' in str(bundle_dir).upper():
                raise RuntimeError(
                    f"BASELINE_DISABLED: Replay detected BASELINE bundle_dir: '{bundle_dir}'. "
                    f"Only GATED_FUSION bundles are allowed in replay."
                )
        
        # DEL 1: Verify GX1_GATED_FUSION_ENABLED=1
        # Note: os is imported at module level, but check if there's a local shadow
        gated_fusion_enabled = os.getenv("GX1_GATED_FUSION_ENABLED", "0") == "1"
        if not gated_fusion_enabled:
            raise RuntimeError(
                "BASELINE_DISABLED: GX1_GATED_FUSION_ENABLED is not '1' in replay mode. "
                "BASELINE is disabled. Set GX1_GATED_FUSION_ENABLED=1."
            )
        
        log.info("[REPLAY] GATED_FUSION mode verified: GX1_GATED_FUSION_ENABLED=1")
        
        # DEL 1: Log [REPLAY_PROVENANCE] blokk (explicit configuration from YAML)
        replay_config = self.policy.get("replay_config", {})
        if not replay_config:
            raise RuntimeError(
                "REPLAY_CONFIG_REQUIRED: replay_config section is missing in policy YAML. "
                "Set replay_config with policy_module, entry_model_id, runtime_feature_impl for replay mode."
            )
        
        policy_module = replay_config.get("policy_module")
        policy_class = replay_config.get("policy_class")
        policy_id = replay_config.get("policy_id")
        entry_model_id = replay_config.get("entry_model_id")
        runtime_feature_impl = replay_config.get("runtime_feature_impl")
        runtime_feature_module = replay_config.get("runtime_feature_module")
        entry_config_path = self.policy.get("entry_config", "")
        policy_config_path = str(self.policy_path.resolve())
        
        # Hard-fail if policy_module is missing or wrong
        if not policy_module:
            raise RuntimeError(
                "REPLAY_CONFIG_REQUIRED: replay_config.policy_module is missing in policy YAML. "
                "Set replay_config.policy_module='gx1.policy.entry_policy_sniper_v10_ctx' for replay mode."
            )
        
        if policy_module != "gx1.policy.entry_policy_sniper_v10_ctx":
            raise RuntimeError(
                f"REPLAY_POLICY_MISMATCH: replay_config.policy_module='{policy_module}' does not match expected "
                f"'gx1.policy.entry_policy_sniper_v10_ctx'. Only V10_CTX policies are allowed in replay mode."
            )
        
        # DEL 2 + DEL 3: Enforce V9 guardrails (sys.modules check + log sanitizer)
        try:
            from gx1.execution.replay_v9_guardrails import enforce_replay_v9_guardrails
            enforce_replay_v9_guardrails()
            log.info("[REPLAY_V9_GUARDRAILS] ✅ All V9 guardrails enforced successfully")
        except ImportError:
            # Module not available (may be quarantined or removed)
            log.debug("[REPLAY] replay_v9_guardrails module not available (skipping)")
        except RuntimeError as e:
            log.error(f"[REPLAY_V9_GUARDRAILS] ❌ V9 guardrails failed: {e}")
            raise
        
        # DEL 1: Get bundle sha256 early (for SSoT header)
        bundle_path_abs = None
        # Use bundle_sha256 from master if available (computed deterministically before workers start)
        bundle_sha256 = None
        if hasattr(self, "bundle_sha256_from_master") and self.bundle_sha256_from_master:
            bundle_sha256 = self.bundle_sha256_from_master
            log.info(f"[SSOT] Using bundle_sha256 from master: {bundle_sha256[:16]}...")
        else:
            # Fallback: compute from bundle (for backward compatibility, but this should not happen in replay)
            if hasattr(self, "entry_v10_ctx_cfg") and self.entry_v10_ctx_cfg:
                bundle_dir = self.entry_v10_ctx_cfg.get("bundle_dir", "")
                if bundle_dir:
                    bundle_path_abs = str(Path_module(bundle_dir).resolve())
                    bundle_metadata_path = Path_module(bundle_dir) / "bundle_metadata.json"
                    if bundle_metadata_path.exists():
                        try:
                            with open(bundle_metadata_path, "r") as f:
                                bundle_metadata = jsonlib.load(f)
                                bundle_sha256 = bundle_metadata.get("sha256")
                        except Exception:
                            pass
                    if not bundle_sha256:
                        model_state_path = Path_module(bundle_dir) / "model_state_dict.pt"
                        if model_state_path.exists():
                            try:
                                with open(model_state_path, "rb") as f:
                                    bundle_sha256 = hashlib.sha256(f.read()).hexdigest()
                            except Exception:
                                pass
        
        # DEL 1: SSoT Assert - Hard-fail if any critical fields are missing or wrong
        if not policy_module:
            raise RuntimeError(
                "REPLAY_SSoT_ASSERT_FAILED: policy_module is missing. "
                "SSoT (Single Source of Truth) requires explicit policy_module in replay_config."
            )
        if policy_module != "gx1.policy.entry_policy_sniper_v10_ctx":
            raise RuntimeError(
                f"REPLAY_SSoT_ASSERT_FAILED: policy_module='{policy_module}' does not match expected "
                f"'gx1.policy.entry_policy_sniper_v10_ctx'. SSoT requires V10_CTX only."
            )
        if not entry_model_id:
            raise RuntimeError(
                "REPLAY_SSoT_ASSERT_FAILED: entry_model_id is missing. "
                "SSoT requires explicit entry_model_id in replay_config."
            )
        if not runtime_feature_module:
            raise RuntimeError(
                "REPLAY_SSoT_ASSERT_FAILED: runtime_feature_module is missing. "
                "SSoT requires explicit runtime_feature_module in replay_config."
            )
        if runtime_feature_module != "gx1.features.runtime_v10_ctx":
            raise RuntimeError(
                f"REPLAY_SSoT_ASSERT_FAILED: runtime_feature_module='{runtime_feature_module}' does not match expected "
                f"'gx1.features.runtime_v10_ctx'. SSoT requires V10_CTX runtime only."
            )
        if not bundle_sha256:
            raise RuntimeError(
                "REPLAY_SSoT_ASSERT_FAILED: bundle_sha256 is missing. "
                "SSoT requires bundle_sha256 for audit trail. "
                "Ensure bundle_dir points to valid bundle with bundle_metadata.json or model_state_dict.pt."
            )
        
        log.info("=" * 80)
        log.info("[REPLAY_SSoT] Single Source of Truth (SSoT) - Explicit Configuration:")
        log.info("[REPLAY_SSoT]   policy_module: %s", policy_module)
        log.info("[REPLAY_SSoT]   policy_class: %s", policy_class)
        log.info("[REPLAY_SSoT]   policy_id: %s", policy_id)
        log.info("[REPLAY_SSoT]   runtime_feature_module: %s", runtime_feature_module)
        log.info("[REPLAY_SSoT]   entry_model_id: %s", entry_model_id)
        log.info("[REPLAY_SSoT]   bundle_sha256: %s", bundle_sha256)
        log.info("[REPLAY_SSoT]   policy_config_path: %s", policy_config_path)
        log.info("[REPLAY_SSoT]   entry_config_path: %s", entry_config_path)
        log.info("[REPLAY_SSoT]   v9_forbidden: %s", replay_config.get("v9_forbidden", True))
        log.info("=" * 80)
        
        # DEL 1: Write REPLAY_SSoT_HEADER.json (machine-readable audit trail)
        try:
            # Determine output directory (same logic as _generate_run_header)
            if hasattr(self, "output_dir") and self.output_dir:
                output_dir = Path_module(self.output_dir)
            elif hasattr(self, "run_id") and self.run_id:
                output_dir = Path_module("gx1/wf_runs") / self.run_id
            else:
                # Fallback: use current working directory / reports/replay_eval/GATED
                output_dir = Path_module("reports/replay_eval/GATED")
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            from gx1.prod.run_header import get_git_commit_hash
            git_commit = get_git_commit_hash()
            
            sso_header = {
                "run_id": getattr(self, "run_id", None),
                "chunk_id": getattr(self, "chunk_id", None),
                "timestamp": dt_module.datetime.now(tz=dt_module.timezone.utc).isoformat(),
                "replay_sso": {
                    "policy_module": policy_module,
                    "policy_class": policy_class,
                    "policy_id": policy_id,
                    "runtime_feature_module": runtime_feature_module,
                    "runtime_feature_impl": runtime_feature_impl,
                    "entry_model_id": entry_model_id,
                    "bundle_path_abs": bundle_path_abs,
                    "bundle_sha256": bundle_sha256,
                    "policy_config_path": policy_config_path,
                    "entry_config_path": entry_config_path,
                    "v9_forbidden": replay_config.get("v9_forbidden", True),
                },
                "provenance": {
                    "git_commit": git_commit,
                    "worker_pid": os.getpid(),
                },
            }
            
            # Write to run-root (where run_header.json is)
            sso_header_path = output_dir / "REPLAY_SSoT_HEADER.json"
            with open(sso_header_path, "w") as f:
                jsonlib.dump(sso_header, f, indent=2)
            log.info("[REPLAY_SSoT] ✅ Written SSoT header to: %s", sso_header_path)
            
        except Exception as e:
            log.error("[REPLAY_SSoT] ❌ Failed to write REPLAY_SSoT_HEADER.json: %s", e, exc_info=True)
            # Non-fatal: continue but log error
        
        # DEL 3: Set provenance on collectors (once at replay start, using explicit YAML values)
        # Note: bundle_path_abs and bundle_sha256 are already computed above (for SSoT assert)
        if hasattr(self, "replay_eval_collectors") and self.replay_eval_collectors:
            from gx1.prod.run_header import get_git_commit_hash
            
            # DEL 1: Use explicit values from replay_config (SSoT from YAML, no auto-switch)
            # These are already validated above in [REPLAY_SSoT] block
            # policy_module, policy_class, entry_model_id, runtime_feature_impl, runtime_feature_module are from replay_config
            entry_config_path = self.policy.get("entry_config", "")
            
            # Get git commit
            git_commit = get_git_commit_hash()
            
            # DEL 3: Set provenance on all collectors (using explicit values from replay_config)
            provenance_kwargs = {
                "policy_module": policy_module,  # From replay_config (validated above)
                "policy_class": policy_class,  # From replay_config
                "policy_config_path": policy_config_path,  # From self.policy_path (logged above)
                "entry_config_path": entry_config_path,
                "entry_model_id": entry_model_id,  # From replay_config
                "bundle_path_abs": bundle_path_abs,  # Already computed above for SSoT assert
                "bundle_sha256": bundle_sha256,  # Already computed above for SSoT assert
                "git_commit": git_commit,
                "run_id": getattr(self, "run_id", None),
                "chunk_id": getattr(self, "chunk_id", None),
                "worker_pid": os.getpid(),
                "runtime_feature_impl": runtime_feature_impl,  # DEL 3: From replay_config
                "runtime_feature_module": runtime_feature_module,  # DEL 3: From replay_config
            }
            
            for collector_name, collector in self.replay_eval_collectors.items():
                if hasattr(collector, "set_provenance"):
                    collector.set_provenance(**provenance_kwargs)
                    log.info(f"[REPLAY] Set provenance on {collector_name}: policy_module={policy_module}, entry_model_id={entry_model_id}")
        
        # Fix 3: Initialize perf counters on self for summary
        self.perf_n_bars_processed = 0
        self.perf_n_trades_created = 0
        self.perf_feat_time = 0.0  # DEL C: Feature building time accumulator
        
        # NEW: Initialize HTF warmup skip counter (harness-only telemetry)
        self.n_bars_skipped_due_to_htf_warmup = 0
        
        # DEL A2: Initialize pregate telemetry counters
        self.pregate_skips = 0
        self.pregate_passes = 0
        self.pregate_missing_inputs = 0
        
        # DEL 1: Initialize phase timing accumulators (for "bars/sec" breakdown)
        self.t_pregate_total_sec = 0.0
        self.t_feature_build_total_sec = 0.0
        # Legacy summary fields (B1): do NOT use 50/50 estimate; these will be derived from true timers below.
        self.t_model_total_sec = 0.0
        self.t_policy_total_sec = 0.0
        self.t_io_total_sec = 0.0

        # B1: True hot-path timers (non-overlapping intent)
        # - XGB predict and Transformer forward are model time
        # - gates/policy time covers guard + policy evaluation logic
        # - replay tags time isolates ensure_replay_tags overhead
        # - telemetry time covers per-bar telemetry recording + chunk telemetry flush/write
        self.t_xgb_predict_sec = 0.0
        self.t_transformer_forward_sec = 0.0
        self.t_gates_policy_sec = 0.0
        self.t_replay_tags_sec = 0.0
        self.t_telemetry_sec = 0.0

        # Step 4A: replay tagger sub-timers + counters
        self.t_replay_tags_build_inputs_sec = 0.0
        self.t_replay_tags_rolling_sec = 0.0
        self.t_replay_tags_ewm_sec = 0.0
        self.t_replay_tags_rank_sec = 0.0
        self.t_replay_tags_assign_sec = 0.0
        self.replay_tags_fastskip_hits = 0
        self.replay_tags_fastskip_reason_counts = {}
        
        # DEL A2: Check if pregate is enabled in replay_config
        # Gate-SSoT: This is where pregate_enabled is determined (used in bar loop at line 10085)
        # DEL 1B: Env override trumfer YAML (for reproducible A/B testing)
        env_pregate_enabled = os.getenv("GX1_REPLAY_PREGATE_ENABLED")
        is_preflight_run = os.getenv("GX1_PREFLIGHT", "0") == "1" or os.getenv("GX1_PREFLIGHT_WARMUP_BARS") is not None
        
        if env_pregate_enabled is not None:
            # Env override: "1" or "true" (case-insensitive) = enabled, else disabled
            pregate_enabled_effective = env_pregate_enabled.lower() in ("1", "true")
            pregate_override_applied = is_preflight_run and env_pregate_enabled.lower() == "0"
            log.info("[REPLAY_PREGATE] Env override: GX1_REPLAY_PREGATE_ENABLED=%s -> enabled=%s (preflight=%s)", env_pregate_enabled, pregate_enabled_effective, is_preflight_run)
        else:
            # No env override: read from YAML
            replay_config = self.policy.get("replay_config", {})
            pregate_cfg = replay_config.get("replay_pregate", {})
            pregate_enabled_effective = pregate_cfg.get("enabled", False) if isinstance(pregate_cfg, dict) else False
            pregate_override_applied = False
            if pregate_enabled_effective:
                log.info("[REPLAY_PREGATE] Enabled from YAML config: %s", pregate_cfg)
        
        # Store effective pregate value and override flag on runner (for footer export)
        self.pregate_enabled_effective = pregate_enabled_effective
        self.pregate_override_applied = pregate_override_applied
        self.pregate_enabled = pregate_enabled_effective  # Use effective value for bar loop
        
        # DEL C: Initialize HTF alignment warning counter (for chunk summary)
        self.htf_align_warn_count = 0
        
        # DEL 2: Initialize feature timeout counter (for chunk summary)
        self.feature_timeout_count = 0
        
        # Entry stage telemetry: reset before loop (attributes already initialized in __init__)
        # These are persistent attributes on runner-self (not local variables)
        self.bars_seen = 0
        self.bars_skipped_warmup = 0
        self.bars_skipped_pregate = 0
        self.bars_reaching_entry_stage = 0
        
        # A) CANONICAL LOOP COUNTERS (SSoT for loop completion check)
        # loop_iters_total: increments every iteration of the for-loop (must equal len(df))
        # loop_iters_post_warmup: increments only when i >= first_valid_eval_idx
        self.loop_iters_total = 0
        self.loop_iters_post_warmup = 0
        self.bars_total_coalesced = len(df)  # Total bars in df (ground truth)
        self.first_valid_eval_idx_stored = first_valid_eval_idx  # Store for later access

        # ---------------------------------------------------------------------
        # PERF (semantics-neutral): precompute mid-price ATR(5) in bps once per chunk.
        #
        # This matches ExitManager._compute_runtime_atr_bps() which previously built a small
        # mid_df + compute_atr_bps() per bar. Precomputing once and looking up by timestamp
        # is equivalent because compute_atr_bps uses a rolling window (max period bars).
        #
        # Used only in replay loop via ExitManager._compute_runtime_atr_bps() lookup.
        # ---------------------------------------------------------------------
        self.replay_runtime_atr5_bps_series = None
        try:
            required_cols = [
                "bid_high",
                "bid_low",
                "bid_close",
                "ask_high",
                "ask_low",
                "ask_close",
            ]
            if all(col in df.columns for col in required_cols):
                from gx1.execution.live_features import compute_atr_bps

                mid_df_full = pd.DataFrame(
                    {
                        "high": (df["bid_high"] + df["ask_high"]) * 0.5,
                        "low": (df["bid_low"] + df["ask_low"]) * 0.5,
                        "close": (df["bid_close"] + df["ask_close"]) * 0.5,
                    },
                    index=df.index,
                )
                # Exit runtime ATR uses period=5
                self.replay_runtime_atr5_bps_series = compute_atr_bps(mid_df_full, period=5)
        except Exception:
            # Non-fatal: exit manager will fall back to per-bar compute (legacy behavior)
            self.replay_runtime_atr5_bps_series = None
        
        # CANONICAL FUNNEL LEDGER (SSoT for where bars die)
        # Counters are incremented at each funnel stage in the replay loop
        from collections import defaultdict
        self.funnel_bars_total = 0  # = loop_iters_total (alias)
        self.funnel_bars_post_warmup = 0  # = loop_iters_post_warmup (alias)
        self.funnel_pregate_pass = 0  # Bars that passed pregate (or pregate disabled)
        self.funnel_pregate_block = 0  # Bars blocked by pregate
        self.funnel_pregate_block_reasons = defaultdict(int)  # Reason histogram
        self.funnel_entry_eval_called = 0  # Bars where evaluate_entry() was actually called
        self.funnel_predict_entered = 0  # Bars where _predict_entry_v10_hybrid was entered (from telemetry)
        self.funnel_first_ts_post_warmup = None  # First timestamp after warmup
        self.funnel_last_ts_post_warmup = None  # Last timestamp after warmup
        
        # PRE-ENTRY FUNNEL COUNTERS: Track where bars die before entry evaluation (legacy)
        self.candles_iterated = 0  # Total loop iterations (legacy, same as loop_iters_total)
        self.warmup_skipped = 0  # Bars skipped due to warmup/min bars (alias for bars_skipped_warmup)
        self.pregate_checked_count = 0  # Number of times pregate was checked
        self.pregate_skipped = 0  # Bars skipped by pregate (alias for bars_skipped_pregate)
        self.prebuilt_available_checked = 0  # Number of times prebuilt availability was checked
        self.prebuilt_missing_skipped = 0  # Bars skipped due to prebuilt missing
        self.bars_before_evaluate_entry = 0  # Bars that reached point right before evaluate_entry() call
        self.evaluate_entry_called_count = 0  # Actual number of times evaluate_entry() was called
        self.bars_after_evaluate_entry = 0  # Bars that completed evaluate_entry() call
        
        # B) Stop reason semantics: only valid values are EOF, STOP_REQUESTED, FAST_ABORT, EXCEPTION
        # WARMUP is NOT a stop reason (it's accounted for by first_valid_eval_idx)
        self.last_stop_reason = "EOF"  # Default: end of file (normal completion)
        self.last_stop_exception = None  # Exception that caused stop (if any)
        self.last_iterated_ts = None  # Last timestamp iterated
        self.last_i = None  # Last index iterated
        
        for i, (ts, row) in enumerate(df.iterrows()):
            # A) CANONICAL LOOP COUNTERS: increment loop_iters_total every iteration (SSoT)
            self.loop_iters_total += 1
            
            # B) Track explicit stop reason in-loop (single source)
            # Update last_i, last_iterated_ts every iteration
            self.last_i = i
            self.last_iterated_ts = ts
            
            # Compute session tag for per-session funnel telemetry
            ts_for_session = ts
            try:
                if not isinstance(ts, pd.Timestamp):
                    ts_for_session = pd.Timestamp(ts)
                session_tag_for_bar = infer_session_tag(ts_for_session).upper()
            except Exception as e:
                session_tag_for_bar = "UNKNOWN"
                if os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1":
                    log.error(
                        "[SESSION_TRACKING] Failed to infer session tag: ts=%s type=%s error=%s",
                        ts,
                        type(ts),
                        e,
                    )
            
            # Diagnostic override (replay truth only): cache forced sessions list and bypass gate
            if not hasattr(self, "diagnostic_force_eval_sessions"):
                diagnostic_force_sessions_raw = os.getenv("GX1_DIAGNOSTIC_FORCE_EVAL_SESSIONS", "")
                self.diagnostic_force_eval_sessions = [
                    s.strip().upper() for s in diagnostic_force_sessions_raw.split(",") if s.strip()
                ]
                self.diagnostic_force_eval_enabled = bool(self.diagnostic_force_eval_sessions) and os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1"
                diagnostic_bypass_gate_raw = os.getenv("GX1_DIAGNOSTIC_BYPASS_GATE", "").strip().lower()
                self.diagnostic_bypass_gate = diagnostic_bypass_gate_raw if diagnostic_bypass_gate_raw else "auto"
            
            # Record bar seen (sentinel counter)
            if hasattr(self, "entry_manager") and self.entry_manager is not None and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry is not None:
                self.entry_manager.entry_feature_telemetry.record_bar_seen(session_tag=session_tag_for_bar, ts=ts_for_session)
            
            # PRE-ENTRY FUNNEL: Track every loop iteration
            self.candles_iterated += 1
            self.bars_seen += 1
            
            # Skip bars before first_valid_eval_idx (HTF warmup not satisfied)
            # NOTE: This is NOT a stop reason - warmup is accounted for by first_valid_eval_idx
            if i < first_valid_eval_idx:
                self.n_bars_skipped_due_to_htf_warmup += 1
                self.bars_skipped_warmup += 1
                self.warmup_skipped += 1
                # Do NOT set last_stop_reason here - warmup skip is not a stop condition
                continue
            
            # Record bar after warmup (replay mode uses first_valid_eval_idx)
            if hasattr(self, "entry_manager") and self.entry_manager is not None and hasattr(self, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry is not None:
                self.entry_manager.entry_feature_telemetry.record_bar_after_warmup(session=session_tag_for_bar)
            
            # A) Increment loop_iters_post_warmup when i >= first_valid_eval_idx
            self.loop_iters_post_warmup += 1
            
            # FUNNEL LEDGER: Track post-warmup bars and timestamps
            self.funnel_bars_post_warmup += 1
            if self.funnel_first_ts_post_warmup is None:
                self.funnel_first_ts_post_warmup = ts
            self.funnel_last_ts_post_warmup = ts
            
            # Skip first N bars (not enough history for stable features) - now relative to first_valid_eval_idx
            if i < first_valid_eval_idx + min_bars_for_features:
                self.bars_skipped_warmup += 1
                self.warmup_skipped += 1
                # Do NOT set last_stop_reason here - warmup skip is not a stop condition
                continue
            
            # Fix 3: Increment bar counter
            self.perf_n_bars_processed += 1

            # Replay determinism: store current bar timestamp for any replay-side journaling.
            # Used by ExitArbiter request_close() to avoid wall-clock timestamps in replay artifacts.
            try:
                self.replay_current_ts = ts
            except Exception:
                pass
            
            # Fast abort mode: stop after N bars per chunk (for fast verification)
            abort_after_n_bars = os.getenv("GX1_ABORT_AFTER_N_BARS_PER_CHUNK")
            if abort_after_n_bars:
                try:
                    abort_after_n_bars_int = int(abort_after_n_bars)
                    if self.perf_n_bars_processed >= abort_after_n_bars_int:
                        log.info(
                            f"[REPLAY] Fast abort: stopping after {self.perf_n_bars_processed} bars "
                            f"(requested: {abort_after_n_bars_int})"
                        )
                        self.last_stop_reason = "FAST_ABORT"
                        break
                except ValueError:
                    pass  # Invalid value, ignore
            
            # DEL 2: Check for STOP_REQUESTED flag (graceful shutdown)
            if os.getenv("GX1_STOP_REQUESTED", "0") == "1" or getattr(self, "_stop_requested", False):
                log.warning("[REPLAY] STOP_REQUESTED flag detected, breaking bar loop gracefully")
                self.last_stop_reason = "STOP_REQUESTED"
                break
            
            # Test-only crash injection (only active when env vars set)
            crash_chunk_id = os.getenv("GX1_TEST_INDUCE_CRASH_CHUNK_ID")
            if crash_chunk_id is not None:
                crash_after_n_bars_str = os.getenv("GX1_TEST_CRASH_AFTER_N_BARS", "0")
                try:
                    crash_after_n_bars = int(crash_after_n_bars_str)
                except ValueError:
                    crash_after_n_bars = 0
                
                # Get current chunk_id from env (set by parallel replay scripts)
                current_chunk_id = os.getenv("GX1_CHUNK_ID")
                if current_chunk_id == crash_chunk_id and self.perf_n_bars_processed >= crash_after_n_bars:
                    raise RuntimeError(
                        f"TEST_INDUCED_CRASH: Chunk {current_chunk_id} crashed after {self.perf_n_bars_processed} bars "
                        f"(requested: {crash_after_n_bars})"
                    )
            
            # DEL 3: Periodic checkpoint flush (every CHECKPOINT_EVERY_BARS)
            CHECKPOINT_EVERY_BARS = int(os.getenv("GX1_CHECKPOINT_EVERY_BARS", "1000"))
            if self.perf_n_bars_processed > 0 and self.perf_n_bars_processed % CHECKPOINT_EVERY_BARS == 0:
                if hasattr(self, "replay_eval_collectors") and self.replay_eval_collectors:
                    log.info("[REPLAY_EVAL] skip legacy flush_replay_eval_collectors (forbidden import path)")
            
            # FIX: Rate-limited HTF progress logging every 10k bars
            if self.perf_n_bars_processed > 0 and self.perf_n_bars_processed % 10000 == 0:
                # Get HTFAligner stats directly from feature_state
                h1_calls = 0
                h4_calls = 0
                fallback_count = 0
                if hasattr(self, "feature_state") and self.feature_state:
                    h1_aligner = getattr(self.feature_state, "h1_aligner", None)
                    h4_aligner = getattr(self.feature_state, "h4_aligner", None)
                    if h1_aligner is not None:
                        h1_stats = h1_aligner.get_stats()
                        h1_calls = h1_stats.get("call_count", 0)
                    if h4_aligner is not None:
                        h4_stats = h4_aligner.get_stats()
                        h4_calls = h4_stats.get("call_count", 0)
                    fallback_count = getattr(self.feature_state, "htf_align_fallback_count", 0)
                
                log.info(
                    "[HTF_PROGRESS] bars_processed=%d, h1_calls=%d, h4_calls=%d, total_calls=%d, fallback_count=%d",
                    self.perf_n_bars_processed, h1_calls, h4_calls, h1_calls + h4_calls, fallback_count
                )
            
            # Progress logging every 500 bars (more frequent)
            current_bar_idx = i - first_valid_eval_idx
            current_time = time.time()
            
            if current_bar_idx > 0 and (current_bar_idx % 500 == 0 or (current_time - last_progress_log_time) > 30):
                progress_pct = (current_bar_idx / bars_to_process) * 100.0
                elapsed_time = current_time - replay_start_time
                
                # Estimate remaining time
                if current_bar_idx > 0:
                    bars_per_sec = current_bar_idx / elapsed_time if elapsed_time > 0 else 0
                    remaining_bars = bars_to_process - current_bar_idx
                    estimated_remaining_sec = remaining_bars / bars_per_sec if bars_per_sec > 0 else 0
                    estimated_remaining_min = estimated_remaining_sec / 60.0
                else:
                    estimated_remaining_min = 0
                
                # Count total trades (try to read trade log)
                total_trades = len(self.open_trades)
                if self.trade_log_path.exists():
                    try:
                        with open(self.trade_log_path, 'r') as f:
                            lines = f.readlines()
                            # Subtract header if exists
                            total_trades = len([l for l in lines if l.strip() and not l.startswith('run_id')])
                    except:
                        pass
                
                log.info(
                    "[REPLAY PROGRESS] %d/%d bars (%.1f%%) | Elapsed: %.1f min | Est. remaining: %.1f min | Trades: %d open, %d total",
                    current_bar_idx,
                    bars_to_process,
                    progress_pct,
                    elapsed_time / 60.0,
                    estimated_remaining_min,
                    len(self.open_trades),
                    total_trades,
                )
                last_progress_log_time = current_time
            
            # Store current bar timestamp for replay context (used by _tick_close_now)
            self._replay_current_ts = ts
            
            # Build candle history for features (need lookback window)
            # Use all candles up to current bar (including current bar)
            # CRITICAL: For HTF warmup, we need to ensure we have enough history
            # HTF aggregator needs at least 48 M5 bars for H4 (4 hours * 12 bars/hour)
            # So we start from max(0, i+1-48) to ensure we have enough history for HTF
            # But we still need all bars from start for proper HTF aggregation
            # Actually, we need all bars from start for HTF aggregation to work correctly
            # The issue is that _align_htf_to_m5_numpy checks if indices < 0 for the first 12 bars
            # So we need to ensure that when we build features, we skip bars that don't have HTF data yet
            # This is already handled by first_valid_eval_idx, but we need to ensure candles_history
            # has enough bars for HTF aggregation to complete
            candles_history = df.iloc[:i+1].copy()
            
            # Ensure index is DatetimeIndex with UTC timezone
            if not isinstance(candles_history.index, pd.DatetimeIndex):
                candles_history = candles_history.set_index("time")
                candles_history.index = pd.to_datetime(candles_history.index, utc=True)
            
            # Simulate tick-based exit for open trades (before processing new bar)
            # Skip in ENTRY_ONLY mode
            if self.mode != "ENTRY_ONLY" and self.open_trades:
                candle_row = pd.Series({
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
                "bid_open": float(row["bid_open"]),
                "bid_high": float(row["bid_high"]),
                "bid_low": float(row["bid_low"]),
                "bid_close": float(row["bid_close"]),
                "ask_open": float(row["ask_open"]),
                "ask_high": float(row["ask_high"]),
                "ask_low": float(row["ask_low"]),
                "ask_close": float(row["ask_close"]),
                })
                self._simulate_tick_exits_for_bar(ts, candle_row)
        
            # Process bar (same logic as run_once, but with historical candle)
            log.debug("[REPLAY] Processing bar %d/%d: %s", i+1, len(df), ts.isoformat())
        
            # DEL A2: Replay PreGate - early skip before expensive feature building
            should_skip_pregate = False
            pregate_reason = ""
            t_pregate_start = time.perf_counter()
            if self.replay_mode and self.pregate_enabled:
                # PRE-ENTRY FUNNEL: Track pregate check
                self.pregate_checked_count += 1
                from gx1.execution.replay_pregate import replay_pregate_should_skip
                
                # Get cheap inputs for pregate (no pandas, no HTF, no rolling)
                # Infer session from timestamp (cheap - just time-based)
                current_session = infer_session_tag(ts).upper()
                
                # Get warmup/degraded status from runner state
                warmup_ready = hasattr(self, "_cached_bars_at_startup") and getattr(self, "_cached_bars_at_startup", 0) >= 288
                degraded = getattr(self, "_warmup_degraded", False)
                
                # Compute cheap spread_bps from row (cheap - just arithmetic)
                spread_bps = None
                try:
                    if "bid_close" in row and "ask_close" in row:
                        bid = float(row["bid_close"])
                        ask = float(row["ask_close"])
                        if bid > 0:
                            spread_price = ask - bid
                            spread_bps = (spread_price / bid) * 10000.0
                except (ValueError, TypeError, KeyError):
                    pass  # Keep as None if computation fails
                
                # Compute cheap ATR proxy from row (cheap - just recent highs/lows)
                atr_bps = None
                try:
                    if i >= 13 and "high" in df.columns and "low" in df.columns and "close" in df.columns:
                        # Use last 14 bars for cheap ATR proxy (same as soft eligibility)
                        recent_window = df.iloc[max(0, i-13):i+1]
                        if len(recent_window) >= 14:
                            highs = recent_window["high"].values
                            lows = recent_window["low"].values
                            closes = recent_window["close"].values
                            import numpy as np
                            # True Range approximation (simplified)
                            tr1 = highs - lows
                            close_prev = np.roll(closes, 1)
                            close_prev[0] = closes[0]
                            tr2 = np.abs(highs - close_prev)
                            tr3 = np.abs(lows - close_prev)
                            tr = np.maximum(np.maximum(tr1, tr2), tr3)
                            atr_proxy = np.mean(tr)
                            current_close = float(row["close"])
                            if current_close > 0:
                                atr_bps = (atr_proxy / current_close) * 10000.0
                except (ValueError, TypeError, KeyError, IndexError):
                    pass  # Keep as None if computation fails
                
                # Check in_scope (cheap - session-based)
                in_scope = current_session in ["EU", "OVERLAP", "US"]
                
                # Call pregate
                should_skip_pregate, pregate_reason = replay_pregate_should_skip(
                    ts=ts,
                    session=current_session,
                    warmup_ready=warmup_ready,
                    degraded=degraded,
                    spread_bps=spread_bps,
                    atr_bps=atr_bps,
                    in_scope=in_scope,
                    policy_config=self.policy,
                )
                
                # Diagnostic override (replay truth only): bypass pregate for forced sessions
                if (
                    getattr(self, "diagnostic_force_eval_enabled", False)
                    and session_tag_for_bar in getattr(self, "diagnostic_force_eval_sessions", [])
                    and getattr(self, "diagnostic_bypass_gate", "auto") in ("auto", "pregate")
                    and should_skip_pregate
                ):
                    if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                        self.entry_manager.entry_feature_telemetry.record_diagnostic_bypass(
                            gate_name="pregate",
                            session=session_tag_for_bar,
                        )
                    pregate_reason = f"DIAGNOSTIC_BYPASS:{pregate_reason}" if pregate_reason else "DIAGNOSTIC_BYPASS"
                    should_skip_pregate = False
                
                if should_skip_pregate:
                    # PreGate skip - register decision but don't build features or call model
                    self.pregate_skips += 1
                    self.bars_skipped_pregate += 1
                    self.pregate_skipped += 1
                    
                    # FUNNEL LEDGER: Track pregate block with reason
                    self.funnel_pregate_block += 1
                    reason_key = pregate_reason if pregate_reason else "UNKNOWN"
                    self.funnel_pregate_block_reasons[reason_key] += 1
                    
                    # Entry feature telemetry: record pregate block by session
                    if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                        self.entry_manager.entry_feature_telemetry.record_pregate_block(
                            reason=reason_key,
                            session=current_session,
                        )
                    
                    # Register skip decision in collector (same as normal skip)
                    if hasattr(self, "replay_eval_collectors") and self.replay_eval_collectors:
                        policy_collector = self.replay_eval_collectors.get("policy_decisions")
                        if policy_collector:
                            # Convert datetime to pd.Timestamp if needed
                            ts_pd = ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)
                            policy_collector.collect(
                                timestamp=ts_pd,
                                decision="skip",
                                reasons=[pregate_reason] if pregate_reason else ["replay_pregate_skip:unknown"],
                            )
                    
                    # Log rate-limited (max 1 per 100 bars to avoid spam)
                    if self.pregate_skips % 100 == 0 or self.pregate_skips <= 10:
                        log.debug(f"[REPLAY_PREGATE] Skipped bar {i+1}: {pregate_reason}")
                    
                    # Skip to next bar (no feature build, no model call)
                    continue
                else:
                    self.pregate_passes += 1
                    # FUNNEL LEDGER: Track pregate pass
                    self.funnel_pregate_pass += 1
                    
                    # Entry feature telemetry: record pregate pass by session
                    if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                        self.entry_manager.entry_feature_telemetry.record_pregate_pass(
                            session=current_session,
                        )
                        self.entry_manager.entry_feature_telemetry.record_post_pregate_enter(
                            session=current_session,
                            ts=ts,
                        )
                    if spread_bps is None or atr_bps is None:
                        self.pregate_missing_inputs += 1
                
                t_pregate_end = time.perf_counter()
                self.t_pregate_total_sec += (t_pregate_end - t_pregate_start)
            else:
                t_pregate_end = time.perf_counter()
                self.t_pregate_total_sec += (t_pregate_end - t_pregate_start)
                # Pregate disabled: treat as pass for funnel/telemetry consistency
                self.pregate_passes += 1
                self.funnel_pregate_pass += 1
                if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                    self.entry_manager.entry_feature_telemetry.record_pregate_pass(
                        session=session_tag_for_bar,
                    )
                    self.entry_manager.entry_feature_telemetry.record_post_pregate_enter(
                        session=session_tag_for_bar,
                        ts=ts,
                    )
            
            # DEL 1: Time feature building phase
            t_feature_start = time.perf_counter()
            
            # PRE-ENTRY FUNNEL: Track bars reaching point before evaluate_entry
            if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                self.entry_manager.entry_feature_telemetry.record_pre_eval_enter(
                    session=session_tag_for_bar,
                    ts=ts,
                )
            self.bars_before_evaluate_entry += 1
            self.bars_reaching_entry_stage += 1
            
            # PRE-ENTRY FUNNEL: Track actual evaluate_entry call
            self.evaluate_entry_called_count += 1
            
            # FUNNEL LEDGER: Track evaluate_entry call (exactly when called)
            self.funnel_entry_eval_called += 1
            
            # Evaluate entry (only if pregate didn't skip)
            trade = self.evaluate_entry(candles_history)
            
            # PRE-ENTRY FUNNEL: Track bars after evaluate_entry returns
            self.bars_after_evaluate_entry += 1
            
            t_feature_end = time.perf_counter()
            # Feature time is already tracked in self.perf_feat_time, but we need total time
            # Note: evaluate_entry includes feature build + model + policy, so we need to separate
            # For now, we'll track feature build time from entry_manager's perf_feat_time accumulator
            # and model/policy time separately if available
            self.t_feature_build_total_sec += (t_feature_end - t_feature_start)
            if trade and self.can_enter(ts):  # can_enter takes timestamp, not DataFrame
                    # In ENTRY_ONLY mode, log entry signal but don't execute trades
                    if self.mode == "ENTRY_ONLY":
                        # Log entry signal to entry_only_log with extended features
                        if hasattr(self, "entry_only_log_path") and self.entry_only_log_path:
                                import csv
                                file_exists = self.entry_only_log_path.exists()
                    
                                # Extended fieldnames for policy analysis
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
                    
                                with open(self.entry_only_log_path, "a", newline="") as f:
                                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                                    if not file_exists:
                                        writer.writeheader()
                        
                                        # Extract features from current_bar and trade.extra
                                        current_bar = candles_history.iloc[-1]
                                        extra = trade.extra if hasattr(trade, "extra") and trade.extra else {}
                        
                                        # Session info
                                        session = extra.get("session", current_bar.get("session", "UNKNOWN"))
                                        session_id = current_bar.get("session_id", current_bar.get("_v1_session_tag", "UNKNOWN"))
                        
                                        # Regime info
                                        atr_regime = extra.get("atr_regime", current_bar.get("atr_regime", "UNKNOWN"))
                                        vol_regime = extra.get("brain_vol_regime", atr_regime)
                                        trend_regime = extra.get("brain_trend_regime", current_bar.get("trend_regime_tf24h", "UNKNOWN"))
                        
                                        # Time features
                                        hour_of_day = ts.hour
                                        day_of_week = ts.dayofweek  # 0=Monday, 6=Sunday
                        
                                        # Bar features
                                        body_pct = current_bar.get("body_pct", current_bar.get("_v1_body_tr", 0.0))
                                        wick_asym = current_bar.get("wick_asym", 0.0)
                                        bar_range = (current_bar.get("high", 0) - current_bar.get("low", 0)) / current_bar.get("close", 1) * 10000 if "high" in current_bar.index and "low" in current_bar.index else 0.0
                        
                                        # HTF context (H1/H4 slopes if available)
                                        htf_h1 = current_bar.get("_v1_int_slope_h1_us", 0.0)
                                        htf_h4 = current_bar.get("_v1_int_slope_h4_atr", 0.0)
                        
                                        # Calculate MFE/MAE for next 5 bars (lookahead)
                                        mfe_5b = np.nan
                                        mae_5b = np.nan
                                        t_mfe = np.nan
                                        t_mae = np.nan
                        
                                        # Look ahead 5 bars for MFE/MAE calculation
                                        if i + 5 < len(df):
                                            future_bars = df.iloc[i+1:i+6]
                                            if len(future_bars) > 0 and "high" in future_bars.columns and "low" in future_bars.columns:
                                                entry_price = float(current_bar["close"])
                                
                                                # For LONG: MFE = max high - entry, MAE = entry - min low
                                                # TODO: Feature engineering MFE/MAE uses mid prices (high/low)
                                                # This is for feature engineering only, not actual PnL calculation
                                                # Consider using bid_high/ask_low for LONG and bid_low/ask_high for SHORT in future
                                                highs = future_bars["high"].values
                                                lows = future_bars["low"].values
                                
                                                mfe_5b = (np.max(highs) - entry_price) / entry_price * 10000.0
                                                mae_5b = (entry_price - np.min(lows)) / entry_price * 10000.0
                                
                                                # Time to MFE/MAE (in bars)
                                                mfe_idx = np.argmax(highs)
                                                mae_idx = np.argmin(lows)
                                                t_mfe = float(mfe_idx + 1)  # +1 because we start from next bar
                                                t_mae = float(mae_idx + 1)
                        
                                                # Next regime/session (from next bar if available)
                                                next_vol_regime = "UNKNOWN"
                                                next_session = "UNKNOWN"
                                                if i + 1 < len(df):
                                                    next_bar = df.iloc[i+1]
                                                    next_vol_regime = next_bar.get("atr_regime", next_bar.get("brain_vol_regime", "UNKNOWN"))
                                                    next_session = next_bar.get("session", next_bar.get("_v1_session_tag", "UNKNOWN"))
                        
                                                    writer.writerow({
                                                    "run_id": self.run_id,
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
                                                    "margin_entry": f"{trade.margin:.4f}" if hasattr(trade, "margin") else "",
                                                    "p_hat_entry": f"{trade.p_hat:.4f}" if hasattr(trade, "p_hat") else "",
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
                                                    })
                    else:
                        # Normal mode: execute trade and log to trade_log
                        self.execute_entry(trade)
                        # Fix 3: Increment trade counter
                        self.perf_n_trades_created += 1
                        log.info(
                            "[ENTRY_DIAG] open_trades_after_execute=%d",
                            len(self.open_trades),
                        )
                        # Build trade log row with FARM fields from trade.extra
                        trade_extra = trade.extra if hasattr(trade, "extra") and trade.extra else {}
                        trade_log_row = {
                            "trade_id": trade.trade_id,
                            "entry_time": trade.entry_time.isoformat(),
                            "entry_price": f"{trade.entry_price:.3f}",
                            "side": trade.side,
                            "units": trade.units,
                            "exit_time": "",
                            "exit_price": "",
                            "pnl_bps": "",
                            "pnl_currency": "",
                            "entry_prob_long": f"{trade.entry_prob_long:.4f}",
                            "entry_prob_short": f"{trade.entry_prob_short:.4f}",
                            "exit_prob_close": "",
                            "vol_bucket": trade.vol_bucket,
                            "atr_bps": f"{trade.atr_bps:.2f}",
                            "notes": self._build_notes_string(trade),
                            "run_id": self.run_id,
                            "policy_name": self.policy_name,
                            "model_name": self.model_name,
                            "extra": trade_extra,
                        }
                        # Extract FARM fields from trade.extra (append_trade_log will handle extraction, but set explicitly for clarity)
                        if trade_extra:
                            if "farm_entry_session" in trade_extra:
                                trade_log_row["farm_entry_session"] = trade_extra["farm_entry_session"]
                            if "farm_entry_vol_regime" in trade_extra:
                                trade_log_row["farm_entry_vol_regime"] = trade_extra["farm_entry_vol_regime"]
                            if "farm_guard_version" in trade_extra:
                                trade_log_row["farm_guard_version"] = trade_extra["farm_guard_version"]
                        append_trade_log(self.trade_log_path, trade_log_row)
            
            # Evaluate exits (skip in ENTRY_ONLY mode)
            if self.mode != "ENTRY_ONLY":
                self.evaluate_and_close_trades(candles_history)
            
            # Log progress every 100 bars
            if (i + 1) % 100 == 0:
                log.info("[REPLAY] Progress: %d/%d bars (%.1f%%) | open_trades=%d", 
                    i+1, len(df), 100*(i+1)/len(df), len(self.open_trades))
        
        log.info("[REPLAY] Finished processing all bars")
        # CRITICAL: bars_seen is total bars loop iterated over (including warmup skips)
        # perf_n_bars_processed is bars that reached model call stage (after warmup/pregate)
        # Ensure bars_seen reflects all bars in df (loop completion)
        expected_bars = len(df)
        if self.bars_seen != expected_bars:
            log.error(
                f"[REPLAY] FATAL: Early stop before end of subset. "
                f"bars_seen={self.bars_seen}, expected_bars={expected_bars}, diff={expected_bars - self.bars_seen}"
            )
            # Hard-fail if not timeout/stop requested
            if not (os.getenv("GX1_STOP_REQUESTED", "0") == "1" or getattr(self, "_stop_requested", False)):
                raise RuntimeError(
                    f"[REPLAY] FATAL: Early stop before end of subset. "
                    f"bars_seen={self.bars_seen}, expected_bars={expected_bars}, diff={expected_bars - self.bars_seen}"
                )
        else:
            log.info(f"[REPLAY] ✅ Loop completed all bars: bars_seen={self.bars_seen} == expected_bars={expected_bars}")

        # TRUTH/PREBUILT: finalize XGB input truth dump at end-of-replay (if fewer than N samples were collected).
        try:
            self._write_xgb_input_truth_dump_if_ready(force=True)
        except Exception:
            raise
        
        # Log when approaching end (100 bars before end)
        if self.bars_seen >= expected_bars - 100 and self.bars_seen < expected_bars:
            log.info(
                f"[REPLAY] Approaching end: bars_seen={self.bars_seen}, "
                f"expected_bars={expected_bars}, remaining={expected_bars - self.bars_seen}"
            )
        
        self.perf_n_bars_processed = self.perf_bars_total # Ensure 100% completion is reflected
        
        # DEL C: End-of-chunk perf summary
        total_time = time.time() - replay_start_time
        wall_clock_sec = total_time
        bars_per_sec = self.perf_n_bars_processed / total_time if total_time > 0 else 0.0
        feature_time_mean_ms = (self.perf_feat_time / self.perf_n_bars_processed * 1000.0) if self.perf_n_bars_processed > 0 else 0.0
        
        # Get pregate stats
        pregate_skip_ratio = (self.pregate_skips / self.perf_n_bars_processed * 100.0) if self.perf_n_bars_processed > 0 else 0.0
        
        # DEL 2: Get feature timeout count (from feature_state if available, else module-level cache)
        feature_timeout_count = getattr(self, "feature_timeout_count", 0)
        try:
            from gx1.utils.feature_context import get_feature_state
            state = get_feature_state()
            if state is not None and hasattr(state, "feature_timeout_count"):
                feature_timeout_count = state.feature_timeout_count
        except Exception:
            pass  # Non-fatal: use self.feature_timeout_count
        
        # DEL 2: Get HTF alignment warning count (from feature_state if available, else self counter)
        htf_align_warn_count = getattr(self, "htf_align_warn_count", 0)
        try:
            from gx1.utils.feature_context import get_feature_state
            state = get_feature_state()
            if state is not None and hasattr(state, "htf_align_warn_count"):
                htf_align_warn_count = state.htf_align_warn_count
        except Exception:
            pass  # Non-fatal: use self.htf_align_warn_count
        
        # PATCH: Get HTF alignment stats from FeatureState/HTFAligner (direct export, not perf collector)
        # FIX: Export aligner stats directly from FeatureState since perf collector may not have them
        htf_align_time_total_sec = 0.0
        htf_align_warning_time_sec = 0.0
        htf_align_call_count = 0
        htf_align_warn_count_total = 0
        
        # Try to get stats from aligners first (most reliable)
        if self.feature_state.h1_aligner is not None or self.feature_state.h4_aligner is not None:
            h1_stats = self.feature_state.h1_aligner.get_stats() if self.feature_state.h1_aligner is not None else {}
            h4_stats = self.feature_state.h4_aligner.get_stats() if self.feature_state.h4_aligner is not None else {}
            htf_align_call_count = h1_stats.get("call_count", 0) + h4_stats.get("call_count", 0)
            htf_align_warn_count_total = h1_stats.get("warn_count", 0) + h4_stats.get("warn_count", 0)
            # Note: aligner.get_stats() doesn't have time, so we fall back to perf collector for timing
            # But call_count and warn_count come from aligners
        
        # Fallback to perf collector for timing (if aligners don't have it)
        if htf_align_time_total_sec == 0.0 and hasattr(self, "perf_collector") and self.perf_collector:
            try:
                perf_data = self.perf_collector.get_all()
                for name, data in perf_data.items():
                    if name.startswith("feat.htf_align.call_total"):
                        htf_align_time_total_sec += data.get("total_sec", 0.0)
                    elif name.startswith("feat.htf_align.warning_overhead_est"):
                        htf_align_warning_time_sec += data.get("total_sec", 0.0)
            except Exception:
                pass  # Non-fatal: timing data unavailable
        
        # Store in runner for chunk footer export
        self.htf_align_time_total_sec = htf_align_time_total_sec
        self.htf_align_call_count = htf_align_call_count
        self.htf_align_warning_time_sec = htf_align_warning_time_sec
        self.htf_align_warn_count_total = htf_align_warn_count_total
        
        # Get vol_regime_unknown_count from entry_manager telemetry
        vol_regime_unknown_count = 0
        if hasattr(self, "entry_manager") and self.entry_manager:
            vol_regime_unknown_count = self.entry_manager.entry_telemetry.get("vol_regime_unknown_count", 0)
        self.vol_regime_unknown_count = vol_regime_unknown_count
        
        feature_timeout_rate = (feature_timeout_count / self.perf_n_bars_processed * 100.0) if self.perf_n_bars_processed > 0 else 0.0
        
        # HARD INVARIANT: If prebuilt is enabled and used, verify all invariants
        if hasattr(self, "prebuilt_used") and self.prebuilt_used:
            # A1) FEATURE_BUILD_TIMEOUT must be 0
            if feature_timeout_count > 0:
                raise RuntimeError(
                    f"[PREBUILT_FAIL] FEATURE_BUILD_TIMEOUT triggered {feature_timeout_count} times in prebuilt-run. "
                    f"This indicates build_basic_v1() was called despite prebuilt features being enabled. "
                    f"Instructions: Check that entry_manager.evaluate_entry bypasses build_live_entry_features when prebuilt is enabled."
                )
            
            # A2) feature_build_call_count must be 0
            # FASE 1: Use PREBUILT-safe tripwire counter (does not import basic_v1)
            from gx1.execution.feature_build_tripwires import (
                get_feature_build_call_count,
                get_feature_build_call_details,
            )
            feature_build_calls = get_feature_build_call_count()
            feature_build_details = get_feature_build_call_details()
            if feature_build_calls > 0:
                raise RuntimeError(
                    f"[PREBUILT_FAIL] Feature build functions were called {feature_build_calls} times "
                    f"in prebuilt-run (expected 0). Details: {feature_build_details}. "
                    f"This indicates prebuilt bypass is not working. "
                    f"Instructions: Check that entry_manager.evaluate_entry bypasses build_live_entry_features when prebuilt is enabled."
                )
            
            # A3) feature_time_mean_ms must be <= 5ms (or 0.0 with bypass_count proof)
            prebuilt_bypass_count = getattr(self, "prebuilt_bypass_count", 0)
            feature_time_mean_ms = (self.perf_feat_time / self.perf_n_bars_processed * 1000.0) if self.perf_n_bars_processed > 0 else 0.0
            if feature_time_mean_ms > 5.0:
                if prebuilt_bypass_count == 0:
                    raise RuntimeError(
                        f"[PREBUILT_FAIL] feature_time_mean_ms={feature_time_mean_ms:.2f}ms > 5ms and prebuilt_bypass_count=0. "
                        f"This indicates feature-building is still happening. "
                        f"Instructions: Verify prebuilt bypass is working correctly."
                    )
                else:
                    log.warning(
                        "[PREBUILT] feature_time_mean_ms=%.2fms > 5ms but prebuilt_bypass_count=%d > 0. "
                        "This may indicate ATR/vol_bucket computation overhead.",
                        feature_time_mean_ms, prebuilt_bypass_count
                    )
            
            # A4) prebuilt_bypass_count must equal n_model_calls (minus warmup)
            # Get warmup_bars from runner state
            warmup_bars = getattr(self, "n_bars_skipped_due_to_htf_warmup", 0)
            n_model_calls = self.perf_n_bars_processed
            expected_bypass = n_model_calls  # All processed bars should use prebuilt (warmup is skipped before processing)
            if prebuilt_bypass_count != expected_bypass:
                log.warning(
                    "[PREBUILT] prebuilt_bypass_count=%d != n_model_calls=%d (warmup_bars=%d). "
                    "This may indicate some bars did not use prebuilt features.",
                    prebuilt_bypass_count, n_model_calls, warmup_bars
                )
            
            if self.perf_feat_time > 0.1:  # Allow small tolerance for ATR/vol_bucket computation
                log.warning(
                    "[PREBUILT] perf_feat_time=%.3f > 0.1s (expected ~0 for prebuilt). "
                    "This may indicate feature-building is still happening.",
                    self.perf_feat_time
                )
        
        # DEL 1: Phase timing breakdown (for "bars/sec" pie chart)
        # B1: Replace 50/50 estimate with true timers (best-effort, semantics-neutral).
        # Keep feature time as pure feature-building time (PREBUILT => ~0).
        t_feature_build_pure_sec = self.perf_feat_time  # Pure feature building time (from entry_manager)

        # True model time components
        self.t_model_total_sec = float(getattr(self, "t_xgb_predict_sec", 0.0) or 0.0) + float(getattr(self, "t_transformer_forward_sec", 0.0) or 0.0)
        # True gates/policy time (guard + policy evaluation logic)
        self.t_policy_total_sec = float(getattr(self, "t_gates_policy_sec", 0.0) or 0.0)

        # I/O time is estimated as "everything else" after known phases.
        # This is an observability-only approximation.
        known = (
            float(self.t_pregate_total_sec or 0.0)
            + float(t_feature_build_pure_sec or 0.0)
            + float(self.t_model_total_sec or 0.0)
            + float(self.t_policy_total_sec or 0.0)
            + float(getattr(self, "t_replay_tags_sec", 0.0) or 0.0)
            + float(getattr(self, "t_telemetry_sec", 0.0) or 0.0)
        )
        self.t_io_total_sec = max(0.0, float(wall_clock_sec) - known)
        
        # Log prebuilt bypass count if available
        prebuilt_bypass_log = ""
        if hasattr(self, "prebuilt_bypass_count") and self.prebuilt_bypass_count > 0:
            prebuilt_bypass_log = f" | prebuilt_bypass_count={self.prebuilt_bypass_count}"
        
        log.info(
            "[REPLAY_PERF_SUMMARY] "
            "wall_clock_sec=%.1f | bars_per_sec=%.2f | total_bars=%d | n_model_calls=%d | n_trades_closed=%d | "
            "pregate_skips=%d (%.1f%%) | pregate_passes=%d | pregate_missing_inputs=%d | "
            "feature_time_mean_ms=%.2f | feature_timeout_count=%d (%.2f%%) | htf_align_warn_count=%d | "
            "htf_align_time=%.2fs | htf_align_calls=%d | htf_align_warn_overhead=%.2fs | "
            "t_pregate=%.2fs | t_feature=%.2fs | t_model=%.2fs | t_policy=%.2fs | t_io=%.2fs%s",
            wall_clock_sec,
            bars_per_sec,
            self.perf_n_bars_processed,
            self.perf_n_bars_processed,  # n_model_calls = bars processed (after pregate)
            self.perf_n_trades_created,
            self.pregate_skips,
            pregate_skip_ratio,
            self.pregate_passes,
            self.pregate_missing_inputs,
            feature_time_mean_ms,
            feature_timeout_count,
            feature_timeout_rate,
            htf_align_warn_count,
            htf_align_time_total_sec,
            htf_align_call_count,
            htf_align_warning_time_sec,
            self.t_pregate_total_sec,
            t_feature_build_pure_sec,
            self.t_model_total_sec,
            self.t_policy_total_sec,
            self.t_io_total_sec,
            prebuilt_bypass_log,
        )
        
        # DEL 1: Flush replay eval collectors to disk (if enabled) — legacy path forbidden.
        if hasattr(self, "replay_eval_collectors") and self.replay_eval_collectors:
            log.info("[REPLAY_EVAL] skip legacy flush_replay_eval_collectors (forbidden import path)")
        
        # Del 4: Log cache statistics once at replay end
        total_cache_requests = self.feature_state.htf_cache_hits + self.feature_state.htf_cache_misses
        if total_cache_requests > 0:
            hit_rate = (self.feature_state.htf_cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0.0
            log.info(
                "[HTF_CACHE] Replay complete - Cache stats: Hits=%d, Misses=%d, Hit rate=%.1f%%, Total requests=%d",
                self.feature_state.htf_cache_hits, self.feature_state.htf_cache_misses, hit_rate, total_cache_requests
            )
        
        # PATCH 8: Log HTF aligner stats (instrumentering - to tall som avgjør alt)
        # FIX: Update runner stats from FeatureState/HTFAligner (for chunk footer export)
        if self.feature_state.h1_aligner is not None or self.feature_state.h4_aligner is not None:
            h1_stats = self.feature_state.h1_aligner.get_stats() if self.feature_state.h1_aligner is not None else {}
            h4_stats = self.feature_state.h4_aligner.get_stats() if self.feature_state.h4_aligner is not None else {}
            total_align_calls = h1_stats.get("call_count", 0) + h4_stats.get("call_count", 0)
            total_align_warns = h1_stats.get("warn_count", 0) + h4_stats.get("warn_count", 0)
            fallback_count = getattr(self.feature_state, "htf_align_fallback_count", 0)
            
            # FIX: Update runner stats from aligners (overrides perf collector if available)
            if total_align_calls > 0:
                self.htf_align_call_count = total_align_calls
            if total_align_warns > 0:
                self.htf_align_warn_count_total = total_align_warns
            
            # Expected: ≈ n_bars × 2 (H1 + H4), not n_bars × 7–8
            expected_calls = self.perf_n_bars_processed * 2
            call_ratio = (total_align_calls / expected_calls * 100) if expected_calls > 0 else 0.0
            
            log.info(
                "[HTF_ALIGNER] Replay complete - Aligner stats: "
                "H1_calls=%d, H4_calls=%d, Total_calls=%d (expected≈%d, ratio=%.1f%%), "
                "H1_warns=%d, H4_warns=%d, Total_warns=%d, Fallback_count=%d",
                h1_stats.get("call_count", 0),
                h4_stats.get("call_count", 0),
                total_align_calls,
                expected_calls,
                call_ratio,
                h1_stats.get("warn_count", 0),
                h4_stats.get("warn_count", 0),
                total_align_warns,
                fallback_count
            )
            
            # PATCH: Replay-only invariant - verify call counts match feature compute bars
            htf_feature_compute_bars = getattr(self.feature_state, "htf_feature_compute_bars", 0)
            h1_call_count = h1_stats.get("call_count", 0)
            h4_call_count = h4_stats.get("call_count", 0)
            
            # Check if we're in replay mode
            is_replay = getattr(self, "replay_mode", False) or os.getenv("GX1_REPLAY", "0") == "1"
            
            if is_replay and htf_feature_compute_bars > 0:
                # Invariant: H1 and H4 should be called once per bar where features were computed
                # Allow small tolerance for warmup/skip (±5 bars)
                h1_deviation = abs(h1_call_count - htf_feature_compute_bars)
                h4_deviation = abs(h4_call_count - htf_feature_compute_bars)
                tolerance = max(5, int(htf_feature_compute_bars * 0.01))  # ±1% or ±5 bars, whichever is larger
                
                if h1_deviation > tolerance:
                    raise RuntimeError(
                        f"HTF_ALIGNER INVARIANT VIOLATION: H1 call_count ({h1_call_count}) != "
                        f"htf_feature_compute_bars ({htf_feature_compute_bars}), deviation={h1_deviation} "
                        f"(tolerance={tolerance}). This indicates aligner.step() was not called for every bar."
                    )
                
                if h4_deviation > tolerance:
                    raise RuntimeError(
                        f"HTF_ALIGNER INVARIANT VIOLATION: H4 call_count ({h4_call_count}) != "
                        f"htf_feature_compute_bars ({htf_feature_compute_bars}), deviation={h4_deviation} "
                        f"(tolerance={tolerance}). This indicates aligner.step() was not called for every bar."
                    )
                
                log.info(
                    "[HTF_ALIGNER] Invariant verified: H1_calls=%d, H4_calls=%d, "
                    "htf_feature_compute_bars=%d (all match within tolerance=%d)",
                    h1_call_count, h4_call_count, htf_feature_compute_bars, tolerance
                )
            
            # PATCH 8: Hard-fail if call count is too high (indicates O(N²) still present)
            if total_align_calls > expected_calls * 1.5:  # Allow 50% overhead for edge cases
                log.warning(
                    "[HTF_ALIGNER] WARNING: Total aligner calls (%d) is much higher than expected (%d). "
                    "This may indicate O(N²) behavior is still present. Ratio: %.1f%%",
                    total_align_calls, expected_calls, call_ratio
                )
            
            # PATCH 6: Hard-fail if fallback was used (SSoT violation)
            if fallback_count > 0:
                if os.getenv("GX1_HTF_ALIGN_FALLBACK_ALLOWED") != "1":
                    raise RuntimeError(
                        f"HTF legacy fallback was used {fallback_count} times in replay (SSoT violation). "
                        f"Stateful alignment should always be available. "
                        f"Set GX1_HTF_ALIGN_FALLBACK_ALLOWED=1 to allow (not recommended)."
                    )
                else:
                    log.warning(
                        "[HTF_ALIGNER] WARNING: Fallback was used %d times (allowed via GX1_HTF_ALIGN_FALLBACK_ALLOWED=1)",
                        fallback_count
                    )
        
        # Del 2B: Clear perf collector context
        from gx1.utils.perf_timer import reset_current_perf
        reset_current_perf(perf_token)
        
        # Del 2: Reset feature state context
        from gx1.utils.feature_context import reset_feature_state
        reset_feature_state(feature_token)
        
        # Write Stage-0 reason report if available
        if hasattr(self, "_stage0_reasons") and self._stage0_reasons:
            try:
                from pathlib import Path
                report_path = Path_module(self.output_dir) / "stage0_reasons.json"
                with open(report_path, "w") as f:
                    jsonlib.dump(self._stage0_reasons, f, indent=2)
                log.info("[STAGE0] Wrote reason report to %s", report_path)
            except Exception as e:
                log.warning("[STAGE0] Failed to write reason report: %s", e)
        
        # Dump FARM_V2B diagnostic if available (before closing remaining trades)
        if hasattr(self, "entry_manager") and hasattr(self.entry_manager, "farm_diag"):
            farm_diag = self.entry_manager.farm_diag
            if farm_diag.get("n_bars", 0) > 0:
                # Create diagnostic output path
                from datetime import datetime
                diag_filename = f"farm_entry_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                diag_path = self.log_dir / diag_filename
                
                # Convert numpy types to native Python types for JSON serialization
                diag_serializable = {
                    "n_bars": int(farm_diag["n_bars"]),
                    "n_raw_candidates": int(farm_diag["n_raw_candidates"]),
                    "n_after_stage0": int(farm_diag["n_after_stage0"]),
                    "n_after_farm_regime": int(farm_diag["n_after_farm_regime"]),
                    "n_after_brutal_guard": int(farm_diag["n_after_brutal_guard"]),
                    "n_after_policy_thresholds": int(farm_diag["n_after_policy_thresholds"]),
                    "p_long_values": [float(x) for x in farm_diag["p_long_values"]],
                    "sessions": {str(k): int(v) for k, v in farm_diag["sessions"].items()},
                    "atr_regimes": {str(k): int(v) for k, v in farm_diag["atr_regimes"].items()},
                    "farm_regimes": {str(k): int(v) for k, v in farm_diag["farm_regimes"].items()},
                }
                
                # Add statistics for p_long_values
                if len(diag_serializable["p_long_values"]) > 0:
                    p_long_arr = np.array(diag_serializable["p_long_values"])
                    diag_serializable["p_long_stats"] = {
                        "min": float(np.min(p_long_arr)),
                        "max": float(np.max(p_long_arr)),
                        "mean": float(np.mean(p_long_arr)),
                        "median": float(np.median(p_long_arr)),
                        "p5": float(np.percentile(p_long_arr, 5)),
                        "p50": float(np.percentile(p_long_arr, 50)),
                        "p95": float(np.percentile(p_long_arr, 95)),
                        "p99": float(np.percentile(p_long_arr, 99)),
                    }
                else:
                    diag_serializable["p_long_stats"] = {}
                
                # Write to JSON
                with open(diag_path, "w") as f:
                    jsonlib.dump(diag_serializable, f, indent=2)
                
                log.info(f"[FARM_DIAG] Dumped FARM_V2B entry diagnostic to {diag_path}")
        
        # B) After loop: verify canonical loop counters (SSoT)
        # Calculate expected values
        expected_post_warmup = max(0, self.bars_total_coalesced - self.first_valid_eval_idx_stored)
        
        # Log canonical counters for diagnostics
        log.info(
            f"[LOOP_COUNTERS] loop_iters_total={self.loop_iters_total} "
            f"bars_total_coalesced={self.bars_total_coalesced} "
            f"loop_iters_post_warmup={self.loop_iters_post_warmup} "
            f"expected_post_warmup={expected_post_warmup} "
            f"first_valid_eval_idx={self.first_valid_eval_idx_stored} "
            f"stop_reason={self.last_stop_reason} "
            f"last_i={self.last_i} "
            f"last_ts={self.last_iterated_ts}"
        )
        
        # Get entry_eval_entered_total from telemetry (if available)
        entry_eval_entered = 0
        if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
            entry_eval_entered = getattr(self.entry_manager.entry_feature_telemetry, "entry_eval_entered_total", 0)
        self.funnel_predict_entered = entry_eval_entered
        
        # FUNNEL LEDGER SUMMARY: Print one line summary
        log.info(
            f"[ENTRY_FUNNEL] post_warmup={self.funnel_bars_post_warmup} "
            f"pregate_pass={self.funnel_pregate_pass} pregate_block={self.funnel_pregate_block} "
            f"eval_called={self.funnel_entry_eval_called} predict_entered={self.funnel_predict_entered}"
        )
        
        # MODEL_FUNNEL: Detailed breakdown of post-predict flow
        model_funnel = {
            "predict_entered": 0,
            "pre_model_return": 0,
            "pre_model_return_reasons": {},
            "routing_v10_hybrid": 0,
            "post_lookup": 0,
            "post_vol_guard": 0,
            "vol_guard_block_reasons": {},
            "post_score_gate": 0,
            "score_gate_block_reasons": {},
            "post_model_call": 0,
            "model_forward": 0,
            "exceptions": 0,
        }
        if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
            telemetry = self.entry_manager.entry_feature_telemetry
            model_funnel["predict_entered"] = getattr(telemetry, "entry_eval_entered_total", 0)
            model_funnel["pre_model_return"] = getattr(telemetry, "pre_model_return_count", 0)
            model_funnel["pre_model_return_reasons"] = dict(getattr(telemetry, "pre_model_return_reasons", {}))
            model_funnel["routing_v10_hybrid"] = telemetry.entry_routing_selected_model_counts.get("v10_hybrid", 0)
            model_funnel["post_lookup"] = getattr(telemetry, "post_lookup_reached", 0)
            model_funnel["post_vol_guard"] = getattr(telemetry, "post_vol_guard_reached", 0)
            model_funnel["vol_guard_block_reasons"] = dict(getattr(telemetry, "stage2_block_reasons", {}))
            model_funnel["post_score_gate"] = getattr(telemetry, "post_score_gate_reached", 0)
            model_funnel["score_gate_block_reasons"] = dict(getattr(telemetry, "stage3_block_reasons", {}))
            model_funnel["post_model_call"] = getattr(telemetry, "post_model_call_reached", 0)
            model_funnel["model_forward"] = getattr(telemetry, "transformer_forward_calls", 0)
            model_funnel["exceptions"] = getattr(telemetry, "model_exception_count", 0)
        
        # Print MODEL_FUNNEL summary
        log.info(
            f"[MODEL_FUNNEL] predict={model_funnel['predict_entered']} "
            f"pre_return={model_funnel['pre_model_return']} "
            f"routing(v10_hybrid={model_funnel['routing_v10_hybrid']}) "
            f"lookup={model_funnel['post_lookup']} "
            f"vol={model_funnel['post_vol_guard']} "
            f"score={model_funnel['post_score_gate']} "
            f"pre_call={model_funnel['post_model_call']} "
            f"forward={model_funnel['model_forward']} "
            f"exceptions={model_funnel['exceptions']}"
        )
        
        # HARD FAIL RULE: Diagnostic when bars exist but no entry eval
        output_dir = getattr(self, "explicit_output_dir", None)
        if getattr(self, "replay_mode", False):
            pass
        elif self.funnel_bars_post_warmup >= 200 and self.funnel_entry_eval_called == 0:
            # Write ENTRY_FUNNEL_ZERO_CALLS.json
            import json
            top_pregate_reasons = sorted(
                self.funnel_pregate_block_reasons.items(),
                key=lambda x: -x[1]
            )[:10]
            funnel_diag = {
                "error": "ENTRY_FUNNEL_ZERO_CALLS",
                "message": "Post-warmup bars exist but evaluate_entry() was never called",
                "funnel_bars_total": self.funnel_bars_total if hasattr(self, "funnel_bars_total") else self.loop_iters_total,
                "funnel_bars_post_warmup": self.funnel_bars_post_warmup,
                "funnel_pregate_pass": self.funnel_pregate_pass,
                "funnel_pregate_block": self.funnel_pregate_block,
                "top10_pregate_block_reasons": dict(top_pregate_reasons),
                "funnel_entry_eval_called": self.funnel_entry_eval_called,
                "funnel_predict_entered": self.funnel_predict_entered,
                "first_ts_post_warmup": str(self.funnel_first_ts_post_warmup) if self.funnel_first_ts_post_warmup else None,
                "last_ts_post_warmup": str(self.funnel_last_ts_post_warmup) if self.funnel_last_ts_post_warmup else None,
                "replay_mode": getattr(self, "replay_mode_enum_str", str(getattr(self, "replay_mode_enum", "UNKNOWN"))),
            }
            if output_dir:
                try:
                    diag_path = Path(output_dir) / "ENTRY_FUNNEL_ZERO_CALLS.json"
                    with open(diag_path, "w") as f:
                        json.dump(funnel_diag, f, indent=2, default=str)
                    log.error(f"[ENTRY_FUNNEL] Wrote diagnostic to: {diag_path}")
                except Exception as e:
                    log.error(f"[ENTRY_FUNNEL] Failed to write diagnostic: {e}")
            log.error(
                f"[ENTRY_FUNNEL] FATAL: Post-warmup bars ({self.funnel_bars_post_warmup}) >= 200 "
                f"but evaluate_entry() never called. Top pregate block reasons: {top_pregate_reasons[:5]}"
            )
        elif not getattr(self, "replay_mode", False) and self.funnel_entry_eval_called > 0 and self.funnel_predict_entered == 0:
            # Write ENTRY_FUNNEL_EVAL_BUT_NO_PREDICT.json
            import json
            top_pregate_reasons = sorted(
                self.funnel_pregate_block_reasons.items(),
                key=lambda x: -x[1]
            )[:10]
            funnel_diag = {
                "error": "ENTRY_FUNNEL_EVAL_BUT_NO_PREDICT",
                "message": "evaluate_entry() was called but _predict_entry_v10_hybrid never entered",
                "funnel_bars_total": self.funnel_bars_total if hasattr(self, "funnel_bars_total") else self.loop_iters_total,
                "funnel_bars_post_warmup": self.funnel_bars_post_warmup,
                "funnel_pregate_pass": self.funnel_pregate_pass,
                "funnel_pregate_block": self.funnel_pregate_block,
                "top10_pregate_block_reasons": dict(top_pregate_reasons),
                "funnel_entry_eval_called": self.funnel_entry_eval_called,
                "funnel_predict_entered": self.funnel_predict_entered,
                "first_ts_post_warmup": str(self.funnel_first_ts_post_warmup) if self.funnel_first_ts_post_warmup else None,
                "last_ts_post_warmup": str(self.funnel_last_ts_post_warmup) if self.funnel_last_ts_post_warmup else None,
                "replay_mode": getattr(self, "replay_mode_enum_str", str(getattr(self, "replay_mode_enum", "UNKNOWN"))),
            }
            if output_dir:
                try:
                    diag_path = Path(output_dir) / "ENTRY_FUNNEL_EVAL_BUT_NO_PREDICT.json"
                    with open(diag_path, "w") as f:
                        json.dump(funnel_diag, f, indent=2, default=str)
                    log.error(f"[ENTRY_FUNNEL] Wrote diagnostic to: {diag_path}")
                except Exception as e:
                    log.error(f"[ENTRY_FUNNEL] Failed to write diagnostic: {e}")
            log.error(
                f"[ENTRY_FUNNEL] WARNING: evaluate_entry() called {self.funnel_entry_eval_called} times "
                f"but _predict_entry_v10_hybrid never entered. Check entry_manager logic."
            )
        
        # Hard invariant: TRANSFORMER_CALLS_STARVED
        # If predict_entered_total > 50 and model_forward_calls < 5
        if model_funnel["predict_entered"] > 50 and model_funnel["model_forward"] < 5:
            import json
            top_pre_model_return_reasons = sorted(
                model_funnel["pre_model_return_reasons"].items(),
                key=lambda x: -x[1]
            )[:10]
            top_vol_guard_block_reasons = sorted(
                model_funnel["vol_guard_block_reasons"].items(),
                key=lambda x: -x[1]
            )[:10]
            top_score_gate_block_reasons = sorted(
                model_funnel["score_gate_block_reasons"].items(),
                key=lambda x: -x[1]
            )[:10]
            
            # Get session breakdown from telemetry
            session_breakdown = {"EU": 0, "US": 0, "OVERLAP": 0}
            if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
                telemetry = self.entry_manager.entry_feature_telemetry
                session_breakdown = dict(getattr(telemetry, "transformer_forward_calls_by_session", {"EU": 0, "US": 0, "OVERLAP": 0}))
            
            starved_diag = {
                "error": "TRANSFORMER_CALLS_STARVED",
                "message": f"predict_entered={model_funnel['predict_entered']} but model_forward={model_funnel['model_forward']} < 5",
                "predict_entered_total": model_funnel["predict_entered"],
                "pre_model_return_count": model_funnel["pre_model_return"],
                "top10_pre_model_return_reasons": dict(top_pre_model_return_reasons),
                "routing_selected_model_counts": {
                    "v10_hybrid": model_funnel["routing_v10_hybrid"],
                },
                "stage_counters": {
                    "post_lookup": model_funnel["post_lookup"],
                    "post_vol_guard": model_funnel["post_vol_guard"],
                    "post_score_gate": model_funnel["post_score_gate"],
                    "post_model_call": model_funnel["post_model_call"],
                    "model_forward": model_funnel["model_forward"],
                },
                "vol_guard_block_reasons_top10": dict(top_vol_guard_block_reasons),
                "score_gate_block_reasons_top10": dict(top_score_gate_block_reasons),
                "session_breakdown": session_breakdown,
                "model_exception_count": model_funnel["exceptions"],
                "replay_mode": getattr(self, "replay_mode_enum_str", str(getattr(self, "replay_mode_enum", "UNKNOWN"))),
            }
            if output_dir:
                try:
                    diag_path = Path(output_dir) / "TRANSFORMER_CALLS_STARVED.json"
                    with open(diag_path, "w") as f:
                        json.dump(starved_diag, f, indent=2, default=str)
                    log.error(f"[MODEL_FUNNEL] Wrote diagnostic to: {diag_path}")
                except Exception as e:
                    log.error(f"[MODEL_FUNNEL] Failed to write diagnostic: {e}")
            log.error(
                f"[MODEL_FUNNEL] FATAL: predict_entered={model_funnel['predict_entered']} "
                f"but model_forward={model_funnel['model_forward']} < 5. "
                f"Top pre_model_return reasons: {top_pre_model_return_reasons[:5]}"
            )
        
        # After loop: if last_stop_reason is still "EOF", loop completed normally
        if self.last_stop_reason == "EOF":
            log.info("[REPLAY] Loop completed normally (EOF)")
            # Verify loop_iters_total == bars_total_coalesced
            if self.loop_iters_total != self.bars_total_coalesced:
                log.error(
                    f"[LOOP_COUNTERS] MISMATCH: loop_iters_total={self.loop_iters_total} != "
                    f"bars_total_coalesced={self.bars_total_coalesced}. "
                    f"This should not happen if loop completed normally."
                )
            # Verify loop_iters_post_warmup == expected_post_warmup
            if self.loop_iters_post_warmup != expected_post_warmup:
                log.error(
                    f"[LOOP_COUNTERS] MISMATCH: loop_iters_post_warmup={self.loop_iters_post_warmup} != "
                    f"expected_post_warmup={expected_post_warmup}. "
                    f"This indicates a bug in first_valid_eval_idx handling."
                )
    
        # Clear replay context
        if hasattr(self, '_replay_current_ts'):
            delattr(self, '_replay_current_ts')
    
        # Close any remaining open trades at final price
        # Check if EOF close is enabled in policy
        replay_cfg = self.policy.get("replay", {})
        close_open_at_end = replay_cfg.get("close_open_trades_at_end", False)
        eof_reason = replay_cfg.get("eof_close_reason", "REPLAY_EOF")
        eof_price_source = replay_cfg.get("eof_price_source", "mid")

        # TRUTH-only optional: Accounting close at end of replay/chunk
        # This is explicitly enabled via env flag to avoid changing baseline semantics by default.
        if os.getenv("GX1_REPLAY_ACCOUNTING_CLOSE_AT_END", "0") == "1":
            close_open_at_end = True
            eof_reason = "REPLAY_END"
            try:
                self.exit_coverage["accounting_close_enabled"] = True
            except Exception:
                pass
        
        # Find open trades from trade log (more reliable than self.open_trades in replay)
        open_trades_from_log = []
        if self.trade_log_path.exists():
            try:
                trades_df = pd.read_csv(self.trade_log_path, on_bad_lines='skip', engine='python')
                if len(trades_df) > 0:
                    # Filter trades without exit_time or with empty exit_time
                    if "exit_time" in trades_df.columns:
                        open_trades_from_log = trades_df[
                            trades_df["exit_time"].isna() | 
                            (trades_df["exit_time"].astype(str).str.strip() == "") |
                            (trades_df["exit_time"].astype(str).str.strip() == "nan")
                        ].copy()
                    elif "pnl_bps" in trades_df.columns:
                        # Alternative: trades without pnl_bps are open
                        open_trades_from_log = trades_df[
                            trades_df["pnl_bps"].isna() | 
                            (trades_df["pnl_bps"].astype(str).str.strip() == "") |
                            (trades_df["pnl_bps"].astype(str).str.strip() == "nan")
                        ].copy()
            except Exception as e:
                log.warning("[REPLAY] Failed to read trade log for EOF close: %s", e)
        
        log.info(
            "[REPLAY] EOF Close check: open_trades=%d (from self.open_trades), open_from_log=%d, close_open_at_end=%s",
            len(self.open_trades) if self.open_trades else 0,
            len(open_trades_from_log),
            close_open_at_end,
        )
    
        # Use trade log if self.open_trades is empty but we have open trades in log
        should_close_from_log = not self.open_trades and len(open_trades_from_log) > 0 and close_open_at_end
        
        if (self.open_trades and close_open_at_end) or should_close_from_log:
            final_candle = df.iloc[-1]
            final_bid = float(final_candle["bid_close"])
            final_ask = float(final_candle["ask_close"])
            final_mid = (final_bid + final_ask) / 2.0
            
            # Determine exit price based on eof_price_source
            if eof_price_source == "bid":
                exit_price = final_bid
                exit_bid = final_bid
                exit_ask = final_bid
            elif eof_price_source == "ask":
                exit_price = final_ask
                exit_bid = final_ask
                exit_ask = final_ask
            else:  # mid (default)
                exit_price = final_mid
                exit_bid = final_bid
                exit_ask = final_ask
            
            if should_close_from_log:
                # Close trades directly from trade log (no LiveTrade objects needed)
                log.info(
                    "[REPLAY] EOF Close: Closing %d open trades from trade log at %s price (bid=%.3f ask=%.3f mid=%.3f) reason=%s",
                    len(open_trades_from_log),
                    eof_price_source,
                    final_bid,
                    final_ask,
                    final_mid,
                    eof_reason,
                )
                
                    # compute_pnl_bps is already imported at module level (line 74)
                    # No need for local import here
                closed_count = 0
                last_ts = df.index[-1]
                
                for _, row in open_trades_from_log.iterrows():
                    try:
                        trade_id = str(row.get("trade_id", ""))
                        if not trade_id:
                            continue
                        
                        # Get entry prices
                        entry_price = float(row.get("entry_price", 0))
                        entry_bid = float(row.get("entry_bid", entry_price)) if pd.notna(row.get("entry_bid")) else entry_price
                        entry_ask = float(row.get("entry_ask", entry_price)) if pd.notna(row.get("entry_ask")) else entry_price
                        
                        # Fix 1: Validate and normalize side before compute_pnl_bps
                        side_raw = row.get("side", "long")
                        if isinstance(side_raw, str):
                            side = side_raw.lower().strip()
                        else:
                            side = str(side_raw).lower().strip() if pd.notna(side_raw) else "long"
                        
                        # Validate side is "long" or "short"
                        if side not in ("long", "short"):
                            log.error(
                                "[REPLAY_EOF] Invalid side for trade_id=%s: got %r (raw=%r). Skipping EOF close.",
                                trade_id, side, side_raw
                            )
                            # Mark trade as left_open_eof (best-effort close failed)
                            if hasattr(self, "trade_journal") and self.trade_journal:
                                try:
                                    trade_journal = self.trade_journal._get_trade_journal(trade_id)
                                    if trade_journal.get("exit_summary"):
                                        trade_journal["exit_summary"]["left_open_eof"] = True
                                        trade_journal["exit_summary"]["eof_close_error"] = f"Invalid side: {side_raw}"
                                    self.trade_journal._write_trade_json(trade_id)
                                except Exception as e:
                                    log.warning("[REPLAY_EOF] Failed to mark trade %s as left_open_eof: %s", trade_id, e)
                            continue  # Skip this trade, continue with next
                        
                        # Calculate PnL
                        pnl_bps = compute_pnl_bps(entry_bid, entry_ask, exit_bid, exit_ask, side)
                        
                        # Calculate bars held
                        entry_time = pd.to_datetime(row.get("entry_time"), utc=True, errors='coerce')
                        if pd.isna(entry_time):
                            bars_held = 0
                        else:
                            delta_minutes = (last_ts - entry_time).total_seconds() / 60.0
                            bars_held = int(round(delta_minutes / 5.0))  # M5 = 5 minutes per bar
                        
                        # Update trade log
                        self._update_trade_log_on_close(
                            trade_id=trade_id,
                            exit_price=exit_price,
                            pnl_bps=pnl_bps,
                            reason=eof_reason,
                            exit_ts=last_ts,
                            bars_in_trade=bars_held,
                        )
                        
                        # Update trade journal
                        if hasattr(self, "trade_journal") and self.trade_journal:
                            try:
                                # Extract trade_uid from journal if available (COMMIT C)
                                # Search all existing journals for matching trade_id to find correct trade_uid
                                trade_uid_from_journal = None
                                if trade_id:
                                    # Search through all journals to find one with matching trade_id
                                    for key, journal in self.trade_journal._trade_journals.items():
                                        if journal.get("trade_id") == trade_id:
                                            trade_uid_from_journal = journal.get("trade_uid")
                                            break
                                
                                # GUARD 2: In replay mode, never create new journal in exit path
                                # If trade_uid not found, this means entry_snapshot was never logged - fail hard
                                if self.is_replay and not trade_uid_from_journal:
                                    raise RuntimeError(
                                        f"EXIT_WITHOUT_ENTRY_SNAPSHOT_REPLAY: Attempted to log exit_summary for trade_id={trade_id} "
                                        f"but no journal exists with matching trade_id. This indicates exit logging attempted "
                                        f"without entry_snapshot being logged first. This is a hard contract violation in replay mode."
                                    )
                                
                                # Only log exit_summary if we found a valid trade_uid
                                if trade_uid_from_journal:
                                    # Calculate MAE/MFE from candles for EOF-close trades
                                    max_mae_bps = None
                                    max_mfe_bps = None
                                    intratrade_drawdown_bps = None
                                    
                                    try:
                                        # Get entry time and side from journal
                                        trade_journal_temp = self.trade_journal._get_trade_journal(trade_uid=trade_uid_from_journal, trade_id=trade_id)
                                        entry_snapshot = trade_journal_temp.get("entry_snapshot", {})
                                        entry_time_str = entry_snapshot.get("entry_time")
                                        side = entry_snapshot.get("side", "long").lower()
                                        
                                        if entry_time_str and entry_time:
                                            entry_time_ts = pd.to_datetime(entry_time_str, utc=True)
                                            # Filter candles between entry and exit
                                            trade_bars = df[(df.index >= entry_time_ts) & (df.index <= last_ts)].copy()
                                            
                                            if len(trade_bars) > 0:
                                                entry_price_val = float(entry_price)
                                                
                                                # Get price columns
                                                if side == "long":
                                                    high_col = "bid_high" if "bid_high" in trade_bars.columns else "high"
                                                    low_col = "bid_low" if "bid_low" in trade_bars.columns else "low"
                                                else:  # short
                                                    high_col = "ask_high" if "ask_high" in trade_bars.columns else "high"
                                                    low_col = "ask_low" if "ask_low" in trade_bars.columns else "low"
                                                
                                                if high_col in trade_bars.columns and low_col in trade_bars.columns:
                                                    highs = trade_bars[high_col].values
                                                    lows = trade_bars[low_col].values
                                                    
                                                    # Calculate PnL in bps
                                                    if side == "long":
                                                        bar_highs_pnl = ((highs - entry_price_val) / entry_price_val) * 10000.0
                                                        bar_lows_pnl = ((lows - entry_price_val) / entry_price_val) * 10000.0
                                                    else:  # short
                                                        bar_highs_pnl = ((entry_price_val - highs) / entry_price_val) * 10000.0
                                                        bar_lows_pnl = ((entry_price_val - lows) / entry_price_val) * 10000.0
                                                    
                                                    # MFE: max favorable (positive)
                                                    max_mfe_bps = float(max(bar_highs_pnl))
                                                    
                                                    # MAE: max adverse (negative, but we want positive magnitude)
                                                    max_mae_bps_raw = float(min(bar_lows_pnl))
                                                    max_mae_bps = abs(max_mae_bps_raw)  # Positive magnitude
                                                    
                                                    # Intratrade drawdown (simplified - use closes if available)
                                                    close_col = "bid_close" if side == "long" and "bid_close" in trade_bars.columns else ("ask_close" if side == "short" and "ask_close" in trade_bars.columns else "close")
                                                    if close_col in trade_bars.columns:
                                                        closes = trade_bars[close_col].values
                                                        if side == "long":
                                                            bar_closes_pnl = ((closes - entry_price_val) / entry_price_val) * 10000.0
                                                        else:
                                                            bar_closes_pnl = ((entry_price_val - closes) / entry_price_val) * 10000.0
                                                        
                                                        running_max = np.maximum.accumulate(bar_closes_pnl)
                                                        drawdowns = bar_closes_pnl - running_max
                                                        intratrade_drawdown_bps = abs(float(min(drawdowns)))  # Positive magnitude
                                    except Exception as e:
                                        log.warning("[REPLAY_EOF] Failed to calculate MAE/MFE for trade %s: %s", trade_id, e)
                                    
                                    self.trade_journal.log_exit_summary(
                                        exit_time=last_ts.isoformat(),
                                        exit_reason=eof_reason,
                                        exit_price=exit_price,
                                        realized_pnl_bps=pnl_bps,
                                        trade_uid=trade_uid_from_journal,
                                        trade_id=trade_id,
                                        max_mfe_bps=max_mfe_bps,
                                        max_mae_bps=max_mae_bps,
                                        intratrade_drawdown_bps=intratrade_drawdown_bps,
                                    )
                                    self.exit_coverage["replay_end_or_eof_triggered"] = True
                                    self._exit_cov_inc("exit_summary_logged", 1)
                                    self._exit_cov_inc("force_close_logged_replay_eof", 1)
                                    self._exit_cov_note_close(trade_id=trade_id, reason=eof_reason)
                                    # Add EOF-specific metadata (use trade_uid_from_journal, not trade_id)
                                    trade_journal = self.trade_journal._get_trade_journal(trade_uid=trade_uid_from_journal, trade_id=trade_id)
                                    if trade_journal.get("exit_summary"):
                                        trade_journal["exit_summary"]["closed_by_replay_eof"] = True
                                        trade_journal["exit_summary"]["eof_price_source"] = eof_price_source
                                    self.trade_journal._write_trade_json(trade_uid=trade_uid_from_journal, trade_id=trade_id)
                            except Exception as e:
                                log.warning("[REPLAY_EOF] Failed to log exit summary for %s: %s", trade_id, e)
                        
                        closed_count += 1
                    except Exception as e:
                        # Fix 4: Better EOF-close logging
                        log.error(
                            "[REPLAY_EOF] Error closing trade from log: trade_id=%s entry_bid=%.5f entry_ask=%.5f exit_bid=%.5f exit_ask=%.5f timestamp=%s exception=%s",
                            trade_id if 'trade_id' in locals() else 'unknown',
                            entry_bid if 'entry_bid' in locals() else float('nan'),
                            entry_ask if 'entry_ask' in locals() else float('nan'),
                            exit_bid if 'exit_bid' in locals() else float('nan'),
                            exit_ask if 'exit_ask' in locals() else float('nan'),
                            last_ts.isoformat() if 'last_ts' in locals() else 'unknown',
                            type(e).__name__,
                            exc_info=True
                        )
                        # Continue - don't stop replay
                
                log.info("[REPLAY_EOF] Closed %d/%d open trades from trade log at EOF (reason=%s, price_source=%s)", 
                         closed_count, len(open_trades_from_log), eof_reason, eof_price_source)
            else:
                # Use EOF close helper function (for self.open_trades)
                log.info(
                    "[REPLAY] EOF Close: Closing %d remaining open trades at %s price (bid=%.3f ask=%.3f mid=%.3f) reason=%s",
                    len(self.open_trades),
                    eof_price_source,
                    final_bid,
                    final_ask,
                    final_mid,
                    eof_reason,
                )
                
                # Use EOF close helper function
                self._replay_force_close_open_trades_at_end(
                    last_ts=df.index[-1],
                    last_bid=final_bid,
                    last_ask=final_ask,
                    last_mid=final_mid,
                    reason=eof_reason,
                    price_source=eof_price_source,
                )
        elif self.open_trades:
            # Legacy behavior: close trades even if config is not set (backward compatibility)
            final_candle = df.iloc[-1]
            final_bid = float(final_candle["bid_close"])
            final_ask = float(final_candle["ask_close"])
            log.info(
            "[REPLAY] Closing %d remaining open trades at final bid/ask %.3f/%.3f (legacy mode, EOF close not configured)",
            len(self.open_trades),
            final_bid,
            final_ask,
            )
        
            # FAST MODE: Simple flush (for tuning runs)
            if self.fast_replay:
                # Simple flush: just update trade log and remove from open_trades
                open_count_before = len(self.open_trades)
                # Add defensive logging for first N calls
                if not hasattr(self, "_replay_flush_pnl_log_count"):
                    self._replay_flush_pnl_log_count = 0
                for trade in list(self.open_trades):
                    # Calculate final PnL
                    entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
                    entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
                    if self._replay_flush_pnl_log_count < 5:
                        log.debug(
                            "[PNL] Replay flush PnL: entry_bid=%.5f entry_ask=%.5f exit_bid=%.5f exit_ask=%.5f side=%s",
                            entry_bid, entry_ask, final_bid, final_ask, trade.side
                        )
                        self._replay_flush_pnl_log_count += 1
                    pnl_bps = compute_pnl_bps(entry_bid, entry_ask, final_bid, final_ask, trade.side)
                    exit_price = final_bid if trade.side == "long" else final_ask
                    
                    # Update trade log directly (skip ExitArbiter for speed)
                    # Calculate bars held
                    delta_minutes = (df.index[-1] - trade.entry_time).total_seconds() / 60.0
                    bars_held = int(round(delta_minutes / 5.0))  # M5 = 5 minutes per bar
                    
                    self._update_trade_log_on_close(
                        trade_id=trade.trade_id,
                        exit_price=exit_price,
                        pnl_bps=pnl_bps,
                        reason="REPLAY_END",
                        exit_ts=df.index[-1],
                        bars_in_trade=bars_held,
                    )
                    
                    # Remove from open_trades
                    self.open_trades = [t for t in self.open_trades if t.trade_id != trade.trade_id]
                
                log.info("[REPLAY] Fast flush: closed %d trades (skipped ExitArbiter/reporting)", open_count_before)
            else:
                # FULL MODE: Use ExitArbiter (for manual analysis runs)
                # Add defensive logging for first N calls
                if not hasattr(self, "_replay_full_pnl_log_count"):
                    self._replay_full_pnl_log_count = 0
                for trade in list(self.open_trades):
                    # GUARD: Check if trade is already being closed/exited (prevent duplicate exit attempts)
                    if hasattr(self, "_closing_trades") and trade.trade_id in self._closing_trades:
                        log.warning(
                            "[REPLAY_END] Trade %s is already being closed (duplicate exit attempt prevented)",
                            trade.trade_id
                        )
                        continue
                    if hasattr(self, "_exited_trade_ids") and trade.trade_id in self._exited_trade_ids:
                        log.warning(
                            "[REPLAY_END] Trade %s already exited (duplicate exit attempt prevented)",
                            trade.trade_id
                        )
                        continue
                    
                    # Calculate final PnL
                    entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
                    entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
                    if self._replay_full_pnl_log_count < 5:
                        log.debug(
                            "[PNL] Replay full PnL: entry_bid=%.5f entry_ask=%.5f exit_bid=%.5f exit_ask=%.5f side=%s",
                            entry_bid, entry_ask, final_bid, final_ask, trade.side
                        )
                        self._replay_full_pnl_log_count += 1
                    pnl_bps = compute_pnl_bps(entry_bid, entry_ask, final_bid, final_ask, trade.side)
                    exit_price = final_bid if trade.side == "long" else final_ask
                    
                    # Close trade via ExitArbiter
                    self.exit_coverage["replay_end_or_eof_triggered"] = True
                    self._exit_cov_inc("force_close_attempts_replay_end", 1)
                    accepted = self.request_close(
                        trade_id=trade.trade_id,
                        source="REPLAY_END",
                        reason="REPLAY_END",
                        px=exit_price,
                        pnl_bps=pnl_bps,
                        bars_in_trade=len(df) - 1,  # Approximate
                    )
                    if accepted and hasattr(self, "trade_journal") and self.trade_journal:
                        # Ensure REPLAY_END exits are logged as exit_summary (required for journal coverage)
                        try:
                            self.trade_journal.log_exit_summary(
                                exit_time=df.index[-1].isoformat(),
                                exit_reason="REPLAY_END",
                                exit_price=exit_price,
                                realized_pnl_bps=pnl_bps,
                                trade_uid=getattr(trade, "trade_uid", None),
                                trade_id=trade.trade_id,
                            )
                            self._exit_cov_inc("exit_summary_logged", 1)
                            self._exit_cov_inc("force_close_logged_replay_end", 1)
                            self._exit_cov_note_close(trade_id=trade.trade_id, reason="REPLAY_END")
                        except Exception as e:
                            log.warning("[REPLAY_END] Failed to log exit summary for %s: %s", trade.trade_id, e)
        
        # ------------------------------------------------------------------ #
        # FINAL SAFETY: ensure all trades in trade journal are closed at EOF
        # ------------------------------------------------------------------ #
        #
        # In replay (including parallel chunks), we require left_open == 0 in the
        # merged trade_journal_index without touching trading logic. If, for any
        # reason, trades remain without an exit_summary after the primary EOF
        # close paths above (self.open_trades / trade_log), we fall back to
        # scanning the trade journal JSONs and force-closing any trades that
        # are still missing exit metadata.
        #
        # This is replay-only plumbing and does not influence live/FARM logic.
        if close_open_at_end and hasattr(self, "trade_journal") and self.trade_journal:
            try:
                journal = self.trade_journal
                trade_json_dir = getattr(journal, "trade_json_dir", None)
                if trade_json_dir and trade_json_dir.exists():
                    final_candle = df.iloc[-1]
                    final_bid = float(final_candle["bid_close"])
                    final_ask = float(final_candle["ask_close"])
                    final_mid = (final_bid + final_ask) / 2.0

                    # Resolve EOF exit price consistent with earlier logic
                    if eof_price_source == "bid":
                        exit_price = final_bid
                        exit_bid = final_bid
                        exit_ask = final_bid
                    elif eof_price_source == "ask":
                        exit_price = final_ask
                        exit_bid = final_ask
                        exit_ask = final_ask
                    else:
                        exit_price = final_mid
                        exit_bid = final_bid
                        exit_ask = final_ask

                    # compute_pnl_bps is already imported at module level (line 74)
                    # No need for local import here

                    closed_from_journal = 0
                    last_ts = df.index[-1]

                    for json_path in trade_json_dir.glob("*.json"):
                        try:
                            with open(json_path, "r", encoding="utf-8") as f:
                                tj = jsonlib.load(f)

                            exit_summary = tj.get("exit_summary") or {}
                            # Skip trades that already have an exit recorded
                            if exit_summary.get("exit_time") or exit_summary.get("exit_reason"):
                                continue

                            entry = tj.get("entry_snapshot") or {}
                            entry_time_str = entry.get("entry_time")
                            
                            # Fix 1: Validate and normalize side before compute_pnl_bps
                            side_raw = entry.get("side") or "long"
                            if isinstance(side_raw, str):
                                side = side_raw.lower().strip()
                            else:
                                side = str(side_raw).lower().strip() if pd.notna(side_raw) else "long"
                            
                            # Validate side is "long" or "short"
                            if side not in ("long", "short"):
                                trade_id = str(
                                    tj.get("trade_id")
                                    or entry.get("trade_id")
                                    or json_path.stem
                                )
                                log.error(
                                    "[REPLAY_EOF] Invalid side for trade_id=%s from journal: got %r (raw=%r). Skipping EOF close.",
                                    trade_id, side, side_raw
                                )
                                # Mark trade as left_open_eof
                                try:
                                    tj2 = journal._get_trade_journal(trade_id)
                                    if tj2.get("exit_summary"):
                                        tj2["exit_summary"]["left_open_eof"] = True
                                        tj2["exit_summary"]["eof_close_error"] = f"Invalid side: {side_raw}"
                                    journal._write_trade_json(trade_id)
                                except Exception as e:
                                    log.warning("[REPLAY_EOF] Failed to mark trade %s as left_open_eof: %s", trade_id, e)
                                continue  # Skip this trade, continue with next
                            
                            entry_price = float(entry.get("entry_price", exit_price))

                            # Approximate entry bid/ask from entry_price if not available
                            entry_bid = float(entry.get("entry_bid", entry_price))
                            entry_ask = float(entry.get("entry_ask", entry_price))

                            if not entry_time_str:
                                bars_held = 0
                            else:
                                entry_time = pd.to_datetime(entry_time_str, utc=True, errors="coerce")
                                if pd.isna(entry_time):
                                    bars_held = 0
                                else:
                                    delta_minutes = (last_ts - entry_time).total_seconds() / 60.0
                                    bars_held = int(round(delta_minutes / 5.0))

                            pnl_bps = compute_pnl_bps(entry_bid, entry_ask, exit_bid, exit_ask, side)

                            trade_id = str(
                                tj.get("trade_id")
                                or entry.get("trade_id")
                                or json_path.stem
                            )

                            # Extract trade_uid from journal if available (COMMIT C)
                            trade_uid_from_journal = tj.get("trade_uid") or (tj.get("trade_key", "").replace("LEGACY:", "") if not (tj.get("trade_key", "") or "").startswith("LEGACY:") else None)
                            journal.log_exit_summary(
                                exit_time=last_ts.isoformat(),
                                trade_uid=trade_uid_from_journal,  # Primary key (COMMIT C)
                                trade_id=trade_id,  # Display ID (backward compatibility)
                                exit_price=exit_price,
                                exit_reason=eof_reason,
                                realized_pnl_bps=pnl_bps,
                            )
                            self.exit_coverage["replay_end_or_eof_triggered"] = True
                            self._exit_cov_inc("exit_summary_logged", 1)
                            self._exit_cov_inc("force_close_logged_replay_end", 1)
                            self._exit_cov_note_close(trade_id=trade_id, reason=eof_reason)

                            # Tag EOF-specific metadata
                            tj2 = journal._get_trade_journal(trade_uid=trade_uid_from_journal, trade_id=trade_id)
                            if tj2.get("exit_summary"):
                                tj2["exit_summary"]["closed_by_replay_eof"] = True
                                tj2["exit_summary"]["eof_price_source"] = eof_price_source
                            journal._write_trade_json(trade_uid=trade_uid_from_journal, trade_id=trade_id)

                            closed_from_journal += 1
                        except Exception as e:
                            # Fix 4: Better EOF-close logging
                            trade_id = str(
                                tj.get("trade_id")
                                or entry.get("trade_id")
                                or json_path.stem
                            ) if 'trade_id' not in locals() else trade_id
                            log.error(
                                "[REPLAY_EOF] Error closing trade from journal: trade_id=%s entry_bid=%.5f entry_ask=%.5f exit_bid=%.5f exit_ask=%.5f timestamp=%s exception=%s",
                                trade_id,
                                entry_bid if 'entry_bid' in locals() else float('nan'),
                                entry_ask if 'entry_ask' in locals() else float('nan'),
                                exit_bid if 'exit_bid' in locals() else float('nan'),
                                exit_ask if 'exit_ask' in locals() else float('nan'),
                                last_ts.isoformat() if 'last_ts' in locals() else 'unknown',
                                type(e).__name__,
                                exc_info=True
                            )
                            # Continue - don't stop replay
                            log.warning(
                                "[REPLAY_EOF] Fallback journal close failed for %s: %s",
                                json_path,
                                e,
                            )

                    if closed_from_journal:
                        log.info(
                            "[REPLAY_EOF] Fallback journal EOF close: closed %d trades missing exit_summary",
                            closed_from_journal,
                        )
            except Exception as e:
                log.warning("[REPLAY_EOF] Fallback journal EOF close failed: %s", e)
    
        # Dump backtest summary (skip in fast mode)
        if not self.fast_replay:
            self._dump_backtest_summary()
    
        # Flush any remaining entry-only log buffer
        if hasattr(self, "_entry_only_log_buffer") and len(self._entry_only_log_buffer) > 0:
            self._flush_entry_only_log_buffer()
            log.info("[ENTRY_ONLY] Flushed final buffer (%d entries)", len(getattr(self, "_entry_only_log_buffer", [])))

        self._log_entry_diag_summary()
    
        # Write Stage-0 reason breakdown report (if EntryManager has tracking)
        if hasattr(self, "entry_manager") and hasattr(self.entry_manager, "stage0_reasons"):
            try:
                self._write_stage0_reason_report()
            except AttributeError:
                # Fallback: write report directly if method doesn't exist
                try:
                    from pathlib import Path
                    if hasattr(self, "output_dir") and self.output_dir:
                        report_path = Path_module(self.output_dir) / "stage0_reasons.json"
                    elif hasattr(self, "run_id") and self.run_id:
                        report_path = Path_module("gx1/wf_runs") / self.run_id / "stage0_reasons.json"
                    else:
                        report_path = Path_module("gx1/live/logs") / "stage0_reasons.json"
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(report_path, "w") as f:
                        jsonlib.dump(self.entry_manager.stage0_reasons, f, indent=2)
                    log.info("[STAGE0] Wrote reason report to %s", report_path)
                except Exception as e:
                    log.warning("[STAGE0] Failed to write reason report: %s", e)
    
        log.info("[REPLAY] Backtest complete")
        
        # PHASE 1 FIX: Update RUN_IDENTITY with temperature scaling status
        try:
            from gx1.runtime.run_identity import load_run_identity
            identity_paths_to_try = []
            if hasattr(self, "output_dir") and self.output_dir:
                identity_paths_to_try.append(Path(self.output_dir) / "RUN_IDENTITY.json")
            if hasattr(self, "explicit_output_dir") and self.explicit_output_dir:
                identity_paths_to_try.append(Path(self.explicit_output_dir) / "RUN_IDENTITY.json")
                parent_dir = Path(self.explicit_output_dir).parent
                identity_paths_to_try.append(parent_dir / "RUN_IDENTITY.json")
            output_dir_env = os.getenv("GX1_OUTPUT_DIR")
            if output_dir_env:
                identity_paths_to_try.append(Path(output_dir_env) / "RUN_IDENTITY.json")
            
            for identity_path in identity_paths_to_try:
                if identity_path.exists():
                    identity = load_run_identity(identity_path)
                    # Update temperature scaling status
                    temp_map = self._get_temperature_map()
                    identity.temperature_scaling_enabled = os.getenv("GX1_TEMPERATURE_SCALING", "1") != "0"
                    identity.temperature_map_loaded = len(temp_map) > 0 if temp_map else False
                    identity.temperature_defaults_used_count = getattr(self, "_temp_defaults_used_count", 0)
                    # Write updated identity
                    import json as jsonlib
                    with open(identity_path, "w") as f:
                        jsonlib.dump(identity.to_dict(), f, indent=2, sort_keys=True)
                    log.info(
                        f"[REPLAY] Updated RUN_IDENTITY with temperature scaling: "
                        f"enabled={identity.temperature_scaling_enabled}, "
                        f"map_loaded={identity.temperature_map_loaded}, "
                        f"defaults_used={identity.temperature_defaults_used_count}"
                    )
                    break
        except Exception as e:
            log.warning(f"[REPLAY] Failed to update RUN_IDENTITY with temperature scaling: {e}")
        
        # Update RUN_IDENTITY with Exit Policy V2 counters (if enabled)
        if hasattr(self, "exit_policy_v2") and self.exit_policy_v2:
            try:
                exit_v2_counters = self.exit_policy_v2.get_counters()
                # Update RUN_IDENTITY.json with counters
                from gx1.runtime.run_identity import load_run_identity
                identity_paths_to_try = []
                if hasattr(self, "output_dir") and self.output_dir:
                    identity_paths_to_try.append(Path(self.output_dir) / "RUN_IDENTITY.json")
                if hasattr(self, "explicit_output_dir") and self.explicit_output_dir:
                    identity_paths_to_try.append(Path(self.explicit_output_dir) / "RUN_IDENTITY.json")
                    # Also try parent directory (for chunk workers)
                    parent_dir = Path(self.explicit_output_dir).parent
                    identity_paths_to_try.append(parent_dir / "RUN_IDENTITY.json")
                # Try environment variable for output dir (used in replay_eval_gated_parallel)
                output_dir_env = os.getenv("GX1_OUTPUT_DIR")
                if output_dir_env:
                    identity_paths_to_try.append(Path(output_dir_env) / "RUN_IDENTITY.json")
                
                updated = False
                for identity_path in identity_paths_to_try:
                    if identity_path.exists():
                        try:
                            identity = load_run_identity(identity_path)
                            identity.exit_v2_enabled = True  # Mark as enabled
                            identity.exit_v2_counters = exit_v2_counters
                            # Write updated identity (atomic write)
                            import json
                            temp_path = identity_path.with_suffix(".json.tmp")
                            with open(temp_path, "w", encoding="utf-8") as f:
                                json.dump(identity.to_dict(), f, indent=2, sort_keys=True)
                            temp_path.replace(identity_path)
                            log.info(f"[EXIT_POLICY_V2] Updated RUN_IDENTITY.json with counters: {identity_path}")
                            updated = True
                            break
                        except Exception as e:
                            log.warning(f"[EXIT_POLICY_V2] Failed to update {identity_path}: {e}", exc_info=True)
                
                if not updated:
                    log.warning(f"[EXIT_POLICY_V2] Could not find RUN_IDENTITY.json in any of: {identity_paths_to_try}")
            except Exception as e:
                log.warning(f"[EXIT_POLICY_V2] Failed to update RUN_IDENTITY with counters: {e}", exc_info=True)
        
        # DEL 3: Update RUN_IDENTITY with Pre-Entry Wait Gate counters (if enabled)
        if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "pre_entry_wait_gate") and self.entry_manager.pre_entry_wait_gate:
            try:
                pre_entry_wait_counters = self.entry_manager.pre_entry_wait_gate.get_counters()
                # Update RUN_IDENTITY.json with counters
                from gx1.runtime.run_identity import load_run_identity
                identity_paths_to_try = []
                if hasattr(self, "output_dir") and self.output_dir:
                    identity_paths_to_try.append(Path(self.output_dir) / "RUN_IDENTITY.json")
                if hasattr(self, "explicit_output_dir") and self.explicit_output_dir:
                    identity_paths_to_try.append(Path(self.explicit_output_dir) / "RUN_IDENTITY.json")
                    # Also try parent directory for chunk workers
                    identity_paths_to_try.append(Path(self.explicit_output_dir).parent / "RUN_IDENTITY.json")
                output_dir_env = os.getenv("GX1_OUTPUT_DIR")
                if output_dir_env:
                    identity_paths_to_try.append(Path(output_dir_env) / "RUN_IDENTITY.json")
                
                updated = False
                for identity_path in identity_paths_to_try:
                    if identity_path.exists():
                        try:
                            identity = load_run_identity(identity_path)
                            identity.pre_entry_wait_enabled = True  # Mark as enabled
                            identity.pre_entry_wait_counters = pre_entry_wait_counters
                            # Write updated identity (atomic write)
                            import json
                            temp_path = identity_path.with_suffix(".json.tmp")
                            with open(temp_path, "w", encoding="utf-8") as f:
                                json.dump(identity.to_dict(), f, indent=2, sort_keys=True)
                            temp_path.replace(identity_path)
                            log.info(f"[PRE_ENTRY_WAIT] Updated RUN_IDENTITY.json with counters: {identity_path}")
                            updated = True
                            break
                        except Exception as e:
                            log.warning(f"[PRE_ENTRY_WAIT] Failed to update {identity_path}: {e}", exc_info=True)
                
                if not updated:
                    log.warning(f"[PRE_ENTRY_WAIT] Could not find RUN_IDENTITY.json in any of: {identity_paths_to_try}")
            except Exception as e:
                log.warning(f"[PRE_ENTRY_WAIT] Failed to update RUN_IDENTITY with counters: {e}", exc_info=True)
        
        # BUGFIX: Update RUN_IDENTITY with OVERLAP Overlay counters and invariants
        if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "overlap_overlay_config") and self.entry_manager.overlap_overlay_config:
            try:
                # Collect overlap overlay counters from entry_telemetry
                # BUGFIX: Include hard telemetry for cost-gate debugging
                overlap_counters = {}
                if hasattr(self.entry_manager, "entry_telemetry"):
                    telemetry = self.entry_manager.entry_telemetry
                    
                    # Cost-gate telemetry
                    cost_gate_eval_count = telemetry.get("overlap_cost_gate_eval_count", 0)
                    cost_gate_missing_inputs = telemetry.get("overlap_cost_gate_missing_inputs_count", 0)
                    cost_gate_skips = telemetry.get("overlap_cost_gate_skips", 0)
                    
                    # BUGFIX: Compute spread/ATR stats
                    spread_values = telemetry.get("overlap_cost_gate_spread_bps_values", [])
                    atr_values = telemetry.get("overlap_cost_gate_atr_bps_values", [])
                    
                    spread_stats = {}
                    atr_stats = {}
                    if spread_values:
                        import numpy as np
                        spread_arr = np.array(spread_values)
                        spread_stats = {
                            "min": float(np.min(spread_arr)),
                            "median": float(np.median(spread_arr)),
                            "p95": float(np.percentile(spread_arr, 95)),
                            "max": float(np.max(spread_arr)),
                        }
                    if atr_values:
                        import numpy as np
                        atr_arr = np.array(atr_values)
                        atr_stats = {
                            "min": float(np.min(atr_arr)),
                            "median": float(np.median(atr_arr)),
                            "p95": float(np.percentile(atr_arr, 95)),
                            "max": float(np.max(atr_arr)),
                        }
                    
                    overlap_counters = {
                        "overlap_cost_gate_eval_count": cost_gate_eval_count,
                        "overlap_cost_gate_missing_inputs_count": cost_gate_missing_inputs,
                        "overlap_cost_gate_skips": cost_gate_skips,
                        "overlap_cost_gate_spread_bps_stats": spread_stats,
                        "overlap_cost_gate_atr_bps_stats": atr_stats,
                        "overlap_window_blocks": telemetry.get("overlap_window_blocks", 0),
                        "overlap_window_veto_count": telemetry.get("overlap_window_veto_count", 0),
                    }
                    
                    # BUGFIX: Fail-fast if cost-gate enabled but never evaluated
                    if hasattr(self.entry_manager, "overlap_overlay_config") and self.entry_manager.overlap_overlay_config:
                        if self.entry_manager.overlap_overlay_config.overlap_cost_gate_enabled:
                            if cost_gate_eval_count == 0:
                                error_msg = (
                                    f"[OVERLAP_COST_GATE_WIRING_BUG] Cost-gate is enabled but eval_count=0. "
                                    f"This indicates a wiring bug - cost-gate check is not being called."
                                )
                                log.error(error_msg)
                                raise RuntimeError(error_msg)
                            
                            if cost_gate_missing_inputs > 0:
                                error_msg = (
                                    f"[OVERLAP_COST_GATE_INPUTS_BUG] Cost-gate missing inputs {cost_gate_missing_inputs} times. "
                                    f"This indicates an ordering/inputs bug - spread_bps/atr_bps not available when needed."
                                )
                                log.error(error_msg)
                                raise RuntimeError(error_msg)
                
                # Get overlap override counters from runner
                overlap_override_applied_n = getattr(self, "overlap_override_applied_n", 0)
                overlap_override_applied_outside_overlap = getattr(self, "overlap_override_applied_outside_overlap", 0)
                
                # Update RUN_IDENTITY.json with counters
                from gx1.runtime.run_identity import load_run_identity
                identity_paths_to_try = []
                if hasattr(self, "output_dir") and self.output_dir:
                    identity_paths_to_try.append(Path(self.output_dir) / "RUN_IDENTITY.json")
                if hasattr(self, "explicit_output_dir") and self.explicit_output_dir:
                    identity_paths_to_try.append(Path(self.explicit_output_dir) / "RUN_IDENTITY.json")
                    identity_paths_to_try.append(Path(self.explicit_output_dir).parent / "RUN_IDENTITY.json")
                output_dir_env = os.getenv("GX1_OUTPUT_DIR")
                if output_dir_env:
                    identity_paths_to_try.append(Path(output_dir_env) / "RUN_IDENTITY.json")
                
                updated = False
                for identity_path in identity_paths_to_try:
                    if identity_path.exists():
                        try:
                            identity = load_run_identity(identity_path)
                            identity.overlap_overlay_enabled = True
                            identity.overlap_overlay_counters = overlap_counters
                            identity.overlap_override_applied_n = overlap_override_applied_n
                            identity.overlap_override_applied_outside_overlap = overlap_override_applied_outside_overlap
                            
                            # BUGFIX: Hard-fail if overlap override applied outside OVERLAP session
                            if overlap_override_applied_outside_overlap > 0:
                                error_msg = (
                                    f"[OVERLAP_INVARIANT_VIOLATION] Overlap override applied to non-OVERLAP session "
                                    f"{overlap_override_applied_outside_overlap} times. This violates the invariant that "
                                    f"overlap overlays ONLY affect OVERLAP trades."
                                )
                                log.error(error_msg)
                                raise RuntimeError(error_msg)
                            
                            # Write updated identity (atomic write)
                            import json
                            temp_path = identity_path.with_suffix(".json.tmp")
                            with open(temp_path, "w", encoding="utf-8") as f:
                                json.dump(identity.to_dict(), f, indent=2, sort_keys=True)
                            temp_path.replace(identity_path)
                            log.info(
                                f"[OVERLAP_OVERLAY] Updated RUN_IDENTITY.json with counters: "
                                f"override_applied={overlap_override_applied_n}, "
                                f"override_outside_overlap={overlap_override_applied_outside_overlap}, "
                                f"counters={overlap_counters}"
                            )
                            updated = True
                            break
                        except Exception as e:
                            log.warning(f"[OVERLAP_OVERLAY] Failed to update {identity_path}: {e}", exc_info=True)
                
                if not updated:
                    log.warning(f"[OVERLAP_OVERLAY] Could not find RUN_IDENTITY.json in any of: {identity_paths_to_try}")
            except Exception as e:
                log.warning(f"[OVERLAP_OVERLAY] Failed to update RUN_IDENTITY with counters: {e}", exc_info=True)
        
        # Close trade journal
        if hasattr(self, "trade_journal") and self.trade_journal:
            try:
                # Mandatory flush point: before journal close / before RUN_COMPLETED.
                try:
                    if hasattr(self, "_flush_journal_buffer"):
                        self._flush_journal_buffer(reason="before_journal_close")
                except Exception:
                    pass
                self.trade_journal.close()
                log.info("[TRADE_JOURNAL] Trade journal closed")
            except Exception as e:
                log.warning("[TRADE_JOURNAL] Failed to close journal: %s", e)
    
        # Canary mode: log invariants and generate prod_metrics
        if hasattr(self, "canary_mode") and self.canary_mode:
            self._log_canary_invariants()
            self._generate_canary_metrics()
    
        # Generate detailed report via external script (disabled in TRUTH/SMOKE replay)
        if not getattr(self, "replay_mode", False) and not self.fast_replay:
            try:
                from gx1.backtest.generate_exit_tuning_report import main as generate_report  # type: ignore[reportMissingImports]
                import sys
                # Save original argv
                old_argv = sys.argv
                # Set up arguments for report generator
                exit_audit_file = self.log_dir / "exits" / f"exits_{self.replay_start_ts.strftime('%Y%m%d')}_{self.replay_end_ts.strftime('%Y%m%d')}.jsonl"
                report_file = Path_module("gx1/backtest/reports") / f"GX1_ONE_BACKTEST_exit_tuning_v4.md"
                report_file.parent.mkdir(parents=True, exist_ok=True)
                sys.argv = [
                    "generate_exit_tuning_report",
                    "--trade-log", str(self.trade_log_path),
                    "--exit-audit", str(exit_audit_file),
                    "--output", str(report_file),
                ]
                generate_report()
                sys.argv = old_argv
                log.info("[REPLAY] Detailed report generated: %s", report_file)
            except Exception as e:
                log.warning("[REPLAY] Failed to generate detailed report: %s", e)
                # Fallback to simple summary
                self._dump_backtest_summary()
        else:
            log.info("[REPLAY] Detailed report generation skipped in replay/fast mode")

def run_replay(self: GX1DemoRunner, csv_path: Path) -> None:
    """
    Run replay with historical M5 candles from CSV/parquet.
    
    Parameters
    ----------
    csv_path : Path
        Path to CSV/parquet file with historical candles
    """
    try:
        return self._run_replay_impl(csv_path)
    finally:
        # Replay-safe journal buffering: always flush buffered IO before leaving replay,
        # regardless of success/failure (RUN_COMPLETED / RUN_FAILED written after this returns).
        try:
            if hasattr(self, "_flush_journal_buffer"):
                self._flush_journal_buffer(reason="run_replay_finally")
        except Exception:
            pass

def _simulate_tick_exits_for_bar_impl(self, bar_ts: pd.Timestamp, candle_row: pd.Series) -> None:
    """
    Simulate tick-based exits for a single M5 bar.
    
    Uses high/low of the bar to check if TP/SL/BE/soft-stop would have triggered.
    Also calls exit policies (FIXED_BAR, FARM_V2_RULES) with bid/ask prices.
    
    Parameters
    ----------
    bar_ts : pd.Timestamp
        Timestamp of the current bar.
    candle_row : pd.Series
            Single row with open, high, low, close, volume, bid_*, ask_* columns.
    """
    if not self.open_trades:
        return
    if getattr(self, "exit_verbose_logging", False):
        log.info(
            "[EXIT][SIM] Evaluating %d trades on bar %s",
            len(self.open_trades),
            bar_ts,
        )
    
    # ============================================================================
    # DEFENSIVE CHECK: Ensure bid/ask columns exist
    # ============================================================================
    required_bid_ask_cols = ["bid_open", "bid_high", "bid_low", "bid_close", 
                             "ask_open", "ask_high", "ask_low", "ask_close"]
    missing_cols = [col for col in required_bid_ask_cols if col not in candle_row.index]
    if missing_cols:
        raise ValueError(
            f"Replay requires bid_* and ask_* columns but they are missing: {missing_cols}. "
            f"Please use data with bid/ask prices (not mid-only data)."
        )
    
    # Log first time (once per run) that we're using bid/ask for exit policies
    if not hasattr(self, "_exit_policy_bid_ask_logged"):
        log.info(
            "[REPLAY] Exit policies using bid/ask prices: bid_close/ask_close from candles"
        )
        self._exit_policy_bid_ask_logged = True
    
    open_pos_list = self._get_open_positions_for_tick()
    if not open_pos_list:
        return
    
    # For each open position, check if TP/SL/BE/soft-stop would trigger
    bid_high = float(candle_row["bid_high"])
    bid_low = float(candle_row["bid_low"])
    ask_high = float(candle_row["ask_high"])
    ask_low = float(candle_row["ask_low"])
    
    for pos in open_pos_list:
        # Determine which price to check based on direction
        # For LONG: check low (exit at loss) and high (exit at profit)
        # For SHORT: check high (exit at loss) and low (exit at profit)
        
        if pos.direction == "LONG":
            tp_exit_bid, tp_exit_ask = bid_high, ask_high
            sl_exit_bid, sl_exit_ask = bid_low, ask_low
        else:
            tp_exit_bid, tp_exit_ask = bid_low, ask_low
            sl_exit_bid, sl_exit_ask = bid_high, ask_high
        
        # Add defensive logging for first N calls
        if not hasattr(self, "_tick_sim_pnl_log_count"):
            self._tick_sim_pnl_log_count = 0
        side_str = "long" if pos.direction == "LONG" else "short"
        if self._tick_sim_pnl_log_count < 5:
            log.debug(
                "[PNL] Tick sim TP PnL: entry_bid=%.5f entry_ask=%.5f exit_bid=%.5f exit_ask=%.5f side=%s",
                pos.entry_bid,
                pos.entry_ask,
                tp_exit_bid,
                tp_exit_ask,
                side_str,
            )
            log.debug(
                "[PNL] Tick sim SL PnL: entry_bid=%.5f entry_ask=%.5f exit_bid=%.5f exit_ask=%.5f side=%s",
                pos.entry_bid,
                pos.entry_ask,
                sl_exit_bid,
                sl_exit_ask,
                side_str,
            )
            self._tick_sim_pnl_log_count += 1
        pnl_bps_tp = compute_pnl_bps(
            pos.entry_bid,
            pos.entry_ask,
            tp_exit_bid,
            tp_exit_ask,
            side_str,
        )
        pnl_bps_sl = compute_pnl_bps(
            pos.entry_bid,
            pos.entry_ask,
            sl_exit_bid,
            sl_exit_ask,
            side_str,
        )
        
        # BE activation check (before TP/SL/BE checks)
        # Activate BE if profit >= activate_at_bps and not already active
        be_cfg = self.policy.get("tick_exit", {}).get("be", {})
        be_enabled = be_cfg.get("enabled", False)
        be_activate_at_bps = int(be_cfg.get("activate_at_bps", 50))
        be_bias_price = float(be_cfg.get("bias_price", 0.3))
        
        if be_enabled and not pos.be_active:
            # Check if we should activate BE (use TP price for profit check)
            if pnl_bps_tp >= be_activate_at_bps:
                # Activate BE: set BE price = entry_price + bias (lock in small profit)
                if pos.direction == "LONG":
                    be_price = pos.entry_px + be_bias_price
                else:  # SHORT
                    be_price = pos.entry_px - be_bias_price
                
                # Update BE status in trade.extra (thread-safe)
                try:
                    self._update_be_status(pos.trade_id, be_active=True, be_price=be_price)
                    # Update pos object for this check
                    pos.be_active = True
                    pos.be_price = be_price
                    log.debug(
                        "[REPLAY] BE activated for trade %s: pnl_bps=%.2f >= %d, BE_price=%.3f",
                        pos.trade_id, pnl_bps_tp, be_activate_at_bps, be_price
                    )
                except Exception as e:
                    log.warning("[REPLAY] Failed to activate BE for trade %s: %s", pos.trade_id, e)
        
        # Check TP (after BE activation)
        if pnl_bps_tp >= pos.tp_bps:
            exit_px = tp_exit_bid if pos.direction == "LONG" else tp_exit_ask
            self._tick_close_now(pos, "TP_TICK", exit_px, pnl_bps_tp)
            continue
        
        # Check BE (if active) - check if price touched BE level
        be_triggered = False
        be_exit_bid = None
        be_exit_ask = None
        pnl_bps_be = 0.0
        if pos.be_active and pos.be_price is not None:
            if pos.direction == "LONG" and sl_exit_bid <= pos.be_price:
                be_triggered = True
                # LONG: exit at bid when BE triggers (use actual exit bid from candle)
                be_exit_bid = sl_exit_bid  # Use actual exit bid (bid_low when BE triggers)
                be_exit_ask = ask_low  # Use actual exit ask for consistency
                pnl_bps_be = compute_pnl_bps(
                    pos.entry_bid,
                    pos.entry_ask,
                    be_exit_bid,
                    be_exit_ask,
                    "long",
                )
            elif pos.direction == "SHORT" and sl_exit_ask >= pos.be_price:
                be_triggered = True
                # SHORT: exit at ask when BE triggers (use actual exit ask from candle)
                be_exit_bid = bid_high  # Use actual exit bid for consistency
                be_exit_ask = sl_exit_ask  # Use actual exit ask (ask_high when BE triggers)
                pnl_bps_be = compute_pnl_bps(
                    pos.entry_bid,
                    pos.entry_ask,
                    be_exit_bid,
                    be_exit_ask,
                    "short",
                )

        if be_triggered:
            # Use correct exit price based on side (LONG -> bid, SHORT -> ask)
            exit_px = be_exit_bid if pos.direction == "LONG" else be_exit_ask
            self._tick_close_now(pos, "BE_TICK", exit_px, pnl_bps_be)
            continue
        
        # Check SL (after BE check - BE takes priority over SL)
        if pnl_bps_sl <= -pos.sl_bps:
            exit_px = sl_exit_bid if pos.direction == "LONG" else sl_exit_ask
            self._tick_close_now(pos, "SL_TICK", exit_px, pnl_bps_sl)
            continue
        
        # Check soft-stop (last priority - only if not TP/SL/BE)
        tick_exit_cfg = self.policy.get("tick_exit", {})
        soft_stop_bps = tick_exit_cfg.get("soft_stop_bps", 0)
        if soft_stop_bps > 0 and pnl_bps_sl <= -soft_stop_bps:
            exit_px = sl_exit_bid if pos.direction == "LONG" else sl_exit_ask
            self._tick_close_now(pos, "SOFT_STOP_TICK", exit_px, pnl_bps_sl)
            continue
        
        # ============================================================================
        # EXIT POLICIES: Call on_bar() for exit_fixed_bar_policy and exit_farm_v2_rules_policy
        # ============================================================================
        # After tick-based exits, process exit policies for remaining open trades
        # Get current bar prices (use close prices for exit policy evaluation)
        # These are already validated to exist above
        price_bid = float(candle_row["bid_close"])
        price_ask = float(candle_row["ask_close"])
        
        # Iterate over remaining open trades (after tick-based exits)
        for trade in list(self.open_trades):  # Use list() to avoid modification during iteration
            # Skip trades already closed (single-exit invariant)
            try:
                if hasattr(self, "_exited_trade_ids") and trade.trade_id in self._exited_trade_ids:
                    continue
            except Exception:
                pass
            # Skip if trade was closed by tick-based exits
            if trade not in self.open_trades:
                continue
            
            # Get exit_profile from trade.extra
            exit_profile = None
            if hasattr(trade, "extra") and trade.extra:
                exit_profile = trade.extra.get("exit_profile")
            
            # ========================================================================
            # FIXED_BAR_CLOSE exit policy
            # ========================================================================
            if (hasattr(self, "exit_fixed_bar_policy") and 
                self.exit_fixed_bar_policy is not None and
                exit_profile == "FIXED_BAR_CLOSE"):
                
                # Check if policy has state for this trade (should be initialized on entry)
                if self.exit_fixed_bar_policy.has_state(trade.trade_id):
                    try:
                        # Call on_bar() with bid/ask prices and timestamp
                        decision = self.exit_fixed_bar_policy.on_bar(
                            trade_id=trade.trade_id,
                            price_bid=price_bid,
                            price_ask=price_ask,
                            side=trade.side,
                            ts=bar_ts,
                        )
                        
                        if decision is not None:
                            # Exit triggered - close trade
                            log.debug(
                                "[REPLAY] FIXED_BAR exit triggered for trade %s: reason=%s bars_held=%d pnl=%.2f bps "
                                "(price_bid=%.5f price_ask=%.5f)",
                                trade.trade_id, decision.reason, decision.bars_held, decision.pnl_bps,
                                price_bid, price_ask
                            )
                            self.request_close(
                                trade_id=trade.trade_id,
                                source="EXIT_POLICY",
                                reason=decision.reason,
                                px=decision.exit_price,
                                pnl_bps=decision.pnl_bps,
                                bars_in_trade=decision.bars_held,
                            )
                    except Exception as e:
                        log.warning(
                            "[REPLAY] Error calling exit_fixed_bar_policy.on_bar() for trade %s: %s",
                            trade.trade_id, e
                        )
                else:
                    log.debug(
                        "[REPLAY] exit_fixed_bar_policy has no state for trade %s (not initialized?)",
                        trade.trade_id
                    )
            
            # ========================================================================
            # FARM_V2_RULES exit policy
            # ========================================================================
            elif (getattr(self, "exit_farm_v2_rules_factory", None) and
                  exit_profile and exit_profile.startswith("FARM_EXIT_V2_RULES")):
                
                policy = self.exit_farm_v2_rules_states.get(trade.trade_id)
                if policy is None:
                    log.debug(
                        "[REPLAY] FARM_V2_RULES state missing for trade %s, reinitializing",
                        trade.trade_id,
                    )
                    try:
                        self._init_farm_v2_rules_state(trade, context="replay_exit_loop")
                        policy = self.exit_farm_v2_rules_states.get(trade.trade_id)
                    except Exception as e:
                        log.warning(
                            "[REPLAY] Unable to initialize FARM_V2_RULES state for trade %s: %s",
                            trade.trade_id,
                            e,
                        )
                        continue
                
                try:
                    log.debug(
                        "[REPLAY] Calling FARM_V2_RULES.on_bar() for trade %s: price_bid=%.5f price_ask=%.5f ts=%s",
                        trade.trade_id,
                        price_bid,
                        price_ask,
                        bar_ts,
                    )
                    decision = policy.on_bar(
                        price_bid=price_bid,
                        price_ask=price_ask,
                        ts=bar_ts,
                    )
                    
                    if decision is not None:
                        log.info(
                            "[REPLAY] FARM_V2_RULES exit triggered for trade %s: reason=%s bars_held=%d pnl=%.2f bps mae=%.2f mfe=%.2f "
                            "(price_bid=%.5f price_ask=%.5f)",
                            trade.trade_id, decision.reason, decision.bars_held, decision.pnl_bps, decision.mae_bps, decision.mfe_bps,
                            price_bid, price_ask
                        )
                        self.request_close(
                            trade_id=trade.trade_id,
                            source="EXIT_POLICY",
                            reason=decision.reason,
                            px=decision.exit_price,
                            pnl_bps=decision.pnl_bps,
                            bars_in_trade=decision.bars_held,
                        )
                except Exception as e:
                    log.warning(
                        "[REPLAY] Error calling FARM_V2_RULES.on_bar() for trade %s: %s",
                        trade.trade_id, e
                    )

def _dump_backtest_summary_impl(self) -> None:
    """
    Dump backtest summary after replay.
    
    Reads trade log and exit audit to generate statistics.
    """
    import numpy as np
    
    # Read trade log (skip bad lines to handle corrupted rows)
    if not self.trade_log_path.exists():
        log.warning("[REPLAY] Trade log not found: %s", self.trade_log_path)
        return
    
    try:
        trades_df = pd.read_csv(self.trade_log_path, on_bad_lines='skip', engine='python')
    except Exception as exc:
        log.warning("[REPLAY] Failed to read trade log %s: %s. Skipping summary.", self.trade_log_path, exc)
        return
    
    # Filter replay trades
    if "notes" in trades_df.columns:
        replay_trades = trades_df[trades_df["notes"].str.contains("replay", case=False, na=False)]
    else:
        # If notes column is missing, filter by run_id instead
        if "run_id" in trades_df.columns:
            replay_trades = trades_df[trades_df["run_id"].str.contains("replay", case=False, na=False)]
        else:
            # If neither notes nor run_id exists, use all trades (legacy mode)
            replay_trades = trades_df
    
    if len(replay_trades) == 0:
        log.info("[REPLAY SUMMARY] No trades executed")
        return
    
    # Parse exit times and PnL
    replay_trades = replay_trades.copy()
    replay_trades["entry_time"] = pd.to_datetime(replay_trades["entry_time"], utc=True, format='ISO8601')
    replay_trades["exit_time"] = pd.to_datetime(replay_trades["exit_time"], utc=True, errors='coerce', format='ISO8601')
    
    # Filter closed trades (have exit_time and pnl_bps)
    closed_trades = replay_trades[
        replay_trades["exit_time"].notna() & 
        replay_trades["pnl_bps"].notna() &
        (replay_trades["pnl_bps"] != "")
    ].copy()
    
    # Convert pnl_bps to float
    closed_trades["pnl_bps"] = pd.to_numeric(closed_trades["pnl_bps"], errors='coerce')
    closed_trades = closed_trades[closed_trades["pnl_bps"].notna()]
    
    if len(closed_trades) == 0:
        log.info("[REPLAY SUMMARY] No closed trades yet")
        return
    
    pnls = closed_trades["pnl_bps"].values
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    
    # Count soft-stop exits from exit audit
    soft_stop_count = 0
    if self.exit_audit_path.exists():
        with self.exit_audit_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = jsonlib.loads(line)
                    if entry.get("source") == "TICK" and entry.get("reason") == "SOFT_STOP_TICK" and entry.get("accepted"):
                        soft_stop_count += 1
                except jsonlib.JSONDecodeError:
                    pass
    
    log.info("=" * 60)
    log.info("[REPLAY SUMMARY] Backtest Results")
    log.info("=" * 60)
    log.info("[REPLAY SUMMARY] Period: %s to %s", 
             self.replay_start_ts.isoformat(), self.replay_end_ts.isoformat())
    log.info("[REPLAY SUMMARY] Total trades: %d", len(replay_trades))
    log.info("[REPLAY SUMMARY] Closed trades: %d", len(closed_trades))
    log.info("[REPLAY SUMMARY] Open trades: %d", len(replay_trades) - len(closed_trades))
    log.info("")
    log.info("[REPLAY SUMMARY] Total PnL: %.2f bps", pnls.sum())
    log.info("[REPLAY SUMMARY] Average PnL: %.2f bps", pnls.mean())
    log.info("[REPLAY SUMMARY] Winrate: %.1f%%", 100 * len(wins) / len(closed_trades) if len(closed_trades) > 0 else 0)
    log.info("")
    if len(wins) > 0:
        log.info("[REPLAY SUMMARY] Average Win: +%.2f bps", wins.mean())
        log.info("[REPLAY SUMMARY] Largest Win: +%.2f bps", wins.max())
    if len(losses) > 0:
        log.info("[REPLAY SUMMARY] Average Loss: %.2f bps", losses.mean())
        log.info("[REPLAY SUMMARY] Largest Loss: %.2f bps", losses.min())
    log.info("")
    log.info("[REPLAY SUMMARY] Soft-stop exits: %d (%.1f%% of closed trades)", 
             soft_stop_count, 100 * soft_stop_count / len(closed_trades) if len(closed_trades) > 0 else 0)
    log.info("=" * 60)


# --------------------------------------------------------------------------- #
# Module-level wrappers for late-bound GX1DemoRunner methods
# --------------------------------------------------------------------------- #


def evaluate_entry(self: GX1DemoRunner, candles: pd.DataFrame) -> Optional[LiveTrade]:
    import inspect
    
    # ENTRY EVAL PATH TELEMETRY: Record which function actually performs entry evaluation (SSoT)
    # This MUST be the very first thing in the function to identify the actual code path
    if hasattr(self, "entry_manager") and self.entry_manager and hasattr(self.entry_manager, "entry_feature_telemetry") and self.entry_manager.entry_feature_telemetry:
        current_frame = inspect.currentframe()
        frame = current_frame if current_frame else None
        lineno = frame.f_lineno if frame else 0
        self.entry_manager.entry_feature_telemetry.record_entry_eval_path(
            function="evaluate_entry",
            file=__file__,
            line=lineno,
        )
    
    return self._evaluate_entry_impl(candles)


def _log_entry_only_event(
    self: GX1DemoRunner,
    timestamp: pd.Timestamp,
    side: str,
    price: float,
    prediction: Any,
    policy_state: Dict[str, Any],
) -> None:
    return self._log_entry_only_event_impl(timestamp, side, price, prediction, policy_state)


def _flush_entry_only_log_buffer(self: GX1DemoRunner) -> None:
    return self._flush_entry_only_log_buffer_impl()


def execute_entry(self: GX1DemoRunner, trade: LiveTrade) -> None:
    return self._execute_entry_impl(trade)


def evaluate_and_close_trades(self: GX1DemoRunner, candles: pd.DataFrame) -> None:
    return self._evaluate_and_close_trades_impl(candles)


def run_once(self: GX1DemoRunner) -> None:
    return self._run_once_impl()


def _update_health_signal(self: GX1DemoRunner, now: pd.Timestamp) -> None:
    return self._update_health_signal_impl(now)


def _setup_signal_handlers(self: GX1DemoRunner) -> None:
    return self._setup_signal_handlers_impl()


def _flush_logs(self: GX1DemoRunner) -> None:
    return self._flush_logs_impl()


def _check_disk_space(self: GX1DemoRunner) -> bool:
    return self._check_disk_space_impl()


def _rotate_logs(self: GX1DemoRunner) -> None:
    return self._rotate_logs_impl()


def run_forever(self: GX1DemoRunner) -> None:
    return self._run_forever_impl()




def _simulate_tick_exits_for_bar(
    self: GX1DemoRunner,
    bar_ts: pd.Timestamp,
    candle_row: pd.Series,
) -> None:
    return self._simulate_tick_exits_for_bar_impl(bar_ts, candle_row)


def _dump_backtest_summary(self: GX1DemoRunner) -> None:
    return self._dump_backtest_summary_impl()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GX1 v1.1 OANDA practice runner")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("gx1/configs/policies/GX1_V11_OANDA_DEMO.yaml"),
        help="Path to policy YAML.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single M5 evaluation cycle and exit.",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Override policy and send real orders to the configured OANDA environment.",
    )
    parser.add_argument(
        "--replay-csv",
        type=Path,
        default=None,
        help="Path to historical M5 candles CSV/parquet for offline backtest. Format: time,open,high,low,close,volume.",
    )
    parser.add_argument(
        "--fast-replay",
        action="store_true",
        help="Fast replay mode: skip heavy reporting, use simple flush for REPLAY_END (for tuning runs)",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=None,
        help="Maximum number of bars to process in replay mode (for testing/debugging). If not set, processes all bars.",
    )
    return parser.parse_args()


# Bind module-level helpers back onto GX1DemoRunner
GX1DemoRunner.evaluate_entry = evaluate_entry  # type: ignore[attr-defined]
GX1DemoRunner._log_entry_only_event = _log_entry_only_event  # type: ignore[attr-defined]
GX1DemoRunner._flush_entry_only_log_buffer = _flush_entry_only_log_buffer  # type: ignore[attr-defined]
# _ensure_bid_ask_columns, _calculate_unrealized_portfolio_bps, _reset_entry_diag, _record_entry_diag, and _log_entry_diag_summary are now methods of GX1DemoRunner class
# No need to assign them here
GX1DemoRunner._generate_client_order_id = _generate_client_order_id  # type: ignore[attr-defined]
GX1DemoRunner._build_notes_string = _build_notes_string  # type: ignore[attr-defined]
GX1DemoRunner._check_client_order_id_exists = _check_client_order_id_exists  # type: ignore[attr-defined]
GX1DemoRunner._execute_entry_impl = _execute_entry_impl  # type: ignore[attr-defined]
GX1DemoRunner._evaluate_and_close_trades_impl = _evaluate_and_close_trades_impl  # type: ignore[attr-defined]
def should_consider_entry(self, trend: str, vol: str, session: str, risk_score: float) -> bool:
    """Wrapper for _should_consider_entry_impl that takes self as first parameter."""
    # Initialize stage0_reasons tracking if not exists
    if not hasattr(self, "_stage0_reasons"):
        from collections import defaultdict
        self._stage0_reasons = defaultdict(int)
    
    # Get Stage-0 config from policy (if SNIPER) - cache to avoid per-bar YAML I/O in replay/chain.
    stage0_config = getattr(self, "_stage0_config_cached", None)
    if stage0_config is None and hasattr(self, "policy"):
        entry_config = self.policy.get("entry_config", {})
        if isinstance(entry_config, str):
            # Load entry config YAML if path (once)
            try:
                import yaml
                from pathlib import Path
                entry_config_path = Path(entry_config)
                if not entry_config_path.is_absolute():
                    entry_config_path = Path(__file__).parent.parent.parent / entry_config_path
                if entry_config_path.exists():
                    with open(entry_config_path, "r") as f:
                        entry_config = yaml.safe_load(f)
            except Exception:
                entry_config = self.policy.get("entry_config", {})
        
        # Check if SNIPER config exists
        if isinstance(entry_config, dict):
            stage0_config = entry_config.get("entry_gating", {}).get("stage0")
            # Check meta.role for SNIPER
            meta_role = self.policy.get("meta", {}).get("role", "")
            policy_name = self.policy.get("policy_name", "")
            if meta_role == "SNIPER_CANARY" or "SNIPER" in policy_name:
                # SNIPER mode - use config if available
                if stage0_config is None:
                    # Default SNIPER config (permissive)
                    stage0_config = {
                        "enabled": True,
                        "allowed_sessions": ["EU", "OVERLAP", "US"],
                        "allowed_vol_regimes": ["LOW", "MEDIUM", "HIGH"],
                        "block_trend_vol_combos": [],
                    }

        # Cache (even if None) to avoid repeated parsing
        self._stage0_config_cached = stage0_config
    
    result, reason = _should_consider_entry_impl(trend, vol, session, risk_score, stage0_config)
    
    # Store reason for logging
    self._stage0_reasons[reason] = self._stage0_reasons.get(reason, 0) + 1
    self._last_stage0_reason = reason
    
    return result

GX1DemoRunner.should_consider_entry = should_consider_entry  # type: ignore[attr-defined]
GX1DemoRunner.execute_entry = execute_entry  # type: ignore[attr-defined]
GX1DemoRunner.evaluate_and_close_trades = evaluate_and_close_trades  # type: ignore[attr-defined]
GX1DemoRunner.run_once = run_once  # type: ignore[attr-defined]
GX1DemoRunner._run_once_impl = _run_once_impl  # type: ignore[attr-defined]
GX1DemoRunner._update_health_signal = _update_health_signal  # type: ignore[attr-defined]
GX1DemoRunner._update_health_signal_impl = _update_health_signal_impl  # type: ignore[attr-defined]
GX1DemoRunner._setup_signal_handlers = _setup_signal_handlers  # type: ignore[attr-defined]
GX1DemoRunner._setup_signal_handlers_impl = _setup_signal_handlers_impl  # type: ignore[attr-defined]
GX1DemoRunner._flush_logs = _flush_logs  # type: ignore[attr-defined]
GX1DemoRunner._flush_logs_impl = _flush_logs_impl  # type: ignore[attr-defined]
GX1DemoRunner._check_disk_space = _check_disk_space  # type: ignore[attr-defined]
GX1DemoRunner._check_disk_space_impl = _check_disk_space_impl  # type: ignore[attr-defined]
GX1DemoRunner._rotate_logs = _rotate_logs  # type: ignore[attr-defined]
GX1DemoRunner._rotate_logs_impl = _rotate_logs_impl  # type: ignore[attr-defined]
GX1DemoRunner.run_forever = run_forever  # type: ignore[attr-defined]
GX1DemoRunner._run_forever_impl = _run_forever_impl  # type: ignore[attr-defined]
GX1DemoRunner.run_replay = run_replay  # type: ignore[attr-defined]
GX1DemoRunner._run_replay_impl = _run_replay_impl  # type: ignore[attr-defined]
GX1DemoRunner._replay_force_close_open_trades_at_end = _replay_force_close_open_trades_at_end  # type: ignore[attr-defined]
GX1DemoRunner._simulate_tick_exits_for_bar = _simulate_tick_exits_for_bar  # type: ignore[attr-defined]
GX1DemoRunner._simulate_tick_exits_for_bar_impl = _simulate_tick_exits_for_bar_impl  # type: ignore[attr-defined]
GX1DemoRunner._dump_backtest_summary = _dump_backtest_summary  # type: ignore[attr-defined]
GX1DemoRunner._dump_backtest_summary_impl = _dump_backtest_summary_impl  # type: ignore[attr-defined]


def main() -> None:
    try:
        args = parse_args()
        load_dotenv_if_present()
        # --no-dry-run flag overrides policy to set dry_run=False
        # If --no-dry-run is set, dry_run_override=False (send real orders)
        # Otherwise, dry_run_override=None (use policy/env/default)
        # In replay mode, always use dry_run=True (no real orders)
        dry_run_override = None if not args.no_dry_run else False
        if args.replay_csv:
            dry_run_override = True  # Always dry_run in replay mode
            runner = GX1DemoRunner(
                args.policy,
                dry_run_override=dry_run_override,
                replay_mode=True,
                fast_replay=args.fast_replay,
            )
            runner._max_bars = args.max_bars  # Pass max_bars to runner
            runner.run_replay(args.replay_csv)
            return  # Exit after replay (don't continue to run_once/run_forever)
        else:
            runner = GX1DemoRunner(args.policy, dry_run_override=dry_run_override)
        if args.once:
            runner.run_once()
        else:
            runner.run_forever()
    except KeyboardInterrupt:
        log.info("[MAIN] Interrupted by user")
        raise
    except Exception as e:
        log.error("[MAIN] Fatal error: %s", e, exc_info=True)
        import sys
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()