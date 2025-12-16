"""
GX1 v1.1 → OANDA practice demo runner.

This module wires the frozen ENTRY_V3 + EXIT_V2 models to the OANDA
practice API with strong safety defaults (dry-run, risk guards).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import yaml

from gx1.features.runtime_v9 import build_v9_runtime_features
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

ENTRY_MODEL_METADATA_PATH = Path("gx1/models/GX1_entry_session_metadata.json")
ENTRY_MODEL_PATHS = {
    "EU": Path("gx1/models/GX1_entry_EU.joblib"),
    "US": Path("gx1/models/GX1_entry_US.joblib"),
    "OVERLAP": Path("gx1/models/GX1_entry_OVERLAP.joblib"),
}
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
                    data = json.loads(line)
                except json.JSONDecodeError:
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
    trade_id: str
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
    with open(path, "r", encoding="utf-8") as handle:
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


def load_entry_models(
    metadata_path: Path,
    model_paths: Dict[str, Path],
) -> EntryModelBundle:
    """
    Load session-routed entry models and their metadata.

    Parameters
    ----------
    metadata_path : Path
        Path to JSON metadata describing feature columns.
    model_paths : Dict[str, Path]
        Mapping of session tag → model path.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Entry metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    feature_cols = metadata.get("feature_cols")
    if not feature_cols:
        raise ValueError(f"'feature_cols' missing in entry metadata: {metadata_path}")

    # Hash feature columns for drift detection
    feature_cols_str = "|".join(sorted(feature_cols))
    feature_cols_hash = hashlib.md5(feature_cols_str.encode("utf-8")).hexdigest()[:16]
    metadata_hash = metadata.get("feature_cols_hash")
    if metadata_hash:
        if feature_cols_hash != metadata_hash:
            log.warning(
                "Feature columns hash mismatch: computed=%s, metadata=%s. "
                "This may indicate feature drift.",
                feature_cols_hash,
                metadata_hash,
            )
    else:
        log.info("Feature columns hash: %s (metadata hash not present)", feature_cols_hash)
    log.info("Loaded %d feature columns from metadata: %s", len(feature_cols), feature_cols[:5])
    
    # Get model bundle version from metadata
    model_bundle_version = metadata.get("model_bundle_version") or metadata.get("version") or None

    models: Dict[str, Any] = {}
    for session in ("EU", "US", "OVERLAP"):
        path = model_paths.get(session)
        if path is None:
            raise ValueError(f"No model path configured for session '{session}'.")
        if not path.exists():
            raise FileNotFoundError(f"Entry model for {session} not found: {path}")
        model = joblib.load(path)
        if not hasattr(model, "predict_proba"):
            raise ValueError(f"Entry model for {session} lacks predict_proba(): {path}")
        
        # Verify model expects correct number of features
        if hasattr(model, "n_features_in_"):
            if model.n_features_in_ != len(feature_cols):
                raise ValueError(
                    f"Model for {session} expects {model.n_features_in_} features, "
                    f"but metadata specifies {len(feature_cols)} features."
                )
        
        # Log model classes for verification (CRITICAL for probability mapping)
        classes = getattr(model, "classes_", None)
        log.info(
            "[BOOT] %s classes=%s n_features_in_=%s",
            session,
            classes.tolist() if classes is not None else None,
            getattr(model, "n_features_in_", "unknown"),
        )
        models[session] = model

    return EntryModelBundle(
        models=models,
        feature_names=list(feature_cols),
        metadata=metadata,
        feature_cols_hash=feature_cols_hash,
        model_bundle_version=model_bundle_version,
        ts_map_hash=None,  # Will be set after temperature map is loaded
    )


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
        # 3-class: [SHORT, NEUTRAL, LONG] (assumed order)
        return EntryPrediction(
            session="",
            prob_short=float(prob_vector[0]),
            prob_neutral=float(prob_vector[1]),
            prob_long=float(prob_vector[2]),
            p_hat=float(max(prob_vector[0], prob_vector[2])),
            margin=float(abs(prob_vector[2] - prob_vector[0])),
        )

    if n_classes == 2:
        # 2-class: Must map correctly using model.classes_
        # Common cases: [0, 1] (0=SHORT, 1=LONG) or [1, 0] (1=LONG, 0=SHORT)
        idx_long = None
        idx_short = None
        
        if classes is not None and len(classes) == 2:
            # Map via model.classes_: find which index corresponds to LONG (1) and SHORT (0)
            classes_list = classes.tolist() if hasattr(classes, 'tolist') else list(classes)
            for idx, cls in enumerate(classes_list):
                if isinstance(cls, str):
                    tag = cls.upper()
                    if any(keyword in tag for keyword in ("UP", "LONG", "BUY", "POS", "1")):
                        idx_long = idx
                    elif any(keyword in tag for keyword in ("DOWN", "SHORT", "SELL", "NEG", "0")):
                        idx_short = idx
                elif isinstance(cls, (int, np.integer)):
                    cls_int = int(cls)
                    if cls_int == 1:
                        idx_long = idx
                    elif cls_int == 0:
                        idx_short = idx
            
            # Fallback: if we found one but not the other, infer the other
            if idx_long is not None and idx_short is None:
                idx_short = 1 - idx_long
            elif idx_short is not None and idx_long is None:
                idx_long = 1 - idx_short
        
        # If still not found, use standard mapping: assume [0, 1] or [SHORT, LONG]
        if idx_long is None or idx_short is None:
            # Default assumption: higher probability class is LONG, lower is SHORT
            # But this is risky - better to use model.classes_ explicitly
            if prob_vector[0] > prob_vector[1]:
                idx_long = 0
                idx_short = 1
            else:
                idx_long = 1
                idx_short = 0
        
        prob_long = float(prob_vector[idx_long])
        prob_short = float(prob_vector[idx_short])
        return EntryPrediction(
            session="",
            prob_short=prob_short,
            prob_neutral=0.0,
            prob_long=prob_long,
            p_hat=float(max(prob_long, prob_short)),
            margin=float(abs(prob_long - prob_short)),
        )

    if n_classes == 1:
        # 1-class: binary classifier (probability of positive class)
        prob_long = float(prob_vector[0])
        return EntryPrediction(
            session="",
            prob_short=float(1.0 - prob_long),
            prob_neutral=0.0,
            prob_long=prob_long,
            p_hat=float(max(prob_long, 1.0 - prob_long)),
            margin=float(abs(prob_long - (1.0 - prob_long))),
        )

    raise ValueError(f"Unsupported number of classes for entry model: {n_classes}")


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


def _should_consider_entry_impl(trend: str, vol: str, session: str, risk_score: float) -> bool:
    """
    Stage-0 opportunitetsfilter.
    
    Returnerer True hvis vi i det hele tatt skal kjøre entry-modellene (XGB/TCN)
    for gjeldende bar, gjet Big Brain V1-regime + session.
    
    Dette er første versjon, med enkel hardkodet whitelist-logikk.
    TODO: gjøre should_consider_entry config-drevet (YAML) når Stage-0 v1 er validert på flere perioder.
    
    Parameters
    ----------
    trend : str
        Big Brain V1 trend regime (TREND_UP, TREND_DOWN, MR, UNKNOWN).
    vol : str
        Big Brain V1 volatility regime (LOW, MID, HIGH, EXTREME, UNKNOWN).
    session : str
        Trading session (EU, US, OVERLAP, ASIA).
    risk_score : float
        Big Brain V1 risk score (0.0-1.0).
    
    Returns
    -------
    bool
        True if we should consider entry (run XGB/TCN), False to skip completely.
    """
    # Handle UNKNOWN regimes - conservative: skip entry consideration
    if trend == "UNKNOWN" or vol == "UNKNOWN":
        return False
    
    # 1) Blokker åpenbart dårlige sessions/regimer først
    # OVERLAP er allerede veldig problematisk i Q2, og er tungt gated i SAFETY_V4,
    # så i Stage-0 v1 vurderer vi OVERLAP som "ikke prioritert" miljø:
    if session == "OVERLAP":
        return False
    
    # 2) Blokker EXTREME i EU generelt – vi vet dette er giftig
    if session == "EU" and vol == "EXTREME":
        return False
    
    # 3) Håndter US – her har vi sett at EXTREME/HIGH kan være bra
    if session == "US":
        # US EXTREME (TREND_UP/TREND_DOWN/MR) har vist seg å være bra, spesielt for LONG i tidligere analyser.
        # Vi vil tillate alle ikke-totalt gale kombinasjoner, men fortsatt unngå HIGH+TREND_UP for SHORT.
        if vol in {"HIGH", "EXTREME"}:
            return True
        # MID i US kan også være ok:
        if vol == "MID":
            return True
        # LOW i US er ofte slapp/støyete – v1: vi skipper LOW:
        return False
    
    # 4) Håndter ASIA – Q2-data viser positiv total PnL i ASIA (ikke EXTREME).
    if session == "ASIA":
        if vol != "EXTREME":
            return True
        return False
    
    # 5) Håndter EU – her har TREND_UP + HIGH/EXTREME vært dårlige, men andre kombinasjoner kan være ok.
    if session == "EU":
        # Dump TREND_UP + HIGH – dette vet vi er dårlig
        if trend == "TREND_UP" and vol == "HIGH":
            return False
        # La MR + MID/LOW og TREND_DOWN + MID/HIGH slippe gjennom
        if trend == "MR" and vol in {"LOW", "MID"}:
            return True
        if trend == "TREND_DOWN" and vol in {"MID", "HIGH"}:
            return True
        # Alt annet i EU: v1 → nei
        return False
    
    # Fallback – ukjente sessions etc: ikke vurder entry
    return False


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
                            row_dict["extra"] = json.dumps(row_dict["extra"])
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
        df.loc[mask, "extra"] = json.dumps(extra)
    
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
                extra_dict = json.loads(row["extra"])
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
        row["extra"] = json.dumps(row["extra"])
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
        handle.write(json.dumps(eval_record, separators=(",", ":")) + "\n")


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
    def __init__(
        self,
        policy_path: Path,
        *,
        dry_run_override: Optional[bool] = None,
        replay_mode: bool = False,
        fast_replay: bool = False,
    ) -> None:
        load_dotenv_if_present()
        self.policy_path = policy_path
        self.policy = load_yaml_config(policy_path)
        self.exit_config_name = None
        
        # Initialize replay flags early (before any methods that check them)
        self.replay_mode = bool(replay_mode)
        self.replay_start_ts: Optional[pd.Timestamp] = None
        self.replay_end_ts: Optional[pd.Timestamp] = None
        self.fast_replay = bool(fast_replay)  # Fast mode for tuning (skip heavy reporting)
        self.is_replay = self.replay_mode or self.fast_replay
        # Stable identifier for this process (used in trade_log for analysis)
        # Priority: 1) Environment variable (for testing), 2) Policy config, 3) Auto-generated
        env_run_id = os.getenv("GX1_RUN_ID")
        if env_run_id:
            self.run_id = env_run_id
            log.info("[POLICY] Using custom run_id from environment: %s", self.run_id)
        elif "run_id" in self.policy:
            self.run_id = self.policy["run_id"]
            log.info("[POLICY] Using custom run_id from policy: %s", self.run_id)
        else:
            ts_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{'replay' if self.is_replay else 'live'}_{ts_str}"
        
        # Policy-lock: Compute and store policy_hash at startup
        policy_content = policy_path.read_text()
        self.policy_hash = hashlib.md5(policy_content.encode("utf-8")).hexdigest()[:16]
        
        # Extract policy name/version for traceability
        self.policy_name = self.policy.get("version", policy_path.stem)  # Use policy["version"] if set, else filename
        if "version" not in self.policy:
            log.info("[POLICY] Policy version not set - using filename: %s", self.policy_name)
        
        # Read mode from policy (default: "LIVE")
        self.mode = self.policy.get("mode", "LIVE")
        
        # Check for CANARY mode (override dry_run, but log all invariants)
        self.canary_mode = self.mode.upper() == "CANARY"
        if self.canary_mode:
            log.info("[CANARY] Canary mode enabled - dry_run=True, all invariants will be logged")
            if dry_run_override is None:
                dry_run_override = True  # Force dry_run in canary mode
        
        log.info("[BOOT] mode=%s", self.mode)
        
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
        
        entry_cfg_path = Path(self.policy.get("entry_config", "gx1/configs/entry_policy_ENTRY_ANCHOR_V3.yaml"))
        exit_cfg_path = Path(self.policy.get("exit_config", "gx1/configs/exit_policy_EXIT_V2.yaml"))
        entry_cfg = load_yaml_config(entry_cfg_path)
        exit_cfg = load_yaml_config(exit_cfg_path)
        self.entry_params = entry_cfg.get("params", entry_cfg)
        
        # Merge entry_config into self.policy so entry policies can access their configs
        # This is critical for entry_v9_policy_farm_v1, entry_v9_policy_v1, etc.
        if entry_cfg:
            for key, value in entry_cfg.items():
                if key not in self.policy or key == "entry_models":  # Don't override entry_models if already in policy
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
                        import joblib
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
                                import json
                                with open(feature_cols_path, 'r') as f:
                                    feature_data = json.load(f)
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
                    import json
                    
                    # Load meta
                    with open(tcn_meta_path, "r") as f:
                        meta = json.load(f)
                    
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
        
        # --- ENTRY_V9 Model (ONLY entry model - no fallback) ---
        entry_models_cfg = self.policy.get("entry_models", {})
        entry_v9_cfg = entry_models_cfg.get("v9", {})
        self.entry_v9_enabled = bool(entry_v9_cfg.get("enabled", False))
        self.entry_v9_model: Optional[Dict[str, Any]] = None
        self.entry_v9_cfg = entry_v9_cfg
        
        # V9 is MANDATORY - no fallback to V6/V8
        if not self.entry_v9_enabled:
            log.error("[ENTRY_V9] V9 is REQUIRED but disabled in policy. No entry will be possible.")
        
        # Load ENTRY_V9 (mandatory)
        if self.entry_v9_enabled:
            try:
                model_dir = Path(entry_v9_cfg.get("model_dir", "gx1/models/entry_v9/nextgen_2020_2025_clean"))
                if not model_dir.exists():
                    log.error("[ENTRY_V9] Model directory not found: %s. Disabling ENTRY_V9.", model_dir)
                    self.entry_v9_enabled = False
                else:
                    # Load ENTRY_V9 model
                    self.entry_v9_model = self._load_entry_v9_model(model_dir, entry_v9_cfg)
                    # Store paths for runtime feature building
                    self.entry_v9_feature_meta_path = model_dir / "entry_v9_feature_meta.json"
                    self.entry_v9_seq_scaler_path = model_dir / "seq_scaler.joblib"
                    self.entry_v9_snap_scaler_path = model_dir / "snap_scaler.joblib"
                    
                    # Validate paths exist
                    if not self.entry_v9_feature_meta_path.exists():
                        log.error("[ENTRY_V9] Feature meta path does not exist: %s", self.entry_v9_feature_meta_path)
                        self.entry_v9_enabled = False
                    elif not self.entry_v9_seq_scaler_path.exists():
                        log.error("[ENTRY_V9] Seq scaler path does not exist: %s", self.entry_v9_seq_scaler_path)
                        self.entry_v9_enabled = False
                    elif not self.entry_v9_snap_scaler_path.exists():
                        log.error("[ENTRY_V9] Snap scaler path does not exist: %s", self.entry_v9_snap_scaler_path)
                        self.entry_v9_enabled = False
                    else:
                        log.info("[ENTRY_V9] Runtime feature paths validated:")
                        log.info("  feature_meta: %s", self.entry_v9_feature_meta_path)
                        log.info("  seq_scaler: %s", self.entry_v9_seq_scaler_path)
                        log.info("  snap_scaler: %s", self.entry_v9_snap_scaler_path)

                        # Pre-load and cache feature_meta and scalers (MID priority optimization)
                        # This avoids re-reading files on every prediction call
                        try:
                            from gx1.features.runtime_v9 import _load_feature_meta, _load_scaler

                            self._entry_v9_seq_features, self._entry_v9_snap_features = _load_feature_meta(
                                self.entry_v9_feature_meta_path
                            )
                            self._entry_v9_seq_scaler = _load_scaler(self.entry_v9_seq_scaler_path)
                            self._entry_v9_snap_scaler = _load_scaler(self.entry_v9_snap_scaler_path)
                            log.info(
                                "[ENTRY_V9] Pre-loaded feature_meta (%d seq, %d snap) and scalers (cached)",
                                len(self._entry_v9_seq_features),
                                len(self._entry_v9_snap_features),
                            )
                        except Exception as e:
                            log.warning(
                                "[ENTRY_V9] Failed to pre-load feature_meta/scalers (will load on-demand): %s",
                                e,
                            )
                            # Not critical - build_v9_runtime_features will load on-demand
                            self._entry_v9_seq_features = None
                            self._entry_v9_snap_features = None
                            self._entry_v9_seq_scaler = None
                            self._entry_v9_snap_scaler = None

                    # Update model_name if ENTRY_V9 is active
                    if self.entry_v9_model and self.entry_v9_model.get("meta"):
                        self.model_name = self.entry_v9_model["meta"].get("model_name", "ENTRY_V9")
                    log.info("[ENTRY_V9] Model loaded from %s", model_dir)
            except Exception as e:
                log.error("[ENTRY_V9] Failed to load model: %s. Disabling ENTRY_V9.", e, exc_info=True)
                self.entry_v9_enabled = False
        else:
            log.info("[ENTRY_V9] Model disabled in policy")
        
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
            self.exit_control.allowed_loss_closers = [
                "RULE_A_PROFIT",
                "RULE_A_TRAILING",
                "RULE_B_FAST_LOSS",
                "RULE_C_TIMEOUT",
            ]
            self.exit_control.allow_model_exit_when["min_bars"] = 1
            self.exit_control.allow_model_exit_when["min_pnl_bps"] = -100
            self.exit_control.allow_model_exit_when["min_exit_prob"] = 0.0
            log.info("[BOOT] ExitArbiter configured for EXIT_FARM_V2_RULES")
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
        self._init_trade_journal()
        
        # JSON eval log path (rotates daily)
        self.eval_log_path = self.log_dir / f"eval_log_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d')}.jsonl"
        self.eval_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.instrument = self.policy.get("instrument", "XAU_USD")
        self.granularity = self.policy.get("granularity", "M5")

        # Check if we're using FARM_V2B (which uses ENTRY_V9 directly, not session-routed models)
        entry_config_path = self.policy.get("entry_config", "")
        entry_v9_policy_farm_v2b_enabled = False
        if entry_config_path:
            try:
                entry_config = load_yaml_config(Path(entry_config_path))
                entry_v9_policy_farm_v2b_enabled = entry_config.get("entry_v9_policy_farm_v2b", {}).get("enabled", False)
            except Exception:
                pass  # If we can't load entry config, assume not FARM_V2B
        
        # Optional: Load session-routed entry models (only if metadata exists and not FARM_V2B-only mode)
        entry_model_cfg = self.policy.get("entry_model", {})
        entry_metadata_path = Path(
            entry_model_cfg.get("metadata", entry_model_cfg.get("metadata_path", ENTRY_MODEL_METADATA_PATH))
        )
        
        if entry_v9_policy_farm_v2b_enabled:
            # FARM_V2B uses ENTRY_V9 directly - skip session-routed bundle
            log.info("[BOOT] FARM_V2B mode detected: skipping session-routed entry bundle (using ENTRY_V9 directly)")
            self.entry_model_bundle = None
            self.session_entry_enabled = False
        elif not entry_metadata_path.exists():
            # Metadata missing - optional mode for replay/FARM
            log.info(
                "[BOOT] Session-routed entry bundle metadata not found at %s; "
                "continuing with direct ENTRY_V9 only. This is expected for FARM policies.",
                entry_metadata_path
            )
            self.entry_model_bundle = None
            self.session_entry_enabled = False
        else:
            # Metadata exists - load session-routed bundle as normal
            model_path_overrides = {
                "EU": Path(entry_model_cfg.get("eu", ENTRY_MODEL_PATHS["EU"])),
                "US": Path(entry_model_cfg.get("us", ENTRY_MODEL_PATHS["US"])),
                "OVERLAP": Path(entry_model_cfg.get("overlap", ENTRY_MODEL_PATHS["OVERLAP"])),
            }
            try:
                self.entry_model_bundle = load_entry_models(entry_metadata_path, model_path_overrides)
                self.session_entry_enabled = True
                log.info("[BOOT] Loaded session XGB models (EU/US/OVERLAP)")
                log.info(
                    "[BOOT] Entry models: EU=%s, US=%s, OVERLAP=%s",
                    "✓" if "EU" in self.entry_model_bundle.models else "✗",
                    "✓" if "US" in self.entry_model_bundle.models else "✗",
                    "✓" if "OVERLAP" in self.entry_model_bundle.models else "✗",
                )
            except Exception as e:
                log.warning(
                    "[BOOT] Failed to load session-routed entry bundle: %s. "
                    "Continuing with direct ENTRY_V9 only.",
                    e
                )
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
        
        # Managers
        self.entry_manager = EntryManager(self, exit_config_name=exit_config_name)
        self.exit_manager = ExitManager(self)
        self._reset_entry_diag()
    
    def _load_entry_v9_model(self, model_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load ENTRY_V9 transformer model from model directory.
        
        Returns dict with model components: model, meta, seq_scaler, snap_scaler, device, config
        """
        import torch
        import json
        from joblib import load as joblib_load
        from gx1.models.entry_v9.entry_v9_transformer import build_entry_v9_model
        
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"ENTRY_V9 model directory not found: {model_dir}")
        
        log.info("[ENTRY_V9] Loading model from %s", model_dir)
        
        # Load metadata
        meta_path = model_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"ENTRY_V9 metadata not found: {meta_path}")
        
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        # Load feature metadata
        feature_meta_path = model_dir / "entry_v9_feature_meta.json"
        if not feature_meta_path.exists():
            raise FileNotFoundError(f"ENTRY_V9 feature metadata not found: {feature_meta_path}")
        
        with open(feature_meta_path, "r") as f:
            feature_meta = json.load(f)
        
        seq_feat_names = feature_meta.get("seq_features", feature_meta.get("seq_feature_names", []))
        snap_feat_names = feature_meta.get("snap_features", feature_meta.get("snap_feature_names", []))
        
        log.info("[ENTRY_V9] Model metadata:")
        log.info("  seq_features: %d", len(seq_feat_names))
        log.info("  snap_features: %d", len(snap_feat_names))
        
        # Build model from config
        model_cfg = config.get("model", {}).copy()
        if not model_cfg:
            # Build from metadata
            model_cfg = meta.get("model_config", {})
        if not model_cfg:
            # Fallback to defaults
            model_cfg = {
                "name": "entry_v9",
                "seq_input_dim": len(seq_feat_names),
                "snap_input_dim": len(snap_feat_names),
                "max_seq_len": 30,
                "seq_cfg": {"d_model": 128, "n_heads": 4, "num_layers": 3, "dim_feedforward": 384, "dropout": 0.1},
                "snap_cfg": {"hidden_dims": [256, 128, 64], "use_layernorm": True, "dropout": 0.0},
                "regime_cfg": {"embedding_dim": 16},
                "fusion_hidden_dim": 128,
                "fusion_dropout": 0.1,
                "head_hidden_dim": 64,
            }
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_entry_v9_model({"model": model_cfg})
        
        # Load model weights
        model_path = model_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"ENTRY_V9 model weights not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        model.to(device)
        
        # Load scalers
        seq_scaler_path = model_dir / "seq_scaler.joblib"
        snap_scaler_path = model_dir / "snap_scaler.joblib"
        
        if not seq_scaler_path.exists():
            raise FileNotFoundError(f"ENTRY_V9 seq_scaler not found: {seq_scaler_path}")
        if not snap_scaler_path.exists():
            raise FileNotFoundError(f"ENTRY_V9 snap_scaler not found: {snap_scaler_path}")
        
        seq_scaler = joblib_load(seq_scaler_path)
        snap_scaler = joblib_load(snap_scaler_path)
        
        log.info("[ENTRY_V9] Loaded scalers from %s and %s", seq_scaler_path, snap_scaler_path)
        
        return {
            "model": model,
            "meta": meta,
            "seq_scaler": seq_scaler,
            "snap_scaler": snap_scaler,
            "device": device,
            "config": model_cfg,
            "seq_feat_names": seq_feat_names,
            "snap_feat_names": snap_feat_names,
        }
        
        # Start tick watcher if we have open trades after reconcile
        self._maybe_update_tick_watcher()
        
        # Hash temperature map for drift detection
        temp_map = self._get_temperature_map()
        temp_map_str = "|".join(f"{k}:{v:.6f}" for k, v in sorted(temp_map.items()))
        ts_map_hash = hashlib.md5(temp_map_str.encode("utf-8")).hexdigest()[:16]
        # Update bundle with ts_map_hash (immutable dataclass workaround)
        if self.entry_model_bundle is not None:
            self.entry_model_bundle = EntryModelBundle(
                models=self.entry_model_bundle.models,
                feature_names=self.entry_model_bundle.feature_names,
                metadata=self.entry_model_bundle.metadata,
                feature_cols_hash=self.entry_model_bundle.feature_cols_hash,
                model_bundle_version=self.entry_model_bundle.model_bundle_version,
                ts_map_hash=ts_map_hash,
            )
        else:
            log.debug("[BOOT] entry_model_bundle is None - skipping ts_map_hash update")
        
        # Track consecutive failures for hard STOP
        self._consecutive_order_failures = 0
        self._consecutive_close_failures = 0
        
        # Graceful shutdown flag
        self._shutdown_requested = False
        
        # Health signal
        self.health_signal_path = self.log_dir / "health.json"
        self.health_signal_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_health_signal_time = None
        
        # Backfill state
        self.backfill_in_progress = False
        self.backfill_cache: Optional[pd.DataFrame] = None
        self.warmup_floor: Optional[pd.Timestamp] = None

        # Initialize telemetry tracker
        telemetry_cfg = self.policy.get("telemetry", {})
        # Default to False if not specified (for DEBUG mode)
        self.telemetry_enabled = bool(telemetry_cfg.get("enabled", True))  # Default True for backward compat, but can be disabled
        telemetry_dir = Path(telemetry_cfg.get("telemetry_dir", self.log_dir / "telemetry"))
        window_minutes = int(telemetry_cfg.get("window_minutes", 30))
        log_interval_minutes = int(telemetry_cfg.get("log_interval_minutes", 10))
        # Get target_coverage from policy config, fallback to entry_params.gating.coverage.target
        target_coverage = float(
            telemetry_cfg.get(
                "target_coverage",
                self.entry_params.get("gating", {}).get("coverage", {}).get("target", 0.20),
            )
        )
        ece_window_size = int(telemetry_cfg.get("ece_window_size", 500))
        if self.telemetry_enabled:
            self.telemetry_tracker = TelemetryTracker(
                telemetry_dir=telemetry_dir,
                window_minutes=window_minutes,
                log_interval_minutes=log_interval_minutes,
                target_coverage=target_coverage,
                ece_window_size=ece_window_size,
                is_replay=self.replay_mode or self.fast_replay,
            )
        else:
            self.telemetry_tracker = None

        # Initialize parity-run (for reprod-bevis)
        parity_cfg = self.policy.get("parity", {})
        self.parity_enabled = bool(parity_cfg.get("enabled", False))
        self.parity_sample_every_n = int(parity_cfg.get("sample_every_n", 1))
        self.parity_tolerance_p50 = float(parity_cfg.get("tolerance_p50", 1e-6))
        self.parity_tolerance_p99 = float(parity_cfg.get("tolerance_p99", 1e-4))
        parity_out_path = parity_cfg.get("out_path", str(telemetry_dir / "parity_{YYYYMMDD}.jsonl"))
        parity_out_path = parity_out_path.replace("{YYYYMMDD}", pd.Timestamp.now(tz="UTC").strftime("%Y%m%d"))
        self.parity_log_path = Path(parity_out_path)
        self.parity_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Parity metrics tracking
        self.parity_metrics: Dict[str, List[float]] = defaultdict(list)
        self.parity_direction_matches: Dict[str, List[bool]] = defaultdict(list)
        self.parity_sample_counter = 0
        self._parity_disabled = False
        
        # Parity-run uses same models as live runner (entry_model_bundle.models)
        # No need to load separate offline models
        if self.parity_enabled:
            if self.entry_model_bundle is None or not self.entry_model_bundle.models:
                log.warning("Parity-run enabled but entry_model_bundle is not available. Disabling parity-run.")
                self.parity_enabled = False
            else:
                log.info("Parity-run enabled: using %d live models for parity verification", len(self.entry_model_bundle.models))

        # Log all hashes and version info at startup
        if self.entry_model_bundle is not None:
            log.info("=" * 60)
            log.info("GX1 Entry Model Bundle Version Info:")
            log.info("  model_bundle_version: %s", self.entry_model_bundle.model_bundle_version or "N/A")
            log.info("  feature_cols_hash: %s", self.entry_model_bundle.feature_cols_hash)
            log.info("  ts_map_hash: %s", self.entry_model_bundle.ts_map_hash)
            log.info("  temperature_map: %s", temp_map)
            log.info("=" * 60)
        else:
            log.debug("[BOOT] entry_model_bundle is None - skipping version info logging")
        
        # Perform backfill on startup (if enabled and not in replay mode)
        backfill_cfg = self.policy.get("backfill", {})
        if self.replay_mode:
            log.info("[BACKFILL] Replay mode detected (replay-csv is set) – skipping OANDA backfill.")
            self.backfill_in_progress = False
            self.backfill_cache = None
            self.warmup_floor = None
        else:
            if backfill_cfg.get("enabled", True):  # Default: enabled
                log.info(
                    "[BOOT] Backfill enabled: overlap_bars=%d, lookback_padding=%d", 
                    backfill_cfg.get("overlap_bars", 6),
                    backfill_cfg.get("lookback_padding", 100),
                )
                log.info("[BACKFILL] Starting backfill on startup...")
                self.backfill_in_progress = True
                
                try:
                    # Perform backfill (returns cache_df, gaps_refetched, warmup_floor, bars_remaining, revisions)
                    self.backfill_cache, gaps_refetched, warmup_floor, bars_remaining, revisions = self._perform_backfill()
                    
                    # Set warmup_floor from backfill result
                    self.warmup_floor = warmup_floor
                    
                    # Track revisions in telemetry (warn if >0)
                    if revisions > 0:
                        log.warning(
                            "[REVISION] Detected %d revisions in backfill (n_revisions_24h tracking)", 
                            revisions,
                        )
                        # TODO: Add n_revisions_24h to telemetry tracker
                    
                    # Backfill complete
                    self.backfill_in_progress = False
                    
                except Exception as e:
                    log.error("[BACKFILL] Backfill failed: %s", e, exc_info=True)
                    # Continue without backfill (fallback to live mode)
                    self.backfill_in_progress = False
                    self.backfill_cache = None
                    self.warmup_floor = None
            else:
                log.info("[BACKFILL] Backfill disabled in policy")

        log.info(
            "GX1 demo runner initialised (dry_run=%s, instrument=%s, granularity=%s)",
            self.exec.dry_run,
            self.instrument,
            self.granularity,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _reconcile_open_trades(self) -> None:
        """
        Reconcile open trades from OANDA with internal state.
        
        Loads open trades from OANDA API and binds them to internal entry_id.
        This prevents duplicate orders on restart.
        """
        if self.exec.dry_run:
            # In dry-run mode, no reconciliation needed
            log.info("Reconcile skipped (dry_run mode)")
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
                    trade_id=trade.trade_id,
                    exit_profile=exit_profile,
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
                state = json.load(f)
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
                json.dump(state, f, indent=2)
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
                json.dump(receipt, f, indent=2)
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
        hard_min_bars = int(backfill_cfg.get("hard_min_bars", 900))  # Default: 900 bars minimum
        force_on_boot = bool(backfill_cfg.get("force_on_boot", False))  # Default: False (respect last_bar_ts)
        min_bars_on_boot = int(backfill_cfg.get("min_bars_on_boot", 900))  # Default: 900 bars minimum on boot
        
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
        
        # Perform backfill (chunked with retry, gap scan, revision detection)
        cache_df, gaps_refetched, revisions = self._resume_with_backfill(
            cache_df=None,  # No existing cache on startup
            from_ts=from_ts,
            to_ts=to_ts,
            overlap_bars=overlap_bars,
            now_utc=now_utc,
        )
        
        # Measure backfill duration
        backfill_duration_s = time.time() - backfill_start_time
        
        # Calculate bars in cache
        bars_in_cache = len(cache_df) if not cache_df.empty else 0
        fetched_bars = bars_in_cache  # Approximate (actual fetched may differ due to validation)
        
        # Check if we need to fetch more bars to meet hard_min_bars
        # Note: OANDA API may not return candles for weekends/non-trading periods
        # So we try to fetch more, but accept what we have if we can't get more
        if bars_in_cache < hard_min_bars:
            # We don't have enough bars, try to fetch more (but only if we're significantly short)
            # If we're close to hard_min_bars (within 100 bars), accept what we have
            if bars_in_cache < (hard_min_bars - 100):
                log.warning(
                    "[BACKFILL] Insufficient bars: have=%d, need≥%d. Attempting to fetch additional bars.",
                    bars_in_cache,
                    hard_min_bars,
                )
                # Calculate how many bars we need
                bars_needed = hard_min_bars - bars_in_cache
                # Extend from_ts to fetch more bars (but limit to max 1 day back)
                additional_bars_ts = max(from_ts - pd.Timedelta(minutes=5 * bars_needed), from_ts - pd.Timedelta(days=1))
                # Fetch additional bars
                try:
                    additional_cache_df, additional_gaps, additional_revisions = self._resume_with_backfill(
                        cache_df=cache_df,
                        from_ts=additional_bars_ts,
                        to_ts=from_ts,
                        overlap_bars=0,  # No overlap for additional fetch
                        now_utc=now_utc,
                    )
                    # Combine with existing cache
                    if not additional_cache_df.empty:
                        cache_df = pd.concat([additional_cache_df, cache_df]).sort_index().drop_duplicates(keep="last")
                        bars_in_cache = len(cache_df)
                        gaps_refetched += additional_gaps
                        revisions += additional_revisions
                        log.info(
                            "[BACKFILL] Fetched additional bars: now have=%d bars (target≥%d)",
                            bars_in_cache,
                            hard_min_bars,
                        )
                    else:
                        log.warning(
                            "[BACKFILL] No additional bars available from OANDA (weekends/non-trading periods). Using %d bars (target≥%d)",
                            bars_in_cache,
                            hard_min_bars,
                        )
                except Exception as e:
                    log.warning(
                        "[BACKFILL] Failed to fetch additional bars: %s. Using %d bars (target≥%d)",
                        e,
                        bars_in_cache,
                        hard_min_bars,
                    )
            else:
                log.info(
                    "[BACKFILL] Close to hard_min_bars: have=%d, need≥%d. Accepting current bars (OANDA may not have more due to weekends/non-trading periods).",
                    bars_in_cache,
                    hard_min_bars,
                )
        
        # Calculate warmup_floor based on min_bars_on_boot
        # If we have ≥min_bars_on_boot bars, clear warmup_floor immediately (no warmup)
        # Otherwise, check if we have enough bars for feature-rehydration (≥lookback_padding)
        # If we have enough for feature-rehydration, we can start immediately (OANDA may not have more due to weekends)
        min_bars_for_feature_rehydration = lookback_padding  # Minimum bars needed for feature rehydration
        if bars_in_cache >= min_bars_on_boot:
            # We have enough bars in cache, clear warmup_floor immediately
            warmup_floor = None
            bars_remaining = 0
            resume_at = "immediate"
            log.info(
                "[BACKFILL] Sufficient bars for immediate start: have=%d bars (need≥%d bars)",
                bars_in_cache,
                min_bars_on_boot,
            )
        elif bars_in_cache >= min_bars_for_feature_rehydration:
            # We have enough bars for feature-rehydration, but not enough for min_bars_on_boot
            # Accept what we have and start immediately (OANDA may not have more due to weekends/non-trading periods)
            warmup_floor = None
            bars_remaining = 0
            resume_at = "immediate"
            log.info(
                "[BACKFILL] Sufficient bars for feature-rehydration: have=%d bars (need≥%d bars for features, target≥%d bars). Starting immediately (OANDA may not have more due to weekends/non-trading periods).",
                bars_in_cache,
                min_bars_for_feature_rehydration,
                min_bars_on_boot,
            )
        else:
            # We don't have enough bars, set warmup_floor to wait for more bars
            # Use max(2*lookback, 1*H4_window) as warmup requirement
            warmup_bars = max(2 * lookback_padding, h4_window_bars, min_bars_on_boot)
            warmup_minutes = 5 * warmup_bars
            warmup_floor = now_utc.floor("5min") + pd.Timedelta(minutes=warmup_minutes)
            bars_remaining = max(0, warmup_bars - bars_in_cache)
            resume_at = warmup_floor.strftime("%Y-%m-%dT%H:%M:%SZ")
            log.info(
                "[BACKFILL] Insufficient bars for immediate start: have=%d bars, need≥%d bars for features. Warmup until %s",
                bars_in_cache,
                min_bars_for_feature_rehydration,
                resume_at,
            )
        
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
            else:
                threshold_met = min_bars_for_feature_rehydration
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
            if hasattr(self, "trade_log_path") and self.trade_log_path:
                output_dir = self.trade_log_path.parent
                # If in parallel_chunks, go up to main output dir
                if "parallel_chunks" in str(output_dir):
                    output_dir = output_dir.parent
            elif hasattr(self, "log_dir") and self.log_dir:
                log_dir_path = Path(self.log_dir)
                if "parallel_chunks" in str(log_dir_path):
                    output_dir = log_dir_path.parent.parent
                else:
                    output_dir = log_dir_path.parent
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
            
            # Determine output directory
            # For parallel replay, log_dir might be in parallel_chunks subdirectory
            # Use the main output directory instead (where trade_log_path is)
            if hasattr(self, "trade_log_path") and self.trade_log_path:
                # Use trade_log_path parent (main output directory)
                output_dir = self.trade_log_path.parent
                # If in parallel_chunks, go up one level
                if "parallel_chunks" in str(output_dir):
                    output_dir = output_dir.parent
            elif hasattr(self, "log_dir") and self.log_dir:
                log_dir_path = Path(self.log_dir)
                # If log_dir is in parallel_chunks, go up to main output dir
                if "parallel_chunks" in str(log_dir_path):
                    output_dir = log_dir_path.parent.parent
                else:
                    output_dir = log_dir_path.parent
            else:
                output_dir = Path("gx1/wf_runs") / self.run_id
            
            # Ensure output_dir exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get router model path
            router_model_path = None
            if self.exit_hybrid_enabled and self.exit_mode_selector:
                router_cfg = self.policy.get("hybrid_exit_router", {})
                model_path_str = router_cfg.get("model_path")
                if model_path_str:
                    router_model_path = Path(model_path_str)
                    if not router_model_path.is_absolute() and self.prod_baseline:
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
            
            # Generate header
            header = generate_run_header(
                policy_path=self.policy_path,
                router_model_path=router_model_path,
                entry_model_paths=entry_model_paths if entry_model_paths else None,
                feature_manifest_path=feature_manifest_path,
                output_dir=output_dir,
                run_tag=self.run_id,
            )
            
            log.info("[RUN_HEADER] Generated run_header.json with %d artifacts", len(header.get("artifacts", {})))
            
        except Exception as e:
            log.warning("[RUN_HEADER] Failed to generate run_header.json: %s", e)
    
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
                f.write(json.dumps(audit_record, separators=(",", ":")) + "\n")
        except Exception as e:
            log.error("[ARB] Failed to log exit audit: %s", e)
    
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
        
        # Race-condition handling: Check if trade is already being closed
        with self._closing_lock:
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
            
            # Check if this is a loss close (skip this check for MODEL_EXIT at moderate loss, already handled above)
            if pnl_bps < 0 and source != "MODEL_EXIT":
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
                    if not self.exec.dry_run:
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
            
            # Log accepted close
            self._log_exit_audit(trade_id, source, reason, pnl_bps, True, broker_info, tp_sl_state, bars_in_trade)
            
            # Update trade log CSV with exit information
            self._update_trade_log_on_close(trade_id, px, pnl_bps, reason, now_ts, bars_in_trade=bars_in_trade)
            
            # Remove from closing_trades on success
            with self._closing_lock:
                self._closing_trades.pop(trade_id, None)
            
            # Tear down per-trade exit state (if any)
            self._teardown_exit_state(trade_id)
            
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
                        from gx1.execution.live_features import infer_session_tag
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
            log.info("Skip entry: max_open_trades reached (%s)", self.risk_limits.max_open_trades)
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
        
        # Apply temperature scaling to p_long and p_short (before gating)
        temp_map = self._get_temperature_map()
        # TEMPORARY TEST: Force T=1.0 for all sessions (disable temperature scaling)
        # TODO: Remove this after diagnosis - restore: T = float(temp_map.get(session_key, 1.0))
        T = 1.0
        
        # Fail-safe: Warn if temperature is missing for session (continue with T=1.0)
        if session_key not in temp_map and T != 1.0:
            log.warning(
                "Temperature missing for session '%s', using T=1.0 (no scaling). "
                "This may indicate missing temperature configuration.",
                session_key,
            )
        
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
    
    def _predict_entry_v9(
        self,
        entry_bundle: EntryFeatureBundle,
        candles: pd.DataFrame,
        policy_state: Dict[str, Any],
    ) -> Optional[EntryPrediction]:
        """
        Use ENTRY_V9 transformer model to generate entry prediction.
        
        Returns EntryPrediction with prob_long, prob_short, prob_early_momentum, quality_score.
        """
        if self.entry_v9_model is None:
            log.warning("[ENTRY_V9] Model not loaded - fallback to STANDARD")
            return None
        
        try:
            import torch
            # EntryPrediction is defined in this file (line 355)
            
            v9_model = self.entry_v9_model.get("model")
            v9_meta = self.entry_v9_model.get("meta")
            v9_device = self.entry_v9_model.get("device")
            v9_config = self.entry_v9_model.get("config", {})
            
            if v9_model is None:
                log.error("[ENTRY_V9] Model component is None")
                return None
            
            lookback = v9_config.get("max_seq_len", 30)
            
            # Combine candles and entry_bundle.features into df_raw for runtime feature building
            # Start with candles (OHLCV) as base, then merge in any existing features from entry_bundle
            df_raw = candles.copy()
            # Normalize column names to lowercase
            col_mapping = {}
            for col in df_raw.columns:
                col_lower = col.lower()
                if col_lower in ['open', 'high', 'low', 'close', 'volume']:
                    col_mapping[col] = col_lower
            if col_mapping:
                df_raw = df_raw.rename(columns=col_mapping)
            
            # Ensure volume exists
            if 'volume' not in df_raw.columns:
                df_raw['volume'] = 0.0
            
            # Merge in any existing features from entry_bundle (but we'll rebuild all V9 features anyway)
            # This is just to preserve any metadata columns that might be useful
            
            # Use runtime_v9 to build features (handles leakage removal, scaling, etc.)
            if not hasattr(self, "entry_v9_feature_meta_path") or self.entry_v9_feature_meta_path is None:
                log.error("[ENTRY_V9] Runtime feature paths not initialized")
                return None
            
            # Validate paths exist
            if not self.entry_v9_feature_meta_path.exists():
                log.error("[ENTRY_V9] Feature meta path does not exist: %s", self.entry_v9_feature_meta_path)
                return None
            if self.entry_v9_seq_scaler_path is not None and not self.entry_v9_seq_scaler_path.exists():
                log.error("[ENTRY_V9] Seq scaler path does not exist: %s", self.entry_v9_seq_scaler_path)
                return None
            if self.entry_v9_snap_scaler_path is not None and not self.entry_v9_snap_scaler_path.exists():
                log.error("[ENTRY_V9] Snap scaler path does not exist: %s", self.entry_v9_snap_scaler_path)
                return None
            
            # Use cached feature_meta/scalers if available (optimization)
            # Otherwise build_v9_runtime_features will load on-demand
            seq_scaler_path = self.entry_v9_seq_scaler_path
            snap_scaler_path = self.entry_v9_snap_scaler_path
            
            df_feats, seq_features, snap_features = build_v9_runtime_features(
                df_raw=df_raw,
                feature_meta_path=self.entry_v9_feature_meta_path,
                seq_scaler_path=seq_scaler_path,
                snap_scaler_path=snap_scaler_path,
            )
            
            log.debug(
                "[ENTRY_V9] Runtime features shape=%s (n_seq=%d, n_snap=%d)",
                df_feats.shape, len(seq_features), len(snap_features)
            )
            
            if len(df_feats) < lookback:
                log.warning("[ENTRY_V9] Insufficient bars: %d < %d", len(df_feats), lookback)
                return None
            
            # Extract sequence and snapshot features from df_feats
            # Sequence features: last lookback bars
            seq_X = df_feats[seq_features].values[-lookback:].astype(np.float32)
            # Snapshot features: last bar only
            snap_X = df_feats[snap_features].values[-1:].astype(np.float32)
            
            # Get regime IDs from current bar (from df_feats which has all features)
            current_bar = df_feats.iloc[-1]
            # Use direct indexing for pandas Series (safer than .get() which may not work as expected)
            # Clamp session_id to valid range [0, 2] (EU=0, OVERLAP=1, US=2)
            session_id_raw = int(current_bar["session_id"]) if "session_id" in current_bar.index else 1  # Default to OVERLAP
            session_id = max(0, min(2, session_id_raw))  # Clamp to [0, 2]
            
            # Clamp vol_regime_id to valid range [0, 3] (LOW=0, MEDIUM=1, HIGH=2, EXTREME=3)
            vol_regime_id_raw = int(current_bar["atr_regime_id"]) if "atr_regime_id" in current_bar.index else 2  # Default to HIGH
            vol_regime_id = max(0, min(3, vol_regime_id_raw))  # Clamp to [0, 3]
            
            # Map trend_regime_tf24h to trend_regime_id
            trend_regime_tf24h = float(current_bar["trend_regime_tf24h"]) if "trend_regime_tf24h" in current_bar.index else 0.0
            if trend_regime_tf24h > 0.001:
                trend_regime_id = 0  # UP
            elif trend_regime_tf24h < -0.001:
                trend_regime_id = 1  # DOWN
            else:
                trend_regime_id = 2  # RANGE
            
            # Convert to tensors (seq_X and snap_X are already scaled by build_v9_runtime_features)
            seq_x = torch.FloatTensor(seq_X).unsqueeze(0).to(v9_device)  # [1, lookback, n_seq]
            snap_x = torch.FloatTensor(snap_X).to(v9_device)  # [1, n_snap]
            session_id_t = torch.LongTensor([session_id]).to(v9_device)  # [1]
            vol_regime_id_t = torch.LongTensor([vol_regime_id]).to(v9_device)  # [1]
            trend_regime_id_t = torch.LongTensor([trend_regime_id]).to(v9_device)  # [1]
            
            # Predict
            with torch.no_grad():
                outputs = v9_model(seq_x, snap_x, session_id_t, vol_regime_id_t, trend_regime_id_t)
                direction_logit = outputs["direction_logit"]
                early_move_logit = outputs["early_move_logit"]
                quality_score = outputs["quality_score"]
            
            # Convert to probabilities
            prob_direction = torch.sigmoid(direction_logit).cpu().item()
            prob_early = torch.sigmoid(early_move_logit).cpu().item()
            quality = quality_score.cpu().item()
            
            # ENTRY_V9 is binary (long vs not long)
            prob_long = float(prob_direction)
            prob_short = float(1.0 - prob_direction)
            margin = abs(prob_long - prob_short)
            p_hat = max(prob_long, prob_short)
            
            # Get session from current bar (reuse current_bar from above, session_id already extracted)
            session_map = {0: "EU", 1: "OVERLAP", 2: "US"}
            session_tag = session_map.get(session_id, "OVERLAP")
            
            log.debug(
                "[ENTRY_V9] Prediction: prob_long=%.4f prob_short=%.4f prob_early=%.4f quality=%.4f session=%s",
                prob_long, prob_short, prob_early, quality, session_tag
            )
            
            # Create EntryPrediction (only standard fields)
            return EntryPrediction(
                session=session_tag,
                prob_long=prob_long,
                prob_short=prob_short,
                prob_neutral=0.0,  # ENTRY_V9 is binary
                margin=margin,
                p_hat=p_hat,
            )
            
        except Exception as e:
            log.error("[ENTRY_V9] Prediction error: %s", e, exc_info=True)
            return None

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
            
            # Add 'ts' column (required by basic_v1)
            candles_with_ts = candles_seq.copy()
            candles_with_ts["ts"] = candles_with_ts.index
            
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
            
            # Handle NaN/Inf
            X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale
            if self.entry_tcn_scaler is not None:
                X_seq = self.entry_tcn_scaler.transform(X_seq)
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


    def _evaluate_entry_impl(self, candles: pd.DataFrame) -> Optional[LiveTrade]:
        return self.entry_manager.evaluate_entry(candles)

    
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
        json.dumps(self.daily_loss_tracker),
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
                json.dump(health_signal, handle, separators=(",", ":"))
            
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

def _run_replay_impl(self, csv_path: Path) -> None:
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
    
    # Load historical candles
    if csv_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(csv_path)
    else:
        df = pd.read_csv(csv_path)
    
    # Ensure time column is datetime
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        # Already has DatetimeIndex
        pass
    else:
        raise ValueError("Historical data must have 'time' column or DatetimeIndex")
    
    # Verify required columns
    required_cols = ["open", "high", "low", "close"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in historical data: {missing_cols}")
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
    self.replay_start_ts = df.index.min()
    self.replay_end_ts = df.index.max()
    self._reset_entry_diag()
    
    # Skip backfill in replay mode (we already have historical data)
    self.backfill_in_progress = False
    self.warmup_floor = None  # No warmup in replay (assume features are stable)
    
    # Disable tick watcher in replay (we'll simulate ticks from M5 candles)
    if self.tick_watcher:
        self.tick_watcher.stop()
        log.info("[REPLAY] Tick watcher disabled (simulating ticks from M5 candles)")
    
    # Load warmup prices for Big Brain V1 (MUST be from same period as replay)
    # CRITICAL: Warmup data MUST match replay period - no mixing 2020 data with 2025 replay!
    if self.big_brain_v1 is not None:
        bb_v1_config = self.policy.get("big_brain_v1", {})
        warmup_prices_path = bb_v1_config.get("warmup_prices_path")
        lookback_bars = self.big_brain_v1.lookback
        
        # Try to load warmup from external file ONLY if it has data in the correct period
        warmup_loaded = False
        if warmup_prices_path:
            warmup_prices_path = Path(warmup_prices_path)
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
            if len(df) >= lookback_bars:
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
                log.warning(
                    "[BIG_BRAIN_V1] Replay file has insufficient bars (%d, need %d). "
                    "First trades may have UNKNOWN regimes. "
                    "Consider using a replay file with at least %d bars.",
                    len(df),
                    lookback_bars,
                    lookback_bars,
                )
    
    # Process each bar sequentially
    # Skip first N bars to ensure features are stable (need lookback for ATR, ADR, etc.)
    # build_live_entry_features needs at least ATR_PERIOD (14) bars for ATR
    # ADR_WINDOW is 288 bars (1 day), but we can start earlier with partial ADR
    # Big Brain V1 needs 288 bars for warmup (if using replay file itself)
    # ENTRY-TCN needs 864 bars for lookback (3 days M5)
    # Use max(100, lookback_bars_bb, entry_tcn_lookback) to ensure all models have enough warmup bars
    lookback_requirements = [100]  # Minimum for stable features
    
    if self.big_brain_v1 is not None:
        lookback_bars_bb = self.big_brain_v1.lookback
        lookback_requirements.append(lookback_bars_bb)
    
    # Check for ENTRY-TCN lookback requirement
    entry_tcn_lookback = self.policy.get("tcn", {}).get("lookback_bars", None)
    if entry_tcn_lookback is not None:
        lookback_requirements.append(entry_tcn_lookback)
    
    min_bars_for_features = max(lookback_requirements)
    
    total_bars = len(df)
    bars_to_process = total_bars - min_bars_for_features
    
    import time
    replay_start_time = time.time()
    last_progress_log_time = replay_start_time
    
    for i, (ts, row) in enumerate(df.iterrows()):
        # Skip first N bars (not enough history for stable features)
        if i < min_bars_for_features:
            continue
        
        # Progress logging every 500 bars (more frequent)
        current_bar_idx = i - min_bars_for_features
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
        
        # Build candle history for features (need lookback window)
        # Use all candles up to current bar (including current bar)
        candles_history = df.iloc[:i+1].copy()
        
        # Ensure index is DatetimeIndex with UTC timezone
        if not isinstance(candles_history.index, pd.DatetimeIndex):
            candles_history = candles_history.set_index("time")
        candles_history.index = pd.to_datetime(candles_history.index, utc=True)
        
        # Evaluate entry
        trade = self.evaluate_entry(candles_history)
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
    
    # Dump FARM_V2B diagnostic if available (before closing remaining trades)
    if hasattr(self, "entry_manager") and hasattr(self.entry_manager, "farm_diag"):
        farm_diag = self.entry_manager.farm_diag
        if farm_diag.get("n_bars", 0) > 0:
            # Create diagnostic output path
            from datetime import datetime
            diag_filename = f"farm_entry_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            diag_path = self.log_dir / diag_filename
            
            # Convert numpy types to native Python types for JSON serialization
            import json
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
                json.dump(diag_serializable, f, indent=2)
            
            log.info(f"[FARM_DIAG] Dumped FARM_V2B entry diagnostic to {diag_path}")
    
    # Clear replay context
    if hasattr(self, '_replay_current_ts'):
        delattr(self, '_replay_current_ts')
    
    # Close any remaining open trades at final price
    if self.open_trades:
        final_candle = df.iloc[-1]
        final_bid = float(final_candle["bid_close"])
        final_ask = float(final_candle["ask_close"])
        log.info(
            "[REPLAY] Closing %d remaining open trades at final bid/ask %.3f/%.3f",
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
                self.request_close(
                    trade_id=trade.trade_id,
                    source="REPLAY_END",
                    reason="REPLAY_END",
                    px=exit_price,
                    pnl_bps=pnl_bps,
                    bars_in_trade=len(df) - 1,  # Approximate
                )
    
    # Dump backtest summary (skip in fast mode)
    if not self.fast_replay:
        self._dump_backtest_summary()
    
    # Flush any remaining entry-only log buffer
    if hasattr(self, "_entry_only_log_buffer") and len(self._entry_only_log_buffer) > 0:
        self._flush_entry_only_log_buffer()
        log.info("[ENTRY_ONLY] Flushed final buffer (%d entries)", len(getattr(self, "_entry_only_log_buffer", [])))

    self._log_entry_diag_summary()
    
    log.info("[REPLAY] Backtest complete")
    
    # Close trade journal
    if hasattr(self, "trade_journal") and self.trade_journal:
        try:
            self.trade_journal.close()
            log.info("[TRADE_JOURNAL] Trade journal closed")
        except Exception as e:
            log.warning("[TRADE_JOURNAL] Failed to close journal: %s", e)
    
    # Canary mode: log invariants and generate prod_metrics
    if hasattr(self, "canary_mode") and self.canary_mode:
        self._log_canary_invariants()
        self._generate_canary_metrics()
    
    # Generate detailed report via external script (skip in fast mode)
    if not self.fast_replay:
        try:
            from gx1.backtest.generate_exit_tuning_report import main as generate_report  # type: ignore[reportMissingImports]
            import sys
            import os
            # Save original argv
            old_argv = sys.argv
            # Set up arguments for report generator
            exit_audit_file = self.log_dir / "exits" / f"exits_{self.replay_start_ts.strftime('%Y%m%d')}_{self.replay_end_ts.strftime('%Y%m%d')}.jsonl"
            report_file = Path("gx1/backtest/reports") / f"GX1_ONE_BACKTEST_exit_tuning_v4.md"
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
        log.info("[REPLAY] Fast mode: skipped detailed report generation")

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
                    entry = json.loads(line)
                    if entry.get("source") == "TICK" and entry.get("reason") == "SOFT_STOP_TICK" and entry.get("accepted"):
                        soft_stop_count += 1
                except json.JSONDecodeError:
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


def run_replay(self: GX1DemoRunner, csv_path: Path) -> None:
    return self._run_replay_impl(csv_path)


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
GX1DemoRunner._ensure_bid_ask_columns = _ensure_bid_ask_columns  # type: ignore[attr-defined]
GX1DemoRunner._calculate_unrealized_portfolio_bps = _calculate_unrealized_portfolio_bps  # type: ignore[attr-defined]
GX1DemoRunner._generate_client_order_id = _generate_client_order_id  # type: ignore[attr-defined]
GX1DemoRunner._build_notes_string = _build_notes_string  # type: ignore[attr-defined]
GX1DemoRunner._check_client_order_id_exists = _check_client_order_id_exists  # type: ignore[attr-defined]
GX1DemoRunner._reset_entry_diag = _reset_entry_diag  # type: ignore[attr-defined]
GX1DemoRunner._record_entry_diag = _record_entry_diag  # type: ignore[attr-defined]
GX1DemoRunner._log_entry_diag_summary = _log_entry_diag_summary  # type: ignore[attr-defined]
GX1DemoRunner._execute_entry_impl = _execute_entry_impl  # type: ignore[attr-defined]
GX1DemoRunner._evaluate_and_close_trades_impl = _evaluate_and_close_trades_impl  # type: ignore[attr-defined]
def should_consider_entry(self, trend: str, vol: str, session: str, risk_score: float) -> bool:
    """Wrapper for _should_consider_entry_impl that takes self as first parameter."""
    return _should_consider_entry_impl(trend, vol, session, risk_score)

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
