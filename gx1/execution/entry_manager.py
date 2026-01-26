from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from collections import deque, defaultdict

import logging
import os
import numpy as np
import pandas as pd
import time
import traceback
import uuid

from gx1.execution.live_features import build_live_entry_features, infer_session_tag
from gx1.sniper.policy.sniper_regime_size_overlay import apply_size_overlay
from gx1.sniper.policy.sniper_q4_cchop_size_overlay import apply_q4_cchop_overlay
from gx1.sniper.policy.sniper_q4_atrend_size_overlay import apply_q4_atrend_overlay
from gx1.sniper.policy.sniper_q4_eu_timing_size_overlay import apply_q4_eu_timing_overlay
from gx1.sniper.policy.runtime_regime_inputs import get_runtime_regime_inputs

# Optional import for V10.1 size_overlay (OFFLINE ONLY)
try:
    from gx1.runtime.overlays.entry_v10_1_size_overlay import load_entry_v10_1_size_overlay
    ENTRY_V10_1_SIZE_OVERLAY_AVAILABLE = True
except ImportError:
    ENTRY_V10_1_SIZE_OVERLAY_AVAILABLE = False

log = logging.getLogger(__name__)

# Fix: Log warning after log is defined
if not ENTRY_V10_1_SIZE_OVERLAY_AVAILABLE:
    log.warning("[ENTRY_V10_1_SIZE_OVERLAY] Module not available - V10.1 size overlay disabled")

if TYPE_CHECKING:
    from gx1.execution.oanda_demo_runner import GX1DemoRunner, LiveTrade


class EntryManager:
    def __init__(self, runner: "GX1DemoRunner", exit_config_name: Optional[str] = None) -> None:
        super().__setattr__("_runner", runner)
        # Explicit exit_config_name (injected from runner, no coupling to runner.policy)
        self.exit_config_name = exit_config_name
        # Initialize entry_feature_telemetry if required
        self.entry_feature_telemetry = None
        require_telemetry = os.environ.get("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
        if require_telemetry:
            from gx1.execution.entry_feature_telemetry import EntryFeatureTelemetryCollector
            output_dir = getattr(runner, "output_dir", None)
            self.entry_feature_telemetry = EntryFeatureTelemetryCollector(output_dir=output_dir)
        # Entry telemetry state (SNIPER/NY-first, accumulated over replay)
        # DESIGN CONTRACT: Stage-0 (precheck) vs Stage-1 (candidate) counters
        self.entry_telemetry = {
            # Core counters (SNIPER telemetry contract)
            "n_cycles": 0,  # Total bar cycles evaluated
            "n_eligible_hard": 0,  # OPPGAVE 2: Cycles that passed hard eligibility
            "n_eligible_cycles": 0,  # OPPGAVE 2: Cycles that passed hard + soft eligibility
            "n_precheck_pass": 0,  # Bars that passed Stage-0 (prediction allowed)
            "n_predictions": 0,  # Predictions actually produced (V10/V9 OK, finite prob_long)
            "n_candidates": 0,  # Valid predictions (finite p_long) - same as n_predictions
            "n_candidate_pass": 0,  # Candidates that passed Stage-1 (after policy gates, before trade)
            "n_trades_created": 0,  # Trades actually created (SINGLE SOURCE OF TRUTH)
            "p_long_values": [],  # p_long values for all candidates (for stats)
            "candidate_sessions": {},  # Candidate session distribution
            "trade_sessions": {},  # Trade session distribution (separate from candidates)
            # Entry journaling atomicity counters (OPPGAVE RUNTIME)
            "n_entry_snapshots_written": 0,  # Number of entry_snapshots successfully logged
            "n_entry_snapshots_failed": 0,  # Number of entry_snapshots that failed to log
            # OPPGAVE 2: Context features telemetry
            "n_context_built": 0,  # Number of context features successfully built
            "n_context_missing_or_invalid": 0,  # Number of context features that failed to build
            # DEL 4: Ctx model telemetry
            "n_ctx_model_calls": 0,  # Number of ctx model calls (only when ctx-modell is active)
            # DEL D: CTX consumption proof telemetry
            "ctx_proof_pass_count": 0,  # Number of ctx proof checks that passed
            "ctx_proof_fail_count": 0,  # Number of ctx proof checks that failed
            # Log noise reduction: vol_regime=UNKNOWN counter (for SNIPER replay)
            "vol_regime_unknown_count": 0,  # Number of times vol_regime=UNKNOWN caused guard rejection
        }
        # PREBUILT telemetry: track evaluate_entry() calls and prebuilt_available gate
        self.eval_calls_total = 0
        self.eval_calls_prebuilt_gate_true = 0
        self.eval_calls_prebuilt_gate_false = 0
        # KILL-CHAIN telemetry (READ-ONLY, SSoT counters for where trades die)
        # Design: monotonic counters, no resets mid-run; stable reason codes only.
        self.killchain_version = 1
        self.killchain_n_entry_pred_total = 0
        self.killchain_n_above_threshold = 0
        self.killchain_n_after_session_guard = 0
        self.killchain_n_after_vol_guard = 0
        self.killchain_n_after_regime_guard = 0
        self.killchain_n_after_risk_sizing = 0
        self.killchain_n_trade_create_attempts = 0
        self.killchain_n_trade_created = 0
        self.killchain_block_reason_counts = {
            "BLOCK_BELOW_THRESHOLD": 0,
            "BLOCK_SESSION": 0,
            "BLOCK_VOL": 0,
            "BLOCK_REGIME": 0,
            "BLOCK_RISK": 0,
            "BLOCK_POSITION_LIMIT": 0,
            "BLOCK_COOLDOWN": 0,
            "BLOCK_UNKNOWN": 0,
        }
        self.killchain_unknown_examples: List[Dict[str, Any]] = []
        # Legacy FARM_V2B diagnostic state (DEPRECATED - SNIPER/NY uses entry_telemetry)
        # Kept as empty dict for backward compatibility (export scripts may reference it)
        # TODO: Remove after verification that no external scripts depend on farm_diag
        self.farm_diag = {}
        # Stage-0 reason tracking (for breakdown reporting)
        self.stage0_reasons = defaultdict(int)
        self.stage0_total_considered = 0
        # Veto counters separated by stage
        # OPPGAVE 2: Hard eligibility veto counters (before feature build)
        self.veto_hard = {
            "veto_hard_warmup": 0,
            "veto_hard_session": 0,
            "veto_hard_spread": 0,
            "veto_hard_killswitch": 0,
        }
        # OPPGAVE 2: Soft eligibility veto counters (after minimal cheap computation)
        self.veto_soft = {
            "veto_soft_vol_regime_extreme": 0,
        }
        # Stage-0 (precheck): before prediction
        self.veto_pre = {
            "veto_pre_warmup": 0,
            "veto_pre_session": 0,
            "veto_pre_regime": 0,
            "veto_pre_spread": 0,
            "veto_pre_atr": 0,
            "veto_pre_killswitch": 0,
            "veto_pre_model_missing": 0,
            "veto_pre_nan_features": 0,
        }
        # Stage-1 (candidate): after prediction, before trade
        self.veto_cand = {
            "veto_cand_threshold": 0,
            "veto_cand_risk_guard": 0,
            "veto_cand_max_trades": 0,
            "veto_cand_big_brain": 0,
        }
        # Legacy veto_counters (DEPRECATED - SNIPER/NY uses veto_pre/veto_cand)
        # Kept as empty dict for backward compatibility (export scripts may reference it)
        # TODO: Remove after verification that no external scripts depend on veto_counters
        self.veto_counters = {}
        # Track threshold used for diagnostics
        self.threshold_used = None
        # Track p_long stats
        self.p_long_values = []
        self.cluster_guard_history = deque(maxlen=600)
        self.cluster_guard_atr_median: Optional[float] = None
        self.spread_history = deque(maxlen=600)
        # V10 diagnostic counters
        self.n_v10_calls = 0
        self.n_v10_pred_ok = 0
        self.n_v10_pred_none_or_nan = 0
        self._v10_log_count = 0  # Track first 3 calls for logging
        # SNIPER risk guard (initialized lazily when needed)
        self._sniper_risk_guard: Optional[Any] = None
        # SNIPER cycle logging state (rate-limited, per engine)
        self._sniper_last_log_reason: Optional[str] = None
        self._sniper_log_counter: int = 0
        # SNIPER shadow threshold sweep (for safe threshold testing)
        self._shadow_thresholds: List[float] = self._load_shadow_thresholds()
        self._shadow_journal_path: Optional[str] = None
        # Entry Critic V1 (shadow-only mode: score but don't gate)
        self._entry_critic_model: Optional[Any] = None
        self._entry_critic_meta: Optional[Dict[str, Any]] = None
        self._entry_critic_feature_order: Optional[List[str]] = None
        self._load_entry_critic()

    def __getattr__(self, name: str):
        return getattr(self._runner, name)

    def __setattr__(self, name: str, value):
        if name == "_runner":
            super().__setattr__(name, value)
        else:
            setattr(self._runner, name, value)

    def _killchain_inc_reason(self, reason_code: str) -> None:
        """
        Increment kill-chain block reason counter.

        Args:
            reason_code: One of the stable BLOCK_* reason codes.
        """
        if reason_code not in self.killchain_block_reason_counts:
            reason_code = "BLOCK_UNKNOWN"
        self.killchain_block_reason_counts[reason_code] += 1

    def _killchain_record_unknown(self, example: Dict[str, Any]) -> None:
        """
        Record up to 5 unknown-block examples (deterministic, small).

        Args:
            example: Context dict (timestamp, session, vol_regime, etc.)
        """
        if len(self.killchain_unknown_examples) >= 5:
            return
        self.killchain_unknown_examples.append(example)

    @staticmethod
    def _percentile_from_history(history: deque, value: Optional[float]) -> Optional[float]:
        """Return percentile rank (0-100) of value relative to a deque history."""
        if value is None or not history:
            return None
        try:
            arr = np.array(history, dtype=float)
        except Exception:
            return None
        if arr.size == 0 or not np.isfinite(value):
            return None
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        sorted_arr = np.sort(arr)
        idx = np.searchsorted(sorted_arr, value, side="right")
        pct = (idx / float(sorted_arr.size)) * 100.0
        return max(0.0, min(100.0, float(pct)))

    @staticmethod
    def _compute_range_features(candles: pd.DataFrame, window: int = 96) -> tuple[float, float]:
        """
        Compute range_pos and distance_to_range features from OHLC candles.
        
        Args:
            candles: DataFrame with 'high', 'low', 'close' columns (or bid/ask variants)
            window: Rolling window size (default 96 bars ≈ 1 day for M15)
        
        Returns:
            (range_pos, distance_to_range) both in [0.0, 1.0]
            Falls back to (0.5, 0.5) if insufficient data or error.
        """
        eps = 1e-12
        default_range_pos = 0.5
        default_distance = 0.5
        
        try:
            if len(candles) < window:
                # Not enough bars - return defaults
                return (default_range_pos, default_distance)
            
            # Try to get high/low/close columns (prefer direct, fallback to bid/ask mid)
            # IMPORTANT: Use consistent source - don't mix direct OHLC with bid/ask
            has_direct = all(col in candles.columns for col in ['high', 'low', 'close'])
            has_bid_ask = all(col in candles.columns for col in ['bid_high', 'ask_high', 'bid_low', 'ask_low', 'bid_close', 'ask_close'])
            
            if not has_direct and not has_bid_ask:
                return (default_range_pos, default_distance)
            
            # Use last window bars (exclude current incomplete bar if possible)
            # For range calculation, use window bars ending at -2 (last closed bar)
            # If not enough bars, use what we have
            if len(candles) >= window + 1:
                # We have at least window+1 bars, use window bars ending at -2 (last closed)
                recent = candles.iloc[-(window+1):-1]  # Exclude last bar (may be incomplete)
            else:
                # Not enough bars to exclude last - use all available
                recent = candles.tail(window)
            
            if has_direct:
                # Use direct OHLC consistently
                high_vals = recent['high'].values
                low_vals = recent['low'].values
                close_vals = recent['close'].values
            else:
                # Use bid/ask mid-price consistently
                high_vals = (recent['bid_high'].values + recent['ask_high'].values) / 2.0
                low_vals = (recent['bid_low'].values + recent['ask_low'].values) / 2.0
                close_vals = (recent['bid_close'].values + recent['ask_close'].values) / 2.0
            
            # Compute rolling max/min: max and min of the last window bars
            # This is explicitly: max(high[-window:]) and min(low[-window:])
            range_hi = float(np.max(high_vals))
            range_lo = float(np.min(low_vals))
            
            # Current price reference: use last CLOSED bar's close
            # If we excluded last bar, use -1 from recent (which is -2 from original)
            # If we didn't exclude, use -1 from recent (which is -1 from original)
            price_ref = float(close_vals[-1])
            
            # Compute range_pos
            denom = max(eps, range_hi - range_lo)
            range_pos_raw = (price_ref - range_lo) / denom
            range_pos = max(0.0, min(1.0, float(range_pos_raw)))
            
            # Compute distance_to_range (edge-distance)
            dist_edge = min(range_pos, 1.0 - range_pos)  # 0 at edge, 0.5 in middle
            distance_to_range = dist_edge * 2.0  # Scale to 0..1 (0=edge, 1=mid)
            distance_to_range = max(0.0, min(1.0, float(distance_to_range)))
            
            return (range_pos, distance_to_range)
            
        except Exception:
            # Any error -> return defaults
            return (default_range_pos, default_distance)
    
    # OPPGAVE 2: Hard eligibility reason constants (before feature build)
    HARD_ELIGIBILITY_WARMUP = "HARD_WARMUP"
    HARD_ELIGIBILITY_SESSION_BLOCK = "HARD_SESSION_BLOCK"
    HARD_ELIGIBILITY_SPREAD_CAP = "HARD_SPREAD_CAP"
    HARD_ELIGIBILITY_KILLSWITCH = "HARD_KILLSWITCH"
    # OPPGAVE 2: Soft eligibility reason constants (after minimal cheap computation)
    SOFT_ELIGIBILITY_VOL_REGIME_EXTREME = "SOFT_VOL_REGIME_EXTREME"
    
    def _check_hard_eligibility(
        self,
        candles: pd.DataFrame,
        policy_state: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """
        OPPGAVE 2: Check hard eligibility BEFORE feature build and model inference.
        
        This is a binary "no trade possible" check. If eligible=False, we STOP
        immediately - no feature build, no model call, no candidate.
        
        Args:
            candles: Candles DataFrame
            policy_state: Policy state dict (may contain regime info)
        
        Returns:
            (eligible: bool, reason: Optional[str])
            If eligible=False, reason is one of the HARD_ELIGIBILITY_* constants
        """
        current_ts = candles.index[-1] if len(candles) > 0 else None
        
        # 1. Warmup check
        warmup_bars = getattr(self, "warmup_bars", 288)
        if len(candles) < warmup_bars:
            return False, self.HARD_ELIGIBILITY_WARMUP
        
        # 2. Session check (SNIPER: ASIA blocked)
        policy_sniper_cfg = self.policy.get("entry_v9_policy_sniper", {})
        use_sniper = policy_sniper_cfg.get("enabled", False)
        
        if use_sniper:
            # Get current session
            current_session = policy_state.get("session")
            if not current_session:
                from gx1.execution.live_features import infer_session_tag
                current_session = infer_session_tag(current_ts).upper()
                policy_state["session"] = current_session
            
            allowed_sessions = policy_sniper_cfg.get("allowed_sessions", ["EU", "OVERLAP", "US"])
            if current_session not in allowed_sessions:
                return False, self.HARD_ELIGIBILITY_SESSION_BLOCK
        
        # 3. Spread check (hard cap) - compute from candles before feature build
        # NOTE: Vol regime check moved to soft eligibility (after minimal ATR computation)
        spread_bps = self._get_spread_bps_before_features(candles)
        if spread_bps is not None:
            spread_hard_cap_bps = policy_sniper_cfg.get("spread_hard_cap_bps", 100.0)  # Default 100 bps
            if spread_bps > spread_hard_cap_bps:
                return False, self.HARD_ELIGIBILITY_SPREAD_CAP
        
        # 5. Kill-switch check
        if self._is_kill_switch_active():
            return False, self.HARD_ELIGIBILITY_KILLSWITCH
        
        return True, None
    
    def _get_spread_bps_before_features(self, candles: pd.DataFrame) -> Optional[float]:
        """
        Get spread in bps from candles BEFORE feature build.
        
        This is needed for hard eligibility check.
        """
        if candles.empty:
            return None
        
        try:
            # Try to get spread from bid/ask columns
            if "bid_close" in candles.columns and "ask_close" in candles.columns:
                bid = candles["bid_close"].iloc[-1]
                ask = candles["ask_close"].iloc[-1]
                if pd.notna(bid) and pd.notna(ask) and bid > 0:
                    spread_price = ask - bid
                    spread_bps = (spread_price / bid) * 10000.0
                    return float(spread_bps)
            # Fallback: try spread column if available
            elif "spread" in candles.columns:
                spread_price = candles["spread"].iloc[-1]
                if pd.notna(spread_price):
                    close = candles.get("close", candles.index)
                    if len(close) > 0:
                        price_ref = float(close.iloc[-1])
                        if price_ref > 0:
                            spread_bps = (spread_price / price_ref) * 10000.0
                            return float(spread_bps)
        except Exception:
            pass
        
        return None
    
    def _is_kill_switch_active(self) -> bool:
        """
        Check if kill-switch is active.
        
        Checks for KILL_SWITCH_ON file in project root.
        """
        try:
            from pathlib import Path
            import os
            
            # Get project root (gx1/execution/../..)
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            kill_flag = project_root / "KILL_SWITCH_ON"
            
            return kill_flag.exists()
        except Exception:
            # If we can't check, assume not active (fail-safe)
            return False
    
    @staticmethod
    def _compute_range_edge_dist_atr(
        candles: pd.DataFrame,
        price_ref: float,
        range_hi: float,
        range_lo: float,
        atr_value: Optional[float],
        window: int = 96,
    ) -> float:
        """
        Compute ATR-normalized distance to nearest range edge.
        
        Args:
            candles: DataFrame with OHLC data (used for consistency check)
            price_ref: Current price reference (last closed bar's close)
            range_hi: Maximum high in the window
            range_lo: Minimum low in the window
            atr_value: ATR in price units (None if unavailable)
            window: Rolling window size (for consistency)
        
        Returns:
            range_edge_dist_atr: ATR-normalized distance to nearest edge [0.0, 10.0]
            Falls back to 0.0 if insufficient data or error.
        """
        eps = 1e-12
        default_value = 0.0
        
        try:
            # Check if we have valid ATR value
            if atr_value is None or not np.isfinite(atr_value) or atr_value <= 0:
                return default_value
            
            # Compute distance to nearest edge in price units
            dist_to_low = max(0.0, price_ref - range_lo)
            dist_to_high = max(0.0, range_hi - price_ref)
            dist_edge_price = min(dist_to_low, dist_to_high)
            
            # Normalize by ATR
            denom = max(eps, atr_value)
            range_edge_dist_atr_raw = dist_edge_price / denom
            
            # Clip to [0.0, 10.0] to cap outliers
            range_edge_dist_atr = max(0.0, min(10.0, float(range_edge_dist_atr_raw)))
            
            return range_edge_dist_atr
            
        except Exception:
            # Any error -> return default
            return default_value
    
    def _compute_cheap_atr_proxy(self, candles: pd.DataFrame, window: int = 14) -> Optional[float]:
        """
        Compute ultra-cheap ATR proxy from raw candles (no feature build required).
        
        This is used for soft eligibility check (vol regime EXTREME detection).
        Cost: O(window) numpy operations, << full feature build.
        
        Args:
            candles: Candles DataFrame with OHLC columns
            window: ATR window (default 14)
        
        Returns:
            ATR proxy in price units, or None if insufficient data
        """
        if candles.empty or len(candles) < window:
            return None
        
        try:
            import numpy as np
            
            # Get OHLC (normalize column names)
            high = candles.get("high", candles.get("high", None))
            low = candles.get("low", candles.get("low", None))
            close = candles.get("close", candles.get("close", None))
            
            if high is None or low is None or close is None:
                return None
            
            # Convert to numpy arrays (last window bars)
            high_arr = high.iloc[-window:].values
            low_arr = low.iloc[-window:].values
            close_arr = close.iloc[-window:].values
            
            # True Range components
            tr1 = high_arr - low_arr  # High - Low
            tr2 = np.abs(high_arr - np.roll(close_arr, 1))  # |High - Prev Close|
            tr3 = np.abs(low_arr - np.roll(close_arr, 1))   # |Low - Prev Close|
            
            # True Range = max of three components
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # ATR = mean of True Range
            atr = np.mean(tr)
            
            return float(atr)
        except Exception:
            return None
    
    def _check_soft_eligibility(
        self,
        candles: pd.DataFrame,
        policy_state: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """
        OPPGAVE 2: Check soft eligibility AFTER minimal cheap computation.
        
        This runs after hard eligibility passes, but before full feature build.
        Uses ultra-cheap ATR proxy to detect EXTREME vol regime.
        
        Args:
            candles: Candles DataFrame
            policy_state: Policy state dict
        
        Returns:
            (eligible: bool, reason: Optional[str])
        """
        # Vol regime EXTREME check using cheap ATR proxy
        atr_proxy = self._compute_cheap_atr_proxy(candles, window=14)
        
        # OPPGAVE 2: Store ATR proxy for context features build (reuse computation)
        self._last_atr_proxy = atr_proxy
        
        # OPPGAVE 2: Store spread_bps for context features build (reuse from hard eligibility)
        spread_bps = self._get_spread_bps_before_features(candles)
        self._last_spread_bps = spread_bps
        
        if atr_proxy is not None:
            # Get current price for normalization
            close = candles.get("close", None)
            if close is not None and len(close) > 0:
                current_price = float(close.iloc[-1])
                if current_price > 0:
                    # Compute ATR in basis points
                    atr_bps = (atr_proxy / current_price) * 10000.0
                    
                    # EXTREME threshold: > 200 bps (conservative, matches typical EXTREME regime)
                    # This is a hard threshold - if ATR is this high, market is too volatile
                    if atr_bps > 200.0:
                        return False, self.SOFT_ELIGIBILITY_VOL_REGIME_EXTREME
        
        return True, None

    def _log_sniper_cycle(
        self,
        ts: pd.Timestamp,
        session: str,
        in_scope: bool,
        warmup_ready: bool,
        degraded: bool,
        spread_bps: Optional[float],
        atr_bps: Optional[float],
        eval_ran: bool,
        reason: str,
        decision: str = "NO-TRADE",
        n_signals: Optional[int] = None,
        p_long: Optional[float] = None,
    ) -> None:
        """
        Log SNIPER cycle summary (rate-limited, SNIPER-only).
        
        Args:
            ts: Current timestamp
            session: Current session (EU/LONDON/NY/OVERLAP/ASIA)
            in_scope: Whether session is in-scope for SNIPER
            warmup_ready: Whether warmup is ready
            degraded: Whether warmup is degraded
            spread_bps: Current spread in bps (if available)
            atr_bps: Current ATR in bps (if available)
            eval_ran: Whether policy evaluation ran
            reason: Reason for no trade (or "TRADE" if trade occurred)
            decision: Final decision (LONG/NO-TRADE)
            n_signals: Number of signals from policy (if available)
            p_long: Probability long (if available)
        """
        # Check if SNIPER policy is enabled
        policy_sniper_cfg = self.policy.get("entry_v9_policy_sniper", {})
        use_sniper = policy_sniper_cfg.get("enabled", False)
        if not use_sniper:
            return  # Not SNIPER - skip logging
        
        # Rate limiting: log every cycle (1 line per cycle max)
        # Format values
        spread_str = f"{spread_bps:.1f}" if spread_bps is not None else "N/A"
        atr_str = f"{atr_bps:.1f}" if atr_bps is not None else "N/A"
        n_signals_str = str(n_signals) if n_signals is not None else "N/A"
        p_long_str = f"{p_long:.3f}" if p_long is not None else "N/A"
        
        # Build log line
        log_line = (
            f"[SNIPER_CYCLE] ts={ts.isoformat()} session={session} in_scope={1 if in_scope else 0} "
            f"warmup_ready={1 if warmup_ready else 0} degraded={1 if degraded else 0} "
            f"spread_bps={spread_str} atr_bps={atr_str} eval={1 if eval_ran else 0} "
            f"reason={reason} decision={decision}"
        )
        
        # Add optional fields if available
        if n_signals is not None:
            log_line += f" n_signals={n_signals_str}"
        if p_long is not None and eval_ran:
            log_line += f" p_long={p_long_str}"
        
        log.info(log_line)

    def _load_shadow_thresholds(self) -> List[float]:
        """
        Load shadow thresholds from env or config (default: [0.55, 0.58, 0.60, 0.62, 0.65]).
        
        Returns:
            List of shadow thresholds (sorted descending)
        """
        import os
        # Try env first
        env_thresholds = os.getenv("SNIPER_SHADOW_THRESHOLDS", "")
        if env_thresholds:
            try:
                thresholds = [float(t.strip()) for t in env_thresholds.split(",")]
                return sorted(thresholds, reverse=True)
            except ValueError:
                log.warning(f"[SNIPER_SHADOW] Invalid SNIPER_SHADOW_THRESHOLDS env: {env_thresholds}, using defaults")
        
        # Default shadow thresholds
        return [0.65, 0.62, 0.60, 0.58, 0.55]
    
    def _load_entry_critic(self) -> None:
        """Load Entry Critic V1 model for shadow-only scoring."""
        try:
            from gx1.rl.entry_critic_runtime import load_entry_critic_v1
            model, meta, feature_order = load_entry_critic_v1()
            self._entry_critic_model = model
            self._entry_critic_meta = meta
            self._entry_critic_feature_order = feature_order
            if model is not None:
                log.info("[ENTRY_CRITIC] Entry Critic V1 loaded (shadow-only mode)")
        except Exception as e:
            log.warning(f"[ENTRY_CRITIC] Failed to load Entry Critic: {e}")
            self._entry_critic_model = None
            self._entry_critic_meta = None
            self._entry_critic_feature_order = None

    def _log_sniper_shadow(
        self,
        ts: pd.Timestamp,
        session: str,
        p_long: float,
        real_threshold: float,
        real_trade: bool,
        spread_bps: Optional[float],
        atr_bps: Optional[float],
        trend_regime: Optional[str],
        vol_regime: Optional[str],
        real_decision: str,
    ) -> None:
        """Log shadow threshold hits and calculate Entry Critic score."""
        """
        Log shadow threshold hits (what would have traded at lower thresholds).
        
        This is SAFE - it only logs and journals, never creates orders.
        """
        # Check if SNIPER policy is enabled
        policy_sniper_cfg = self.policy.get("entry_v9_policy_sniper", {})
        use_sniper = policy_sniper_cfg.get("enabled", False)
        if not use_sniper or not self._shadow_thresholds:
            return  # Not SNIPER or shadow disabled
        
        # Only log when eval=1 and we have p_long (policy evaluation ran)
        # Calculate shadow hits: would_trade = (p_long >= shadow_thr) AND (policy otherwise would approve)
        # For shadow, we assume policy would approve if p_long >= threshold (simplified)
        shadow_hits = {}
        best_hit: Optional[float] = None
        
        for shadow_thr in self._shadow_thresholds:
            # Shadow hit: p_long >= shadow_thr (for trades/day estimate)
            # Also track if it's a "near-miss" (p_long >= shadow_thr AND p_long < real_threshold)
            shadow_hit = p_long >= shadow_thr
            near_miss = shadow_hit and p_long < real_threshold
            
            shadow_hits[shadow_thr] = shadow_hit
            if near_miss and (best_hit is None or shadow_thr < best_hit):
                best_hit = shadow_thr
        
        # Always write to journal (for trades/day estimate, independent of real_threshold)
        # But only log to console if there's a near-miss (to reduce noise)
        has_near_miss = any(
            p_long >= shadow_thr and p_long < real_threshold 
            for shadow_thr in self._shadow_thresholds
        )
        
        # Build shadow hits string
        hits_str = ",".join([f"{thr}:{1 if hit else 0}" for thr, hit in shadow_hits.items()])
        best_hit_str = f"{best_hit:.2f}" if best_hit is not None else "None"
        
        # Entry Critic V1 scoring (shadow-only: calculate but don't gate)
        entry_critic_score: Optional[float] = None
        if self._entry_critic_model is not None and self._entry_critic_feature_order is not None:
            try:
                from gx1.rl.entry_critic_runtime import (
                    prepare_entry_critic_features,
                    score_entry_critic,
                )
                feature_vector = prepare_entry_critic_features(
                    p_long=p_long,
                    spread_bps=spread_bps,
                    atr_bps=atr_bps,
                    trend_regime=trend_regime,
                    vol_regime=vol_regime,
                    session=session,
                    shadow_hits=shadow_hits,
                    real_threshold=real_threshold,
                    feature_order=self._entry_critic_feature_order,
                )
                if feature_vector is not None:
                    entry_critic_score = score_entry_critic(
                        model=self._entry_critic_model,
                        feature_vector=feature_vector,
                    )
            except Exception as e:
                log.debug(f"[ENTRY_CRITIC] Failed to score entry: {e}")
        
        # Write to shadow journal (jsonl) - ALWAYS for trades/day estimate
        self._write_shadow_journal(
            ts=ts,
            session=session,
            p_long=p_long,
            spread_bps=spread_bps,
            atr_bps=atr_bps,
            trend_regime=trend_regime,
            vol_regime=vol_regime,
            shadow_hits=shadow_hits,
            real_threshold=real_threshold,
            real_decision=real_decision,
            entry_critic_score=entry_critic_score,
        )
        
        # Only log to console if near-miss or real trade (to reduce noise)
        if has_near_miss or real_trade:
            log.info(
                f"[SNIPER_SHADOW] ts={ts.isoformat()} session={session} p_long={p_long:.3f} "
                f"real_thr={real_threshold:.2f} real_trade={1 if real_trade else 0} "
                f"shadow_hits={hits_str} best_hit={best_hit_str}"
            )

    def _write_shadow_journal(
        self,
        ts: pd.Timestamp,
        session: str,
        p_long: float,
        spread_bps: Optional[float],
        atr_bps: Optional[float],
        trend_regime: Optional[str],
        vol_regime: Optional[str],
        shadow_hits: Dict[float, bool],
        real_threshold: float,
        real_decision: str,
        entry_critic_score: Optional[float] = None,
    ) -> None:
        """Write shadow hit record to jsonl journal."""
        import json
        from pathlib import Path
        
        # Initialize shadow journal path on first write
        if self._shadow_journal_path is None:
            # Get run-dir from runner if available
            run_dir = None
            if hasattr(self._runner, "run_dir") and self._runner.run_dir:
                run_dir = Path(self._runner.run_dir)
            else:
                # Try to infer from policy or create default
                run_dir = Path("runs/live_demo/shadow")
            
            # Create shadow subdirectory
            shadow_dir = run_dir / "shadow"
            shadow_dir.mkdir(parents=True, exist_ok=True)
            self._shadow_journal_path = shadow_dir / "shadow_hits.jsonl"
        
        # Write record
        record = {
            "ts": ts.isoformat(),
            "session": session,
            "p_long": float(p_long),
            "spread_bps": float(spread_bps) if spread_bps is not None else None,
            "atr_bps": float(atr_bps) if atr_bps is not None else None,
            "trend_regime": trend_regime,
            "vol_regime": vol_regime,
            "shadow_hits": {str(k): bool(v) for k, v in shadow_hits.items()},
            "real_threshold": float(real_threshold),
            "real_decision": real_decision,
            "entry_critic_score_v1": float(entry_critic_score) if entry_critic_score is not None else None,
            "entry_critic_model": self._entry_critic_meta.get("model_type", "entry_critic_v1") if self._entry_critic_meta else None,
        }
        
        try:
            with open(self._shadow_journal_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            log.warning(f"[SNIPER_SHADOW] Failed to write shadow journal: {e}")

    def evaluate_entry(self, candles: pd.DataFrame) -> Optional[LiveTrade]:
        import time
        
        # DEL 3: Reset routing telemetry for this bar (must be called at start of each bar evaluation)
        if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
            self.entry_feature_telemetry.reset_routing_for_next_bar()
        
        # PREBUILT telemetry: track that evaluate_entry() was called
        self.eval_calls_total += 1
        
        # Entry telemetry: increment cycle counter (bar-level)
        self.entry_telemetry["n_cycles"] += 1  # SNIPER telemetry contract
        
        # STEG 1: lookup_attempts må telles FØR hard eligibility (så vi fanger alle bars i PREBUILT mode)
        # Dette må skje helt i starten, før noen early returns
        import os as os_module
        prebuilt_enabled = os_module.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1"
        is_replay = getattr(self._runner, "replay_mode", False)
        
        if is_replay and prebuilt_enabled:
            if not hasattr(self._runner, "lookup_attempts"):
                self._runner.lookup_attempts = 0
                self._runner.lookup_hits = 0
                self._runner.lookup_misses = 0
                self._runner.lookup_miss_details = []  # Store first 3 miss details for footer
            
            # STEG 1: lookup_attempts++ ALLTID når vi er i PREBUILT replay-mode (FØR hard eligibility)
            self._runner.lookup_attempts += 1
        
        # OPPGAVE 2: Hard eligibility check BEFORE feature build and model inference
        # Initialize policy_state early (needed for hard eligibility check)
        policy_state = {}
        
        # Get current session for hard eligibility check
        current_ts = candles.index[-1] if len(candles) > 0 else pd.Timestamp.now(tz="UTC")
        from gx1.execution.live_features import infer_session_tag
        current_session = infer_session_tag(current_ts).upper()
        policy_state["session"] = current_session
        
        # Hard eligibility check (before feature build)
        eligible, eligibility_reason = self._check_hard_eligibility(candles, policy_state)
        
        # DEL 1: Record hard eligibility gate in telemetry
        if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
            if eligible:
                self.entry_feature_telemetry.record_gate(
                    gate_name="hard_eligibility",
                    executed=True,
                    blocked=False,
                    passed=True,
                    reason=None,
                )
            else:
                self.entry_feature_telemetry.record_gate(
                    gate_name="hard_eligibility",
                    executed=True,
                    blocked=True,
                    passed=False,
                    reason=eligibility_reason,
                )
        
        if not eligible:
            # Hard eligibility failed - STOP immediately
            # NOTE: lookup_attempts er allerede økt (før hard eligibility), så vi har sporet at denne bar nådde evaluate_entry
            # Men lookup ble ikke forsøkt fordi hard eligibility blokkerte før lookup
            # I PREBUILT mode: dette er en "miss" fordi vi ikke kunne forsøke lookup (hard eligibility blokkerte)
            if is_replay and prebuilt_enabled:
                # Telle som miss fordi lookup ble aldri forsøkt (hard eligibility blokkerte først)
                # Dette sikrer at lookup_attempts == lookup_hits + lookup_misses
                if not hasattr(self._runner, "lookup_misses"):
                    self._runner.lookup_misses = 0
                self._runner.lookup_misses += 1
            
            # Increment hard veto counter
            reason_to_veto_key = {
                self.HARD_ELIGIBILITY_WARMUP: "veto_hard_warmup",
                self.HARD_ELIGIBILITY_SESSION_BLOCK: "veto_hard_session",
                self.HARD_ELIGIBILITY_SPREAD_CAP: "veto_hard_spread",
                self.HARD_ELIGIBILITY_KILLSWITCH: "veto_hard_killswitch",
            }
            veto_key = reason_to_veto_key.get(eligibility_reason)
            if veto_key and veto_key in self.veto_hard:
                self.veto_hard[veto_key] += 1
            
            # Rate-limited logging
            if not hasattr(self, "_hard_eligibility_log_counter"):
                self._hard_eligibility_log_counter = 0
            self._hard_eligibility_log_counter += 1
            
            if self._hard_eligibility_log_counter % 100 == 0 or self._hard_eligibility_log_counter <= 10:
                session_info = f" session={policy_state.get('session', 'UNKNOWN')}" if eligibility_reason == self.HARD_ELIGIBILITY_SESSION_BLOCK else ""
                log.info(
                    "[HARD_ELIGIBILITY] blocked reason=%s%s",
                    eligibility_reason,
                    session_info,
                )
            
            return None  # STOP - no feature build, no model call, no candidate
        
        # Hard eligibility passed - increment hard eligible cycles counter
        if "n_eligible_hard" not in self.entry_telemetry:
            self.entry_telemetry["n_eligible_hard"] = 0
        self.entry_telemetry["n_eligible_hard"] += 1
        
        # Soft eligibility check (after minimal cheap computation, before feature build)
        soft_eligible, soft_reason = self._check_soft_eligibility(candles, policy_state)
        
        # DEL 1: Record soft eligibility gate in telemetry
        if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
            if soft_eligible:
                self.entry_feature_telemetry.record_gate(
                    gate_name="soft_eligibility",
                    executed=True,
                    blocked=False,
                    passed=True,
                    reason=None,
                )
                # Record control flow for execution-truth validation
                self.entry_feature_telemetry.record_control_flow("SOFT_ELIGIBILITY_RETURN_TRUE")
            else:
                self.entry_feature_telemetry.record_gate(
                    gate_name="soft_eligibility",
                    executed=True,
                    blocked=True,
                    passed=False,
                    reason=soft_reason,
                )
                # Record control flow for execution-truth validation
                self.entry_feature_telemetry.record_control_flow("SOFT_ELIGIBILITY_RETURN_FALSE")
        
        if not soft_eligible:
            # Soft eligibility failed - STOP before feature build
            if soft_reason == self.SOFT_ELIGIBILITY_VOL_REGIME_EXTREME:
                self.veto_soft["veto_soft_vol_regime_extreme"] += 1
            
            # Rate-limited logging
            if not hasattr(self, "_soft_eligibility_log_counter"):
                self._soft_eligibility_log_counter = 0
            self._soft_eligibility_log_counter += 1
            
            if self._soft_eligibility_log_counter % 100 == 0 or self._soft_eligibility_log_counter <= 10:
                log.info(
                    "[SOFT_ELIGIBILITY] blocked reason=%s",
                    soft_reason,
                )
            
            return None  # STOP - no feature build, no model call, no candidate
        
        # Both hard and soft eligibility passed - increment eligible cycles counter
        self.entry_telemetry["n_eligible_cycles"] += 1
        
        # Record control flow: past soft eligibility, proceeding to feature build
        if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
            self.entry_feature_telemetry.record_control_flow("AFTER_SOFT_ELIGIBILITY_PASSED")
        
        # OPPGAVE 2: Build context features (after soft eligibility, before full feature build)
        # This is cheaper than full V9 feature build and uses ATR proxy from soft eligibility
        import os
        context_features_enabled = os.getenv("ENTRY_CONTEXT_FEATURES_ENABLED", "false").lower() == "true"
        entry_context_features = None
        
        if context_features_enabled:
            try:
                from gx1.execution.entry_context_features import build_entry_context_features
                
                # Get ATR proxy from soft eligibility (already computed)
                atr_proxy = None
                if hasattr(self, "_last_atr_proxy"):
                    atr_proxy = self._last_atr_proxy
                
                # Get spread_bps from hard eligibility (already computed)
                spread_bps = None
                if hasattr(self, "_last_spread_bps"):
                    spread_bps = self._last_spread_bps
                
                # Build context features
                is_replay = getattr(self._runner, "replay_mode", False)
                entry_context_features = build_entry_context_features(
                    candles=candles,
                    policy_state=policy_state,
                    atr_proxy=atr_proxy,
                    spread_bps=spread_bps,
                    is_replay=is_replay,
                )
                
                # Track context build
                if "n_context_built" not in self.entry_telemetry:
                    self.entry_telemetry["n_context_built"] = 0
                self.entry_telemetry["n_context_built"] += 1
                
            except Exception as e:
                is_replay = getattr(self._runner, "replay_mode", False)
                if is_replay:
                    raise RuntimeError(
                        f"CONTEXT_FEATURES_BUILD_FAILED: Failed to build context features: {e}"
                    ) from e
                else:
                    log.warning(
                        "[CONTEXT_FEATURES] Failed to build context features (live mode): %s. "
                        "Continuing without context features.",
                        e
                    )
                    if "n_context_missing_or_invalid" not in self.entry_telemetry:
                        self.entry_telemetry["n_context_missing_or_invalid"] = 0
                    self.entry_telemetry["n_context_missing_or_invalid"] += 1
        
        # SNIPER cycle logging: collect info at start
        policy_sniper_cfg = self.policy.get("entry_v9_policy_sniper", {})
        use_sniper = policy_sniper_cfg.get("enabled", False)
        
        # PREBUILT BYPASS: If prebuilt features are enabled and validated, skip build_live_entry_features
        # NOTE: lookup_attempts er allerede økt tidligere (før hard eligibility)
        # prebuilt_enabled og is_replay er allerede satt tidligere
        
        # Gate dump for diagnosis: capture all conditions
        prebuilt_features_df_exists = hasattr(self._runner, "prebuilt_features_df")
        prebuilt_features_df_is_none = not prebuilt_features_df_exists or self._runner.prebuilt_features_df is None
        prebuilt_features_df_len = len(self._runner.prebuilt_features_df) if not prebuilt_features_df_is_none else 0
        prebuilt_features_df_index_type = type(self._runner.prebuilt_features_df.index).__name__ if not prebuilt_features_df_is_none else "N/A"
        prebuilt_features_df_index_tz = str(getattr(self._runner.prebuilt_features_df.index, 'tz', None)) if not prebuilt_features_df_is_none else "N/A"
        prebuilt_used_flag = getattr(self._runner, "prebuilt_used", False) if hasattr(self._runner, "prebuilt_used") else False
        
        # Store gate dump for export to chunk_footer.json
        self._prebuilt_gate_dump = {
            "is_replay": is_replay,
            "prebuilt_enabled": prebuilt_enabled,
            "prebuilt_used_flag": prebuilt_used_flag,
            "prebuilt_features_df_exists": prebuilt_features_df_exists,
            "prebuilt_features_df_is_none": prebuilt_features_df_is_none,
            "prebuilt_features_df_len": prebuilt_features_df_len,
            "prebuilt_features_df_index_type": prebuilt_features_df_index_type,
            "prebuilt_features_df_index_tz": prebuilt_features_df_index_tz,
        }
        
        # Calculate prebuilt_available with explicit checks
        has_prebuilt_used_attr = hasattr(self._runner, "prebuilt_used")
        prebuilt_available = (
            is_replay
            and prebuilt_enabled
            and prebuilt_features_df_exists
            and not prebuilt_features_df_is_none
            and has_prebuilt_used_attr
            and prebuilt_used_flag
        )
        
        # Update gate dump with final prebuilt_available result
        self._prebuilt_gate_dump["has_prebuilt_used_attr"] = has_prebuilt_used_attr
        self._prebuilt_gate_dump["prebuilt_available"] = prebuilt_available
        
        # STEG 2: Hard-fail hvis PREBUILT mode men prebuilt_available == False
        if is_replay and prebuilt_enabled and not prebuilt_available:
            raise RuntimeError(
                f"[PREBUILT_FAIL] PREBUILT mode enabled but prebuilt_available=False. "
                f"Gate dump: {self._prebuilt_gate_dump}. "
                f"This indicates prebuilt features failed to load or validate. "
                f"Instructions: Check logs for [PREBUILT_FAIL] errors during prebuilt loading."
            )
        
        # PREBUILT telemetry: track gate result
        if prebuilt_available:
            self.eval_calls_prebuilt_gate_true += 1
        else:
            self.eval_calls_prebuilt_gate_false += 1
            # I PREBUILT mode: hvis prebuilt_available == False, skal vi hard-faile (allerede gjort tidligere)
            # Men lookup_attempts er allerede økt, så vi må telle lookup_misses for å balansere
            if is_replay and prebuilt_enabled:
                # Dette skal ikke skje (hard-fail tidligere), men defensive coding
                self._runner.lookup_misses += 1
        
        if prebuilt_available:
            # PREBUILT BYPASS: Use prebuilt features directly (no build_basic_v1 call)
            current_ts = candles.index[-1] if len(candles) > 0 else None
            if current_ts is None:
                raise RuntimeError("[PREBUILT_FAIL] Cannot get current timestamp from candles")
            
            # STEG 2: Hard-fail on lookup miss (PREBUILT mode)
            # lookup_attempts er allerede økt (før gate-sjekk)
            try:
                features_row = self._runner.prebuilt_features_df.loc[current_ts]
                # STEG 3: lookup_hits++ når lookup faktisk returnerer feature-row
                self._runner.lookup_hits += 1
            except KeyError:
                # STEG 2: lookup_misses++ når lookup feiler
                self._runner.lookup_misses += 1
                
                # STEG 5: Debug info for index alignment (store first 3 misses for footer)
                idx = self._runner.prebuilt_features_df.index
                ts_tz = getattr(getattr(current_ts, 'tzinfo', None), 'zone', None) if hasattr(current_ts, 'tzinfo') else None
                idx_tz = str(getattr(idx, 'tz', None))
                ts_repr = repr(current_ts)
                ts_type = type(current_ts).__name__
                
                # Find nearest timestamp and diff
                nearest_ts = None
                diff_sec = None
                try:
                    nearest_idx = idx.get_indexer([current_ts], method='nearest')[0]
                    if nearest_idx >= 0:
                        nearest_ts = idx[nearest_idx]
                        diff_sec = abs((current_ts - nearest_ts).total_seconds())
                except Exception:
                    pass
                
                miss_detail = {
                    "timestamp": ts_repr,
                    "timestamp_type": ts_type,
                    "timestamp_tz": ts_tz,
                    "df_index_tz": idx_tz,
                    "df_index_min": str(idx.min()) if len(idx) > 0 else "N/A",
                    "df_index_max": str(idx.max()) if len(idx) > 0 else "N/A",
                    "nearest_timestamp": str(nearest_ts) if nearest_ts is not None else "N/A",
                    "nearest_diff_sec": diff_sec if diff_sec is not None else "N/A",
                }
                
                # Store first 3 misses for footer export
                if len(self._runner.lookup_miss_details) < 3:
                    self._runner.lookup_miss_details.append(miss_detail)
                
                # Log miss (rate-limited to first 3)
                if len(self._runner.lookup_miss_details) <= 3:
                    log.error(
                        "[PREBUILT_MISS] ts=%s ts_type=%s ts_tz=%s idx_tz=%s idx_min=%s idx_max=%s nearest=%s diff=%.1fs",
                        current_ts, ts_type, ts_tz, idx_tz, idx.min(), idx.max(), nearest_ts, diff_sec if diff_sec is not None else 0.0
                    )
                
                # STEG 2: Hard-fail umiddelbart på miss (fail-fast, ingen silent fallback)
                raise RuntimeError(
                    f"[PREBUILT_LOOKUP_MISS] Timestamp {current_ts} (type={ts_type}, tz={ts_tz}) not found in prebuilt features. "
                    f"Prebuilt range: {idx.min()} to {idx.max()} (tz={idx_tz}). "
                    f"Nearest: {nearest_ts} (diff={diff_sec:.1f}s)" if nearest_ts is not None else ""
                )
            
            # Build EntryFeatureBundle from prebuilt features + raw candles
            from gx1.execution.live_features import EntryFeatureBundle, compute_atr_bps, infer_vol_bucket
            from gx1.tuning.feature_manifest import align_features, load_manifest
            
            # Convert features_row to DataFrame (single row)
            features_df = features_row.to_frame().T
            features_df.index = [current_ts]
            
            # Align features (same as build_live_entry_features)
            manifest = load_manifest()
            aligned = align_features(features_df, manifest=manifest, training_stats=manifest.get("training_stats"))
            aligned_last = aligned.tail(1).copy()
            
            # Compute ATR and vol_bucket from raw candles (needed for EntryFeatureBundle)
            # Import numpy explicitly (np is already imported at module level, but ensure it's available)
            import numpy as np_module
            
            atr_series = compute_atr_bps(candles[["high", "low", "close"]])
            atr_bps = float(atr_series.iloc[-1])
            
            from gx1.execution.live_features import ADR_WINDOW, PIPS_PER_PERCENT
            recent_window = candles.tail(ADR_WINDOW)
            adr = (recent_window["high"].max() - recent_window["low"].min()) if len(recent_window) >= 2 else np_module.nan
            adr_bps = (adr / recent_window["close"].iloc[-1]) * PIPS_PER_PERCENT if not np_module.isnan(adr) else np_module.nan
            atr_adr_ratio = float(atr_bps / adr_bps) if adr_bps and adr_bps > 0 else np_module.nan
            vol_bucket = infer_vol_bucket(atr_adr_ratio)
            
            close_price = float(candles["close"].iloc[-1])
            bid_open = float(candles["bid_open"].iloc[-1]) if "bid_open" in candles.columns else None
            bid_close = float(candles["bid_close"].iloc[-1]) if "bid_close" in candles.columns else None
            ask_open = float(candles["ask_open"].iloc[-1]) if "ask_open" in candles.columns else None
            ask_close = float(candles["ask_close"].iloc[-1]) if "ask_close" in candles.columns else None
            
            # Get raw_row from candles (last row)
            raw_row = candles.iloc[-1]
            
            entry_bundle = EntryFeatureBundle(
                features=aligned_last,
                raw_row=raw_row,
                close_price=close_price,
                atr_bps=atr_bps,
                vol_bucket=vol_bucket,
                bid_open=bid_open,
                bid_close=bid_close,
                ask_open=ask_open,
                ask_close=ask_close,
            )
            
            # Feature time is 0 (prebuilt features used)
            feat_time = 0.0
            
            # STEG 3: prebuilt_bypass_count skal økes nøyaktig når lookup_hits økes (SSoT)
            # Dette sikrer at bypass_count == lookup_hits (assert i footer)
            if not hasattr(self._runner, "prebuilt_bypass_count"):
                self._runner.prebuilt_bypass_count = 0
            self._runner.prebuilt_bypass_count += 1
            
            # Log bypass (rate-limited)
            if self._runner.prebuilt_bypass_count % 1000 == 0 or self._runner.prebuilt_bypass_count <= 10:
                log.info(
                    "[PREBUILT_BYPASS] bars=%d, basic_v1_calls=0 (prebuilt features used directly)",
                    self._runner.prebuilt_bypass_count
                )
            
            # Assert that prebuilt_used is True (fail-fast if not)
            if not self._runner.prebuilt_used:
                raise RuntimeError(
                    "[PREBUILT_FAIL] prebuilt_features_df is available but prebuilt_used=False. "
                    "This indicates validation failed but we're still trying to use prebuilt features."
                )
        else:
            # Normal path: build features using build_live_entry_features
            feat_start = time.perf_counter()
            entry_bundle = build_live_entry_features(candles)
            feat_time = time.perf_counter() - feat_start
        
        # Accumulate feature time (store on runner for performance tracking)
        if hasattr(self._runner, 'perf_feat_time'):
            self._runner.perf_feat_time += feat_time
        else:
            # Initialize if not exists (should not happen, but defensive)
            self._runner.perf_feat_time = feat_time
        current_atr_bps: Optional[float] = None
        current_atr_pct: Optional[float] = None
        try:
            if entry_bundle.atr_bps is not None:
                current_atr_bps = float(entry_bundle.atr_bps)
        except (TypeError, ValueError):
            current_atr_bps = None
        
        # Collect spread info for SNIPER logging
        current_spread_bps: Optional[float] = None
        if hasattr(entry_bundle, "features") and not entry_bundle.features.empty:
            try:
                feat_row = entry_bundle.features.iloc[-1]
                spread_bps_raw = feat_row.get("spread_bps") or feat_row.get("_v1_spread_bps")
                if spread_bps_raw is not None:
                    current_spread_bps = float(spread_bps_raw)
            except (TypeError, ValueError, KeyError):
                pass
        if current_atr_bps is not None and current_atr_bps > 0:
            self.cluster_guard_history.append(current_atr_bps)
            if len(self.cluster_guard_history) >= 25:
                try:
                    self.cluster_guard_atr_median = float(np.median(self.cluster_guard_history))
                except Exception:
                    pass
            current_atr_pct = self._percentile_from_history(self.cluster_guard_history, current_atr_bps)
        
        # Entry telemetry: increment cycle counter (bar-level)
        # NOTE: This is now incremented at the start of evaluate_entry() for hard eligibility
        # self.entry_telemetry["n_cycles"] += 1  # Moved to start of evaluate_entry()
        
        # Big Brain V1 Runtime: observe-only (adds to policy_state for logging/analysis)
        # CRITICAL: Always run inference if Big Brain V1 is enabled (loaded at startup)
        policy_state = {}
        
        # DEPRECATED: FARM_V2B mode check (SNIPER/NY uses entry_v9_policy_sniper, not FARM_V2B)
        # This code path is unreachable for SNIPER/NY (is_farm_v2b = False)
        # Kept for backward compatibility with FARM policies (low risk - guarded by is_farm_v2b)
        # Check if we're in FARM_V2B mode early (needed for FARM regime inference)
        # EntryManager uses __getattr__ to access runner attributes
        is_farm_v2b = False
        # Try to get farm_v2b_mode from runner (via __getattr__)
        try:
            is_farm_v2b = bool(getattr(self, "farm_v2b_mode", False))
        except (AttributeError, TypeError):
            pass
        # Also check policy config
        if not is_farm_v2b:
            is_farm_v2b = bool(self.policy.get("entry_v9_policy_farm_v2b", {}).get("enabled", False))
        # Also check entry_config path
        if not is_farm_v2b:
            entry_config_str = str(self.policy.get("entry_config", "")).upper()
            is_farm_v2b = "FARM_V2B" in entry_config_str
        # Debug log once (use INFO level so it shows up)
        if not hasattr(self, "_farm_v2b_check_logged"):
            log.info("[FARM_REGIME] Checking FARM_V2B mode: is_farm_v2b=%s (farm_v2b_mode=%s, policy_enabled=%s, entry_config=%s)",
                     is_farm_v2b,
                     getattr(self, "farm_v2b_mode", "NOT_SET"),
                     self.policy.get("entry_v9_policy_farm_v2b", {}).get("enabled", False),
                     self.policy.get("entry_config", ""))
            self._farm_v2b_check_logged = True
        
        if self.big_brain_v1 is not None:
            # Big Brain V1 needs raw OHLC+volume+atr data, not feature data
            # Build raw candles DataFrame with required columns for Big Brain V1
            # If warmup buffer is set, infer_from_df will combine warmup + new data
            # So we send the current bar (or recent bars) to infer_from_df
            lookback_bars = self.big_brain_v1.lookback if hasattr(self.big_brain_v1, 'lookback') else 288
            
            # Use candles DataFrame (should contain OHLC + volume)
            # If warmup buffer is set, we can send just the current bar (or recent bars)
            # If no warmup buffer, we need at least lookback_bars rows
            has_warmup = hasattr(self.big_brain_v1, '_warmup_buffer') and self.big_brain_v1._warmup_buffer is not None and not self.big_brain_v1._warmup_buffer.empty
            
            if has_warmup or len(candles) >= lookback_bars:
                try:
                    # If warmup buffer is set, send recent bars (infer_from_df will combine with warmup)
                    # If no warmup, send last lookback_bars rows
                    if has_warmup:
                        # Send ALL candles up to current bar (infer_from_df will combine with warmup and deduplicate)
                        # This ensures we have full history for proper sequence building
                        brain_candles = candles.copy()
                    else:
                        # No warmup: need full lookback
                        brain_candles = candles.tail(lookback_bars).copy()
                    
                    # Ensure we have required columns (open, high, low, close, volume, atr)
                    required_brain_cols = ['open', 'high', 'low', 'close', 'volume', 'atr']
                    missing_cols = [c for c in required_brain_cols if c not in brain_candles.columns]
                    
                    if 'atr' not in brain_candles.columns:
                        # Calculate ATR if missing
                        high_low = brain_candles['high'] - brain_candles['low']
                        high_close = (brain_candles['high'] - brain_candles['close'].shift(1)).abs()
                        low_close = (brain_candles['low'] - brain_candles['close'].shift(1)).abs()
                        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                        brain_candles['atr'] = tr.rolling(window=14, min_periods=1).mean()
                    
                    # Call Big Brain V1 with raw candles
                    # If warmup buffer is set, infer_from_df will combine warmup + new data
                    log.debug("[BIG_BRAIN_V1] Calling infer_from_df with %d bars (warmup=%s)", 
                             len(brain_candles), has_warmup)
                    brain_out = self.big_brain_v1.infer_from_df(brain_candles)
                    
                    # Add Big Brain outputs to policy_state (observe-only)
                    policy_state["brain_trend_regime"] = brain_out.get("trend_regime", "UNKNOWN")
                    policy_state["brain_vol_regime"] = brain_out.get("vol_regime", "UNKNOWN")
                    policy_state["brain_risk_score"] = brain_out.get("brain_risk_score", 0.0)
                    
                    # Optional: add probabilities for analysis/logging
                    policy_state["brain_trend_probs"] = brain_out.get("trend_probs", [])
                    policy_state["brain_vol_probs"] = brain_out.get("vol_probs", [])
                    
                    # Log Big Brain V1 inference (debug level)
                    log.debug(
                        "[BIG_BRAIN_V1] Inference OK: trend=%s vol=%s risk=%.3f",
                        policy_state["brain_trend_regime"],
                        policy_state["brain_vol_regime"],
                        policy_state["brain_risk_score"],
                    )
                except KeyError as e:
                    # Feature mismatch - log error but continue
                    log.error("[BIG_BRAIN_V1] KeyError during inference (feature mismatch?): %s", e, exc_info=True)
                    policy_state["brain_trend_regime"] = "UNKNOWN"
                    policy_state["brain_vol_regime"] = "UNKNOWN"
                    policy_state["brain_risk_score"] = 0.0
                except Exception as e:
                    # Unexpected error during inference - log but continue with UNKNOWN
                    log.error("[BIG_BRAIN_V1] Inference failed (unexpected error): %s", e, exc_info=True)
                    policy_state["brain_trend_regime"] = "UNKNOWN"
                    policy_state["brain_vol_regime"] = "UNKNOWN"
                    policy_state["brain_risk_score"] = 0.0
            else:
                # Not enough bars yet - set UNKNOWN but don't log (expected in early replay)
                policy_state["brain_trend_regime"] = "UNKNOWN"
                policy_state["brain_vol_regime"] = "UNKNOWN"
                policy_state["brain_risk_score"] = 0.0
        else:
            # Big Brain V1 not enabled - check if we should use FARM-only regime
            # is_farm_v2b was already determined above
            if is_farm_v2b:
                # FARM_V2B mode: use FARM-only regime inference instead of Big Brain
                from gx1.regime.farm_regime import infer_farm_regime
                from gx1.execution.live_features import infer_session_tag as _infer_session_tag
                
                # Get current session
                current_ts = candles.index[-1]
                current_session = _infer_session_tag(current_ts).upper()
                
                # Get ATR regime from features
                atr_regime_id = "UNKNOWN"
                atr_regime_source = "unknown"
                if hasattr(entry_bundle, "features") and not entry_bundle.features.empty:
                    feat_row = entry_bundle.features.iloc[-1]
                    if "_v1_atr_regime_id" in entry_bundle.features.columns:
                        atr_regime_id_raw = feat_row.get("_v1_atr_regime_id")
                        if not pd.isna(atr_regime_id_raw):
                            try:
                                regime_val = int(atr_regime_id_raw)
                                # Map: 0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME
                                mapping = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
                                atr_regime_id = mapping.get(regime_val, "UNKNOWN")
                                atr_regime_source = "features_v1_atr_regime_id"
                            except (TypeError, ValueError):
                                atr_regime_id = "UNKNOWN"
                    elif "atr_regime_id" in entry_bundle.features.columns:
                        atr_regime_id_raw = feat_row.get("atr_regime_id")
                        if not pd.isna(atr_regime_id_raw):
                            try:
                                regime_val = int(atr_regime_id_raw)
                                mapping = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
                                atr_regime_id = mapping.get(regime_val, "UNKNOWN")
                                atr_regime_source = "features_atr_regime_id"
                            except (TypeError, ValueError):
                                atr_regime_id = "UNKNOWN"
                
                # REPLAY FIX: If ATR regime is still UNKNOWN and we have candles, compute it from candles
                # This ensures REPLAY mode can determine regime even if features are missing
                if atr_regime_id == "UNKNOWN" and hasattr(self, "replay_mode") and self.replay_mode:
                    if candles is not None and len(candles) >= 14:
                        try:
                            # Compute ATR14 from candles (same logic as basic_v1.py)
                            # Use mid-price if bid/ask available, otherwise use OHLC
                            if "bid_close" in candles.columns and "ask_close" in candles.columns:
                                close = (candles["bid_close"] + candles["ask_close"]) / 2.0
                                high = (candles.get("bid_high", candles.get("high", close)) + 
                                       candles.get("ask_high", candles.get("high", close))) / 2.0
                                low = (candles.get("bid_low", candles.get("low", close)) + 
                                      candles.get("ask_low", candles.get("low", close))) / 2.0
                            else:
                                close = candles.get("close", candles.index)
                                high = candles.get("high", close)
                                low = candles.get("low", close)
                            
                            # Calculate ATR14 (True Range then 14-period SMA)
                            tr = pd.concat([high - low, 
                                          (high - close.shift(1)).abs(), 
                                          (low - close.shift(1)).abs()], axis=1).max(axis=1)
                            atr14 = tr.rolling(window=14, min_periods=1).mean()
                            
                            # Get current ATR14 value
                            current_atr14 = atr14.iloc[-1]
                            
                            # Compute percentile-based regime (same as basic_v1.py qcut logic)
                            # Use rolling percentile over last 100 bars for stability
                            lookback = min(100, len(atr14))
                            if lookback >= 14:
                                atr14_window = atr14.iloc[-lookback:]
                                atr14_pct = (atr14_window.rank(pct=True).iloc[-1])
                                
                                # Map to regime: 0=LOW (0-33%), 1=MEDIUM (33-67%), 2=HIGH (67-100%)
                                if atr14_pct < 0.33:
                                    regime_val = 0  # LOW
                                elif atr14_pct < 0.67:
                                    regime_val = 1  # MEDIUM
                                else:
                                    regime_val = 2  # HIGH
                                
                                mapping = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
                                atr_regime_id = mapping.get(regime_val, "UNKNOWN")
                                atr_regime_source = "candles_fallback"
                                
                                # Log once per run with source
                                if not hasattr(self, "_atr_regime_fallback_logged"):
                                    log.info(
                                        "[REPLAY_FIX] Computed ATR regime from candles (FALLBACK): atr14_pct=%.3f -> regime=%s "
                                        "(source=candles_fallback, features_missing=_v1_atr_regime_id)",
                                        atr14_pct, atr_regime_id
                                    )
                                    self._atr_regime_fallback_logged = True
                        except Exception as e:
                            log.warning("[REPLAY_FIX] Failed to compute ATR regime from candles: %s", e)
                
                # Infer FARM regime
                farm_regime = infer_farm_regime(current_session, atr_regime_id)
                
                # FARM_V2B diagnostic: track FARM regime (after inference, but before Stage-0)
                # Note: This is counted even if Stage-0 blocks later, to see regime distribution
                if is_farm_v2b:
                    # Store for later use in raw_candidates tracking
                    policy_state["_atr_regime_id"] = atr_regime_id
                    policy_state["_farm_regime"] = farm_regime
                
                # Log once per run with source
                if not hasattr(self, "_farm_regime_logged"):
                    log.info(
                        "[BOOT] FARM_V2B mode detected: using FARM-only regime (session+ATR) instead of Big Brain. "
                        "First bar: session=%s atr_regime=%s (source=%s) -> farm_regime=%s",
                        current_session, atr_regime_id, atr_regime_source, farm_regime
                    )
                    self._farm_regime_logged = True
                
                # Set policy_state with FARM regime info
                # For FARM_ASIA_LOW/MEDIUM, we'll allow Stage-0 to pass
                # For FARM_OUT_OF_SCOPE, Stage-0 can still block (or let brutal guard handle it)
                if farm_regime in ("FARM_ASIA_LOW", "FARM_ASIA_MEDIUM"):
                    # Map to Big Brain-like regime names for Stage-0 compatibility
                    # Stage-0 expects trend/vol, so we'll set reasonable defaults
                    policy_state["brain_trend_regime"] = "TREND_UP"  # FARM assumes uptrend bias
                    policy_state["brain_vol_regime"] = atr_regime_id if atr_regime_id != "UNKNOWN" else "LOW"
                    policy_state["brain_risk_score"] = 0.5  # Moderate risk
                    policy_state["farm_regime"] = farm_regime  # Store FARM regime explicitly
                else:
                    # Out of scope - let Stage-0 or brutal guard handle it
                    policy_state["brain_trend_regime"] = "UNKNOWN"
                    policy_state["brain_vol_regime"] = "UNKNOWN"
                    policy_state["brain_risk_score"] = 0.0
                    policy_state["farm_regime"] = farm_regime
                
                # REPLAY INVARIANT CHECK: After warmup, trend/vol should not be UNKNOWN
                # This ensures REPLAY mode can determine regime from historical data
                if hasattr(self, "replay_mode") and self.replay_mode:
                    # Check if we're past warmup (288 bars = 1 day)
                    warmup_bars = getattr(self, "warmup_bars", 288)
                    if candles is not None and len(candles) > warmup_bars:
                        # Check if trend/vol are still UNKNOWN after warmup
                        trend = policy_state.get("brain_trend_regime", "UNKNOWN")
                        vol = policy_state.get("brain_vol_regime", "UNKNOWN")
                        
                        # Rate-limited debug logging (every 1000 bars)
                        if not hasattr(self, "_replay_unknown_debug_counter"):
                            self._replay_unknown_debug_counter = 0
                        self._replay_unknown_debug_counter += 1
                        
                        if trend == "UNKNOWN" or vol == "UNKNOWN":
                            # Log ERROR with diagnostic info (always)
                            if not hasattr(self, "_replay_invariant_checked"):
                                log.error(
                                    "[REPLAY_INVARIANT] After warmup (%d bars), trend/vol still UNKNOWN: trend=%s vol=%s. "
                                    "Diagnostics: atr_regime_id=%s, farm_regime=%s, session=%s, "
                                    "features_available=%s, candles_len=%d",
                                    len(candles), trend, vol, atr_regime_id, farm_regime, current_session,
                                    hasattr(entry_bundle, "features") and not entry_bundle.features.empty,
                                    len(candles) if candles is not None else 0
                                )
                                self._replay_invariant_checked = True
                            
                            # Rate-limited debug report (every 1000 bars)
                            if self._replay_unknown_debug_counter % 1000 == 0:
                                # Compute ATR for debug
                                try:
                                    if "bid_close" in candles.columns and "ask_close" in candles.columns:
                                        close = (candles["bid_close"] + candles["ask_close"]) / 2.0
                                    else:
                                        close = candles.get("close", candles.index)
                                    tr = pd.concat([
                                        candles.get("high", close) - candles.get("low", close),
                                        (candles.get("high", close) - close.shift(1)).abs(),
                                        (candles.get("low", close) - close.shift(1)).abs()
                                    ], axis=1).max(axis=1)
                                    atr14 = tr.rolling(window=14, min_periods=1).mean()
                                    current_atr = atr14.iloc[-1] if len(atr14) > 0 else None
                                except:
                                    current_atr = None
                                
                                log.warning(
                                    "[REPLAY_UNKNOWN_DEBUG] Bar %d: ts=%s, cached_bars=%d, computed_atr=%s, "
                                    "vol_bucket=%s, trend_tag=%s, session=%s",
                                    self._replay_unknown_debug_counter,
                                    current_ts.isoformat() if current_ts else "N/A",
                                    len(candles),
                                    f"{current_atr:.2f}" if current_atr is not None else "N/A",
                                    vol,
                                    trend,
                                    current_session
                                )
                        else:
                            # Log success once
                            if not hasattr(self, "_replay_invariant_checked"):
                                log.info(
                                    "[REPLAY_INVARIANT] After warmup (%d bars), trend/vol known: trend=%s vol=%s",
                                    len(candles), trend, vol
                                )
                                self._replay_invariant_checked = True
            else:
                # Not FARM_V2B mode - compute tags from candles for SNIPER in live mode
                # This ensures SNIPER and other non-FARM policies get trend/vol tags BEFORE Stage-0
                # Option C: Use replay tags computation in live mode for SNIPER
                if use_sniper and candles is not None:
                    # SNIPER in live mode: compute trend/vol from candles (same as replay)
                    from gx1.execution.replay_features import ensure_replay_tags
                    current_ts = candles.index[-1] if len(candles) > 0 else None
                    # Create a dummy row for ensure_replay_tags (it will compute from candles anyway)
                    dummy_row = pd.Series({"ts": current_ts} if current_ts else {})
                    dummy_row, policy_state = ensure_replay_tags(
                        dummy_row,
                        candles,
                        policy_state,
                        current_ts=current_ts,
                    )
                elif hasattr(self, "replay_mode") and self.replay_mode and candles is not None:
                    # Replay mode (non-SNIPER): use replay tags
                    from gx1.execution.replay_features import ensure_replay_tags
                    current_ts = candles.index[-1] if len(candles) > 0 else None
                    dummy_row = pd.Series({"ts": current_ts} if current_ts else {})
                    dummy_row, policy_state = ensure_replay_tags(
                        dummy_row,
                        candles,
                        policy_state,
                        current_ts=current_ts,
                    )
                else:
                    # Not SNIPER, not replay mode, or no candles - set UNKNOWN as before
                    policy_state["brain_trend_regime"] = "UNKNOWN"
                    policy_state["brain_vol_regime"] = "UNKNOWN"
                    policy_state["brain_risk_score"] = 0.0
        
        # ============================================================
        # Stage-0 Opportunitetsfilter (should_consider_entry)
        # ============================================================
        # Check if we should even run XGB/TCN for this bar, based on
        # Big Brain V1 regime + session. This happens BEFORE any model inference.
        trend = policy_state.get("brain_trend_regime", "UNKNOWN")
        vol = policy_state.get("brain_vol_regime", "UNKNOWN")
        risk_score = policy_state.get("brain_risk_score", 0.0)
        
        # Get current session (same as used for XGB model routing)
        # Note: If FARM_V2B mode, current_session was already set above
        if "session" not in policy_state:
            from gx1.execution.live_features import infer_session_tag as _infer_session_tag
            current_ts_for_session = candles.index[-1]
            current_session = _infer_session_tag(current_ts_for_session).upper()
            policy_state["session"] = current_session
        else:
            current_session = policy_state["session"]
        
        # SNIPER logging: check if session is in-scope
        sniper_in_scope = False
        if use_sniper:
            allowed_sessions = policy_sniper_cfg.get("allowed_sessions", ["EU", "OVERLAP", "US"])
            sniper_in_scope = current_session in allowed_sessions
        
        # SNIPER logging: check warmup status
        sniper_warmup_ready = True
        sniper_degraded = False
        if use_sniper:
            warmup_bars = getattr(self, "warmup_bars", 288)
            cached_bars = len(candles) if candles is not None else 0
            sniper_warmup_ready = cached_bars >= warmup_bars
            # Check if degraded (less than ideal but still usable)
            sniper_degraded = cached_bars < warmup_bars and cached_bars >= 100
        
        # For FARM_V2B mode with FARM regime, don't block solely on UNKNOWN
        # The brutal guard will handle further filtering
        is_farm_v2b_with_regime = (
            policy_state.get("farm_regime") in ("FARM_ASIA_LOW", "FARM_ASIA_MEDIUM")
        )
        
        # Stage-0 filter: skip entry consideration if regime/session is not promising
        # Check if Stage-0 is enabled (config-styrt, default=True for backward compatibility)
        stage0_enabled = getattr(self, "stage0_enabled", True)
        
        # ============================================================
        # DEBUG FORCE ENTRY (CANARY only - kun for plumbing testing)
        # ============================================================
        # Check if force entry should be triggered (extremely safe, CANARY only)
        debug_force_cfg = self.policy.get("debug_force", {})
        force_enabled = (
            debug_force_cfg.get("enabled", False) and
            self.policy.get("meta", {}).get("role") == "CANARY" and
            hasattr(self, "oanda_env") and self.oanda_env == "practice"
        )
        
        if force_enabled:
            # Initialize force entry tracking
            if not hasattr(self, "_force_entry_start_time"):
                self._force_entry_start_time = time.time()
                self._force_entry_trade_count = 0
                log.info("[FORCE_ENTRY] Force entry enabled (CANARY only, practice mode)")
            
            # Check if we've exceeded max trades
            max_trades = debug_force_cfg.get("max_trades", 1)
            if self._force_entry_trade_count >= max_trades:
                # Already hit max trades - disable force entry
                force_enabled = False
            else:
                # Check if timeout has been reached
                timeout_minutes = debug_force_cfg.get("force_entry_if_no_trade_after_minutes", 30)
                elapsed_minutes = (time.time() - self._force_entry_start_time) / 60.0
                
                # Check if we should force entry
                should_force = (
                    elapsed_minutes >= timeout_minutes and
                    self._force_entry_trade_count == 0  # Only force if no trades yet
                )
                
                if should_force:
                    # Check allowed session (null/None = any session)
                    allowed_session = debug_force_cfg.get("allowed_session")
                    session_allowed = (
                        allowed_session is None or
                        allowed_session == "" or
                        current_session.upper() == allowed_session.upper()
                    )
                    
                    if not session_allowed:
                        # Not in allowed session
                        log.debug(
                            "[FORCE_ENTRY] Not in allowed session (current=%s, allowed=%s)",
                            current_session, allowed_session
                        )
                        should_force = False
                    
                    if should_force:
                        # Log warmup progress
                        warmup_bars_seen = len(candles) if candles is not None else 0
                        warmup_progress_pct = min(100.0, (warmup_bars_seen / warmup_bars) * 100.0) if warmup_bars > 0 else 0.0
                        log.info(
                            "[FORCE_ENTRY] Warmup progress: %d/%d bars (%.1f%%)",
                            warmup_bars_seen, warmup_bars, warmup_progress_pct
                        )
                        
                        # Log session/regime
                        log.info(
                            "[FORCE_ENTRY] Session=%s regime=%s trend=%s vol=%s",
                            current_session, policy_state.get("farm_regime", "UNKNOWN"),
                            trend, vol
                        )
                        
                        # Check spread (use existing spread guard, don't bypass)
                        min_spread_pct = debug_force_cfg.get("min_spread_pct")
                        spread_pct = None
                        if hasattr(entry_bundle, "features") and not entry_bundle.features.empty:
                            feat_row = entry_bundle.features.iloc[-1]
                            spread_pct = feat_row.get("spread_pct") or feat_row.get("_v1_spread_pct")
                        
                        if spread_pct is not None:
                            log.info(
                                "[FORCE_ENTRY] Spread guard: %.2f%% %s",
                                spread_pct,
                                "OK" if (min_spread_pct is None or spread_pct <= min_spread_pct) else f"BLOCKED (> {min_spread_pct}%)"
                            )
                        
                        if min_spread_pct is not None and spread_pct is not None and spread_pct > min_spread_pct:
                            log.info(
                                "[FORCE_ENTRY] Spread too high (%.2f > %.2f), not forcing entry",
                                spread_pct, min_spread_pct
                            )
                            should_force = False
                        
                        # Check warmup requirement for force entry
                        min_start_bars = 100  # Minimum bars for CANARY degraded warmup
                        require_full_warmup = debug_force_cfg.get("require_full_warmup", False)
                        warmup_bars_required = self.policy.get("warmup_bars", 288)
                        
                        # Get cached bars from runner
                        cached_bars = getattr(self._runner, '_cached_bars_at_startup', None)
                        if cached_bars is None:
                            # Fallback: try to get from backfill_cache or candles
                            if hasattr(self._runner, 'backfill_cache') and self._runner.backfill_cache is not None:
                                cached_bars = len(self._runner.backfill_cache)
                            elif candles is not None:
                                cached_bars = len(candles)
                            else:
                                cached_bars = 0
                        
                        # Check warmup requirement
                        if require_full_warmup:
                            warmup_minimum = warmup_bars_required
                        else:
                            warmup_minimum = min_start_bars
                        
                        if cached_bars < warmup_minimum:
                            log.info(
                                "[FORCE_ENTRY] Warmup requirement not met: cached_bars=%d < min=%d (require_full_warmup=%s), not forcing entry",
                                cached_bars, warmup_minimum, require_full_warmup
                            )
                            should_force = False
                        
                        # Log force countdown
                        minutes_remaining = max(0, timeout_minutes - elapsed_minutes)
                        log.info(
                            "[FORCE_ENTRY] Countdown: %.1f minutes remaining (timeout=%d min, elapsed=%.1f min)",
                            minutes_remaining, timeout_minutes, elapsed_minutes
                        )
                        
                        if should_force:
                            # FORCE ENTRY TRIGGERED
                            log.warning(
                                "[FORCE_ENTRY] TRIGGERED: reason=timeout_after_%.1f_minutes session=%s "
                                "elapsed=%.1f_minutes trades=0",
                                timeout_minutes, current_session, elapsed_minutes
                            )
                            
                            # Set force entry flag in policy_state
                            policy_state["force_entry"] = True
                            policy_state["force_entry_reason"] = f"timeout_after_{timeout_minutes}_minutes"
                            
                            # Bypass Stage-0 if configured (but still respect other guards)
                            bypass_trend_vol = debug_force_cfg.get("bypass_trend_vol_unknown", False)
                            if bypass_trend_vol and (trend == "UNKNOWN" or vol == "UNKNOWN"):
                                log.warning(
                                    "[FORCE_ENTRY] Bypassing trend/vol UNKNOWN check (bypass_trend_vol_unknown=true)"
                                )
                                # Set reasonable defaults for trend/vol
                                if trend == "UNKNOWN":
                                    policy_state["brain_trend_regime"] = "TREND_UP"
                                    trend = "TREND_UP"
                                if vol == "UNKNOWN":
                                    policy_state["brain_vol_regime"] = "MEDIUM"
                                    vol = "MEDIUM"
                            
                            # Skip Stage-0 blocking for force entry
                            log.info(
                                "[FORCE_ENTRY] Bypassing Stage-0 filter for force entry: "
                                "trend=%s vol=%s session=%s",
                                trend, vol, current_session
                            )
                            # Continue to model inference (don't return None)
        
        # For FARM_V2B with valid FARM regime, don't block on Stage-0
        # The brutal guard and FARM_V2B policy will handle filtering
        if is_farm_v2b_with_regime:
            # FARM_V2B with valid regime: skip Stage-0 blocking, let brutal guard handle it
            log.debug(
                "[STAGE_0] FARM_V2B mode with valid regime (%s): skipping Stage-0 filter, "
                "brutal guard will handle filtering",
                policy_state.get("farm_regime")
            )
            # Record control flow: Stage-0 bypassed (FARM_V2B mode)
            if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
                self.entry_feature_telemetry.record_control_flow("BEFORE_STAGE0_CHECK")
            # FARM_V2B diagnostic: Stage-0 passed (skipped for valid regime)
            if is_farm_v2b:
                if "n_after_stage0" not in self.farm_diag:
                    self.farm_diag["n_after_stage0"] = 0
                self.farm_diag["n_after_stage0"] += 1
        elif not policy_state.get("force_entry", False) and stage0_enabled:
            # Record control flow: entering Stage-0 check
            if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
                self.entry_feature_telemetry.record_control_flow("BEFORE_STAGE0_CHECK")
            
            # Track total considered
            self.stage0_total_considered += 1
            
            # Call should_consider_entry (now returns bool, reason stored in runner)
            should_consider = self.should_consider_entry(trend=trend, vol=vol, session=current_session, risk_score=risk_score)
            
            if not should_consider:
                # Get reason from runner's tracking (if available)
                reason = getattr(self._runner, "_last_stage0_reason", "stage0_unknown")
                missing_field = getattr(self._runner, "_last_stage0_missing_field", None)
                self.stage0_reasons[reason] += 1
                
                # TELEMETRY CONTRACT: Map Stage-0 veto reasons to veto_pre_*
                if reason == "stage0_session_block":
                    self.veto_pre["veto_pre_session"] += 1
                elif reason == "stage0_vol_block":
                    self.veto_pre["veto_pre_atr"] += 1
                elif reason == "stage0_trend_vol_block":
                    self.veto_pre["veto_pre_regime"] += 1
                elif reason == "stage0_unknown_field":
                    # Could be regime-related, but track as separate category for now
                    pass  # Handled separately if needed
                # Legacy (backward compatibility)
                # DEPRECATED: veto_counters removed (SNIPER/NY uses veto_pre)
                # Legacy veto_counters updates removed (veto_pre counters are updated above)
                
                # SNIPER logging: stage0 blocked with missing field info
                if use_sniper and reason == "stage0_unknown_field":
                    # Collect available keys from policy_state for context
                    available_keys_sample = []
                    if policy_state:
                        # Sample up to 10 keys from policy_state
                        available_keys_sample = list(policy_state.keys())[:10]
                    
                    # Rate-limited: log when missing_field changes or max once per 10 minutes
                    if not hasattr(self, "_last_stage0_missing_field_logged"):
                        self._last_stage0_missing_field_logged = None
                        self._last_stage0_log_time = None
                    
                    # time is already imported at module level
                    should_log_stage0 = False
                    current_time = time.time()
                    
                    if missing_field != self._last_stage0_missing_field_logged:
                        should_log_stage0 = True
                    elif self._last_stage0_log_time is None or (current_time - self._last_stage0_log_time) >= 600:
                        should_log_stage0 = True
                    
                    if should_log_stage0:
                        available_keys_str = ",".join(available_keys_sample) if available_keys_sample else "N/A"
                        log.info(
                            "[SNIPER_STAGE0] ts=%s session=%s reason=%s missing_field=%s available_keys=%s",
                            current_ts.isoformat(),
                            current_session,
                            reason,
                            missing_field or "unknown",
                            available_keys_str,
                        )
                        self._last_stage0_missing_field_logged = missing_field
                        self._last_stage0_log_time = current_time
                
                # Rate-limited logging (once per 500 bars)
                if self.stage0_total_considered % 500 == 0:
                    log.info(
                        "[STAGE_0] Skip entry consideration (rate-limited): trend=%s vol=%s session=%s risk=%.3f reason=%s",
                        trend,
                        vol,
                        current_session,
                        risk_score,
                        reason,
                    )
                else:
                    log.debug(
                        "[STAGE_0] Skip entry consideration: trend=%s vol=%s session=%s risk=%.3f reason=%s",
                        trend,
                        vol,
                        current_session,
                        risk_score,
                        reason,
                    )
                
                # Store stage-0 skip in policy_state for logging (if needed)
                policy_state["stage_0_skip"] = {
                    "trend": trend,
                    "vol": vol,
                    "session": current_session,
                    "risk_score": float(risk_score),
                    "reason": reason,
                }
                self._last_policy_state = policy_state
                # FARM_V2B diagnostic: Stage-0 blocked (only if not FARM_V2B with valid regime)
                if is_farm_v2b and not is_farm_v2b_with_regime:
                    # Stage-0 blocked for FARM_V2B (but not counted in n_after_stage0)
                    pass
                
                # SNIPER logging: stage0 blocked
                if use_sniper:
                    # Reason already has stage0_ prefix from should_consider_entry
                    reason_str = reason if reason.startswith("stage0_") else f"stage0_{reason}"
                    self._log_sniper_cycle(
                        ts=current_ts,
                        session=current_session,
                        in_scope=sniper_in_scope,
                        warmup_ready=sniper_warmup_ready,
                        degraded=sniper_degraded,
                        spread_bps=current_spread_bps,
                        atr_bps=current_atr_bps,
                        eval_ran=False,
                        reason=reason_str,
                        decision="NO-TRADE",
                    )
                
                return None  # Skip XGB/TCN inference entirely
            else:
                # TELEMETRY CONTRACT: Stage-0 passed (prediction allowed)
                self.entry_telemetry["n_precheck_pass"] += 1
                # Legacy (backward compatibility)
                if is_farm_v2b:
                    self.farm_diag["n_after_stage0"] += 1
        else:
            # Stage-0 bypassed (force_entry or stage0_enabled=False)
            # Record control flow: Stage-0 bypassed (force_entry or disabled)
            if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
                self.entry_feature_telemetry.record_control_flow("BEFORE_STAGE0_CHECK")
            # TELEMETRY CONTRACT: Stage-0 passed (prediction allowed)
            self.entry_telemetry["n_precheck_pass"] += 1
        
        # Get base cutoff from entry_params or entry_gating (for adaptive thresholding)
        entry_gating = self.policy.get("entry_gating", None)
        if entry_gating and "coverage" in entry_gating:
            base_cutoff = float(entry_gating.get("coverage", {}).get("target", 0.20))
        else:
            # Fallback to entry_params or telemetry target_coverage
            base_cutoff = float(
                self.entry_params.get("gating", {}).get("coverage", {}).get("target", 0.20)
            )
        
        policy_state["coverage_cutoff_base"] = base_cutoff
        
        # Store policy_state temporarily (will be updated after hybrid blending)
        self._last_policy_state = policy_state
        
        # Get current regime from Big Brain V1 (already in policy_state)
        brain_trend_regime = policy_state.get("brain_trend_regime", "UNKNOWN")
        brain_vol_regime = policy_state.get("brain_vol_regime", "UNKNOWN")
        
        # Initialize entry_model_active (set based on which model is actually used)
        if hasattr(self._runner, "entry_v10_enabled") and self._runner.entry_v10_enabled:
            policy_state["entry_model_active"] = "ENTRY_V10"
        else:
            policy_state["entry_model_active"] = "ENTRY_V9"
        
        # ============================================================
        # ENTRY_V9 ONLY - No fallback to V6/V8
        # ============================================================
        # V9 gets ALL data (all regimes) - only policy filters coverage
        # Check if V9 is required for entry (default: True for backward compatibility)
        require_v9_for_entry = self.policy.get("require_v9_for_entry", True)
        
        if require_v9_for_entry:
            # V9 is required (legacy behavior for P4.1, etc.)
            if not self.entry_v9_enabled or self.entry_v9_model is None:
                # DEL 1: Use generic [ENTRY] prefix in replay-mode (no V9 references)
                log_prefix = "[ENTRY_V10_CTX]" if (hasattr(self._runner, "replay_mode") and self._runner.replay_mode) else "[ENTRY_V9]"
                log.error(f"{log_prefix} Entry model is REQUIRED but not loaded. No entry possible.")
                
                # TELEMETRY CONTRACT: Model missing (Stage-0, before prediction)
                self.veto_pre["veto_pre_model_missing"] += 1
                # DEPRECATED: veto_counters removed (SNIPER/NY uses veto_pre)
                
                # SNIPER logging: entry_v9 disabled
                if use_sniper:
                    self._log_sniper_cycle(
                        ts=current_ts,
                        session=current_session,
                        in_scope=sniper_in_scope,
                        warmup_ready=sniper_warmup_ready,
                        degraded=sniper_degraded,
                        spread_bps=current_spread_bps,
                        atr_bps=current_atr_bps,
                        eval_ran=False,
                        reason="entry_v9_disabled",
                        decision="NO-TRADE",
                    )
                
                return None
        else:
            # V9 is NOT required (V10-only mode for V10.1 configs)
            if not self.entry_v9_enabled and not (hasattr(self._runner, "entry_v10_enabled") and self._runner.entry_v10_enabled):
                # Neither V9 nor V10 enabled - this is an error
                log.error("[ENTRY] require_v9_for_entry=False but neither V9 nor V10 enabled. No entry possible.")
                
                # TELEMETRY CONTRACT: Model missing (Stage-0, before prediction)
                self.veto_pre["veto_pre_model_missing"] += 1
                # DEPRECATED: veto_counters removed (SNIPER/NY uses veto_pre)
                
                if use_sniper:
                    self._log_sniper_cycle(
                        ts=current_ts,
                        session=current_session,
                        in_scope=sniper_in_scope,
                        warmup_ready=sniper_warmup_ready,
                        degraded=sniper_degraded,
                        spread_bps=current_spread_bps,
                        atr_bps=current_atr_bps,
                        eval_ran=False,
                        reason="no_entry_model",
                        decision="NO-TRADE",
                    )
                
                return None
            
            # Log V10-only mode if V9 is disabled and V10 is enabled
            if not self.entry_v9_enabled and hasattr(self._runner, "entry_v10_enabled") and self._runner.entry_v10_enabled:
                log.debug("[ENTRY] Running in V10-only mode (V9 disabled, V10 enabled, require_v9_for_entry=False)")
            elif self.entry_v9_enabled and self.entry_v9_model is None:
                # DEL 1: Use generic [ENTRY] prefix in replay-mode (no V9 references)
                log_prefix = "[ENTRY_V10_CTX]" if (hasattr(self._runner, "replay_mode") and self._runner.replay_mode) else "[ENTRY_V9]"
                log.warning(f"{log_prefix} Entry model enabled in config but not loaded. Continuing because require_v9_for_entry=False.")
        
        # Route to V10 (if enabled) or V9
        # V10 takes priority if both are enabled
        entry_pred = None
        
        # DEL 1: Record V10 enable state in telemetry (before routing)
        if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
            v10_enabled = hasattr(self._runner, "entry_v10_enabled") and self._runner.entry_v10_enabled
            if v10_enabled:
                reason = "ENABLED"
            else:
                reason = "POLICY_DISABLED" if hasattr(self._runner, "entry_v10_enabled") else "NOT_CONFIGURED"
            self.entry_feature_telemetry.record_v10_enable_state(enabled=v10_enabled, reason=reason)
        
        # Model inference timing
        model_start = time.perf_counter()
        if hasattr(self._runner, "entry_v10_enabled") and self._runner.entry_v10_enabled:
            # Session gating: V10 hybrid requires XGB which only exists for EU/US/OVERLAP
            # Skip ASIA (and other unsupported sessions) BEFORE counting as V10 call
            # NOTE: Session filtering is already handled by v10_session_supported gate earlier
            # This check should not trigger if gate is working correctly
            v10_supported_sessions = {"EU", "US", "OVERLAP"}
            if current_session not in v10_supported_sessions:
                # ASIA session: skip V10 (no XGB model)
                # This is expected and should not count as a V10 call or failure
                log.debug(
                    "[ENTRY_V10] Session %s not supported (no XGB), skipping entry evaluation",
                    current_session
                )
                # DEL 3: Do NOT record routing here - session gate already handled this
                # Recording here causes duplicate routing telemetry (wiring bug)
                return None
            
            # DEL 1: Record routing for V10 hybrid
            if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
                self.entry_feature_telemetry.record_entry_routing(
                    selected_model="v10_hybrid",
                    reason="V10_ENABLED_AND_SESSION_SUPPORTED"
                )
            
            self.n_v10_calls += 1
            if self._v10_log_count < 3:
                log.info(
                    "[ENTRY_V10] Call #%d: Evaluating in regime trend=%s vol=%s session=%s",
                    self.n_v10_calls, brain_trend_regime, brain_vol_regime, current_session
                )
            # OPPGAVE 2: Pass context features to V10 prediction (if enabled)
            entry_pred = self._runner._predict_entry_v10_hybrid(
                entry_bundle, 
                candles, 
                policy_state,
                entry_context_features=entry_context_features,  # New parameter
            )
            
            # DEL 1: Hook RawSignalCollector - collect pre-policy raw signal (right after model call)
            # Hook point: After _predict_entry_v10_hybrid returns EntryPrediction
            if entry_pred is not None and hasattr(self._runner, "replay_eval_collectors") and self._runner.replay_eval_collectors:
                raw_collector = self._runner.replay_eval_collectors.get("raw_signals")
                if raw_collector:
                    # Extract gate value from telemetry (stored during model call)
                    gate_value = 0.0  # Default if not available
                    gate_values = self.entry_telemetry.get("gate_values", [])
                    if gate_values:
                        gate_value = gate_values[-1]
                    
                    # Extract uncertainty from XGB calibration (if available)
                    uncertainty = None
                    gate_vs_uncertainty = self.entry_telemetry.get("gate_vs_uncertainty", [])
                    if gate_vs_uncertainty:
                        uncertainty = gate_vs_uncertainty[-1].get("uncertainty_score")
                    
                    # Get raw logit (approximate from prob_long via inverse sigmoid)
                    import numpy as np
                    raw_logit = np.log(entry_pred.prob_long / (1.0 - entry_pred.prob_long + 1e-8))
                    
                    # Get timestamp from candles
                    current_ts = candles.index[-1] if len(candles) > 0 else pd.Timestamp.now(tz="UTC")
                    
                    raw_collector.collect(
                        timestamp=current_ts,
                        session=current_session,
                        raw_logit=raw_logit,
                        p_calibrated=entry_pred.prob_long,  # p_cal from XGB calibration
                        gate_value=gate_value,
                        uncertainty=uncertainty,
                        margin=entry_pred.margin,
                    )
        elif hasattr(self._runner, "entry_v9_enabled") and self._runner.entry_v9_enabled:
            # DEL 1: Use generic [ENTRY] prefix in replay-mode (no V9 references)
            log_prefix = "[ENTRY_V10_CTX]" if (hasattr(self._runner, "replay_mode") and self._runner.replay_mode) else "[ENTRY_V9]"
            log.debug(
                f"{log_prefix} Evaluating in regime trend=%s vol=%s session=%s",
                brain_trend_regime, brain_vol_regime, current_session
            )
            entry_pred = self._runner._predict_entry_v9(entry_bundle, candles, policy_state)
        else:
            log.error("[ENTRY] Neither V10 nor V9 enabled - no entry prediction possible")
            return None
        model_time = time.perf_counter() - model_start
        
        # Accumulate model time
        if hasattr(self._runner, 'perf_model_time'):
            self._runner.perf_model_time += model_time
        else:
            # Initialize if not exists (should not happen, but defensive)
            self._runner.perf_model_time = model_time
        
        v9_pred = entry_pred  # Keep variable name for compatibility
        
        # STEG 1: Entry-score logging (READ-ONLY) - log score for ALL eligible bars reaching entry-stage
        # This happens BEFORE policy evaluation, so we capture all scores regardless of threshold
        if entry_pred is not None and hasattr(self._runner, "replay_mode") and self._runner.replay_mode:
            # Initialize entry_score accumulator if not exists
            if not hasattr(self._runner, "entry_score_samples"):
                self._runner.entry_score_samples = []
                self._runner.entry_score_by_session = {"ASIA": [], "EU": [], "OVERLAP": [], "US": []}
                self._runner.entry_score_by_regime = {}  # Will be populated dynamically
            
            # Get session and regime for breakdown
            current_session = policy_state.get("session", "UNKNOWN")
            trend_regime = policy_state.get("brain_trend_regime") or policy_state.get("trend_regime")
            vol_regime = policy_state.get("brain_vol_regime") or policy_state.get("vol_regime")
            regime_key = f"{trend_regime}_{vol_regime}" if trend_regime and vol_regime else "UNKNOWN"
            
            # Extract scores
            prob_long = float(entry_pred.prob_long) if hasattr(entry_pred, "prob_long") else None
            prob_short = float(entry_pred.prob_short) if hasattr(entry_pred, "prob_short") else None
            margin = float(entry_pred.margin) if hasattr(entry_pred, "margin") else None
            
            # Use prob_long as primary entry score (main threshold metric)
            if prob_long is not None:
                score_entry = {
                    "prob_long": prob_long,
                    "prob_short": prob_short,
                    "margin": margin,
                    "session": current_session,
                    "trend_regime": trend_regime,
                    "vol_regime": vol_regime,
                    "regime_key": regime_key,
                }
                
                # Add to global accumulator
                self._runner.entry_score_samples.append(score_entry)
                
                # Add to session breakdown
                if current_session in self._runner.entry_score_by_session:
                    self._runner.entry_score_by_session[current_session].append(score_entry)
                
                # Add to regime breakdown
                if regime_key not in self._runner.entry_score_by_regime:
                    self._runner.entry_score_by_regime[regime_key] = []
                self._runner.entry_score_by_regime[regime_key].append(score_entry)
        
        # Store V10.1 p_long in policy_state for size overlay (if V10 is enabled)
        if hasattr(self._runner, "entry_v10_enabled") and self._runner.entry_v10_enabled and v9_pred:
            # Store p_long_v10_1 for V10.1 size overlay
            policy_state["p_long_v10_1"] = v9_pred.prob_long
        
        # Track V10 prediction results (for diagnostics)
        if hasattr(self._runner, "entry_v10_enabled") and self._runner.entry_v10_enabled:
            if v9_pred is None:
                self.n_v10_pred_none_or_nan += 1
                if self._v10_log_count < 3:
                    log.warning("[ENTRY_V10] Call #%d: Prediction returned None", self.n_v10_calls)
                    self._v10_log_count += 1
            elif not np.isfinite(v9_pred.prob_long) or not np.isfinite(v9_pred.prob_short):
                self.n_v10_pred_none_or_nan += 1
                if self._v10_log_count < 3:
                    log.warning(
                        "[ENTRY_V10] Call #%d: Prediction has NaN/Inf: prob_long=%.4f prob_short=%.4f",
                        self.n_v10_calls, v9_pred.prob_long, v9_pred.prob_short
                    )
                    self._v10_log_count += 1
            else:
                self.n_v10_pred_ok += 1
                if self._v10_log_count < 3:
                    log.info(
                        "[ENTRY_V10] Call #%d: OK prediction: prob_long=%.4f prob_short=%.4f margin=%.4f",
                        self.n_v10_calls, v9_pred.prob_long, v9_pred.prob_short, v9_pred.margin
                    )
                    self._v10_log_count += 1
                
                # TELEMETRY CONTRACT: Track candidates for V10 predictions (SNIPER-first)
                # n_candidates and n_predictions are the same for V10 (valid prediction = candidate)
                self.entry_telemetry["n_candidates"] += 1
                self.entry_telemetry["n_predictions"] += 1
                self.entry_telemetry["p_long_values"].append(float(v9_pred.prob_long))
                session_key = current_session
                self.entry_telemetry["candidate_sessions"][session_key] = self.entry_telemetry["candidate_sessions"].get(session_key, 0) + 1
                # DEPRECATED: farm_diag removed (SNIPER/NY uses entry_telemetry)
                # Legacy farm_diag updates removed
        
        if v9_pred is None:
            # DEL 1: Use generic [ENTRY] prefix in replay-mode (no V9 references)
            log_prefix = "[ENTRY_V10_CTX]" if (hasattr(self._runner, "replay_mode") and self._runner.replay_mode) else "[ENTRY_V9]"
            log.warning(f"{log_prefix} entry_pred=None - no entry this bar")
            
            # SNIPER logging: v9_pred None
            if use_sniper:
                self._log_sniper_cycle(
                    ts=current_ts,
                    session=current_session,
                    in_scope=sniper_in_scope,
                    warmup_ready=sniper_warmup_ready,
                    degraded=sniper_degraded,
                    spread_bps=current_spread_bps,
                    atr_bps=current_atr_bps,
                    eval_ran=False,
                    reason="v9_pred_none",
                    decision="NO-TRADE",
                )
            
            return None

        # KILL-CHAIN: EntryPrediction produced (SSoT)
        self.killchain_n_entry_pred_total += 1
        
        # FARM_V2B diagnostic: raw candidate (V9 gave a prediction)
        # Note: If V10 is enabled, candidates are already counted above (to avoid double-counting)
        if is_farm_v2b and not (hasattr(self._runner, "entry_v10_enabled") and self._runner.entry_v10_enabled):
            self.farm_diag["n_raw_candidates"] += 1
            # Track session/ATR regime for this candidate
            session_key = current_session
            self.farm_diag["sessions"][session_key] = self.farm_diag["sessions"].get(session_key, 0) + 1
            # Get ATR regime from policy_state (set during FARM regime inference)
            atr_key = policy_state.get("_atr_regime_id", vol)
            self.farm_diag["atr_regimes"][atr_key] = self.farm_diag["atr_regimes"].get(atr_key, 0) + 1
            # Track FARM regime if available
            farm_regime_key = policy_state.get("_farm_regime", policy_state.get("farm_regime", "UNKNOWN"))
            self.farm_diag["farm_regimes"][farm_regime_key] = self.farm_diag["farm_regimes"].get(farm_regime_key, 0) + 1
            # Store p_long for diagnosis
            self.farm_diag["p_long_values"].append(float(v9_pred.prob_long))
            
            # FARM_V2B diagnostic: track FARM regime (after inference, counted here when V9 gives signal)
            if farm_regime_key != "FARM_OUT_OF_SCOPE":
                self.farm_diag["n_after_farm_regime"] += 1
        elif is_farm_v2b:
            # V10 is enabled - candidates already counted, but track FARM-specific diagnostics
            # Get ATR regime from policy_state (set during FARM regime inference)
            atr_key = policy_state.get("_atr_regime_id", vol)
            self.farm_diag["atr_regimes"][atr_key] = self.farm_diag["atr_regimes"].get(atr_key, 0) + 1
            # Track FARM regime if available
            farm_regime_key = policy_state.get("_farm_regime", policy_state.get("farm_regime", "UNKNOWN"))
            self.farm_diag["farm_regimes"][farm_regime_key] = self.farm_diag["farm_regimes"].get(farm_regime_key, 0) + 1
            
            # FARM_V2B diagnostic: track FARM regime (after inference, counted here when V9 gives signal)
            if farm_regime_key != "FARM_OUT_OF_SCOPE":
                self.farm_diag["n_after_farm_regime"] += 1
        
        # Apply ENTRY_V9_POLICY (FARM_V2B, FARM_V2, FARM_V1, BASE_V1, or V1) if enabled
        # Check in priority order: FARM_V2B > FARM_V2 > FARM_V1 > BASE_V1 > V1
        policy_sniper_cfg = self.policy.get("entry_v9_policy_sniper", {})
        policy_farm_v2b_cfg = self.policy.get("entry_v9_policy_farm_v2b", {})
        policy_farm_v2_cfg = self.policy.get("entry_v9_policy_farm_v2", {})
        policy_farm_v1_cfg = self.policy.get("entry_v9_policy_farm_v1", {})
        policy_base_v1_cfg = self.policy.get("entry_v9_policy_base_v1", {})
        policy_v1_cfg = self.policy.get("entry_v9_policy_v1", {})
        
        use_sniper = policy_sniper_cfg.get("enabled", False)
        use_farm_v2b = policy_farm_v2b_cfg.get("enabled", False) and not use_sniper
        use_farm_v2 = policy_farm_v2_cfg.get("enabled", False) and not use_sniper and not use_farm_v2b
        use_farm_v1 = policy_farm_v1_cfg.get("enabled", False) and not use_sniper and not use_farm_v2b and not use_farm_v2
        use_base_v1 = policy_base_v1_cfg.get("enabled", False) and not use_sniper and not use_farm_v2b and not use_farm_v2 and not use_farm_v1
        use_v1 = policy_v1_cfg.get("enabled", False) and not use_sniper and not use_farm_v2b and not use_farm_v2 and not use_farm_v1 and not use_base_v1
        
        if use_sniper:
            # DEL 1: Read policy_module from YAML (SSoT - source of truth in YAML, no auto-switch)
            replay_config = self.policy.get("replay_config", {})
            if hasattr(self._runner, "replay_mode") and self._runner.replay_mode:
                # Replay mode: MUST use policy_module from YAML (hard-fail if missing or wrong)
                policy_module = replay_config.get("policy_module")
                if not policy_module:
                    raise RuntimeError(
                        "REPLAY_CONFIG_REQUIRED: replay_config.policy_module is missing in policy YAML. "
                        "Set replay_config.policy_module to 'gx1.policy.entry_policy_sniper_v10_ctx' in replay policy."
                    )
                
                # Hard-fail if policy_module is V9
                if "v9" in policy_module.lower() or "entry_v9" in policy_module.lower():
                    raise RuntimeError(
                        f"REPLAY_V9_POLICY_FORBIDDEN: replay_config.policy_module='{policy_module}' contains V9 reference. "
                        f"V9 policies are forbidden in replay mode. Set policy_module='gx1.policy.entry_policy_sniper_v10_ctx'"
                    )
                
                # Hard-fail if policy_module is not the expected V10 wrapper
                expected_module = "gx1.policy.entry_policy_sniper_v10_ctx"
                if policy_module != expected_module:
                    raise RuntimeError(
                        f"REPLAY_POLICY_MISMATCH: replay_config.policy_module='{policy_module}' does not match expected '{expected_module}'. "
                        f"Only V10_CTX policies are allowed in replay mode."
                    )
                
                # Import policy module from YAML (explicit, no auto-switch)
                # Map policy_module to known function imports (explicit, no dynamic import)
                if policy_module == "gx1.policy.entry_policy_sniper_v10_ctx":
                    from gx1.policy.entry_policy_sniper_v10_ctx import apply_entry_policy_sniper_v10_ctx
                    apply_policy_fn = apply_entry_policy_sniper_v10_ctx
                else:
                    raise RuntimeError(
                        f"REPLAY_POLICY_UNKNOWN: Unknown policy_module='{policy_module}'. "
                        f"Only 'gx1.policy.entry_policy_sniper_v10_ctx' is allowed in replay mode."
                    )
                
                from gx1.policy.farm_guards import sniper_guard_v1
                
                # Get policy_flag_col_name from config (or infer from policy_module)
                policy_flag_col_name = replay_config.get("policy_id", "entry_policy_sniper_v10_ctx")
                
                log.info(f"[ENTRY_MANAGER] Replay mode: Using policy_module from YAML: {policy_module}")
                log.info(f"[ENTRY_MANAGER] Policy flag column: {policy_flag_col_name}")
            else:
                # Live mode: use V9 policy (backward compatible)
                from gx1.policy.entry_v9_policy_sniper import apply_entry_v9_policy_sniper
                from gx1.policy.farm_guards import sniper_guard_v1
                apply_policy_fn = apply_entry_v9_policy_sniper
                policy_flag_col_name = "entry_v9_policy_sniper"
            
            # Get allow_high_vol and allow_extreme_vol from config
            policy_cfg = self.policy.get("entry_v9_policy_sniper", {})
            allow_high_vol = policy_cfg.get("allow_high_vol", True)
            allow_extreme_vol = policy_cfg.get("allow_extreme_vol", False)

            # KILL-CHAIN: Above-threshold check (BEFORE post-gates / guards)
            # This is the "model signal exists?" counter, independent of later rejections.
            killchain_min_prob_long = float(policy_sniper_cfg.get("min_prob_long", 0.67))
            killchain_min_prob_short = float(policy_sniper_cfg.get("min_prob_short", 0.72))
            killchain_allow_short = bool(policy_sniper_cfg.get("allow_short", False))
            import os as os_module
            killchain_analysis_mode = os_module.getenv("GX1_ANALYSIS_MODE") == "1"
            killchain_threshold_override = os_module.getenv("GX1_ENTRY_THRESHOLD_OVERRIDE")
            if killchain_analysis_mode and killchain_threshold_override is not None:
                try:
                    override_value = float(killchain_threshold_override)
                    killchain_min_prob_long = override_value
                    killchain_min_prob_short = override_value
                except (ValueError, TypeError):
                    # Ignore invalid override (policy evaluation will log warning)
                    pass
            killchain_above_threshold = bool(
                (float(v9_pred.prob_long) >= killchain_min_prob_long)
                or (killchain_allow_short and float(v9_pred.prob_short) >= killchain_min_prob_short)
            )
            if killchain_above_threshold:
                self.killchain_n_above_threshold += 1
            
            # Build current_row for SNIPER (needed for guard)
            current_row = entry_bundle.features.iloc[-1:].copy()
            current_row["prob_long"] = v9_pred.prob_long
            current_row["prob_short"] = v9_pred.prob_short
            
            # CRITICAL: Add close and _v1_atr14 for meta-model feature mapping
            if "close" not in current_row.columns:
                current_row["close"] = entry_bundle.close_price
            if "_v1_atr14" not in current_row.columns and "_v1_atr14" in entry_bundle.raw_row.index:
                current_row["_v1_atr14"] = entry_bundle.raw_row["_v1_atr14"]
            
            # Ensure ts column exists
            if "ts" not in current_row.columns:
                if isinstance(current_row.index, pd.DatetimeIndex):
                    current_row = current_row.reset_index()
                    if len(current_row.columns) > 0:
                        current_row = current_row.rename(columns={current_row.columns[0]: "ts"})
            
            # REPLAY FEATURE PARITY: Ensure tags are set from candles if missing/UNKNOWN
            # This must happen BEFORE guards run, but AFTER current_row is built
            if hasattr(self, "replay_mode") and self.replay_mode and candles is not None:
                # Check if tags are missing or UNKNOWN
                needs_tags = (
                    policy_state.get("brain_trend_regime", "UNKNOWN") == "UNKNOWN" or
                    policy_state.get("brain_vol_regime", "UNKNOWN") == "UNKNOWN" or
                    "session" not in policy_state or policy_state.get("session") == "UNKNOWN"
                )
                if needs_tags:
                    from gx1.execution.replay_features import ensure_replay_tags
                    current_ts = candles.index[-1] if len(candles) > 0 else None
                    current_row_for_tags = current_row.iloc[0] if isinstance(current_row, pd.DataFrame) and len(current_row) > 0 else current_row
                    current_row_for_tags, policy_state = ensure_replay_tags(
                        current_row_for_tags,
                        candles,
                        policy_state,
                        current_ts=current_ts,
                    )
                    # CRITICAL FIX: Update current_row DataFrame with tags from Series
                    if isinstance(current_row, pd.DataFrame) and len(current_row) > 0:
                        # Update each tag column individually using .loc (more reliable than direct assignment)
                        for col in ["session", "vol_regime", "atr_regime", "trend_regime", "_v1_atr_regime_id"]:
                            if col in current_row_for_tags.index:
                                current_row.loc[current_row.index[0], col] = current_row_for_tags[col]
                        # Also ensure session is set from policy_state if available (CRITICAL FIX)
                        if "session" in policy_state:
                            current_row.loc[current_row.index[0], "session"] = policy_state["session"]
            
            # CRITICAL FIX: Ensure session is ALWAYS set in current_row before guard (fallback)
            if isinstance(current_row, pd.DataFrame) and len(current_row) > 0:
                current_session_value = None
                if "session" in current_row.columns:
                    current_session_value = current_row["session"].iloc[0]
                
                if current_session_value is None or current_session_value == "UNKNOWN" or pd.isna(current_session_value):
                    # Try policy_state first
                    if "session" in policy_state and policy_state["session"] != "UNKNOWN":
                        current_row.loc[current_row.index[0], "session"] = policy_state["session"]
                    else:
                        # Infer from timestamp as last resort
                        current_ts = candles.index[-1] if candles is not None and len(candles) > 0 else None
                        if current_ts:
                            session_tag = infer_session_tag(current_ts)
                            current_row.loc[current_row.index[0], "session"] = session_tag
                            policy_state["session"] = session_tag
                
                # DIAGNOSTIC: Log if session is still UNKNOWN after all fixes (rate-limited)
                final_session = current_row["session"].iloc[0] if "session" in current_row.columns else None
                if final_session == "UNKNOWN" or final_session is None:
                    if not hasattr(self, "_session_unknown_log_counter"):
                        self._session_unknown_log_counter = 0
                    self._session_unknown_log_counter += 1
                    if self._session_unknown_log_counter % 500 == 0:
                        current_ts = candles.index[-1] if candles is not None and len(candles) > 0 else None
                        log.warning(
                            "[SESSION_UNKNOWN_DIAG] session=UNKNOWN after all fixes: ts=%s candles_len=%d policy_state_session=%s",
                            current_ts.isoformat() if current_ts else "N/A",
                            len(candles) if candles is not None else 0,
                            policy_state.get("session", "N/A")
                        )
            
            # SNIPER GUARD: Apply centralized guard BEFORE policy (EU/OVERLAP/US + (LOW|MEDIUM|HIGH))
            try:
                # KILL-CHAIN: Session/vol guard pass counters (funnel, conditioned on above-threshold)
                if killchain_above_threshold:
                    from gx1.policy.farm_guards import _extract_session_vol_regime
                    kc_session, kc_vol_regime = _extract_session_vol_regime(current_row.iloc[0])
                    if kc_session in ["EU", "OVERLAP", "US"]:
                        self.killchain_n_after_session_guard += 1
                        allowed_vol = ["LOW", "MEDIUM"]
                        if allow_high_vol:
                            allowed_vol.append("HIGH")
                        if allow_extreme_vol:
                            allowed_vol.append("EXTREME")
                        if kc_vol_regime in allowed_vol:
                            self.killchain_n_after_vol_guard += 1

                sniper_guard_v1(
                    current_row.iloc[0], 
                    context="live_runner_pre_policy_sniper",
                    allow_high_vol=allow_high_vol,
                    allow_extreme_vol=allow_extreme_vol
                )
            except AssertionError as e:
                # Track vol_regime=UNKNOWN rejections (expected in SNIPER replay, not an error)
                error_str = str(e)
                if "vol_regime=UNKNOWN" in error_str:
                    self.entry_telemetry["vol_regime_unknown_count"] += 1
                    # Log as debug (not error) to reduce log noise
                    log.debug(f"[ENTRY] SNIPER entry rejected by guard: vol_regime=UNKNOWN (expected in SNIPER replay)")
                else:
                    log.debug(f"[ENTRY] SNIPER entry rejected by sniper guard: {e}")

                # KILL-CHAIN: classify guard block (stable reason codes)
                error_str = str(e)
                if "session=" in error_str and "not in" in error_str:
                    self._killchain_inc_reason("BLOCK_SESSION")
                elif "vol_regime=" in error_str and "not in" in error_str:
                    self._killchain_inc_reason("BLOCK_VOL")
                else:
                    self._killchain_inc_reason("BLOCK_UNKNOWN")
                    try:
                        current_ts = candles.index[-1] if candles is not None and len(candles) > 0 else None
                        self._killchain_record_unknown(
                            {
                                "where": "SNIPER_GUARD_V1",
                                "ts": current_ts.isoformat() if current_ts is not None else None,
                                "error": error_str[:300],
                                "session": str(current_row.get("session", [None])[0]) if isinstance(current_row, pd.DataFrame) and len(current_row) > 0 else None,
                                "vol_regime": str(current_row.get("vol_regime", [None])[0]) if isinstance(current_row, pd.DataFrame) and len(current_row) > 0 else None,
                            }
                        )
                    except Exception:
                        pass
                return None
            
            # SNIPER RISK GUARD V1: Apply risk guard (spread/vol blocks, cooldown, clamps)
            # Initialize guard if not already initialized
            if self._sniper_risk_guard is None:
                risk_guard_cfg_path = self.policy.get("risk_guard", {}).get("config_path")
                if risk_guard_cfg_path:
                    try:
                        import yaml
                        from pathlib import Path
                        from gx1.policy.sniper_risk_guard import SniperRiskGuardV1
                        
                        guard_cfg_path = Path(risk_guard_cfg_path)
                        if not guard_cfg_path.is_absolute():
                            # Resolve relative to project root
                            project_root = Path(__file__).resolve().parent.parent.parent
                            guard_cfg_path = project_root / guard_cfg_path
                        
                        if guard_cfg_path.exists():
                            with open(guard_cfg_path) as f:
                                guard_cfg = yaml.safe_load(f)
                            self._sniper_risk_guard = SniperRiskGuardV1(guard_cfg)
                            log.info(f"[SNIPER_RISK_GUARD] Initialized from {risk_guard_cfg_path}")
                        else:
                            log.warning(f"[SNIPER_RISK_GUARD] Config file not found: {guard_cfg_path}")
                    except Exception as e:
                        log.warning(f"[SNIPER_RISK_GUARD] Failed to initialize: {e}")
            
            # Apply risk guard if initialized
            risk_guard_blocked = False
            risk_guard_reason = None
            risk_guard_details = {}
            risk_guard_clamp = None
            
            if self._sniper_risk_guard is not None and self._sniper_risk_guard.enabled:
                # Build entry snapshot and feature context for guard
                entry_snapshot = {
                    "session": current_row["session"].iloc[0] if "session" in current_row.columns and len(current_row) > 0 else None,
                    "vol_regime": current_row["vol_regime"].iloc[0] if "vol_regime" in current_row.columns and len(current_row) > 0 else None,
                    "spread_bps": current_row.get("spread_bps", [None])[0] if "spread_bps" in current_row.columns and len(current_row) > 0 else None,
                    "atr_bps": current_row.get("atr_bps", [None])[0] if "atr_bps" in current_row.columns and len(current_row) > 0 else None,
                }
                
                feature_context = {
                    "spread_bps": entry_bundle.spread_bps if hasattr(entry_bundle, "spread_bps") else None,
                    "atr_bps": current_atr_bps,
                }
                
                # Use len(candles) as bar index proxy
                current_bar_index = len(candles) if candles is not None else 0
                
                # Check if entry should be blocked
                should_block, reason_code, details = self._sniper_risk_guard.should_block(
                    entry_snapshot,
                    feature_context,
                    policy_state,
                    current_bar_index,
                )
                
                if should_block:
                    risk_guard_blocked = True
                    risk_guard_reason = reason_code
                    risk_guard_details = details
                    log.info(f"[SNIPER_RISK_GUARD] Blocked entry: reason={reason_code}, details={details}")

                    # KILL-CHAIN: risk/cooldown block (stable reason codes)
                    if isinstance(reason_code, str) and "cooldown" in reason_code.lower():
                        self._killchain_inc_reason("BLOCK_COOLDOWN")
                    else:
                        self._killchain_inc_reason("BLOCK_RISK")
                    
                    # Log blocked entry attempt (if journal supports it)
                    if hasattr(self, "trade_journal") and self.trade_journal:
                        try:
                            # Try to log as entry attempt (if method exists)
                            if hasattr(self.trade_journal, "log_entry_attempt"):
                                self.trade_journal.log_entry_attempt(
                                    entry_time=candles.index[-1].isoformat() if candles is not None and len(candles) > 0 else None,
                                    reason=reason_code,
                                    details=details,
                                )
                        except Exception as e:
                            log.debug(f"[SNIPER_RISK_GUARD] Failed to log entry attempt: {e}")
                    
                    # SNIPER logging: risk guard blocked
                    if use_sniper:
                        self._log_sniper_cycle(
                            ts=current_ts,
                            session=current_session,
                            in_scope=sniper_in_scope,
                            warmup_ready=sniper_warmup_ready,
                            degraded=sniper_degraded,
                            spread_bps=current_spread_bps,
                            atr_bps=current_atr_bps,
                            eval_ran=False,
                            reason=f"risk_guard_{reason_code}",
                            decision="NO-TRADE",
                        )
                    
                    return None
                
                # Check for session clamp (adjust min_prob_long locally)
                session = entry_snapshot.get("session") or policy_state.get("session")
                clamp = self._sniper_risk_guard.get_session_clamp(session)
                if clamp is not None and clamp > 0:
                    risk_guard_clamp = clamp
                    # Store clamp info for later use in policy evaluation
                    policy_state["risk_guard_min_prob_long_clamp"] = clamp
                    log.debug(f"[SNIPER_RISK_GUARD] Session clamp applied: session={session}, clamp=+{clamp}")
                
                # Store risk guard metadata in policy_state for trade journal
                policy_state["risk_guard_blocked"] = risk_guard_blocked
                policy_state["risk_guard_reason"] = risk_guard_reason
                policy_state["risk_guard_details"] = risk_guard_details
                policy_state["risk_guard_clamp"] = risk_guard_clamp
            
            # Use meta-model loaded at init (same as FARM_V2B)
            meta_model = getattr(self, "farm_entry_meta_model", None)
            meta_feature_cols = getattr(self, "farm_entry_meta_feature_cols", None)
            
            # If feature columns not loaded, auto-detect (fallback)
            if meta_model is not None and meta_feature_cols is None:
                numeric_cols = current_row.select_dtypes(include=[np.number]).columns.tolist()
                exclude_cols = {'prob_long', 'prob_short', 'prob_neutral', 'p_long', 'p_profitable', 
                                'y_profitable_8b', 'pnl_bps_signed', 'mfe_8b_signed', 'mae_8b_signed',
                                'trade_id', 'entry_ts', 'exit_ts'}
                meta_feature_cols = [c for c in numeric_cols if c not in exclude_cols]
                log.info(f"[ENTRY] Auto-detected {len(meta_feature_cols)} feature columns for meta-model")
            
            # Check if we have any candidates at all
            if len(current_row) == 0:
                log.debug("[ENTRY] SNIPER: No entry candidates in batch (empty current_row)")
                policy_state["p_long"] = v9_pred.prob_long
                policy_state["p_profitable"] = None
            else:
                # Apply policy (will compute p_profitable for logging, but NOT use it for filtering)
                # Store threshold_used BEFORE policy evaluation (for telemetry)
                min_prob_long = float(policy_sniper_cfg.get("min_prob_long", 0.67))
                min_prob_short = float(policy_sniper_cfg.get("min_prob_short", 0.72))
                allow_short = policy_sniper_cfg.get("allow_short", False)
                
                # STEG 1: Threshold override (ANALYSIS MODE ONLY)
                # Allow runtime override via env var if GX1_ANALYSIS_MODE=1
                import os as os_module
                analysis_mode = os_module.getenv("GX1_ANALYSIS_MODE") == "1"
                threshold_override = os_module.getenv("GX1_ENTRY_THRESHOLD_OVERRIDE")
                
                if analysis_mode and threshold_override is not None:
                    try:
                        override_value = float(threshold_override)
                        min_prob_long = override_value
                        # Also override short threshold to same value (for simplicity in analysis)
                        min_prob_short = override_value
                        log.info(f"[ANALYSIS_MODE] Entry threshold override: {min_prob_long} (original: {policy_sniper_cfg.get('min_prob_long', 0.67)})")
                    except (ValueError, TypeError) as e:
                        log.warning(f"[ANALYSIS_MODE] Invalid GX1_ENTRY_THRESHOLD_OVERRIDE={threshold_override}, using config value: {e}")
                
                # Format threshold_used as string (similar to entry_gating path)
                if allow_short:
                    self.threshold_used = f"long={min_prob_long},short={min_prob_short}"
                else:
                    self.threshold_used = f"long={min_prob_long}"
                
                # Override threshold in policy config if analysis mode
                if analysis_mode and threshold_override is not None:
                    # Create modified policy config with override
                    policy_config_override = self.policy.copy()
                    if "entry_policy_sniper_v10_ctx" in policy_config_override:
                        policy_config_override["entry_policy_sniper_v10_ctx"]["min_prob_long"] = min_prob_long
                        policy_config_override["entry_policy_sniper_v10_ctx"]["min_prob_short"] = min_prob_short
                    elif "entry_v9_policy_sniper" in policy_config_override:
                        policy_config_override["entry_v9_policy_sniper"]["min_prob_long"] = min_prob_long
                        policy_config_override["entry_v9_policy_sniper"]["min_prob_short"] = min_prob_short
                    
                    df_policy = apply_policy_fn(
                        current_row,
                        policy_config_override,  # Use override config
                        meta_model=meta_model,
                        meta_feature_cols=meta_feature_cols,
                    )
                else:
                    df_policy = apply_policy_fn(
                        current_row,
                        self.policy,
                        meta_model=meta_model,
                        meta_feature_cols=meta_feature_cols,
                    )
                policy_flag_col = policy_flag_col_name
                
                # Store p_long and p_profitable for trade creation (if signal passed)
                # Handle both V9 and V10 policy flag columns (backward compatible)
                if policy_flag_col not in df_policy.columns and "entry_v9_policy_sniper" in df_policy.columns:
                    # Fallback: V9 wrapper may have set both columns, use V9 column if V10 column missing
                    policy_flag_col = "entry_v9_policy_sniper"
                
                if len(df_policy) > 0 and policy_flag_col in df_policy.columns and df_policy[policy_flag_col].sum() > 0:
                    # Signal passed - extract values from accepted row
                    accepted_row = df_policy[df_policy[policy_flag_col]].iloc[0]
                    policy_state["p_long"] = accepted_row.get("p_long", v9_pred.prob_long)
                    policy_state["p_profitable"] = accepted_row.get("p_profitable", None)
                    # Store policy-determined side if available
                    if "_policy_side" in accepted_row:
                        policy_state["_policy_side"] = accepted_row["_policy_side"]
                else:
                    # Signal rejected - track as threshold veto (Stage-1)
                    self.veto_cand["veto_cand_threshold"] += 1
                    # KILL-CHAIN: policy-stage block (threshold gate)
                    self._killchain_inc_reason("BLOCK_BELOW_THRESHOLD")
                    # DEPRECATED: veto_counters removed (SNIPER/NY uses veto_cand)
                    # Store p_long for logging
                    policy_state["p_long"] = df_policy["p_long"].iloc[0] if len(df_policy) > 0 and "p_long" in df_policy.columns else v9_pred.prob_long
                    policy_state["p_profitable"] = df_policy["p_profitable"].iloc[0] if len(df_policy) > 0 and "p_profitable" in df_policy.columns else None
                    # SNIPER logging: policy eval ran but no signals
                    if use_sniper:
                        n_signals = int(df_policy[policy_flag_col].sum()) if len(df_policy) > 0 else 0
                        p_long_val = policy_state.get("p_long")
                        self._log_sniper_cycle(
                            ts=current_ts,
                            session=current_session,
                            in_scope=sniper_in_scope,
                            warmup_ready=sniper_warmup_ready,
                            degraded=sniper_degraded,
                            spread_bps=current_spread_bps,
                            atr_bps=current_atr_bps,
                            eval_ran=True,
                            reason="policy_no_signals",
                            decision="NO-TRADE",
                            n_signals=n_signals,
                            p_long=p_long_val,
                        )
                        # Shadow threshold logging (safe - no orders)
                        if p_long_val is not None:
                            min_prob_long = float(policy_sniper_cfg.get("min_prob_long", 0.67))
                            trend_regime = policy_state.get("brain_trend_regime")
                            vol_regime = policy_state.get("brain_vol_regime")
                            self._log_sniper_shadow(
                                ts=current_ts,
                                session=current_session,
                                p_long=p_long_val,
                                real_threshold=min_prob_long,
                                real_trade=False,
                                spread_bps=current_spread_bps,
                                atr_bps=current_atr_bps,
                                trend_regime=trend_regime,
                                vol_regime=vol_regime,
                                real_decision="NO-TRADE",
                            )
                    
                    # DEL 1: Hook PolicyDecisionCollector - collect "skip" decision
                    if hasattr(self._runner, "replay_eval_collectors") and self._runner.replay_eval_collectors:
                        decision_collector = self._runner.replay_eval_collectors.get("policy_decisions")
                        if decision_collector:
                            reasons = ["policy_no_signals"]
                            decision_collector.collect(
                                timestamp=current_ts,
                                decision="skip",
                                reasons=reasons,
                            )
                    
                    return None
                
                # Check if signal passed
                if df_policy[policy_flag_col].sum() == 0:
                    # SNIPER logging: policy eval ran but signal sum is 0
                    if use_sniper:
                        n_signals = 0
                        p_long_val = policy_state.get("p_long")
                        self._log_sniper_cycle(
                            ts=current_ts,
                            session=current_session,
                            in_scope=sniper_in_scope,
                            warmup_ready=sniper_warmup_ready,
                            degraded=sniper_degraded,
                            spread_bps=current_spread_bps,
                            atr_bps=current_atr_bps,
                            eval_ran=True,
                            reason="policy_no_signals",
                            decision="NO-TRADE",
                            n_signals=n_signals,
                            p_long=p_long_val,
                        )
                        # Shadow threshold logging (safe - no orders)
                        if p_long_val is not None:
                            min_prob_long = float(policy_sniper_cfg.get("min_prob_long", 0.67))
                            trend_regime = policy_state.get("brain_trend_regime")
                            vol_regime = policy_state.get("brain_vol_regime")
                            self._log_sniper_shadow(
                                ts=current_ts,
                                session=current_session,
                                p_long=p_long_val,
                                real_threshold=min_prob_long,
                                real_trade=False,
                                spread_bps=current_spread_bps,
                                atr_bps=current_atr_bps,
                                trend_regime=trend_regime,
                                vol_regime=vol_regime,
                                real_decision="NO-TRADE",
                            )
                    
                    # DEL 1: Hook PolicyDecisionCollector - collect "skip" decision
                    if hasattr(self._runner, "replay_eval_collectors") and self._runner.replay_eval_collectors:
                        decision_collector = self._runner.replay_eval_collectors.get("policy_decisions")
                        if decision_collector:
                            reasons = ["policy_no_signals"]
                            decision_collector.collect(
                                timestamp=current_ts,
                                decision="skip",
                                reasons=reasons,
                            )
                    
                    # KILL-CHAIN: policy-stage block (threshold gate)
                    self._killchain_inc_reason("BLOCK_BELOW_THRESHOLD")
                    return None
                
                # Signal passed - continue to trade creation
                # SNIPER shadow logging: log even when real trade occurs (for trades/day estimate)
                if use_sniper:
                    p_long_val = policy_state.get("p_long")
                    if p_long_val is not None:
                        min_prob_long = float(policy_sniper_cfg.get("min_prob_long", 0.67))
                        trend_regime = policy_state.get("brain_trend_regime")
                        vol_regime = policy_state.get("brain_vol_regime")
                        self._log_sniper_shadow(
                            ts=current_ts,
                            session=current_session,
                            p_long=p_long_val,
                            real_threshold=min_prob_long,
                            real_trade=True,
                            spread_bps=current_spread_bps,
                            atr_bps=current_atr_bps,
                            trend_regime=trend_regime,
                            vol_regime=vol_regime,
                            real_decision="TRADE",
                        )
                
                policy_state["policy_name"] = "V9_SNIPER"
                prediction = v9_pred
                policy_state["entry_model_active"] = "ENTRY_V9"
                # SNIPER policy complete - skip other policy blocks and continue to trade creation
        elif use_farm_v2b and not use_sniper:
            # Use FARM_V2B policy (p_long-driven, no meta-filter)
            policy_cfg = policy_farm_v2b_cfg
        elif use_farm_v2:
            # Use FARM_V2 policy (meta-model enhanced)
            policy_cfg = policy_farm_v2_cfg
        elif use_farm_v1:
            # Use FARM_V1 policy (bondegård-modus)
            policy_cfg = policy_farm_v1_cfg
        elif use_base_v1:
            # Use BASE_V1 policy (data-driven)
            policy_cfg = policy_base_v1_cfg
        elif use_v1:
            # Use V1 policy (original)
            policy_cfg = policy_v1_cfg
        else:
            policy_cfg = None
        
        if use_sniper:
            # SNIPER policy already handled above - skip other policy blocks
            pass
        elif policy_cfg is not None and policy_cfg.get("enabled", False):
            # Build DataFrame with single row for policy evaluation
            current_row = entry_bundle.features.iloc[-1:].copy()
            current_row["prob_long"] = v9_pred.prob_long
            current_row["prob_short"] = v9_pred.prob_short
            
            # CRITICAL: Add close and _v1_atr14 for meta-model feature mapping
            # These are needed by build_meta_feature_vector to correctly compute atr_bps
            if "close" not in current_row.columns:
                current_row["close"] = entry_bundle.close_price
            if "_v1_atr14" not in current_row.columns and "_v1_atr14" in entry_bundle.raw_row.index:
                current_row["_v1_atr14"] = entry_bundle.raw_row["_v1_atr14"]
            
            # Ensure ts column exists
            if "ts" not in current_row.columns:
                if isinstance(current_row.index, pd.DatetimeIndex):
                    current_row = current_row.reset_index()
                    if len(current_row.columns) > 0:
                        current_row = current_row.rename(columns={current_row.columns[0]: "ts"})
            
            # REPLAY FEATURE PARITY: Ensure tags are set from candles if missing/UNKNOWN
            # This must happen BEFORE guards run, but AFTER current_row is built
            if hasattr(self, "replay_mode") and self.replay_mode and candles is not None:
                # Check if tags are missing or UNKNOWN
                needs_tags = (
                    policy_state.get("brain_trend_regime", "UNKNOWN") == "UNKNOWN" or
                    policy_state.get("brain_vol_regime", "UNKNOWN") == "UNKNOWN" or
                    "session" not in policy_state or policy_state.get("session") == "UNKNOWN"
                )
                if needs_tags:
                    from gx1.execution.replay_features import ensure_replay_tags
                    current_ts = candles.index[-1] if len(candles) > 0 else None
                    current_row_for_tags = current_row.iloc[0] if isinstance(current_row, pd.DataFrame) and len(current_row) > 0 else current_row
                    current_row_for_tags, policy_state = ensure_replay_tags(
                        current_row_for_tags,
                        candles,
                        policy_state,
                        current_ts=current_ts,
                    )
                    # Update current_row DataFrame with tags from Series
                    if isinstance(current_row, pd.DataFrame) and len(current_row) > 0:
                        # Update each tag column individually (iloc assignment doesn't work for Series -> DataFrame)
                        for col in ["session", "vol_regime", "atr_regime", "trend_regime", "_v1_atr_regime_id"]:
                            if col in current_row_for_tags.index:
                                current_row[col] = current_row_for_tags[col]
                        # Also ensure session is set from policy_state if available
                        if "session" in policy_state:
                            current_row["session"] = policy_state["session"]
            
            # Add regime columns from Big Brain V1 if available
            if hasattr(self, "big_brain_v1") and self.big_brain_v1 is not None:
                try:
                    # Get current bar's Big Brain predictions
                    bb_pred = self.big_brain_v1.predict(current_row)
                    if bb_pred is not None:
                        # Map Big Brain regime predictions to policy columns
                        if hasattr(bb_pred, "vol_regime") and bb_pred.vol_regime:
                            current_row["brain_vol_regime"] = bb_pred.vol_regime
                        if hasattr(bb_pred, "trend_regime") and bb_pred.trend_regime:
                            current_row["brain_trend_regime"] = bb_pred.trend_regime
                except Exception as e:
                    # DEL 1: Use generic [ENTRY] prefix in replay-mode (no V9 references)
                    log_prefix = "[ENTRY_V10_CTX]" if (hasattr(self._runner, "replay_mode") and self._runner.replay_mode) else "[ENTRY_V9]"
                    log.debug(f"{log_prefix} Could not add Big Brain regime columns: %s", e)
            
            # Add session column if missing (try to infer from timestamp or existing columns)
            if "session" not in current_row.columns:
                # Try session_id mapping (from build_sequence_features)
                if "session_id" in current_row.columns:
                    session_map = {0: "EU", 1: "OVERLAP", 2: "US"}
                    current_row["session"] = current_row["session_id"].map(session_map).fillna("UNKNOWN")
                # Try _v1_session_tag
                elif "_v1_session_tag" in current_row.columns:
                    current_row["session"] = current_row["_v1_session_tag"]
                # Try session_tag
                elif "session_tag" in current_row.columns:
                    current_row["session"] = current_row["session_tag"]
                # Infer from timestamp if available
                elif "ts" in current_row.columns or isinstance(current_row.index, pd.DatetimeIndex):
                    ts = current_row.index[0] if isinstance(current_row.index, pd.DatetimeIndex) else pd.to_datetime(current_row["ts"].iloc[0])
                    current_row["session"] = infer_session_tag(ts)
            
            # Add atr_regime/vol_regime column if missing
            if "atr_regime" not in current_row.columns and "vol_regime" not in current_row.columns and "brain_vol_regime" not in current_row.columns:
                # Try atr_regime_id mapping (from build_sequence_features)
                if "_v1_atr_regime_id" in current_row.columns:
                    regime_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
                    current_row["atr_regime"] = current_row["_v1_atr_regime_id"].map(regime_map).fillna("UNKNOWN")
                    current_row["vol_regime"] = current_row["atr_regime"]  # Also set vol_regime for consistency
                elif "atr_regime_id" in current_row.columns:
                    regime_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
                    current_row["atr_regime"] = current_row["atr_regime_id"].map(regime_map).fillna("UNKNOWN")
                    current_row["vol_regime"] = current_row["atr_regime"]  # Also set vol_regime for consistency
            
            # Add trend_regime column if missing
            if "trend_regime" not in current_row.columns and "brain_trend_regime" not in current_row.columns:
                # Try trend_regime_tf24h (from build_sequence_features)
                if "trend_regime_tf24h" in current_row.columns:
                    # Map numeric to string: >0.001 = TREND_UP, < -0.001 = TREND_DOWN, else TREND_NEUTRAL
                    trend_val = float(current_row["trend_regime_tf24h"].iloc[0])
                    if trend_val > 0.001:
                        current_row["trend_regime"] = "TREND_UP"
                    elif trend_val < -0.001:
                        current_row["trend_regime"] = "TREND_DOWN"
                    else:
                        current_row["trend_regime"] = "TREND_NEUTRAL"
                # Also check for brain_trend_regime from Big Brain V1
                elif hasattr(self, "big_brain_v1") and self.big_brain_v1 is not None:
                    try:
                        bb_pred = self.big_brain_v1.predict(current_row)
                        if bb_pred and hasattr(bb_pred, "trend_regime") and bb_pred.trend_regime:
                            current_row["trend_regime"] = bb_pred.trend_regime
                    except:
                        pass
            
            # Apply policy (FARM_V2B, FARM_V2, FARM_V1, BASE_V1, or V1)
            if use_farm_v2b:
                from gx1.policy.entry_v9_policy_farm_v2b import apply_entry_v9_policy_farm_v2b
                from gx1.policy.farm_guards import farm_brutal_guard_v2, get_farm_entry_metadata_v2
                
                # Get allow_medium_vol from config
                policy_cfg = self.policy.get("entry_v9_policy_farm_v2b", {})
                allow_medium_vol = policy_cfg.get("allow_medium_vol", True)
                
                # Note: REPLAY FEATURE PARITY is already applied above (after current_row is built)
                # FARM_V2B already computes tags above, but replay_features ensures they're set even if that path fails
                
                # BRUTAL GUARD V2: Apply centralized guard BEFORE policy (ASIA + (LOW ∪ MEDIUM))
                try:
                    farm_brutal_guard_v2(current_row.iloc[0], context="live_runner_pre_policy_v2b", allow_medium_vol=allow_medium_vol)
                    # FARM_V2B diagnostic: brutal guard passed
                    self.farm_diag["n_after_brutal_guard"] += 1
                except AssertionError as e:
                    log.debug(f"[ENTRY] FARM_V2B entry rejected by brutal guard V2: {e}")
                    return None
                
                # Use meta-model loaded at init (optional for V2B - computed but not used for filtering)
                meta_model = getattr(self, "farm_entry_meta_model", None)
                meta_feature_cols = getattr(self, "farm_entry_meta_feature_cols", None)
                
                # If feature columns not loaded, auto-detect (fallback)
                if meta_model is not None and meta_feature_cols is None:
                    numeric_cols = current_row.select_dtypes(include=[np.number]).columns.tolist()
                    exclude_cols = {'prob_long', 'prob_short', 'prob_neutral', 'p_long', 'p_profitable', 
                                    'y_profitable_8b', 'pnl_bps_signed', 'mfe_8b_signed', 'mae_8b_signed',
                                    'trade_id', 'entry_ts', 'exit_ts'}
                    meta_feature_cols = [c for c in numeric_cols if c not in exclude_cols]
                    log.info(f"[ENTRY] Auto-detected {len(meta_feature_cols)} feature columns for meta-model")
                
                # Check if we have any candidates at all
                if len(current_row) == 0:
                    log.debug("[ENTRY] FARM_V2B: No entry candidates in batch (empty current_row)")
                    policy_state["p_long"] = v9_pred.prob_long
                    policy_state["p_profitable"] = None
                else:
                    # Apply policy (will compute p_profitable for logging, but NOT use it for filtering)
                    df_policy = apply_entry_v9_policy_farm_v2b(
                        current_row,
                        self.policy,
                        meta_model=meta_model,
                        meta_feature_cols=meta_feature_cols,
                    )
                    policy_flag_col = "entry_v9_policy_farm_v2b"
                    
                    # Store p_long and p_profitable for trade creation (if signal passed)
                    if len(df_policy) > 0 and df_policy["entry_v9_policy_farm_v2b"].sum() > 0:
                        # Signal passed - extract values from accepted row
                        accepted_row = df_policy[df_policy["entry_v9_policy_farm_v2b"]].iloc[0]
                        policy_state["p_long"] = accepted_row.get("p_long", v9_pred.prob_long)
                        policy_state["p_profitable"] = accepted_row.get("p_profitable", None)
                        # Store policy-determined side if available (for side-aware filtering)
                        if "_policy_side" in accepted_row:
                            policy_state["_policy_side"] = accepted_row["_policy_side"]
                    else:
                        # Signal rejected - still store p_long for logging
                        policy_state["p_long"] = df_policy["p_long"].iloc[0] if len(df_policy) > 0 and "p_long" in df_policy.columns else v9_pred.prob_long
                        policy_state["p_profitable"] = df_policy["p_profitable"].iloc[0] if len(df_policy) > 0 and "p_profitable" in df_policy.columns else None
                        return None
                    
                    # Check if signal passed
                    if df_policy[policy_flag_col].sum() == 0:
                        return None
                    
                    # Signal passed - continue to trade creation
                    policy_state["policy_name"] = "V9_FARM_V2B"
            elif use_farm_v2:
                from gx1.policy.entry_v9_policy_farm_v2 import apply_entry_v9_policy_farm_v2
                from gx1.policy.farm_guards import farm_brutal_guard_v2, get_farm_entry_metadata_v2
                
                # Get allow_medium_vol from config
                policy_cfg = self.policy.get("entry_v9_policy_farm_v2", {})
                allow_medium_vol = policy_cfg.get("allow_medium_vol", True)
                
                # BRUTAL GUARD V2: Apply centralized guard BEFORE policy (ASIA + (LOW ∪ MEDIUM))
                try:
                    farm_brutal_guard_v2(current_row.iloc[0], context="live_runner_pre_policy_v2", allow_medium_vol=allow_medium_vol)
                except AssertionError as e:
                    log.debug(f"[ENTRY] FARM_V2 entry rejected by brutal guard V2: {e}")
                    return None
                
                # Use meta-model loaded at init (FARM_V2 requires it)
                meta_model = getattr(self, "farm_entry_meta_model", None)
                meta_feature_cols = getattr(self, "farm_entry_meta_feature_cols", None)
                
                # If feature columns not loaded, auto-detect (fallback)
                if meta_model is not None and meta_feature_cols is None:
                    numeric_cols = current_row.select_dtypes(include=[np.number]).columns.tolist()
                    exclude_cols = {'prob_long', 'prob_short', 'prob_neutral', 'p_long', 'p_profitable', 
                                    'y_profitable_8b', 'pnl_bps_signed', 'mfe_8b_signed', 'mae_8b_signed',
                                    'trade_id', 'entry_ts', 'exit_ts'}
                    meta_feature_cols = [c for c in numeric_cols if c not in exclude_cols]
                    log.info(f"[ENTRY] Auto-detected {len(meta_feature_cols)} feature columns for meta-model")
                
                # Check if we have any candidates at all
                if len(current_row) == 0:
                    log.debug("[ENTRY] FARM_V2: No entry candidates in batch (empty current_row)")
                    policy_state["last_p_profitable_mean"] = 0.0
                    policy_state["last_p_profitable_gt_thresh"] = 0.0
                    policy_state["last_batch_size"] = 0
                    policy_state["p_long"] = v9_pred.prob_long
                    policy_state["p_profitable"] = None
                else:
                    # Apply policy (will compute p_profitable for ALL candidates before filtering)
                    df_policy = apply_entry_v9_policy_farm_v2(
                        current_row,
                        self.policy,
                        meta_model=meta_model,
                        meta_feature_cols=meta_feature_cols,
                    )
                    policy_flag_col = "entry_v9_policy_farm_v2"
                    
                    # Extract p_profitable stats from input candidates (before policy filtering)
                    # df_policy should have p_profitable for all rows, even if policy rejected them
                    if "p_profitable" in df_policy.columns and len(df_policy) > 0:
                        policy_cfg = self.policy.get("entry_v9_policy_farm_v2", {})
                        min_prob_profitable = float(policy_cfg.get("min_prob_profitable", 0.50))
                        
                        # Always update stats for input candidates (before filtering)
                        policy_state["last_p_profitable_mean"] = float(df_policy["p_profitable"].mean())
                        policy_state["last_p_profitable_gt_thresh"] = float((df_policy["p_profitable"] >= min_prob_profitable).mean())
                        policy_state["last_batch_size"] = int(len(df_policy))
                        
                        # Log if no candidates passed policy (but we had input candidates)
                        if df_policy["entry_v9_policy_farm_v2"].sum() == 0:
                            log.debug(
                                f"[ENTRY] FARM_V2: No candidates passed policy (input: {len(df_policy)}, "
                                f"p_profitable_mean={policy_state['last_p_profitable_mean']:.4f}, "
                                f"p>={min_prob_profitable}: {policy_state['last_p_profitable_gt_thresh']:.2%})"
                            )
                    else:
                        # This should not happen - policy should always compute p_profitable
                        log.warning("[ENTRY] FARM_V2: Policy did not compute p_profitable for input candidates")
                        policy_state["last_p_profitable_mean"] = 0.0
                        policy_state["last_p_profitable_gt_thresh"] = 0.0
                        policy_state["last_batch_size"] = int(len(df_policy)) if len(df_policy) > 0 else 0
                    
                    # Store p_long and p_profitable for trade creation (if signal passed)
                    if len(df_policy) > 0 and df_policy["entry_v9_policy_farm_v2"].sum() > 0:
                        # Signal passed - extract values from accepted row
                        accepted_row = df_policy[df_policy["entry_v9_policy_farm_v2"]].iloc[0]
                        policy_state["p_long"] = accepted_row.get("p_long", v9_pred.prob_long)
                        policy_state["p_profitable"] = accepted_row.get("p_profitable", None)
                    else:
                        # Signal rejected - still store p_long for logging, but p_profitable is None
                        policy_state["p_long"] = df_policy["p_long"].iloc[0] if len(df_policy) > 0 and "p_long" in df_policy.columns else v9_pred.prob_long
                        policy_state["p_profitable"] = None
                    
                    # Sanity logging: Track p_profitable distribution (every N candidates)
                    if not hasattr(self, "_farm_v2_stats"):
                        self._farm_v2_stats = {
                            "n_batches": 0,
                            "n_candidates_total": 0,
                            "n_accepted": 0,
                            "p_profitable_values": [],
                            "p_long_values": [],
                        }
                    
                    # Update stats from policy_state (computed before policy filtering)
                    self._farm_v2_stats["n_batches"] += 1
                    batch_size = policy_state.get("last_batch_size", 0)
                    self._farm_v2_stats["n_candidates_total"] += batch_size
                    
                    if batch_size > 0:
                        p_prof_mean = policy_state.get("last_p_profitable_mean", 0.0)
                        p_prof_gt_thresh = policy_state.get("last_p_profitable_gt_thresh", 0.0)
                        self._farm_v2_stats["p_profitable_values"].append(p_prof_mean)
                    
                    # Track accepted signals
                    if len(df_policy) > 0 and df_policy[policy_flag_col].sum() > 0:
                        self._farm_v2_stats["n_accepted"] += 1
                        if policy_state.get("p_profitable") is not None:
                            self._farm_v2_stats["p_profitable_values"].append(float(policy_state["p_profitable"]))
                        if policy_state.get("p_long") is not None:
                            self._farm_v2_stats["p_long_values"].append(float(policy_state["p_long"]))
                    
                    # Log every 50 batches
                    if self._farm_v2_stats["n_batches"] % 50 == 0:
                        n_batches = self._farm_v2_stats["n_batches"]
                        n_candidates_total = self._farm_v2_stats["n_candidates_total"]
                        n_accepted = self._farm_v2_stats["n_accepted"]
                        coverage = n_accepted / max(n_candidates_total, 1)
                        if len(self._farm_v2_stats["p_profitable_values"]) > 0:
                            mean_p_profitable = np.mean(self._farm_v2_stats["p_profitable_values"])
                            pct_above_threshold = sum(1 for p in self._farm_v2_stats["p_profitable_values"] if p >= 0.50) / len(self._farm_v2_stats["p_profitable_values"]) * 100
                            log.info(
                                "[FARM_V2_STATS] After %d batches (%d total candidates): "
                                "Accepted: %d (coverage: %.2f%%), "
                                "Mean p_profitable: %.4f, %% > 0.50: %.1f%%",
                                n_batches, n_candidates_total, n_accepted, coverage * 100, mean_p_profitable, pct_above_threshold
                            )
                
                # BRUTAL GUARD: Apply again AFTER policy (double-check)
                if len(df_policy) > 0 and df_policy[policy_flag_col].iloc[0]:
                    try:
                        farm_brutal_guard(df_policy.iloc[0], context="live_runner_post_policy_v2")
                    except AssertionError as e:
                        raise RuntimeError(
                            f"[FARM_V2_ASSERT] Policy passed but brutal guard failed: {e}. "
                            f"This should never happen."
                        )
            elif use_farm_v1:
                from gx1.policy.entry_v9_policy_farm_v1 import apply_entry_v9_policy_farm_v1  # type: ignore[reportMissingImports]
                from gx1.policy.farm_guards import farm_brutal_guard, get_farm_entry_metadata
                
                # BRUTAL GUARD: Apply centralized guard BEFORE policy
                # This is the SINGLE SOURCE OF TRUTH for FARM entry validation
                try:
                    farm_brutal_guard(current_row.iloc[0], context="live_runner_pre_policy")
                except AssertionError as e:
                    # Guard failed - reject entry immediately
                    log.debug(f"[ENTRY] FARM_V1 entry rejected by brutal guard: {e}")
                    return None
                
                # Now apply policy (which will also apply brutal guard internally)
                df_policy = apply_entry_v9_policy_farm_v1(current_row, self.policy)
                policy_flag_col = "entry_v9_policy_farm_v1"
                
                # BRUTAL GUARD: Apply again AFTER policy (double-check)
                if len(df_policy) > 0 and df_policy[policy_flag_col].iloc[0]:
                    try:
                        farm_brutal_guard(df_policy.iloc[0], context="live_runner_post_policy")
                    except AssertionError as e:
                        # Guard failed after policy - this should never happen
                        raise RuntimeError(
                            f"[FARM_V1_ASSERT] Policy passed but brutal guard failed: {e}. "
                            f"This should never happen."
                        )
            elif use_base_v1:
                from gx1.policy.entry_v9_policy_base_v1 import apply_entry_v9_policy_base_v1  # type: ignore[reportMissingImports]
                df_policy = apply_entry_v9_policy_base_v1(current_row, self.policy)
                policy_flag_col = "entry_v9_policy_base_v1"
            else:
                from gx1.policy.entry_v9_policy_v1 import apply_entry_v9_policy_v1  # type: ignore[reportMissingImports]
                df_policy = apply_entry_v9_policy_v1(current_row, self.policy)
                policy_flag_col = "entry_v9_policy_v1"
            
            # Check if policy approves this signal
            if len(df_policy) > 0 and df_policy[policy_flag_col].iloc[0]:
                prediction = v9_pred
                policy_state["entry_model_active"] = "ENTRY_V9"
                if use_sniper:
                    policy_name = "V9_SNIPER"
                elif use_farm_v2b:
                    policy_name = "V9_FARM_V2B"
                elif use_farm_v2:
                    policy_name = "V9_FARM_V2"
                elif use_farm_v1:
                    policy_name = "V9_FARM_V1"
                elif use_base_v1:
                    policy_name = "POLICY_BASE_V1"
                else:
                    policy_name = "POLICY_V1_V9"
                # DEL 1: Use generic message in replay-mode (no V9 references)
                log_prefix = "[ENTRY_V10_CTX]" if (hasattr(self._runner, "replay_mode") and self._runner.replay_mode) else "[ENTRY]"
                pred_type = "ENTRY_V10_CTX" if (hasattr(self._runner, "replay_mode") and self._runner.replay_mode) else "ENTRY_V9"
                log.debug(f"{log_prefix} Using {pred_type} prediction (%s approved)", policy_name)
                # Store policy_name and current_row for later use in trade creation
                policy_state["policy_name"] = policy_name
                policy_state["_current_row"] = current_row
            else:
                # Policy rejected - no entry
                if use_sniper:
                    policy_name = "POLICY_SNIPER"
                elif use_farm_v2:
                    policy_name = "POLICY_FARM_V2"
                elif use_farm_v1:
                    policy_name = "POLICY_FARM_V1"
                elif use_base_v1:
                    policy_name = "POLICY_BASE_V1"
                else:
                    policy_name = "POLICY_V1_V9"
                # DEL 1: Use generic message in replay-mode (no V9 references)
                log_prefix = "[ENTRY_V10_CTX]" if (hasattr(self._runner, "replay_mode") and self._runner.replay_mode) else "[ENTRY]"
                pred_type = "ENTRY_V10_CTX" if (hasattr(self._runner, "replay_mode") and self._runner.replay_mode) else "ENTRY_V9"
                log.debug(f"{log_prefix} {pred_type} prediction rejected by %s (prob_long=%.4f)", policy_name, v9_pred.prob_long)
                
                # SNIPER logging: policy rejected (non-SNIPER policies also handled above)
                if use_sniper:
                    p_long_val = v9_pred.prob_long if v9_pred else None
                    self._log_sniper_cycle(
                        ts=current_ts,
                        session=current_session,
                        in_scope=sniper_in_scope,
                        warmup_ready=sniper_warmup_ready,
                        degraded=sniper_degraded,
                        spread_bps=current_spread_bps,
                        atr_bps=current_atr_bps,
                        eval_ran=True,
                        reason="policy_rejected",
                        decision="NO-TRADE",
                        n_signals=0,
                        p_long=p_long_val,
                    )
                
                # DEL 1: Hook PolicyDecisionCollector - collect "skip" decision
                if hasattr(self._runner, "replay_eval_collectors") and self._runner.replay_eval_collectors:
                    decision_collector = self._runner.replay_eval_collectors.get("policy_decisions")
                    if decision_collector:
                        reasons = ["policy_rejected"]
                        decision_collector.collect(
                            timestamp=current_ts,
                            decision="skip",
                            reasons=reasons,
                        )
                
                return None  # No entry this bar
        else:
            # Policy disabled - use V9 prediction directly (RAW_ENTRY_V9 mode)
            prediction = v9_pred
            policy_state["entry_model_active"] = "ENTRY_V9"
            # DEL 1: Use generic message in replay-mode (no V9 references)
            log_prefix = "[ENTRY_V10_CTX]" if (hasattr(self._runner, "replay_mode") and self._runner.replay_mode) else "[ENTRY]"
            pred_type = "ENTRY_V10_CTX" if (hasattr(self._runner, "replay_mode") and self._runner.replay_mode) else "ENTRY_V9"
            log.debug(f"{log_prefix} Using {pred_type} prediction (POLICY_V1_V9 disabled - RAW mode)")
        
        # Store policy_state for logging
        self._last_policy_state = policy_state
        
        # V9 is complete - skip all other entry logic
        hybrid_entry_enabled = False
        xgb_prediction = prediction  # Use V9 prediction as xgb_prediction for compatibility
        tcn_prediction = None
        use_v9 = True  # V9 is always active
        hybrid_entry_data = None  # No hybrid blending with V9
        
        # V9 is the ONLY entry model - all V6/V8 code removed
        # Continue directly to gating section with V9 prediction
        prediction = xgb_prediction
        
        # ============================================================
        # Adaptive Thresholding (V9-only - no hybrid conflict calculation)
        # ============================================================
        # Adjust coverage cutoff based on regime (trend + vol from Big Brain V1)
        # V9 uses policy-based filtering, so cutoff adjustment is minimal
        
        # V9 is the ONLY entry model - no hybrid conflict calculation
        # Explicitly set conflict variables to 0.0 for V9 (not a bug, by design)
        model_version = policy_state.get("entry_model_active", "ENTRY_V9")
        if model_version.startswith("ENTRY_V9") or model_version == "ENTRY_V9":
            conflict = 0.0  # V9-only: no conflict calculation (no hybrid blending)
            conflict_norm = 0.0  # V9-only: no conflict normalization
            cutoff_adj_conf = 0.0  # V9-only: no conflict-based cutoff adjustment
        else:
            # Legacy V6/V8 hybrid logic (should not be reached in V9-only mode)
            # This branch exists for code clarity and future-proofing
            log.warning(
                "[ADAPTIVE_THRESHOLD] Unexpected model version: %s (expected ENTRY_V9)",
                model_version
            )
            conflict = 0.0
            conflict_norm = 0.0
            cutoff_adj_conf = 0.0
        
        regime_adj = 0.0
        trend = policy_state.get("brain_trend_regime", "UNKNOWN")
        vol = policy_state.get("brain_vol_regime", "UNKNOWN")
        
        # Regime-based cutoff adjustment (minimal for V9 - policy handles filtering)
        if trend == "TREND_UP" and vol in ("LOW", "MID"):
            regime_adj = -0.01
        elif trend == "TREND_UP" and vol == "HIGH":
            regime_adj = +0.015
        elif trend == "MR":
            regime_adj = 0.0
        elif trend == "TREND_DOWN" and vol == "HIGH":
            regime_adj = +0.02
        else:
            regime_adj = +0.005
        
        # Get base_cutoff from entry_params or policy
        base_cutoff = self.entry_params.get("coverage_cutoff", 0.55)  # Default threshold
        
        # Coverage cutoff (V9 policy handles most filtering)
        coverage_cutoff_new = base_cutoff + regime_adj
        
        # Clamp to ±3% for stability
        coverage_cutoff_new = max(
            min(coverage_cutoff_new, base_cutoff + 0.03),
            base_cutoff - 0.03
        )
        
        policy_state["coverage_cutoff"] = coverage_cutoff_new
        policy_state["coverage_cutoff_adj"] = coverage_cutoff_new - base_cutoff
        
        # D) Update margin_min based on new cutoff
        # Higher cutoff → stricter margin_min
        # Lower cutoff → softer margin_min
        delta = coverage_cutoff_new - base_cutoff
        
        # Get base margin_min from entry_gating or entry_params
        margin_min_cfg = entry_gating.get("margin_min", {}) if entry_gating else {}
        margin_min_long_base = float(margin_min_cfg.get("long", 0.08)) if margin_min_cfg else 0.08
        margin_min_short_base = float(margin_min_cfg.get("short", 0.10)) if margin_min_cfg else 0.10
        
        # Adjust margin_min proportionally to cutoff change
        margin_min_long_new = margin_min_long_base + 0.5 * delta
        margin_min_short_new = margin_min_short_base + 0.5 * delta
        
        # Clamp margin_min to valid range [0.02, 0.15]
        margin_min_long_new = max(0.02, min(0.15, margin_min_long_new))
        margin_min_short_new = max(0.02, min(0.15, margin_min_short_new))
        
        # Update entry_gating with new margin_min values
        if entry_gating:
            if "margin_min" not in entry_gating:
                entry_gating["margin_min"] = {}
            entry_gating["margin_min"]["long"] = margin_min_long_new
            entry_gating["margin_min"]["short"] = margin_min_short_new
        
        # Store adaptive thresholding data in policy_state for logging
        policy_state["adaptive_thresholding"] = {
            "conflict": float(conflict),
            "conflict_norm": float(conflict_norm) if hybrid_entry_data is not None else 0.0,
            "regime_adj": float(regime_adj),
            "cutoff_adj_conf": float(cutoff_adj_conf),
            "coverage_cutoff_new": float(coverage_cutoff_new),
            "margin_min_long": float(margin_min_long_new),
            "margin_min_short": float(margin_min_short_new),
            "margin_min_long_base": float(margin_min_long_base),
            "margin_min_short_base": float(margin_min_short_base),
        }
        
        # Log first 10 adaptive thresholding decisions
        if not hasattr(self, "_adaptive_threshold_log_count"):
            self._adaptive_threshold_log_count = 0
        if self._adaptive_threshold_log_count < 10:
            log.info(
                "[ADAPTIVE_THRESHOLD] #%d: conflict=%.4f norm=%.3f regime_adj=%.4f cutoff_adj_conf=%.4f "
                "cutoff=%.3f→%.3f margin_min=%.3f/%.3f→%.3f/%.3f trend=%s vol=%s",
                self._adaptive_threshold_log_count + 1,
                conflict,
                conflict_norm if hybrid_entry_data is not None else 0.0,
                regime_adj,
                cutoff_adj_conf,
                base_cutoff,
                coverage_cutoff_new,
                margin_min_long_base,
                margin_min_short_base,
                margin_min_long_new,
                margin_min_short_new,
                trend,
                vol,
            )
            self._adaptive_threshold_log_count += 1
        
        # Update policy_state for trade creation
        self._last_policy_state = policy_state

        # Get temperature for logging (TEMPORARY TEST: T=1.0 for all sessions)
        # TODO: Remove this after diagnosis - restore: temp_map = self._get_temperature_map(); T = float(temp_map.get(prediction.session, 1.0))
        T = 1.0

        now_ts = candles.index[-1]
        now_ts_utc = now_ts.tz_convert("UTC") if now_ts.tzinfo else pd.Timestamp(now_ts, tz="UTC")
        
        # Get entry_gating config from policy (if available)
        # Note: margin_min will be adjusted by adaptive thresholding below
        # GUARD: Check if guard is enabled - if disabled, skip entry_gating (ghost guard fix)
        guard_cfg = self.policy.get("guard", {})
        guard_enabled = guard_cfg.get("enabled", True)  # Default to True for backward compatibility
        if not guard_enabled:
            # Guard disabled: skip entry_gating to allow all trades through (scan-mode)
            entry_gating = None
            log.debug("[GUARD] Guard disabled - skipping entry_gating (scan-mode)")
        else:
            entry_gating = self.policy.get("entry_gating", None)
        
        # Directional Soft Bias removed - using adaptive thresholding only
        
        # Get last side and bars since last side (for sticky-side logic)
        last_side = self.last_entry_side
        bars_since_last_side = 999
        if self.last_entry_timestamp is not None:
            # Calculate bars since last entry (M5 = 300 seconds per bar)
            bars_since_last_side = int((now_ts - self.last_entry_timestamp).total_seconds() / 300.0)
        
        # PRE-GATES: Determine side before gates (diagnostic)
        if prediction.prob_long >= prediction.prob_short:
            side_pre = "LONG"
            p_side_pre = prediction.prob_long
            p_other_pre = prediction.prob_short
        else:
            side_pre = "SHORT"
            p_side_pre = prediction.prob_short
            p_other_pre = prediction.prob_long
        ratio_pre = p_side_pre / (p_other_pre + 1e-6)
        
        # Soft Score Shaping removed - using blended scores directly (no trend-bias or risk-damping)
        
        # For FARM_V2B with side-aware policy, use policy-determined side if available
        # NOTE: This is FARM-specific, but we track threshold veto for SNIPER paths
        policy_side = policy_state.get("_policy_side", None)
        if policy_side and policy_side in ("long", "short", "both"):
            # Policy has determined which side(s) are allowed
            if policy_side == "long":
                side = "long"
            elif policy_side == "short":
                side = "short"
            elif policy_side == "both":
                # Both thresholds met - use default logic (max prob)
                side = "long" if prediction.prob_long >= prediction.prob_short else "short"
            else:
                side = None
                # TELEMETRY CONTRACT: Policy veto (Stage-1, after prediction) - FARM-specific
                # For SNIPER, this path is not used, so we don't track veto_cand here
        else:
            # Default: use should_enter_trade function
            # should_enter_trade is a module-level function in oanda_demo_runner
            # EntryManager accesses it via __getattr__ from runner
            should_enter_trade_func = getattr(self, "should_enter_trade", None)
            if should_enter_trade_func is None:
                # Fallback: import directly if not available via runner
                from gx1.execution.oanda_demo_runner import should_enter_trade as should_enter_trade_func
            side = should_enter_trade_func(
                prediction, 
                self.entry_params,
                entry_gating=entry_gating,
                last_side=last_side,
                bars_since_last_side=bars_since_last_side,
            )
            
            # TELEMETRY CONTRACT: Track threshold veto (Stage-1, after prediction)
            if side is None:
                self.veto_cand["veto_cand_threshold"] += 1
                # DEPRECATED: veto_counters removed (SNIPER/NY uses veto_cand)
                # Store threshold used for diagnostics
                if entry_gating:
                    if "long" in entry_gating:
                        long_thresh = entry_gating["long"].get("p_side_min", entry_gating["long"].get("threshold", None))
                        self.threshold_used = f"long={long_thresh}" if long_thresh is not None else None
                    elif "short" in entry_gating:
                        short_thresh = entry_gating["short"].get("p_side_min", entry_gating["short"].get("threshold", None))
                        self.threshold_used = f"short={short_thresh}" if short_thresh is not None else None
                    elif "threshold" in entry_gating:
                        self.threshold_used = entry_gating["threshold"]
                return None  # Early return - no need to check other Stage-1 gates
        
        # POST-GATES: side after gates (diagnostic)
        side_post = side.upper() if side else "NO_ENTRY"
        
        # Big Brain V1 Entry Gating: Apply hard gating based on regime
        # SAFETY_V4: Flow: blend-scorer + big_brain_regime → safety_profile (SAFETY_Vx) → gate-decision
        # This runs AFTER XGB/gating but BEFORE momentum-veto/volatility-brake
        # Inputs: prediction (blend-scorer), policy_state (big_brain_regime), session_tag → safety_profile → gate-decision
        if side is not None and hasattr(self, 'big_brain_v1_entry_gater') and self.big_brain_v1_entry_gater is not None:
            brain_trend = policy_state.get("brain_trend_regime", None)
            brain_vol = policy_state.get("brain_vol_regime", None)
            
            # SAFETY_V4: Use original session_tag (from infer_session_tag), not resolved model key
            # Entry gates need the actual session (EU/US/OVERLAP/ASIA), not model routing key
            # Get session from prediction if available, otherwise infer from current timestamp
            if prediction.session:
                session_key = prediction.session.upper()
            else:
                # Infer session from current bar timestamp (last candle timestamp)
                current_ts = candles.index[-1]
                session_key = infer_session_tag(current_ts).upper()
            
            # SAFETY_V4: Ensure session is valid (EU/US/OVERLAP/ASIA), never UNKNOWN/ALL
            if session_key not in ("EU", "US", "OVERLAP", "ASIA"):
                log.warning(
                    "[SAFETY_V4] Invalid session '%s' in entry gating, inferring from timestamp",
                    session_key
                )
                current_ts = candles.index[-1]
                session_key = infer_session_tag(current_ts).upper()
            
            # Check if side is allowed
            allowed, entry_action = self.big_brain_v1_entry_gater.should_allow_side(
                side=side,
                trend_regime=brain_trend,
                vol_regime=brain_vol,
                session=session_key,
            )
            
            if not allowed:
                log.info(
                    "[BIG_BRAIN_V1_ENTRY] BLOCK side=%s trend=%s vol=%s session=%s action=%s",
                    side.upper(),
                    brain_trend or "UNKNOWN",
                    brain_vol or "UNKNOWN",
                    session_key,
                    entry_action.value,
                )
                # DEPRECATED: veto_counters removed (SNIPER/NY uses veto_cand)
                # Track veto (veto_cand_big_brain already incremented above)
                return None  # Block entry
            
            # Log if entry was allowed (debug level)
            log.debug(
                "[BIG_BRAIN_V1_ENTRY] ALLOW side=%s trend=%s vol=%s session=%s action=%s",
                side.upper(),
                brain_trend or "UNKNOWN",
                brain_vol or "UNKNOWN",
                session_key,
                entry_action.value,
            )
        
        # Extract momentum/volatility features for diagnostic logging (BEFORE creating eval_record)
        gates_cfg = self.policy.get("gates", {})
        momentum_veto = gates_cfg.get("momentum_veto", {})
        volatility_brake = gates_cfg.get("volatility_brake", {})
        
        feat_row = entry_bundle.features.iloc[-1]
        r5 = 0.0
        r8 = 0.0
        atr_z = 0.0
        ema5_pct = 0.0
        ema20_pct = 0.0
        
        # Get r5 and r8 (momentum features)
        if "_v1_r5" in entry_bundle.features.columns:
            r5 = float(feat_row["_v1_r5"]) if not pd.isna(feat_row["_v1_r5"]) else 0.0
        elif "r5" in entry_bundle.features.columns:
            r5 = float(feat_row["r5"]) if not pd.isna(feat_row["r5"]) else 0.0
        
        if "_v1_r8" in entry_bundle.features.columns:
            r8 = float(feat_row["_v1_r8"]) if not pd.isna(feat_row["_v1_r8"]) else 0.0
        elif "r8" in entry_bundle.features.columns:
            r8 = float(feat_row["r8"]) if not pd.isna(feat_row["r8"]) else 0.0
        
        # Get ATR proxy (volatility feature) for diagnostics using ENTRY_V3 features
        if "_v1_atr_regime_id" in entry_bundle.features.columns:
            atr_regime = feat_row.get("_v1_atr_regime_id")
            if not pd.isna(atr_regime):
                # Map discrete regime id to a coarse "z-like" scale for logging only
                try:
                    regime_val = int(atr_regime)
                except (TypeError, ValueError):
                    regime_val = 1
                if regime_val <= 0:
                    atr_z = 0.5
                elif regime_val == 1:
                    atr_z = 1.5
                else:
                    atr_z = 3.0
        elif "_v1_atr14" in entry_bundle.features.columns:
            atr14 = feat_row.get("_v1_atr14")
            if atr14 is not None and not pd.isna(atr14):
                atr_z = float(atr14)
        
        # Get EMA5 and EMA20 (if available) - approximate from close price
        # Note: These are not in the feature manifest, but we can compute them from candles
        if len(candles) >= 20:
            close_series = candles["close"]
            ema5 = float(close_series.ewm(span=5, adjust=False).mean().iloc[-1])
            ema20 = float(close_series.ewm(span=20, adjust=False).mean().iloc[-1])
            # Normalize by current close price
            ema5_pct = ((ema5 - entry_bundle.close_price) / entry_bundle.close_price) * 100.0
            ema20_pct = ((ema20 - entry_bundle.close_price) / entry_bundle.close_price) * 100.0
        
        # Prepare eval record for JSON logging (with all diagnostic fields)
        eval_record = {
            "ts_utc": now_ts_utc.isoformat(),
            "session": prediction.session,
            "p_long": float(prediction.prob_long),
            "p_short": float(prediction.prob_short),
            "p_hat": float(prediction.p_hat),
            "margin": float(prediction.margin),
            "T": float(T),
            "side_pre": side_pre,  # PRE-GATES: side before gates
            "ratio_pre": float(ratio_pre),  # PRE-GATES: p_side/p_other ratio
            "decision": side_post,  # POST-GATES: side after gates
            "r5": float(r5),
            "r8": float(r8),
            "atr_z": float(atr_z),
            "ema5_pct": float(ema5_pct),
            "ema20_pct": float(ema20_pct),
            "ratio": float(ratio_pre),  # Same as ratio_pre (for backward compatibility)
            "price": float(entry_bundle.close_price),
            "units": int(self.exec.default_units if side == "long" else (-self.exec.default_units if side == "short" else 0)),
        }
        
        # Log eval record to JSON file
        # Lazy import to avoid circular dependency
        from gx1.execution.oanda_demo_runner import append_eval_log
        append_eval_log(self.eval_log_path, eval_record)
        
        # Log eval summary (diagnostic format)
        log.info(
            "[DBG] side_pre=%s p_long_adj=%.4f p_short_adj=%.4f ratio=%.2f r5=%.4f r8=%.4f ema5=%.2f%% ema20=%.2f%% atr_z=%.2f session=%s decision=%s",
            side_pre,
            prediction.prob_long,
            prediction.prob_short,
            ratio_pre,
            r5,
            r8,
            ema5_pct,
            ema20_pct,
            atr_z,
            prediction.session,
            side_post,
        )
        
        # Record eval to telemetry tracker (if available)
        if hasattr(self, "telemetry_tracker") and self.telemetry_tracker is not None:
            try:
                self.telemetry_tracker.record_eval(
                    ts_utc=now_ts_utc,
                    session=prediction.session,
                    p_hat=prediction.p_hat,
                    margin=prediction.margin,
                    decision=side.upper() if side else "NO_ENTRY",
                )
            except Exception as e:
                log.debug("Failed to record eval in telemetry: %s", e)
        
        # Parity-run: Compare live vs offline predictions (for reprod-bevis)
        # Note: We use same models as live, but call them separately to verify parity
        if hasattr(self, "parity_enabled") and self.parity_enabled and not getattr(self, "_parity_disabled", False) and hasattr(self, "entry_model_bundle") and self.entry_model_bundle.models:
            self.parity_sample_counter += 1
            if self.parity_sample_counter % self.parity_sample_every_n == 0:
                try:
                    # Get feature row for offline prediction (use same entry_bundle as live)
                    feat_df = entry_bundle.features
                    feature_cols = self.entry_model_bundle.feature_names
                    
                    # Align features to manifest (same as live)
                    aligned = feat_df.reindex(columns=feature_cols, fill_value=0.0)
                    feat_row = aligned.iloc[-1]
                    
                    # Temperature map for parity (uses same configuration as live)
                    temp_map = self._get_temperature_map()
                    
                    # Call offline prediction (using same models as live)
                    # Lazy import to avoid circular dependency
                    from gx1.execution.oanda_demo_runner import mirror_offline_predict
                    p_hat_off, margin_off, dir_off = mirror_offline_predict(
                        feat_row=feat_row,
                        session_tag=prediction.session,
                        models=self.entry_model_bundle.models,
                        feature_cols=feature_cols,
                        temperature_map=temp_map,
                        manifest=self.manifest,
                    )
                    
                    # Get live predictions
                    p_hat_live = prediction.p_hat
                    margin_live = prediction.margin
                    dir_live = side.upper() if side else "NO_ENTRY"
                    
                    # Calculate absolute error
                    abs_err_p_hat = abs(p_hat_live - p_hat_off)
                    abs_err_margin = abs(margin_live - margin_off)
                    
                    # Check direction match
                    dir_match = (dir_live == dir_off) or (dir_live == "NO_ENTRY" and dir_off == "NO_ENTRY")
                    
                    # Log parity record
                    parity_record = {
                        "ts_utc": now_ts_utc.isoformat(),
                        "session": prediction.session,
                        "p_hat_live": float(p_hat_live),
                        "p_hat_off": float(p_hat_off),
                        "margin_live": float(margin_live),
                        "margin_off": float(margin_off),
                        "dir_live": dir_live,
                        "dir_off": dir_off,
                        "abs_err_p_hat": float(abs_err_p_hat),
                        "abs_err_margin": float(abs_err_margin),
                        "dir_match": dir_match,
                    }
                    
                    # Lazy import to avoid circular dependency
                    from gx1.execution.oanda_demo_runner import append_eval_log
                    append_eval_log(self.parity_log_path, parity_record)
                    
                    # Track parity metrics per session
                    session_key = prediction.session
                    self.parity_metrics[session_key].append(abs_err_p_hat)
                    self.parity_direction_matches[session_key].append(dir_match)
                    
                    # Keep only last 1000 samples per session
                    if len(self.parity_metrics[session_key]) > 1000:
                        self.parity_metrics[session_key] = self.parity_metrics[session_key][-1000:]
                        self.parity_direction_matches[session_key] = self.parity_direction_matches[session_key][-1000:]
                    
                    # Calculate p50 and p99
                    if len(self.parity_metrics[session_key]) >= 10:
                        abs_errs = np.array(self.parity_metrics[session_key])
                        p50_abs_err = float(np.percentile(abs_errs, 50))
                        p99_abs_err = float(np.percentile(abs_errs, 99))
                        
                        # Direction match rate
                        dir_match_rate = float(np.mean(self.parity_direction_matches[session_key]))
                        
                        # Check tolerances (always compute/log, but guard behaviour depends on mode)
                        if p50_abs_err > self.parity_tolerance_p50:
                            log.warning(
                                "[PARITY %s] P50 abs_err > tolerance: p50=%.6f, tolerance=%.6f",
                                session_key,
                                p50_abs_err,
                                self.parity_tolerance_p50,
                            )
                        
                        if p99_abs_err > self.parity_tolerance_p99:
                            log.warning(
                                "[PARITY %s] P99 abs_err > tolerance: p99=%.6f, tolerance=%.6f",
                                session_key,
                                p99_abs_err,
                                self.parity_tolerance_p99,
                            )
                        
                        # In replay/fast-replay: parity is metrics-only (no guard / kill-switch)
                        if self.is_replay:
                            if p50_abs_err > self.parity_tolerance_p50 or p99_abs_err > self.parity_tolerance_p99:
                                log.debug(
                                    "[PARITY %s] Drift above tolerance in replay mode – metrics only, no guard/kill-switch "
                                    "(p50=%.6f, p99=%.6f, tol_p50=%.6e, tol_p99=%.6e)",
                                    session_key,
                                    p50_abs_err,
                                    p99_abs_err,
                                    self.parity_tolerance_p50,
                                    self.parity_tolerance_p99,
                                )
                        else:
                            # Live/demo: parity drift is a guard signal (logged here; actual blocking happens below)
                            if p50_abs_err > self.parity_tolerance_p50 or p99_abs_err > self.parity_tolerance_p99:
                                log.error(
                                    "[GUARD %s] Parity drift — blocking orders: p50=%.6f, p99=%.6f",
                                    session_key,
                                    p50_abs_err,
                                    p99_abs_err,
                                )
                        
                        # Log parity metrics (every 100 samples)
                        if len(self.parity_metrics[session_key]) % 100 == 0:
                            log.info(
                                "[PARITY %s] n=%d, dir_match_rate=%.3f, p50_abs_err=%.6f, p99_abs_err=%.6f",
                                session_key,
                                len(self.parity_metrics[session_key]),
                                dir_match_rate,
                                p50_abs_err,
                                p99_abs_err,
                            )
                
                except NameError as e:
                    log.warning("[PARITY] Disabling parity due to config error: %r", e)
                    self._parity_disabled = True
                except Exception as e:
                    log.warning("Parity-run failed: %s", e, exc_info=True)
        
        # Kill-switch checks (before entry decision)
        # If any kill-switch is triggered, block orders and log explicitly (not DRY-RUN)
        session_key = self._resolve_session_key(prediction.session)
        guard_blocked = False
        block_reason = None
        
        # Kill-switch 1: ECE > 0.18
        ece = None
        if hasattr(self, "telemetry_tracker") and self.telemetry_tracker is not None:
            ece = self.telemetry_tracker.get_ece(session_key)
        if ece is not None and ece > 0.18:
            if not self.is_replay:
                guard_blocked = True
                block_reason = f"ECE > 0.18: ece={ece:.4f}"
                log.error(
                    "[KILL-SWITCH %s] %s. Trading disabled.",
                    session_key,
                    block_reason,
                )
                log.warning("[GUARD] BLOCKED ORDER (live_mode=%s) reason=%s", not self.exec.dry_run, block_reason)
                # DEPRECATED: veto_counters removed (SNIPER/NY uses veto_pre)
                # Track veto (veto_pre_killswitch already incremented above)
                # Block orders if kill-switch is triggered (even in live mode)
                return None
            else:
                # Replay/fast-replay: ECE is metrics-only (no guard/kill-switch)
                log.error(
                    "[KILL-SWITCH %s] ECE > 0.18: ece=%.4f (replay mode – NOT blocking trades)",
                    session_key,
                    ece,
                )
        
        # Kill-switch 2: Parity p99 > tolerance
        if not self.is_replay and self.parity_enabled and len(self.parity_metrics.get(session_key, [])) >= 10:
            parity_errs = np.array(self.parity_metrics[session_key])
            p99_abs_err = float(np.percentile(parity_errs, 99))
            if p99_abs_err > self.parity_tolerance_p99:
                guard_blocked = True
                block_reason = f"Parity p99 > tolerance: p99={p99_abs_err:.6f}, tolerance={self.parity_tolerance_p99:.6f}"
                log.error(
                    "[KILL-SWITCH %s] %s. Trading disabled.",
                    session_key,
                    block_reason,
                )
                log.warning("[GUARD] BLOCKED ORDER (live_mode=%s) reason=%s", not self.exec.dry_run, block_reason)
                # Block orders if kill-switch is triggered (even in live mode)
                return None
        
        # Kill-switch 3: Coverage avvik >50% (only check if we have enough evals)
        coverage = None
        if hasattr(self, "telemetry_tracker") and self.telemetry_tracker is not None:
            coverage = self.telemetry_tracker.get_coverage(session_key)
        if coverage is not None:
            # Only check coverage kill-switch if we have enough evals (at least 30 evals = 30 minutes)
            eval_buffer = self.telemetry_tracker.eval_buffers.get(session_key, deque())
            if len(eval_buffer) >= 30:  # At least 30 evals before checking coverage kill-switch
                coverage_diff = abs(coverage - self.telemetry_tracker.target_coverage)
                coverage_threshold = 0.50 * self.telemetry_tracker.target_coverage  # 50% avvik
                if coverage_diff > coverage_threshold:
                    if not self.is_replay:
                        guard_blocked = True
                        block_reason = f"Coverage avvik >50%: coverage={coverage:.3f}, target={self.telemetry_tracker.target_coverage:.3f}, diff={coverage_diff:.3f}"
                        log.error(
                            "[KILL-SWITCH %s] %s. Trading disabled.",
                            session_key,
                            block_reason,
                        )
                        log.warning("[GUARD] BLOCKED ORDER (live_mode=%s) reason=%s", not self.exec.dry_run, block_reason)
                        # Block orders if kill-switch is triggered (even in live mode)
                        return None
                    else:
                        # Replay/fast-replay: coverage is metrics-only (no guard/kill-switch)
                        log.error(
                            "[KILL-SWITCH %s] Coverage avvik >50%%: coverage=%.3f, target=%.3f, diff=%.3f "
                            "(replay mode – NOT blocking trades)",
                            session_key,
                            coverage,
                            self.telemetry_tracker.target_coverage,
                            coverage_diff,
                        )
            else:
                # Not enough evals yet, skip coverage kill-switch check
                log.debug(
                    "[GUARD %s] Coverage kill-switch skipped: n_evals=%d < 30 (insufficient data)",
                    session_key,
                    len(eval_buffer),
                )
        
        if side is None:
            log.info(
                "[GUARD] NO ENTRY — session=%s p_hat=%.4f margin=%.4f p_long=%.4f p_short=%.4f T=%.2f",
                prediction.session,
                prediction.p_hat,
                prediction.margin,
                prediction.prob_long,
                prediction.prob_short,
                T,
            )
            return None
        
        # Big Brain V0 removed - using V1 entry gating only (already applied above)
        
        # Apply momentum-veto and volatility-brake gates (before order placement)
        gates_cfg = self.policy.get("gates", {})
        
        # Momentum-veto: block trades against momentum
        momentum_veto = gates_cfg.get("momentum_veto", {})
        if momentum_veto.get("enabled", False):
            # Get r5 from features (TEMPORARY: use only r5 when require_both=false)
            feat_row = entry_bundle.features.iloc[-1]
            # Check if column exists first
            if "_v1_r5" in entry_bundle.features.columns:
                r5 = float(feat_row["_v1_r5"]) if not pd.isna(feat_row["_v1_r5"]) else 0.0
            elif "r5" in entry_bundle.features.columns:
                r5 = float(feat_row["r5"]) if not pd.isna(feat_row["r5"]) else 0.0
            else:
                r5 = 0.0
                log.warning("[DEBUG] _v1_r5 and r5 not found in features columns (available: %s)", list(entry_bundle.features.columns)[:10])
            
            # Get r8 only if require_both=true (TEMPORARY: skip r8 when require_both=false)
            require_both = momentum_veto.get("require_both", True)
            r8 = 0.0
            if require_both:
                # Use _v1_r8 as proxy for r10 (r10 doesn't exist in features, but r8 is close)
                if "_v1_r8" in entry_bundle.features.columns:
                    r8 = float(feat_row["_v1_r8"]) if not pd.isna(feat_row["_v1_r8"]) else 0.0
                elif "r8" in entry_bundle.features.columns:
                    r8 = float(feat_row["r8"]) if not pd.isna(feat_row["r8"]) else 0.0
                else:
                    r8 = 0.0
                    log.warning("[DEBUG] _v1_r8 and r8 not found in features columns (available: %s)", list(entry_bundle.features.columns)[:10])
            
            short_veto = momentum_veto.get("short_veto", {})
            long_veto = momentum_veto.get("long_veto", {})
            
            if side == "short":
                r5_min = float(short_veto.get("r5_min", -0.00))
                r10_min = float(short_veto.get("r10_min", -0.00))  # Keep r10_min for policy compatibility
                
                if require_both:
                    # Require BOTH r5<=r5_min AND r8<=r10_min (using r8 as proxy for r10)
                    # Block SHORT if r5 > r5_min OR r8 > r10_min (OR logic: block if ANY is > 0)
                    # This means we block if r5 > 0 OR r8 > 0 (market pointing up)
                    if r5 > r5_min or r8 > r10_min:
                        log.info(
                            "[GUARD] momentum_veto: block SHORT (r5=%.4f r8=%.4f, required: r5<=%.2f AND r8<=%.2f)",
                            r5, r8, r5_min, r10_min
                        )
                        return None
                    else:
                        # Log when SHORT passes momentum_veto (for debugging)
                        log.debug(
                            "[GUARD] momentum_veto: SHORT passed (r5=%.4f<=%.2f AND r8=%.4f<=%.2f)",
                            r5, r5_min, r8, r10_min
                        )
                else:
                    # TEMPORARY: Use only r5 (not r8) for faster diagnosis
                    # Block SHORT if r5 > r5_min (simpler, faster)
                    if r5 > r5_min:
                        log.info(
                            "[GUARD] momentum_veto: block SHORT (r5=%.4f > %.2f, require_both=false)",
                            r5, r5_min
                        )
                        return None
                    else:
                        # Log when SHORT passes momentum_veto (for debugging)
                        log.debug(
                            "[GUARD] momentum_veto: SHORT passed (r5=%.4f <= %.2f, require_both=false)",
                            r5, r5_min
                        )
            elif side == "long":
                r5_max = float(long_veto.get("r5_max", 0.00))
                r10_max = float(long_veto.get("r10_max", 0.00))  # Keep r10_max for policy compatibility
                
                if require_both:
                    # Require BOTH r5>=r5_max AND r8>=r10_max (using r8 as proxy for r10)
                    # Block LONG if r5 < r5_max OR r8 < r10_max (OR logic: block if ANY is < 0)
                    # This means we block if r5 < 0 OR r8 < 0 (market pointing down)
                    if r5 < r5_max or r8 < r10_max:
                        log.info(
                            "[GUARD] momentum_veto: block LONG (r5=%.4f r8=%.4f, required: r5>=%.2f AND r8>=%.2f)",
                            r5, r8, r5_max, r10_max
                        )
                        return None
                    else:
                        # Log when LONG passes momentum_veto (for debugging)
                        log.debug(
                            "[GUARD] momentum_veto: LONG passed (r5=%.4f>=%.2f AND r8=%.4f>=%.2f)",
                            r5, r5_max, r8, r10_max
                        )
                else:
                    # TEMPORARY: Use only r5 (not r8) for faster diagnosis
                    # Block LONG if r5 < r5_max (simpler, faster)
                    if r5 < r5_max:
                        log.info(
                            "[GUARD] momentum_veto: block LONG (r5=%.4f < %.2f, require_both=false)",
                            r5, r5_max
                        )
                        # KILL-CHAIN: regime-style block (momentum veto)
                        self._killchain_inc_reason("BLOCK_REGIME")
                        return None
                    else:
                        # Log when LONG passes momentum_veto (for debugging)
                        log.debug(
                            "[GUARD] momentum_veto: LONG passed (r5=%.4f >= %.2f, require_both=false)",
                            r5, r5_max
                        )
        
        # Volatility-brake: block entries in spike-regime
        volatility_brake = gates_cfg.get("volatility_brake", {})
        if volatility_brake.get("enabled", False):
            feat_row = entry_bundle.features.iloc[-1]
            # Primary ATR input for volatility_brake is still atr_z (for backward compatibility)
            if "_v1_atr_z_10_100" in entry_bundle.features.columns:
                atr_z = float(feat_row["_v1_atr_z_10_100"]) if not pd.isna(feat_row["_v1_atr_z_10_100"]) else 0.0
            elif "atr_z" in entry_bundle.features.columns:
                atr_z = float(feat_row["atr_z"]) if not pd.isna(feat_row["atr_z"]) else 0.0
            else:
                # For ENTRY_V3 we expect _v1_atr_regime_id / _v1_atr14. Use them only for diagnostics,
                # keep existing atr_z gating semantics unchanged.
                atr_z = 0.0
                atr_regime = feat_row.get("_v1_atr_regime_id") if "_v1_atr_regime_id" in entry_bundle.features.columns else None
                atr14 = feat_row.get("_v1_atr14") if "_v1_atr14" in entry_bundle.features.columns else None
                if atr_regime is not None and not pd.isna(atr_regime):
                    log.debug(
                        "[DEBUG] volatility_brake: using _v1_atr_regime_id=%s as ATR proxy (no atr_z column present)",
                        atr_regime,
                    )
                elif atr14 is not None and not pd.isna(atr14):
                    log.debug(
                        "[DEBUG] volatility_brake: using _v1_atr14=%.4f as ATR proxy (no atr_z column present)",
                        float(atr14),
                    )
                else:
                    log.debug(
                        "[DEBUG] No ATR feature available in row (no _v1_atr_z_10_100 / atr_z / _v1_atr_regime_id / _v1_atr14)",
                    )
            atr_z_max = float(volatility_brake.get("atr_z_max", 2.5))
            if atr_z > atr_z_max:
                log.info(
                    "[GUARD] volatility_brake: atr_z=%.2f > %.2f (spike-regime)",
                    atr_z, atr_z_max
                )
                # KILL-CHAIN: volatility-style block
                self._killchain_inc_reason("BLOCK_VOL")
                return None
            else:
                # Log when volatility_brake passes (for debugging)
                log.debug(
                    "[GUARD] volatility_brake: passed (atr_z=%.2f <= %.2f)",
                    atr_z, atr_z_max
                )
        
            # Portfolio protection: max concurrent positions
            # Use self.max_concurrent_positions (set in __init__ from policy)
            open_count = len(self.open_trades)
            if open_count >= self.max_concurrent_positions:
                log.debug(
                    "[GUARD] max_concurrent_positions: open_count=%d >= %d",
                    open_count, self.max_concurrent_positions
                )
                # DEPRECATED: veto_counters removed (SNIPER/NY uses veto_cand)
                # Track veto (veto_cand_max_trades already incremented above)
                # KILL-CHAIN: position limit block
                self._killchain_inc_reason("BLOCK_POSITION_LIMIT")
                return None
        
            # Portfolio protection: intraday drawdown limit
            risk_cfg = self.policy.get("risk", {})
            intraday_drawdown_bps_limit = float(risk_cfg.get("intraday_drawdown_bps_limit", 120))
            if intraday_drawdown_bps_limit > 0:
                # Calculate unrealized portfolio PnL in bps
                if entry_bundle.bid_close is None or entry_bundle.ask_close is None:
                    raise ValueError("Bid/ask required for portfolio guard, but missing in entry bundle.")
                unrealized_portfolio_bps = self._calculate_unrealized_portfolio_bps(
                    entry_bundle.bid_close,
                    entry_bundle.ask_close,
                )
                if unrealized_portfolio_bps <= -intraday_drawdown_bps_limit:
                    log.info(
                        "[GUARD] portfolio drawdown limit: unrealized_portfolio_bps=%.1f <= -%.1f (blocking new entries)",
                        unrealized_portfolio_bps, intraday_drawdown_bps_limit
                    )
                    # TELEMETRY CONTRACT: Risk guard veto (Stage-1, after prediction)
                    self.veto_cand["veto_cand_risk_guard"] += 1
                    # KILL-CHAIN: risk block
                    self._killchain_inc_reason("BLOCK_RISK")
                    return None

            if side == "long":
                atr_median = getattr(self, "cluster_guard_atr_median", None)
                if atr_median is not None and current_atr_bps is not None:
                    same_dir = sum(
                        1 for t in self.open_trades
                        if getattr(t, "side", "").lower() == "long"
                    )
                    if same_dir >= 3 and current_atr_bps > atr_median:
                        log.info(
                            "[GUARD] cluster_guard: open_longs=%d atr=%.2f bps > median=%.2f bps",
                            same_dir,
                            current_atr_bps,
                            atr_median,
                        )
                        # KILL-CHAIN: volatility-style block (cluster guard)
                        self._killchain_inc_reason("BLOCK_VOL")
                        return None

        # KILL-CHAIN: passed post-policy regime/portfolio guards (candidate is still alive)
        self.killchain_n_after_regime_guard += 1

        # ENTRY_ONLY mode: Log entry event instead of creating trade
        if self.mode == "ENTRY_ONLY":
            self._log_entry_only_event(
                timestamp=now_ts,
                side=side,
                price=entry_bundle.close_price,
                prediction=prediction,
                policy_state=policy_state,
            )
            log.info(
                "[ENTRY_ONLY] Hypothetical entry logged: side=%s price=%.3f p_hat=%.4f (no trade created)",
                side.upper(),
                entry_bundle.close_price,
                prediction.p_hat,
            )
            return None  # No trade created in ENTRY_ONLY mode
        
        # Normal mode: Create LiveTrade
        # COMMIT B: Generate trade_uid (globally unique) and trade_id (display ID)
        if not hasattr(self, "_next_trade_id"):
            self._next_trade_id = 0
        self._next_trade_id += 1
        
        # Generate trade_id (display ID, kept for backward compatibility)
        trade_id = f"SIM-{int(time.time())}-{self._next_trade_id:06d}"
        
        # Generate trade_uid (globally unique primary key for journaling)
        # Format: {run_id}:{chunk_id}:{local_seq}:{uuid4_short}
        run_id = getattr(self._runner, "run_id", "unknown")
        chunk_id = getattr(self._runner, "chunk_id", "single")
        local_seq = self._next_trade_id
        uuid_short = uuid.uuid4().hex[:12]  # 12 hex chars for uniqueness
        trade_uid = f"{run_id}:{chunk_id}:{local_seq:06d}:{uuid_short}"
        
        # GUARD 1: Replay-only fail-fast - trade_uid format invariant
        if self.is_replay:
            expected_prefix = f"{run_id}:{chunk_id}:"
            if not trade_uid.startswith(expected_prefix):
                raise RuntimeError(
                    f"BAD_TRADE_UID_FORMAT_REPLAY: Generated trade_uid={trade_uid} does not start with "
                    f"expected prefix={expected_prefix}. run_id={run_id}, chunk_id={chunk_id}. "
                    f"This is a hard contract violation in replay mode."
                )
        
        # Force entry tracking: increment trade count when trade is created
        if hasattr(self, "_force_entry_trade_count"):
            self._force_entry_trade_count += 1
            log.info(
                "[FORCE_ENTRY] Trade created: count=%d (max=%d)",
                self._force_entry_trade_count,
                self.policy.get("debug_force", {}).get("max_trades", 1)
            )
        
        current_bar = candles.iloc[-1]
        try:
            entry_bid_price = float(current_bar["bid_close"])
            entry_ask_price = float(current_bar["ask_close"])
        except KeyError as exc:
            raise ValueError(
                "Bid/ask required for FARM_V2B 2025 replay, but missing in candles during entry creation."
            ) from exc
        if side == "long":
            entry_price = entry_ask_price
        else:
            entry_price = entry_bid_price
        
        # Base units (before any SNIPER overlays)
        base_units = self.exec.default_units if side == "long" else -self.exec.default_units

        # Get policy_state_snapshot early (needed for regime inputs extractor)
        policy_state_snapshot = self._last_policy_state or {}

        # Compute spread metrics BEFORE overlays (needed for regime classification)
        spread_bps_for_overlay: Optional[float] = None
        spread_pct_for_overlay: Optional[float] = None
        try:
            spread_raw = float(entry_ask_price) - float(entry_bid_price)
            if spread_raw > 0:
                spread_bps_for_overlay = spread_raw * 10000.0
                # Try to get spread_pct from percentile history if available
                if hasattr(self, "spread_history") and len(self.spread_history) > 0:
                    try:
                        spread_pct_for_overlay = self._percentile_from_history(self.spread_history, spread_bps_for_overlay)
                    except Exception:
                        pass
        except (TypeError, ValueError):
            pass

        # Composable SNIPER overlays (runtime-only, no change to entry/exit logic)
        overlays_meta: List[Dict[str, Any]] = []
        units_current = base_units

        # Extract regime inputs from all available sources (single source of truth)
        # This ensures overlays see the same data as offline analysis
        # Build feature_context-like dict from available sources
        feature_context_dict: Dict[str, Any] = {}
        if current_atr_bps is not None:
            feature_context_dict["atr_bps"] = current_atr_bps
        if spread_bps_for_overlay is not None:
            feature_context_dict["spread_bps"] = spread_bps_for_overlay
        if spread_pct_for_overlay is not None:
            feature_context_dict["spread_pct"] = spread_pct_for_overlay
        # Also check entry_bundle for additional fields
        if entry_bundle is not None:
            if hasattr(entry_bundle, "atr_bps") and entry_bundle.atr_bps is not None:
                feature_context_dict.setdefault("atr_bps", entry_bundle.atr_bps)
            if hasattr(entry_bundle, "spread_bps") and entry_bundle.spread_bps is not None:
                feature_context_dict.setdefault("spread_bps", entry_bundle.spread_bps)
        
        regime_inputs = get_runtime_regime_inputs(
            prediction=prediction,
            feature_context=feature_context_dict if feature_context_dict else None,
            spread_pct=spread_pct_for_overlay,  # Use pre-computed spread_pct
            current_atr_bps=current_atr_bps,
            entry_bundle=entry_bundle,
            policy_state=policy_state_snapshot,
            entry_time=now_ts,
        )
        
        # Extract individual fields for overlay calls
        _trend_regime = regime_inputs["trend_regime"]
        _vol_regime = regime_inputs["vol_regime"]
        _atr_bps = regime_inputs["atr_bps"]
        _spread_bps = regime_inputs["spread_bps"]
        _session = regime_inputs["session"]
        
        # Debug logging for first few trades per chunk (to verify regime inputs are correct)
        if not hasattr(self, "_overlay_debug_count"):
            self._overlay_debug_count = 0
        if self._overlay_debug_count < 5:
            self._overlay_debug_count += 1
            sources = regime_inputs.get("_sources", {})
            log.info(
                "[REGIME_EXTRACTOR] Trade %d - trend=%s (source=%s) vol=%s (source=%s) atr_bps=%s (source=%s) spread_bps=%s (source=%s) session=%s (source=%s)",
                self._overlay_debug_count,
                _trend_regime,
                sources.get("trend_regime", "unknown"),
                _vol_regime,
                sources.get("vol_regime", "unknown"),
                _atr_bps,
                sources.get("atr_bps", "unknown"),
                _spread_bps,
                sources.get("spread_bps", "unknown"),
                _session,
                sources.get("session", "unknown"),
            )
            
            # Runtime vs offline parity check (replay-only)
            try:
                from gx1.sniper.analysis.regime_classifier import classify_regime
                offline_row = {
                    "trend_regime": _trend_regime,
                    "vol_regime": _vol_regime,
                    "atr_bps": _atr_bps,
                    "spread_bps": _spread_bps,
                }
                offline_regime_class, offline_reason = classify_regime(offline_row)
                # Store for later comparison with overlay
                regime_inputs["_offline_regime_class"] = offline_regime_class
                regime_inputs["_offline_regime_reason"] = offline_reason
            except Exception as e:
                log.debug("[REGIME_EXTRACTOR] Offline classification failed: %s", e)

        # Overlay 1: Q4 × B_MIXED size gate
        regime_overlay_cfg: Dict[str, Any] = {}
        if hasattr(self, "policy") and getattr(self, "policy"):
            regime_overlay_cfg = self.policy.get("sniper_regime_size_overlay", {}) or {}
        try:
            units_1, meta_1 = apply_size_overlay(
                base_units=units_current,
                entry_time=now_ts,
                trend_regime=_trend_regime,
                vol_regime=_vol_regime,
                atr_bps=_atr_bps,
                spread_bps=_spread_bps,
                session=_session,
                cfg=regime_overlay_cfg,
            )
        except Exception as e:
            log.exception("SNIPER B_MIXED size overlay failed; falling back to current units")
            units_1 = units_current
            tb = traceback.format_exc(limit=20)
            meta_1 = {
                "overlay_applied": False,
                "overlay_name": "Q4_B_MIXED_SIZE",
                "reason": f"overlay_error:{type(e).__name__}",
                "error": str(e),
                "traceback": tb,
            }
        overlays_meta.append(meta_1)
        units_current = units_1

        # Overlay 2: Q4 × C_CHOP session-based size gate
        cchop_cfg: Dict[str, Any] = {}
        if hasattr(self, "policy") and getattr(self, "policy"):
            cchop_cfg = self.policy.get("sniper_q4_cchop_overlay", {}) or {}
        try:
            units_2, meta_2 = apply_q4_cchop_overlay(
                base_units=units_current,
                entry_time=now_ts,
                trend_regime=_trend_regime,
                vol_regime=_vol_regime,
                atr_bps=_atr_bps,
                spread_bps=_spread_bps,
                session=_session,
                cfg=cchop_cfg,
            )
        except Exception as e:
            log.exception("SNIPER Q4 C_CHOP session overlay failed; falling back to current units")
            units_2 = units_current
            tb = traceback.format_exc(limit=20)
            meta_2 = {
                "overlay_applied": False,
                "overlay_name": "Q4_C_CHOP_SESSION_SIZE",
                "reason": f"overlay_error:{type(e).__name__}",
                "error": str(e),
                "traceback": tb,
            }
        overlays_meta.append(meta_2)
        units_current = units_2

        # Overlay 3: Q4 × A_TREND size gate
        atrend_cfg: Dict[str, Any] = {}
        if hasattr(self, "policy") and getattr(self, "policy"):
            atrend_cfg = self.policy.get("sniper_q4_atrend_overlay", {}) or {}
        try:
            units_3, meta_3 = apply_q4_atrend_overlay(
                base_units=units_current,
                entry_time=now_ts,
                trend_regime=_trend_regime,
                vol_regime=_vol_regime,
                atr_bps=_atr_bps,
                spread_bps=_spread_bps,
                session=_session,
                cfg=atrend_cfg,
            )
        except Exception as e:
            log.exception("SNIPER Q4 A_TREND size overlay failed; falling back to current units")
            units_3 = units_current
            tb = traceback.format_exc(limit=20)
            meta_3 = {
                "overlay_applied": False,
                "overlay_name": "Q4_A_TREND_SIZE",
                "reason": f"overlay_error:{type(e).__name__}",
                "error": str(e),
                "traceback": tb,
            }
        overlays_meta.append(meta_3)
        units_current = units_3

        # Overlay 4: EU timing-based size gate (P4.1)
        eu_timing_cfg: Dict[str, Any] = {}
        if hasattr(self, "policy") and getattr(self, "policy"):
            eu_timing_cfg = self.policy.get("sniper_q4_eu_timing_overlay", {}) or {}
        try:
            units_4, meta_4 = apply_q4_eu_timing_overlay(
                base_units=units_current,
                entry_time=now_ts,
                trend_regime=_trend_regime,
                vol_regime=_vol_regime,
                atr_bps=_atr_bps,
                spread_bps=_spread_bps,
                session=_session,
                cfg=eu_timing_cfg,
            )
        except Exception as e:
            log.exception("SNIPER EU timing overlay failed; falling back to current units")
            units_4 = units_current
            tb = traceback.format_exc(limit=20)
            meta_4 = {
                "overlay_applied": False,
                "overlay_name": "EU_TIMING_SIZE",
                "reason": f"overlay_error:{type(e).__name__}",
                "error": str(e),
                "traceback": tb,
            }
        overlays_meta.append(meta_4)
        units_current = units_4
        
        # Overlay 5: ENTRY_V10.1 Size Overlay (OFFLINE ONLY - AGGR mode)
        # This overlay is only active when entry_v10_1_size_overlay is enabled in policy
        if ENTRY_V10_1_SIZE_OVERLAY_AVAILABLE:
            v10_1_overlay_cfg: Dict[str, Any] = {}
            if hasattr(self, "policy") and getattr(self, "policy"):
                # Check entry_config for overlay config
                entry_config_path = self.policy.get("entry_config")
                if entry_config_path:
                    try:
                        import yaml
                        from pathlib import Path
                        entry_cfg = yaml.safe_load(Path(entry_config_path).read_text())
                        v10_1_overlay_cfg = entry_cfg.get("entry_v10_1_size_overlay", {}) or {}
                    except Exception as e:
                        log.debug("[ENTRY_V10_1_SIZE_OVERLAY] Failed to load entry config: %s", e)
            
            # Initialize overlay if enabled (lazy initialization)
            if not hasattr(self, "_entry_v10_1_size_overlay"):
                if v10_1_overlay_cfg.get("enabled", False):
                    try:
                        self._entry_v10_1_size_overlay = load_entry_v10_1_size_overlay(v10_1_overlay_cfg)
                        log.info("[ENTRY_V10_1_SIZE_OVERLAY] Loaded successfully")
                    except Exception as e:
                        log.warning("[ENTRY_V10_1_SIZE_OVERLAY] Failed to load: %s", e)
                        self._entry_v10_1_size_overlay = None
                else:
                    self._entry_v10_1_size_overlay = None
            
            # Apply overlay if available
            if self._entry_v10_1_size_overlay is not None:
                try:
                    # Get p_long_v10_1 from policy_state or entry prediction
                    p_long_v10_1 = policy_state.get("p_long_v10_1") or policy_state.get("p_long")
                    if p_long_v10_1 is None:
                        # Try to get from entry prediction if V10 is enabled
                        if hasattr(self._runner, "entry_v10_enabled") and self._runner.entry_v10_enabled:
                            # p_long_v10_1 should be in policy_state from V10 prediction
                            p_long_v10_1 = policy_state.get("p_long")
                    
                    if p_long_v10_1 is None:
                        log.warning("[ENTRY_V10_1_SIZE_OVERLAY] p_long_v10_1 not available, skipping overlay")
                        units_5 = units_current
                        meta_5 = {
                            "overlay_applied": False,
                            "overlay_name": "ENTRY_V10_1_SIZE",
                            "reason": "p_long_v10_1_not_available",
                        }
                    else:
                        # Build regime string (e.g., "UP×LOW")
                        trend_name = "UP" if _trend_regime == "TREND_UP" else ("DOWN" if _trend_regime == "TREND_DOWN" else "NEUTRAL")
                        vol_name = _vol_regime.upper() if isinstance(_vol_regime, str) else ("LOW" if _vol_regime == 0 else ("MEDIUM" if _vol_regime == 1 else ("HIGH" if _vol_regime == 2 else "EXTREME")))
                        regime_str = f"{trend_name}×{vol_name}"
                        
                        # Get SL bps from trade config
                        sl_bps = abs(int(self.tick_cfg.get("sl_bps", 100)))
                        
                        units_5, meta_5 = self._entry_v10_1_size_overlay.apply_overlay(
                            base_units=units_current,
                            p_long_v10_1=float(p_long_v10_1),
                            session=_session,
                            regime=regime_str,
                            sl_bps=sl_bps,
                        )
                except Exception as e:
                    log.exception("ENTRY_V10.1 size overlay failed; falling back to current units")
                    units_5 = units_current
                    tb = traceback.format_exc(limit=20)
                    meta_5 = {
                        "overlay_applied": False,
                        "overlay_name": "ENTRY_V10_1_SIZE",
                        "reason": f"overlay_error:{type(e).__name__}",
                        "error": str(e),
                        "traceback": tb,
                    }
                overlays_meta.append(meta_5)
                units_current = units_5
            else:
                # Overlay not enabled - add no-op metadata
                meta_5 = {
                    "overlay_applied": False,
                    "overlay_name": "ENTRY_V10_1_SIZE",
                    "reason": "disabled",
                }
                overlays_meta.append(meta_5)
        else:
            # Module not available - add no-op metadata
            meta_5 = {
                "overlay_applied": False,
                "overlay_name": "ENTRY_V10_1_SIZE",
                "reason": "module_not_available",
            }
            overlays_meta.append(meta_5)
        
        units_out = units_current
        
        # NO-TRADE check: if overlay set units to 0, skip trade creation
        if units_out == 0:
            # Log the reason from the last overlay that set units to 0
            reason_no_trade = "unknown"
            overlay_name_no_trade = "unknown"
            for meta in reversed(overlays_meta):
                if meta.get("size_after_units") == 0:
                    reason_no_trade = meta.get("reason", "unknown")
                    overlay_name_no_trade = meta.get("overlay_name", "unknown")
                    break
            log.info(
                "[NO_TRADE] Overlay set units to 0, skipping trade creation. Overlay=%s, Reason=%s",
                overlay_name_no_trade,
                reason_no_trade,
            )
            # KILL-CHAIN: risk sizing / overlays blocked trade
            self._killchain_inc_reason("BLOCK_RISK")
            return None

        # KILL-CHAIN: passed risk sizing (units decided)
        self.killchain_n_after_risk_sizing += 1
        
        # Debug logging: log overlay regime classification for first few trades
        if self._overlay_debug_count <= 5:
            for meta in overlays_meta:
                if meta.get("overlay_name") == "Q4_A_TREND_SIZE":
                    overlay_regime = meta.get("regime_class")
                    offline_regime = regime_inputs.get("_offline_regime_class")
                    if overlay_regime != offline_regime:
                        log.warning(
                            "[REGIME_PARITY_MISMATCH] Trade %d - Overlay regime=%s (reason=%s) vs Offline regime=%s (reason=%s). Runtime inputs: trend=%s vol=%s atr_bps=%s spread_bps=%s",
                            self._overlay_debug_count,
                            overlay_regime,
                            meta.get("reason"),
                            offline_regime,
                            regime_inputs.get("_offline_regime_reason"),
                            _trend_regime,
                            _vol_regime,
                            _atr_bps,
                            _spread_bps,
                        )
                    else:
                        log.info(
                            "[OVERLAY_DEBUG] Q4_A_TREND overlay: applied=%s regime_class=%s reason=%s size_before=%s size_after=%s",
                            meta.get("overlay_applied"),
                            meta.get("regime_class"),
                            meta.get("reason"),
                            meta.get("size_before_units"),
                            meta.get("size_after_units"),
                        )
                    break
        
        # SNIPER logging: trade created (policy passed)
        if use_sniper:
            p_long_val = policy_state.get("p_long")
            n_signals_val = 1  # Trade created = 1 signal passed
            self._log_sniper_cycle(
                ts=current_ts,
                session=current_session,
                in_scope=sniper_in_scope,
                warmup_ready=sniper_warmup_ready,
                degraded=sniper_degraded,
                spread_bps=current_spread_bps,
                atr_bps=current_atr_bps,
                eval_ran=True,
                reason="TRADE",
                decision="LONG",
                n_signals=n_signals_val,
                p_long=p_long_val,
            )
        
        # TELEMETRY CONTRACT: Increment n_candidate_pass (candidate passed all Stage-1 gates)
        self.entry_telemetry["n_candidate_pass"] += 1
        
        # Lazy import to avoid circular dependency
        from gx1.execution.oanda_demo_runner import LiveTrade
        
        # TELEMETRY CONTRACT: Single source of truth for trade creation
        # n_trades_created is incremented here (not in policy-specific branches)
        # KILL-CHAIN: trade creation attempt (right before LiveTrade is instantiated)
        self.killchain_n_trade_create_attempts += 1
        trade = LiveTrade(
            trade_id=trade_id,
            trade_uid=trade_uid,
            entry_time=now_ts,
            side=side,
            units=units_out,
            entry_price=entry_price,
            entry_bid=entry_bid_price,
            entry_ask=entry_ask_price,
            atr_bps=entry_bundle.atr_bps,
            vol_bucket=entry_bundle.vol_bucket,
            entry_prob_long=prediction.prob_long,
            entry_prob_short=prediction.prob_short,
            dry_run=self.exec.dry_run,
        )
        
        # TELEMETRY CONTRACT: Increment n_trades_created (SINGLE SOURCE OF TRUTH)
        # This counter always matches trades_total in replay, regardless of policy mode
        self.entry_telemetry["n_trades_created"] += 1
        # KILL-CHAIN: trade created
        self.killchain_n_trade_created += 1
        # Track trade session distribution
        session_key = policy_state.get("session") or infer_session_tag(trade.entry_time).upper()
        self.entry_telemetry["trade_sessions"][session_key] = self.entry_telemetry["trade_sessions"].get(session_key, 0) + 1
        # Legacy counter (for backward compatibility)
        if is_farm_v2b:
            self.farm_diag["n_after_policy_thresholds"] += 1
        
        # Log entry decision with temperature
        log.info(
            "[ENTRY SIGNAL] Session=%s, T=%.3f, side=%s, p_hat=%.4f, margin=%.4f, p_long=%.4f, p_short=%.4f",
            prediction.session,
            T,
            side.upper(),
            prediction.p_hat,
            prediction.margin,
            prediction.prob_long,
            prediction.prob_short,
        )
        
        # Generate client_order_id for idempotency
        client_order_id = self._generate_client_order_id(trade.entry_time, trade.entry_price, trade.side)
        trade.client_order_id = client_order_id
        
        # Log FORCED_CANARY_TRADE if this is a force entry trade
        # policy_state_snapshot is set later, but we can check _last_policy_state which was updated earlier
        policy_state_for_logging = self._last_policy_state or {}
        if policy_state_for_logging.get("force_entry", False):
            log.warning(
                "[FORCED_CANARY_TRADE] Trade created: trade_id=%s client_ext_id=%s reason=%s",
                trade.trade_id,
                client_order_id,
                policy_state_for_logging.get("force_entry_reason", "timeout")
            )
        
        # Store TP/SL thresholds in trade.extra for TickWatcher and broker-side orders
        # Get TP/SL from tick_exit config (or use defaults: EXIT_V3 profile = 180/100)
        tp_bps = int(self.tick_cfg.get("tp_bps", 180))
        sl_bps = int(self.tick_cfg.get("sl_bps", 100))
        be_trigger_bps = int(self.tick_cfg.get("be_trigger_bps", 50))
        
        if not hasattr(trade, "extra"):
            trade.extra = {}
        trade.extra["tp_bps"] = tp_bps
        trade.extra["sl_bps"] = sl_bps
        trade.extra["be_trigger_bps"] = be_trigger_bps
        trade.extra["be_active"] = False
        trade.extra["be_price"] = None

        # Compute spread metrics for hybrid exit routing / diagnostics
        spread_bps: Optional[float] = None
        spread_pct: Optional[float] = None
        try:
            spread_raw = float(entry_ask_price) - float(entry_bid_price)
            if spread_raw > 0:
                spread_bps = spread_raw * 10000.0
        except (TypeError, ValueError):
            spread_bps = None
        if spread_bps is not None:
            spread_pct = self._percentile_from_history(self.spread_history, spread_bps)
            if np.isfinite(spread_bps):
                self.spread_history.append(spread_bps)

        # Note: policy_state_snapshot is now set earlier (before overlay calls)

        # Compute range features BEFORE router selection (needed for V3 router with range features)
        range_window = 96
        range_pos, distance_to_range = self._compute_range_features(candles, window=range_window)
        trade.extra["range_pos"] = float(range_pos)
        trade.extra["distance_to_range"] = float(distance_to_range)
        
        # Compute range_edge_dist_atr (ATR-normalized distance to nearest edge)
        atr_value: Optional[float] = None
        if current_atr_bps is not None and current_atr_bps > 0 and entry_price > 0:
            atr_value = (current_atr_bps / 10000.0) * entry_price
        
        if len(candles) >= range_window + 1:
            recent = candles.iloc[-(range_window+1):-1]
        else:
            recent = candles.tail(range_window)
        
        has_direct = all(col in candles.columns for col in ['high', 'low', 'close'])
        has_bid_ask = all(col in candles.columns for col in ['bid_high', 'ask_high', 'bid_low', 'ask_low', 'bid_close', 'ask_close'])
        
        range_hi = None
        range_lo = None
        price_ref = None
        range_edge_dist_atr = 0.0
        
        if has_direct or has_bid_ask:
            if has_direct:
                high_vals = recent['high'].values
                low_vals = recent['low'].values
                close_vals = recent['close'].values
            else:
                high_vals = (recent['bid_high'].values + recent['ask_high'].values) / 2.0
                low_vals = (recent['bid_low'].values + recent['ask_low'].values) / 2.0
                close_vals = (recent['bid_close'].values + recent['ask_close'].values) / 2.0
            
            range_hi = float(np.max(high_vals))
            range_lo = float(np.min(low_vals))
            price_ref = float(close_vals[-1])
            
            range_edge_dist_atr = self._compute_range_edge_dist_atr(
                candles=candles,
                price_ref=price_ref,
                range_hi=range_hi,
                range_lo=range_lo,
                atr_value=atr_value,
                window=range_window,
            )
        
        trade.extra["range_edge_dist_atr"] = float(range_edge_dist_atr)

        if getattr(self, "exit_hybrid_enabled", False) and getattr(self, "exit_mode_selector", None):
            session_for_exit = policy_state_snapshot.get("session")
            if not session_for_exit:
                from gx1.execution.live_features import infer_session_tag as _infer_session_tag

                session_for_exit = _infer_session_tag(trade.entry_time).upper()
            farm_regime = (
                policy_state_snapshot.get("farm_regime")
                or policy_state_snapshot.get("_farm_regime")
                or "UNKNOWN"
            )
            selected_profile = self.exit_mode_selector.choose_exit_profile(
                atr_bps=current_atr_bps if current_atr_bps is not None else trade.atr_bps,
                atr_pct=current_atr_pct,
                spread_bps=spread_bps,
                spread_pct=spread_pct,
                session=session_for_exit,
                regime=farm_regime,
                range_pos=range_pos,
                distance_to_range=distance_to_range,
                range_edge_dist_atr=range_edge_dist_atr,
            )
            trade.extra["exit_profile"] = selected_profile
            trade.extra.setdefault("exit_hybrid", {})
            trade.extra["exit_hybrid"].update(
                {
                    "mode": getattr(self, "exit_hybrid_mode", "RULE5_RULE6A_ATR_SPREAD_V1"),
                    "atr_bps": current_atr_bps,
                    "atr_pct": current_atr_pct,
                    "spread_bps": spread_bps,
                    "spread_pct": spread_pct,
                    "session": session_for_exit,
                    "regime": farm_regime,
                }
            )
            log.info(
                "[HYBRID_EXIT] trade=%s session=%s regime=%s atr_bps=%s atr_pct=%s spread_bps=%s spread_pct=%s -> %s",
                trade.trade_id,
                session_for_exit,
                farm_regime,
                f"{current_atr_bps:.2f}" if current_atr_bps is not None else "nan",
                f"{current_atr_pct:.1f}" if current_atr_pct is not None else "nan",
                f"{spread_bps:.2f}" if spread_bps is not None else "nan",
                f"{spread_pct:.1f}" if spread_pct is not None else "nan",
                selected_profile,
            )

        # Optional debug log
        log.debug(
            "[RANGE_FEAT] trade_id=%s range_pos=%.4f distance_to_range=%.4f range_edge_dist_atr=%.4f window=%d",
            trade.trade_id,
            range_pos,
            distance_to_range,
            range_edge_dist_atr,
            range_window,
        )

        # Ensure exit profile + exit policy initialization through shared helper
        self._ensure_exit_profile(trade, context="entry_manager")
        if self.exit_config_name and not (getattr(trade, "extra", {}) or {}).get("exit_profile"):
            raise RuntimeError(
                f"[EXIT_PROFILE] Trade created without exit_profile under exit-config {self.exit_config_name}: {trade.trade_id}"
            )
        policy_state_snapshot = self._last_policy_state or policy_state_snapshot
        if hasattr(self, "_record_entry_diag"):
            try:
                self._record_entry_diag(trade, policy_state_snapshot, prediction)
            except Exception as diag_exc:
                log.warning("[ENTRY_DIAG] Failed to record entry diagnostics for %s: %s", trade.trade_id, diag_exc)
        
        # DEL 1: Hook PolicyDecisionCollector - collect "enter" decision (after trade creation)
        # Hook point: After trade is created, we have all info (size, exit_profile, etc.)
        if hasattr(self._runner, "replay_eval_collectors") and self._runner.replay_eval_collectors:
            decision_collector = self._runner.replay_eval_collectors.get("policy_decisions")
            if decision_collector:
                reasons = []
                # Collect reasons from policy_state and guards
                if policy_state.get("veto_cand_threshold", 0) > 0:
                    reasons.append("threshold")
                if current_spread_bps and current_spread_bps > 100:  # Spread cap
                    reasons.append("spread")
                if risk_guard_blocked:
                    reasons.append("safety")
                if policy_state.get("brain_vol_regime") not in ["LOW", "MEDIUM", "HIGH"]:
                    reasons.append("regime")
                if not reasons:
                    reasons.append("policy_pass")  # Default if no specific reason
                
                exit_profile = trade.extra.get("exit_profile") if hasattr(trade, "extra") and trade.extra else None
                chosen_size = float(trade.units) if hasattr(trade, "units") else None
                
                decision_collector.collect(
                    timestamp=current_ts,
                    decision="enter",
                    reasons=reasons,
                    chosen_size=chosen_size,
                    exit_policy_id=exit_profile,
                )
        
        # Log entry signal to trade journal
        if hasattr(self._runner, "trade_journal") and self._runner.trade_journal:
            try:
                from gx1.monitoring.trade_journal import EVENT_ENTRY_SIGNAL
                
                # Collect gate information
                gates = {}
                if spread_bps is not None:
                    gates["spread_bps"] = {"value": spread_bps, "passed": True}  # Simplified
                if current_atr_bps is not None:
                    gates["atr_bps"] = {"value": current_atr_bps, "passed": True}
                if policy_state_snapshot:
                    gates["regime"] = {"value": policy_state_snapshot.get("farm_regime", "UNKNOWN"), "passed": True}
                    gates["session"] = {"value": policy_state_snapshot.get("session", "UNKNOWN"), "passed": True}
                
                # Features snapshot
                features_snapshot = {
                    "atr_bps": current_atr_bps,
                    "atr_pct": current_atr_pct,
                    "spread_bps": spread_bps,
                    "spread_pct": spread_pct,
                    "range_pos": float(range_pos),
                    "distance_to_range": float(distance_to_range),
                    "range_edge_dist_atr": float(range_edge_dist_atr),
                }
                
                self._runner.trade_journal.log(
                    EVENT_ENTRY_SIGNAL,
                    {
                        "entry_time": trade.entry_time.isoformat() if hasattr(trade.entry_time, "isoformat") else str(trade.entry_time),
                        "entry_price": trade.entry_price,
                        "side": trade.side,
                        "entry_model_outputs": {
                            "p_long": prediction.prob_long,
                            "p_short": prediction.prob_short,
                            "p_hat": prediction.p_hat,
                            "margin": prediction.margin,
                            "session": prediction.session,
                        },
                        "gates": gates,
                        "features_snapshot": features_snapshot,
                        "warmup_state": {
                            "bars_since_start": len(candles) if candles is not None else None,
                        },
                    },
                    trade_key={
                        "entry_time": trade.entry_time.isoformat() if hasattr(trade.entry_time, "isoformat") else str(trade.entry_time),
                        "entry_price": trade.entry_price,
                        "side": trade.side,
                    },
                    trade_id=trade.trade_id,
                )
            except Exception as e:
                log.warning("[TRADE_JOURNAL] Failed to log ENTRY_SIGNAL: %s", e)
        
        # Log structured entry snapshot and feature context
        if hasattr(self._runner, "trade_journal") and self._runner.trade_journal:
            try:
                # Collect entry filters
                entry_filters_passed = []
                entry_filters_blocked = []
                
                # Check which filters passed (simplified - actual gates checked earlier)
                if spread_bps is not None and spread_pct is not None:
                    entry_filters_passed.append("spread_ok")
                if current_atr_bps is not None and current_atr_pct is not None:
                    entry_filters_passed.append("atr_ok")
                if policy_state_snapshot:
                    if policy_state_snapshot.get("farm_regime"):
                        entry_filters_passed.append("regime_ok")
                    if policy_state_snapshot.get("session"):
                        entry_filters_passed.append("session_ok")
                
                # Entry snapshot
                entry_time_iso = trade.entry_time.isoformat() if hasattr(trade.entry_time, "isoformat") else str(trade.entry_time)
                
                # Safely get instrument and model_name (avoid AttributeError from __getattr__)
                instrument_val = "XAU_USD"
                try:
                    if hasattr(self._runner, "instrument"):
                        instrument_val = self._runner.instrument
                except AttributeError:
                    pass
                
                model_name_val = None
                try:
                    if hasattr(self._runner, "model_name"):
                        model_name_val = self._runner.model_name
                except AttributeError:
                    pass
                
                # Check if this is a force entry trade (CANARY only)
                is_force_entry = policy_state_snapshot.get("force_entry", False) if policy_state_snapshot else False
                force_entry_reason = policy_state_snapshot.get("force_entry_reason", None) if policy_state_snapshot else None
                
                # Set test_mode and reason for force entry trades
                test_mode = is_force_entry
                reason = "FORCED_CANARY_TRADE" if is_force_entry else None
                
                # Check for degraded warmup (CANARY only)
                warmup_degraded = getattr(self._runner, '_warmup_degraded', False)
                cached_bars_at_entry = getattr(self._runner, '_cached_bars_at_startup', None)
                warmup_bars_required = self._runner.policy.get("warmup_bars", 288) if hasattr(self._runner, 'policy') else None
                
                # Extract vol_regime and trend_regime from policy_state
                vol_regime = None
                trend_regime = None
                if policy_state_snapshot:
                    # Prefer brain_vol_regime/brain_trend_regime (Big Brain V1)
                    vol_regime = policy_state_snapshot.get("brain_vol_regime") or policy_state_snapshot.get("vol_regime")
                    trend_regime = policy_state_snapshot.get("brain_trend_regime") or policy_state_snapshot.get("trend_regime")
                    # Fallback to farm_regime if available (for FARM policies)
                    if not vol_regime:
                        farm_regime = policy_state_snapshot.get("farm_regime")
                        if farm_regime:
                            # farm_regime is like "ASIA_LOW", extract vol part
                            if "_" in farm_regime:
                                vol_regime = farm_regime.split("_")[-1]  # Extract "LOW" from "ASIA_LOW"
                            else:
                                vol_regime = farm_regime
                
                # Extract risk guard metadata from policy_state
                risk_guard_blocked = policy_state_snapshot.get("risk_guard_blocked", False) if policy_state_snapshot else False
                risk_guard_reason = policy_state_snapshot.get("risk_guard_reason") if policy_state_snapshot else None
                risk_guard_details = policy_state_snapshot.get("risk_guard_details", {}) if policy_state_snapshot else {}
                risk_guard_clamp = policy_state_snapshot.get("risk_guard_clamp") if policy_state_snapshot else None
                
                # Entry Critic V1 scoring (shadow-only: calculate but don't gate)
                entry_critic_data: Optional[Dict[str, Any]] = None
                if self._entry_critic_model is not None and self._entry_critic_feature_order is not None:
                    try:
                        from gx1.rl.entry_critic_runtime import (
                            prepare_entry_critic_features,
                            score_entry_critic,
                        )
                        # Get p_long from prediction
                        p_long_val = prediction.prob_long if hasattr(prediction, "prob_long") else None
                        if p_long_val is not None:
                            # Prepare shadow hits for Entry Critic (simplified: use real threshold)
                            shadow_hits_for_critic: Dict[float, bool] = {}
                            if self._shadow_thresholds:
                                for shadow_thr in self._shadow_thresholds:
                                    shadow_hits_for_critic[shadow_thr] = p_long_val >= shadow_thr
                            
                            feature_vector = prepare_entry_critic_features(
                                p_long=p_long_val,
                                spread_bps=regime_inputs.get("spread_bps"),
                                atr_bps=regime_inputs.get("atr_bps"),
                                trend_regime=regime_inputs.get("trend_regime"),
                                vol_regime=regime_inputs.get("vol_regime"),
                                session=regime_inputs.get("session"),
                                shadow_hits=shadow_hits_for_critic,
                                real_threshold=policy_sniper_cfg.get("min_prob_long", 0.67) if use_sniper else None,
                                feature_order=self._entry_critic_feature_order,
                            )
                            if feature_vector is not None:
                                entry_critic_score = score_entry_critic(
                                    model=self._entry_critic_model,
                                    feature_vector=feature_vector,
                                )
                                if entry_critic_score is not None:
                                    entry_critic_data = {
                                        "score_v1": float(entry_critic_score),
                                        "model": self._entry_critic_meta.get("model_type", "entry_critic_v1") if self._entry_critic_meta else "entry_critic_v1",
                                        "target": self._entry_critic_meta.get("target", "label_profitable_10bps") if self._entry_critic_meta else "label_profitable_10bps",
                                        "features_used": self._entry_critic_meta.get("features", []) if self._entry_critic_meta else [],
                                    }
                    except Exception as e:
                        log.debug(f"[ENTRY_CRITIC] Failed to score entry for trade journal: {e}")
                
                # Use regime inputs from extractor (same as overlays see)
                # This ensures entry_snapshot has the same regime data as overlay metadata
                journal_start = time.perf_counter()
                # OPPGAVE RUNTIME: Entry journaling atomicity - hard contract in replay
                try:
                    self._runner.trade_journal.log_entry_snapshot(
                        entry_time=entry_time_iso,
                        trade_uid=trade.trade_uid,  # Primary key (COMMIT C)
                        trade_id=trade.trade_id,  # Display ID (backward compatibility)
                        instrument=instrument_val,
                        side=trade.side,
                        entry_price=trade.entry_price,
                        units=units_out,
                        base_units=base_units,
                        session=regime_inputs["session"],  # Use extractor result
                        regime=policy_state_snapshot.get("farm_regime") if policy_state_snapshot else None,  # Legacy field
                        vol_regime=regime_inputs["vol_regime"],  # Use extractor result
                        trend_regime=regime_inputs["trend_regime"],  # Use extractor result
                        entry_model_version=model_name_val,
                        entry_score={
                            "p_long": prediction.prob_long,
                            "p_short": prediction.prob_short,
                            "p_hat": prediction.p_hat,
                            "margin": prediction.margin,
                        },
                        entry_filters_passed=entry_filters_passed,
                        entry_filters_blocked=entry_filters_blocked,
                        test_mode=test_mode,
                        reason=reason,
                        warmup_degraded=warmup_degraded,
                        cached_bars_at_entry=cached_bars_at_entry,
                        warmup_bars_required=warmup_bars_required,
                        risk_guard_blocked=risk_guard_blocked,
                        risk_guard_reason=risk_guard_reason,
                        risk_guard_details=risk_guard_details,
                        risk_guard_min_prob_long_clamp=risk_guard_clamp,
                        sniper_overlays=overlays_meta,
                        sniper_overlay=overlays_meta[-1] if overlays_meta else None,
                        # Note: V10.1 size overlay metadata is included in sniper_overlays list above
                        atr_bps=regime_inputs["atr_bps"],  # Store for consistency with overlay inputs
                        spread_bps=regime_inputs["spread_bps"],  # Store for consistency with overlay inputs
                        entry_critic=entry_critic_data,  # Entry Critic V1 score (shadow-only)
                    )
                    # Increment success counter
                    self.entry_telemetry["n_entry_snapshots_written"] += 1
                except Exception as e:
                    # Increment failure counter
                    self.entry_telemetry["n_entry_snapshots_failed"] += 1
                    # OPPGAVE RUNTIME: Fail-fast in replay mode
                    if self.is_replay:
                        raise RuntimeError(
                            f"ENTRY_SNAPSHOT_MISSING: Failed to log entry_snapshot for trade_uid={trade.trade_uid}, "
                            f"trade_id={trade.trade_id}. This is a hard contract violation in replay mode. "
                            f"Error: {e}"
                        ) from e
                    else:
                        # Live mode: log error and set degraded flag (non-fatal)
                        log.error(
                            "[TRADE_JOURNAL] Failed to log structured entry snapshot for trade_uid=%s, trade_id=%s: %s",
                            trade.trade_uid,
                            trade.trade_id,
                            e,
                            exc_info=True,
                        )
                        # Set degraded flag (if runner supports it)
                        if hasattr(self._runner, "data_integrity_degraded"):
                            self._runner.data_integrity_degraded = True
                
                # Feature context (immutable snapshot)
                atr_value_price = None
                if current_atr_bps is not None and current_atr_bps > 0 and trade.entry_price > 0:
                    atr_value_price = (current_atr_bps / 10000.0) * trade.entry_price
                
                # Get last closed candle
                candle_close = None
                candle_high = None
                candle_low = None
                if candles is not None and len(candles) > 0:
                    last_bar = candles.iloc[-1]
                    if "close" in candles.columns:
                        candle_close = float(last_bar["close"])
                        candle_high = float(last_bar["high"])
                        candle_low = float(last_bar["low"])
                    elif "bid_close" in candles.columns and "ask_close" in candles.columns:
                        candle_close = float((last_bar["bid_close"] + last_bar["ask_close"]) / 2.0)
                        candle_high = float((last_bar["bid_high"] + last_bar["ask_high"]) / 2.0)
                        candle_low = float((last_bar["bid_low"] + last_bar["ask_low"]) / 2.0)
                
                spread_value_price = None
                if entry_ask_price and entry_bid_price:
                    spread_value_price = float(entry_ask_price) - float(entry_bid_price)
                
                self._runner.trade_journal.log_feature_context(
                    trade_uid=trade.trade_uid,  # Primary key (COMMIT C)
                    trade_id=trade.trade_id,  # Display ID (backward compatibility)
                    atr_bps=current_atr_bps,
                    atr_price=atr_value_price,
                    atr_percentile=current_atr_pct,
                    range_pos=float(range_pos),
                    distance_to_range=float(distance_to_range),
                    range_edge_dist_atr=float(range_edge_dist_atr),
                    spread_price=spread_value_price,
                    spread_pct=spread_pct,
                    candle_close=candle_close,
                    candle_high=candle_high,
                    candle_low=candle_low,
                )
                journal_time = time.perf_counter() - journal_start
                # Accumulate journal time
                if hasattr(self._runner, 'perf_journal_time'):
                    self._runner.perf_journal_time += journal_time
                else:
                    # Initialize if not exists (should not happen, but defensive)
                    self._runner.perf_journal_time = journal_time
            except Exception as e:
                log.warning("[TRADE_JOURNAL] Failed to log structured entry snapshot: %s", e)
        
        # CRITICAL: Store ATR regime for trade log reporting (ENTRY-regime)
        # This is the source of truth for vol_regime_entry in CSV
        # Get ATR regime from features (same logic as Big Brain)
        atr_regime_name = "ALL"  # Default fallback
        if hasattr(self, "_last_policy_state") and self._last_policy_state:
            policy_state = policy_state_snapshot
            brain_vol = policy_state.get("brain_vol_regime", None)
            if brain_vol and brain_vol != "UNKNOWN":
                atr_regime_name = brain_vol  # Use Big Brain V1 vol_regime
        else:
            # Fallback: extract from features if Big Brain V1 not available
            feat_row = entry_bundle.features.iloc[-1]
            if "_v1_atr_regime_id" in entry_bundle.features.columns:
                atr_regime_id = feat_row.get("_v1_atr_regime_id")
                if not pd.isna(atr_regime_id):
                    try:
                        regime_val = int(atr_regime_id)
                        # Simple mapping: 0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME
                        mapping = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
                        atr_regime_name = mapping.get(regime_val, "ALL")
                    except (TypeError, ValueError):
                        atr_regime_name = "ALL"
        # CRITICAL: Store atr_regime (entry-regime) ALWAYS
        # For FARM_V1, we MUST NOT set atr_regime here because it's computed BEFORE policy filtering
        # We'll set it later from vol_regime_entry (computed AFTER policy filtering)
        # For non-FARM modes, set it now
        if not (hasattr(self, "farm_v1_mode") and self.farm_v1_mode):
            # For non-FARM modes, set atr_regime now
            if atr_regime_name == "ALL" and vol_regime_entry:
                # Use vol_regime_entry if atr_regime_name is default
                atr_regime_name = vol_regime_entry
        trade.extra["atr_regime"] = atr_regime_name
        # For FARM_V1, atr_regime will be set later from vol_regime_entry (after policy filtering)
        
        # Store Big Brain V1 policy_state in trade.extra["big_brain_v1"] (observe-only, for logging/analysis)
        # CRITICAL: Always store Big Brain V1 data, even if UNKNOWN (for coverage tracking)
        if hasattr(self, "_last_policy_state") and self._last_policy_state:
            policy_state = self._last_policy_state
            
            # Store all Big Brain V1 data in nested dict for consistency
            trade.extra["big_brain_v1"] = {
                "brain_trend_regime": policy_state.get("brain_trend_regime", "UNKNOWN"),
                "brain_vol_regime": policy_state.get("brain_vol_regime", "UNKNOWN"),
                "brain_risk_score": policy_state.get("brain_risk_score", 0.0),
                # Coverage cutoff adjustment
                "coverage_cutoff": policy_state.get("coverage_cutoff", 0.20),
                "coverage_cutoff_base": policy_state.get("coverage_cutoff_base", 0.20),
                "coverage_cutoff_adj": policy_state.get("coverage_cutoff_adj", 0.0),
            }
            
            # Directional bias removed - margin_min adjustments now only from adaptive thresholding
            
            # Store hybrid entry data in trade.extra (if available)
            if "hybrid_entry" in policy_state:
                trade.extra["hybrid_entry"] = policy_state["hybrid_entry"]
                log.debug("[CREATE_TRADE] hybrid_entry stored in trade.extra")
            
            # Store adaptive thresholding data in trade.extra (if available)
            if "adaptive_thresholding" in policy_state:
                trade.extra["adaptive_thresholding"] = policy_state["adaptive_thresholding"]
                log.debug("[CREATE_TRADE] adaptive_thresholding stored in trade.extra: conflict_norm=%.3f cutoff=%.3f", 
                         policy_state["adaptive_thresholding"].get("conflict_norm", 0.0),
                         policy_state["adaptive_thresholding"].get("coverage_cutoff_new", 0.0))
            else:
                log.warning("[CREATE_TRADE] adaptive_thresholding NOT in policy_state - this should not happen!")
            
            # Always log Big Brain V1 data presence
            if "big_brain_v1" in trade.extra:
                bb_v1 = trade.extra["big_brain_v1"]
                log.debug("[CREATE_TRADE] big_brain_v1 stored: trend=%s vol=%s risk=%.3f", 
                         bb_v1.get("brain_trend_regime", "UNKNOWN"),
                         bb_v1.get("brain_vol_regime", "UNKNOWN"),
                         bb_v1.get("brain_risk_score", 0.0))
            else:
                log.warning("[CREATE_TRADE] big_brain_v1 NOT in trade.extra - this should not happen!")
            
            # Score shaping removed - no longer storing shaping adjustments
            
            # Optional: store probabilities for analysis
            if "brain_trend_probs" in policy_state:
                trade.extra["big_brain_v1"]["brain_trend_probs"] = policy_state["brain_trend_probs"]
            if "brain_vol_probs" in policy_state:
                trade.extra["big_brain_v1"]["brain_vol_probs"] = policy_state["brain_vol_probs"]
            
            # Backward compatibility: also store top-level fields for notes parsing
            trade.extra["brain_trend_regime"] = policy_state.get("brain_trend_regime", "UNKNOWN")
            trade.extra["brain_vol_regime"] = policy_state.get("brain_vol_regime", "UNKNOWN")
            trade.extra["brain_risk_score"] = policy_state.get("brain_risk_score", 0.0)
            
            # Store ENTRY-regime explicitly (for FARM_V1 hygiene checks)
            # Use SAME normalization logic as policy uses (for consistency)
            # CRITICAL: Compute vol_regime_entry from entry_bundle.features (same as policy uses)
            # This ensures consistency with policy filtering
            # Do NOT use trade.extra["atr_regime"] because it's set BEFORE policy filtering
            vol_regime_entry = None
            # Use entry_bundle.features (same as policy uses)
            feat_row = entry_bundle.features.iloc[-1] if hasattr(entry_bundle, 'features') and len(entry_bundle.features) > 0 else None
            if feat_row is not None:
                if "vol_regime" in entry_bundle.features.columns:
                    vol_regime_entry = str(feat_row["vol_regime"])
                elif "atr_regime" in entry_bundle.features.columns:
                    vol_regime_entry = str(feat_row["atr_regime"])
                elif "_v1_atr_regime_id" in entry_bundle.features.columns or "atr_regime_id" in entry_bundle.features.columns:
                    # Use SAME mapping as policy (ATR_ID_TO_VOL)
                    ATR_ID_TO_VOL = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
                    atr_id_col = "_v1_atr_regime_id" if "_v1_atr_regime_id" in entry_bundle.features.columns else "atr_regime_id"
                    atr_id = int(feat_row[atr_id_col])
                    vol_regime_entry = ATR_ID_TO_VOL.get(atr_id, "UNKNOWN")
            
            # Fallback: use atr_regime from trade.extra if features not available
            if vol_regime_entry is None and "atr_regime" in trade.extra:
                # NOTE: This might be HIGH/EXTREME even if policy filtered it out
                vol_regime_entry = str(trade.extra["atr_regime"])
            
            # Final fallback: use brain_vol_regime
            if vol_regime_entry is None:
                vol_regime_entry = policy_state.get("brain_vol_regime", "UNKNOWN")
            
            # HARD ASSERTS: FARM_V1 trades must be ASIA/LOW only, FARM_V2B permits MEDIUM
            if hasattr(self, "farm_v2_mode") and self.farm_v2_mode:
                allowed_regimes = {"LOW"}
                if hasattr(self, "farm_v2b_mode") and self.farm_v2b_mode:
                    allowed_regimes.add("MEDIUM")
                if vol_regime_entry not in allowed_regimes:
                    allowed_str = "/".join(sorted(allowed_regimes))
                    raise RuntimeError(
                        f"[FARM_ASSERT] Illegal vol_regime_entry when creating trade: {vol_regime_entry} "
                        f"(expected {allowed_str}). Policy should have filtered this out. "
                        f"atr_regime={trade.extra.get('atr_regime')}"
                    )
            elif hasattr(self, "farm_v1_mode") and self.farm_v1_mode:
                if vol_regime_entry != "LOW":
                    raise RuntimeError(
                        f"[FARM_ASSERT] Illegal vol_regime_entry when creating trade: {vol_regime_entry} (expected LOW for FARM_V1). "
                        f"Policy should have filtered this out. atr_regime={trade.extra.get('atr_regime')}"
                    )
            
            # CRITICAL: Store entry-regime fields ALWAYS (not in conditional block)
            # These MUST be set for every trade, regardless of policy or mode
            # Use vol_regime_entry (already computed from current_row)
            trade.extra["vol_regime_entry"] = vol_regime_entry
            trade.extra["vol_regime"] = vol_regime_entry  # Keep backward compatibility
            
            # CRITICAL: Store atr_regime (entry-regime) ALWAYS
            # For FARM_V1/V2, we MUST use vol_regime_entry (computed after policy filtering)
            # as the source of truth, NOT atr_regime_name which is computed before filtering.
            if (hasattr(self, "farm_v1_mode") and self.farm_v1_mode) or (hasattr(self, "farm_v2_mode") and self.farm_v2_mode):
                # For FARM_V1/V2, ALWAYS use vol_regime_entry (which should be LOW after policy filtering)
                # This OVERWRITES any atr_regime that was set earlier (before policy filtering)
                trade.extra["atr_regime"] = vol_regime_entry
                
                # CRITICAL: Store explicit FARM entry metadata (single source of truth)
                # Use feat_row (from entry_bundle.features) or current_row from policy_state
                # Use V2 metadata function for FARM_V2/V2B, V1 for FARM_V1
                if (hasattr(self, "farm_v2_mode") and self.farm_v2_mode) or (hasattr(self, "farm_v2b_mode") and self.farm_v2b_mode):
                    from gx1.policy.farm_guards import get_farm_entry_metadata_v2
                    # Get config from V2B or V2 (V2B takes priority)
                    policy_cfg = self.policy.get("entry_v9_policy_farm_v2b", {})
                    if not policy_cfg or not policy_cfg.get("enabled", False):
                        policy_cfg = self.policy.get("entry_v9_policy_farm_v2", {})
                    allow_medium_vol = policy_cfg.get("allow_medium_vol", True)
                    
                    row_for_metadata = None
                    if feat_row is not None:
                        row_for_metadata = feat_row
                    elif "_current_row" in policy_state:
                        # Use current_row from policy evaluation (if available)
                        current_row_df = policy_state["_current_row"]
                        if len(current_row_df) > 0:
                            row_for_metadata = current_row_df.iloc[0]
                    
                    if row_for_metadata is not None:
                        farm_metadata = get_farm_entry_metadata_v2(row_for_metadata, allow_medium_vol=allow_medium_vol)
                        trade.extra["farm_entry_session"] = farm_metadata["farm_entry_session"]
                        trade.extra["farm_entry_vol_regime"] = farm_metadata["farm_entry_vol_regime"]
                        trade.extra["farm_guard_version"] = farm_metadata["farm_guard_version"]
                    else:
                        # Fallback: use vol_regime_entry and session from policy_state
                        trade.extra["farm_entry_session"] = policy_state.get("session", "UNKNOWN")
                        trade.extra["farm_entry_vol_regime"] = vol_regime_entry
                        trade.extra["farm_guard_version"] = "FARM_V2_BRUTAL_V2"
                else:
                    from gx1.policy.farm_guards import get_farm_entry_metadata
                    row_for_metadata = None
                    if feat_row is not None:
                        row_for_metadata = feat_row
                    elif "_current_row" in policy_state:
                        # Use current_row from policy evaluation (if available)
                        current_row_df = policy_state["_current_row"]
                        if len(current_row_df) > 0:
                            row_for_metadata = current_row_df.iloc[0]
                    
                    if row_for_metadata is not None:
                        farm_metadata = get_farm_entry_metadata(row_for_metadata)
                        trade.extra["farm_entry_session"] = farm_metadata["farm_entry_session"]
                        trade.extra["farm_entry_vol_regime"] = farm_metadata["farm_entry_vol_regime"]
                        trade.extra["farm_guard_version"] = farm_metadata["farm_guard_version"]
                    else:
                        # Fallback: use vol_regime_entry and session from policy_state
                        trade.extra["farm_entry_session"] = policy_state.get("session", "UNKNOWN")
                        trade.extra["farm_entry_vol_regime"] = vol_regime_entry
                        trade.extra["farm_guard_version"] = "FARM_V1_BRUTAL_V1"
            else:
                # For non-FARM modes, use atr_regime_name if already set, otherwise vol_regime_entry
                if "atr_regime" not in trade.extra:
                    trade.extra["atr_regime"] = vol_regime_entry
                # If atr_regime already set, keep it (it's the entry-regime)
            
            # Store session tag (from policy_state, which was set at start of evaluate_entry)
            # This ensures session in trade.extra matches the session used by Stage-0, Big Brain V1, and routing
            session_tag_from_policy = policy_state.get("session")
            
            # CRITICAL: Store session ALWAYS (not in conditional block)
            # Infer session if not in policy_state
            if session_tag_from_policy and session_tag_from_policy in ("EU", "US", "OVERLAP", "ASIA"):
                session_entry_value = session_tag_from_policy
            else:
                # Fallback: infer session from entry_time if not in policy_state
                session_entry_value = infer_session_tag(trade.entry_time).upper()
            
            # ALWAYS set session (regardless of how we got it)
            # This is the source of truth for entry session_entry in CSV
            trade.extra["session"] = session_entry_value
            trade.extra["session_entry"] = session_entry_value  # Also store explicitly for backward compatibility
            
            # HARD ASSERT: FARM_V1 must only have ASIA session_entry
            if hasattr(self, "farm_v1_mode") and self.farm_v1_mode:
                if session_entry_value != "ASIA":
                    raise RuntimeError(
                        f"[FARM_V1_ASSERT] Illegal session_entry when creating trade: {session_entry_value} (expected ASIA). "
                        f"Policy should have filtered this out."
                    )
            
            # Store trade_id in extra for analysis
            trade.extra["trade_id"] = trade_id
            
            # Store FARM_V2/V2B meta-model predictions in trade.extra (for logging)
            if (hasattr(self, "farm_v2_mode") and self.farm_v2_mode) or (hasattr(self, "farm_v2b_mode") and self.farm_v2b_mode):
                p_long = policy_state.get("p_long")
                p_profitable = policy_state.get("p_profitable")
                if p_long is not None:
                    trade.extra["p_long"] = float(p_long)
                if p_profitable is not None:
                    trade.extra["p_profitable"] = float(p_profitable)
                # Set policy version based on mode
                if hasattr(self, "farm_v2b_mode") and self.farm_v2b_mode:
                    trade.extra["entry_policy_version"] = "FARM_V2B"
                    trade.extra["entry_policy_name"] = "FARM_V2B"
                else:
                    trade.extra["entry_policy_version"] = "FARM_V2"
                    trade.extra["entry_policy_name"] = "FARM_V2"
                
                # Sanity check: p_profitable must be present for FARM_V2
                if p_profitable is None:
                    raise RuntimeError(
                        f"FARM_V2 trade {trade_id} missing p_profitable in policy_state. "
                        f"Meta-model must compute p_profitable for all FARM_V2 trades."
                    )
            
            # FARM_V1/V2 DEBUGGING: Log ALL trade openings with session/vol_regime
            if (hasattr(self, "farm_v1_mode") and self.farm_v1_mode) or (hasattr(self, "farm_v2_mode") and self.farm_v2_mode):
                # Get active policy name from policy_state (set in evaluate_entry)
                policy_name = policy_state.get("policy_name", "UNKNOWN")
                # Fallback: check which FARM mode is active if not in policy_state
                if policy_name == "UNKNOWN":
                    use_farm_v2 = hasattr(self, "farm_v2_mode") and self.farm_v2_mode
                    use_farm_v1 = hasattr(self, "farm_v1_mode") and self.farm_v1_mode and not use_farm_v2
                    if use_farm_v2:
                        policy_name = "V9_FARM_V2"
                    elif use_farm_v1:
                        policy_name = "V9_FARM_V1"
                    else:
                        policy_name = policy_state.get("entry_model_active", "UNKNOWN")
                
                log.warning(
                    f"[FARM_OPEN] trade_id={trade_id} "
                    f"session={session_entry_value} vol_regime={vol_regime_entry} policy={policy_name}"
                )
                
                # BRUTAL ASSERT: FARM modes must only allow ASIA + allowed vol regimes
                allowed_policy_names = {"V9_FARM_V1", "V9_FARM_V2"}
                allowed_vol_regimes = {"LOW"}
                if hasattr(self, "farm_v2_mode") and self.farm_v2_mode:
                    allowed_policy_names.add("V9_FARM_V2B")
                    if hasattr(self, "farm_v2b_mode") and self.farm_v2b_mode:
                        allowed_vol_regimes.add("MEDIUM")
                assert policy_name in allowed_policy_names, f"[FARM_ASSERT] Unexpected policy in FARM mode: {policy_name} (trade_id={trade_id})"
                assert session_entry_value == "ASIA", f"[FARM_ASSERT] Illegal session at ENTRY: {session_entry_value} (expected ASIA, trade_id={trade_id})"
                assert vol_regime_entry in allowed_vol_regimes, (
                    f"[FARM_ASSERT] Illegal vol_regime at ENTRY: {vol_regime_entry} "
                    f"(expected {'/'.join(sorted(allowed_vol_regimes))}, trade_id={trade_id})"
                )
                # Debug log for ENTRY_V6 trades to verify session tagging
                if policy_state.get("entry_model_active") == "ENTRY_V6":
                    log.info("[DEBUG_SESSION] ENTRY_V6 trade created: session=%s entry_time=%s", 
                             session_tag_from_policy, trade.entry_time)
            else:
                # Fallback: infer from entry_time if not in policy_state or invalid (should not happen)
                inferred_session = infer_session_tag(trade.entry_time).upper()
                trade.extra["session"] = inferred_session
                log.warning("[DEBUG_SESSION] session not in policy_state or invalid (%s), inferred from entry_time: %s", 
                           session_tag_from_policy, inferred_session)
        else:
            # No policy_state available - store UNKNOWN for coverage tracking
            trade.extra["big_brain_v1"] = {
                "brain_trend_regime": "UNKNOWN",
                "brain_vol_regime": "UNKNOWN",
                "brain_risk_score": 0.0,
            }
            trade.extra["brain_trend_regime"] = "UNKNOWN"
            trade.extra["brain_vol_regime"] = "UNKNOWN"
            trade.extra["brain_risk_score"] = 0.0
            trade.extra["entry_model_active"] = "STANDARD"
            # Infer session from entry_time as fallback (should not happen in normal operation)
            inferred_session = infer_session_tag(trade.entry_time).upper()
            trade.extra["session"] = inferred_session
            trade.extra["session_entry"] = inferred_session  # Store entry session explicitly
            
            # Store entry vol_regime (fallback)
            if "atr_regime" in trade.extra:
                trade.extra["vol_regime_entry"] = trade.extra["atr_regime"]
            else:
                trade.extra["vol_regime_entry"] = "UNKNOWN"
            trade.extra["vol_regime"] = trade.extra["vol_regime_entry"]  # Keep backward compatibility
            
            log.warning("[DEBUG_SESSION] no policy_state available, inferred session from entry_time: %s", 
                        trade.extra["session"])
        
        return trade
    
