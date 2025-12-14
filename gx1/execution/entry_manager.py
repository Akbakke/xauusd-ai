from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from collections import deque

import logging
import numpy as np
import pandas as pd
import time

from gx1.execution.live_features import build_live_entry_features

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gx1.execution.oanda_demo_runner import GX1DemoRunner, LiveTrade


class EntryManager:
    def __init__(self, runner: "GX1DemoRunner", exit_config_name: Optional[str] = None) -> None:
        super().__setattr__("_runner", runner)
        # Explicit exit_config_name (injected from runner, no coupling to runner.policy)
        self.exit_config_name = exit_config_name
        # FARM_V2B diagnostic state (accumulated over replay)
        self.farm_diag = {
            "n_bars": 0,
            "n_raw_candidates": 0,
            "n_after_stage0": 0,
            "n_after_farm_regime": 0,
            "n_after_brutal_guard": 0,
            "n_after_policy_thresholds": 0,
            "p_long_values": [],
            "sessions": {},
            "atr_regimes": {},
            "farm_regimes": {},
        }
        self.cluster_guard_history = deque(maxlen=600)
        self.cluster_guard_atr_median: Optional[float] = None
        self.spread_history = deque(maxlen=600)

    def __getattr__(self, name: str):
        return getattr(self._runner, name)

    def __setattr__(self, name: str, value):
        if name == "_runner":
            super().__setattr__(name, value)
        else:
            setattr(self._runner, name, value)

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

    def evaluate_entry(self, candles: pd.DataFrame) -> Optional[LiveTrade]:
        entry_bundle = build_live_entry_features(candles)
        current_atr_bps: Optional[float] = None
        current_atr_pct: Optional[float] = None
        try:
            if entry_bundle.atr_bps is not None:
                current_atr_bps = float(entry_bundle.atr_bps)
        except (TypeError, ValueError):
            current_atr_bps = None
        if current_atr_bps is not None and current_atr_bps > 0:
            self.cluster_guard_history.append(current_atr_bps)
            if len(self.cluster_guard_history) >= 25:
                try:
                    self.cluster_guard_atr_median = float(np.median(self.cluster_guard_history))
                except Exception:
                    pass
            current_atr_pct = self._percentile_from_history(self.cluster_guard_history, current_atr_bps)
        
        # FARM_V2B diagnostic: increment bar counter
        self.farm_diag["n_bars"] += 1
        
        # Big Brain V1 Runtime: observe-only (adds to policy_state for logging/analysis)
        # CRITICAL: Always run inference if Big Brain V1 is enabled (loaded at startup)
        policy_state = {}
        
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
                            except (TypeError, ValueError):
                                atr_regime_id = "UNKNOWN"
                    elif "atr_regime_id" in entry_bundle.features.columns:
                        atr_regime_id_raw = feat_row.get("atr_regime_id")
                        if not pd.isna(atr_regime_id_raw):
                            try:
                                regime_val = int(atr_regime_id_raw)
                                mapping = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
                                atr_regime_id = mapping.get(regime_val, "UNKNOWN")
                            except (TypeError, ValueError):
                                atr_regime_id = "UNKNOWN"
                
                # Infer FARM regime
                farm_regime = infer_farm_regime(current_session, atr_regime_id)
                
                # FARM_V2B diagnostic: track FARM regime (after inference, but before Stage-0)
                # Note: This is counted even if Stage-0 blocks later, to see regime distribution
                if is_farm_v2b:
                    # Store for later use in raw_candidates tracking
                    policy_state["_atr_regime_id"] = atr_regime_id
                    policy_state["_farm_regime"] = farm_regime
                
                # Log once per run
                if not hasattr(self, "_farm_regime_logged"):
                    log.info(
                        "[BOOT] FARM_V2B mode detected: using FARM-only regime (session+ATR) instead of Big Brain. "
                        "First bar: session=%s atr_regime=%s -> farm_regime=%s",
                        current_session, atr_regime_id, farm_regime
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
            else:
                # Not FARM_V2B mode - set UNKNOWN as before
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
            current_ts = candles.index[-1]
            current_session = _infer_session_tag(current_ts).upper()
            policy_state["session"] = current_session
        else:
            current_session = policy_state["session"]
        
        # For FARM_V2B mode with FARM regime, don't block solely on UNKNOWN
        # The brutal guard will handle further filtering
        is_farm_v2b_with_regime = (
            policy_state.get("farm_regime") in ("FARM_ASIA_LOW", "FARM_ASIA_MEDIUM")
        )
        
        # Stage-0 filter: skip entry consideration if regime/session is not promising
        # Check if Stage-0 is enabled (config-styrt, default=True for backward compatibility)
        stage0_enabled = getattr(self, "stage0_enabled", True)
        
        # For FARM_V2B with valid FARM regime, don't block on Stage-0
        # The brutal guard and FARM_V2B policy will handle filtering
        if is_farm_v2b_with_regime:
            # FARM_V2B with valid regime: skip Stage-0 blocking, let brutal guard handle it
            log.debug(
                "[STAGE_0] FARM_V2B mode with valid regime (%s): skipping Stage-0 filter, "
                "brutal guard will handle filtering",
                policy_state.get("farm_regime")
            )
            # FARM_V2B diagnostic: Stage-0 passed (skipped for valid regime)
            if is_farm_v2b:
                self.farm_diag["n_after_stage0"] += 1
        elif stage0_enabled and not self.should_consider_entry(trend=trend, vol=vol, session=current_session, risk_score=risk_score):
            log.info(
                "[STAGE_0] Skip entry consideration: trend=%s vol=%s session=%s risk=%.3f",
                trend,
                vol,
                current_session,
                risk_score,
            )
            # Store stage-0 skip in policy_state for logging (if needed)
            policy_state["stage_0_skip"] = {
                "trend": trend,
                "vol": vol,
                "session": current_session,
                "risk_score": float(risk_score),
            }
            self._last_policy_state = policy_state
            # FARM_V2B diagnostic: Stage-0 blocked (only if not FARM_V2B with valid regime)
            if is_farm_v2b and not is_farm_v2b_with_regime:
                # Stage-0 blocked for FARM_V2B (but not counted in n_after_stage0)
                pass
            return None  # Skip XGB/TCN inference entirely
        
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
        
        # Initialize entry_model_active
        policy_state["entry_model_active"] = "ENTRY_V9"
        
        # ============================================================
        # ENTRY_V9 ONLY - No fallback to V6/V8
        # ============================================================
        # V9 gets ALL data (all regimes) - only policy filters coverage
        if not self.entry_v9_enabled or self.entry_v9_model is None:
            log.error("[ENTRY_V9] V9 is REQUIRED but not loaded. No entry possible.")
            return None
        
        # V9 is always used (no regime filtering at model level)
        log.debug(
            "[ENTRY_V9] Evaluating in regime trend=%s vol=%s session=%s",
                brain_trend_regime, brain_vol_regime, current_session
            )
        
        v9_pred = self._predict_entry_v9(entry_bundle, candles, policy_state)
        if v9_pred is None:
            log.warning("[ENTRY_V9] v9_pred=None - no entry this bar")
            return None
        
        # FARM_V2B diagnostic: raw candidate (V9 gave a prediction)
        if is_farm_v2b:
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
        
        # Apply ENTRY_V9_POLICY (FARM_V2B, FARM_V2, FARM_V1, BASE_V1, or V1) if enabled
        # Check in priority order: FARM_V2B > FARM_V2 > FARM_V1 > BASE_V1 > V1
        policy_farm_v2b_cfg = self.policy.get("entry_v9_policy_farm_v2b", {})
        policy_farm_v2_cfg = self.policy.get("entry_v9_policy_farm_v2", {})
        policy_farm_v1_cfg = self.policy.get("entry_v9_policy_farm_v1", {})
        policy_base_v1_cfg = self.policy.get("entry_v9_policy_base_v1", {})
        policy_v1_cfg = self.policy.get("entry_v9_policy_v1", {})
        
        use_farm_v2b = policy_farm_v2b_cfg.get("enabled", False)
        use_farm_v2 = policy_farm_v2_cfg.get("enabled", False) and not use_farm_v2b
        use_farm_v1 = policy_farm_v1_cfg.get("enabled", False) and not use_farm_v2b and not use_farm_v2
        use_base_v1 = policy_base_v1_cfg.get("enabled", False) and not use_farm_v2b and not use_farm_v2 and not use_farm_v1
        use_v1 = policy_v1_cfg.get("enabled", False) and not use_farm_v2b and not use_farm_v2 and not use_farm_v1 and not use_base_v1
        
        if use_farm_v2b:
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
        
        if policy_cfg is not None and policy_cfg.get("enabled", False):
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
                    log.debug("[ENTRY_V9] Could not add Big Brain regime columns: %s", e)
            
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
                    from gx1.execution.live_features import infer_session_tag
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
                if use_farm_v2b:
                    policy_name = "V9_FARM_V2B"
                elif use_farm_v2:
                    policy_name = "V9_FARM_V2"
                elif use_farm_v1:
                    policy_name = "V9_FARM_V1"
                elif use_base_v1:
                    policy_name = "POLICY_BASE_V1"
                else:
                    policy_name = "POLICY_V1_V9"
                log.debug("[ENTRY] Using ENTRY_V9 prediction (%s approved)", policy_name)
                # Store policy_name and current_row for later use in trade creation
                policy_state["policy_name"] = policy_name
                policy_state["_current_row"] = current_row
            else:
                # Policy rejected - no entry
                if use_farm_v2:
                    policy_name = "POLICY_FARM_V2"
                elif use_farm_v1:
                    policy_name = "POLICY_FARM_V1"
                elif use_base_v1:
                    policy_name = "POLICY_BASE_V1"
                else:
                    policy_name = "POLICY_V1_V9"
                log.debug("[ENTRY] ENTRY_V9 prediction rejected by %s (prob_long=%.4f)", policy_name, v9_pred.prob_long)
                return None  # No entry this bar
        else:
            # Policy disabled - use V9 prediction directly (RAW_ENTRY_V9 mode)
            prediction = v9_pred
            policy_state["entry_model_active"] = "ENTRY_V9"
            log.debug("[ENTRY] Using ENTRY_V9 prediction (POLICY_V1_V9 disabled - RAW mode)")
        
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
                        return None

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
        # Use unique sequential trade_id for FARM_V1 debugging
        if not hasattr(self, "_next_trade_id"):
            self._next_trade_id = 0
        self._next_trade_id += 1
        trade_id = f"SIM-{int(time.time())}-{self._next_trade_id:06d}"
        
        # FARM_V2B diagnostic: trade actually created (final step)
        if is_farm_v2b:
            self.farm_diag["n_after_policy_thresholds"] += 1
        
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
        
        # Lazy import to avoid circular dependency
        from gx1.execution.oanda_demo_runner import LiveTrade
        
        trade = LiveTrade(
            trade_id=trade_id,
            entry_time=now_ts,
            side=side,
            units=self.exec.default_units if side == "long" else -self.exec.default_units,
            entry_price=entry_price,
            entry_bid=entry_bid_price,
            entry_ask=entry_ask_price,
            atr_bps=entry_bundle.atr_bps,
            vol_bucket=entry_bundle.vol_bucket,
            entry_prob_long=prediction.prob_long,
            entry_prob_short=prediction.prob_short,
            dry_run=self.exec.dry_run,
        )
        
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

        policy_state_snapshot = self._last_policy_state or {}

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
    
