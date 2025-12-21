from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import logging
import pandas as pd

from gx1.execution.live_features import build_live_exit_snapshot, infer_session_tag, compute_atr_bps
from gx1.utils.pnl import compute_pnl_bps

if TYPE_CHECKING:
    from gx1.execution.oanda_demo_runner import GX1DemoRunner

log = logging.getLogger(__name__)
_prepare_exit_features_fn = None


def _get_prepare_exit_features():
    global _prepare_exit_features_fn
    if _prepare_exit_features_fn is None:
        from gx1.execution.oanda_demo_runner import prepare_exit_features as _fn

        _prepare_exit_features_fn = _fn
    return _prepare_exit_features_fn

class ExitManager:
    def __init__(self, runner: "GX1DemoRunner") -> None:
        super().__setattr__("_runner", runner)

    def __getattr__(self, name: str):
        return getattr(self._runner, name)

    def __setattr__(self, name: str, value):
        if name == "_runner":
            super().__setattr__(name, value)
        else:
            setattr(self._runner, name, value)

    def evaluate_and_close_trades(self, candles: pd.DataFrame) -> None:
        # Skip exit evaluation in ENTRY_ONLY mode (no trades to close)
        if self.mode == "ENTRY_ONLY":
            return
        
        # FARM_V1 mode: Skip tick-exit evaluation entirely
        if hasattr(self, "farm_v1_mode") and self.farm_v1_mode:
            # In FARM_V1 mode, only FARM exits are allowed - skip all tick-exit logic
            pass  # Continue to FARM_V1 exit evaluation below
        elif self.tick_cfg.get("enabled", False) and not self.replay_mode:
            # Normal mode: tick-exit is handled by TickWatcher thread (live mode only)
            # In replay mode, tick-exit is evaluated here if enabled
            pass
        
        # Get snapshot of open trades
        if not self.open_trades:
            return
        open_trades_copy = list(self.open_trades)
        
        self._ensure_bid_ask_columns(candles, context="exit_manager")
        now_ts = candles.index[-1]
        current_bid = float(candles["bid_close"].iloc[-1])
        current_ask = float(candles["ask_close"].iloc[-1])
        runtime_atr_bps = self._compute_runtime_atr_bps(candles)
        if self.exit_verbose_logging:
            log.info(
                "[EXIT] Evaluating %d open trades on bar %s (replay=%s)",
                len(open_trades_copy),
                now_ts,
                self.replay_mode,
            )
        else:
            log.debug(
                "[EXIT] Evaluating %d open trades on bar %s (replay=%s)",
                len(open_trades_copy),
                now_ts,
                self.replay_mode,
            )
        closes_requested = 0
        closes_accepted = 0
        
        # Collect price trace for open trades (for intratrade metrics)
        self._collect_price_trace(candles, now_ts, open_trades_copy)

        # FARM_V1 mode: Skip tick-exit evaluation, go straight to FARM_V1 exits
        if hasattr(self, "farm_v1_mode") and self.farm_v1_mode:
            # Skip all tick-exit logic - only FARM_V1 exits allowed
            pass

        # Check if EXIT_FARM_V2_RULES is enabled
        if getattr(self, "exit_farm_v2_rules_factory", None):
            # Use EXIT_FARM_V2_RULES bar-based exit logic
            for trade in open_trades_copy:
                # Thread-safe check: skip if trade was already closed by tick-exit
                if trade not in self.open_trades:
                    continue
                
                # EXIT_FARM_V2_RULES only supports LONG positions
                if trade.side != "long":
                    log.debug(
                        "[EXIT_FARM_V2_RULES] Skipping SHORT trade %s (EXIT_FARM_V2_RULES only supports LONG)",
                        trade.trade_id
                    )
                    continue
                delta_minutes = (now_ts - trade.entry_time).total_seconds() / 60.0
                est_bars = max(1, int(round(delta_minutes / 5.0)))
                entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
                entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
                est_pnl = compute_pnl_bps(entry_bid, entry_ask, current_bid, current_ask, trade.side)
                log.debug(
                    "[EXIT] Trade %s profile=%s bars=%d pnl=%.2f -> evaluate FARM_V2_RULES",
                    trade.trade_id,
                    trade.extra.get("exit_profile"),
                    est_bars,
                    est_pnl,
                )
                
                policy = self.exit_farm_v2_rules_states.get(trade.trade_id)
                if policy is None:
                    try:
                        self._init_farm_v2_rules_state(trade, context="exit_manager")
                        policy = self.exit_farm_v2_rules_states.get(trade.trade_id)
                    except Exception as exc:
                        log.error(
                            "[EXIT_FARM_V2_RULES] Failed to initialize state for trade %s: %s",
                            trade.trade_id,
                            exc,
                        )
                        continue

                # Check for exit on current bar
                if len(candles) == 0:
                    continue
                exit_decision = policy.on_bar(
                    price_bid=current_bid,
                    price_ask=current_ask,
                    ts=now_ts,
                    atr_bps=runtime_atr_bps,
                )
                
                if exit_decision is None:
                    log.debug(
                        "[EXIT] Trade %s profile=%s bars=%d pnl=%.2f -> HOLD (farm rules)",
                        trade.trade_id,
                        trade.extra.get("exit_profile"),
                        est_bars,
                        est_pnl,
                    )
                    continue
                
                if exit_decision is not None:
                    # Exit triggered
                    exit_reason = exit_decision.reason
                    pnl_bps = exit_decision.pnl_bps
                    bars_in_trade = exit_decision.bars_held
                    
                    # Log exit triggered to trade journal
                    if hasattr(self._runner, "trade_journal") and self._runner.trade_journal:
                        try:
                            from gx1.monitoring.trade_journal import EVENT_EXIT_TRIGGERED
                            
                            exit_state = self.exit_farm_v2_rules_states.get(trade.trade_id)
                            exit_state_fields = {}
                            if exit_state and hasattr(exit_state, "trail_level"):
                                exit_state_fields["trail_level"] = getattr(exit_state, "trail_level", None)
                            if exit_state and hasattr(exit_state, "tp_level"):
                                exit_state_fields["tp_level"] = getattr(exit_state, "tp_level", None)
                            if exit_state and hasattr(exit_state, "sl_level"):
                                exit_state_fields["sl_level"] = getattr(exit_state, "sl_level", None)
                            
                            # Log exit event (structured)
                            exit_time_iso = now_ts.isoformat() if hasattr(now_ts, "isoformat") else str(now_ts)
                            self._runner.trade_journal.log_exit_event(
                                trade_id=trade.trade_id,
                                timestamp=exit_time_iso,
                                event_type="EXIT_TRIGGERED",
                                price=exit_decision.exit_price,
                                pnl_bps=pnl_bps,
                                bars_held=bars_in_trade,
                            )
                            
                            # Log exit event (backward compatibility JSONL)
                            self._runner.trade_journal.log(
                                EVENT_EXIT_TRIGGERED,
                                {
                                    "exit_time": exit_time_iso,
                                    "exit_price": exit_decision.exit_price,
                                    "exit_profile": trade.extra.get("exit_profile", "UNKNOWN"),
                                    "exit_reason": exit_reason,
                                    "bars_held": bars_in_trade,
                                    "pnl_bps": pnl_bps,
                                    "exit_state_fields": exit_state_fields,
                                },
                                trade_key={
                                    "entry_time": trade.entry_time.isoformat() if hasattr(trade.entry_time, "isoformat") else str(trade.entry_time),
                                    "entry_price": trade.entry_price,
                                    "side": trade.side,
                                },
                                trade_id=trade.trade_id,
                            )
                        except Exception as e:
                            log.warning("[TRADE_JOURNAL] Failed to log EXIT_TRIGGERED: %s", e)
                    
                    # Request close
                    accepted = self.request_close(
                        trade_id=trade.trade_id,
                        source="EXIT_FARM_V2_RULES",
                        reason=exit_reason,
                        px=exit_decision.exit_price,
                        pnl_bps=pnl_bps,
                        bars_in_trade=bars_in_trade,
                    )
                    closes_requested += 1
                    
                    if not accepted:
                        log.warning("[EXIT] close rejected by ExitArbiter for trade %s", trade.trade_id)
                        log.error("Exit signaled but not applied for trade %s", trade.trade_id)
                        continue
                    closes_accepted += 1
                    
                    # Remove from open_trades
                    if trade in self.open_trades:
                        self.open_trades.remove(trade)
                    self._teardown_exit_state(trade.trade_id)
                    
                    # Record realized PnL
                    self.record_realized_pnl(now_ts, pnl_bps)
                    
                    # Log trade closure
                    log.info(
                        "[LIVE] CLOSED TRADE (FARM_V2_RULES) %s %s @ %.3f | pnl=%.1f bps | reason=%s | bars=%d",
                        trade.side.upper(),
                        trade.trade_id,
                        exit_decision.exit_price,
                        pnl_bps,
                        exit_reason,
                        bars_in_trade,
                    )
                    
                    # Log trade closed to trade journal (with intratrade metrics)
                    self._log_trade_close_with_metrics(
                        trade=trade,
                        exit_time=now_ts,
                        exit_price=exit_decision.exit_price,
                        exit_reason=exit_reason,
                        realized_pnl_bps=pnl_bps,
                        bars_held=bars_in_trade,
                    )
                    
                    # Record exit in trade log
                    self._update_trade_log_on_close(
                        trade.trade_id,
                        exit_decision.exit_price,
                        pnl_bps,
                        exit_reason,
                        now_ts,
                        bars_in_trade=bars_in_trade,
                    )
            
            # If FARM_V2_RULES mode is active, return early (no other exits allowed)
            if self.exit_only_v2_drift:
                return
        
        if hasattr(self, "exit_fixed_bar_policy") and self.exit_fixed_bar_policy is not None:
            if len(candles) == 0:
                return
            for trade in open_trades_copy:
                if trade not in self.open_trades or trade.side != "long":
                    continue
                entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
                entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
                est_pnl = compute_pnl_bps(entry_bid, entry_ask, current_bid, current_ask, trade.side)
                delta_minutes = (now_ts - trade.entry_time).total_seconds() / 60.0
                est_bars = max(1, int(round(delta_minutes / 5.0)))
                if not getattr(trade, "extra", None):
                    trade.extra = {}
                if (
                    not trade.extra.get("fixed_bar_exit_initialized")
                    or not self.exit_fixed_bar_policy.has_state(trade.trade_id)
                ):
                    self.exit_fixed_bar_policy.reset_on_entry(
                        trade.entry_bid,
                        trade.entry_ask,
                        trade.trade_id,
                        trade.side,
                    )
                    trade.extra["fixed_bar_exit_initialized"] = True
                decision = self.exit_fixed_bar_policy.on_bar(
                    trade.trade_id,
                    price_bid=current_bid,
                    price_ask=current_ask,
                    side=trade.side,
                )
                if decision is None:
                    log.debug(
                        "[EXIT] Trade %s profile=%s bars=%d pnl=%.2f -> HOLD (fixed bar)",
                        trade.trade_id,
                        trade.extra.get("exit_profile"),
                        est_bars,
                        est_pnl,
                    )
                    continue
                accepted = self.request_close(
                    trade_id=trade.trade_id,
                    source="FIXED_BAR_CLOSE",
                    reason=decision.reason,
                    px=decision.exit_price,
                    pnl_bps=decision.pnl_bps,
                    bars_in_trade=decision.bars_held,
                )
                closes_requested += 1
                if not accepted:
                    log.error("Exit signaled but not applied for trade %s", trade.trade_id)
                    continue
                closes_accepted += 1
                if trade in self.open_trades:
                    self.open_trades.remove(trade)
                self._teardown_exit_state(trade.trade_id)
                self.record_realized_pnl(now_ts, decision.pnl_bps)
                
                # Log trade closed to trade journal (with intratrade metrics)
                self._log_trade_close_with_metrics(
                    trade=trade,
                    exit_time=now_ts,
                    exit_price=decision.exit_price,
                    exit_reason=decision.reason,
                    realized_pnl_bps=decision.pnl_bps,
                    bars_held=decision.bars_held,
                )
                
                self._update_trade_log_on_close(
                    trade.trade_id,
                    decision.exit_price,
                    decision.pnl_bps,
                    decision.reason,
                    now_ts,
                    bars_in_trade=decision.bars_held,
                )
            if self.exit_only_v2_drift:
                return
        
        # Check if EXIT_FARM_V1 is enabled
        if hasattr(self, "exit_farm_v1_policy") and self.exit_farm_v1_policy is not None:
            # Use EXIT_FARM_V1 bar-based exit logic
            for trade in open_trades_copy:
                # Thread-safe check: skip if trade was already closed by tick-exit
                if trade not in self.open_trades:
                    continue
                
                # EXIT_FARM_V1 only supports LONG positions
                if trade.side != "long":
                    log.debug(
                        "[EXIT_FARM_V1] Skipping SHORT trade %s (EXIT_FARM_V1 only supports LONG)",
                        trade.trade_id,
                    )
                    continue
                
                # Initialize exit policy for this trade if not already done
                if "exit_farm_v1_initialized" not in trade.extra:
                    # GUARDRAIL: FARM_V2 entry MUST use FARM_EXIT_V1_STABLE exit
                    if hasattr(self, "farm_v2_mode") and self.farm_v2_mode:
                        if not hasattr(self, "exit_farm_v1_policy") or self.exit_farm_v1_policy is None:
                            raise RuntimeError(
                                "[FARM_EXIT_GUARDRAIL] FARM exit policy not initialized. This should never happen."
                            )
                        
                        # Determine expected exit_profile based on FARM mode
                        if hasattr(self, "farm_v2_mode") and self.farm_v2_mode:
                            expected_exit_profile = "FARM_EXIT_V2_AGGRO"
                            log.info(
                                "[FARM_V2_GUARDRAIL] FARM_V2 entry detected - enforcing exit_profile=%s",
                                expected_exit_profile
                            )
                        elif hasattr(self, "farm_v1_mode") and self.farm_v1_mode:
                            expected_exit_profile = "FARM_EXIT_V1_STABLE"
                            log.info(
                                "[FARM_V1_GUARDRAIL] FARM_V1 entry detected - enforcing exit_profile=%s",
                                expected_exit_profile
                            )
                        else:
                            # Fallback: use FARM_EXIT_V1_STABLE for backward compatibility
                            expected_exit_profile = "FARM_EXIT_V1_STABLE"
                            log.warning(
                                "[FARM_EXIT_GUARDRAIL] Unknown FARM mode - defaulting to exit_profile=%s",
                                expected_exit_profile
                            )
                    
                    self.exit_farm_v1_policy.reset_on_entry(
                        trade.entry_price,
                        trade.entry_time,
                        side="long",  # Always "long" for EXIT_FARM_V1
                    )
                    trade.extra["exit_farm_v1_initialized"] = True
                    # Set exit_profile for traceability
                    trade.extra["exit_profile"] = expected_exit_profile
                    # Set baseline version for traceability
                    if hasattr(self, "farm_baseline_version"):
                        trade.extra["farm_baseline_version"] = self.farm_baseline_version
                    
                    # GUARDRAIL: Final check - reject if exit_profile doesn't match mode
                    if hasattr(self, "farm_v2_mode") and self.farm_v2_mode:
                        if trade.extra.get("exit_profile") != "FARM_EXIT_V2_AGGRO":
                            raise RuntimeError(
                                f"[FARM_V2_GUARDRAIL] FARM_V2 entry requires exit_profile='FARM_EXIT_V2_AGGRO'. "
                                f"Found: {trade.extra.get('exit_profile')}. "
                                f"FARM_V2 must use FARM_EXIT_V2_AGGRO."
                            )
                    elif hasattr(self, "farm_v1_mode") and self.farm_v1_mode:
                        if trade.extra.get("exit_profile") != "FARM_EXIT_V1_STABLE":
                            raise RuntimeError(
                                f"[FARM_V1_GUARDRAIL] FARM_V1 entry requires exit_profile='FARM_EXIT_V1_STABLE'. "
                                f"Found: {trade.extra.get('exit_profile')}. "
                                f"FARM_V1 must use FARM_EXIT_V1_STABLE."
                            )
                
                # Evaluate exit on current bar
                exit_decision = self.exit_farm_v1_policy.on_bar(current_bid, now_ts)
                
                if exit_decision is not None:
                    # Exit triggered
                    reason_map = {
                        "EXIT_FARM_SL": "EXIT_FARM_SL",
                        "EXIT_FARM_SL_BREAKEVEN": "EXIT_FARM_SL_BREAKEVEN",
                        "EXIT_FARM_TP": "EXIT_FARM_TP",
                        "EXIT_FARM_TIMEOUT": "EXIT_FARM_TIMEOUT",
                    }
                    exit_reason = reason_map.get(exit_decision.reason, exit_decision.reason)
                    
                    log.info(
                        "[EXIT_FARM_V1] Trade %s: %s triggered at bar %d, price=%.5f, pnl_bps=%.2f",
                        trade.trade_id,
                        exit_decision.reason,
                        exit_decision.bars_held,
                        exit_decision.exit_price,
                        exit_decision.pnl_bps,
                    )
                    
                    # HARD ASSERT: FARM_V1 mode must only use FARM exit reasons
                    if self.farm_v1_mode and not exit_reason.startswith("EXIT_FARM"):
                        raise RuntimeError(
                            f"[FARM_V1_ASSERT] Non-FARM exit reason in FARM_V1 mode: {exit_reason}. "
                            f"Trade: {trade.trade_id}"
                        )
                    
                    # Request close via ExitArbiter
                    accepted = self.request_close(
                        trade_id=trade.trade_id,
                        source="EXIT_FARM_V1",
                        reason=exit_reason,
                        px=exit_decision.exit_price,
                        pnl_bps=exit_decision.pnl_bps,
                        bars_in_trade=exit_decision.bars_held,
                    )
                    closes_requested += 1
                    
                    if accepted and trade in self.open_trades:
                        self.open_trades.remove(trade)
                        if not self.replay_mode:
                            self._maybe_update_tick_watcher()
                        closes_accepted += 1
                        
                        # Log trade closed to trade journal (with intratrade metrics)
                        self._log_trade_close_with_metrics(
                            trade=trade,
                            exit_time=now_ts,
                            exit_price=exit_decision.exit_price,
                            exit_reason=exit_reason,
                            realized_pnl_bps=exit_decision.pnl_bps,
                            bars_held=exit_decision.bars_held,
                        )
                    elif not accepted:
                        log.error("Exit signaled but not applied for trade %s", trade.trade_id)
                    continue  # Continue to next trade, do not fall through
            
            # If FARM_V1 mode is active, return early (no other exits allowed)
            if self.farm_v1_mode:
                return
            
            # If only_v2_drift_mode is active (or equivalent for FARM_V1), return early
            if self.exit_only_v2_drift:
                return

        # Fallback to original exit model logic
        for trade in open_trades_copy:
            # Thread-safe check: skip if trade was already closed by tick-exit
            if trade not in self.open_trades:
                continue  # Trade was closed by tick-exit, skip
            
            # CRITICAL: Minimum hold time guard (prevent exit model from triggering immediately after entry)
            # Exit model is not calibrated for bars_in_trade=1 (immediate exit)
            # Require at least 2 bars (10 minutes) before evaluating exit model
            min_hold_bars = 2  # Minimum bars before exit model evaluation
            delta_minutes = (now_ts - trade.entry_time).total_seconds() / 60.0
            bars_in_trade_min = int(round(delta_minutes / 5.0))  # M5 = 5 minutes per bar
            
            if bars_in_trade_min < min_hold_bars:
                # Skip exit model evaluation if trade is too new (prevent immediate exit)
                log.debug(
                    "[EXIT] Skipping exit model evaluation for trade %s: bars_in_trade=%d < min_hold_bars=%d (trade too new)",
                    trade.trade_id,
                    bars_in_trade_min,
                    min_hold_bars,
                )
                continue  # Skip exit evaluation for new trades
            
            snapshot = build_live_exit_snapshot(
                {
                    "trade_id": trade.trade_id,
                    "entry_time": trade.entry_time,
                    "entry_price": trade.entry_price,
                    "side": trade.side,
                    "units": trade.units,
                    "cost_bps": 0.0,
                },
                candles,
            )
            prepare_exit_features = _get_prepare_exit_features()
            features = prepare_exit_features(snapshot, self.exit_bundle.feature_names)
            prob_close = float(self.exit_bundle.model.predict_proba(features.to_numpy())[0, 1])

            # Exit model hysteresis: track prob_close history and require consecutive bars above threshold
            trade.prob_close_history.append(prob_close)
            
            # Log exit model evaluation (for debugging)
            log.debug(
                "[EXIT] Trade %s: bars_in_trade=%d prob_close=%.4f threshold=%.4f history=%s (require_consecutive=%d)",
                trade.trade_id,
                bars_in_trade_min,
                prob_close,
                self.exit_threshold,
                list(trade.prob_close_history),
                self.exit_require_consecutive,
            )
            
            # Check if prob_close >= threshold in required consecutive bars
            # Only exit if ALL last N bars have prob_close >= threshold (hysteresis)
            should_exit = False
            if len(trade.prob_close_history) >= self.exit_require_consecutive:
                # Check if all last N bars have prob_close >= threshold
                recent_bars = list(trade.prob_close_history)[-self.exit_require_consecutive:]
                should_exit = all(p >= self.exit_threshold for p in recent_bars)
                
                if should_exit:
                    log.info(
                        "[EXIT] Trade %s: prob_close >= threshold in %d consecutive bars: %s (hysteresis passed)",
                        trade.trade_id,
                        self.exit_require_consecutive,
                        [f"{p:.4f}" for p in recent_bars],
                    )
                else:
                    log.debug(
                        "[EXIT] Trade %s: prob_close < threshold in some bars: %s (hysteresis: need %d consecutive)",
                        trade.trade_id,
                        [f"{p:.4f}" for p in recent_bars],
                        self.exit_require_consecutive,
                    )
            else:
                # Not enough bars yet, wait for more
                log.debug(
                    "[EXIT] Trade %s: not enough bars for hysteresis check: %d < %d",
                    trade.trade_id,
                    len(trade.prob_close_history),
                    self.exit_require_consecutive,
                )

            if should_exit:
                # Thread-safe check: skip if trade was already closed by tick-exit
                if trade not in self.open_trades:
                    continue  # Trade was closed by tick-exit, skip
                entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
                entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
                # Add defensive logging for first N calls
                if not hasattr(self, "_exit_mgr_pnl_log_count"):
                    self._exit_mgr_pnl_log_count = 0
                if self._exit_mgr_pnl_log_count < 5:
                    log.debug(
                        "[PNL] ExitManager PnL: entry_bid=%.5f entry_ask=%.5f exit_bid=%.5f exit_ask=%.5f side=%s",
                        entry_bid, entry_ask, current_bid, current_ask, trade.side
                    )
                    self._exit_mgr_pnl_log_count += 1
                pnl_bps = compute_pnl_bps(entry_bid, entry_ask, current_bid, current_ask, trade.side)
                mark_price = current_bid if trade.side == "long" else current_ask
                pnl_currency = pnl_bps / 10000.0 * mark_price * (abs(trade.units) / 100.0)
                
                # Calculate bars_in_trade for ExitArbiter
                delta_minutes = (now_ts - trade.entry_time).total_seconds() / 60.0
                bars_in_trade = int(round(delta_minutes / 5.0))  # M5 = 5 minutes per bar
                
                # Log exit model propose close
                log.info("[EXIT] propose close reason=THRESHOLD pnl=%.1f bars_in_trade=%d", pnl_bps, bars_in_trade)
                
                # Request close via ExitArbiter
                accepted = self.request_close(
                    trade_id=trade.trade_id,
                    source="MODEL_EXIT",
                    reason="THRESHOLD",
                    px=mark_price,
                    pnl_bps=pnl_bps,
                    bars_in_trade=bars_in_trade,
                )
                closes_requested += 1
                
                if not accepted:
                    log.warning("[EXIT] close rejected by ExitArbiter for trade %s", trade.trade_id)
                    log.error("Exit signaled but not applied for trade %s", trade.trade_id)
                    continue  # Exit rejected, keep trade open
                closes_accepted += 1
                
                # Remove from open_trades (after successful close)
                if trade in self.open_trades:
                    self.open_trades.remove(trade)
                
                # Record realized PnL (after successful close)
                self.record_realized_pnl(now_ts, pnl_bps)
                
                # Log trade closed to trade journal (with intratrade metrics)
                self._log_trade_close_with_metrics(
                    trade=trade,
                    exit_time=now_ts,
                    exit_price=mark_price,
                    exit_reason="THRESHOLD",
                    realized_pnl_bps=pnl_bps,
                    bars_held=bars_in_trade,
                )

                # Get session for trade (from entry_time)
                entry_session = infer_session_tag(trade.entry_time)
                session_key = self._resolve_session_key(entry_session)
                
                # Calculate entry p_hat from probabilities
                entry_p_hat = max(trade.entry_prob_long, trade.entry_prob_short)
                
                # Convert timestamps to UTC (needed for both telemetry and trade log)
                now_ts_utc = now_ts.tz_convert("UTC") if now_ts.tzinfo else pd.Timestamp(now_ts, tz="UTC")
                entry_ts_utc = trade.entry_time.tz_convert("UTC") if trade.entry_time.tzinfo else pd.Timestamp(trade.entry_time, tz="UTC")
                
                # Record closed trade to telemetry tracker (for delayed ECE) - only if enabled
                if hasattr(self, "telemetry_tracker") and self.telemetry_tracker is not None:
                    self.telemetry_tracker.record_closed_trade(
                        entry_id=trade.trade_id,
                        session=session_key,
                        entry_time=entry_ts_utc,
                        exit_time=now_ts_utc,
                        p_hat=entry_p_hat,
                        side=trade.side,
                        pnl_bps=pnl_bps,
                    )
                    
                    # Record exit latency for EXIT_V2 (real exit)
                    hold_time_s = (now_ts_utc - entry_ts_utc).total_seconds()
                    try:
                        self.telemetry_tracker.record_exit_latency(session_key, "EXIT_V2", hold_time_s)
                    except Exception as e:
                        log.warning("[EXIT] Failed to record exit latency in telemetry: %s", e)
                
                # Log trade closure
                log.info("[LIVE] CLOSED TRADE (BAR) %s %s @ %.3f | pnl=%.1f bps", trade.side.upper(), trade.trade_id, mark_price, pnl_bps)

                # Append to trade log (only after successful close)
                # Build trade log row with FARM fields from trade.extra
                trade_extra = trade.extra if hasattr(trade, "extra") and trade.extra else {}
                trade_log_row = {
                    "trade_id": trade.trade_id,
                    "entry_time": trade.entry_time.isoformat(),
                    "entry_price": f"{trade.entry_price:.3f}",
                    "side": trade.side,
                    "units": trade.units,
                    "exit_time": now_ts_utc.isoformat(),
                    "exit_price": f"{mark_price:.3f}",
                    "pnl_bps": f"{pnl_bps:.2f}",
                    "pnl_currency": f"{pnl_currency:.2f}",
                    "entry_prob_long": f"{trade.entry_prob_long:.4f}",
                    "entry_prob_short": f"{trade.entry_prob_short:.4f}",
                    "exit_prob_close": f"{prob_close:.4f}",
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
                
                # Extract FARM_V2/V2B meta-model predictions (entry_p_long, entry_p_profitable, entry_policy_version)
                if (hasattr(self, "farm_v2_mode") and self.farm_v2_mode) or (hasattr(self, "farm_v2b_mode") and self.farm_v2b_mode):
                    # Get p_long from trade.extra or policy_state
                    p_long = trade_extra.get("p_long")
                    if p_long is None:
                        p_long = trade.entry_prob_long  # Fallback to entry_prob_long
                    trade_log_row["entry_p_long"] = f"{p_long:.4f}" if p_long is not None else ""
                    
                    # Get p_profitable from trade.extra (should be set by policy)
                    p_profitable = trade_extra.get("p_profitable")
                    if p_profitable is None:
                        log.warning(f"[TRADE_LOG] FARM_V2 trade {trade.trade_id} missing p_profitable in trade.extra")
                    trade_log_row["entry_p_profitable"] = f"{p_profitable:.4f}" if p_profitable is not None else ""
                    
                    # Get policy version
                    policy_version = trade_extra.get("entry_policy_version", "FARM_V2")
                    trade_log_row["entry_policy_version"] = policy_version
                else:
                    # For non-FARM_V2 trades, set empty values
                    trade_log_row["entry_p_long"] = ""
                    trade_log_row["entry_p_profitable"] = ""
                    trade_log_row["entry_policy_version"] = ""
                    # Extract exit_profile if present (for FARM_EXIT_V1_STABLE traceability)
                    if "exit_profile" in trade_extra:
                        trade_log_row["exit_profile"] = trade_extra["exit_profile"]
                    # Also set exit_profile if FARM_V1 mode is active and exit_farm_v1_initialized is set
                    elif hasattr(self, "farm_v1_mode") and self.farm_v1_mode and trade_extra.get("exit_farm_v1_initialized"):
                        trade_log_row["exit_profile"] = "FARM_EXIT_V1_STABLE"
                        # Also update trade.extra for consistency
                        trade_extra["exit_profile"] = "FARM_EXIT_V1_STABLE"
                # Lazy import to avoid circular dependency
                from gx1.execution.oanda_demo_runner import append_trade_log
                append_trade_log(self.trade_log_path, trade_log_row)

        if closes_requested:
            log.debug(
                "[EXIT] Close summary: requested=%d accepted=%d remaining_open=%d",
                closes_requested,
                closes_accepted,
                len(self.open_trades),
            )
        
        # Update tick watcher (stop if no more open trades)
        self._maybe_update_tick_watcher()
    
    def _collect_price_trace(self, candles: pd.DataFrame, now_ts: pd.Timestamp, open_trades: List[Any]) -> None:
        """
        Collect price trace for open trades (for intratrade metrics calculation).
        
        Stores high/low/close per bar in trade.extra["_price_trace"].
        Limited to last 5000 points to prevent memory bloat.
        """
        if len(candles) == 0:
            return
        
        try:
            # Get last closed bar prices
            last_bar = candles.iloc[-1]
            
            # Determine price source (prefer direct OHLC, fallback to bid/ask mid)
            if "high" in candles.columns and "low" in candles.columns and "close" in candles.columns:
                bar_high = float(last_bar["high"])
                bar_low = float(last_bar["low"])
                bar_close = float(last_bar["close"])
            elif "bid_high" in candles.columns and "ask_high" in candles.columns:
                bar_high = float((last_bar["bid_high"] + last_bar["ask_high"]) / 2.0)
                bar_low = float((last_bar["bid_low"] + last_bar["ask_low"]) / 2.0)
                bar_close = float((last_bar["bid_close"] + last_bar["ask_close"]) / 2.0)
            else:
                # Fallback: use close price for all
                if "close" in candles.columns:
                    bar_close = float(last_bar["close"])
                elif "bid_close" in candles.columns and "ask_close" in candles.columns:
                    bar_close = float((last_bar["bid_close"] + last_bar["ask_close"]) / 2.0)
                else:
                    return  # Cannot collect trace without price data
                bar_high = bar_close
                bar_low = bar_close
            
            # Append to price trace for each open trade
            for trade in open_trades:
                if not hasattr(trade, "extra") or trade.extra is None:
                    trade.extra = {}
                
                if "_price_trace" not in trade.extra:
                    trade.extra["_price_trace"] = []
                
                trace_point = {
                    "ts": now_ts.isoformat() if hasattr(now_ts, "isoformat") else str(now_ts),
                    "high": bar_high,
                    "low": bar_low,
                    "close": bar_close,
                }
                
                trade.extra["_price_trace"].append(trace_point)
                
                # Limit trace size (keep last 5000 points)
                max_trace_size = 5000
                if len(trade.extra["_price_trace"]) > max_trace_size:
                    trade.extra["_price_trace"] = trade.extra["_price_trace"][-max_trace_size:]
        
        except Exception as e:
            # Never break trading due to trace collection failure
            log.debug("[EXIT] Failed to collect price trace: %s", e)
    
    def _log_trade_close_with_metrics(
        self,
        trade: Any,
        exit_time: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
        realized_pnl_bps: float,
        bars_held: int,
    ) -> None:
        """
        Log trade close to journal with intratrade metrics calculation.
        
        Helper function to ensure consistent logging across all exit paths.
        """
        if not hasattr(self._runner, "trade_journal") or not self._runner.trade_journal:
            return
        
        try:
            from gx1.monitoring.trade_journal import EVENT_TRADE_CLOSED
            
            exit_time_iso = exit_time.isoformat() if hasattr(exit_time, "isoformat") else str(exit_time)
            
            # Calculate intratrade metrics from price trace
            intratrade_metrics = self._compute_intratrade_metrics(trade, exit_price)
            
            # Validate invariants with realized_pnl
            self._validate_intratrade_metrics(trade.trade_id, intratrade_metrics, realized_pnl_bps)
            
            # Remove _price_trace from trade.extra to prevent bloating trade logs
            # (metrics are already calculated and logged)
            if "_price_trace" in trade.extra:
                del trade.extra["_price_trace"]
            
            # Log exit summary (structured)
            self._runner.trade_journal.log_exit_summary(
                trade_id=trade.trade_id,
                exit_time=exit_time_iso,
                exit_price=exit_price,
                exit_reason=exit_reason,
                realized_pnl_bps=realized_pnl_bps,
                max_mfe_bps=intratrade_metrics.get("max_mfe_bps"),
                max_mae_bps=intratrade_metrics.get("max_mae_bps"),
                intratrade_drawdown_bps=intratrade_metrics.get("intratrade_drawdown_bps"),
            )
            
            # Log trade closed (backward compatibility JSONL)
            self._runner.trade_journal.log(
                EVENT_TRADE_CLOSED,
                {
                    "exit_time": exit_time_iso,
                    "exit_price": exit_price,
                    "exit_profile": trade.extra.get("exit_profile", "UNKNOWN"),
                    "exit_reason": exit_reason,
                    "bars_held": bars_held,
                    "pnl_bps": realized_pnl_bps,
                    "final_status": "CLOSED",
                },
                trade_key={
                    "entry_time": trade.entry_time.isoformat() if hasattr(trade.entry_time, "isoformat") else str(trade.entry_time),
                    "entry_price": trade.entry_price,
                    "side": trade.side,
                },
                trade_id=trade.trade_id,
            )
        except Exception as e:
            log.warning("[TRADE_JOURNAL] Failed to log TRADE_CLOSED: %s", e)
    
    def _compute_intratrade_metrics(
        self,
        trade: Any,
        exit_price: float,
    ) -> Dict[str, Optional[float]]:
        """
        Compute intratrade metrics (MFE, MAE, intratrade drawdown) from price trace.
        
        Args:
            trade: Trade object with entry_price, side, and _price_trace
            exit_price: Final exit price
            
        Returns:
            Dict with max_mfe_bps, max_mae_bps, intratrade_drawdown_bps
        """
        try:
            price_trace = trade.extra.get("_price_trace", [])
            if not price_trace:
                return {
                    "max_mfe_bps": None,
                    "max_mae_bps": None,
                    "intratrade_drawdown_bps": None,
                }
            
            entry_price = trade.entry_price
            side = trade.side.lower()
            
            # Convert price trace to unrealized PnL curve (in bps)
            pnl_curve = []
            for point in price_trace:
                # Use high for favorable (long), low for adverse (long)
                # Use low for favorable (short), high for adverse (short)
                high_price = point["high"]
                low_price = point["low"]
                close_price = point["close"]
                
                if side == "long":
                    # Favorable: use high (best case)
                    favorable_pnl_bps = (high_price - entry_price) / entry_price * 10000.0
                    # Adverse: use low (worst case)
                    adverse_pnl_bps = (low_price - entry_price) / entry_price * 10000.0
                    # Current unrealized: use close
                    current_pnl_bps = (close_price - entry_price) / entry_price * 10000.0
                else:  # short
                    # Favorable: use low (best case for short)
                    favorable_pnl_bps = (entry_price - low_price) / entry_price * 10000.0
                    # Adverse: use high (worst case for short)
                    adverse_pnl_bps = (entry_price - high_price) / entry_price * 10000.0
                    # Current unrealized: use close
                    current_pnl_bps = (entry_price - close_price) / entry_price * 10000.0
                
                pnl_curve.append({
                    "favorable": favorable_pnl_bps,
                    "adverse": adverse_pnl_bps,
                    "current": current_pnl_bps,
                })
            
            # Add exit price to curve
            if side == "long":
                exit_pnl_bps = (exit_price - entry_price) / entry_price * 10000.0
            else:  # short
                exit_pnl_bps = (entry_price - exit_price) / entry_price * 10000.0
            
            pnl_curve.append({
                "favorable": exit_pnl_bps,
                "adverse": exit_pnl_bps,
                "current": exit_pnl_bps,
            })
            
            # Calculate MFE (max favorable excursion)
            max_mfe_bps = max(p["favorable"] for p in pnl_curve)
            
            # Calculate MAE (max adverse excursion)
            max_mae_bps = min(p["adverse"] for p in pnl_curve)
            
            # Calculate intratrade drawdown (largest peak-to-trough drawdown)
            # Build unrealized PnL curve from current values
            unrealized_curve = [p["current"] for p in pnl_curve]
            
            max_drawdown_bps = 0.0
            peak = unrealized_curve[0] if unrealized_curve else 0.0
            
            for pnl in unrealized_curve:
                if pnl > peak:
                    peak = pnl
                drawdown = peak - pnl
                if drawdown > max_drawdown_bps:
                    max_drawdown_bps = drawdown
            
            metrics = {
                "max_mfe_bps": float(max_mfe_bps),
                "max_mae_bps": float(max_mae_bps),
                "intratrade_drawdown_bps": float(max_drawdown_bps),
            }
            
            # Note: Invariants validated in _log_trade_close_with_metrics() with realized_pnl
            
            return metrics
        
        except Exception as e:
            log.warning("[EXIT] Failed to compute intratrade metrics for trade %s: %s", trade.trade_id, e)
            return {
                "max_mfe_bps": None,
                "max_mae_bps": None,
                "intratrade_drawdown_bps": None,
            }
    
    def _validate_intratrade_metrics(
        self,
        trade_id: str,
        metrics: Dict[str, Optional[float]],
        realized_pnl_bps: Optional[float] = None,
    ) -> None:
        """
        Validate intratrade metrics invariants (soft warnings in prod).
        
        Invariants:
        - MFE >= 0 (favorable excursion cannot be negative)
        - MAE <= 0 (adverse excursion cannot be positive)
        - Intratrade DD <= 0 (drawdown is negative or zero)
        - If realized_pnl > 0, MFE should be >= realized_pnl (MFE is best case)
        
        On violation: log WARNING but do not block trading.
        """
        eps = 1e-6
        
        mfe_bps = metrics.get("max_mfe_bps")
        mae_bps = metrics.get("max_mae_bps")
        dd_bps = metrics.get("intratrade_drawdown_bps")
        
        # Validate MFE >= 0
        if mfe_bps is not None and mfe_bps < -eps:
            log.warning(
                "[INTRATRADE_INVARIANT] Trade %s: MFE violation (MFE=%.2f < 0). "
                "Metrics snapshot: MFE=%.2f, MAE=%.2f, DD=%.2f",
                trade_id, mfe_bps, mfe_bps, mae_bps, dd_bps
            )
        
        # Validate MAE <= 0
        if mae_bps is not None and mae_bps > eps:
            log.warning(
                "[INTRATRADE_INVARIANT] Trade %s: MAE violation (MAE=%.2f > 0). "
                "Metrics snapshot: MFE=%.2f, MAE=%.2f, DD=%.2f",
                trade_id, mae_bps, mfe_bps, mae_bps, dd_bps
            )
        
        # Validate DD <= 0
        if dd_bps is not None and dd_bps > eps:
            log.warning(
                "[INTRATRADE_INVARIANT] Trade %s: Drawdown violation (DD=%.2f > 0). "
                "Metrics snapshot: MFE=%.2f, MAE=%.2f, DD=%.2f",
                trade_id, dd_bps, mfe_bps, mae_bps, dd_bps
            )
        
        # Validate MFE >= realized_pnl (if realized_pnl > 0)
        if realized_pnl_bps is not None and realized_pnl_bps > eps:
            if mfe_bps is not None and mfe_bps + eps < realized_pnl_bps:
                log.warning(
                    "[INTRATRADE_INVARIANT] Trade %s: MFE < realized_pnl (MFE=%.2f < realized=%.2f). "
                    "Metrics snapshot: MFE=%.2f, MAE=%.2f, DD=%.2f, realized=%.2f",
                    trade_id, mfe_bps, realized_pnl_bps, mfe_bps, mae_bps, dd_bps, realized_pnl_bps
                )

    def _compute_runtime_atr_bps(self, candles: pd.DataFrame, period: int = 5) -> Optional[float]:
        """Compute mid-price ATR(5) in bps for adaptive exit overlays."""
        required_cols = [
            "bid_high",
            "bid_low",
            "bid_close",
            "ask_high",
            "ask_low",
            "ask_close",
        ]
        if not all(col in candles.columns for col in required_cols):
            return None
        window = candles[required_cols].tail(period + 2)
        if len(window) < period:
            return None
        mid_df = pd.DataFrame(
            {
                "high": (window["bid_high"] + window["ask_high"]) * 0.5,
                "low": (window["bid_low"] + window["ask_low"]) * 0.5,
                "close": (window["bid_close"] + window["ask_close"]) * 0.5,
            },
            index=window.index,
        )
        atr_series = compute_atr_bps(mid_df, period=period)
        if atr_series.empty:
            return None
        latest = atr_series.iloc[-1]
        if pd.isna(latest):
            return None
        return float(latest)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
