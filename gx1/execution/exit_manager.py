from __future__ import annotations

import os
import json
from collections import deque
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pathlib import Path
from datetime import datetime, timezone

import logging
import pandas as pd
import numpy as np
from gx1.exits.contracts.exit_io_v0_ctx19 import (
    EXIT_IO_V0_CTX19_FEATURES,
    EXIT_IO_V0_CTX19_FEATURE_NAMES_HASH,
    EXIT_IO_V0_CTX19_IO_VERSION,
    assert_exit_io_v0_ctx19_contract,
    compute_feature_names_hash,
)

# DEL 3: PREBUILT mode fix - move live_features imports to lazy imports
# live_features is forbidden in PREBUILT mode, so we only import it when needed (live mode only)
# These functions are used at runtime, so we import them locally where needed
from gx1.utils.pnl import compute_pnl_bps
from gx1.execution.exit_critic_controller import ExitCriticController

# RL exit critic is disabled; keep stub constants to avoid runtime surprises.
def _rl_disabled(*_args, **_kwargs):
    raise RuntimeError(
        "RL_DISABLED: exit_critic is archived. ENTRY/EXIT transformer + replay plumbing only."
    )

ACTION_EXIT_NOW = "RL_DISABLED_EXIT_NOW"
ACTION_SCALP_PROFIT = "RL_DISABLED_SCALP_PROFIT"

if TYPE_CHECKING:
    from gx1.execution.oanda_demo_runner import GX1DemoRunner

log = logging.getLogger(__name__)

class ExitManager:
    def __init__(self, runner: "GX1DemoRunner") -> None:
        object.__setattr__(self, "_runner", runner)
        try:
            assert_exit_io_v0_ctx19_contract()
        except Exception:
            # Propagate to fail fast; ensures contract checked even if import guard skipped
            raise

    def __getattr__(self, name: str):
        return getattr(self._runner, name)

    def __setattr__(self, name: str, value):
        # ExitManager owns all private attrs (including _runner); public attrs proxy to runner
        if name == "_runner" or name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        setattr(self._runner, name, value)
    

    def evaluate_and_close_trades(self, candles: pd.DataFrame) -> None:
        """
        Evaluate and close trades using the canonical exit transformer path.

        Raises if the configured exit_mode deviates from exit_transformer_v0 to
        enforce single-universe execution without fallbacks.
        """
        # Skip exit evaluation in ENTRY_ONLY mode (no trades to close)
        if self.mode == "ENTRY_ONLY":
            return
        
        exit_mode = (
            (getattr(self, "exit_params", {}) or {})
            .get("exit", {})
            .get("params", {})
            .get("exit_ml", {})
            .get("mode")
            if isinstance(getattr(self, "exit_params", {}), dict)
            else None
        )
        if exit_mode != "exit_transformer_v0":
            raise RuntimeError("[EXIT_CONTRACT] only exit_transformer_v0 supported in canonical truth")
        
        # Tick-exit handling (no FARM paths)
        if self.tick_cfg.get("enabled", False) and not self.replay_mode:
            # Normal mode: tick-exit is handled by TickWatcher thread (live mode only)
            # In replay mode, tick-exit is evaluated here if enabled
            pass
        
        # Get snapshot of open trades
        if not self.open_trades:
            return
        open_trades_copy = list(self.open_trades)
        
        # One-shot exit window_len report (observability only; no behavior change)
        def _maybe_log_exit_windowlen_once() -> None:
            if getattr(self, "_exit_windowlen_reported", False):
                return
            try:
                policy_window_len = None
                model_window_len = None
                model_input_dim = None
                model_io_version = None
                model_feat_hash = None
                bundle_dir = None

                # Policy hint
                try:
                    exit_cfg = getattr(self, "exit_params", {}) or {}
                    policy_window_len = (
                        exit_cfg.get("exit", {})
                        .get("params", {})
                        .get("exit_ml", {})
                        .get("exit_transformer", {})
                        .get("window_len")
                    )
                except Exception:
                    policy_window_len = None

                # Resolve bundle dir from policy model_path
                try:
                    exit_cfg = getattr(self, "exit_params", {}) or {}
                    model_path_raw = (
                        exit_cfg.get("exit", {})
                        .get("params", {})
                        .get("exit_ml", {})
                        .get("exit_transformer", {})
                        .get("model_path")
                    )
                    if model_path_raw:
                        model_path = Path(model_path_raw)
                        if not model_path.is_absolute():
                            gx1_data_root = Path(os.environ.get("GX1_DATA", "")).expanduser()
                            model_path = gx1_data_root / model_path
                        bundle_dir = model_path
                        config_path = bundle_dir / "exit_transformer_config.json"
                        with open(config_path, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                        model_window_len = cfg.get("window_len")
                        model_input_dim = cfg.get("input_dim")
                        model_io_version = cfg.get("exit_ml_io_version")
                        model_feat_hash = cfg.get("feature_names_hash")
                except Exception:
                    pass

                # Effective window_len: prefer model config if available, else policy hint
                effective_window_len = model_window_len or policy_window_len

                # Mismatch notice (no crash)
                if policy_window_len is not None and model_window_len is not None and policy_window_len != model_window_len:
                    log.info(
                        "[EXIT_WINDOWLEN_MISMATCH] policy=%s model=%s using=%s",
                        policy_window_len,
                        model_window_len,
                        model_window_len,
                    )

                # Build tensor shape guess (B, T, D) using current open trades count
                built_tensor_shape = None
                try:
                    T = int(effective_window_len) if effective_window_len is not None else None
                    D = int(model_input_dim) if model_input_dim is not None else None
                    if T and D:
                        built_tensor_shape = (1, T, D)
                except Exception:
                    built_tensor_shape = None

                # Write report to run root if available
                out_dir = getattr(self, "explicit_output_dir", None) or getattr(self, "output_dir", None)
                if out_dir:
                    report_path = Path(out_dir) / "EXIT_WINDOWLEN_REPORT.json"
                    payload = {
                        "run_id": getattr(self, "run_id", None),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "policy_window_len": policy_window_len,
                        "model_config_window_len": model_window_len,
                        "effective_window_len": effective_window_len,
                        "model_input_dim": model_input_dim,
                        "model_io_version": model_io_version,
                        "model_feature_names_hash": model_feat_hash,
                        "bundle_dir": str(bundle_dir) if bundle_dir else None,
                        "built_tensor_shape": built_tensor_shape,
                    }
                    try:
                        from gx1.utils.atomic_json import atomic_write_json

                        atomic_write_json(report_path, payload)
                    except Exception:
                        try:
                            with open(report_path, "w", encoding="utf-8") as f:
                                json.dump(payload, f, indent=2)
                        except Exception:
                            pass

                log.info(
                    "[EXIT_WINDOWLEN_REPORT] policy_window_len=%s model_config_window_len=%s effective_window_len=%s model_input_dim=%s model_io_version=%s model_feature_names_hash=%s built_tensor_shape=%s",
                    policy_window_len,
                    model_window_len,
                    effective_window_len,
                    model_input_dim,
                    model_io_version,
                    model_feat_hash,
                    built_tensor_shape,
                )
                # Additional boot proof when transformer mode declared
                exit_mode = (
                    (getattr(self, "exit_params", {}) or {})
                    .get("exit", {})
                    .get("params", {})
                    .get("exit_ml", {})
                    .get("mode")
                    if isinstance(getattr(self, "exit_params", {}), dict)
                    else None
                )
                if exit_mode == "exit_transformer_v0":
                    log.info(
                        "[EXIT_T8_PROOF_BOOT] mode=%s model_path=%s model_window_len=%s input_dim=%s feature_hash=%s",
                        exit_mode,
                        bundle_dir,
                        model_window_len,
                        model_input_dim,
                        model_feat_hash,
                    )
            finally:
                setattr(self, "_exit_windowlen_reported", True)

        _maybe_log_exit_windowlen_once()

        self._ensure_bid_ask_columns(candles, context="exit_manager")
        now_ts = candles.index[-1]
        current_bid = float(candles["bid_close"].iloc[-1])
        current_ask = float(candles["ask_close"].iloc[-1])
        runtime_atr_bps = self._compute_runtime_atr_bps(candles)
        # Update per-bar extrema once per bar for determinism
        self._update_trade_extremes_current_bar(open_trades_copy, now_ts, current_bid, current_ask)
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

        # FARM paths disabled in canonical truth
        if False and hasattr(self, "farm_v1_mode") and self.farm_v1_mode:
            pass

        if False and getattr(self, "exit_farm_v2_rules_factory", None):
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
                
                # Build exit snapshot for ExitCritic (if enabled)
                exit_snapshot = None
                critic_action = None
                if hasattr(self, "exit_critic_controller") and self.exit_critic_controller:
                    try:
                        from gx1.execution.live_features import build_live_exit_snapshot
                        exit_snapshot = build_live_exit_snapshot(
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
                        # Get regime info from trade if available
                        session = getattr(trade, "session", None) or exit_snapshot["session_tag"].iloc[0] if len(exit_snapshot) > 0 else None
                        vol_regime = getattr(trade, "vol_regime", None) or exit_snapshot["vol_bucket"].iloc[0] if len(exit_snapshot) > 0 else None
                        trend_regime = getattr(trade, "trend_regime", None) or None
                        
                        critic_action = self.exit_critic_controller.maybe_apply_exit_critic(
                            trade_snapshot=exit_snapshot.iloc[0].to_dict() if len(exit_snapshot) > 0 else {},
                            bars_held=est_bars,
                            current_pnl_bps=est_pnl,
                            session=session,
                            vol_regime=vol_regime,
                            trend_regime=trend_regime,
                        )
                    except Exception as e:
                        log.warning("[EXIT_CRITIC] Failed to evaluate: %s", e, exc_info=True)
                
                # ExitCritic override: EXIT_NOW
                if critic_action == ACTION_EXIT_NOW:
                    exit_price = current_bid if trade.side == "long" else current_ask
                    accepted = self.request_close(
                        trade_id=trade.trade_id,
                        source="EXIT_CRITIC",
                        reason="EXIT_CRITIC_EXIT_NOW",
                        px=exit_price,
                        pnl_bps=est_pnl,
                        bars_in_trade=est_bars,
                    )
                    if accepted:
                        if trade in self.open_trades:
                            self.open_trades.remove(trade)
                        self._teardown_exit_state(trade.trade_id)
                        self.record_realized_pnl(now_ts, est_pnl)
                        log.info(
                            "[LIVE] CLOSED TRADE (EXIT_CRITIC_EXIT_NOW) %s %s @ %.3f | pnl=%.1f bps | bars=%d",
                            trade.side.upper(),
                            trade.trade_id,
                            exit_price,
                            est_pnl,
                            est_bars,
                        )
                        self._log_trade_close_with_metrics(
                            trade=trade,
                            exit_time=now_ts,
                            exit_price=exit_price,
                            exit_reason="EXIT_CRITIC_EXIT_NOW",
                            realized_pnl_bps=est_pnl,
                            bars_held=est_bars,
                        )
                        self._update_trade_log_on_close(
                            trade.trade_id,
                            exit_price,
                            est_pnl,
                            "EXIT_CRITIC_EXIT_NOW",
                            now_ts,
                            bars_in_trade=est_bars,
                        )
                        continue
                
                # ExitCritic override: SCALP_PROFIT (partial exit)
                if critic_action == ACTION_SCALP_PROFIT:
                    # Partial exit: close 50% of position
                    exit_price = current_bid if trade.side == "long" else current_ask
                    partial_units = abs(trade.units) // 2
                    if partial_units > 0:
                        # Note: request_close doesn't support partial closes directly
                        # For now, we'll log and let normal exit logic handle it
                        # TODO: Implement partial close support
                        log.info(
                            "[EXIT_CRITIC] SCALP_PROFIT signal for trade %s (pnl=%.2f bps) - partial exit not yet implemented, continuing with normal exit",
                            trade.trade_id,
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
                                timestamp=exit_time_iso,
                                trade_uid=trade.trade_uid,  # Primary key (COMMIT C)
                                trade_id=trade.trade_id,  # Display ID (backward compatibility)
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

        # Exit transformer path (single supported mode)
        exit_mode = (
            (getattr(self, "exit_params", {}) or {})
            .get("exit", {})
            .get("params", {})
            .get("exit_ml", {})
            .get("mode")
            if isinstance(getattr(self, "exit_params", {}), dict)
            else None
        )
        if exit_mode != "exit_transformer_v0":
            raise RuntimeError(
                "[EXIT_CONTRACT] only exit_transformer_v0 supported in canonical truth"
            )
        if exit_mode == "exit_transformer_v0":
            decider = getattr(self, "exit_transformer_decider", None)
            if decider is None:
                raise RuntimeError(
                    "[EXIT_MODEL_REQUIRED] Exit transformer model is required for configured exit policy; decider missing."
                )
            window_len = int(getattr(decider, "window_len", -1))
            input_dim = int(getattr(decider, "input_dim", -1))
            if window_len != 8 or input_dim != 19:
                raise RuntimeError(
                    f"[EXIT_IO_CONTRACT_VIOLATION] expected window_len=8,input_dim=19 got window_len={window_len},input_dim={input_dim}"
                )
            if not getattr(self, "_exit_features_proof_logged", False):
                computed_hash = compute_feature_names_hash(EXIT_IO_V0_CTX19_FEATURES)
                log.info(
                    "[EXIT_FEATURES_PROOF] n=%d hash=%s expected_hash=%s first3=%s last3=%s",
                    len(EXIT_IO_V0_CTX19_FEATURES),
                    computed_hash,
                    EXIT_IO_V0_CTX19_FEATURE_NAMES_HASH,
                    EXIT_IO_V0_CTX19_FEATURES[:3],
                    EXIT_IO_V0_CTX19_FEATURES[-3:],
                )
                self._exit_features_proof_logged = True
            if not getattr(self, "_exit_t8_forward_logged", False):
                log.info(
                    "[EXIT_T8_PROOF_FWD] built_tensor_shape=%s n_open_trades=%s",
                    (1, window_len, input_dim),
                    len(open_trades_copy),
                )
                self._exit_t8_forward_logged = True

            for trade in open_trades_copy:
                if trade not in self.open_trades:
                    continue
                min_hold_bars = 2
                delta_minutes = (now_ts - trade.entry_time).total_seconds() / 60.0
                bars_in_trade_min = int(round(delta_minutes / 5.0))
                if bars_in_trade_min < min_hold_bars:
                    log.debug(
                        "[EXIT] Skipping exit model evaluation for trade %s: bars_in_trade=%d < min_hold_bars=%d (trade too new)",
                        trade.trade_id,
                        bars_in_trade_min,
                        min_hold_bars,
                    )
                    continue
                window_arr = self._build_exit_ctx19_window(trade, candles, window_len)
                if window_arr is None:
                    continue
                prob_close, _, _ = decider.predict(window_arr)
                if not hasattr(trade, "prob_close_history") or trade.prob_close_history is None:
                    trade.prob_close_history = deque(maxlen=int(self.exit_require_consecutive) * 4)
                try:
                    if trade.side == "long":
                        self._runner.exit_eval_long += 1
                    else:
                        self._runner.exit_eval_short += 1
                except Exception:
                    pass
                trade.prob_close_history.append(prob_close)

                log.debug(
                    "[EXIT] Trade %s: bars_in_trade=%d prob_close=%.4f threshold=%.4f history=%s (require_consecutive=%d)",
                    trade.trade_id,
                    bars_in_trade_min,
                    prob_close,
                    self.exit_threshold,
                    list(trade.prob_close_history),
                    self.exit_require_consecutive,
                )

                should_exit = False
                if len(trade.prob_close_history) >= self.exit_require_consecutive:
                    recent_bars = list(trade.prob_close_history)[-self.exit_require_consecutive:]
                    should_exit = all(p >= self.exit_threshold for p in recent_bars)
                    if should_exit:
                        log.info(
                            "[EXIT] Trade %s: prob_close >= threshold in %d consecutive bars: %s (hysteresis passed)",
                            trade.trade_id,
                            self.exit_require_consecutive,
                            [f"{p:.4f}" for p in recent_bars],
                        )
                        try:
                            if trade.side == "long":
                                self._runner.exit_close_long += 1
                            else:
                                self._runner.exit_close_short += 1
                        except Exception:
                            pass
                    else:
                        log.debug(
                            "[EXIT] Trade %s: prob_close < threshold in some bars: %s (hysteresis: need %d consecutive)",
                            trade.trade_id,
                            [f"{p:.4f}" for p in recent_bars],
                            self.exit_require_consecutive,
                        )
                else:
                    log.debug(
                        "[EXIT] Trade %s: not enough bars for hysteresis check: %d < %d",
                        trade.trade_id,
                        len(trade.prob_close_history),
                        self.exit_require_consecutive,
                    )

                if should_exit:
                    if trade not in self.open_trades:
                        continue
                    entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
                    entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
                    if not hasattr(self, "_exit_mgr_pnl_log_count"):
                        self._exit_mgr_pnl_log_count = 0
                    if self._exit_mgr_pnl_log_count < 5:
                        log.debug(
                            "[PNL] ExitManager PnL: entry_bid=%.5f entry_ask=%.5f exit_bid=%.5f exit_ask=%.5f side=%s",
                            entry_bid,
                            entry_ask,
                            current_bid,
                            current_ask,
                            trade.side,
                        )
                        self._exit_mgr_pnl_log_count += 1
                    pnl_bps = compute_pnl_bps(entry_bid, entry_ask, current_bid, current_ask, trade.side)
                    mark_price = current_bid if trade.side == "long" else current_ask
                    delta_minutes = (now_ts - trade.entry_time).total_seconds() / 60.0
                    bars_in_trade = int(round(delta_minutes / 5.0))
                    log.info("[EXIT] propose close reason=THRESHOLD pnl=%.1f bars_in_trade=%d", pnl_bps, bars_in_trade)
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
                        continue
                    closes_accepted += 1
                    if trade in self.open_trades:
                        self.open_trades.remove(trade)
                    self.record_realized_pnl(now_ts, pnl_bps)
                    self._log_trade_close_with_metrics(
                        trade=trade,
                        exit_time=now_ts,
                        exit_price=mark_price,
                        exit_reason="THRESHOLD",
                        realized_pnl_bps=pnl_bps,
                        bars_held=bars_in_trade,
                    )
            return

        if closes_requested:
            log.debug(
                "[EXIT] Close summary: requested=%d accepted=%d remaining_open=%d",
                closes_requested,
                closes_accepted,
                len(self.open_trades),
            )
        
        # Update tick watcher (stop if no more open trades)
        self._maybe_update_tick_watcher()
    
    def _build_exit_ctx19_window(self, trade: Any, candles: pd.DataFrame, window_len: int) -> Optional[np.ndarray]:
        """
        Build (T,19) exit transformer input window for a single trade.
        """
        signal_history = list(getattr(self._runner, "exit_signal7_history", []))
        if len(signal_history) < window_len:
            if not getattr(trade, "_exit_warmup_logged", False):
                log.info(
                    "[EXIT_WARMUP_SKIP] insufficient signal7 history: have=%d need=%d trade_uid=%s trade_id=%s",
                    len(signal_history),
                    window_len,
                    getattr(trade, "trade_uid", None),
                    getattr(trade, "trade_id", None),
                )
                trade._exit_warmup_logged = True
            return None
        candle_tail = candles.tail(window_len)
        if len(candle_tail) < window_len:
            raise RuntimeError(
                f"[EXIT_TRANSFORMER_INPUT] insufficient bars for window_len={window_len}, have {len(candle_tail)}"
            )
        candle_rows = list(candle_tail.itertuples(index=True, name="CandleRow"))
        signal_slice = signal_history[-window_len:]
        runtime_atr_bps = self._compute_runtime_atr_bps(candles, period=5)
        if runtime_atr_bps is None:
            raise RuntimeError("[EXIT_IO_CONTRACT_VIOLATION] atr_bps_now unavailable")
        for attr in ("p_long_entry", "p_hat_entry", "uncertainty_entry", "entropy_entry", "margin_entry"):
            if not hasattr(trade, attr):
                raise RuntimeError(f"[EXIT_IO_CONTRACT_VIOLATION] trade missing {attr} snapshot")

        entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
        entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
        local_mfe = float(getattr(trade, "mfe_bps", 0.0))
        local_mae = float(getattr(trade, "mae_bps", 0.0))
        local_mfe_last_bar = int(getattr(trade, "_mfe_last_bar", 0))

        window_rows: List[List[float]] = []
        for idx, signal_dict in enumerate(signal_slice):
            candle_row = candle_rows[idx]
            try:
                bid_now = float(getattr(candle_row, "bid_close"))
                ask_now = float(getattr(candle_row, "ask_close"))
                bar_ts = pd.Timestamp(candle_row.Index)
            except Exception as e:
                raise RuntimeError(f"[EXIT_IO_CONTRACT_VIOLATION] missing bid/ask or ts in candle row: {e}") from e

            pnl_bps_now = compute_pnl_bps(entry_bid, entry_ask, bid_now, ask_now, trade.side)
            bars_held = int(round((bar_ts - trade.entry_time).total_seconds() / 300.0))
            if pnl_bps_now > local_mfe:
                local_mfe = pnl_bps_now
                local_mfe_last_bar = bars_held
            if pnl_bps_now < local_mae:
                local_mae = pnl_bps_now
            dd_from_mfe = local_mfe - pnl_bps_now
            time_since_mfe_bars = bars_held - local_mfe_last_bar

            row = [
                float(signal_dict["p_long"]),
                float(signal_dict["p_short"]),
                float(signal_dict["p_flat"]),
                float(signal_dict["p_hat"]),
                float(signal_dict["uncertainty_score"]),
                float(signal_dict["margin_top1_top2"]),
                float(signal_dict["entropy"]),
                float(trade.p_long_entry),
                float(trade.p_hat_entry),
                float(trade.uncertainty_entry),
                float(trade.entropy_entry),
                float(trade.margin_entry),
                float(pnl_bps_now),
                float(local_mfe),
                float(local_mae),
                float(dd_from_mfe),
                float(bars_held),
                float(time_since_mfe_bars),
                float(runtime_atr_bps),
            ]
            window_rows.append(row)

        window_arr = np.asarray(window_rows, dtype=np.float32)
        expected_shape = (window_len, 19)
        if window_arr.shape != expected_shape:
            raise RuntimeError(
                f"[EXIT_TRANSFORMER_INPUT] bad shape built: {window_arr.shape}, expected {expected_shape}"
            )
        if not np.isfinite(window_arr).all():
            raise RuntimeError("[EXIT_IO_CONTRACT_VIOLATION] non-finite values in exit transformer window")
        return window_arr

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

    def _update_trade_extremes_current_bar(
        self, open_trades: List[Any], now_ts: pd.Timestamp, current_bid: float, current_ask: float
    ) -> None:
        """
        Update MFE/MAE per trade once per bar using current bid/ask.
        """
        for trade in open_trades:
            entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
            entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
            pnl_now = compute_pnl_bps(entry_bid, entry_ask, current_bid, current_ask, trade.side)
            bars_held = int(round((now_ts - trade.entry_time).total_seconds() / 300.0))
            if not hasattr(trade, "mfe_bps"):
                trade.mfe_bps = pnl_now
            if not hasattr(trade, "mae_bps"):
                trade.mae_bps = pnl_now
            if not hasattr(trade, "_mfe_last_bar"):
                trade._mfe_last_bar = bars_held
            if pnl_now > trade.mfe_bps:
                trade.mfe_bps = pnl_now
                trade._mfe_last_bar = bars_held
            if pnl_now < trade.mae_bps:
                trade.mae_bps = pnl_now
    
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
                exit_time=exit_time_iso,
                trade_uid=trade.trade_uid,  # Primary key (COMMIT C)
                trade_id=trade.trade_id,  # Display ID (backward compatibility)
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
            
            # Calculate MAE (max adverse excursion) - convert to positive magnitude
            max_mae_bps_raw = min(p["adverse"] for p in pnl_curve)
            max_mae_bps = abs(max_mae_bps_raw)  # Positive magnitude (>=0)
            
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
            
            # Ensure MFE is positive (should already be, but enforce)
            max_mfe_bps_abs = abs(float(max_mfe_bps)) if max_mfe_bps < 0 else float(max_mfe_bps)
            max_drawdown_bps_abs = abs(float(max_drawdown_bps)) if max_drawdown_bps < 0 else float(max_drawdown_bps)
            
            metrics = {
                "max_mfe_bps": max_mfe_bps_abs,
                "max_mae_bps": float(max_mae_bps),  # Already positive magnitude from line 1031
                "intratrade_drawdown_bps": max_drawdown_bps_abs,
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
        - MFE >= 0 (favorable excursion magnitude)
        - MAE >= 0 (adverse excursion magnitude)
        - Intratrade DD >= 0 (drawdown magnitude)
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
        
        # Validate MAE magnitude >= 0
        if mae_bps is not None and mae_bps < -eps:
            log.warning(
                "[INTRATRADE_INVARIANT] Trade %s: MAE violation (MAE=%.2f < 0). "
                "Metrics snapshot: MFE=%.2f, MAE=%.2f, DD=%.2f",
                trade_id, mae_bps, mfe_bps, mae_bps, dd_bps
            )
        
        # Validate DD magnitude >= 0
        if dd_bps is not None and dd_bps < -eps:
            log.warning(
                "[INTRATRADE_INVARIANT] Trade %s: Drawdown violation (DD=%.2f < 0). "
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
        # DEL 3: Lazy import - only import when actually needed (baseline mode only)
        # live_features is forbidden in PREBUILT mode, so we only import it when needed
        from gx1.execution.live_features import compute_atr_bps
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


if __name__ == "__main__":
    print("ExitManager smoke test: module import succeeded.")
