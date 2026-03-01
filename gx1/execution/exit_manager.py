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
        self._exit_input_audit_logged_once = False
        self._exit_input_audit_samples_count = 0
        self._exit_ctx_audit_logged_once = False
        self._exit_effective_cfg_logged = False
        self._exit_prob_n = 0
        self._exit_prob_ge_thr = 0
        self._exit_prob_min = float("inf")
        self._exit_prob_max = float("-inf")
        self._exit_prob_buckets = {
            "<0.001": 0,
            "<0.01": 0,
            "<0.05": 0,
            "<0.1": 0,
            "<0.2": 0,
            ">=0.2": 0,
        }

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
        
        if not self._exit_effective_cfg_logged:
            runner = self._runner
            arb = getattr(runner, "exit_control", None) or {}
            arb_allow = getattr(arb, "allow_model_exit_when", {}) if hasattr(arb, "allow_model_exit_when") else arb.get("allow_model_exit_when", {}) if isinstance(arb, dict) else {}
            log.info(
                "[EXIT_EFFECTIVE_CFG] threshold=%.4f require_consecutive=%d arb_min_pnl_bps=%s arb_min_exit_prob=%s arb_exit_prob_hysteresis=%s arb_min_bars=%s",
                float(getattr(runner, "exit_threshold", 0.0)),
                int(getattr(runner, "exit_require_consecutive", 1)),
                arb_allow.get("min_pnl_bps"),
                arb_allow.get("min_exit_prob"),
                arb_allow.get("exit_prob_hysteresis"),
                arb_allow.get("min_bars"),
            )
            self._exit_effective_cfg_logged = True
        
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
        # Snapshot ctx once per bar using the same prebuilt row as entry ctx (if available)
        ctx_cont_snapshot: Optional[List[float]] = None
        ctx_cat_snapshot: Optional[List[int]] = None
        audit_strict = os.environ.get("GX1_EXIT_AUDIT_STRICT") == "1"
        try:
            prebuilt_df = getattr(self._runner, "prebuilt_features_df", None)
            ctx_cont_cols = getattr(self._runner, "ctx_cont_required_columns", None)
            ctx_all_cols = getattr(self._runner, "ctx_required_columns", None)
            if prebuilt_df is not None:
                if ctx_cont_cols is None or ctx_all_cols is None:
                    from gx1.contracts.signal_bridge_v1 import get_canonical_ctx_contract
                    ctx_contract = get_canonical_ctx_contract()
                    ctx_cont_cols = ctx_contract.get("ctx_cont_names")
                    ctx_all_cols = (ctx_contract.get("ctx_cont_names") or []) + (ctx_contract.get("ctx_cat_names") or [])
                ctx_cont_cols = list(ctx_cont_cols or [])
                ctx_all_cols = list(ctx_all_cols or [])
                if len(ctx_cont_cols) == 6 and len(ctx_all_cols) >= 12 and now_ts in prebuilt_df.index:
                    row = prebuilt_df.loc[now_ts]
                    def _extract(cols):
                        out = []
                        for c in cols:
                            if c not in row.index:
                                return None
                            out.append(float(row[c]))
                        return out
                    ctx_cont_snapshot = _extract(ctx_cont_cols)
                    ctx_cat_raw = _extract(ctx_all_cols[len(ctx_cont_cols):len(ctx_cont_cols)+6])
                    if ctx_cont_snapshot is not None and ctx_cat_raw is not None and len(ctx_cat_raw) == 6:
                        ctx_cat_snapshot = [int(x) for x in ctx_cat_raw]
                    elif audit_strict:
                        raise RuntimeError("[EXIT_CTX_AUDIT] ctx 6/6 missing for ts with prebuilt row present")
        except Exception:
            ctx_cont_snapshot = None
            ctx_cat_snapshot = None
            if audit_strict:
                raise
        if not getattr(self, "_exit_ctx_audit_logged_once", False):
            log.info(
                "[EXIT_CTX_AUDIT] have_ctx_cont=%s have_ctx_cat=%s ts=%s",
                bool(ctx_cont_snapshot and len(ctx_cont_snapshot) == 6),
                bool(ctx_cat_snapshot and len(ctx_cat_snapshot) == 6),
                now_ts,
            )
            self._exit_ctx_audit_logged_once = True
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
                # Track probability stats
                self._exit_prob_n += 1
                if prob_close >= self.exit_threshold:
                    self._exit_prob_ge_thr += 1
                self._exit_prob_min = min(self._exit_prob_min, prob_close)
                self._exit_prob_max = max(self._exit_prob_max, prob_close)
                b = self._exit_prob_buckets
                if prob_close < 0.001:
                    b["<0.001"] += 1
                elif prob_close < 0.01:
                    b["<0.01"] += 1
                elif prob_close < 0.05:
                    b["<0.05"] += 1
                elif prob_close < 0.1:
                    b["<0.1"] += 1
                elif prob_close < 0.2:
                    b["<0.2"] += 1
                else:
                    b[">=0.2"] += 1
                try:
                    should_log_io = self.replay_mode
                    if not should_log_io:
                        try:
                            exit_cfg = getattr(self, "exit_params", {}) or {}
                            log_flag = (
                                exit_cfg.get("exit", {})
                                .get("params", {})
                                .get("exit_ml", {})
                                .get("exit_transformer", {})
                                .get("log_io_features", False)
                            )
                            should_log_io = bool(log_flag)
                        except Exception:
                            should_log_io = False
                    if should_log_io:
                        self._append_exit_io_record(
                            event_ts=now_ts,
                            trade=trade,
                            prob_close=prob_close,
                            window_arr=window_arr,
                            ctx_cont=ctx_cont_snapshot,
                            ctx_cat=ctx_cat_snapshot,
                        )
                except Exception:
                    pass
                if not hasattr(self, "_exit_runtime_gt_count"):
                    self._exit_runtime_gt_count = 0
                if not hasattr(self, "_exit_hysteresis_dbg_count"):
                    self._exit_hysteresis_dbg_count = 0
                hist = self._ensure_prob_history(trade)
                hist_before = list(hist)
                try:
                    if trade.side == "long":
                        self._runner.exit_eval_long += 1
                    else:
                        self._runner.exit_eval_short += 1
                except Exception:
                    pass
                len_before = len(hist)
                hist.append(float(prob_close))
                len_after = len(hist)
                if self._exit_hysteresis_dbg_count < 30:
                    recent = list(hist)[-self.exit_require_consecutive:]
                    recent_f = [float(x) for x in recent]
                    should_exit_dbg = (
                        len_after >= self.exit_require_consecutive
                        and all(x >= float(self.exit_threshold) for x in recent_f)
                    )
                    log.info(
                        "[EXIT_HYSTERESIS_DBG] trade_uid=%s trade_id=%s side=%s trade_obj_id=%s hist_type=%s maxlen=%s len_before=%d len_after=%d thr=%.6f req=%d history=%s recent=%s recent_f=%s should_exit=%s",
                        getattr(trade, "trade_uid", None) or getattr(trade, "trade_id", None),
                        getattr(trade, "trade_id", None),
                        getattr(trade, "side", None),
                        id(trade),
                        type(hist).__name__,
                        getattr(hist, "maxlen", None),
                        len_before,
                        len_after,
                        float(self.exit_threshold),
                        int(self.exit_require_consecutive),
                        list(hist),
                        recent,
                        recent_f,
                        should_exit_dbg,
                    )
                    self._exit_hysteresis_dbg_count += 1
                if self._exit_runtime_gt_count < 5:
                    hist_after = list(hist)
                    log.info(
                        "[EXIT_RUNTIME_GROUND_TRUTH] trade_uid=%s trade_id=%s side=%s prob_close=%.6f threshold=%.6f require_consecutive=%d hist_before=%s hist_after=%s model_path=%s model_sha=%s window_len=%s input_dim=%s bars_held=%s",
                        getattr(trade, "trade_uid", None) or getattr(trade, "trade_id", None),
                        getattr(trade, "trade_id", None),
                        getattr(trade, "side", None),
                        prob_close,
                        self.exit_threshold,
                        self.exit_require_consecutive,
                        [f"{p:.6f}" for p in hist_before],
                        [f"{p:.6f}" for p in hist_after],
                        getattr(decider, "model_path", getattr(decider, "bundle_dir", None)),
                        getattr(decider, "model_sha", None),
                        getattr(decider, "window_len", None),
                        getattr(decider, "input_dim", None),
                        bars_in_trade_min,
                    )
                    self._exit_runtime_gt_count += 1

                # Best-effort: log one context-bearing exit ML event (ensures exits jsonl has ctx 6/6)
                if not getattr(self, "_exit_ml_ctx_logged_once", False) and ctx_cont_snapshot and ctx_cat_snapshot:
                    self._maybe_log_exit_ml_event(
                        event_ts=now_ts,
                        prob_close=prob_close,
                        trade=trade,
                        ctx_cont=ctx_cont_snapshot,
                        ctx_cat=ctx_cat_snapshot,
                    )
                    self._exit_ml_ctx_logged_once = True

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
                    recent_bars = [float(p) for p in list(trade.prob_close_history)][-self.exit_require_consecutive:]
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
                    entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
                    entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
                    pnl_bps = compute_pnl_bps(entry_bid, entry_ask, current_bid, current_ask, trade.side)
                    mark_price = current_bid if trade.side == "long" else current_ask
                    delta_minutes = (now_ts - trade.entry_time).total_seconds() / 60.0
                    bars_in_trade = int(round(delta_minutes / 5.0))
                    did_eval_any = True
                    if not hasattr(self, "_exit_decision_logged"):
                        self._exit_decision_logged = set()
                    tuid = getattr(trade, "trade_uid", None) or getattr(trade, "trade_id", None)
                    if tuid not in self._exit_decision_logged:
                        arb_allow = getattr(getattr(self._runner, "exit_control", None), "allow_model_exit_when", {}) or {}
                        log.info(
                            "[EXIT_DECISION_PROOF] trade_uid=%s trade_id=%s side=%s prob_close=%.6f threshold=%.6f require_consecutive=%d prob_history=%s arb_cfg={min_exit_prob=%s, exit_prob_hysteresis=%s, min_pnl_bps=%s} bars_held=%s pnl_bps=%s will_call_request_close=true",
                            tuid,
                            getattr(trade, "trade_id", None),
                            getattr(trade, "side", None),
                            prob_close,
                            self.exit_threshold,
                            self.exit_require_consecutive,
                            [f"{p:.6f}" for p in list(trade.prob_close_history)],
                            arb_allow.get("min_exit_prob"),
                            arb_allow.get("exit_prob_hysteresis"),
                            arb_allow.get("min_pnl_bps"),
                            bars_in_trade,
                            pnl_bps,
                        )
                        self._exit_decision_logged.add(tuid)
                    if trade not in self.open_trades:
                        continue
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
                        log.error(
                            "[EXIT_DECISION_RESULT] trade_uid=%s trade_id=%s accepted=False reject_reason=arbiter_reject still_in_open_trades=%s",
                            tuid,
                            getattr(trade, "trade_id", None),
                            trade in self.open_trades,
                        )
                        continue
                    closes_accepted += 1
                    if trade in self.open_trades:
                        self.open_trades.remove(trade)
                    log.info(
                        "[EXIT_DECISION_RESULT] trade_uid=%s trade_id=%s accepted=True reject_reason=None still_in_open_trades=%s",
                        tuid,
                        getattr(trade, "trade_id", None),
                        trade in self.open_trades,
                    )
                    self.record_realized_pnl(now_ts, pnl_bps)
                    self._log_trade_close_with_metrics(
                        trade=trade,
                        exit_time=now_ts,
                        exit_price=mark_price,
                        exit_reason="THRESHOLD",
                        realized_pnl_bps=pnl_bps,
                        bars_held=bars_in_trade,
                    )
            self._maybe_log_exit_prob_audit(decider=decider)
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
        self._maybe_log_exit_prob_audit(decider=decider)

    def _ensure_prob_history(self, trade: Any) -> deque:
        """
        Ensure trade.prob_close_history exists and has sufficient maxlen for hysteresis.
        """
        req = int(getattr(self, "exit_require_consecutive", 1))
        target = max(req * 4, req, 8)
        h = getattr(trade, "prob_close_history", None)
        if h is None:
            h = deque(maxlen=target)
            trade.prob_close_history = h
            return h
        cur_maxlen = getattr(h, "maxlen", None)
        if cur_maxlen is None or int(cur_maxlen) < target:
            old = []
            try:
                old = list(h)
            except Exception:
                old = []
            new = deque(old[-target:], maxlen=target)
            trade.prob_close_history = new
            return new
        return h

    def _maybe_log_exit_prob_audit(self, decider: Any = None) -> None:
        """
        One-shot probability audit log for exit transformer evaluations.
        """
        try:
            if getattr(self, "_exit_prob_audit_logged", False):
                return
            n = int(getattr(self, "_exit_prob_n", 0))
            if n <= 0:
                return
            b = getattr(self, "_exit_prob_buckets", {})
            model_dir = None
            if decider is not None:
                model_dir = getattr(decider, "bundle_dir", None) or getattr(decider, "model_path", None)
            if model_dir is None:
                model_dir = getattr(getattr(self._runner, "exit_model_decider", None), "bundle_dir", None)
            log.info(
                "[EXIT_PROB_AUDIT] n=%d ge_thr=%d thr=%.4f min=%.4f max=%.4f buckets=%s model_dir=%s",
                n,
                int(getattr(self, "_exit_prob_ge_thr", 0)),
                float(getattr(self._runner, "exit_threshold", getattr(self, "exit_threshold", 0.0))),
                float(self._exit_prob_min if getattr(self, "_exit_prob_min", float("inf")) != float("inf") else 0.0),
                float(self._exit_prob_max if getattr(self, "_exit_prob_max", float("-inf")) != float("-inf") else 0.0),
                b,
                model_dir,
            )
            self._exit_prob_audit_logged = True
        except Exception:
            return

    def _maybe_log_exit_ml_event(
        self,
        event_ts: pd.Timestamp,
        prob_close: float,
        trade: Any,
        ctx_cont: Optional[List[float]] = None,
        ctx_cat: Optional[List[int]] = None,
    ) -> None:
        """
        Append one exit ML event to exits_<run_id>.jsonl with ctx_cont/ctx_cat (6/6).
        Does nothing if required context columns are missing or dims mismatch.
        """
        try:
            runner = self._runner
            log_dir = getattr(runner, "log_dir", None)
            run_id = getattr(runner, "run_id", None)
            if log_dir is None or run_id is None:
                return
            if ctx_cont is None or ctx_cat is None:
                return
            if len(ctx_cont) != 6 or len(ctx_cat) != 6:
                return
            exits_path = Path(log_dir) / "exits" / f"exits_{run_id}.jsonl"
            exits_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": event_ts.isoformat() if hasattr(event_ts, "isoformat") else str(event_ts),
                "trade_id": getattr(trade, "trade_id", None),
                "computed": {"mode": "exit_transformer_v0", "prob_close": float(prob_close)},
                "context": {"ctx_cont": ctx_cont, "ctx_cat": ctx_cat},
            }
            with exits_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        except Exception:
            return

    def _append_exit_io_record(
        self,
        *,
        event_ts: pd.Timestamp,
        trade: Any,
        prob_close: float,
        window_arr: Any,
        ctx_cont: Optional[List[float]],
        ctx_cat: Optional[List[int]],
    ) -> None:
        """
        Append full IO (T=8, D=19) for exit transformer evaluation to exits_<run_id>.jsonl.
        """
        try:
            runner = self._runner
            log_dir = getattr(runner, "log_dir", None)
            run_id = getattr(runner, "run_id", None)
            if log_dir is None or run_id is None:
                return

            exits_path = Path(log_dir) / "exits" / f"exits_{run_id}.jsonl"
            exits_path.parent.mkdir(parents=True, exist_ok=True)

            # Normalize window to float list
            arr = np.asarray(window_arr, dtype=np.float32)
            if arr.shape != (8, 19):
                raise RuntimeError(f"[EXIT_IO_SHAPE] expected (8,19), got {arr.shape}")

            # Index map for scalars
            idx = {name: i for i, name in enumerate(EXIT_IO_V0_CTX19_FEATURES)}
            last = arr[-1]
            try:
                scalars = {
                    "pnl_bps_now": float(last[idx["pnl_bps_now"]]),
                    "mfe_bps": float(last[idx["mfe_bps"]]),
                    "mae_bps": float(last[idx["mae_bps"]]),
                    "dd_from_mfe_bps": float(last[idx["dd_from_mfe_bps"]]),
                    "bars_held": float(last[idx["bars_held"]]),
                    "time_since_mfe_bars": float(last[idx["time_since_mfe_bars"]]),
                    "atr_bps_now": float(last[idx["atr_bps_now"]]),
                }
            except KeyError as e:
                raise RuntimeError(f"[EXIT_IO_INDEX_MISSING] {e}")

            ctx_payload = None
            if ctx_cont and ctx_cat and len(ctx_cont) == 6 and len(ctx_cat) == 6:
                ctx_payload = {"ctx_cont": ctx_cont, "ctx_cat": ctx_cat}

            record = {
                "ts": event_ts.isoformat() if hasattr(event_ts, "isoformat") else str(event_ts),
                "run_id": run_id,
                "trade_uid": getattr(trade, "trade_uid", None),
                "trade_id": getattr(trade, "trade_id", None),
                "side": getattr(trade, "side", None),
                "computed": {
                    "mode": "exit_transformer_v0",
                    "prob_close": float(prob_close),
                    "threshold": float(self.exit_threshold),
                },
                "context": ctx_payload,
                "io": {
                    "io_version": EXIT_IO_V0_CTX19_IO_VERSION,
                    "feature_names_hash": EXIT_IO_V0_CTX19_FEATURE_NAMES_HASH,
                    "window_len": 8,
                    "input_dim": 19,
                    "io_features": arr.tolist(),
                },
                "scalars": scalars,
            }

            with exits_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        except Exception:
            return

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
        pre_entry_count = 0
        for idx, signal_dict in enumerate(signal_slice):
            candle_row = candle_rows[idx]
            try:
                bid_now = float(getattr(candle_row, "bid_close"))
                ask_now = float(getattr(candle_row, "ask_close"))
                bar_ts = pd.Timestamp(candle_row.Index)
            except Exception as e:
                raise RuntimeError(f"[EXIT_IO_CONTRACT_VIOLATION] missing bid/ask or ts in candle row: {e}") from e

            is_post_entry = bar_ts >= trade.entry_time
            if is_post_entry:
                pnl_bps_now = compute_pnl_bps(entry_bid, entry_ask, bid_now, ask_now, trade.side)
                bars_held_raw = int(round((bar_ts - trade.entry_time).total_seconds() / 300.0))
                bars_held = max(0, bars_held_raw)
                if pnl_bps_now > local_mfe:
                    local_mfe = pnl_bps_now
                    local_mfe_last_bar = bars_held
                if pnl_bps_now < local_mae:
                    local_mae = pnl_bps_now
                dd_from_mfe = max(0.0, local_mfe - pnl_bps_now)
                time_since_mfe_bars = max(0.0, bars_held - local_mfe_last_bar)
                local_mfe_for_row = local_mfe
                local_mae_for_row = local_mae
                bars_held_for_row = float(bars_held)
            else:
                pre_entry_count += 1
                pnl_bps_now = 0.0
                local_mfe_for_row = 0.0
                local_mae_for_row = 0.0
                dd_from_mfe = 0.0
                bars_held_for_row = 0.0
                time_since_mfe_bars = 0.0

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
                float(local_mfe_for_row),
                float(local_mae_for_row),
                float(dd_from_mfe),
                float(bars_held_for_row),
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

        idx = {name: i for i, name in enumerate(EXIT_IO_V0_CTX19_FEATURES)}
        strict = os.environ.get("GX1_EXIT_AUDIT_STRICT") == "1"
        eps = 1e-6

        # One-shot audit log with snapshots
        if not getattr(self, "_exit_input_audit_logged_once", False):
            first_row_first7 = window_arr[0, :7].tolist()
            last_row_first7 = window_arr[-1, :7].tolist()
            entry_snapshots = {
                "p_long_entry": float(window_arr[0, idx["p_long_entry"]]),
                "p_hat_entry": float(window_arr[0, idx["p_hat_entry"]]),
                "uncertainty_entry": float(window_arr[0, idx["uncertainty_entry"]]),
                "entropy_entry": float(window_arr[0, idx["entropy_entry"]]),
                "margin_entry": float(window_arr[0, idx["margin_entry"]]),
            }
            last_row_scalars = {
                "pnl_bps_now": float(window_arr[-1, idx["pnl_bps_now"]]),
                "mfe_bps": float(window_arr[-1, idx["mfe_bps"]]),
                "mae_bps": float(window_arr[-1, idx["mae_bps"]]),
                "dd_from_mfe_bps": float(window_arr[-1, idx["dd_from_mfe_bps"]]),
                "bars_held": float(window_arr[-1, idx["bars_held"]]),
                "time_since_mfe_bars": float(window_arr[-1, idx["time_since_mfe_bars"]]),
                "atr_bps_now": float(window_arr[-1, idx["atr_bps_now"]]),
            }
            log.info(
                "[EXIT_INPUT_AUDIT_ONESHOT] io_version=%s feature_hash=%s window_len=%d input_dim=%d first_row_first7=%s last_row_first7=%s entry_snapshots=%s last_row_scalars=%s pre_entry_count=%d",
                EXIT_IO_V0_CTX19_IO_VERSION,
                EXIT_IO_V0_CTX19_FEATURE_NAMES_HASH,
                window_len,
                window_arr.shape[1],
                first_row_first7,
                last_row_first7,
                entry_snapshots,
                last_row_scalars,
                pre_entry_count,
            )
            self._exit_input_audit_logged_once = True

        # Sample a few windows for additional visibility
        if getattr(self, "_exit_input_audit_samples_count", 0) < 5:
            bars_series = window_arr[:, idx["bars_held"]]
            pnl_series = window_arr[:, idx["pnl_bps_now"]]
            mfe_series = window_arr[:, idx["mfe_bps"]]
            dd_series = window_arr[:, idx["dd_from_mfe_bps"]]
            ts_mfe_series = window_arr[:, idx["time_since_mfe_bars"]]
            atr_series = window_arr[:, idx["atr_bps_now"]]
            log.info(
                "[EXIT_INPUT_AUDIT_SAMPLE] trade_uid=%s side=%s bars_last=%.3f pnl_last=%.3f mfe_last=%.3f dd_last=%.3f ts_mfe_last=%.3f atr_last=%.3f pnl_min_max=%.3f/%.3f mfe_min_max=%.3f/%.3f dd_min_max=%.3f/%.3f bars_min_max=%.3f/%.3f",
                getattr(trade, "trade_uid", None),
                getattr(trade, "side", None),
                float(bars_series[-1]),
                float(pnl_series[-1]),
                float(mfe_series[-1]),
                float(dd_series[-1]),
                float(ts_mfe_series[-1]),
                float(atr_series[-1]),
                float(pnl_series.min()),
                float(pnl_series.max()),
                float(mfe_series.min()),
                float(mfe_series.max()),
                float(dd_series.min()),
                float(dd_series.max()),
                float(bars_series.min()),
                float(bars_series.max()),
            )
            self._exit_input_audit_samples_count += 1

        if strict:
            bars_series = window_arr[:, idx["bars_held"]]
            if not np.all(np.diff(bars_series) >= -eps):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] bars_held not non-decreasing")
            ts_mfe_series = window_arr[:, idx["time_since_mfe_bars"]]
            if not np.all(ts_mfe_series >= -eps):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] time_since_mfe_bars negative")
            pnl_series = window_arr[:, idx["pnl_bps_now"]]
            mfe_series = window_arr[:, idx["mfe_bps"]]
            dd_series = window_arr[:, idx["dd_from_mfe_bps"]]
            if not np.all(mfe_series + eps >= pnl_series):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] mfe_bps < pnl_bps_now")
            if not np.all(dd_series + eps >= 0):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] dd_from_mfe_bps negative beyond eps")
            atr_series = window_arr[:, idx["atr_bps_now"]]
            if not np.all(atr_series >= 0):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] atr_bps_now negative")
            # signal7 sanity: probabilities in [0,1] and sum≈1; other channels finite & >=0
            sig = window_arr[:, :7]
            if not np.isfinite(sig).all():
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] non-finite signal7")
            tol = 1e-3
            proba = sig[:, :3]
            if not np.all((proba >= -tol) & (proba <= 1 + tol)):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] p_long/p_short/p_flat out of [0,1]")
            sums = proba.sum(axis=1)
            if not np.all(np.abs(sums - 1.0) <= tol):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] p_long+p_short+p_flat != 1")
            p_hat = sig[:, 3]
            if not np.all((p_hat >= -tol) & (p_hat <= 1 + tol)):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] p_hat out of [0,1]")
            others = sig[:, 4:]
            if not np.all(others >= -tol):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] signal7 other channels negative")

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
