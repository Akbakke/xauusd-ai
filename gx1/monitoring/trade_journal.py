#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trade Journal - Structured logging for trade lifecycle events.

Production-ready trade journal with two output formats:
1. Per-trade JSON files (structured, human-readable)
2. Aggregated index CSV (for filtering and analysis)

Logs complete trade lifecycle:
- Entry snapshot (why trade was taken)
- Feature context (immutable snapshot at entry)
- Router & guardrail explainability
- Exit lifecycle (events and summary)

Usage:
    journal = TradeJournal(run_dir, run_tag, header)
    journal.log_entry_snapshot(trade_id, entry_data)
    journal.log_router_decision(trade_id, router_data)
    journal.log_exit_event(trade_id, event_data)
    journal.log_exit_summary(trade_id, summary_data)
    journal.close()
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# Event type constants (for JSONL compatibility)
EVENT_RUN_START = "RUN_START"
EVENT_ENTRY_SIGNAL = "ENTRY_SIGNAL"
EVENT_ENTRY_BLOCKED = "ENTRY_BLOCKED"
EVENT_ROUTER_DECISION = "ROUTER_DECISION"
EVENT_GUARDRAIL_OVERRIDE = "GUARDRAIL_OVERRIDE"
EVENT_EXIT_TRIGGERED = "EXIT_TRIGGERED"
EVENT_ORDER_SUBMITTED = "ORDER_SUBMITTED"
EVENT_ORDER_REJECTED = "ORDER_REJECTED"
EVENT_ORDER_CANCELLED = "ORDER_CANCELLED"
EVENT_ORDER_FILLED = "ORDER_FILLED"
EVENT_TRADE_OPENED_OANDA = "TRADE_OPENED_OANDA"
EVENT_TRADE_CLOSED_OANDA = "TRADE_CLOSED_OANDA"
EVENT_EXECUTION_RECONCILE_SUMMARY = "EXECUTION_RECONCILE_SUMMARY"
EVENT_TRADE_CLOSED = "TRADE_CLOSED"


def _mask_account_id(account_id: str) -> str:
    """Mask account ID for logging (e.g., '101-004-12345-001' -> '101-***-001')."""
    if not account_id:
        return "MISSING"
    parts = account_id.split("-")
    if len(parts) >= 3:
        return f"{parts[0]}-***-{parts[-1]}"
    return "***"


def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize payload to remove secrets.
    
    Removes:
    - API tokens
    - Full account IDs (masks them)
    """
    sanitized = payload.copy()
    
    # Remove API tokens
    for key in list(sanitized.keys()):
        if "token" in key.lower() or "api_key" in key.lower():
            sanitized.pop(key)
    
    # Mask account IDs
    if "account_id" in sanitized and isinstance(sanitized["account_id"], str):
        sanitized["account_id"] = _mask_account_id(sanitized["account_id"])
    
    # Recursively sanitize nested dicts
    for key, value in sanitized.items():
        if isinstance(value, dict):
            sanitized[key] = _sanitize_payload(value)
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_payload(item) if isinstance(item, dict) else item
                for item in value
            ]
    
    return sanitized


class TradeJournal:
    """
    Production-ready structured trade journal.
    
    Maintains per-trade JSON files and aggregated index CSV.
    """
    
    def __init__(
        self,
        run_dir: Path,
        run_tag: str,
        header: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        """
        Initialize trade journal.
        
        Args:
            run_dir: Output directory for this run
            run_tag: Run tag/identifier
            header: Run header dict (from run_header.json) containing artifact hashes
            enabled: Whether journal is enabled (default: True, always enabled in PROD_BASELINE)
        """
        self.run_dir = Path(run_dir)
        self.run_tag = run_tag
        self.header = header or {}
        self.enabled = enabled
        
        # Create journal directory structure
        self.journal_dir = self.run_dir / "trade_journal"
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-trade JSON directory
        self.trade_json_dir = self.journal_dir / "trades"
        self.trade_json_dir.mkdir(parents=True, exist_ok=True)
        
        # Index CSV path
        self.index_path = self.journal_dir / "trade_journal_index.csv"
        
        # JSONL file path (for backward compatibility)
        self.journal_path = self.journal_dir / "trade_journal.jsonl"
        
        # Extract artifact hashes from header
        artifacts = self.header.get("artifacts", {})
        self.policy_sha256 = artifacts.get("policy", {}).get("sha256", "N/A")
        self.router_sha256 = artifacts.get("router_model", {}).get("sha256", "N/A")
        self.manifest_sha256 = artifacts.get("feature_manifest", {}).get("sha256", "N/A")
        
        # In-memory trade journals (for building complete JSON before write)
        self._trade_journals: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Index data (for CSV)
        self._index_rows: List[Dict[str, Any]] = []
        
        # Open JSONL file in append mode (for backward compatibility)
        self._file_handle = None
        if self.enabled:
            try:
                self._file_handle = open(self.journal_path, "a", encoding="utf-8")
            except Exception as e:
                logger.warning(f"[TRADE_JOURNAL] Failed to open JSONL file: {e}")
                self._file_handle = None
        
        # Initialize index CSV if it doesn't exist
        if self.enabled and not self.index_path.exists():
            self._write_index_header()
    
    def _write_index_header(self) -> None:
        """Write CSV header for index file."""
        try:
            with open(self.index_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "trade_id",
                    "entry_time",
                    "exit_time",
                    "side",
                    "exit_profile",
                    "pnl_bps",
                    "guardrail_applied",
                    "range_edge_dist_atr",
                    "router_decision",
                    "exit_reason",
                    "oanda_trade_id",
                    "oanda_last_txn_id",
                    "execution_status",
                ])
                writer.writeheader()
        except Exception as e:
            logger.warning(f"[TRADE_JOURNAL] Failed to write index header: {e}")
    
    def _get_trade_journal(self, trade_id: str) -> Dict[str, Any]:
        """Get or create trade journal dict."""
        if trade_id not in self._trade_journals:
            self._trade_journals[trade_id] = {
                "trade_id": trade_id,
                "run_tag": self.run_tag,
                "policy_sha256": self.policy_sha256,
                "router_sha256": self.router_sha256,
                "manifest_sha256": self.manifest_sha256,
                "entry_snapshot": None,
                "feature_context": None,
                "router_explainability": None,
                "exit_configuration": None,
                "exit_events": [],
                "execution_events": [],  # Order submission, fills, OANDA events
                "exit_summary": None,
            }
        return self._trade_journals[trade_id]
    
    def _write_trade_json(self, trade_id: str) -> None:
        """Write complete trade journal to JSON file."""
        if not self.enabled:
            return
        
        trade_journal = self._get_trade_journal(trade_id)
        trade_json_path = self.trade_json_dir / f"{trade_id}.json"
        
        try:
            with open(trade_json_path, "w", encoding="utf-8") as f:
                json.dump(trade_journal, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"[TRADE_JOURNAL] Failed to write trade JSON for {trade_id}: {e}")
    
    def log_entry_snapshot(
        self,
        trade_id: str,
        entry_time: str,
        instrument: str,
        side: str,
        entry_price: float,
        session: Optional[str] = None,
        regime: Optional[str] = None,
        entry_model_version: Optional[str] = None,
        entry_score: Optional[Dict[str, float]] = None,
        entry_filters_passed: Optional[List[str]] = None,
        entry_filters_blocked: Optional[List[str]] = None,
        test_mode: bool = False,
        reason: Optional[str] = None,
        warmup_degraded: bool = False,
        cached_bars_at_entry: Optional[int] = None,
        warmup_bars_required: Optional[int] = None,
    ) -> None:
        """
        Log entry snapshot (why trade was taken).
        
        Args:
            trade_id: Trade identifier
            entry_time: Entry timestamp (ISO8601 UTC)
            instrument: Instrument symbol
            side: Trade side (long/short)
            entry_price: Entry price
            session: Trading session (EU/US/OVERLAP)
            regime: Volatility regime
            entry_model_version: Entry model version
            entry_score: Entry model scores/probabilities
            entry_filters_passed: List of filters that passed
            entry_filters_blocked: List of filters that blocked (if any)
            test_mode: If True, marks trade as test mode (e.g., force entry, smoke test)
            reason: Reason for trade (e.g., "FORCED_CANARY_TRADE", "EXECUTION_SMOKE_TEST")
        """
        if not self.enabled:
            return
        
        try:
            trade_journal = self._get_trade_journal(trade_id)
            trade_journal["entry_snapshot"] = {
                "trade_id": trade_id,
                "entry_time": entry_time,
                "instrument": instrument,
                "side": side,
                "entry_price": entry_price,
                "session": session,
                "regime": regime,
                "entry_model_version": entry_model_version,
                "entry_score": entry_score or {},
                "entry_filters_passed": entry_filters_passed or [],
                "entry_filters_blocked": entry_filters_blocked or [],
                "test_mode": test_mode,
            }
            if reason:
                trade_journal["entry_snapshot"]["reason"] = reason
            # Add degraded warmup fields if applicable
            if warmup_degraded:
                trade_journal["entry_snapshot"]["warmup_degraded"] = True
                if cached_bars_at_entry is not None:
                    trade_journal["entry_snapshot"]["cached_bars_at_entry"] = cached_bars_at_entry
                if warmup_bars_required is not None:
                    trade_journal["entry_snapshot"]["warmup_bars_required"] = warmup_bars_required
            self._write_trade_json(trade_id)
        except Exception as e:
            logger.warning(f"[TRADE_JOURNAL] Failed to log entry snapshot for {trade_id}: {e}")
    
    def log_feature_context(
        self,
        trade_id: str,
        atr_bps: Optional[float] = None,
        atr_price: Optional[float] = None,
        atr_percentile: Optional[float] = None,
        range_pos: Optional[float] = None,
        distance_to_range: Optional[float] = None,
        range_edge_dist_atr: Optional[float] = None,
        spread_price: Optional[float] = None,
        spread_pct: Optional[float] = None,
        candle_close: Optional[float] = None,
        candle_high: Optional[float] = None,
        candle_low: Optional[float] = None,
    ) -> None:
        """
        Log feature context (immutable snapshot at entry).
        
        Args:
            trade_id: Trade identifier
            atr_bps: ATR in basis points
            atr_price: ATR in price units
            atr_percentile: ATR percentile rank
            range_pos: Range position [0.0, 1.0]
            distance_to_range: Distance to range [0.0, 1.0]
            range_edge_dist_atr: Range edge distance (ATR-normalized)
            spread_price: Spread in price units
            spread_pct: Spread percentile
            candle_close: Last closed bar close price
            candle_high: Last closed bar high price
            candle_low: Last closed bar low price
        """
        if not self.enabled:
            return
        
        try:
            trade_journal = self._get_trade_journal(trade_id)
            trade_journal["feature_context"] = {
                "atr": {
                    "atr_bps": atr_bps,
                    "atr_price": atr_price,
                    "atr_percentile": atr_percentile,
                },
                "range": {
                    "range_pos": range_pos,
                    "distance_to_range": distance_to_range,
                    "range_edge_dist_atr": range_edge_dist_atr,
                },
                "spread": {
                    "spread_price": spread_price,
                    "spread_pct": spread_pct,
                },
                "candle": {
                    "close": candle_close,
                    "high": candle_high,
                    "low": candle_low,
                },
            }
            self._write_trade_json(trade_id)
        except Exception as e:
            logger.warning(f"[TRADE_JOURNAL] Failed to log feature context for {trade_id}: {e}")
    
    def log_router_decision(
        self,
        trade_id: str,
        router_version: str,
        router_raw_decision: str,
        final_exit_profile: str,
        router_model_hash: Optional[str] = None,
        router_features_used: Optional[Dict[str, Any]] = None,
        guardrail_applied: bool = False,
        guardrail_reason: Optional[str] = None,
        guardrail_cutoff: Optional[float] = None,
        range_edge_dist_atr: Optional[float] = None,
    ) -> None:
        """
        Log router decision and guardrail explainability.
        
        Args:
            trade_id: Trade identifier
            router_version: Router version (V3, V3_RANGE, etc.)
            router_model_hash: Router model SHA256 hash
            router_features_used: Dict of features used by router
            router_raw_decision: Raw router decision (RULE5 or RULE6A)
            guardrail_applied: Whether guardrail was applied
            guardrail_reason: Reason for guardrail override
            guardrail_cutoff: Guardrail cutoff value
            range_edge_dist_atr: Range edge distance ATR (for guardrail)
            final_exit_profile: Final exit profile after guardrail
        """
        if not self.enabled:
            return
        
        try:
            trade_journal = self._get_trade_journal(trade_id)
            trade_journal["router_explainability"] = {
                "router_version": router_version,
                "router_model_hash": router_model_hash or self.router_sha256,
                "router_features_used": router_features_used or {},
                "router_raw_decision": router_raw_decision,
                "guardrail_applied": guardrail_applied,
                "guardrail_reason": guardrail_reason,
                "guardrail_cutoff": guardrail_cutoff,
                "range_edge_dist_atr": range_edge_dist_atr,
                "final_exit_profile": final_exit_profile,
            }
            self._write_trade_json(trade_id)
        except Exception as e:
            logger.warning(f"[TRADE_JOURNAL] Failed to log router decision for {trade_id}: {e}")
    
    def log_exit_configuration(
        self,
        trade_id: str,
        exit_profile: str,
        tp_levels: Optional[List[float]] = None,
        sl: Optional[float] = None,
        trailing_enabled: bool = False,
        be_rules: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log exit configuration.
        
        Args:
            trade_id: Trade identifier
            exit_profile: Exit profile name
            tp_levels: Take profit levels
            sl: Stop loss level
            trailing_enabled: Whether trailing stop is enabled
            be_rules: Break-even rules dict
        """
        if not self.enabled:
            return
        
        try:
            trade_journal = self._get_trade_journal(trade_id)
            trade_journal["exit_configuration"] = {
                "exit_profile": exit_profile,
                "tp_levels": tp_levels or [],
                "sl": sl,
                "trailing_enabled": trailing_enabled,
                "be_rules": be_rules or {},
            }
            self._write_trade_json(trade_id)
        except Exception as e:
            logger.warning(f"[TRADE_JOURNAL] Failed to log exit configuration for {trade_id}: {e}")
    
    def log_exit_event(
        self,
        trade_id: str,
        timestamp: str,
        event_type: str,
        price: Optional[float] = None,
        pnl_bps: Optional[float] = None,
        bars_held: Optional[int] = None,
    ) -> None:
        """
        Log exit event (append-only).
        
        Args:
            trade_id: Trade identifier
            timestamp: Event timestamp (ISO8601 UTC)
            event_type: Event type (TP1_HIT, BE_MOVED, TRAIL_ACTIVATED, EXIT_TRIGGERED, etc.)
            price: Price at event
            pnl_bps: PnL in basis points (unrealized or realized)
            bars_held: Bars held at event
        """
        if not self.enabled:
            return
        
        try:
            trade_journal = self._get_trade_journal(trade_id)
            event = {
                "timestamp": timestamp,
                "event_type": event_type,
                "price": price,
                "pnl_bps": pnl_bps,
                "bars_held": bars_held,
            }
            trade_journal["exit_events"].append(event)
            self._write_trade_json(trade_id)
        except Exception as e:
            logger.warning(f"[TRADE_JOURNAL] Failed to log exit event for {trade_id}: {e}")
    
    def log_exit_summary(
        self,
        trade_id: str,
        exit_time: str,
        exit_price: float,
        exit_reason: str,
        realized_pnl_bps: float,
        max_mfe_bps: Optional[float] = None,
        max_mae_bps: Optional[float] = None,
        intratrade_drawdown_bps: Optional[float] = None,
    ) -> None:
        """
        Log exit summary (final trade closure).
        
        Args:
            trade_id: Trade identifier
            exit_time: Exit timestamp (ISO8601 UTC)
            exit_price: Exit price
            exit_reason: Exit reason (TP, SL, BE, TIMEOUT, TRAIL, etc.)
            realized_pnl_bps: Realized PnL in basis points
            max_mfe_bps: Maximum favorable excursion (bps)
            max_mae_bps: Maximum adverse excursion (bps)
            intratrade_drawdown_bps: Intratrade drawdown (bps)
        """
        if not self.enabled:
            return
        
        try:
            trade_journal = self._get_trade_journal(trade_id)
            trade_journal["exit_summary"] = {
                "exit_time": exit_time,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "realized_pnl_bps": realized_pnl_bps,
                "max_mfe_bps": max_mfe_bps,
                "max_mae_bps": max_mae_bps,
                "intratrade_drawdown_bps": intratrade_drawdown_bps,
            }
            self._write_trade_json(trade_id)
            
            # Update index CSV
            self._update_index(trade_id)
        except Exception as e:
            logger.warning(f"[TRADE_JOURNAL] Failed to log exit summary for {trade_id}: {e}")
    
    def _update_index(self, trade_id: str) -> None:
        """Update index CSV with trade summary."""
        if not self.enabled:
            return
        
        try:
            trade_journal = self._get_trade_journal(trade_id)
            entry_snapshot = trade_journal.get("entry_snapshot") or {}
            router_explainability = trade_journal.get("router_explainability") or {}
            exit_summary = trade_journal.get("exit_summary") or {}
            
            # Extract OANDA IDs from execution events
            execution_events = trade_journal.get("execution_events", [])
            oanda_trade_id = None
            oanda_last_txn_id = None
            execution_status = "UNKNOWN"
            
            for event in execution_events:
                if event.get("event_type") == "ORDER_FILLED":
                    oanda_trade_id = event.get("oanda_trade_id") or oanda_trade_id
                    oanda_last_txn_id = event.get("oanda_transaction_id") or oanda_last_txn_id
                    execution_status = "OK"
                elif event.get("event_type") == "ORDER_REJECTED":
                    execution_status = "REJECTED"
                elif event.get("event_type") == "TRADE_CLOSED_OANDA":
                    oanda_last_txn_id = event.get("oanda_transaction_id") or oanda_last_txn_id
            
            index_row = {
                "trade_id": trade_id,
                "entry_time": entry_snapshot.get("entry_time", ""),
                "exit_time": exit_summary.get("exit_time", ""),
                "side": entry_snapshot.get("side", ""),
                "exit_profile": router_explainability.get("final_exit_profile", ""),
                "pnl_bps": exit_summary.get("realized_pnl_bps", ""),
                "guardrail_applied": router_explainability.get("guardrail_applied", False),
                "range_edge_dist_atr": router_explainability.get("range_edge_dist_atr", ""),
                "router_decision": router_explainability.get("router_raw_decision", ""),
                "exit_reason": exit_summary.get("exit_reason", ""),
                "oanda_trade_id": oanda_trade_id or "",
                "oanda_last_txn_id": oanda_last_txn_id or "",
                "execution_status": execution_status,
            }
            
            # Append to CSV
            # Ensure CSV has correct headers (write header if file is new)
            write_header = not self.index_path.exists() or self.index_path.stat().st_size == 0
            
            fieldnames = [
                "trade_id",
                "entry_time",
                "exit_time",
                "side",
                "exit_profile",
                "pnl_bps",
                "guardrail_applied",
                "range_edge_dist_atr",
                "router_decision",
                "exit_reason",
                "oanda_trade_id",
                "oanda_last_txn_id",
                "execution_status",
            ]
            
            with open(self.index_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(index_row)
        except Exception as e:
            logger.warning(f"[TRADE_JOURNAL] Failed to update index for {trade_id}: {e}")
    
    # Backward compatibility: JSONL logging
    def log(
        self,
        event_type: str,
        payload: Dict[str, Any],
        trade_key: Optional[Dict[str, Any]] = None,
        trade_id: Optional[str] = None,
    ) -> None:
        """
        Log an event to JSONL file (backward compatibility).
        
        Args:
            event_type: Event type constant (e.g., EVENT_ENTRY_SIGNAL)
            payload: Event-specific payload dict
            trade_key: Trade identifier dict (entry_time, entry_price, side)
            trade_id: Trade ID if available
        """
        if not self.enabled or self._file_handle is None:
            return
        
        try:
            # Build event record
            event = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "run_tag": self.run_tag,
                "policy_sha256": self.policy_sha256,
                "router_sha256": self.router_sha256,
                "manifest_sha256": self.manifest_sha256,
            }
            
            # Add trade identifiers
            if trade_id:
                event["trade_id"] = trade_id
            if trade_key:
                event["trade_key"] = trade_key
            
            # Add sanitized payload
            event["payload"] = _sanitize_payload(payload)
            
            # Write as JSONL (one JSON object per line)
            json_line = json.dumps(event, ensure_ascii=False, default=str)
            self._file_handle.write(json_line + "\n")
            self._file_handle.flush()  # Crash safety: flush immediately
            
        except Exception as e:
            # Never break trading due to journal failure
            logger.warning(f"[TRADE_JOURNAL] Failed to log event {event_type}: {e}")
    
    def log_order_submitted(
        self,
        trade_id: str,
        instrument: str,
        side: str,
        units: int,
        order_type: str,
        client_order_id: Optional[str] = None,
        client_ext_id: Optional[str] = None,
        client_ext_tag: Optional[str] = None,
        client_ext_comment: Optional[str] = None,
        requested_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        oanda_env: Optional[str] = None,
        account_id_masked: Optional[str] = None,
    ) -> None:
        """
        Log order submitted event.
        
        Args:
            trade_id: Trade identifier
            instrument: Instrument symbol
            side: Trade side (long/short)
            units: Number of units
            order_type: Order type (MARKET/LIMIT/STOP)
            client_order_id: Client order ID
            client_ext_id: Client extensions ID
            client_ext_tag: Client extensions tag
            client_ext_comment: Client extensions comment
            requested_price: Requested price (if applicable)
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            oanda_env: OANDA environment (practice/live)
            account_id_masked: Masked account ID
        """
        if not self.enabled:
            return
        
        try:
            trade_journal = self._get_trade_journal(trade_id)
            if "execution_events" not in trade_journal:
                trade_journal["execution_events"] = []
            
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "ORDER_SUBMITTED",
                "instrument": instrument,
                "side": side,
                "units": units,
                "order_type": order_type,
                "client_order_id": client_order_id,
                "client_extensions": {
                    "id": client_ext_id,
                    "tag": client_ext_tag,
                    "comment": client_ext_comment,
                },
                "requested_price": requested_price,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "oanda_env": oanda_env,
                "account_id_masked": account_id_masked,
            }
            
            trade_journal["execution_events"].append(event)
            self._write_trade_json(trade_id)
        except Exception as e:
            logger.warning(f"[TRADE_JOURNAL] Failed to log ORDER_SUBMITTED for {trade_id}: {e}")
    
    def log_order_rejected(
        self,
        trade_id: str,
        client_order_id: Optional[str] = None,
        status_code: Optional[int] = None,
        reject_reason: Optional[str] = None,
        response_body: Optional[str] = None,
    ) -> None:
        """
        Log order rejected event.
        
        Args:
            trade_id: Trade identifier
            client_order_id: Client order ID
            status_code: HTTP status code
            reject_reason: Rejection reason
            response_body: Response body snippet (sanitized)
        """
        if not self.enabled:
            return
        
        try:
            trade_journal = self._get_trade_journal(trade_id)
            if "execution_events" not in trade_journal:
                trade_journal["execution_events"] = []
            
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "ORDER_REJECTED",
                "client_order_id": client_order_id,
                "status_code": status_code,
                "reject_reason": reject_reason,
                "response_body": response_body[:500] if response_body else None,  # Limit size
            }
            
            trade_journal["execution_events"].append(event)
            self._write_trade_json(trade_id)
        except Exception as e:
            logger.warning(f"[TRADE_JOURNAL] Failed to log ORDER_REJECTED for {trade_id}: {e}")
    
    def log_order_filled(
        self,
        trade_id: str,
        oanda_order_id: Optional[str] = None,
        oanda_trade_id: Optional[str] = None,
        oanda_transaction_id: Optional[str] = None,
        fill_price: Optional[float] = None,
        fill_units: Optional[int] = None,
        commission: Optional[float] = None,
        financing: Optional[float] = None,
        pl: Optional[float] = None,
        ts_oanda: Optional[str] = None,
    ) -> None:
        """
        Log order filled event.
        
        Args:
            trade_id: Trade identifier
            oanda_order_id: OANDA order ID
            oanda_trade_id: OANDA trade ID
            oanda_transaction_id: OANDA transaction ID
            fill_price: Fill price
            fill_units: Fill units
            commission: Commission
            financing: Financing
            pl: Profit/loss
            ts_oanda: OANDA timestamp (RFC3339)
        """
        if not self.enabled:
            return
        
        try:
            trade_journal = self._get_trade_journal(trade_id)
            if "execution_events" not in trade_journal:
                trade_journal["execution_events"] = []
            
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "ORDER_FILLED",
                "oanda_order_id": oanda_order_id,
                "oanda_trade_id": oanda_trade_id,
                "oanda_transaction_id": oanda_transaction_id,
                "fill_price": fill_price,
                "fill_units": fill_units,
                "commission": commission,
                "financing": financing,
                "pl": pl,
                "ts_oanda": ts_oanda,
            }
            
            trade_journal["execution_events"].append(event)
            self._write_trade_json(trade_id)
        except Exception as e:
            logger.warning(f"[TRADE_JOURNAL] Failed to log ORDER_FILLED for {trade_id}: {e}")
    
    def log_oanda_trade_update(
        self,
        trade_id: str,
        event_type: str,  # TRADE_OPENED_OANDA or TRADE_CLOSED_OANDA
        oanda_trade_id: Optional[str] = None,
        oanda_transaction_id: Optional[str] = None,
        price: Optional[float] = None,
        units: Optional[int] = None,
        pl: Optional[float] = None,
        ts_oanda: Optional[str] = None,
    ) -> None:
        """
        Log OANDA trade update (opened/closed).
        
        Args:
            trade_id: Trade identifier
            event_type: Event type (TRADE_OPENED_OANDA or TRADE_CLOSED_OANDA)
            oanda_trade_id: OANDA trade ID
            oanda_transaction_id: OANDA transaction ID
            price: Price
            units: Units
            pl: Profit/loss
            ts_oanda: OANDA timestamp (RFC3339)
        """
        if not self.enabled:
            return
        
        try:
            trade_journal = self._get_trade_journal(trade_id)
            if "execution_events" not in trade_journal:
                trade_journal["execution_events"] = []
            
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "oanda_trade_id": oanda_trade_id,
                "oanda_transaction_id": oanda_transaction_id,
                "price": price,
                "units": units,
                "pl": pl,
                "ts_oanda": ts_oanda,
            }
            
            trade_journal["execution_events"].append(event)
            self._write_trade_json(trade_id)
        except Exception as e:
            logger.warning(f"[TRADE_JOURNAL] Failed to log {event_type} for {trade_id}: {e}")
    
    def close(self) -> None:
        """Close journal and flush all pending writes."""
        # Write any remaining trade JSONs
        for trade_id in list(self._trade_journals.keys()):
            self._write_trade_json(trade_id)
        
        # Close JSONL file
        if self._file_handle:
            try:
                self._file_handle.close()
            except Exception:
                pass
            self._file_handle = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
