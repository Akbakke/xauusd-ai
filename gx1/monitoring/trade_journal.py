#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trade Journal - Structured logging for trade lifecycle events.

Strict trade journal with two output formats:
1. Per-trade JSON files (structured, human-readable)
2. Aggregated index CSV (for filtering and analysis)

Logs complete trade lifecycle:
- Entry snapshot (why trade was taken)
- Feature context (immutable snapshot at entry)
- Router & guardrail explainability
- Exit lifecycle (events and summary)

Usage:
    journal = TradeJournal(run_dir, run_tag, header)
    journal.log_entry_snapshot(..., model_evidence=validated_entry_snapshot)
    journal.log_exit_summary(trade_id, summary_data)
    journal.close()
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS,
    MODEL_NATIVE_RUNTIME_POLICY,
    require_model_native_fill_time,
    require_model_native_runtime_evidence,
)
from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    require_model_native_sizing_application_record,
)
from gx1.models.entry_v10.direction_decision_contract import (
    require_unified_exit_output,
    require_unified_exit_path_envelope,
)

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


# One versioned CSV contract.  The old SMART/SNIPER/anchor index mixed learned
# evidence with live-only gates and could not prove what actually selected the
# direction.  Lists/dicts are serialized as canonical JSON by _csv_value().
MODEL_NATIVE_INDEX_FIELDS = (
    "trade_key",
    "trade_uid",
    "trade_id",
    "run_tag",
    "entry_time",
    "exit_time",
    "side",
    "instrument",
    "model_policy",
    "runtime_evidence_schema_version",
    "session",
    "session_id",
    "entry_price",
    "entry_bid",
    "entry_ask",
    "entry_spread_bps",
    "atr_bps",
    "units",
    "applied_size_multiplier",
    "sizing_mode",
    "calibrated_size_fraction",
    "capacity_units",
    "reference_pre_round_units",
    "pre_round_units",
    "sizing_calibration_artifact_sha256",
    "sizing_oos_proof_artifact_sha256",
    "sizing_adoption_artifact_sha256",
    "execution_checks",
    "model_direction",
    "model_direction_index",
    "direction_logits",
    "direction_probs",
    "public_trade_flat_decision",
    "public_trade_flat_decision_index",
    "public_trade_flat_decision_logits",
    "public_trade_flat_decision_probs",
    "p_trade",
    "p_flat_hier",
    "p_long_given_trade",
    "p_short_given_trade",
    "path_quality",
    "path_quality_log_var",
    "path_quality_std",
    "mfe_first_n",
    "tradable_prob",
    "bad_path_prob",
    "clean_edge_prob",
    "survival_prob",
    "tf_agreement_logit",
    "tf_agreement_pred",
    "position_size_logit",
    "position_size_pred",
    "side_utility",
    "side_bad_path_logit",
    "long_bad_path_prob",
    "short_bad_path_prob",
    "side_validity_logit",
    "long_validity_prob",
    "short_validity_prob",
    "side_mae",
    "mtf_dir_logits",
    "mtf_dir_probs",
    "mtf_trend_evidence",
    "specialist_names",
    "specialist_gate",
    "trendline_rail_logits",
    "trendline_rail_probs",
    "geometry_channel_edge_pressure",
    "geometry_rising_support_rail_long_pressure",
    "geometry_rising_support_rail_short_trap_pressure",
    "geometry_falling_resistance_rail_short_pressure",
    "geometry_falling_resistance_rail_long_trap_pressure",
    "calibration_version",
    "direction_calibration_enabled",
    "direction_calibration_temperature",
    "path_calibration_enabled",
    "path_calibration",
    "decision_ts",
    "decision_available_ts",
    "entry_signal_latency_sec",
    "context_cutoff_ts",
    "context_age_m5_bars",
    "pnl_bps",
    "exit_reason",
    "oanda_trade_id",
    "oanda_last_txn_id",
    "execution_status",
)


def _csv_value(value: Any) -> Any:
    """Return a deterministic CSV cell without erasing valid zero/False."""
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return value


def _require_complete_journal_model_evidence(
    evidence: Dict[str, Any],
    *,
    context: str,
) -> Dict[str, Any]:
    validated = require_model_native_runtime_evidence(evidence, context=context)
    if not MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS.issubset(validated):
        raise RuntimeError(
            f"[{context}_MODEL_NATIVE_TIMING_EVIDENCE_MISSING] "
            "the persisted live snapshot must contain the complete timing contract"
        )
    return validated


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
        
        # The versioned filename prevents stale SMART-era rows/headers from
        # being appended to the model-native evidence contract.
        self.index_path = self.journal_dir / "trade_journal_index_model_native_v1.csv"
        
        # One JSONL event stream.  The former run_dir/journal mirror was an
        # unverified mutable duplicate and is deliberately retired.
        self.journal_path = self.journal_dir / "trade_journal.jsonl"
        
        # Extract artifact hashes from header
        artifacts = self.header.get("artifacts", {})
        self.policy_sha256 = artifacts.get("policy", {}).get("sha256", "N/A")
        self.router_sha256 = artifacts.get("router_model", {}).get("sha256", "N/A")
        self.manifest_sha256 = artifacts.get("feature_manifest", {}).get("sha256", "N/A")
        
        # In-memory trade journals (for building complete JSON before write)
        self._trade_journals: Dict[str, Dict[str, Any]] = {}
        
        # Open JSONL file in append mode.
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

    def _is_replay_context(self) -> bool:
        """Replay hard-fail guards are driven by replay env/tag, not generic TEST names."""
        tag = str(self.run_tag or "").upper()
        return (
            os.getenv("GX1_REPLAY") == "1"
            or os.getenv("REPLAY_MODE") == "1"
            or tag.startswith("REPLAY_")
            or "REPLAY" in tag
        )

    def _is_execution_smoke_context(self) -> bool:
        """Allow empty model evidence only for an explicitly marked broker smoke."""
        meta = self.header.get("meta")
        return (
            isinstance(meta, dict)
            and meta.get("test_mode") is True
            and str(self.run_tag or "").upper().startswith("EXEC_SMOKE")
        )

    def _normalize_ids(
        self,
        trade_uid: Optional[str] = None,
        trade_id: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Normalize the optional global UID and broker/display trade ID."""
        if trade_id is None and trade_uid:
            value = str(trade_uid)
            if ":" not in value:
                return None, value
        return trade_uid, trade_id
    
    def _key(self, trade_uid: Optional[str] = None, trade_id: Optional[str] = None) -> str:
        """
        Normalize trade identifier to internal key (COMMIT C).
        
        Priority:
        1. trade_uid (globally unique replay ID)
        2. trade_id (broker/display ID, namespaced as TRADE:{trade_id})
        3. Raise error if neither provided
        
        Parameters
        ----------
        trade_uid : str, optional
            Globally unique trade identifier (run_id:chunk_id:seq:uuid)
        trade_id : str, optional
            Legacy trade identifier (SIM-...)
        
        Returns
        -------
        str
            Internal key for storage/indexing
        """
        trade_uid, trade_id = self._normalize_ids(trade_uid=trade_uid, trade_id=trade_id)
        if trade_uid:
            return trade_uid
        elif trade_id:
            return f"TRADE:{trade_id}"
        else:
            raise ValueError("Either trade_uid or trade_id must be provided")
    
    def _write_index_header(self) -> None:
        """Write CSV header for index file."""
        try:
            with open(self.index_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=MODEL_NATIVE_INDEX_FIELDS)
                writer.writeheader()
        except Exception as e:
            logger.warning(f"[TRADE_JOURNAL] Failed to write index header: {e}")
    
    def _get_trade_journal(self, trade_uid: Optional[str] = None, trade_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get or create trade journal dict (COMMIT C: backward-compatible wrapper).
        
        Parameters
        ----------
        trade_uid : str, optional
            Globally unique trade identifier (preferred)
        trade_id : str, optional
            Legacy trade identifier (fallback)
        
        Returns
        -------
        Dict[str, Any]
            Trade journal dict
        """
        trade_uid, trade_id = self._normalize_ids(trade_uid=trade_uid, trade_id=trade_id)
        # GUARD 1: Replay-only fail-fast - trade_uid format invariant
        is_replay = self._is_replay_context()
        
        if is_replay and trade_uid:
            # In replay, trade_uid must start with GX1_RUN_ID:GX1_CHUNK_ID:
            env_run_id = os.getenv("GX1_RUN_ID")
            env_chunk_id = os.getenv("GX1_CHUNK_ID")
            if env_run_id and env_chunk_id:
                expected_prefix = f"{env_run_id}:{env_chunk_id}:"
                if not trade_uid.startswith(expected_prefix):
                    raise RuntimeError(
                        f"BAD_TRADE_UID_FORMAT_REPLAY: trade_uid={trade_uid} does not start with "
                        f"expected prefix={expected_prefix}. GX1_RUN_ID={env_run_id}, GX1_CHUNK_ID={env_chunk_id}. "
                        f"This is a hard contract violation in replay mode."
                    )
        
        key = self._key(trade_uid=trade_uid, trade_id=trade_id)
        if key not in self._trade_journals:
            file_stem = trade_id if trade_uid is None and trade_id else key
            persisted_path = self.trade_json_dir / f"{file_stem}.json"
            if persisted_path.is_file():
                try:
                    persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise RuntimeError(
                        f"[TRADE_JOURNAL_PERSISTED_READ_FAILED] path={persisted_path}"
                    ) from exc
                if not isinstance(persisted, dict) or persisted.get("trade_key") != key:
                    raise RuntimeError(
                        "[TRADE_JOURNAL_PERSISTED_ID_MISMATCH] "
                        f"path={persisted_path} expected_key={key!r} "
                        f"observed_key={getattr(persisted, 'get', lambda *_: None)('trade_key')!r}"
                    )
                if persisted.get("trade_uid") != trade_uid or persisted.get("trade_id") != trade_id:
                    raise RuntimeError(
                        "[TRADE_JOURNAL_PERSISTED_ID_MISMATCH] "
                        f"path={persisted_path}"
                    )
                persisted_entry = persisted.get("entry_snapshot")
                if persisted_entry is not None:
                    if not isinstance(persisted_entry, dict):
                        raise RuntimeError(
                            f"[TRADE_JOURNAL_PERSISTED_ENTRY_INVALID] path={persisted_path}"
                        )
                    persisted_evidence = persisted_entry.get("model_evidence")
                    if persisted_evidence:
                        persisted_evidence = _require_complete_journal_model_evidence(
                            dict(persisted_evidence),
                            context="TRADE_JOURNAL_PERSISTED",
                        )
                        require_model_native_fill_time(
                            persisted_evidence,
                            persisted_entry.get("entry_time"),
                            context="TRADE_JOURNAL_PERSISTED",
                        )
                    elif persisted_evidence != {} or not self._is_execution_smoke_context():
                        raise RuntimeError(
                            "[TRADE_JOURNAL_PERSISTED_ENTRY_SCHEMA_RETIRED] "
                            f"path={persisted_path}"
                        )
                self._trade_journals[key] = persisted

        journal_exists = key in self._trade_journals
        
        # GUARD 2: In replay mode, never create new journal in exit path
        # If journal doesn't exist and we're in a context where entry_snapshot should exist, fail hard
        if is_replay and not journal_exists:
            # Check if this is being called from exit path (heuristic: trade_id provided but no trade_uid, or trade_uid doesn't match expected format)
            if trade_id and (not trade_uid or (trade_uid and not trade_uid.startswith(f"{env_run_id}:{env_chunk_id}:") if env_run_id and env_chunk_id else False)):
                raise RuntimeError(
                    f"EXIT_WITHOUT_ENTRY_SNAPSHOT_REPLAY: Attempted to create new journal for trade_id={trade_id}, "
                    f"trade_uid={trade_uid} but journal does not exist. This indicates exit logging attempted "
                    f"without entry_snapshot being logged first. This is a hard contract violation in replay mode."
                )
        
        if key not in self._trade_journals:
            self._trade_journals[key] = {
                "trade_key": key,  # Internal primary key
                "trade_uid": trade_uid,  # Globally unique ID (COMMIT C)
                "trade_id": trade_id,
                "run_tag": self.run_tag,
                "policy_sha256": self.policy_sha256,
                "router_sha256": self.router_sha256,
                "manifest_sha256": self.manifest_sha256,
                "entry_snapshot": None,
                "execution_events": [],  # Order submission, fills, OANDA events
                "exit_summary": None,
            }
        return self._trade_journals[key]
    
    def _write_trade_json(self, trade_uid: Optional[str] = None, trade_id: Optional[str] = None) -> None:
        """
        Write complete trade journal to JSON file (COMMIT C: backward-compatible wrapper).
        
        Parameters
        ----------
        trade_uid : str, optional
            Globally unique trade identifier (preferred)
        trade_id : str, optional
            Legacy trade identifier (fallback)
        """
        import time
        
        if not self.enabled:
            return
        trade_uid, trade_id = self._normalize_ids(trade_uid=trade_uid, trade_id=trade_id)
        
        # GUARD 1: Replay-only fail-fast - trade_uid format invariant when writing
        is_replay = self._is_replay_context()
        
        if is_replay and trade_uid:
            env_run_id = os.getenv("GX1_RUN_ID")
            env_chunk_id = os.getenv("GX1_CHUNK_ID")
            if env_run_id and env_chunk_id:
                expected_prefix = f"{env_run_id}:{env_chunk_id}:"
                if not trade_uid.startswith(expected_prefix):
                    raise RuntimeError(
                        f"BAD_TRADE_UID_FORMAT_REPLAY: Attempted to write journal with trade_uid={trade_uid} "
                        f"that does not start with expected prefix={expected_prefix}. "
                        f"GX1_RUN_ID={env_run_id}, GX1_CHUNK_ID={env_chunk_id}. "
                        f"This is a hard contract violation in replay mode."
                    )
        
        key = self._key(trade_uid=trade_uid, trade_id=trade_id)
        trade_journal = self._get_trade_journal(trade_uid=trade_uid, trade_id=trade_id)
        file_stem = trade_id if trade_uid is None and trade_id else key
        trade_json_path = self.trade_json_dir / f"{file_stem}.json"
        
        try:
            # Time the actual I/O operation
            io_start = time.perf_counter()
            tmp_path = trade_json_path.with_suffix(
                trade_json_path.suffix + ".tmp"
            )
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(trade_journal, f, indent=2, ensure_ascii=False, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, trade_json_path)
            io_time = time.perf_counter() - io_start
            
            # Log journal I/O time (only periodically to avoid spam)
            if not hasattr(self, "_journal_io_log_count"):
                self._journal_io_log_count = 0
            self._journal_io_log_count += 1
            if self._journal_io_log_count % 100 == 0:
                logger.debug(f"[JOURNAL_PERF] Wrote {self._journal_io_log_count} trade JSONs, last I/O time: {io_time*1000:.2f}ms")
        except Exception as e:
            logger.warning(f"[TRADE_JOURNAL] Failed to write trade JSON for key={key}: {e}")
            raise
    
    def log_entry_snapshot(
        self,
        entry_time: str,
        instrument: str,
        side: str,
        entry_price: float,
        model_evidence: Dict[str, Any],
        entry_bid: Optional[float] = None,
        entry_ask: Optional[float] = None,
        entry_spread_bps: Optional[float] = None,
        trade_uid: Optional[str] = None,  # Globally unique ID (COMMIT C)
        trade_id: Optional[str] = None,
        session: Optional[str] = None,
        model_policy: Optional[str] = None,
        execution_checks: Optional[List[str]] = None,
        capacity_units: Optional[int] = None,
        reference_pre_round_units: Optional[float] = None,
        pre_round_units: Optional[float] = None,
        units: Optional[int] = None,
        applied_size_multiplier: Optional[float] = None,
        sizing_application: Optional[Dict[str, Any]] = None,
        atr_bps: Optional[float] = None,
        model_bundle_binding: Optional[Dict[str, Any]] = None,
        entry_source_pair_binding: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist the exact model evidence and the separate execution facts.

        A non-empty ``model_evidence`` is the frozen snapshot already consumed
        by TradeState.  It is revalidated here so the journal cannot silently
        replace a missing head, record a different direction, or imply that the
        learned sizing head controlled exposure.  Empty evidence is accepted
        only for non-direction execution smoke records.
        """
        if not self.enabled:
            return

        try:
            trade_journal = self._get_trade_journal(trade_uid=trade_uid, trade_id=trade_id)
            evidence = dict(model_evidence)
            if evidence:
                evidence = _require_complete_journal_model_evidence(
                    evidence,
                    context="TRADE_JOURNAL_ENTRY",
                )
                require_model_native_fill_time(
                    evidence,
                    entry_time,
                    context="TRADE_JOURNAL_ENTRY",
                )
                expected_side = str(evidence["model_direction"]).lower()
                if expected_side not in ("long", "short") or side != expected_side:
                    raise RuntimeError(
                        "[TRADE_JOURNAL_DIRECTION_PARITY] "
                        f"journal_side={side!r} model_direction={evidence['model_direction']!r}"
                    )
                if (
                    model_policy != MODEL_NATIVE_RUNTIME_POLICY
                    or model_policy != evidence["model_policy"]
                ):
                    raise RuntimeError(
                        "[TRADE_JOURNAL_MODEL_POLICY_MISMATCH] "
                        f"observed={model_policy!r} "
                        f"snapshot={evidence['model_policy']!r} "
                        f"expected={MODEL_NATIVE_RUNTIME_POLICY!r}"
                    )
                if session != evidence["session"]:
                    raise RuntimeError(
                        "[TRADE_JOURNAL_SESSION_EVIDENCE_MISMATCH] "
                        f"observed={session!r} snapshot={evidence['session']!r}"
                    )
                if not execution_checks:
                    raise RuntimeError("[TRADE_JOURNAL_EXECUTION_CHECKS_MISSING]")
                if (
                    capacity_units is None
                    or reference_pre_round_units is None
                    or pre_round_units is None
                    or units is None
                    or applied_size_multiplier is None
                ):
                    raise RuntimeError("[TRADE_JOURNAL_APPLIED_SIZE_CONTRACT_MISSING]")
                validated_sizing = require_model_native_sizing_application_record(
                    sizing_application,
                    context="TRADE_JOURNAL_ENTRY",
                )
                sizing_parity = {
                    "capacity_units": capacity_units,
                    "reference_pre_round_units": reference_pre_round_units,
                    "pre_round_units": pre_round_units,
                    "units": units,
                    "applied_size_multiplier": applied_size_multiplier,
                    "model_direction": evidence["model_direction"],
                    "position_size_logit": evidence["position_size_logit"],
                }
                mismatched_sizing = sorted(
                    key
                    for key, value in sizing_parity.items()
                    if validated_sizing.get(key) != value
                )
                if mismatched_sizing:
                    raise RuntimeError(
                        "[TRADE_JOURNAL_APPLIED_SIZE_PARITY_MISMATCH] "
                        + ",".join(mismatched_sizing)
                    )
                required_market_values = {
                    "entry_price": entry_price,
                    "entry_bid": entry_bid,
                    "entry_ask": entry_ask,
                    "entry_spread_bps": entry_spread_bps,
                    "atr_bps": atr_bps,
                }
                parsed_market_values: Dict[str, float] = {}
                for name, value in required_market_values.items():
                    try:
                        parsed = float(value)
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            f"[TRADE_JOURNAL_MARKET_EVIDENCE_INVALID] {name}={value!r}"
                        ) from exc
                    if not math.isfinite(parsed):
                        raise RuntimeError(
                            f"[TRADE_JOURNAL_MARKET_EVIDENCE_INVALID] {name}={value!r}"
                        )
                    parsed_market_values[name] = parsed
                if (
                    parsed_market_values["entry_price"] <= 0.0
                    or parsed_market_values["entry_bid"] <= 0.0
                    or parsed_market_values["entry_ask"] < parsed_market_values["entry_bid"]
                    or parsed_market_values["entry_spread_bps"] < 0.0
                    or parsed_market_values["atr_bps"] <= 0.0
                ):
                    raise RuntimeError(
                        "[TRADE_JOURNAL_MARKET_EVIDENCE_RANGE] "
                        + json.dumps(parsed_market_values, sort_keys=True)
                    )
                from gx1.execution.v12_trade_state import (
                    require_trade_model_bundle_binding,
                    require_trade_source_pair_binding,
                )

                model_bundle_binding = (
                    require_trade_model_bundle_binding(
                        model_bundle_binding,
                        executable=True,
                    )
                )
                entry_source_pair_binding = (
                    require_trade_source_pair_binding(
                        entry_source_pair_binding,
                        executable=True,
                    )
                )
            elif not self._is_execution_smoke_context():
                raise RuntimeError(
                    "[TRADE_JOURNAL_MODEL_EVIDENCE_MISSING] empty evidence is allowed "
                    "only for an explicit EXEC_SMOKE test context"
                )

            entry_snapshot: Dict[str, Any] = {
                "trade_uid": trade_uid,
                "trade_id": trade_id,
                "entry_time": entry_time,
                "instrument": instrument,
                "side": side,
                "entry_price": entry_price,
                "entry_bid": entry_bid,
                "entry_ask": entry_ask,
                "entry_spread_bps": entry_spread_bps,
                "session": session,
                "model_policy": model_policy,
                "model_evidence": evidence,
                "execution_checks": list(execution_checks or ()),
                "capacity_units": (
                    int(capacity_units) if capacity_units is not None else None
                ),
                "reference_pre_round_units": (
                    float(reference_pre_round_units)
                    if reference_pre_round_units is not None
                    else None
                ),
                "pre_round_units": (
                    float(pre_round_units) if pre_round_units is not None else None
                ),
                "units": int(units) if units is not None else None,
                "applied_size_multiplier": (
                    float(applied_size_multiplier)
                    if applied_size_multiplier is not None
                    else None
                ),
                "sizing_application": dict(sizing_application or {}),
                "atr_bps": float(atr_bps) if atr_bps is not None else None,
                "model_bundle_binding": dict(
                    model_bundle_binding or {}
                ),
                "entry_source_pair_binding": dict(
                    entry_source_pair_binding or {}
                ),
            }
            trade_journal["entry_snapshot"] = entry_snapshot
            self._write_trade_json(trade_uid=trade_uid, trade_id=trade_id)
        except Exception as e:
            key_str = self._key(trade_uid=trade_uid, trade_id=trade_id) if (trade_uid or trade_id) else "UNKNOWN"
            logger.exception(f"[TRADE_JOURNAL] Failed to log entry snapshot for key={key_str}")
            raise RuntimeError(
                f"[TRADE_JOURNAL_ENTRY_EVIDENCE_FAILED] trade_key={key_str}"
            ) from e
    
    def log_exit_summary(
        self,
        exit_time: str,
        exit_price: float,
        exit_reason: str,
        realized_pnl_bps: float,
        exit_bid: Optional[float] = None,
        exit_ask: Optional[float] = None,
        exit_spread_bps: Optional[float] = None,
        exit_price_used: Optional[float] = None,
        trade_uid: Optional[str] = None,
        trade_id: Optional[str] = None,
        max_mfe_bps: Optional[float] = None,
        max_mae_bps: Optional[float] = None,
        intratrade_drawdown_bps: Optional[float] = None,
    ) -> None:
        """
        Log exit summary (final trade closure).
        
        Args:
            exit_time: Exit timestamp (ISO8601 UTC)
            exit_price: Exit price
            exit_reason: Exit reason (TP, SL, BE, TIMEOUT, TRAIL, etc.)
            realized_pnl_bps: Realized PnL in basis points
            trade_uid: Globally unique trade identifier (preferred)
            trade_id: Broker/display trade identifier
            max_mfe_bps: Maximum favorable excursion (bps)
            max_mae_bps: Maximum adverse excursion (bps)
            intratrade_drawdown_bps: Intratrade drawdown (bps)
        """
        if not self.enabled:
            return
        
        try:
            key = self._key(trade_uid=trade_uid, trade_id=trade_id)
            trade_journal = self._get_trade_journal(trade_uid=trade_uid, trade_id=trade_id)
            entry_snapshot = trade_journal.get("entry_snapshot")
            if not isinstance(entry_snapshot, dict):
                raise RuntimeError(
                    f"[TRADE_JOURNAL_EXIT_WITHOUT_ENTRY_EVIDENCE] trade_key={key}"
                )
            
            trade_journal["exit_summary"] = {
                "exit_time": exit_time,
                "exit_price": exit_price,
                "exit_bid": exit_bid,
                "exit_ask": exit_ask,
                "exit_spread_bps": exit_spread_bps,
                "exit_price_used": exit_price_used,
                "exit_reason": exit_reason,
                "realized_pnl_bps": realized_pnl_bps,
                "max_mfe_bps": max_mfe_bps,
                "max_mae_bps": max_mae_bps,
                "intratrade_drawdown_bps": intratrade_drawdown_bps,
            }
            self._write_trade_json(trade_uid=trade_uid, trade_id=trade_id)
            
            # Update index CSV
            self._update_index(trade_uid=trade_uid, trade_id=trade_id)
        except Exception as e:
            key_str = self._key(trade_uid=trade_uid, trade_id=trade_id) if (trade_uid or trade_id) else "UNKNOWN"
            logger.exception("[TRADE_JOURNAL] Failed to log exit summary for key=%s", key_str)
            raise RuntimeError(
                f"[TRADE_JOURNAL_EXIT_SUMMARY_FAILED] trade_key={key_str}"
            ) from e
    
    def _update_index(self, trade_uid: Optional[str] = None, trade_id: Optional[str] = None) -> None:
        """Append one row using the versioned model-native evidence schema."""
        if not self.enabled:
            return

        key = self._key(trade_uid=trade_uid, trade_id=trade_id)
        try:
            trade_journal = self._get_trade_journal(trade_uid=trade_uid, trade_id=trade_id)
            entry_snapshot = trade_journal.get("entry_snapshot")
            if not isinstance(entry_snapshot, dict):
                raise RuntimeError(
                    f"[TRADE_JOURNAL_INDEX_ENTRY_MISSING] trade_key={key}"
                )
            evidence = entry_snapshot.get("model_evidence")
            if not isinstance(evidence, dict) or not evidence:
                raise RuntimeError(
                    f"[TRADE_JOURNAL_INDEX_MODEL_EVIDENCE_MISSING] trade_key={key}"
                )
            evidence = _require_complete_journal_model_evidence(
                dict(evidence),
                context="TRADE_JOURNAL_INDEX",
            )
            require_model_native_fill_time(
                evidence,
                entry_snapshot.get("entry_time"),
                context="TRADE_JOURNAL_INDEX",
            )

            exit_summary = trade_journal.get("exit_summary") or {}
            oanda_trade_id: Optional[str] = None
            oanda_last_txn_id: Optional[str] = None
            execution_status = "UNKNOWN"
            for event in trade_journal.get("execution_events") or ():
                event_type = event.get("event_type")
                if event_type == "ORDER_FILLED":
                    oanda_trade_id = event.get("oanda_trade_id") or oanda_trade_id
                    oanda_last_txn_id = (
                        event.get("oanda_transaction_id") or oanda_last_txn_id
                    )
                    execution_status = "OK"
                elif event_type == "ORDER_REJECTED":
                    execution_status = "REJECTED"
                elif event_type == "TRADE_CLOSED_OANDA":
                    oanda_last_txn_id = (
                        event.get("oanda_transaction_id") or oanda_last_txn_id
                    )

            row: Dict[str, Any] = {
                "trade_key": key,
                "trade_uid": trade_uid,
                "trade_id": trade_id,
                "run_tag": trade_journal.get("run_tag"),
                "entry_time": entry_snapshot.get("entry_time"),
                "exit_time": exit_summary.get("exit_time"),
                "side": entry_snapshot.get("side"),
                "instrument": entry_snapshot.get("instrument"),
                "model_policy": entry_snapshot.get("model_policy"),
                "session": entry_snapshot.get("session"),
                "entry_price": entry_snapshot.get("entry_price"),
                "entry_bid": entry_snapshot.get("entry_bid"),
                "entry_ask": entry_snapshot.get("entry_ask"),
                "entry_spread_bps": entry_snapshot.get("entry_spread_bps"),
                "atr_bps": entry_snapshot.get("atr_bps"),
                "capacity_units": entry_snapshot.get("capacity_units"),
                "reference_pre_round_units": entry_snapshot.get(
                    "reference_pre_round_units"
                ),
                "pre_round_units": entry_snapshot.get("pre_round_units"),
                "units": entry_snapshot.get("units"),
                "applied_size_multiplier": entry_snapshot.get(
                    "applied_size_multiplier"
                ),
                "execution_checks": entry_snapshot.get("execution_checks"),
                "pnl_bps": exit_summary.get("realized_pnl_bps"),
                "exit_reason": exit_summary.get("exit_reason"),
                "oanda_trade_id": oanda_trade_id,
                "oanda_last_txn_id": oanda_last_txn_id,
                "execution_status": execution_status,
            }
            sizing_application = entry_snapshot.get("sizing_application")
            if isinstance(sizing_application, dict):
                row.update(
                    {
                        "sizing_mode": sizing_application.get("sizing_mode"),
                        "calibrated_size_fraction": sizing_application.get(
                            "calibrated_size_fraction"
                        ),
                        "capacity_units": sizing_application.get("capacity_units"),
                        "reference_pre_round_units": sizing_application.get(
                            "reference_pre_round_units"
                        ),
                        "pre_round_units": sizing_application.get("pre_round_units"),
                        "sizing_calibration_artifact_sha256": sizing_application.get(
                            "calibration_artifact_sha256"
                        ),
                        "sizing_oos_proof_artifact_sha256": sizing_application.get(
                            "oos_proof_artifact_sha256"
                        ),
                        "sizing_adoption_artifact_sha256": sizing_application.get(
                            "adoption_artifact_sha256"
                        ),
                    }
                )
            for field in MODEL_NATIVE_INDEX_FIELDS:
                if field not in row and field in evidence:
                    row[field] = evidence[field]
            serialized = {
                field: _csv_value(row.get(field))
                for field in MODEL_NATIVE_INDEX_FIELDS
            }

            write_header = (
                not self.index_path.exists() or self.index_path.stat().st_size == 0
            )
            if not write_header:
                with open(self.index_path, newline="", encoding="utf-8") as f:
                    observed_header = tuple(next(csv.reader(f), ()))
                if observed_header != MODEL_NATIVE_INDEX_FIELDS:
                    raise RuntimeError(
                        "[TRADE_JOURNAL_INDEX_SCHEMA_MISMATCH] "
                        f"observed={observed_header} expected={MODEL_NATIVE_INDEX_FIELDS}"
                    )
            with open(self.index_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=MODEL_NATIVE_INDEX_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow(serialized)
        except Exception as exc:
            logger.exception(
                "[TRADE_JOURNAL] Failed to update model-native index for key=%s",
                key,
            )
            raise RuntimeError(
                f"[TRADE_JOURNAL_INDEX_FAILED] trade_key={key}"
            ) from exc

    def log(
        self,
        event_type: str,
        payload: Dict[str, Any],
        trade_key: Optional[Dict[str, Any]] = None,
        trade_id: Optional[str] = None,
    ) -> None:
        """Log a sanitized run/execution event to the sole JSONL stream.
        
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
            self._file_handle.flush()
            os.fsync(self._file_handle.fileno())

        except Exception as e:
            logger.exception("[TRADE_JOURNAL] Failed to log event %s", event_type)
            raise RuntimeError(
                f"[TRADE_JOURNAL_EVENT_WRITE_FAILED] event_type={event_type}"
            ) from e
    
    def log_order_submitted(
        self,
        trade_id: str,
        instrument: str,
        side: str,
        units: int,
        order_type: str,
        trade_uid: Optional[str] = None,
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
            trade_journal = self._get_trade_journal(trade_uid=trade_uid, trade_id=trade_id)
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
            self._write_trade_json(trade_uid=trade_uid, trade_id=trade_id)
        except Exception as e:
            key_str = self._key(trade_uid=trade_uid, trade_id=trade_id) if (trade_uid or trade_id) else "UNKNOWN"
            logger.warning(f"[TRADE_JOURNAL] Failed to log ORDER_SUBMITTED for key={key_str}: {e}")
    
    def log_order_rejected(
        self,
        trade_id: str,
        trade_uid: Optional[str] = None,
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
            trade_journal = self._get_trade_journal(trade_uid=trade_uid, trade_id=trade_id)
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
            self._write_trade_json(trade_uid=trade_uid, trade_id=trade_id)
        except Exception as e:
            key_str = self._key(trade_uid=trade_uid, trade_id=trade_id) if (trade_uid or trade_id) else "UNKNOWN"
            logger.warning(f"[TRADE_JOURNAL] Failed to log ORDER_REJECTED for key={key_str}: {e}")
    
    def log_order_filled(
        self,
        trade_id: str,
        trade_uid: Optional[str] = None,
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
            trade_journal = self._get_trade_journal(trade_uid=trade_uid, trade_id=trade_id)
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
            self._write_trade_json(trade_uid=trade_uid, trade_id=trade_id)
        except Exception as e:
            key_str = self._key(trade_uid=trade_uid, trade_id=trade_id) if (trade_uid or trade_id) else "UNKNOWN"
            logger.warning(f"[TRADE_JOURNAL] Failed to log ORDER_FILLED for key={key_str}: {e}")
    
    def log_oanda_trade_update(
        self,
        trade_id: str,
        event_type: str,  # TRADE_OPENED_OANDA or TRADE_CLOSED_OANDA
        trade_uid: Optional[str] = None,
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
            trade_journal = self._get_trade_journal(trade_uid=trade_uid, trade_id=trade_id)
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
            self._write_trade_json(trade_uid=trade_uid, trade_id=trade_id)
        except Exception as e:
            key_str = self._key(trade_uid=trade_uid, trade_id=trade_id) if (trade_uid or trade_id) else "UNKNOWN"
            logger.warning(f"[TRADE_JOURNAL] Failed to log {event_type} for key={key_str}: {e}")
    
    def log_v12_bar_decision(
        self,
        trade_id: str,
        timestamp: str,
        bars_in_trade: int,
        bid: float,
        ask: float,
        current_pnl_bps: float,
        cum_mfe_bps: float,
        cum_mae_bps: float,
        exit_action: str,
        exit_action_index: int,
        exit_action_logits: list[float],
        exit_action_probs: list[float],
        exit_decision_source: str,
        bundle_sha256: str,
        entry_snapshot_sha256: str,
        exit_path_envelope_sha256: str,
        output_evidence_sha256: str,
        exit_path_envelope: Dict[str, Any],
        executable_range_bps: Optional[float] = None,
        bars_since_mfe_peak: Optional[int] = None,
        trade_uid: Optional[str] = None,
    ) -> None:
        """Log the exact unified-model per-bar Exit decision and identities."""
        if not self.enabled:
            return
        try:
            trade_journal = self._get_trade_journal(trade_uid=trade_uid, trade_id=trade_id)
            entry_record = trade_journal.get("entry_snapshot")
            if not isinstance(entry_record, dict):
                raise RuntimeError("unified Exit journal requires Entry snapshot")
            entry_evidence = entry_record.get("model_evidence")
            if not isinstance(entry_evidence, dict) or not entry_evidence:
                raise RuntimeError("unified Exit journal requires model Entry evidence")
            exit_output = {
                "exit_action_logits": list(exit_action_logits),
                "exit_action_probs": list(exit_action_probs),
                "exit_action_index": exit_action_index,
                "action": exit_action,
                "decision_source": exit_decision_source,
                "bundle_sha256": bundle_sha256,
                "entry_snapshot_sha256": entry_snapshot_sha256,
                "exit_path_envelope_sha256": exit_path_envelope_sha256,
                "output_evidence_sha256": output_evidence_sha256,
            }
            validated_path = require_unified_exit_path_envelope(
                exit_path_envelope,
                context="TRADE_JOURNAL_UNIFIED_EXIT",
            )
            if (
                timestamp != validated_path["last_closed_m1_bar_ts"]
                or bars_in_trade != validated_path["bars_in_trade"]
                or bid != validated_path["path_rows"][-1]["bid_close"]
                or ask != validated_path["path_rows"][-1]["ask_close"]
            ):
                raise RuntimeError(
                    "unified Exit journal bar identity differs from "
                    "the exact closed M1 envelope"
                )
            require_unified_exit_output(
                exit_output,
                context="TRADE_JOURNAL_UNIFIED_EXIT",
                expected_bundle_sha256=bundle_sha256,
                entry_snapshot=entry_evidence,
                exit_path_envelope=validated_path,
            )
            bar_record = {
                "schema_version": "gx1_unified_exit_journal_bar_v1",
                "timestamp": timestamp,
                "bars_in_trade": bars_in_trade,
                "bid": bid, "ask": ask,
                "current_pnl_bps": current_pnl_bps,
                "cum_mfe_bps": cum_mfe_bps,
                "cum_mae_bps": cum_mae_bps,
                "bars_since_mfe_peak": bars_since_mfe_peak,
                "executable_range_bps": executable_range_bps,
                "exit_action": exit_action,
                "exit_action_index": exit_action_index,
                "exit_action_logits": list(exit_action_logits),
                "exit_action_probs": list(exit_action_probs),
                "exit_decision_source": exit_decision_source,
                "bundle_sha256": bundle_sha256,
                "entry_snapshot_sha256": entry_snapshot_sha256,
                "exit_path_envelope_sha256": exit_path_envelope_sha256,
                "output_evidence_sha256": output_evidence_sha256,
                "exit_path_envelope": validated_path,
            }
            decisions = trade_journal.setdefault(
                "v12_bar_decisions",
                [],
            )
            if not isinstance(decisions, list):
                raise RuntimeError(
                    "unified Exit journal decision history is invalid"
                )
            exact_replay = [
                existing
                for existing in decisions
                if isinstance(existing, dict)
                and existing.get("output_evidence_sha256")
                == output_evidence_sha256
            ]
            if exact_replay:
                if len(exact_replay) != 1 or exact_replay[0] != bar_record:
                    raise RuntimeError(
                        "unified Exit journal hash replay differs from "
                        "the existing decision"
                    )
                return
            identity_collision = [
                existing
                for existing in decisions
                if isinstance(existing, dict)
                and (
                    existing.get("timestamp") == timestamp
                    or existing.get("bars_in_trade") == bars_in_trade
                )
            ]
            if identity_collision:
                raise RuntimeError(
                    "unified Exit journal bar identity already has "
                    "a different model decision"
                )
            decisions.append(bar_record)
            self._write_trade_json(trade_uid=trade_uid, trade_id=trade_id)
        except Exception as exc:
            key_str = self._key(trade_uid=trade_uid, trade_id=trade_id) if (trade_uid or trade_id) else "UNKNOWN"
            logger.exception(
                "[TRADE_JOURNAL] Failed to log v12 bar decision for key=%s",
                key_str,
            )
            raise RuntimeError(
                f"[TRADE_JOURNAL_V12_BAR_DECISION_FAILED] trade_key={key_str}"
            ) from exc

    def close(self) -> None:
        """Close journal and flush all pending writes."""
        # Write any remaining trade JSONs
        for key, trade_journal in list(self._trade_journals.items()):
            self._write_trade_json(
                trade_uid=trade_journal.get("trade_uid"),
                trade_id=trade_journal.get("trade_id"),
            )
        
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
