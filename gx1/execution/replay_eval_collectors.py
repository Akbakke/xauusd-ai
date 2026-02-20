#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay Evaluation Collectors - Passive data collection for replay analysis

DEL 1: Three passive collectors that accumulate data during replay:
- RawSignalCollector: Pre-policy raw model outputs
- PolicyDecisionCollector: Post-policy decisions with attribution
- TradeOutcomeCollector: Trading outcomes (PnL, MAE, MFE)

All collectors are passive - they only accumulate data, no thresholds or filtering.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class RawSignalCollector:
    """
    Collects pre-policy raw model outputs (never thresholded).
    
    Hook point: Right after model call returns EntryPrediction.
    Invariant: One row per model call.
    """
    signals: List[Dict[str, Any]] = field(default_factory=list)
    # Provenance (set once at replay start, copied to all rows)
    _provenance: Dict[str, Any] = field(default_factory=dict)
    
    def set_provenance(
        self,
        policy_module: Optional[str] = None,
        policy_class: Optional[str] = None,
        policy_config_path: Optional[str] = None,
        entry_config_path: Optional[str] = None,
        entry_model_id: Optional[str] = None,
        bundle_path_abs: Optional[str] = None,
        bundle_sha256: Optional[str] = None,
        git_commit: Optional[str] = None,
        run_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
        worker_pid: Optional[int] = None,
        runtime_feature_impl: Optional[str] = None,  # DEL 3: Runtime feature implementation
        runtime_feature_module: Optional[str] = None,  # DEL 3: Runtime feature module path
    ):
        """Set provenance information (called once at replay start)."""
        self._provenance = {
            "policy_module": policy_module,
            "policy_class": policy_class,
            "policy_config_path": policy_config_path,
            "entry_config_path": entry_config_path,
            "entry_model_id": entry_model_id,
            "bundle_path_abs": bundle_path_abs,
            "bundle_sha256": bundle_sha256,
            "git_commit": git_commit,
            "run_id": run_id,
            "chunk_id": chunk_id,
            "worker_pid": worker_pid,
            "runtime_feature_impl": runtime_feature_impl,  # DEL 3: Runtime feature implementation
            "runtime_feature_module": runtime_feature_module,  # DEL 3: Runtime feature module path
        }
    
    def collect(
        self,
        timestamp: pd.Timestamp,
        session: str,
        raw_logit: float,
        p_calibrated: float,
        gate_value: float,
        uncertainty: Optional[float] = None,
        margin: Optional[float] = None,
    ):
        """
        Collect raw signal (pre-policy, never thresholded).
        
        Args:
            timestamp: Bar timestamp
            session: Trading session (EU/US/OVERLAP/ASIA)
            raw_logit: Raw logit from model (before calibration)
            p_calibrated: Calibrated probability (p_cal)
            gate_value: Gate value from gated fusion
            uncertainty: Uncertainty score (if available)
            margin: Margin (if available)
        """
        self.signals.append({
            "ts": timestamp.isoformat() if isinstance(timestamp, pd.Timestamp) else str(timestamp),
            "session": session,
            "raw_logit": float(raw_logit),
            "p_calibrated": float(p_calibrated),
            "gate_value": float(gate_value),
            "uncertainty": float(uncertainty) if uncertainty is not None else None,
            "margin": float(margin) if margin is not None else None,
        })
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected signals to DataFrame, adding provenance columns."""
        if not self.signals:
            return pd.DataFrame()
        df = pd.DataFrame(self.signals)
        # Add provenance columns to all rows
        for key, value in self._provenance.items():
            df[key] = value
        return df


@dataclass
class PolicyDecisionCollector:
    """
    Collects post-policy decisions with attribution.
    
    Hook point: After policy decision (enter/skip).
    """
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    attribution_counts: Dict[str, int] = field(default_factory=dict)
    # Provenance (set once at replay start, copied to all rows)
    _provenance: Dict[str, Any] = field(default_factory=dict)
    
    def set_provenance(
        self,
        policy_module: Optional[str] = None,
        policy_class: Optional[str] = None,
        policy_config_path: Optional[str] = None,
        entry_config_path: Optional[str] = None,
        entry_model_id: Optional[str] = None,
        bundle_path_abs: Optional[str] = None,
        bundle_sha256: Optional[str] = None,
        git_commit: Optional[str] = None,
        run_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
        worker_pid: Optional[int] = None,
        runtime_feature_impl: Optional[str] = None,  # DEL 3: Runtime feature implementation
        runtime_feature_module: Optional[str] = None,  # DEL 3: Runtime feature module path
    ):
        """Set provenance information (called once at replay start)."""
        self._provenance = {
            "policy_module": policy_module,
            "policy_class": policy_class,
            "policy_config_path": policy_config_path,
            "entry_config_path": entry_config_path,
            "entry_model_id": entry_model_id,
            "bundle_path_abs": bundle_path_abs,
            "bundle_sha256": bundle_sha256,
            "git_commit": git_commit,
            "run_id": run_id,
            "chunk_id": chunk_id,
            "worker_pid": worker_pid,
            "runtime_feature_impl": runtime_feature_impl,  # DEL 3: Runtime feature implementation
            "runtime_feature_module": runtime_feature_module,  # DEL 3: Runtime feature module path
        }
    
    def collect(
        self,
        timestamp: pd.Timestamp,
        decision: str,  # "enter" or "skip"
        reasons: List[str],  # ["threshold", "spread", "safety", etc.]
        chosen_size: Optional[float] = None,
        exit_policy_id: Optional[str] = None,
    ):
        """
        Collect policy decision (post-policy).
        
        Args:
            timestamp: Bar timestamp
            decision: "enter" or "skip"
            reasons: List of reason codes (threshold, spread, safety, regime, etc.)
            chosen_size: Trade size if entering
            exit_policy_id: Exit policy identifier if entering
        """
        self.decisions.append({
            "ts": timestamp.isoformat() if isinstance(timestamp, pd.Timestamp) else str(timestamp),
            "decision": decision,
            "reasons": reasons,
            "chosen_size": float(chosen_size) if chosen_size is not None else None,
            "exit_policy_id": exit_policy_id,
        })
        
        # Update attribution counts
        for reason in reasons:
            key = f"{decision}:{reason}"
            self.attribution_counts[key] = self.attribution_counts.get(key, 0) + 1
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected decisions to DataFrame, adding provenance columns."""
        if not self.decisions:
            return pd.DataFrame()
        # Flatten reasons list to string for parquet compatibility
        df_data = []
        for d in self.decisions:
            row = d.copy()
            row["reasons"] = ",".join(row["reasons"])  # Convert list to comma-separated string
            df_data.append(row)
        df = pd.DataFrame(df_data)
        # Add provenance columns to all rows
        for key, value in self._provenance.items():
            df[key] = value
        return df
    
    def get_attribution(self) -> Dict[str, Any]:
        """Get attribution summary (counts and percentages)."""
        total = len(self.decisions)
        if total == 0:
            return {
                "total_decisions": 0,
                "by_decision": {},
                "by_reason": {},
            }
        
        attribution = {
            "total_decisions": total,
            "by_decision": {},
            "by_reason": {},
        }
        
        # Aggregate by decision and reason
        for key, count in self.attribution_counts.items():
            decision, reason = key.split(":", 1)
            attribution["by_decision"][decision] = attribution["by_decision"].get(decision, 0) + count
            attribution["by_reason"][reason] = attribution["by_reason"].get(reason, 0) + count
        
        # Add percentages
        # Create a copy to avoid "dictionary changed size during iteration"
        for decision in list(attribution["by_decision"].keys()):
            count = attribution["by_decision"][decision]
            attribution["by_decision"][f"{decision}_pct"] = (count / total * 100) if total > 0 else 0.0
        
        # Create a copy to avoid "dictionary changed size during iteration"
        for reason in list(attribution["by_reason"].keys()):
            count = attribution["by_reason"][reason]
            attribution["by_reason"][f"{reason}_pct"] = (count / total * 100) if total > 0 else 0.0
        
        return attribution


@dataclass
class TradeOutcomeCollector:
    """
    Collects trading outcomes when trades close.
    
    Hook point: When trade closes (request_close or _update_trade_log_on_close).
    
    SSoT cache: Maintains trade_uid -> entry_time mapping for reliable access
    even if trade object is removed from open_trades before close.
    
    Per-run state: _collected_trade_ids is reset for each replay/run (not singleton/global).
    """
    outcomes: List[Dict[str, Any]] = field(default_factory=list)
    # SSoT cache: trade_uid -> entry_time (set at trade creation, used at close)
    _entry_time_cache: Dict[str, pd.Timestamp] = field(default_factory=dict)
    # One-shot guard: prevent duplicate collection (trade_id -> True if already collected)
    # Per-run: reset for each replay/run, does not leak between chunks
    _collected_trade_ids: set[str] = field(default_factory=set)
    
    def reset_for_new_run(self) -> None:
        """
        Reset per-run state for a new replay/run.
        
        This ensures _collected_trade_ids does not leak between runs or chunks.
        Should be called at the start of each replay/run.
        """
        self._collected_trade_ids.clear()
        # Note: _entry_time_cache is kept (persists across chunks in same run)
        # outcomes list is kept (accumulates across chunks in same run)
    # Statistics: track cache usage for diagnostics
    _stats: Dict[str, int] = field(default_factory=lambda: {
        "n_cache_set": 0,  # Number of trades registered in cache
        "n_cache_hit": 0,  # Number of times cache was used successfully
        "n_cache_miss": 0,  # Number of times cache lookup failed (must be 0 in TRUTH)
        "n_entry_time_from_open_trades": 0,  # Number of times entry_time came from open_trades
        "n_close_open_trades_missing": 0,  # Number of times trade was missing from open_trades at close
        # UID drift statistics
        "n_registered_uid": 0,  # Number of times registered with trade_uid
        "n_registered_id": 0,  # Number of times registered with trade_id
        "n_close_lookup_by_uid": 0,  # Number of times looked up by trade_uid at close
        "n_close_lookup_by_id": 0,  # Number of times looked up by trade_id at close
        "n_uid_id_disagree": 0,  # Number of times both exist but point to different entry_time (must be 0 in TRUTH)
        "n_duplicate_collection_attempts": 0,  # Number of times collect() was called for already-collected trade_id
    })
    # Sample disagree cases (max 3)
    _sample_disagree: List[Dict[str, Any]] = field(default_factory=list)
    
    def register_trade_entry(self, trade_uid: str, entry_time: pd.Timestamp, is_trade_id: bool = False) -> None:
        """
        Register trade entry in SSoT cache (called at trade creation).
        
        Args:
            trade_uid: Globally unique trade identifier (can be trade_uid or trade_id)
            entry_time: Trade entry timestamp
            is_trade_id: If True, this is trade_id (not trade_uid)
        """
        import os
        debug_enabled = os.getenv("GX1_ENTRYTIME_CACHE_DEBUG", "0") == "1"
        debug_max = int(os.getenv("GX1_ENTRYTIME_CACHE_DEBUG_MAX", "0") or "0")
        
        if trade_uid:
            # Check for disagree: if key already exists with different entry_time
            existing_entry_time = self._entry_time_cache.get(trade_uid)
            if existing_entry_time is not None and existing_entry_time != entry_time:
                # Disagree detected
                self._stats["n_uid_id_disagree"] += 1
                if len(self._sample_disagree) < 3:
                    self._sample_disagree.append({
                        "key": trade_uid,
                        "existing_entry_time": existing_entry_time.isoformat() if existing_entry_time else None,
                        "new_entry_time": entry_time.isoformat() if entry_time else None,
                        "is_trade_id": is_trade_id,
                    })
            
            self._entry_time_cache[trade_uid] = entry_time
            self._stats["n_cache_set"] += 1
            
            # Track registration type
            if is_trade_id:
                self._stats["n_registered_id"] += 1
            else:
                self._stats["n_registered_uid"] += 1
            
            if debug_enabled and self._stats["n_cache_set"] <= debug_max:
                import logging
                log = logging.getLogger(__name__)
                log.info(
                    f"[ENTRYTIME_CACHE_DEBUG] SET {'trade_id' if is_trade_id else 'trade_uid'}={trade_uid} entry_time={entry_time.isoformat() if entry_time else None} "
                    f"(n_cache_set={self._stats['n_cache_set']}, n_registered_{'id' if is_trade_id else 'uid'}={self._stats['n_registered_id' if is_trade_id else 'n_registered_uid']})"
                )
    
    def get_entry_time(self, trade_uid: str) -> Optional[pd.Timestamp]:
        """
        Get entry_time from SSoT cache.
        
        Args:
            trade_uid: Globally unique trade identifier
            
        Returns:
            Entry timestamp if found in cache, None otherwise
        """
        return self._entry_time_cache.get(trade_uid)
    
    def collect(
        self,
        trade_id: str,
        pnl_bps: float,
        mae_bps: Optional[float] = None,
        mfe_bps: Optional[float] = None,
        time_to_mae_bars: Optional[int] = None,
        time_to_mfe_bars: Optional[int] = None,
        close_to_mfe_bps: Optional[float] = None,
        exit_efficiency: Optional[float] = None,
        post_exit_mfe_12b_bps: Optional[float] = None,
        post_exit_mae_12b_bps: Optional[float] = None,
        duration_bars: Optional[int] = None,
        session: Optional[str] = None,
        exit_reason: Optional[str] = None,
        trade_uid: Optional[str] = None,
        entry_time: Optional[pd.Timestamp] = None,
        exit_time: Optional[pd.Timestamp] = None,
    ):
        """
        Collect trade outcome - ONLY for closed trades.
        
        CONTRACT: This method should ONLY be called when a trade is fully closed.
        - entry_time MUST be set (from trade object or SSoT cache)
        - exit_time MUST be set (from close event)
        - pnl_bps MUST be set (realized PnL)
        
        In TRUTH mode, all of these are REQUIRED and will hard-fail if missing.
        
        Args:
            trade_id: Trade identifier
            pnl_bps: Realized PnL in basis points (REQUIRED)
            mae_bps: Maximum Adverse Excursion (if available)
            mfe_bps: Maximum Favorable Excursion (if available)
            duration_bars: Bars held (if available)
            session: Entry session (if available)
            exit_reason: Exit reason (if available)
            trade_uid: Globally unique trade identifier (SSoT for merge)
            entry_time: Trade entry timestamp (SSoT for merge) - REQUIRED in TRUTH mode
            exit_time: Trade exit timestamp (SSoT for merge) - REQUIRED in TRUTH mode
        """
        # SSoT fallback: If entry_time not provided, try to get from cache
        import os
        debug_enabled = os.getenv("GX1_ENTRYTIME_CACHE_DEBUG", "0") == "1"
        debug_max = int(os.getenv("GX1_ENTRYTIME_CACHE_DEBUG_MAX", "0") or "0")
        
        # TRUTH: One-shot guard - prevent duplicate collection
        import os
        is_truth = os.getenv("GX1_RUN_MODE", "").upper() == "TRUTH" or os.getenv("GX1_TRUTH_MODE", "0") == "1"
        
        # Check if this trade_id has already been collected
        if trade_id in self._collected_trade_ids:
            self._stats["n_duplicate_collection_attempts"] += 1
            if is_truth:
                # In TRUTH mode, log deterministic debug event
                import logging
                log = logging.getLogger(__name__)
                log.debug(
                    f"[TRADE_OUTCOME_COLLECTOR] Duplicate collection attempt for trade_id={trade_id} (uid={trade_uid}). "
                    f"This is expected if trade was already closed. Skipping duplicate collection. "
                    f"(n_duplicate_attempts={self._stats['n_duplicate_collection_attempts']})"
                )
            # Return early - idempotent behavior
            return
        
        effective_trade_uid = trade_uid if trade_uid is not None else trade_id
        
        # TRUTH: Hard fail if legacy trade_uid format is used
        if is_truth and effective_trade_uid and effective_trade_uid.startswith("SIM-"):
            error_msg = (
                f"[TRADE_OUTCOME_COLLECTOR] Legacy trade_uid format detected: {effective_trade_uid}. "
                f"This violates TRUTH invariant. trade_uid must use canonical format (run_id:chunk_id:sequence:hash), "
                f"not legacy SIM-* format. trade_id={trade_id}"
            )
            raise RuntimeError(error_msg)
        
        cached_entry_time = None
        used_cache = False
        if entry_time is None and effective_trade_uid:
            # Cache lookup needed
            cached_entry_time = self.get_entry_time(effective_trade_uid)
            if cached_entry_time is not None:
                entry_time = cached_entry_time
                used_cache = True
                self._stats["n_cache_hit"] += 1
                
                if debug_enabled and self._stats["n_cache_hit"] <= debug_max:
                    import logging
                    log = logging.getLogger(__name__)
                    log.info(
                        f"[ENTRYTIME_CACHE_DEBUG] HIT trade_uid={effective_trade_uid} entry_time={entry_time.isoformat() if entry_time else None} "
                        f"(n_cache_hit={self._stats['n_cache_hit']})"
                    )
            else:
                # Cache miss
                self._stats["n_cache_miss"] += 1
                
                if debug_enabled and self._stats["n_cache_miss"] <= debug_max:
                    import logging
                    log = logging.getLogger(__name__)
                    log.warning(
                        f"[ENTRYTIME_CACHE_DEBUG] MISS trade_uid={effective_trade_uid} trade_id={trade_id} "
                        f"(n_cache_miss={self._stats['n_cache_miss']})"
                    )
        elif entry_time is not None:
            # entry_time was provided directly (from open_trades)
            self._stats["n_entry_time_from_open_trades"] += 1
        
        # TRUTH-only: Hard fail if entry_time, exit_time, or pnl_bps is None (violates TRUTH invariant)
        import os
        import inspect
        import traceback
        from pathlib import Path
        from datetime import datetime, timezone
        
        is_truth = os.getenv("GX1_RUN_MODE", "").upper() == "TRUTH" or os.getenv("GX1_TRUTH_MODE", "0") == "1"
        if is_truth:
            # Hard fail if pnl_bps is None or NaN
            if pnl_bps is None or (isinstance(pnl_bps, float) and (pnl_bps != pnl_bps)):  # NaN check
                error_msg = (
                    f"[TRADE_OUTCOME_COLLECTOR] pnl_bps is None or NaN for trade {trade_id} (uid={effective_trade_uid}). "
                    f"This violates TRUTH invariant. All closed trades must have realized PnL. "
                    f"entry_time={entry_time.isoformat() if entry_time else None}, "
                    f"exit_time={exit_time.isoformat() if exit_time else None}"
                )
                # Write fatal capsule
                try:
                    output_dir = getattr(self, "_output_dir", None)
                    if output_dir is None:
                        output_dir_str = os.getenv("GX1_OUTPUT_DIR")
                        if output_dir_str:
                            output_dir = Path(output_dir_str)
                    
                    if output_dir:
                        capsule_path = output_dir / f"TRADE_OUTCOME_SSOT_FAIL_PNL_NULL_{effective_trade_uid or trade_id}.json"
                        stack = traceback.format_stack(limit=15)
                        callsite = "\n".join(stack[-5:]) if len(stack) >= 5 else "\n".join(stack)
                        capsule = {
                            "error_type": "TRADE_OUTCOME_PNL_NULL",
                            "trade_id": trade_id,
                            "trade_uid": effective_trade_uid,
                            "entry_time": entry_time.isoformat() if entry_time else None,
                            "exit_time": exit_time.isoformat() if exit_time else None,
                            "pnl_bps": pnl_bps,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "message": error_msg,
                            "callsite": callsite,
                        }
                        with open(capsule_path, "w") as f:
                            import json
                            json.dump(capsule, f, indent=2)
                except Exception as capsule_error:
                    log.warning(f"[TRADE_OUTCOME_COLLECTOR] Failed to write fatal capsule: {capsule_error}")
                
                raise RuntimeError(error_msg)
            
            if entry_time is None:
                cache_status = "found" if cached_entry_time is not None else "not found"
                error_msg = (
                    f"[TRADE_OUTCOME_COLLECTOR] entry_time is None for trade {trade_id} (uid={effective_trade_uid}). "
                    f"This violates TRUTH invariant. All trades must have entry_time. "
                    f"Cache lookup: {cache_status}"
                )
                # Write fatal capsule
                try:
                    output_dir = getattr(self, "_output_dir", None)
                    if output_dir is None:
                        output_dir_str = os.getenv("GX1_OUTPUT_DIR")
                        if output_dir_str:
                            output_dir = Path(output_dir_str)
                    
                    if output_dir:
                        capsule_path = output_dir / f"TRADE_OUTCOME_SSOT_FAIL_ENTRY_NULL_{effective_trade_uid or trade_id}.json"
                        stack = traceback.format_stack(limit=15)
                        callsite = "\n".join(stack[-5:]) if len(stack) >= 5 else "\n".join(stack)
                        capsule = {
                            "error_type": "TRADE_OUTCOME_ENTRY_NULL",
                            "trade_id": trade_id,
                            "trade_uid": effective_trade_uid,
                            "entry_time": None,
                            "exit_time": exit_time.isoformat() if exit_time else None,
                            "pnl_bps": pnl_bps,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "message": error_msg,
                            "callsite": callsite,
                        }
                        with open(capsule_path, "w") as f:
                            import json
                            json.dump(capsule, f, indent=2)
                except Exception as capsule_error:
                    log.warning(f"[TRADE_OUTCOME_COLLECTOR] Failed to write fatal capsule: {capsule_error}")
                
                raise RuntimeError(error_msg)
            
            if exit_time is None:
                # Hard fail in TRUTH mode - exit_time must be provided
                # Get callsite info for debugging
                stack = traceback.format_stack(limit=15)
                callsite = "\n".join(stack[-5:]) if len(stack) >= 5 else "\n".join(stack)
                
                # Try to get run_id, chunk_id, bar_i, ts from environment or collector
                run_id = os.getenv("GX1_RUN_ID", "unknown")
                chunk_id = os.getenv("GX1_CHUNK_ID", "unknown")
                bar_i = os.getenv("GX1_CURRENT_BAR_I", "unknown")
                current_ts = os.getenv("GX1_CURRENT_TS", "unknown")
                
                error_msg = (
                    f"[TRADE_OUTCOME_COLLECTOR] exit_time is None for trade {trade_id} (uid={effective_trade_uid}). "
                    f"This violates TRUTH invariant. All trades must have exit_time when closing. "
                    f"entry_time={entry_time.isoformat() if entry_time else None}, "
                    f"run_id={run_id}, chunk_id={chunk_id}, bar_i={bar_i}, current_ts={current_ts}"
                )
                
                # Write fatal capsule
                try:
                    # Try to get output_dir from collector or environment
                    output_dir = getattr(self, "_output_dir", None)
                    if output_dir is None:
                        output_dir_str = os.getenv("GX1_OUTPUT_DIR")
                        if output_dir_str:
                            output_dir = Path(output_dir_str)
                    
                    if output_dir:
                        capsule_path = output_dir / f"TRADE_OUTCOME_EXITTIME_NULL_{effective_trade_uid or trade_id}.json"
                        capsule = {
                            "error_type": "TRADE_OUTCOME_EXITTIME_NULL",
                            "trade_id": trade_id,
                            "trade_uid": effective_trade_uid,
                            "entry_time": entry_time.isoformat() if entry_time else None,
                            "exit_time": None,
                            "run_id": run_id,
                            "chunk_id": chunk_id,
                            "bar_i": bar_i,
                            "current_ts": current_ts,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "message": error_msg,
                            "callsite": callsite,
                        }
                        with open(capsule_path, "w") as f:
                            import json
                            json.dump(capsule, f, indent=2)
                except Exception as capsule_error:
                    log.warning(f"[TRADE_OUTCOME_COLLECTOR] Failed to write fatal capsule: {capsule_error}")
                
                raise RuntimeError(error_msg)
        
        # Normalize session field: never use string "None", use actual None or proper value
        normalized_session = None
        if session is not None:
            session_str = str(session).strip()
            if session_str and session_str.lower() != "none":
                normalized_session = session_str
            # If session is "None" string, convert to None (null)
            # This ensures consistent representation
        
        # Mark trade_id as collected (one-shot guard)
        self._collected_trade_ids.add(trade_id)
        
        # Use 0/0.0/1.0 for missing numerics to avoid NaN in parquet (TRUTH hash requires no NaNs)
        self.outcomes.append({
            "trade_id": trade_id,
            "trade_uid": effective_trade_uid,
            "pnl_bps": float(pnl_bps),
            "mae_bps": float(mae_bps) if mae_bps is not None else 0.0,
            "mfe_bps": float(mfe_bps) if mfe_bps is not None else 0.0,
            "time_to_mae_bars": int(time_to_mae_bars) if time_to_mae_bars is not None else 0,
            "time_to_mfe_bars": int(time_to_mfe_bars) if time_to_mfe_bars is not None else 0,
            "close_to_mfe_bps": float(close_to_mfe_bps) if close_to_mfe_bps is not None else 0.0,
            "exit_efficiency": float(exit_efficiency) if exit_efficiency is not None else 1.0,
            "post_exit_mfe_12b_bps": float(post_exit_mfe_12b_bps) if post_exit_mfe_12b_bps is not None else 0.0,
            "post_exit_mae_12b_bps": float(post_exit_mae_12b_bps) if post_exit_mae_12b_bps is not None else 0.0,
            "duration_bars": int(duration_bars) if duration_bars is not None else 0,
            "session": normalized_session,  # Normalized: never string "None"
            "exit_reason": exit_reason or "",
            "entry_time": entry_time.isoformat() if entry_time is not None else None,
            "exit_time": exit_time.isoformat() if exit_time is not None else None,
        })
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected outcomes to DataFrame."""
        if not self.outcomes:
            return pd.DataFrame()
        return pd.DataFrame(self.outcomes)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about cache usage."""
        stats = self._stats.copy()
        stats["sample_disagree"] = self._sample_disagree.copy() if self._sample_disagree else []
        return stats
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute trading outcome metrics."""
        if not self.outcomes:
            return {
                "n_trades": 0,
                "total_pnl_bps": 0.0,
                "mean_pnl_bps": 0.0,
            }
        
        df = pd.DataFrame(self.outcomes)
        
        metrics = {
            "n_trades": len(df),
            "total_pnl_bps": float(df["pnl_bps"].sum()),
            "mean_pnl_bps": float(df["pnl_bps"].mean()),
            "median_pnl_bps": float(df["pnl_bps"].median()),
            "std_pnl_bps": float(df["pnl_bps"].std()),
        }
        
        # MAE/MFE if available
        if "mae_bps" in df.columns and df["mae_bps"].notna().any():
            metrics["mae_bps"] = float(df["mae_bps"].mean())
        if "mfe_bps" in df.columns and df["mfe_bps"].notna().any():
            metrics["mfe_bps"] = float(df["mfe_bps"].mean())
        
        # Tail metrics
        sorted_pnl = df["pnl_bps"].sort_values()
        if len(sorted_pnl) > 0:
            metrics["p1_loss"] = float(sorted_pnl.quantile(0.01))
            metrics["p5_loss"] = float(sorted_pnl.quantile(0.05))
            metrics["max_dd"] = float(sorted_pnl.min())
            
            # Loss streaks
            loss_streaks = []
            current_streak = 0
            for pnl in sorted_pnl:
                if pnl < 0:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        loss_streaks.append(current_streak)
                    current_streak = 0
            if current_streak > 0:
                loss_streaks.append(current_streak)
            
            metrics["max_loss_streak"] = max(loss_streaks) if loss_streaks else 0
            metrics["mean_loss_streak"] = float(pd.Series(loss_streaks).mean()) if loss_streaks else 0.0
        else:
            metrics["p1_loss"] = 0.0
            metrics["p5_loss"] = 0.0
            metrics["max_dd"] = 0.0
            metrics["max_loss_streak"] = 0
            metrics["mean_loss_streak"] = 0.0
        
        return metrics
