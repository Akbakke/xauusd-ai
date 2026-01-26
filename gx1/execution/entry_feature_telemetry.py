#!/usr/bin/env python3
"""
Entry Feature Flow Telemetry

Collects instrumentation data for entry feature flow:
- Gate execution (executed, blocked, passed)
- Transformer input (feature names, shapes, xgb channels)
- XGB flow (pre/post transformer, veto)
- Feature masking proof (applied, sample check)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

log = logging.getLogger(__name__)


@dataclass
class GateTelemetry:
    """Telemetry for a single gate execution."""
    gate_name: str
    executed: bool = False
    blocked: bool = False
    passed: bool = False
    reason: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class TransformerInputTelemetry:
    """Telemetry for transformer input."""
    seq_shape: Optional[tuple] = None
    snap_shape: Optional[tuple] = None
    seq_feature_names: List[str] = field(default_factory=list)
    snap_feature_names: List[str] = field(default_factory=list)
    xgb_seq_channels: List[str] = field(default_factory=list)
    xgb_snap_channels: List[str] = field(default_factory=list)
    xgb_seq_values: Optional[Dict[str, float]] = None
    xgb_snap_values: Optional[Dict[str, float]] = None
    input_aliases_applied: Dict[str, str] = field(default_factory=dict)  # DEL 2: CLOSE -> candles.close
    timestamp: Optional[str] = None


@dataclass
class XGBFlowTelemetry:
    """Telemetry for XGB flow."""
    xgb_called: bool = False
    xgb_session: Optional[str] = None
    xgb_p_long_raw: Optional[float] = None
    xgb_p_long_cal: Optional[float] = None
    xgb_margin: Optional[float] = None
    xgb_p_hat: Optional[float] = None
    xgb_uncertainty_score: Optional[float] = None
    xgb_veto: bool = False  # DEPRECATED: XGB POST removed 2026-01-24
    xgb_veto_reason: Optional[str] = None  # DEPRECATED: XGB POST removed 2026-01-24
    xgb_used_as: Optional[str] = None  # Now always "pre" (XGB POST removed 2026-01-24)
    post_predict_called: bool = False  # DEPRECATED: Always False (XGB POST removed 2026-01-24)
    timestamp: Optional[str] = None


@dataclass
class ToggleStateTelemetry:
    """Telemetry for toggle states."""
    disable_xgb_channels_in_transformer_requested: bool = False
    disable_xgb_channels_in_transformer_effective: bool = False
    disable_xgb_post_transformer_requested: bool = False
    disable_xgb_post_transformer_effective: bool = False
    n_xgb_channels_in_transformer_input: int = 0
    timestamp: Optional[str] = None


@dataclass
class FeatureMaskTelemetry:
    """Telemetry for feature masking."""
    mask_enabled: bool = False
    mask_families: List[str] = field(default_factory=list)
    mask_strategy: Optional[str] = None
    masked_features: List[str] = field(default_factory=list)
    sample_masked_values: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[str] = None


class EntryFeatureTelemetryCollector:
    """Collects telemetry for entry feature flow."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir
        self.gates: List[GateTelemetry] = []
        self.transformer_inputs: List[TransformerInputTelemetry] = []
        self.xgb_flows: List[XGBFlowTelemetry] = []
        self.mask_telemetry: Optional[FeatureMaskTelemetry] = None
        self.toggle_state: Optional[ToggleStateTelemetry] = None
        
        # Aggregated stats
        self.gate_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"executed": 0, "blocked": 0, "passed": 0})
        self.xgb_stats: Dict[str, int] = defaultdict(int)
        self.mask_stats: Dict[str, int] = defaultdict(int)
        
        # XGB usage tracking
        self.xgb_pre_predict_count = 0  # XGB predict called before transformer
        self.xgb_post_predict_count = 0  # XGB predict called after transformer
        self.xgb_veto_applied_count = 0  # XGB veto applied
        
        # Entry universe counters (standardized)
        self.bars_passed_hard_eligibility = 0  # Bars that passed hard eligibility gate
        self.bars_blocked_hard_eligibility = 0  # Bars that were blocked by hard eligibility gate
        self.bars_passed_soft_eligibility = 0  # Bars that passed soft eligibility gate (if exists)
        self.bars_blocked_soft_eligibility = 0  # Bars that were blocked by soft eligibility gate (if exists)
        
        # Transformer forward call tracking
        self.transformer_forward_calls = 0  # Number of times transformer model was called
        self.transformer_input_recorded = False  # Whether transformer input was recorded at least once
        
        # Model entry telemetry (for tracking why transformer may not be called)
        self.model_attempt_calls: Dict[str, int] = defaultdict(int)  # Per model name
        self.model_forward_calls: Dict[str, int] = defaultdict(int)  # Per model name
        self.model_block_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))  # model_name -> reason -> count
        
        # Entry routing telemetry (for tracking which model is selected and why)
        # Per-bar state (reset each bar)
        self.entry_routing_selected_model: Optional[str] = None  # "v10_hybrid", "legacy", "xgb_only", None
        self.entry_routing_reason: Optional[str] = None  # Reason for routing decision
        self.entry_routing_recorded: bool = False  # Whether routing was recorded for this bar
        
        # Aggregated routing telemetry (persist across bars, SSoT for master)
        self.entry_routing_selected_model_counts: Dict[str, int] = defaultdict(int)  # Histogram of selected models
        self.entry_routing_reason_counts: Dict[str, int] = defaultdict(int)  # Histogram of reasons
        self.entry_routing_total_recorded: int = 0  # Total number of bars where routing was recorded
        
        # V10 callsite bridge telemetry (tracks gap between routing and model call)
        self.v10_callsite_entered: int = 0  # Number of times we entered the V10 call site
        self.v10_callsite_returned: int = 0  # Number of times V10 call returned successfully
        self.v10_callsite_exception: int = 0  # Number of times V10 call raised exception
        self.v10_callsite_last: Optional[Dict[str, Any]] = None  # Last callsite event metadata
        self.v10_callsite_last_exception: Optional[Dict[str, Any]] = None  # Last exception metadata
        
        # V10 enable-state telemetry (SSoT for why V10 is enabled/disabled)
        self.entry_v10_enabled: Optional[bool] = None  # Whether V10 is enabled (per-bar state, reset each bar)
        self.entry_v10_enabled_reason: Optional[str] = None  # Reason for enable/disable state
        self.entry_v10_enabled_reason_counts: Dict[str, int] = defaultdict(int)  # Histogram of reasons (aggregated)
        self.entry_v10_enabled_true_count: int = 0  # Number of bars where V10 was enabled
        self.entry_v10_enabled_false_count: int = 0  # Number of bars where V10 was disabled
        
        # Control-flow telemetry (tracks execution path through entry evaluation)
        self.control_flow_counts: Dict[str, int] = defaultdict(int)  # Histogram of control-flow events
        self.control_flow_last: Optional[Dict[str, Any]] = None  # Last control-flow event metadata
        
        # Entry eval path telemetry (SSoT for which function actually performs entry evaluation)
        self.entry_eval_path_counts: Dict[str, int] = defaultdict(int)  # Histogram of entry eval paths
        self.entry_eval_path_last: Optional[Dict[str, Any]] = None  # Last entry eval path metadata
        
        # Exception gap telemetry (tracks exceptions between soft eligibility and Stage-0 check)
        self.exception_gap: Optional[Dict[str, Any]] = None  # Last exception in soft-to-stage0 gap
        
        # Sample collection (first N samples)
        self.max_samples = 100
        self.sample_count = 0
    
    def record_gate(
        self,
        gate_name: str,
        executed: bool,
        blocked: bool = False,
        passed: bool = False,
        reason: Optional[str] = None,
    ) -> None:
        """Record gate execution."""
        if self.sample_count < self.max_samples:
            gate = GateTelemetry(
                gate_name=gate_name,
                executed=executed,
                blocked=blocked,
                passed=passed,
                reason=reason,
                timestamp=datetime.utcnow().isoformat(),
            )
            self.gates.append(gate)
            self.sample_count += 1
        
        # Update stats
        self.gate_stats[gate_name]["executed"] += 1
        if blocked:
            self.gate_stats[gate_name]["blocked"] += 1
        if passed:
            self.gate_stats[gate_name]["passed"] += 1
        
        # Update entry universe counters (standardized)
        if gate_name == "hard_eligibility":
            if passed:
                self.bars_passed_hard_eligibility += 1
            elif blocked:
                self.bars_blocked_hard_eligibility += 1
        elif gate_name == "soft_eligibility":
            if passed:
                self.bars_passed_soft_eligibility += 1
            elif blocked:
                self.bars_blocked_soft_eligibility += 1
    
    def record_transformer_input(
        self,
        seq_shape: tuple,
        snap_shape: tuple,
        seq_feature_names: List[str],
        snap_feature_names: List[str],
        xgb_seq_channels: List[str],
        xgb_snap_channels: List[str],
        xgb_seq_values: Optional[Dict[str, float]] = None,
        xgb_snap_values: Optional[Dict[str, float]] = None,
        input_aliases_applied: Optional[Dict[str, str]] = None,  # DEL 2: CLOSE -> candles.close
    ) -> None:
        """Record transformer input.
        
        NOTE: Always records at least one transformer input (first call), regardless of sample_count.
        This is critical for verification - we need evidence that transformer was actually called.
        """
        # Always record the first transformer input (critical for verification)
        # Subsequent samples respect the max_samples limit
        if len(self.transformer_inputs) == 0 or self.sample_count < self.max_samples:
            telemetry = TransformerInputTelemetry(
                seq_shape=seq_shape,
                snap_shape=snap_shape,
                seq_feature_names=seq_feature_names,
                snap_feature_names=snap_feature_names,
                xgb_seq_channels=xgb_seq_channels,
                xgb_snap_channels=xgb_snap_channels,
                xgb_seq_values=xgb_seq_values,
                xgb_snap_values=xgb_snap_values,
                input_aliases_applied=input_aliases_applied or {},
                timestamp=datetime.utcnow().isoformat(),
            )
            self.transformer_inputs.append(telemetry)
            self.sample_count += 1
    
    def record_xgb_flow(
        self,
        xgb_called: bool,
        xgb_session: Optional[str] = None,
        xgb_p_long_raw: Optional[float] = None,
        xgb_p_long_cal: Optional[float] = None,
        xgb_margin: Optional[float] = None,
        xgb_p_hat: Optional[float] = None,
        xgb_uncertainty_score: Optional[float] = None,
        xgb_veto: bool = False,  # DEPRECATED: XGB POST removed 2026-01-24
        xgb_veto_reason: Optional[str] = None,  # DEPRECATED
        xgb_used_as: Optional[str] = None,  # Now always "pre"
        post_predict_called: bool = False,  # DEPRECATED: Always False
    ) -> None:
        """Record XGB flow. NOTE: XGB POST (calibration/veto) removed 2026-01-24."""
        if self.sample_count < self.max_samples:
            telemetry = XGBFlowTelemetry(
                xgb_called=xgb_called,
                xgb_session=xgb_session,
                xgb_p_long_raw=xgb_p_long_raw,
                xgb_p_long_cal=xgb_p_long_cal,
                xgb_margin=xgb_margin,
                xgb_p_hat=xgb_p_hat,
                xgb_uncertainty_score=xgb_uncertainty_score,
                xgb_veto=xgb_veto,
                xgb_veto_reason=xgb_veto_reason,
                xgb_used_as=xgb_used_as,
                post_predict_called=post_predict_called,
                timestamp=datetime.utcnow().isoformat(),
            )
            self.xgb_flows.append(telemetry)
            self.sample_count += 1
        
        # Update stats
        if xgb_called:
            self.xgb_stats["xgb_called"] += 1
            if xgb_used_as == "pre" or xgb_used_as == "both":
                self.xgb_pre_predict_count += 1
            if xgb_used_as == "post" or xgb_used_as == "both":
                self.xgb_post_predict_count += 1
        if post_predict_called:
            self.xgb_post_predict_count += 1
        if xgb_veto:
            self.xgb_stats["xgb_veto"] += 1
            self.xgb_veto_applied_count += 1
    
    def record_toggle_state(
        self,
        disable_xgb_channels_in_transformer_requested: bool,
        disable_xgb_channels_in_transformer_effective: bool,
        disable_xgb_post_transformer_requested: bool = False,
        disable_xgb_post_transformer_effective: bool = False,
        n_xgb_channels_in_transformer_input: int = 0,
    ) -> None:
        """Record toggle state."""
        self.toggle_state = ToggleStateTelemetry(
            disable_xgb_channels_in_transformer_requested=disable_xgb_channels_in_transformer_requested,
            disable_xgb_channels_in_transformer_effective=disable_xgb_channels_in_transformer_effective,
            disable_xgb_post_transformer_requested=disable_xgb_post_transformer_requested,
            disable_xgb_post_transformer_effective=disable_xgb_post_transformer_effective,
            n_xgb_channels_in_transformer_input=n_xgb_channels_in_transformer_input,
            timestamp=datetime.utcnow().isoformat(),
        )
    
    def record_mask(
        self,
        mask_enabled: bool,
        mask_families: List[str],
        mask_strategy: Optional[str] = None,
        masked_features: List[str] = None,
        sample_masked_values: Dict[str, float] = None,
    ) -> None:
        """Record feature masking."""
        self.mask_telemetry = FeatureMaskTelemetry(
            mask_enabled=mask_enabled,
            mask_families=mask_families or [],
            mask_strategy=mask_strategy,
            masked_features=masked_features or [],
            sample_masked_values=sample_masked_values or {},
            timestamp=datetime.utcnow().isoformat(),
        )
        
        # Update stats
        if mask_enabled:
            self.mask_stats["mask_enabled"] = 1
            self.mask_stats["masked_features_count"] = len(masked_features or [])
    
    def record_channel_mask(
        self,
        masked_channels: List[str],
        kept_channels: List[str],
        effective_seq_channels: List[str],
        effective_snap_channels: List[str],
    ) -> None:
        """Record XGB channel masking for ablation analysis (DEL C)."""
        if not hasattr(self, "channel_mask_info"):
            self.channel_mask_info = {
                "masked_channels": masked_channels,
                "kept_channels": kept_channels,
                "effective_seq_channels": effective_seq_channels,
                "effective_snap_channels": effective_snap_channels,
                "n_masked": len(masked_channels),
                "n_kept": len(kept_channels),
            }
        # Only record once (first call)
    
    def record_model_attempt(self, model_name: str) -> None:
        """Record that a model entry was attempted."""
        self.model_attempt_calls[model_name] += 1
    
    def record_model_forward(self, model_name: str) -> None:
        """Record that a model forward pass was executed."""
        self.model_forward_calls[model_name] += 1
        # Also increment transformer_forward_calls for backward compatibility
        if model_name == "v10_hybrid":
            self.transformer_forward_calls += 1
    
    def record_model_block(self, model_name: str, reason: str) -> None:
        """Record that a model entry was blocked with a specific reason."""
        self.model_block_counts[model_name][reason] += 1
    
    def record_entry_routing(self, selected_model: Optional[str], reason: str) -> None:
        """
        Record entry routing decision (which model was selected and why).
        
        This should be called once per bar evaluation, right before the routing decision
        is executed. It updates both per-bar state and aggregated histogram.
        
        Args:
            selected_model: The model that was selected ("v10_hybrid", "legacy", "xgb_only", or None)
            reason: Reason for the routing decision (e.g., "ROUTED_TO_V10", "ENTRY_V10_DISABLED_BY_POLICY")
        """
        import os
        require_telemetry = os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
        
        # Check if already recorded for this bar (idempotency check)
        if self.entry_routing_recorded:
            if require_telemetry:
                raise RuntimeError(
                    f"[ENTRY_ROUTING_TELEMETRY] FATAL: record_entry_routing() called multiple times for the same bar. "
                    f"Current: selected_model={self.entry_routing_selected_model}, reason={self.entry_routing_reason}. "
                    f"New: selected_model={selected_model}, reason={reason}. "
                    f"This indicates a wiring bug - routing should be recorded exactly once per bar."
                )
            else:
                log.debug(
                    f"[ENTRY_ROUTING_TELEMETRY] record_entry_routing() called multiple times for the same bar. "
                    f"Ignoring subsequent call."
                )
                return
        
        # Update per-bar state
        self.entry_routing_selected_model = selected_model
        self.entry_routing_reason = reason
        self.entry_routing_recorded = True
        
        # Update aggregated histogram (SSoT for master)
        selected_model_key = selected_model if selected_model is not None else "NONE"
        self.entry_routing_selected_model_counts[selected_model_key] += 1
        self.entry_routing_reason_counts[reason] += 1
        self.entry_routing_total_recorded += 1
    
    def record_v10_enable_state(self, enabled: bool, reason: str) -> None:
        """
        Record V10 enable state and reason (SSoT for why V10 is enabled/disabled).
        
        This should be called once per bar evaluation, right before routing decision.
        
        Args:
            enabled: Whether V10 is enabled for this bar
            reason: Reason for enable/disable state (e.g., "ENABLED", "POLICY_DISABLED", "BUNDLE_NOT_LOADED")
        """
        # Update per-bar state
        self.entry_v10_enabled = enabled
        self.entry_v10_enabled_reason = reason
        
        # Update aggregated histogram
        self.entry_v10_enabled_reason_counts[reason] += 1
        if enabled:
            self.entry_v10_enabled_true_count += 1
        else:
            self.entry_v10_enabled_false_count += 1
    
    def reset_routing_for_next_bar(self) -> None:
        """
        Reset routing telemetry for the next bar evaluation.
        
        Only resets per-bar state. Aggregated histogram fields are NOT reset
        (they persist across bars as SSoT for master telemetry).
        """
        self.entry_routing_selected_model = None
        self.entry_routing_reason = None
        self.entry_routing_recorded = False
        self.entry_v10_enabled = None
        self.entry_v10_enabled_reason = None
        self.entry_eval_path_last = None
    
    def record_control_flow(self, event: str, **meta: Any) -> None:
        """
        Record control-flow event (execution path sentinel).
        
        This tracks execution flow through entry evaluation to identify
        where early returns or exceptions occur.
        
        Args:
            event: Event name (e.g., "AFTER_HARD_ELIGIBILITY_PASSED", "EARLY_RETURN_IN_GAP")
            **meta: Optional metadata (e.g., reason, line, ts, context)
        """
        # Update aggregated histogram
        self.control_flow_counts[event] += 1
        
        # Update last event (for debugging)
        self.control_flow_last = {
            "event": event,
            **meta,
        }
    
    def record_entry_eval_path(self, function: str, file: str, line: int, **meta: Any) -> None:
        """
        Record which function actually performs entry evaluation (SSoT).
        
        This should be called at the very top of every entry evaluation function
        to identify which code path is actually being executed.
        
        Args:
            function: Function name (e.g., "evaluate_entry", "evaluate_entry_v9")
            file: File path (e.g., __file__)
            line: Line number (e.g., inspect.currentframe().f_lineno)
            **meta: Optional metadata
        """
        # Create path identifier (function@file:line)
        path_id = f"{function}@{Path(file).name}:{line}"
        
        # Update aggregated histogram
        self.entry_eval_path_counts[path_id] += 1
        
        # Update last path (for debugging)
        self.entry_eval_path_last = {
            "function": function,
            "file": file,
            "line": line,
            "path_id": path_id,
            **meta,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry to dictionary."""
        # Compute xgb_used_as summary
        xgb_used_as_summary = "none"
        if self.xgb_pre_predict_count > 0 and self.xgb_post_predict_count > 0:
            xgb_used_as_summary = "both"
        elif self.xgb_pre_predict_count > 0:
            xgb_used_as_summary = "pre"
        elif self.xgb_post_predict_count > 0:
            xgb_used_as_summary = "post"
        
        return {
            "gates": [asdict(g) for g in self.gates],
            "transformer_inputs": [asdict(t) for t in self.transformer_inputs],
            "xgb_flows": [asdict(x) for x in self.xgb_flows],
            "mask_telemetry": asdict(self.mask_telemetry) if self.mask_telemetry else None,
            "toggle_state": asdict(self.toggle_state) if self.toggle_state else None,
            "gate_stats": dict(self.gate_stats),
            "xgb_stats": dict(self.xgb_stats),
            "mask_stats": dict(self.mask_stats),
            "xgb_usage_summary": {
                "xgb_used_as": xgb_used_as_summary,
                "xgb_pre_predict_count": self.xgb_pre_predict_count,
                "xgb_post_predict_count": self.xgb_post_predict_count,
                "xgb_veto_applied_count": self.xgb_veto_applied_count,
            },
        }
    
    def write_entry_features_used(self, output_path: Path) -> None:
        """Write ENTRY_FEATURES_USED.json."""
        # Compute xgb_used_as summary
        xgb_used_as_summary = "none"
        if self.xgb_pre_predict_count > 0 and self.xgb_post_predict_count > 0:
            xgb_used_as_summary = "both"
        elif self.xgb_pre_predict_count > 0:
            xgb_used_as_summary = "pre"
        elif self.xgb_post_predict_count > 0:
            xgb_used_as_summary = "post"
        
        n_xgb_channels = -1  # Default to -1 if not set (indicates telemetry not collected)
        if self.toggle_state:
            n_xgb_channels = self.toggle_state.n_xgb_channels_in_transformer_input
        
        # Handle case where transformer_inputs is empty
        # Validation logic:
        # - If transformer_forward_calls > 0: must have recorded transformer input (fail-fast if required)
        # - If transformer_forward_calls == 0: allow empty only if no_entry_evaluations=true
        import os
        require_telemetry = os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
        no_entry_evaluations = os.getenv("GX1_TELEMETRY_NO_ENTRY_EVALUATIONS", "0") == "1"
        
        if not self.transformer_inputs:
            # Check if transformer was actually called
            if self.transformer_forward_calls > 0:
                # Transformer was called but input was not recorded - this is a wiring bug
                if require_telemetry:
                    raise RuntimeError(
                        f"[ENTRY_FEATURES_TELEMETRY] FATAL: transformer_forward_calls={self.transformer_forward_calls} > 0 "
                        f"but transformer_input_recorded={self.transformer_input_recorded}. "
                        f"This indicates telemetry wiring bug: transformer was called but input was not recorded. "
                        f"Check that record_transformer_input() is called in the first-call capture hook."
                    )
                else:
                    log.warning(
                        f"[ENTRY_FEATURES_TELEMETRY] transformer_forward_calls={self.transformer_forward_calls} > 0 "
                        f"but transformer_input_recorded={self.transformer_input_recorded}. "
                        f"This indicates telemetry wiring bug."
                    )
            elif no_entry_evaluations:
                # Explicitly tagged as no entry evaluations (e.g., warmup, pregate blocks all)
                log.info(
                    f"[ENTRY_FEATURES_TELEMETRY] No transformer_inputs recorded (no_entry_evaluations=true, transformer_forward_calls=0). "
                    f"Writing ENTRY_FEATURES_USED.json with no_entry_evaluations flag."
                )
                data = {
                    "no_entry_evaluations": True,
                    "no_entry_evaluations_reason": os.getenv("GX1_TELEMETRY_NO_ENTRY_REASON", "unknown"),
                    "transformer_forward_calls": self.transformer_forward_calls,
                    "transformer_input_recorded": self.transformer_input_recorded,
                    "seq_features": {"names": [], "count": 0},
                    "snap_features": {"names": [], "count": 0},
                    "xgb_seq_channels": {"names": [], "count": 0},
                    "xgb_snap_channels": {"names": [], "count": 0},
                    "xgb_flow": {
                        "xgb_used_as": xgb_used_as_summary,
                        "n_xgb_channels_in_transformer_input": n_xgb_channels,
                        "xgb_pre_predict_count": self.xgb_pre_predict_count,
                        "xgb_post_predict_count": self.xgb_post_predict_count,
                        "post_predict_called": self.xgb_post_predict_count > 0,
                        "veto_applied_count": self.xgb_veto_applied_count,
                    },
                    "toggles": {
                        "disable_xgb_channels_in_transformer_requested": self.toggle_state.disable_xgb_channels_in_transformer_requested if self.toggle_state else False,
                        "disable_xgb_channels_in_transformer_effective": self.toggle_state.disable_xgb_channels_in_transformer_effective if self.toggle_state else False,
                        "disable_xgb_post_transformer_requested": self.toggle_state.disable_xgb_post_transformer_requested if self.toggle_state else False,
                        "disable_xgb_post_transformer_effective": self.toggle_state.disable_xgb_post_transformer_effective if self.toggle_state else False,
                    },
                    "gate_stats": dict(self.gate_stats),
                    "xgb_stats": dict(self.xgb_stats),
                    "entry_routing": {
                        "selected_model": self.entry_routing_selected_model,
                        "reason": self.entry_routing_reason,
                        "recorded": self.entry_routing_recorded,
                    },
                    "entry_routing_aggregate": {
                        "total_recorded": self.entry_routing_total_recorded,
                        "selected_model_counts": dict(self.entry_routing_selected_model_counts),
                        "reason_counts": dict(self.entry_routing_reason_counts),
                    },
                    "v10_callsite": {
                        "entered": self.v10_callsite_entered,
                        "returned": self.v10_callsite_returned,
                        "exception": self.v10_callsite_exception,
                        "last": self.v10_callsite_last,
                        "last_exception": self.v10_callsite_last_exception,
                    },
                    "entry_v10_enable_state": {
                        "enabled": self.entry_v10_enabled,
                        "reason": self.entry_v10_enabled_reason,
                        "enabled_true_count": self.entry_v10_enabled_true_count,
                        "enabled_false_count": self.entry_v10_enabled_false_count,
                        "reason_counts": dict(self.entry_v10_enabled_reason_counts),
                    },
                    "control_flow": {
                        "counts": dict(self.control_flow_counts),
                        "last": self.control_flow_last,
                    },
                    "entry_eval_path": {
                        "counts": dict(self.entry_eval_path_counts),
                        "last": self.entry_eval_path_last,
                    },
                    "soft_eligibility_truth": {
                        "return_true_count": self.control_flow_counts.get("SOFT_ELIGIBILITY_RETURN_TRUE", 0),
                        "return_false_count": self.control_flow_counts.get("SOFT_ELIGIBILITY_RETURN_FALSE", 0),
                        "after_passed_count": self.control_flow_counts.get("AFTER_SOFT_ELIGIBILITY_PASSED", 0),
                        "blocked_branch_count": self.control_flow_counts.get("SOFT_ELIGIBILITY_BLOCKED_BRANCH", 0),
                    },
                    "exception_gap": self.exception_gap,
                }
            else:
                # transformer_forward_calls == 0 and no_entry_evaluations not set
                # This can happen if bars reached entry stage but transformer was never called
                # (e.g., all bars blocked by gates before transformer call)
                # In this case, we should allow empty lists but log a warning
                if require_telemetry:
                    # Check if bars actually reached entry stage (from entry_universe counters)
                    bars_reached = self.bars_passed_hard_eligibility + self.bars_blocked_hard_eligibility
                    if bars_reached > 0:
                        # Bars reached entry stage but transformer was never called
                        # This is a valid scenario (all bars blocked by gates)
                        log.warning(
                            f"[ENTRY_FEATURES_TELEMETRY] transformer_forward_calls={self.transformer_forward_calls}, "
                            f"but bars_reached_entry_stage={bars_reached} > 0. "
                            f"This indicates all bars were blocked by gates before transformer call. "
                            f"Writing ENTRY_FEATURES_USED.json with no_entry_evaluations=true."
                        )
                        data = {
                            "no_entry_evaluations": True,
                            "no_entry_evaluations_reason": "all_bars_blocked_by_gates",
                            "transformer_forward_calls": self.transformer_forward_calls,
                            "transformer_input_recorded": self.transformer_input_recorded,
                            "bars_passed_hard_eligibility": self.bars_passed_hard_eligibility,
                            "bars_blocked_hard_eligibility": self.bars_blocked_hard_eligibility,
                            "bars_passed_soft_eligibility": self.bars_passed_soft_eligibility,
                            "bars_blocked_soft_eligibility": self.bars_blocked_soft_eligibility,
                            "seq_features": {"names": [], "count": 0},
                            "snap_features": {"names": [], "count": 0},
                            "xgb_seq_channels": {"names": [], "count": 0},
                            "xgb_snap_channels": {"names": [], "count": 0},
                            "input_aliases_applied": {},  # DEL 2: CLOSE -> candles.close (empty when no evaluations)
                            "xgb_flow": {
                                "xgb_used_as": xgb_used_as_summary,
                                "n_xgb_channels_in_transformer_input": n_xgb_channels,
                                "xgb_pre_predict_count": self.xgb_pre_predict_count,
                                "xgb_post_predict_count": self.xgb_post_predict_count,
                                "post_predict_called": self.xgb_post_predict_count > 0,
                                "veto_applied_count": self.xgb_veto_applied_count,
                            },
                            "toggles": {
                                "disable_xgb_channels_in_transformer_requested": self.toggle_state.disable_xgb_channels_in_transformer_requested if self.toggle_state else False,
                                "disable_xgb_channels_in_transformer_effective": self.toggle_state.disable_xgb_channels_in_transformer_effective if self.toggle_state else False,
                                "disable_xgb_post_transformer_requested": self.toggle_state.disable_xgb_post_transformer_requested if self.toggle_state else False,
                                "disable_xgb_post_transformer_effective": self.toggle_state.disable_xgb_post_transformer_effective if self.toggle_state else False,
                            },
                            "gate_stats": dict(self.gate_stats),
                            "xgb_stats": dict(self.xgb_stats),
                            "entry_universe": {
                                "bars_passed_hard_eligibility": self.bars_passed_hard_eligibility,
                                "bars_blocked_hard_eligibility": self.bars_blocked_hard_eligibility,
                                "bars_passed_soft_eligibility": self.bars_passed_soft_eligibility,
                                "bars_blocked_soft_eligibility": self.bars_blocked_soft_eligibility,
                            },
                            "entry_routing": {
                                "selected_model": self.entry_routing_selected_model,
                                "reason": self.entry_routing_reason,
                                "recorded": self.entry_routing_recorded,
                            },
                            "entry_routing_aggregate": {
                                "total_recorded": self.entry_routing_total_recorded,
                                "selected_model_counts": dict(self.entry_routing_selected_model_counts),
                                "reason_counts": dict(self.entry_routing_reason_counts),
                            },
                            "v10_callsite": {
                                "entered": self.v10_callsite_entered,
                                "returned": self.v10_callsite_returned,
                                "exception": self.v10_callsite_exception,
                                "last": self.v10_callsite_last,
                                "last_exception": self.v10_callsite_last_exception,
                            },
                            "model_entry": {
                                "model_attempt_calls": dict(self.model_attempt_calls),
                                "model_forward_calls": dict(self.model_forward_calls),
                                "model_block_counts": {k: dict(v) for k, v in self.model_block_counts.items()},
                            },
                            "entry_v10_enable_state": {
                                "enabled": self.entry_v10_enabled,
                                "reason": self.entry_v10_enabled_reason,
                                "enabled_true_count": self.entry_v10_enabled_true_count,
                                "enabled_false_count": self.entry_v10_enabled_false_count,
                                "reason_counts": dict(self.entry_v10_enabled_reason_counts),
                            },
                            "control_flow": {
                                "counts": dict(self.control_flow_counts),
                                "last": self.control_flow_last,
                            },
                            "entry_eval_path": {
                                "counts": dict(self.entry_eval_path_counts),
                                "last": self.entry_eval_path_last,
                            },
                            "exception_gap": self.exception_gap,
                        }
                    else:
                        # No bars reached entry stage - this is the normal no_entry_evaluations case
                        raise RuntimeError(
                            f"[ENTRY_FEATURES_TELEMETRY] FATAL: transformer_forward_calls={self.transformer_forward_calls}, "
                            f"transformer_input_recorded={self.transformer_input_recorded}, "
                            f"no_entry_evaluations={no_entry_evaluations}, "
                            f"bars_reached_entry_stage={bars_reached}. "
                            f"This indicates telemetry was not collected during entry evaluation. "
                            f"Check that record_transformer_input() is called in entry flow."
                        )
                # Otherwise, log warning and write stub
                log.warning(
                    f"[ENTRY_FEATURES_TELEMETRY] No transformer_inputs recorded. "
                    f"Writing stub ENTRY_FEATURES_USED.json with empty feature lists. "
                    f"This may indicate telemetry was not collected."
                )
                data = {
                    "seq_features": {"names": [], "count": 0},
                    "snap_features": {"names": [], "count": 0},
                    "xgb_seq_channels": {"names": [], "count": 0},
                    "xgb_snap_channels": {"names": [], "count": 0},
                    "xgb_flow": {
                        "xgb_used_as": xgb_used_as_summary,
                        "n_xgb_channels_in_transformer_input": n_xgb_channels,
                        "xgb_pre_predict_count": self.xgb_pre_predict_count,
                        "xgb_post_predict_count": self.xgb_post_predict_count,
                        "post_predict_called": self.xgb_post_predict_count > 0,
                        "veto_applied_count": self.xgb_veto_applied_count,
                    },
                    "toggles": {
                        "disable_xgb_channels_in_transformer_requested": self.toggle_state.disable_xgb_channels_in_transformer_requested if self.toggle_state else False,
                        "disable_xgb_channels_in_transformer_effective": self.toggle_state.disable_xgb_channels_in_transformer_effective if self.toggle_state else False,
                        "disable_xgb_post_transformer_requested": self.toggle_state.disable_xgb_post_transformer_requested if self.toggle_state else False,
                        "disable_xgb_post_transformer_effective": self.toggle_state.disable_xgb_post_transformer_effective if self.toggle_state else False,
                    },
                    "gate_stats": dict(self.gate_stats),
                    "xgb_stats": dict(self.xgb_stats),
                    "entry_universe": {
                        "bars_passed_hard_eligibility": self.bars_passed_hard_eligibility,
                        "bars_blocked_hard_eligibility": self.bars_blocked_hard_eligibility,
                        "bars_passed_soft_eligibility": self.bars_passed_soft_eligibility,
                        "bars_blocked_soft_eligibility": self.bars_blocked_soft_eligibility,
                    },
                    "transformer_forward_calls": self.transformer_forward_calls,
                    "transformer_input_recorded": self.transformer_input_recorded,
                    "model_entry": {
                        "model_attempt_calls": dict(self.model_attempt_calls),
                        "model_forward_calls": dict(self.model_forward_calls),
                        "model_block_counts": {k: dict(v) for k, v in self.model_block_counts.items()},
                    },
                    "entry_routing": {
                        "selected_model": self.entry_routing_selected_model,
                        "reason": self.entry_routing_reason,
                        "recorded": self.entry_routing_recorded,
                    },
                    "entry_routing_aggregate": {
                        "total_recorded": self.entry_routing_total_recorded,
                        "selected_model_counts": dict(self.entry_routing_selected_model_counts),
                        "reason_counts": dict(self.entry_routing_reason_counts),
                    },
                    "v10_callsite": {
                        "entered": self.v10_callsite_entered,
                        "returned": self.v10_callsite_returned,
                        "exception": self.v10_callsite_exception,
                        "last": self.v10_callsite_last,
                        "last_exception": self.v10_callsite_last_exception,
                    },
                    "entry_v10_enable_state": {
                        "enabled": self.entry_v10_enabled,
                        "reason": self.entry_v10_enabled_reason,
                        "enabled_true_count": self.entry_v10_enabled_true_count,
                        "enabled_false_count": self.entry_v10_enabled_false_count,
                        "reason_counts": dict(self.entry_v10_enabled_reason_counts),
                    },
                    "control_flow": {
                        "counts": dict(self.control_flow_counts),
                        "last": self.control_flow_last,
                    },
                    "entry_eval_path": {
                        "counts": dict(self.entry_eval_path_counts),
                        "last": self.entry_eval_path_last,
                    },
                    "exception_gap": self.exception_gap,
                }
        else:
            # DEL 2: Get input_aliases_applied from first transformer input (if available)
            input_aliases_applied = {}
            if self.transformer_inputs and hasattr(self.transformer_inputs[0], "input_aliases_applied"):
                input_aliases_applied = self.transformer_inputs[0].input_aliases_applied or {}
            
            data = {
                "seq_features": {
                    "names": self.transformer_inputs[0].seq_feature_names if self.transformer_inputs else [],
                    "count": len(self.transformer_inputs[0].seq_feature_names) if self.transformer_inputs else 0,
                },
                "snap_features": {
                    "names": self.transformer_inputs[0].snap_feature_names if self.transformer_inputs else [],
                    "count": len(self.transformer_inputs[0].snap_feature_names) if self.transformer_inputs else 0,
                },
                "input_aliases_applied": input_aliases_applied,  # DEL 2: CLOSE -> candles.close
                "xgb_seq_channels": {
                    "names": self.transformer_inputs[0].xgb_seq_channels if self.transformer_inputs else [],
                    "count": len(self.transformer_inputs[0].xgb_seq_channels) if self.transformer_inputs else 0,
                },
                "xgb_snap_channels": {
                    "names": self.transformer_inputs[0].xgb_snap_channels if self.transformer_inputs else [],
                    "count": len(self.transformer_inputs[0].xgb_snap_channels) if self.transformer_inputs else 0,
                },
                "xgb_flow": {
                    "xgb_used_as": xgb_used_as_summary,
                    "n_xgb_channels_in_transformer_input": n_xgb_channels,
                    "xgb_pre_predict_count": self.xgb_pre_predict_count,
                    "xgb_post_predict_count": self.xgb_post_predict_count,
                    "post_predict_called": self.xgb_post_predict_count > 0,
                    "veto_applied_count": self.xgb_veto_applied_count,
                },
                "toggles": {
                    "disable_xgb_channels_in_transformer_requested": self.toggle_state.disable_xgb_channels_in_transformer_requested if self.toggle_state else False,
                    "disable_xgb_channels_in_transformer_effective": self.toggle_state.disable_xgb_channels_in_transformer_effective if self.toggle_state else False,
                    "disable_xgb_post_transformer_requested": self.toggle_state.disable_xgb_post_transformer_requested if self.toggle_state else False,
                    "disable_xgb_post_transformer_effective": self.toggle_state.disable_xgb_post_transformer_effective if self.toggle_state else False,
                },
                "gate_stats": dict(self.gate_stats),
                "xgb_stats": dict(self.xgb_stats),
                "transformer_forward_calls": self.transformer_forward_calls,
                "transformer_input_recorded": self.transformer_input_recorded,
                "model_entry": {
                    "model_attempt_calls": dict(self.model_attempt_calls),
                    "model_forward_calls": dict(self.model_forward_calls),
                    "model_block_counts": {k: dict(v) for k, v in self.model_block_counts.items()},
                },
                "entry_routing": {
                    "selected_model": self.entry_routing_selected_model,
                    "reason": self.entry_routing_reason,
                    "recorded": self.entry_routing_recorded,
                },
                "entry_routing_aggregate": {
                    "total_recorded": self.entry_routing_total_recorded,
                    "selected_model_counts": dict(self.entry_routing_selected_model_counts),
                    "reason_counts": dict(self.entry_routing_reason_counts),
                },
                "v10_callsite": {
                    "entered": self.v10_callsite_entered,
                    "returned": self.v10_callsite_returned,
                    "exception": self.v10_callsite_exception,
                    "last": self.v10_callsite_last,
                    "last_exception": self.v10_callsite_last_exception,
                },
                "entry_v10_enable_state": {
                    "enabled": self.entry_v10_enabled,
                    "reason": self.entry_v10_enabled_reason,
                    "enabled_true_count": self.entry_v10_enabled_true_count,
                    "enabled_false_count": self.entry_v10_enabled_false_count,
                    "reason_counts": dict(self.entry_v10_enabled_reason_counts),
                },
                "control_flow": {
                    "counts": dict(self.control_flow_counts),
                    "last": self.control_flow_last,
                },
                "entry_eval_path": {
                    "counts": dict(self.entry_eval_path_counts),
                    "last": self.entry_eval_path_last,
                },
                "soft_eligibility_truth": {
                    "return_true_count": self.control_flow_counts.get("SOFT_ELIGIBILITY_RETURN_TRUE", 0),
                    "return_false_count": self.control_flow_counts.get("SOFT_ELIGIBILITY_RETURN_FALSE", 0),
                    "after_passed_count": self.control_flow_counts.get("AFTER_SOFT_ELIGIBILITY_PASSED", 0),
                    "blocked_branch_count": self.control_flow_counts.get("SOFT_ELIGIBILITY_BLOCKED_BRANCH", 0),
                },
                "exception_gap": self.exception_gap,
            }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        log.info(f"[ENTRY_FEATURES_TELEMETRY] Wrote ENTRY_FEATURES_USED.json to {output_path}")
    
    def write_entry_features_used_md(self, output_path: Path) -> None:
        """Write ENTRY_FEATURES_USED.md."""
        lines = [
            "# Entry Features Used",
            "",
            "This document lists all features used in entry evaluation.",
            "",
            "## Sequence Features",
            "",
        ]
        
        if self.transformer_inputs:
            seq_names = self.transformer_inputs[0].seq_feature_names
            for i, name in enumerate(seq_names, 1):
                lines.append(f"{i}. `{name}`")
        else:
            lines.append("(No sequence features recorded)")
        
        lines.extend([
            "",
            "## Snapshot Features",
            "",
        ])
        
        if self.transformer_inputs:
            snap_names = self.transformer_inputs[0].snap_feature_names
            for i, name in enumerate(snap_names, 1):
                lines.append(f"{i}. `{name}`")
        else:
            lines.append("(No snapshot features recorded)")
        
        lines.extend([
            "",
            "## XGB Sequence Channels",
            "",
        ])
        
        if self.transformer_inputs:
            xgb_seq = self.transformer_inputs[0].xgb_seq_channels
            for i, name in enumerate(xgb_seq, 1):
                lines.append(f"{i}. `{name}`")
        else:
            lines.append("(No XGB sequence channels recorded)")
        
        lines.extend([
            "",
            "## XGB Snapshot Channels",
            "",
        ])
        
        if self.transformer_inputs:
            xgb_snap = self.transformer_inputs[0].xgb_snap_channels
            for i, name in enumerate(xgb_snap, 1):
                lines.append(f"{i}. `{name}`")
        else:
            lines.append("(No XGB snapshot channels recorded)")
        
        lines.extend([
            "",
            "## Gate Statistics",
            "",
            "| Gate Name | Executed | Blocked | Passed |",
            "|-----------|----------|---------|--------|",
        ])
        
        for gate_name, stats in self.gate_stats.items():
            lines.append(
                f"| {gate_name} | {stats['executed']} | {stats['blocked']} | {stats['passed']} |"
            )
        
        lines.extend([
            "",
            "## XGB Statistics",
            "",
            "| Metric | Count |",
            "|--------|-------|",
        ])
        
        for metric, count in self.xgb_stats.items():
            lines.append(f"| {metric} | {count} |")
        
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        
        log.info(f"[ENTRY_FEATURES_TELEMETRY] Wrote ENTRY_FEATURES_USED.md to {output_path}")
    
    def write_mask_applied(self, output_path: Path) -> None:
        """Write FEATURE_MASK_APPLIED.json."""
        if not self.mask_telemetry or not self.mask_telemetry.mask_enabled:
            return
        
        data = asdict(self.mask_telemetry)
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        log.info(f"[ENTRY_FEATURES_TELEMETRY] Wrote FEATURE_MASK_APPLIED.json to {output_path}")
    
    def write_all(self, output_dir: Path) -> None:
        """Write all telemetry files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.write_entry_features_used(output_dir / "ENTRY_FEATURES_USED.json")
        self.write_entry_features_used_md(output_dir / "ENTRY_FEATURES_USED.md")
        
        if self.mask_telemetry and self.mask_telemetry.mask_enabled:
            self.write_mask_applied(output_dir / "FEATURE_MASK_APPLIED.json")
        
        # Write full telemetry JSON
        telemetry_path = output_dir / "ENTRY_FEATURES_TELEMETRY.json"
        with open(telemetry_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        log.info(f"[ENTRY_FEATURES_TELEMETRY] Wrote full telemetry to {telemetry_path}")
