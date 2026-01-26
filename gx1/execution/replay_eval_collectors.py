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
    """
    outcomes: List[Dict[str, Any]] = field(default_factory=list)
    
    def collect(
        self,
        trade_id: str,
        pnl_bps: float,
        mae_bps: Optional[float] = None,
        mfe_bps: Optional[float] = None,
        duration_bars: Optional[int] = None,
        session: Optional[str] = None,
        exit_reason: Optional[str] = None,
    ):
        """
        Collect trade outcome.
        
        Args:
            trade_id: Trade identifier
            pnl_bps: Realized PnL in basis points
            mae_bps: Maximum Adverse Excursion (if available)
            mfe_bps: Maximum Favorable Excursion (if available)
            duration_bars: Bars held (if available)
            session: Entry session (if available)
            exit_reason: Exit reason (if available)
        """
        self.outcomes.append({
            "trade_id": trade_id,
            "pnl_bps": float(pnl_bps),
            "mae_bps": float(mae_bps) if mae_bps is not None else None,
            "mfe_bps": float(mfe_bps) if mfe_bps is not None else None,
            "duration_bars": int(duration_bars) if duration_bars is not None else None,
            "session": session,
            "exit_reason": exit_reason,
        })
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected outcomes to DataFrame."""
        if not self.outcomes:
            return pd.DataFrame()
        return pd.DataFrame(self.outcomes)
    
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
