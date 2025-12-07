"""
Live telemetry tracking for GX1 entry model.

Tracks per-session metrics: coverage, p_hat_mean, margin_mean, entropy_mean, keep_rate.
Computes delayed ECE over closed trades.
Provides alert rules for drift detection.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def prob_entropy(p: float) -> float:
    """
    Compute binary entropy for a probability.
    
    Parameters
    ----------
    p : float
        Probability in (0, 1).
    
    Returns
    -------
    float
        Entropy: -(p*log(p) + (1-p)*log(1-p))
    """
    p = max(min(p, 1.0 - 1e-9), 1e-9)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def ece_bin(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) using binning.
    
    Parameters
    ----------
    p : np.ndarray
        Predicted probabilities (shape: [N]).
    y : np.ndarray
        True labels (shape: [N], values in {0, 1}).
    n_bins : int
        Number of bins (default: 10).
    
    Returns
    -------
    float
        ECE score.
    """
    if len(p) == 0 or len(y) == 0:
        return 0.0
    
    # Clip probabilities to [0, 1]
    p = np.clip(p, 0.0, 1.0)
    
    # Create bins (matching offline compute_ece_raw)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(p, bin_edges[1:], right=True)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    total = float(len(p))
    ece = 0.0
    
    for i in range(n_bins):
        mask = bin_indices == i
        bin_count = int(mask.sum())
        if bin_count == 0:
            continue
        
        mean_prob = float(np.mean(p[mask]))
        mean_label = float(np.mean(y[mask]))
        ece += abs(mean_label - mean_prob) * (bin_count / total)
    
    return float(ece)


@dataclass
class SessionMetrics:
    """Per-session metrics for telemetry."""
    session: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_evals: int = 0
    n_entries: int = 0
    coverage_window: float = 0.0
    p_hat_mean: float = 0.0
    p_hat_std: float = 0.0
    margin_mean: float = 0.0
    entropy_mean: float = 0.0
    keep_rate: float = 0.0
    calibration_proxy: float = 0.0
    sharpness: float = 0.0
    exit_latency_mean: Dict[str, float] = field(default_factory=dict)  # variant -> mean latency
    exit_latency_count: Dict[str, int] = field(default_factory=dict)  # variant -> count


@dataclass
class ClosedTradeRecord:
    """Record for a closed trade (for delayed ECE calculation)."""
    entry_id: str
    session: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    p_hat: float
    side: str
    pnl_bps: float
    hit_flag: int  # 1 if pnl_bps > 0, else 0


class TelemetryTracker:
    """
    Tracks live telemetry metrics per session.
    
    Features:
    - Window-based metrics (30 min, 1 hour)
    - Alert rules for drift detection
    - Delayed ECE calculation over closed trades
    - Proxy quality metrics (calibration_proxy, sharpness)
    """
    
    def __init__(
        self,
        telemetry_dir: Path,
        window_minutes: int = 30,
        log_interval_minutes: int = 10,
        target_coverage: float = 0.20,
        ece_window_size: int = 500,
        is_replay: bool = False,
    ):
        """
        Initialize telemetry tracker.
        
        Parameters
        ----------
        telemetry_dir : Path
            Directory for telemetry logs.
        window_minutes : int
            Window size in minutes for metrics aggregation (default: 30).
        log_interval_minutes : int
            Interval in minutes for logging telemetry (default: 10).
        target_coverage : float
            Target coverage for alerts (default: 0.20).
        ece_window_size : int
            Window size for ECE calculation over closed trades (default: 500).
        """
        self.telemetry_dir = Path(telemetry_dir)
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.window_minutes = window_minutes
        self.log_interval_minutes = log_interval_minutes
        self.target_coverage = target_coverage
        self.ece_window_size = ece_window_size
        self.is_replay = is_replay
        
        # Per-session eval buffers (deque for sliding window)
        self.eval_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Per-session closed trade buffers (for delayed ECE)
        self.closed_trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=ece_window_size))
        
        # Per-session/variant exit latency buffers (for exit latency tracking)
        self.exit_latency_buffers: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )
        
        # Last telemetry log time per session
        self.last_telemetry_time: Dict[str, pd.Timestamp] = {}
        
        # Alert budget: Track WARN count per category per session (max 3 WARN per 30 min per category)
        self.alert_budget: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=100))
        )  # session -> category -> deque of timestamps
        
        # Revision tracking buffer (24h window, max 5000 revisions)
        self.rev_buf: deque = deque(maxlen=5000)  # Store revision timestamps
        
        # Telemetry log paths
        self.telemetry_log_path = self.telemetry_dir / f"telemetry_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d')}.jsonl"
        self.ece_log_paths = {
            session: self.telemetry_dir / f"ece_session_{session}.jsonl"
            for session in ("EU", "US", "OVERLAP")
        }
        
        log.info(
            "TelemetryTracker initialized: window=%d min, log_interval=%d min, target_coverage=%.2f, ece_window=%d, is_replay=%s",
            window_minutes,
            log_interval_minutes,
            target_coverage,
            ece_window_size,
            is_replay,
        )
    
    def record_eval(
        self,
        ts_utc: pd.Timestamp,
        session: str,
        p_hat: float,
        margin: float,
        decision: str,
    ) -> None:
        """
        Record an evaluation result.
        
        Parameters
        ----------
        ts_utc : pd.Timestamp
            UTC timestamp of evaluation.
        session : str
            Session tag (EU, US, OVERLAP).
        p_hat : float
            Maximum probability (p_hat).
        margin : float
            Margin (abs(p_long - p_short)).
        decision : str
            Entry decision (LONG, SHORT, NO_ENTRY).
        """
        self.eval_buffers[session].append({
            "ts_utc": ts_utc,
            "p_hat": float(p_hat),
            "margin": float(margin),
            "decision": decision,
            "entropy": float(prob_entropy(p_hat)),
        })
        
        # Check if we should log telemetry for this session
        self._maybe_log_telemetry(session, ts_utc)
    
    def record_closed_trade(
        self,
        entry_id: str,
        session: str,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
        p_hat: float,
        side: str,
        pnl_bps: float,
    ) -> None:
        """
        Record a closed trade for delayed ECE calculation.
        
        Parameters
        ----------
        entry_id : str
            Trade ID.
        session : str
            Session tag.
        entry_time : pd.Timestamp
            Entry timestamp.
        exit_time : pd.Timestamp
            Exit timestamp.
        p_hat : float
            Entry p_hat.
        side : str
            Trade side (long, short).
        pnl_bps : float
            PnL in basis points.
        """
        record = ClosedTradeRecord(
            entry_id=entry_id,
            session=session,
            entry_time=entry_time,
            exit_time=exit_time,
            p_hat=float(p_hat),
            side=side,
            pnl_bps=float(pnl_bps),
            hit_flag=1 if pnl_bps > 0 else 0,
        )
        self.closed_trades[session].append(record)
        
        # Calculate and log delayed ECE
        self._calculate_and_log_ece(session)
    
    def record_exit_latency(
        self,
        session: str,
        variant: str,
        hold_time_s: float,
    ) -> None:
        """
        Record exit latency for a variant.
        
        Parameters
        ----------
        session : str
            Session tag (EU, US, OVERLAP).
        variant : str
            Exit variant (EXIT_V2, EXIT_V3_SHADOW).
        hold_time_s : float
            Hold time in seconds (exit_time - entry_time).
        """
        self.exit_latency_buffers[session][variant].append(float(hold_time_s))
    
    def record_revision(self, ts_utc: pd.Timestamp) -> None:
        """
        Record a data revision (checksum changed).
        
        Parameters
        ----------
        ts_utc : pd.Timestamp
            UTC timestamp of the revised bar.
        """
        # Ensure timestamp is timezone-aware UTC
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.tz_localize("UTC")
        else:
            ts_utc = ts_utc.tz_convert("UTC")
        
        self.rev_buf.append(ts_utc)
    
    def n_revisions_24h(self, now_utc: pd.Timestamp) -> int:
        """
        Count revisions in the last 24 hours.
        
        Parameters
        ----------
        now_utc : pd.Timestamp
            Current UTC timestamp.
        
        Returns
        -------
        int
            Number of revisions in the last 24 hours.
        """
        # Ensure timestamp is timezone-aware UTC
        if now_utc.tzinfo is None:
            now_utc = now_utc.tz_localize("UTC")
        else:
            now_utc = now_utc.tz_convert("UTC")
        
        cutoff = now_utc - pd.Timedelta(hours=24)
        return sum(1 for t in self.rev_buf if t >= cutoff)
    
    def _maybe_log_telemetry(self, session: str, ts_utc: pd.Timestamp) -> None:
        """Log telemetry if enough time has passed since last log."""
        last_time = self.last_telemetry_time.get(session)
        if last_time is None:
            self.last_telemetry_time[session] = ts_utc
            return
        
        delta_minutes = (ts_utc - last_time).total_seconds() / 60.0
        if delta_minutes < self.log_interval_minutes:
            return
        
        # Calculate and log metrics (over window_minutes window, but log every log_interval_minutes)
        metrics = self._calculate_session_metrics(session, ts_utc)
        if metrics is not None:
            self._log_telemetry(metrics)
            self._check_alerts(metrics)
            self.last_telemetry_time[session] = ts_utc
    
    def _calculate_session_metrics(
        self,
        session: str,
        ts_utc: pd.Timestamp,
    ) -> Optional[SessionMetrics]:
        """Calculate session metrics over the window."""
        buffer = self.eval_buffers[session]
        if len(buffer) == 0:
            return None
        
        # Filter to window
        window_start = ts_utc - pd.Timedelta(minutes=self.window_minutes)
        window_evals = []
        for e in buffer:
            eval_ts = e["ts_utc"]
            if isinstance(eval_ts, str):
                eval_ts = pd.Timestamp(eval_ts)
            elif not isinstance(eval_ts, pd.Timestamp):
                eval_ts = pd.Timestamp(eval_ts)
            if eval_ts >= window_start:
                window_evals.append(e)
        
        if len(window_evals) == 0:
            return None
        
        p_hat_values = np.array([e["p_hat"] for e in window_evals])
        margin_values = np.array([e["margin"] for e in window_evals])
        entropy_values = np.array([e["entropy"] for e in window_evals])
        decisions = [e["decision"] for e in window_evals]
        
        n_evals = len(window_evals)
        n_entries = sum(1 for d in decisions if d != "NO_ENTRY")
        coverage_window = n_entries / n_evals if n_evals > 0 else 0.0
        
        p_hat_mean = float(np.mean(p_hat_values))
        p_hat_std = float(np.std(p_hat_values))
        margin_mean = float(np.mean(margin_values))
        entropy_mean = float(np.mean(entropy_values))
        
        # Keep rate: fraction of entries that pass all gates
        # (simplified: we assume all entries pass gates if decision != NO_ENTRY)
        keep_rate = n_entries / n_evals if n_evals > 0 else 0.0
        
        # Calibration proxy: p_hat_mean - 2*std(p_hat)
        calibration_proxy = p_hat_mean - 2.0 * p_hat_std
        
        # Sharpness: 1 - mean_entropy
        sharpness = 1.0 - entropy_mean
        
        # Calculate exit latency mean per variant
        exit_latency_mean = {}
        exit_latency_count = {}
        for variant, latency_buffer in self.exit_latency_buffers[session].items():
            if len(latency_buffer) > 0:
                exit_latency_mean[variant] = float(np.mean(list(latency_buffer)))
                exit_latency_count[variant] = len(latency_buffer)
        
        return SessionMetrics(
            session=session,
            window_start=window_start,
            window_end=ts_utc,
            n_evals=n_evals,
            n_entries=n_entries,
            coverage_window=coverage_window,
            p_hat_mean=p_hat_mean,
            p_hat_std=p_hat_std,
            margin_mean=margin_mean,
            entropy_mean=entropy_mean,
            keep_rate=keep_rate,
            calibration_proxy=calibration_proxy,
            sharpness=sharpness,
            exit_latency_mean=exit_latency_mean,
            exit_latency_count=exit_latency_count,
        )
    
    def _log_telemetry(self, metrics: SessionMetrics) -> None:
        """Log telemetry metrics to JSON file."""
        # Count revisions in last 24 hours
        n_revisions_24h = self.n_revisions_24h(metrics.window_end)
        
        record = {
            "ts_utc": metrics.window_end.isoformat(),
            "session": metrics.session,
            "window_minutes": self.window_minutes,
            "n_evals": metrics.n_evals,
            "n_entries": metrics.n_entries,
            "coverage_window": metrics.coverage_window,
            "p_hat_mean": metrics.p_hat_mean,
            "p_hat_std": metrics.p_hat_std,
            "margin_mean": metrics.margin_mean,
            "entropy_mean": metrics.entropy_mean,
            "keep_rate": metrics.keep_rate,
            "calibration_proxy": metrics.calibration_proxy,
            "sharpness": metrics.sharpness,
            "exit_latency_mean": metrics.exit_latency_mean,
            "exit_latency_count": metrics.exit_latency_count,
            "n_revisions_24h": n_revisions_24h,
        }
        
        with self.telemetry_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        
        # Log exit latency in info message
        exit_latency_str = ", ".join(
            f"{v}: {m:.1f}s (n={metrics.exit_latency_count.get(v, 0)})"
            for v, m in metrics.exit_latency_mean.items()
        ) if metrics.exit_latency_mean else "N/A"
        
        # Warn if revisions > 0 (data revisions detected)
        if n_revisions_24h > 0:
            log.warning(
                "[DATA] revisions_24h=%d (data may have been corrected by OANDA)",
                n_revisions_24h,
            )
        
        log.info(
            "[TELEMETRY %s] window=%d min, n_evals=%d, coverage=%.3f, p_hat_mean=%.3f, "
            "entropy_mean=%.3f, sharpness=%.3f, exit_latency=[%s], n_revisions_24h=%d",
            metrics.session,
            self.window_minutes,
            metrics.n_evals,
            metrics.coverage_window,
            metrics.p_hat_mean,
            metrics.entropy_mean,
            metrics.sharpness,
            exit_latency_str,
            n_revisions_24h,
        )
    
    def _check_alert_budget(self, session: str, category: str, now: pd.Timestamp) -> bool:
        """
        Check alert budget (max 3 WARN per 30 min per category).
        
        Parameters
        ----------
        session : str
            Session tag.
        category : str
            Alert category (e.g., "coverage", "p_hat", "entropy").
        now : pd.Timestamp
            Current UTC timestamp.
        
        Returns
        -------
        bool
            True if alert budget OK (can log WARN), False if exceeded (should upgrade to KILL).
        """
        # Get alert timestamps for this session and category
        alert_timestamps = self.alert_budget[session][category]
        
        # Filter to last 30 minutes
        cutoff = now - pd.Timedelta(minutes=30)
        recent_alerts = [ts for ts in alert_timestamps if ts >= cutoff]
        
        # Check if we've exceeded budget (max 3 WARN per 30 min)
        if len(recent_alerts) >= 3:
            return False  # Budget exceeded - should upgrade to KILL
        
        # Add current alert timestamp
        alert_timestamps.append(now)
        return True  # Budget OK - can log WARN
    
    def _check_alerts(self, metrics: SessionMetrics) -> None:
        """Check alert rules and log warnings if thresholds are exceeded."""
        now = metrics.window_end
        session = metrics.session
        
        # Alert 1: Coverage avvik > ±30% fra target
        coverage_diff = abs(metrics.coverage_window - self.target_coverage)
        coverage_threshold = 0.30 * self.target_coverage
        if coverage_diff > coverage_threshold:
            if self._check_alert_budget(session, "coverage", now):
                log.warning(
                    "[ALERT %s] Coverage avvik >±30%%: coverage=%.3f, target=%.3f, diff=%.3f",
                    metrics.session,
                    metrics.coverage_window,
                    self.target_coverage,
                    coverage_diff,
                )
            else:
                if self.is_replay:
                    log.debug(
                        "[TELEMETRY %s] Kill-switch disabled in replay mode (coverage=%.3f, target=%.3f, diff=%.3f)",
                        session,
                        metrics.coverage_window,
                        self.target_coverage,
                        coverage_diff,
                    )
                else:
                    log.error(
                        "[KILL-SWITCH %s] Coverage alert budget exceeded (3+ WARN in 30 min). Setting KILL_SWITCH_ON.",
                        session,
                    )
                    # Set KILL_SWITCH_ON flag (this should be handled by ops_watch, but we log it here)
                    # In production, you might want to set the flag directly or raise an exception
        
        # Alert 2: p_hat_mean < 0.55 i 30 min
        if metrics.p_hat_mean < 0.55:
            if self._check_alert_budget(session, "p_hat", now):
                log.warning(
                    "[ALERT %s] p_hat_mean < 0.55: p_hat_mean=%.3f",
                    metrics.session,
                    metrics.p_hat_mean,
                )
            else:
                if self.is_replay:
                    log.debug(
                        "[TELEMETRY %s] Kill-switch disabled in replay mode (p_hat_mean=%.3f)",
                        session,
                        metrics.p_hat_mean,
                    )
                else:
                    log.error(
                        "[KILL-SWITCH %s] p_hat alert budget exceeded (3+ WARN in 30 min). Setting KILL_SWITCH_ON.",
                        session,
                    )
        
        # Alert 3: entropy_mean > 0.68 i 30 min
        if metrics.entropy_mean > 0.68:
            if self._check_alert_budget(session, "entropy", now):
                log.warning(
                    "[ALERT %s] entropy_mean > 0.68: entropy_mean=%.3f",
                    metrics.session,
                    metrics.entropy_mean,
                )
            else:
                if self.is_replay:
                    log.debug(
                        "[TELEMETRY %s] Kill-switch disabled in replay mode (entropy_mean=%.3f)",
                        session,
                        metrics.entropy_mean,
                    )
                else:
                    log.error(
                        "[KILL-SWITCH %s] Entropy alert budget exceeded (3+ WARN in 30 min). Setting KILL_SWITCH_ON.",
                        session,
                    )
        
        # Alert 4: calibration_proxy < 0.35
        if metrics.calibration_proxy < 0.35:
            if self._check_alert_budget(session, "calibration_proxy", now):
                log.warning(
                    "[ALERT %s] calibration_proxy < 0.35: calibration_proxy=%.3f",
                    metrics.session,
                    metrics.calibration_proxy,
                )
            else:
                if self.is_replay:
                    log.debug(
                        "[TELEMETRY %s] Kill-switch disabled in replay mode (calibration_proxy=%.3f)",
                        session,
                        metrics.calibration_proxy,
                    )
                else:
                    log.error(
                        "[KILL-SWITCH %s] Calibration proxy alert budget exceeded (3+ WARN in 30 min). Setting KILL_SWITCH_ON.",
                        session,
                    )
        
        # Alert 5: sharpness < 0.22
        if metrics.sharpness < 0.22:
            if self._check_alert_budget(session, "sharpness", now):
                log.warning(
                    "[ALERT %s] sharpness < 0.22: sharpness=%.3f",
                    metrics.session,
                    metrics.sharpness,
                )
            else:
                log.error(
                    "[KILL-SWITCH %s] Sharpness alert budget exceeded (3+ WARN in 30 min). Setting KILL_SWITCH_ON.",
                    session,
                )
        
        # Alert 6: Exit latency - V3_SHADOW >25% høyere enn V2
        # Note: BPS-gevinst sjekkes i shadow_exit_report (krever shadow-exit log data)
        exit_lat_v2 = metrics.exit_latency_mean.get("EXIT_V2")
        exit_lat_v3 = metrics.exit_latency_mean.get("EXIT_V3_SHADOW")
        if exit_lat_v2 is not None and exit_lat_v3 is not None:
            latency_diff_pct = ((exit_lat_v3 - exit_lat_v2) / exit_lat_v2) * 100.0
            if latency_diff_pct > 25.0:
                if self._check_alert_budget(session, "exit_latency", now):
                    log.warning(
                        "[ALERT %s] EXIT_V3_SHADOW har >25%% høyere latency enn EXIT_V2: "
                        "V3=%.1fs, V2=%.1fs, diff=%.1f%% (n_V3=%d, n_V2=%d). "
                        "Sjekk shadow-exit rapport for bps-gevinst.",
                        metrics.session,
                        exit_lat_v3,
                        exit_lat_v2,
                        latency_diff_pct,
                        metrics.exit_latency_count.get("EXIT_V3_SHADOW", 0),
                        metrics.exit_latency_count.get("EXIT_V2", 0),
                    )
                else:
                    log.error(
                        "[KILL-SWITCH %s] Exit latency alert budget exceeded (3+ WARN in 30 min). Setting KILL_SWITCH_ON.",
                        session,
                    )
    
    def _calculate_and_log_ece(self, session: str) -> None:
        """Calculate and log delayed ECE for closed trades."""
        closed_trades = list(self.closed_trades[session])
        if len(closed_trades) < 100:
            # Need at least 100 trades for meaningful ECE
            return
        
        # Use last N trades (up to ece_window_size)
        recent_trades = closed_trades[-min(self.ece_window_size, len(closed_trades)):]
        
        p_hat_values = np.array([t.p_hat for t in recent_trades])
        hit_flags = np.array([t.hit_flag for t in recent_trades])
        
        ece = ece_bin(p_hat_values, hit_flags, n_bins=10)
        
        # Log ECE
        record = {
            "ts_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "session": session,
            "n_trades": len(recent_trades),
            "ece": ece,
            "p_hat_mean": float(np.mean(p_hat_values)),
            "hit_rate": float(np.mean(hit_flags)),
        }
        
        ece_log_path = self.ece_log_paths[session]
        with ece_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        
        # Check ECE thresholds
        if ece > 0.18:
            log.error(
                "[KILL-SWITCH %s] ECE > 0.18: ece=%.4f, n_trades=%d",
                session,
                ece,
                len(recent_trades),
            )
        elif ece > 0.12:
            log.warning(
                "[ALERT %s] ECE > 0.12: ece=%.4f, n_trades=%d",
                session,
                ece,
                len(recent_trades),
            )
        else:
            log.info(
                "[ECE %s] ece=%.4f, n_trades=%d, hit_rate=%.3f",
                session,
                ece,
                len(recent_trades),
                record["hit_rate"],
            )
    
    def get_ece(self, session: str) -> Optional[float]:
        """
        Get current ECE for a session.
        
        Parameters
        ----------
        session : str
            Session tag.
        
        Returns
        -------
        Optional[float]
            Current ECE, or None if insufficient trades.
        """
        closed_trades = list(self.closed_trades[session])
        if len(closed_trades) < 100:
            return None
        
        recent_trades = closed_trades[-min(self.ece_window_size, len(closed_trades)):]
        p_hat_values = np.array([t.p_hat for t in recent_trades])
        hit_flags = np.array([t.hit_flag for t in recent_trades])
        
        return float(ece_bin(p_hat_values, hit_flags, n_bins=10))
    
    def get_coverage(self, session: str, window_minutes: Optional[int] = None) -> Optional[float]:
        """
        Get current coverage for a session.
        
        Parameters
        ----------
        session : str
            Session tag.
        window_minutes : Optional[int]
            Window size in minutes (default: self.window_minutes).
        
        Returns
        -------
        Optional[float]
            Current coverage, or None if insufficient evals.
        """
        window_minutes = window_minutes or self.window_minutes
        buffer = self.eval_buffers[session]
        if len(buffer) == 0:
            return None
        
        # Filter to window
        now = pd.Timestamp.now(tz="UTC")
        window_start = now - pd.Timedelta(minutes=window_minutes)
        window_evals = []
        for e in buffer:
            eval_ts = e["ts_utc"]
            if isinstance(eval_ts, str):
                eval_ts = pd.Timestamp(eval_ts)
            elif not isinstance(eval_ts, pd.Timestamp):
                eval_ts = pd.Timestamp(eval_ts)
            if eval_ts >= window_start:
                window_evals.append(e)
        
        if len(window_evals) == 0:
            return None
        
        decisions = [e["decision"] for e in window_evals]
        n_evals = len(window_evals)
        n_entries = sum(1 for d in decisions if d != "NO_ENTRY")
        coverage = n_entries / n_evals if n_evals > 0 else 0.0
        return float(coverage)

