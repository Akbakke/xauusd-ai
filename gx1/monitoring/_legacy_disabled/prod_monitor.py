#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production monitoring module.

Logs rolling metrics:
- trades/day
- EV/trade
- RULE6A_rate
- blocked_RULE6A_rate
- max_loss_streak (or cluster proxy)
- range_edge_dist_atr distribution per exit_profile
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProdMetrics:
    """Production metrics snapshot."""
    timestamp: str
    trades_day: float
    ev_trade: float
    rule6a_rate: float
    blocked_rule6a_rate: float
    max_loss_streak: int
    range_edge_dist_atr_mean_rule5: float
    range_edge_dist_atr_mean_rule6a: float
    n_trades: int
    n_rule5: int
    n_rule6a: int
    n_blocked: int


class ProdMonitor:
    """
    Production monitor that tracks rolling metrics.
    
    Maintains a rolling window of trades and computes metrics.
    """
    
    def __init__(
        self,
        window_days: int = 7,
        output_dir: Path = Path("gx1/wf_runs/prod_monitor"),
    ):
        """
        Initialize production monitor.
        
        Args:
            window_days: Rolling window size in days (default: 7)
            output_dir: Directory for CSV output
        """
        self.window_days = window_days
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Rolling window of trades
        self.trades_window: deque = deque(maxlen=10000)  # Max 10k trades in memory
        
        # Metrics history
        self.metrics_history: List[ProdMetrics] = []
        
        logger.info("[PROD_MONITOR] Initialized with window_days=%d, output_dir=%s", window_days, output_dir)
    
    def add_trade(
        self,
        trade_id: str,
        entry_time: datetime,
        exit_time: Optional[datetime],
        pnl_bps: Optional[float],
        exit_profile: str,
        range_edge_dist_atr: Optional[float] = None,
        was_blocked: bool = False,
    ) -> None:
        """
        Add a trade to the monitoring window.
        
        Args:
            trade_id: Trade identifier
            entry_time: Entry timestamp
            exit_time: Exit timestamp (None if still open)
            pnl_bps: PnL in basis points (None if still open)
            exit_profile: Exit profile (RULE5 or RULE6A)
            range_edge_dist_atr: Range edge distance ATR (if available)
            was_blocked: Whether this trade was blocked by guardrail
        """
        trade = {
            "trade_id": trade_id,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "pnl_bps": pnl_bps,
            "exit_profile": exit_profile,
            "range_edge_dist_atr": range_edge_dist_atr,
            "was_blocked": was_blocked,
        }
        self.trades_window.append(trade)
    
    def compute_metrics(self, now: Optional[datetime] = None) -> ProdMetrics:
        """
        Compute current metrics from rolling window.
        
        Args:
            now: Current timestamp (default: now)
            
        Returns:
            ProdMetrics snapshot
        """
        if now is None:
            now = datetime.now()
        
        # Filter to window
        window_start = now - timedelta(days=self.window_days)
        trades_in_window = [
            t for t in self.trades_window
            if t["entry_time"] >= window_start and t["exit_time"] is not None and t["pnl_bps"] is not None
        ]
        
        if len(trades_in_window) == 0:
            return ProdMetrics(
                timestamp=now.isoformat(),
                trades_day=0.0,
                ev_trade=0.0,
                rule6a_rate=0.0,
                blocked_rule6a_rate=0.0,
                max_loss_streak=0,
                range_edge_dist_atr_mean_rule5=0.0,
                range_edge_dist_atr_mean_rule6a=0.0,
                n_trades=0,
                n_rule5=0,
                n_rule6a=0,
                n_blocked=0,
            )
        
        df = pd.DataFrame(trades_in_window)
        
        # Basic metrics
        n_trades = len(df)
        period_days = (df["exit_time"].max() - df["entry_time"].min()).total_seconds() / (24 * 3600)
        period_days = max(period_days, 1.0)  # Avoid division by zero
        trades_day = n_trades / period_days
        
        # EV/trade
        ev_trade = float(df["pnl_bps"].mean()) if len(df) > 0 else 0.0
        
        # RULE6A rate
        n_rule6a = (df["exit_profile"] == "RULE6A").sum()
        rule6a_rate = n_rule6a / n_trades if n_trades > 0 else 0.0
        
        # Blocked RULE6A rate
        n_blocked = df["was_blocked"].sum()
        blocked_rule6a_rate = n_blocked / n_trades if n_trades > 0 else 0.0
        
        # Max loss streak
        pnl_series = df["pnl_bps"].values
        max_loss_streak = self._compute_max_loss_streak(pnl_series)
        
        # Range edge dist ATR distribution
        rule5_trades = df[df["exit_profile"] == "RULE5"]
        rule6a_trades = df[df["exit_profile"] == "RULE6A"]
        
        range_edge_dist_atr_mean_rule5 = float(rule5_trades["range_edge_dist_atr"].mean()) if len(rule5_trades) > 0 and rule5_trades["range_edge_dist_atr"].notna().any() else 0.0
        range_edge_dist_atr_mean_rule6a = float(rule6a_trades["range_edge_dist_atr"].mean()) if len(rule6a_trades) > 0 and rule6a_trades["range_edge_dist_atr"].notna().any() else 0.0
        
        metrics = ProdMetrics(
            timestamp=now.isoformat(),
            trades_day=trades_day,
            ev_trade=ev_trade,
            rule6a_rate=rule6a_rate,
            blocked_rule6a_rate=blocked_rule6a_rate,
            max_loss_streak=max_loss_streak,
            range_edge_dist_atr_mean_rule5=range_edge_dist_atr_mean_rule5,
            range_edge_dist_atr_mean_rule6a=range_edge_dist_atr_mean_rule6a,
            n_trades=n_trades,
            n_rule5=n_trades - n_rule6a,
            n_rule6a=n_rule6a,
            n_blocked=n_blocked,
        )
        
        return metrics
    
    def _compute_max_loss_streak(self, pnl_series: np.ndarray) -> int:
        """
        Compute maximum consecutive loss streak.
        
        Args:
            pnl_series: Array of PnL values
            
        Returns:
            Maximum number of consecutive losses
        """
        if len(pnl_series) == 0:
            return 0
        
        losses = pnl_series < 0
        max_streak = 0
        current_streak = 0
        
        for is_loss in losses:
            if is_loss:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def check_thresholds(self, metrics: ProdMetrics, thresholds_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Check metrics against thresholds and return alerts.
        
        Args:
            metrics: Metrics to check
            thresholds_path: Path to thresholds YAML (default: gx1/monitoring/thresholds.yaml)
            
        Returns:
            List of alert dicts
        """
        import yaml
        
        if thresholds_path is None:
            thresholds_path = Path(__file__).parent / "thresholds.yaml"
        
        if not thresholds_path.exists():
            logger.warning("[PROD_MONITOR] Thresholds file not found: %s", thresholds_path)
            return []
        
        with open(thresholds_path) as f:
            thresholds_config = yaml.safe_load(f)
        
        thresholds = thresholds_config.get("thresholds", {})
        alerts = []
        
        # Check trades/day
        if "trades_per_day" in thresholds:
            tpd_min = thresholds["trades_per_day"].get("min")
            tpd_max = thresholds["trades_per_day"].get("max")
            if tpd_min is not None and metrics.trades_day < tpd_min:
                alerts.append({
                    "metric": "trades_per_day",
                    "value": metrics.trades_day,
                    "threshold": tpd_min,
                    "severity": "WARNING",
                    "message": f"Trades/day ({metrics.trades_day:.2f}) below minimum ({tpd_min:.2f})",
                })
            if tpd_max is not None and metrics.trades_day > tpd_max:
                alerts.append({
                    "metric": "trades_per_day",
                    "value": metrics.trades_day,
                    "threshold": tpd_max,
                    "severity": "WARNING",
                    "message": f"Trades/day ({metrics.trades_day:.2f}) above maximum ({tpd_max:.2f})",
                })
        
        # Check EV/trade
        if "ev_per_trade" in thresholds:
            ev_min = thresholds["ev_per_trade"].get("min")
            ev_max = thresholds["ev_per_trade"].get("max")
            if ev_min is not None and metrics.ev_trade < ev_min:
                alerts.append({
                    "metric": "ev_per_trade",
                    "value": metrics.ev_trade,
                    "threshold": ev_min,
                    "severity": "ERROR",
                    "message": f"EV/trade ({metrics.ev_trade:.2f} bps) below minimum ({ev_min:.2f} bps)",
                })
            if ev_max is not None and metrics.ev_trade > ev_max:
                alerts.append({
                    "metric": "ev_per_trade",
                    "value": metrics.ev_trade,
                    "threshold": ev_max,
                    "severity": "WARNING",
                    "message": f"EV/trade ({metrics.ev_trade:.2f} bps) above maximum ({ev_max:.2f} bps)",
                })
        
        # Check RULE6A rate
        if "rule6a_rate" in thresholds:
            r6a_min = thresholds["rule6a_rate"].get("min")
            r6a_max = thresholds["rule6a_rate"].get("max")
            if r6a_min is not None and metrics.rule6a_rate < r6a_min:
                alerts.append({
                    "metric": "rule6a_rate",
                    "value": metrics.rule6a_rate,
                    "threshold": r6a_min,
                    "severity": "WARNING",
                    "message": f"RULE6A rate ({metrics.rule6a_rate*100:.1f}%) below minimum ({r6a_min*100:.1f}%)",
                })
            if r6a_max is not None and metrics.rule6a_rate > r6a_max:
                alerts.append({
                    "metric": "rule6a_rate",
                    "value": metrics.rule6a_rate,
                    "threshold": r6a_max,
                    "severity": "WARNING",
                    "message": f"RULE6A rate ({metrics.rule6a_rate*100:.1f}%) above maximum ({r6a_max*100:.1f}%)",
                })
        
        # Check blocked RULE6A rate
        if "blocked_rule6a_rate" in thresholds:
            blocked_max = thresholds["blocked_rule6a_rate"].get("max")
            if blocked_max is not None and metrics.blocked_rule6a_rate > blocked_max:
                alerts.append({
                    "metric": "blocked_rule6a_rate",
                    "value": metrics.blocked_rule6a_rate,
                    "threshold": blocked_max,
                    "severity": "WARNING",
                    "message": f"Blocked RULE6A rate ({metrics.blocked_rule6a_rate*100:.1f}%) above maximum ({blocked_max*100:.1f}%) - guardrail may be too aggressive",
                })
        
        # Check max loss streak
        if "max_loss_streak" in thresholds:
            streak_max = thresholds["max_loss_streak"].get("max")
            if streak_max is not None and metrics.max_loss_streak > streak_max:
                alerts.append({
                    "metric": "max_loss_streak",
                    "value": metrics.max_loss_streak,
                    "threshold": streak_max,
                    "severity": "ERROR",
                    "message": f"Max loss streak ({metrics.max_loss_streak}) above maximum ({streak_max})",
                })
        
        # Check range edge dist ATR difference
        if "range_edge_dist_atr_diff" in thresholds:
            diff = metrics.range_edge_dist_atr_mean_rule6a - metrics.range_edge_dist_atr_mean_rule5
            diff_min = thresholds["range_edge_dist_atr_diff"].get("min")
            diff_max = thresholds["range_edge_dist_atr_diff"].get("max")
            if diff_min is not None and diff < diff_min:
                alerts.append({
                    "metric": "range_edge_dist_atr_diff",
                    "value": diff,
                    "threshold": diff_min,
                    "severity": "WARNING",
                    "message": f"Range edge dist ATR difference ({diff:.3f}) below minimum ({diff_min:.3f}) - RULE6A may not be edge-focused",
                })
            if diff_max is not None and diff > diff_max:
                alerts.append({
                    "metric": "range_edge_dist_atr_diff",
                    "value": diff,
                    "threshold": diff_max,
                    "severity": "WARNING",
                    "message": f"Range edge dist ATR difference ({diff:.3f}) above maximum ({diff_max:.3f})",
                })
        
        return alerts
    
    def log_metrics(self, metrics: Optional[ProdMetrics] = None, check_thresholds: bool = True) -> None:
        """
        Log metrics to CSV and terminal, and check thresholds.
        
        Args:
            metrics: Metrics to log (if None, computes from current window)
            check_thresholds: Whether to check thresholds and generate alerts.json
        """
        if metrics is None:
            metrics = self.compute_metrics()
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Save to CSV
        csv_path = self.output_dir / "prod_metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame()
        
        new_row = pd.DataFrame([asdict(metrics)])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_path, index=False)
        
        # Check thresholds and generate alerts
        alerts = []
        if check_thresholds:
            alerts = self.check_thresholds(metrics)
            if alerts:
                alerts_path = self.output_dir / "alerts.json"
                alerts_data = {
                    "timestamp": metrics.timestamp,
                    "alerts": alerts,
                }
                with open(alerts_path, "w") as f:
                    json.dump(alerts_data, f, indent=2)
                logger.warning("[PROD_MONITOR] %d alerts triggered - saved to %s", len(alerts), alerts_path)
                for alert in alerts:
                    logger.warning("[ALERT] %s: %s", alert["severity"], alert["message"])
        
        # Terminal summary
        logger.info("=" * 80)
        logger.info("[PROD_MONITOR] Metrics Summary (last %d days)", self.window_days)
        logger.info("=" * 80)
        logger.info("Trades/day: %.2f", metrics.trades_day)
        logger.info("EV/trade: %.2f bps", metrics.ev_trade)
        logger.info("RULE6A rate: %.1f%% (%d/%d)", metrics.rule6a_rate * 100, metrics.n_rule6a, metrics.n_trades)
        logger.info("Blocked RULE6A rate: %.1f%% (%d/%d)", metrics.blocked_rule6a_rate * 100, metrics.n_blocked, metrics.n_trades)
        logger.info("Max loss streak: %d", metrics.max_loss_streak)
        logger.info("Range edge dist ATR - RULE5: %.3f", metrics.range_edge_dist_atr_mean_rule5)
        logger.info("Range edge dist ATR - RULE6A: %.3f", metrics.range_edge_dist_atr_mean_rule6a)
        if alerts:
            logger.info("Alerts: %d triggered", len(alerts))
        logger.info("=" * 80)
        logger.info("Metrics saved to: %s", csv_path)
    
    def load_from_trade_log(self, trade_log_path: Path) -> None:
        """
        Load trades from trade log CSV.
        
        Args:
            trade_log_path: Path to trade log CSV
        """
        if not trade_log_path.exists():
            logger.warning("[PROD_MONITOR] Trade log not found: %s", trade_log_path)
            return
        
        df = pd.read_csv(trade_log_path)
        
        # Parse extra column for range_edge_dist_atr and was_blocked
        import json
        
        for _, row in df.iterrows():
            extra_str = row.get("extra", "{}")
            try:
                extra = json.loads(extra_str) if isinstance(extra_str, str) else extra_str
            except:
                extra = {}
            
            range_edge_dist_atr = extra.get("range_edge_dist_atr")
            exit_profile = row.get("exit_profile", "RULE5")
            
            # Infer was_blocked: if exit_profile is RULE5 but router might have chosen RULE6A
            # This is approximate - full verification requires baseline comparison
            was_blocked = False  # Default: not blocked
            
            self.add_trade(
                trade_id=str(row.get("trade_id", "")),
                entry_time=pd.to_datetime(row.get("entry_time")),
                exit_time=pd.to_datetime(row.get("exit_time")) if pd.notna(row.get("exit_time")) else None,
                pnl_bps=float(row.get("pnl_bps")) if pd.notna(row.get("pnl_bps")) else None,
                exit_profile=exit_profile,
                range_edge_dist_atr=float(range_edge_dist_atr) if range_edge_dist_atr is not None else None,
                was_blocked=was_blocked,
            )
        
        logger.info("[PROD_MONITOR] Loaded %d trades from %s", len(df), trade_log_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production monitor")
    parser.add_argument("--trade-log", type=Path, help="Path to trade log CSV")
    parser.add_argument("--window-days", type=int, default=7, help="Rolling window size in days")
    parser.add_argument("--output-dir", type=Path, default=Path("gx1/wf_runs/prod_monitor"), help="Output directory")
    args = parser.parse_args()
    
    monitor = ProdMonitor(window_days=args.window_days, output_dir=args.output_dir)
    
    if args.trade_log:
        monitor.load_from_trade_log(args.trade_log)
        monitor.log_metrics()
    else:
        logger.info("[PROD_MONITOR] No trade log provided - monitor ready for runtime integration")

