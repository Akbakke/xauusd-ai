"""
Exit Policy V2 for Trial 160

Data-driven exit policy based on MAE/MFE analysis.
Only modifies exit/risk/guards, does not change entry model or features.

Dependencies (explicit install line):
  (no external dependencies beyond stdlib and existing gx1 modules)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from gx1.execution.oanda_demo_runner import LiveTrade

log = logging.getLogger(__name__)


@dataclass
class ExitPolicyV2Config:
    """Configuration for Exit Policy V2."""
    
    enabled: bool = False
    mode: str = "replay"  # replay | live
    
    # Time stop
    time_stop_t1_bars: int = 50
    time_stop_require_mfe_atr_ge: float = 1.47
    
    # MAE kill
    mae_kill_mae_atr_ge: float = 7.30
    mae_kill_grace_bars: int = 3
    
    # Profit trail
    profit_trail_activate_mfe_atr_ge: float = 1.47
    profit_trail_giveback_frac: float = 0.45
    profit_trail_tighten_after_bars: Optional[int] = None
    
    # News guard (scaffold only, disabled by default)
    news_guard_enabled: bool = False
    news_guard_pre_minutes: int = 15
    news_guard_post_minutes: int = 30
    news_guard_event_filter: str = "HIGH_IMPACT_USD"
    
    # Logging
    log_exit_reason: bool = True
    counters_per_reason: bool = True


@dataclass
class ExitPolicyV2State:
    """Per-trade state for Exit Policy V2."""
    
    trade_id: str
    entry_atr_bps: float
    entry_time: Any  # pd.Timestamp
    
    # MFE/MAE tracking
    mfe_bps: float = 0.0
    mae_bps: float = 0.0
    age_bars: int = 0
    
    # Trail state
    trail_activated: bool = False
    trail_peak_bps: float = 0.0
    trail_level_bps: Optional[float] = None
    
    # Counters
    mae_kill_grace_bars_remaining: int = 0


@dataclass
class ExitPolicyV2Decision:
    """Exit decision from Exit Policy V2."""
    
    should_exit: bool
    exit_reason: Optional[str] = None
    exit_price: Optional[float] = None
    pnl_bps: Optional[float] = None
    bars_held: Optional[int] = None


class ExitPolicyV2:
    """
    Exit Policy V2: Data-driven exit policy based on MAE/MFE analysis.
    
    Rules:
    1. TIME_STOP: Close after T1 bars if MFE < Y1×ATR
    2. MAE_KILL: Kill if MAE >= X×ATR (after grace period)
    3. TRAIL_PROTECT: Trail stop when MFE >= Y2×ATR, giveback G×MFE
    4. NEWS_GUARD: Exit before/after high-impact news (disabled by default)
    """
    
    def __init__(self, config: ExitPolicyV2Config):
        """
        Initialize Exit Policy V2.
        
        Args:
            config: Exit Policy V2 configuration
        """
        self.config = config
        self.states: Dict[str, ExitPolicyV2State] = {}
        
        # Counters
        self.counters = {
            "exit_v2_n_time_stop": 0,
            "exit_v2_n_mae_kill": 0,
            "exit_v2_n_trail": 0,
            "exit_v2_n_news_exit": 0,
            "exit_v2_n_total_exits": 0,
        }
        
        if not config.enabled:
            log.info("[EXIT_POLICY_V2] Disabled in config")
    
    def initialize_trade(self, trade: "LiveTrade", entry_atr_bps: float) -> None:
        """
        Initialize state for a new trade.
        
        Args:
            trade: Trade object
            entry_atr_bps: ATR at entry (bps)
        """
        if not self.config.enabled:
            return
        
        state = ExitPolicyV2State(
            trade_id=trade.trade_id,
            entry_atr_bps=entry_atr_bps,
            entry_time=trade.entry_time,
            mae_kill_grace_bars_remaining=self.config.mae_kill_grace_bars,
        )
        self.states[trade.trade_id] = state
        log.debug(f"[EXIT_POLICY_V2] Initialized state for trade {trade.trade_id}")
    
    def update_trade_state(
        self,
        trade: "LiveTrade",
        current_bid: float,
        current_ask: float,
        current_atr_bps: float,
        bars_held: int,
    ) -> None:
        """
        Update per-trade state with current unrealized PnL.
        
        Args:
            trade: Trade object
            current_bid: Current bid price
            current_ask: Current ask price
            current_atr_bps: Current ATR (bps)
            bars_held: Number of bars held
        """
        if not self.config.enabled:
            return
        
        state = self.states.get(trade.trade_id)
        if not state:
            # Initialize if not already initialized
            self.initialize_trade(trade, current_atr_bps)
            state = self.states.get(trade.trade_id)
            if not state:
                return
        
        # Calculate unrealized PnL
        entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
        entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
        
        from gx1.utils.pnl import compute_pnl_bps
        unrealized_pnl_bps = compute_pnl_bps(
            entry_bid, entry_ask, current_bid, current_ask, trade.side
        )
        
        # Update MFE/MAE
        if unrealized_pnl_bps > state.mfe_bps:
            state.mfe_bps = unrealized_pnl_bps
        if unrealized_pnl_bps < state.mae_bps:
            state.mae_bps = unrealized_pnl_bps
        
        # Update age
        state.age_bars = bars_held
        
        # Update trail state
        if not state.trail_activated:
            mfe_atr = state.mfe_bps / state.entry_atr_bps if state.entry_atr_bps > 0 else 0.0
            if mfe_atr >= self.config.profit_trail_activate_mfe_atr_ge:
                state.trail_activated = True
                state.trail_peak_bps = state.mfe_bps
                state.trail_level_bps = state.mfe_bps * (1.0 - self.config.profit_trail_giveback_frac)
                log.debug(
                    f"[EXIT_POLICY_V2] Trail activated for trade {trade.trade_id}: "
                    f"MFE={state.mfe_bps:.2f} bps, trail_level={state.trail_level_bps:.2f} bps"
                )
        else:
            # Update trail peak
            if unrealized_pnl_bps > state.trail_peak_bps:
                state.trail_peak_bps = unrealized_pnl_bps
                state.trail_level_bps = state.trail_peak_bps * (1.0 - self.config.profit_trail_giveback_frac)
        
        # Update MAE kill grace period
        if state.mae_kill_grace_bars_remaining > 0:
            state.mae_kill_grace_bars_remaining -= 1
    
    def evaluate_exit(
        self,
        trade: "LiveTrade",
        current_bid: float,
        current_ask: float,
        current_atr_bps: float,
        bars_held: int,
        session: Optional[str] = None,
    ) -> ExitPolicyV2Decision:
        """
        Evaluate exit conditions for a trade.
        
        Args:
            trade: Trade object
            current_bid: Current bid price
            current_ask: Current ask price
            current_atr_bps: Current ATR (bps)
            bars_held: Number of bars held
            session: Trading session (EU/OVERLAP/US)
        
        Returns:
            ExitPolicyV2Decision
        """
        if not self.config.enabled:
            return ExitPolicyV2Decision(should_exit=False)
        
        state = self.states.get(trade.trade_id)
        if not state:
            return ExitPolicyV2Decision(should_exit=False)
        
        # Calculate unrealized PnL
        entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
        entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
        
        from gx1.utils.pnl import compute_pnl_bps
        unrealized_pnl_bps = compute_pnl_bps(
            entry_bid, entry_ask, current_bid, current_ask, trade.side
        )
        
        # Normalize with entry ATR
        mae_atr = abs(state.mae_bps) / state.entry_atr_bps if state.entry_atr_bps > 0 else 0.0
        mfe_atr = state.mfe_bps / state.entry_atr_bps if state.entry_atr_bps > 0 else 0.0
        
        # 1. TIME_STOP: Close after T1 bars if MFE < Y1×ATR
        if bars_held >= self.config.time_stop_t1_bars:
            if mfe_atr < self.config.time_stop_require_mfe_atr_ge:
                self.counters["exit_v2_n_time_stop"] += 1
                self.counters["exit_v2_n_total_exits"] += 1
                exit_price = current_bid if trade.side == "long" else current_ask
                return ExitPolicyV2Decision(
                    should_exit=True,
                    exit_reason="TIME_STOP",
                    exit_price=exit_price,
                    pnl_bps=unrealized_pnl_bps,
                    bars_held=bars_held,
                )
        
        # 2. MAE_KILL: Kill if MAE >= X×ATR (after grace period)
        if state.mae_kill_grace_bars_remaining == 0:
            if mae_atr >= self.config.mae_kill_mae_atr_ge:
                self.counters["exit_v2_n_mae_kill"] += 1
                self.counters["exit_v2_n_total_exits"] += 1
                exit_price = current_bid if trade.side == "long" else current_ask
                return ExitPolicyV2Decision(
                    should_exit=True,
                    exit_reason="MAE_KILL",
                    exit_price=exit_price,
                    pnl_bps=unrealized_pnl_bps,
                    bars_held=bars_held,
                )
        
        # 3. TRAIL_PROTECT: Trail stop when MFE >= Y2×ATR, giveback G×MFE
        if state.trail_activated and state.trail_level_bps is not None:
            if unrealized_pnl_bps <= state.trail_level_bps:
                self.counters["exit_v2_n_trail"] += 1
                self.counters["exit_v2_n_total_exits"] += 1
                exit_price = current_bid if trade.side == "long" else current_ask
                return ExitPolicyV2Decision(
                    should_exit=True,
                    exit_reason="TRAIL_PROTECT",
                    exit_price=exit_price,
                    pnl_bps=unrealized_pnl_bps,
                    bars_held=bars_held,
                )
        
        # 4. NEWS_GUARD: Exit before/after high-impact news (disabled by default)
        if self.config.news_guard_enabled:
            # Scaffold only - not implemented yet
            pass
        
        return ExitPolicyV2Decision(should_exit=False)
    
    def cleanup_trade(self, trade_id: str) -> None:
        """
        Clean up state for a closed trade.
        
        Args:
            trade_id: Trade ID
        """
        if trade_id in self.states:
            del self.states[trade_id]
    
    def get_counters(self) -> Dict[str, int]:
        """Get exit counters."""
        return self.counters.copy()


def load_exit_policy_v2_config(config_path: Optional[Path] = None, config_dict: Optional[Dict[str, Any]] = None) -> ExitPolicyV2Config:
    """
    Load Exit Policy V2 configuration from YAML or dict.
    
    Args:
        config_path: Path to YAML config file
        config_dict: Config dict (if loading from policy YAML)
    
    Returns:
        ExitPolicyV2Config
    """
    import yaml
    
    if config_dict:
        exit_v2_config = config_dict.get("exit_policy_v2", {})
    elif config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        exit_v2_config = config.get("exit_policy_v2", {})
    else:
        raise ValueError("Either config_path or config_dict must be provided")
    
    if not exit_v2_config.get("enabled", False):
        return ExitPolicyV2Config(enabled=False)
    
    time_stop = exit_v2_config.get("time_stop", {})
    mae_kill = exit_v2_config.get("mae_kill", {})
    profit_trail = exit_v2_config.get("profit_trail", {})
    news_guard = exit_v2_config.get("news_guard", {})
    logging_config = exit_v2_config.get("logging", {})
    
    return ExitPolicyV2Config(
        enabled=exit_v2_config.get("enabled", False),
        mode=exit_v2_config.get("mode", "replay"),
        time_stop_t1_bars=int(time_stop.get("t1_bars", 50)),
        time_stop_require_mfe_atr_ge=float(time_stop.get("require_mfe_atr_ge", 1.47)),
        mae_kill_mae_atr_ge=float(mae_kill.get("mae_atr_ge", 7.30)),
        mae_kill_grace_bars=int(mae_kill.get("grace_bars", 3)),
        profit_trail_activate_mfe_atr_ge=float(profit_trail.get("activate_mfe_atr_ge", 1.47)),
        profit_trail_giveback_frac=float(profit_trail.get("giveback_frac", 0.45)),
        profit_trail_tighten_after_bars=profit_trail.get("tighten_after_bars"),
        news_guard_enabled=news_guard.get("enabled", False),
        news_guard_pre_minutes=int(news_guard.get("pre_minutes", 15)),
        news_guard_post_minutes=int(news_guard.get("post_minutes", 30)),
        news_guard_event_filter=news_guard.get("event_filter", "HIGH_IMPACT_USD"),
        log_exit_reason=logging_config.get("log_exit_reason", True),
        counters_per_reason=logging_config.get("counters_per_reason", True),
    )
