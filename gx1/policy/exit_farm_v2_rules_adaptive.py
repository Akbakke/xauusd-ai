#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXIT_FARM_V2_RULES_ADAPTIVE (RULE6A)

Adaptive long-only exit controller that mirrors the RULE5 plumbing but scales
its thresholds using the current ATR(5) context and live MFE/MAE observations.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from gx1.policy.exit_farm_v2_rules import ExitDecision
from gx1.utils.pnl import compute_pnl_bps

logger = logging.getLogger(__name__)


class ExitFarmV2RulesAdaptive:
    """ATR-aware exit engine (RULE6A)."""

    def __init__(
        self,
        *,
        enable_tp: bool = True,
        enable_trailing: bool = True,
        enable_be: bool = True,
        atr_floor_bps: float = 4.0,
        tp2_cap_bps: float = 12.0,
        trailing_floor_bps: float = 2.0,
        trailing_ratio: float = 0.30,
        be_activation_bps: float = 3.0,
        be_level_factor: float = -0.15,
        mfe_trailing_activation_bps: float = 6.0,
        verbose_logging: bool = False,
        log_every_n_bars: int = 5,
    ) -> None:
        self.enable_tp = enable_tp
        self.enable_trailing = enable_trailing
        self.enable_be = enable_be
        self.atr_floor_bps = float(max(0.1, atr_floor_bps))
        self.tp2_cap_bps = float(max(tp2_cap_bps, 1.0))
        self.trailing_floor_bps = float(max(0.5, trailing_floor_bps))
        self.trailing_ratio = float(max(0.0, trailing_ratio))
        self.be_activation_bps = float(be_activation_bps)
        self.be_level_factor = float(be_level_factor)
        self.mfe_trailing_activation_bps = float(mfe_trailing_activation_bps)
        self.verbose_logging = verbose_logging
        self.log_every_n_bars = max(1, int(log_every_n_bars))

        # Runtime state (populated on reset)
        self.entry_bid: Optional[float] = None
        self.entry_ask: Optional[float] = None
        self.entry_ts = None
        self.trade_id: Optional[str] = None
        self.entry_atr_bps: Optional[float] = None

        self.side: str = "long"
        self.bars_held: int = 0
        self.mae_bps: float = 0.0
        self.mfe_bps: float = 0.0
        self.trailing_active: bool = False
        self.trailing_high: float = 0.0
        self.be_active: bool = False
        self.be_level_bps: float = 0.0
        self._last_logged_bar = 0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _resolve_atr_bps(self, atr_bps: Optional[float]) -> float:
        candidates: Sequence[float] = [
            v for v in (atr_bps, self.entry_atr_bps, self.atr_floor_bps) if v is not None
        ]
        atr_value = max(self.atr_floor_bps, max(candidates) if candidates else self.atr_floor_bps)
        return float(atr_value)

    def _compute_tp_bands(self, atr_bps: float) -> tuple[float, float]:
        tp1 = max(4.0, 0.35 * atr_bps)
        tp2 = max(8.0, min(self.tp2_cap_bps, 0.80 * atr_bps))
        return tp1, tp2

    def _compute_trailing_stop(self) -> float:
        if self.mfe_bps >= self.mfe_trailing_activation_bps:
            return max(self.trailing_floor_bps, self.trailing_ratio * self.mfe_bps)
        return self.trailing_floor_bps

    def _maybe_log_state(self, ts, pnl_bps: float, atr_bps: float) -> None:
        if not self.verbose_logging:
            return
        if self.bars_held - self._last_logged_bar < self.log_every_n_bars:
            return
        self._last_logged_bar = self.bars_held
        logger.info(
            "[RULE6A] %s bars=%d pnl=%.2f mfe=%.2f mae=%.2f atr=%.2f be_active=%s trailing=%s",
            self.trade_id or "?",
            self.bars_held,
            pnl_bps,
            self.mfe_bps,
            self.mae_bps,
            atr_bps,
            self.be_active,
            self.trailing_active,
        )

    def _exit_decision(self, reason: str, price_bid: float, price_ask: float, pnl_bps: float) -> ExitDecision:
        exit_price = float(price_bid if self.side == "long" else price_ask)
        decision = ExitDecision(
            exit_price=exit_price,
            reason=reason,
            bars_held=self.bars_held,
            pnl_bps=pnl_bps,
            mae_bps=self.mae_bps,
            mfe_bps=self.mfe_bps,
        )
        return decision

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset_on_entry(
        self,
        entry_bid: float,
        entry_ask: float,
        entry_ts,
        side: str = "long",
        trade_id: Optional[str] = None,
        atr_bps: Optional[float] = None,
    ) -> None:
        if side != "long":
            raise ValueError("ExitFarmV2RulesAdaptive currently supports LONG positions only")
        self.entry_bid = entry_bid
        self.entry_ask = entry_ask
        self.entry_ts = entry_ts
        self.trade_id = trade_id
        self.side = side
        self.entry_atr_bps = atr_bps

        self.bars_held = 0
        self.mae_bps = 0.0
        self.mfe_bps = 0.0
        self.trailing_active = False
        self.trailing_high = 0.0
        self.be_active = False
        self.be_level_bps = 0.0
        self._last_logged_bar = 0

        logger.debug(
            "[RULE6A] reset trade_id=%s entry_bid=%.5f entry_ask=%.5f atr=%.2f",
            trade_id,
            entry_bid,
            entry_ask,
            atr_bps if atr_bps is not None else -1.0,
        )

    def on_bar(self, price_bid: float, price_ask: float, ts, atr_bps: Optional[float] = None) -> Optional[ExitDecision]:
        if self.entry_bid is None or self.entry_ask is None:
            raise RuntimeError("reset_on_entry() must be called before on_bar()")

        self.bars_held += 1
        pnl_bps = compute_pnl_bps(self.entry_bid, self.entry_ask, price_bid, price_ask, self.side)
        if pnl_bps < self.mae_bps:
            self.mae_bps = pnl_bps
        if pnl_bps > self.mfe_bps:
            self.mfe_bps = pnl_bps

        atr_context = self._resolve_atr_bps(atr_bps)
        tp1_bps, tp2_bps = self._compute_tp_bands(atr_context)
        trailing_stop_bps = self._compute_trailing_stop()

        self._maybe_log_state(ts, pnl_bps, atr_context)

        # Break-even activation
        if self.enable_be and not self.be_active and pnl_bps >= self.be_activation_bps:
            self.be_active = True
            self.be_level_bps = float(self.be_level_factor * atr_context)
            logger.debug(
                "[RULE6A] BE activated trade_id=%s level=%.2f atr=%.2f",
                self.trade_id,
                self.be_level_bps,
                atr_context,
            )

        if self.enable_be and self.be_active and pnl_bps <= self.be_level_bps:
            logger.debug("[RULE6A] BE level hit trade_id=%s pnl=%.2f level=%.2f", self.trade_id, pnl_bps, self.be_level_bps)
            return self._exit_decision("RULE6A_BE", price_bid, price_ask, pnl_bps)

        # Trailing activation after TP1 (unless TP2 fires first)
        if (
            self.enable_trailing
            and not self.trailing_active
            and self.mfe_bps >= tp1_bps
        ):
            self.trailing_active = True
            self.trailing_high = self.mfe_bps
            logger.debug(
                "[RULE6A] Trailing activated trade_id=%s mfe=%.2f tp1=%.2f",
                self.trade_id,
                self.mfe_bps,
                tp1_bps,
            )

        if self.trailing_active:
            if pnl_bps > self.trailing_high:
                self.trailing_high = pnl_bps
            trailing_threshold = self.trailing_high - trailing_stop_bps
            if pnl_bps <= trailing_threshold:
                logger.debug(
                    "[RULE6A] Trailing stop hit trade_id=%s pnl=%.2f high=%.2f stop=%.2f",
                    self.trade_id,
                    pnl_bps,
                    self.trailing_high,
                    trailing_threshold,
                )
                return self._exit_decision("RULE6A_TRAILING", price_bid, price_ask, pnl_bps)

        if self.enable_tp:
            if pnl_bps >= tp2_bps:
                logger.debug(
                    "[RULE6A] TP2 reached trade_id=%s pnl=%.2f tp2=%.2f",
                    self.trade_id,
                    pnl_bps,
                    tp2_bps,
                )
                return self._exit_decision("RULE6A_TP2", price_bid, price_ask, pnl_bps)
            if not self.trailing_active and pnl_bps >= tp1_bps:
                logger.debug(
                    "[RULE6A] TP1 reached trade_id=%s pnl=%.2f tp1=%.2f",
                    self.trade_id,
                    pnl_bps,
                    tp1_bps,
                )
                return self._exit_decision("RULE6A_TP1", price_bid, price_ask, pnl_bps)

        return None


def get_exit_policy_farm_v2_rules_adaptive(
    *,
    enable_tp: bool = True,
    enable_trailing: bool = True,
    enable_be: bool = True,
    verbose_logging: bool = False,
    log_every_n_bars: int = 5,
) -> ExitFarmV2RulesAdaptive:
    """
    Factory helper (mirrors RULE5 helper) used by GX1DemoRunner.
    """
    return ExitFarmV2RulesAdaptive(
        enable_tp=enable_tp,
        enable_trailing=enable_trailing,
        enable_be=enable_be,
        verbose_logging=verbose_logging,
        log_every_n_bars=log_every_n_bars,
    )
