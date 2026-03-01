"""
ARCHIVED: forbidden exit modes removed from runtime; canonical truth uses exit_transformer_v0 only.

This file preserves historical branches for EXIT_POLICY_V2/V3, FARM exits, hybrid exit routers,
and related diagnostics that were stripped from `gx1/execution/oanda_demo_runner.py`.
"""

from __future__ import annotations

import json as jsonlib
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

log = logging.getLogger(__name__)


def archived_exit_mode_setup(self, exit_cfg: Dict[str, Any]) -> None:
    """
    Historical exit-mode initialization supporting EXIT_V2_DRIFT, EXIT_V3_ADAPTIVE, and FARM exits.
    """
    self.exit_only_v2_drift = bool(exit_cfg.get("exit", {}).get("only_v2_drift", False))
    if self.exit_only_v2_drift:
        raise RuntimeError(
            "[FORBIDDEN] exit.only_v2_drift is not allowed in CANONICAL_TRUTH_SIGNAL_ONLY_V1; use exit_transformer_v0 only."
        )
    exit_type = exit_cfg.get("exit", {}).get("type") if isinstance(exit_cfg.get("exit"), dict) else None
    self.exit_farm_v1_policy = None
    self.exit_farm_v2_rules_policy = None
    self.exit_farm_v2_rules_factory = None
    self.exit_farm_v2_rules_states: Dict[str, Any] = {}
    self.exit_verbose_logging = False
    self.exit_fixed_bar_policy = None
    self.exit_ml_mode = (
        exit_cfg.get("exit", {}).get("params", {}).get("exit_ml", {}).get("mode")
        if isinstance(exit_cfg.get("exit"), dict)
        else None
    )
    if (exit_type and "FARM" in str(exit_type).upper()) or (
        self.exit_ml_mode and "FARM" in str(self.exit_ml_mode).upper()
    ):
        raise RuntimeError("[FORBIDDEN] FARM exits are disabled in CANONICAL_TRUTH_SIGNAL_ONLY_V1")


def archived_exit_control_v3_patch(self) -> None:
    """Historical EXIT_V3_ADAPTIVE ExitArbiter override."""
    if hasattr(self, "exit_v3_drift_adaptive_policy") and self.exit_v3_drift_adaptive_policy is not None and self.exit_only_v2_drift:
        self.exit_control.allowed_loss_closers = [
            "EXIT_V3_ADAPTIVE_SL",
            "EXIT_V3_ADAPTIVE_SL_BREAKEVEN",
            "EXIT_V3_ADAPTIVE_TP2",
            "EXIT_V3_ADAPTIVE_TIMEOUT",
        ]
        self.exit_control.allow_model_exit_when["min_bars"] = 1
        self.exit_control.allow_model_exit_when["min_pnl_bps"] = -100
        self.exit_control.allow_model_exit_when["min_exit_prob"] = 0.0
        log.info("[BOOT] ExitArbiter configured for EXIT_V3_ADAPTIVE")


def archived_exit_bundle_rules(self) -> None:
    """Historical exit bundle handling for FARM and REN EXIT_V2."""
    if self.exit_ml_mode == "exit_transformer_v0":
        self.exit_bundle = None  # Transformer decider already loaded
    elif self.exit_only_v2_drift or (hasattr(self, "farm_v2b_mode") and self.farm_v2b_mode):
        log.info("[BOOT] FARM exit mode detected: skipping exit model bundle (using rule-based exits)")
        self.exit_bundle = None
    else:
        self.exit_bundle = None
        if self.replay_mode or self.fast_replay or self.policy.get("exit_policy"):
            raise RuntimeError(
                "[EXIT_MODEL_REQUIRED] Exit model bundle is required for configured exit policy; "
                "legacy XGB bundle path has been removed. Provide an explicit exit bundle."
            )


def archived_tick_watcher_disable(self) -> None:
    """Historical TickWatcher gating for REN EXIT_V2 and FARM_V1 modes."""
    self.tick_cfg = self.policy.get("tick_exit", {}) or {}
    if self.exit_only_v2_drift:
        self.tick_cfg["enabled"] = False
    if not self.exit_only_v2_drift:
        self.tick_watcher = self.tick_watcher  # Placeholder for original TickWatcher wiring
    else:
        self.tick_watcher = None
        if hasattr(self, "farm_v1_mode") and self.farm_v1_mode:
            log.info("[BOOT] TickWatcher disabled (FARM_V1 mode)")
        else:
            log.info("[BOOT] TickWatcher disabled (REN EXIT_V2 mode)")


def archived_teardown_exit_state(self, trade_id: str) -> None:
    """Historical per-trade FARM_V2_RULES state cleanup."""
    if hasattr(self, "exit_farm_v2_rules_states"):
        if trade_id in self.exit_farm_v2_rules_states:
            self.exit_farm_v2_rules_states.pop(trade_id, None)
            log.debug("[EXIT_PROFILE] Cleared FARM_V2_RULES state for trade %s", trade_id)


def archived_farm_diag_dump(self, farm_diag: Dict[str, Any], log_dir: Path) -> None:
    """Historical FARM_V2B diagnostic dump."""
    if farm_diag.get("n_bars", 0) <= 0:
        return
    diag_filename = f"farm_entry_diag_{Path(log_dir).name}.json"
    diag_path = log_dir / diag_filename
    diag_serializable = {
        "n_bars": int(farm_diag["n_bars"]),
        "n_raw_candidates": int(farm_diag["n_raw_candidates"]),
        "n_after_stage0": int(farm_diag["n_after_stage0"]),
        "n_after_farm_regime": int(farm_diag["n_after_farm_regime"]),
        "n_after_brutal_guard": int(farm_diag["n_after_brutal_guard"]),
        "n_after_policy_thresholds": int(farm_diag["n_after_policy_thresholds"]),
        "p_long_values": [float(x) for x in farm_diag["p_long_values"]],
        "sessions": {str(k): int(v) for k, v in farm_diag["sessions"].items()},
        "atr_regimes": {str(k): int(v) for k, v in farm_diag["atr_regimes"].items()},
        "farm_regimes": {str(k): int(v) for k, v in farm_diag["farm_regimes"].items()},
    }
    if len(diag_serializable["p_long_values"]) > 0:
        p_long_arr = np.array(diag_serializable["p_long_values"])
        diag_serializable["p_long_stats"] = {
            "min": float(np.min(p_long_arr)),
            "max": float(np.max(p_long_arr)),
            "mean": float(np.mean(p_long_arr)),
            "median": float(np.median(p_long_arr)),
            "p5": float(np.percentile(p_long_arr, 5)),
            "p50": float(np.percentile(p_long_arr, 50)),
            "p95": float(np.percentile(p_long_arr, 95)),
            "p99": float(np.percentile(p_long_arr, 99)),
        }
    else:
        diag_serializable["p_long_stats"] = {}
    with open(diag_path, "w") as f:
        jsonlib.dump(diag_serializable, f, indent=2)
    log.info(f"[FARM_DIAG] Dumped FARM_V2B entry diagnostic to {diag_path}")


def archived_ensure_bid_ask_columns(df, context: str) -> None:
    """Historical bid/ask validation that referenced FARM_V2B replay."""
    required = ["bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Bid/ask required for FARM_V2B 2025 replay, but missing in candles (context={context}): {missing}"
        )


if __name__ == "__main__":
    print("ARCHIVE_ONLY: forbidden exit mode branches snapshot")
