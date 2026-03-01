# ARCHIVED: forbidden in canonical truth; removed from runtime on 2026-02-28.
# Source: gx1/execution/oanda_demo_runner.py FARM exit modes.

from pathlib import Path
from typing import Any, Dict
import json as jsonlib
import joblib

# Archived FARM exit branching (FARM_V2_RULES, FARM_V1).

def _archived_farm_exit_block(self, exit_cfg: Dict[str, Any], exit_cfg_path: Path) -> None:
    """
    Non-runtime archive of FARM exit initialization previously in GX1DemoRunner.
    """
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
    self.farm_v1_mode = False
    self.farm_v2_mode = False
    self.farm_v2b_mode = False

    if exit_type == "FARM_V2_RULES":
        from gx1.policy.exit_farm_v2_rules import get_exit_policy_farm_v2_rules
        exit_params = exit_cfg.get("exit", {}).get("params", {})
        self.exit_verbose_logging = bool(exit_params.get("verbose_logging", False))

        def _build_exit_farm_v2_rules_policy():
            return get_exit_policy_farm_v2_rules(
                enable_rule_a=exit_params.get("enable_rule_a", False),
                enable_rule_b=exit_params.get("enable_rule_b", False),
                enable_rule_c=exit_params.get("enable_rule_c", False),
                rule_a_profit_min_bps=exit_params.get("rule_a_profit_min_bps", 6.0),
                rule_a_profit_max_bps=exit_params.get("rule_a_profit_max_bps", 9.0),
                rule_a_adaptive_threshold_bps=exit_params.get("rule_a_adaptive_threshold_bps", 4.0),
                rule_a_trailing_stop_bps=exit_params.get("rule_a_trailing_stop_bps", 2.0),
                rule_a_adaptive_bars=exit_params.get("rule_a_adaptive_bars", 3),
                rule_b_mae_threshold_bps=exit_params.get("rule_b_mae_threshold_bps", -4.0),
                rule_b_max_bars=exit_params.get("rule_b_max_bars", 6),
                rule_c_timeout_bars=exit_params.get("rule_c_timeout_bars", 8),
                rule_c_min_profit_bps=exit_params.get("rule_c_min_profit_bps", 2.0),
                debug_trade_ids=exit_params.get("debug_trade_ids", []),
                force_exit_bars=exit_params.get("force_exit_bars"),
                verbose_logging=exit_params.get("verbose_logging", False),
                log_every_n_bars=exit_params.get("log_every_n_bars", 5),
            )

        self.exit_farm_v2_rules_factory = _build_exit_farm_v2_rules_policy
        self.exit_farm_v2_rules_policy = self.exit_farm_v2_rules_factory()
        self.exit_farm_v2_rules_states = {}

        rules_str = []
        if exit_params.get("enable_rule_a", False):
            rules_str.append("A")
        if exit_params.get("enable_rule_b", False):
            rules_str.append("B")
        if exit_params.get("enable_rule_c", False):
            rules_str.append("C")

        log.info(f"[BOOT] EXIT_FARM_V2_RULES enabled: Rules={'+'.join(rules_str) if rules_str else 'NONE'}")

        self.exit_farm_v1_policy = None
        tick_exit_cfg = self.policy.get("tick_exit", {})
        tick_exit_cfg["enabled"] = False
        self.policy["tick_exit"] = tick_exit_cfg
        self.policy["broker_side_tp_sl"] = False
        if hasattr(self, "tick_cfg"):
            self.tick_cfg["enabled"] = False

        log.info("[BOOT] FARM_V2_RULES mode: All non-FARM exits disabled")
        self.exit_only_v2_drift = True

    elif exit_type == "FARM_V1":
        self.farm_v1_mode = True
        FARM_BASELINE_VERSION = "FARM_V1_STABLE_FINAL_2025_12_05"
        self.farm_baseline_version = FARM_BASELINE_VERSION

        entry_farm_v2b_cfg = self.policy.get("entry_v9_policy_farm_v2b", {})
        entry_farm_v2_cfg = self.policy.get("entry_v9_policy_farm_v2", {})
        if entry_farm_v2b_cfg.get("enabled", False):
            self.farm_v2_mode = True
            self.farm_v2b_mode = True
        elif entry_farm_v2_cfg.get("enabled", False):
            self.farm_v2_mode = True
            self.farm_v2b_mode = False
        else:
            self.farm_v2b_mode = False

        exit_config_name = exit_cfg_path.stem
        if self.farm_v2_mode or (hasattr(self, "farm_v2b_mode") and self.farm_v2b_mode):
            if exit_config_name != "FARM_EXIT_V2_AGGRO":
                raise RuntimeError(
                    f"FARM_V2/V2B mode MUST use FARM_EXIT_V2_AGGRO exit policy. Found: {exit_config_name}. "
                    f"FARM_V2/V2B must use FARM_EXIT_V2_AGGRO."
                )
        else:
            if exit_config_name != "FARM_EXIT_V1_STABLE":
                raise RuntimeError(
                    f"FARM_V1 mode MUST use FARM_EXIT_V1_STABLE exit policy. Found: {exit_config_name}. "
                    f"This is a frozen baseline (version {FARM_BASELINE_VERSION}). "
                    f"Changes require explicit approval and new major version."
                )

        from gx1.policy.exit_farm_v1_policy import get_exit_policy_farm_v1  # type: ignore[reportMissingImports]
        exit_params = exit_cfg.get("exit", {}).get("params", {})

        expected_sl = -20.0
        expected_tp = 8.0
        expected_timeout = 8
        actual_sl = exit_params.get("sl_bps", -6.0)
        actual_tp = exit_params.get("tp1_bps", 6.0)
        actual_timeout = exit_params.get("timeout_bars", 8)

        if abs(actual_sl - expected_sl) > 0.01 or abs(actual_tp - expected_tp) > 0.01 or actual_timeout != expected_timeout:
            log.warning(
                "[BOOT] FARM_EXIT_V1_STABLE parameter mismatch: "
                f"Expected SL={expected_sl}, TP={expected_tp}, TIMEOUT={expected_timeout}, "
                f"Found SL={actual_sl}, TP={actual_tp}, TIMEOUT={actual_timeout}. "
                f"Baseline version: {FARM_BASELINE_VERSION}"
            )

        self.exit_farm_v1_policy = get_exit_policy_farm_v1(
            sl_bps=exit_params.get("sl_bps", -6.0),
            tp1_bps=exit_params.get("tp1_bps", 6.0),
            tp2_bps=exit_params.get("tp2_bps", 6.0),
            timeout_bars=exit_params.get("timeout_bars", 8),
        )
        log.info(
            "[BOOT] EXIT_FARM_V1 enabled: SL=%.0f bps, TP1=%.0f bps, TP2=%.0f bps, TIMEOUT=%d bars (Baseline: %s)",
            exit_params.get("sl_bps", -6.0),
            exit_params.get("tp1_bps", 6.0),
            exit_params.get("tp2_bps", 6.0),
            exit_params.get("timeout_bars", 8),
            FARM_BASELINE_VERSION,
        )

        tick_exit_cfg = self.policy.get("tick_exit", {})
        tick_exit_cfg["enabled"] = False
        self.policy["tick_exit"] = tick_exit_cfg
        self.policy["broker_side_tp_sl"] = False

        if hasattr(self, "tick_cfg"):
            self.tick_cfg["enabled"] = False

        log.info("[BOOT] FARM_V1 mode: All non-FARM exits disabled (tick_exit, broker TP/SL)")

        if self.farm_v2_mode:
            log.info("[BOOT] FARM_V2 entry policy enabled - using FARM_EXIT_V2_AGGRO exit")

            self.farm_entry_meta_model = None
            self.farm_entry_meta_feature_cols = None

            meta_cfg = self.policy.get("meta_model", {})
            if not meta_cfg:
                meta_cfg = entry_farm_v2_cfg.get("meta_model", {})

            model_path = meta_cfg.get("model_path")
            if not model_path or model_path == "null":
                model_path = "gx1/models/farm_entry_meta/baseline_model.pkl"
                alt_paths = [
                    "gx1/models/farm_entry_meta_baseline.joblib",
                    "gx1/models/farm_entry_meta/baseline_model.joblib",
                ]
                for alt_path in alt_paths:
                    if Path(alt_path).exists():
                        model_path = alt_path
                        break

            if model_path and Path(model_path).exists():
                try:
                    self.farm_entry_meta_model = joblib.load(model_path)
                    log.info(f"[BOOT] Loaded FARM entry meta-model from {model_path}")

                    feature_cols_path = meta_cfg.get("feature_cols_path")
                    if not feature_cols_path or feature_cols_path == "null":
                        feature_cols_paths = [
                            Path(model_path).parent / "feature_cols.json",
                            Path(model_path).parent / "feature_cols.txt",
                            Path("gx1/models/farm_entry_meta/feature_cols.json"),
                        ]
                        for fcp in feature_cols_paths:
                            if fcp.exists():
                                feature_cols_path = str(fcp)
                                break

                    if feature_cols_path and Path(feature_cols_path).exists():
                        if Path(feature_cols_path).suffix == ".json":
                            with open(feature_cols_path, 'r') as f:
                                feature_data = jsonlib.load(f)
                                if isinstance(feature_data, list):
                                    self.farm_entry_meta_feature_cols = feature_data
                                elif isinstance(feature_data, dict) and "feature_cols" in feature_data:
                                    self.farm_entry_meta_feature_cols = feature_data["feature_cols"]
                                else:
                                    log.warning(f"[BOOT] Unexpected feature_cols format in {feature_cols_path}")
                        else:
                            with open(feature_cols_path, 'r') as f:
                                self.farm_entry_meta_feature_cols = [line.strip() for line in f if line.strip()]

                        if self.farm_entry_meta_feature_cols:
                            log.info(f"[BOOT] Loaded {len(self.farm_entry_meta_feature_cols)} feature columns from {feature_cols_path}")
                        else:
                            log.warning("[BOOT] No feature columns loaded - will auto-detect at runtime")
                    else:
                        log.warning("[BOOT] Feature columns file not found - will auto-detect at runtime")
                except Exception as e:
                    log.error(f"[BOOT] Failed to load FARM entry meta-model: {e}")
                    raise RuntimeError(
                        f"FARM_V2 requires meta-model but loading failed: {e}. "
                        f"Model path: {model_path}. "
                        f"FARM_V2 cannot run without meta-model."
                    )
            else:
                raise RuntimeError(
                    f"FARM_V2 requires meta-model but file not found: {model_path}. "
                    f"FARM_V2 cannot run without meta-model. "
                    f"Please train and save the model first."
                )

        exit_model_cfg = self.policy.get("exit_model", {})
        if exit_model_cfg.get("enabled", False) or exit_model_cfg.get("ai_overlay", False):
            raise RuntimeError(
                "AI exit overlay is disabled for FARM_V1. "
                "FARM_V1 uses FARM_EXIT_V1_STABLE only (frozen baseline). "
                "See gx1/docs/FARM_EXIT_V1_FINAL.md for details."
            )

        if self.exit_only_v2_drift:
            exit_control_cfg = self.policy.get("exit_control", {})
            exit_control_cfg["allowed_loss_closers"] = [
                "EXIT_FARM_SL",
                "EXIT_FARM_SL_BREAKEVEN",
                "EXIT_FARM_TP",
                "EXIT_FARM_TIMEOUT",
            ]
            exit_control_cfg.setdefault("allow_model_exit_when", {})["min_bars"] = 1
            exit_control_cfg.setdefault("allow_model_exit_when", {})["min_pnl_bps"] = -100
            exit_control_cfg.setdefault("allow_model_exit_when", {})["min_exit_prob"] = 0.0
            self.policy["exit_control"] = exit_control_cfg

# Archived ExitArbiter overrides for FARM/EXIT_V3_ADAPTIVE (non-runtime).
def _archived_exit_control_overrides(self, exit_control_cfg: Dict[str, Any]) -> None:
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
    elif hasattr(self, "exit_farm_v2_rules_policy") and self.exit_farm_v2_rules_policy is not None and self.exit_only_v2_drift:
        if not exit_control_cfg.get("allowed_loss_closers"):
            self.exit_control.allowed_loss_closers = [
                "RULE_A_PROFIT",
                "RULE_A_TRAILING",
                "RULE_B_FAST_LOSS",
                "RULE_C_TIMEOUT",
            ]
        self.exit_control.allow_model_exit_when["min_bars"] = 1
        self.exit_control.allow_model_exit_when["min_pnl_bps"] = -100
        self.exit_control.allow_model_exit_when["min_exit_prob"] = 0.0
    elif hasattr(self, "exit_farm_v1_policy") and self.exit_farm_v1_policy is not None and self.exit_only_v2_drift:
        self.exit_control.allowed_loss_closers = [
            "EXIT_FARM_SL",
            "EXIT_FARM_SL_BREAKEVEN",
            "EXIT_FARM_TP",
            "EXIT_FARM_TIMEOUT",
        ]
        self.exit_control.allow_model_exit_when["min_bars"] = 1
        self.exit_control.allow_model_exit_when["min_pnl_bps"] = -100
        self.exit_control.allow_model_exit_when["min_exit_prob"] = 0.0

# Archived per-trade FARM_V2_RULES state initialization (non-runtime).
def _archived_init_farm_v2_rules_state(self, trade, *, context: str) -> None:
    if not getattr(self, "exit_farm_v2_rules_factory", None):
        raise RuntimeError("FARM_V2_RULES factory not configured, cannot initialize exit state")
    if trade.entry_bid is None or trade.entry_ask is None:
        raise ValueError(
            f"[EXIT_PROFILE] Missing bid/ask for FARM_V2_RULES init trade_id={trade.trade_id} context={context}"
        )
    policy = self.exit_farm_v2_rules_states.get(trade.trade_id)
    if policy is None:
        policy = self.exit_farm_v2_rules_factory()
        self.exit_farm_v2_rules_states[trade.trade_id] = policy
    policy.reset_on_entry(
        entry_bid=trade.entry_bid,
        entry_ask=trade.entry_ask,
        entry_ts=trade.entry_time,
        side=trade.side,
        trade_id=trade.trade_id,
    )
    if not getattr(trade, "extra", None):
        trade.extra = {}
    trade.extra["exit_farm_v2_rules_initialized"] = True

# Archived replay-time FARM_V2_RULES on_bar loop (non-runtime).
def _archived_replay_farm_v2_rules(self, trade, price_bid, price_ask, bar_ts, exit_profile):
    if (getattr(self, "exit_farm_v2_rules_factory", None) and
        exit_profile and exit_profile.startswith("FARM_EXIT_V2_RULES")):
        policy = self.exit_farm_v2_rules_states.get(trade.trade_id)
        if policy is None:
            try:
                self._init_farm_v2_rules_state(trade, context="replay_exit_loop")
                policy = self.exit_farm_v2_rules_states.get(trade.trade_id)
            except Exception:
                return
        decision = policy.on_bar(price_bid=price_bid, price_ask=price_ask, ts=bar_ts)
        if decision is not None:
            self.request_close(
                trade_id=trade.trade_id,
                source="EXIT_POLICY",
                reason=decision.reason,
                px=decision.exit_price,
                pnl_bps=decision.pnl_bps,
                bars_in_trade=decision.bars_held,
            )
