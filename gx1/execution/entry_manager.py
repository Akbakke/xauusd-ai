from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from collections import deque, defaultdict

import logging
import os
import numpy as np
import pandas as pd
import time
import traceback
import uuid

from gx1.execution.live_features import build_live_entry_features, infer_session_tag
from gx1.sniper.policy.sniper_regime_size_overlay import apply_size_overlay
from gx1.sniper.policy.sniper_q4_cchop_size_overlay import apply_q4_cchop_overlay
from gx1.sniper.policy.sniper_q4_atrend_size_overlay import apply_q4_atrend_overlay
from gx1.sniper.policy.sniper_q4_eu_timing_size_overlay import apply_q4_eu_timing_overlay
from gx1.sniper.policy.runtime_regime_inputs import get_runtime_regime_inputs

try:
    from gx1.runtime.overlays.entry_v10_1_size_overlay import load_entry_v10_1_size_overlay
    ENTRY_V10_1_SIZE_OVERLAY_AVAILABLE = True
except ImportError:
    ENTRY_V10_1_SIZE_OVERLAY_AVAILABLE = False

log = logging.getLogger(__name__)

if not ENTRY_V10_1_SIZE_OVERLAY_AVAILABLE:
    log.warning("[ENTRY_V10_1_SIZE_OVERLAY] Module not available - V10.1 size overlay disabled")

if TYPE_CHECKING:
    from gx1.execution.oanda_demo_runner import GX1DemoRunner, LiveTrade


class EntryManager:
    def __init__(self, runner: "GX1DemoRunner", exit_config_name: Optional[str] = None) -> None:
        object.__setattr__(self, "_runner", runner)
        object.__setattr__(self, "exit_config_name", exit_config_name)
        object.__setattr__(self, "entry_feature_telemetry", None)
        require_telemetry = os.environ.get("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
        if require_telemetry:
            from gx1.execution.entry_feature_telemetry import EntryFeatureTelemetryCollector
            output_dir = getattr(runner, "output_dir", None)
            object.__setattr__(self, "entry_feature_telemetry", EntryFeatureTelemetryCollector(output_dir=output_dir))
        object.__setattr__(self, "entry_telemetry", {
            "n_cycles": 0,
            "n_eligible_hard": 0,
            "n_eligible_cycles": 0,
            "n_precheck_pass": 0,
            "n_predictions": 0,
            "n_candidates": 0,
            "n_candidate_pass": 0,
            "n_trades_created": 0,
            "p_long_values": [],
            "candidate_sessions": {},
            "trade_sessions": {},
            "n_entry_snapshots_written": 0,
            "n_entry_snapshots_failed": 0,
            "n_context_built": 0,
            "n_context_missing_or_invalid": 0,
            "n_ctx_model_calls": 0,
            "ctx_proof_pass_count": 0,
            "ctx_proof_fail_count": 0,
            "vol_regime_unknown_count": 0,
        })
        object.__setattr__(self, "eval_calls_total", 0)
        object.__setattr__(self, "eval_calls_prebuilt_gate_true", 0)
        object.__setattr__(self, "eval_calls_prebuilt_gate_false", 0)
        object.__setattr__(self, "killchain_version", 1)
        object.__setattr__(self, "killchain_n_entry_pred_total", 0)
        object.__setattr__(self, "killchain_n_above_threshold", 0)
        object.__setattr__(self, "killchain_n_after_session_guard", 0)
        object.__setattr__(self, "killchain_n_after_vol_guard", 0)
        object.__setattr__(self, "killchain_n_after_regime_guard", 0)
        object.__setattr__(self, "killchain_n_after_risk_sizing", 0)
        object.__setattr__(self, "killchain_n_trade_create_attempts", 0)
        object.__setattr__(self, "killchain_n_trade_created", 0)
        object.__setattr__(self, "killchain_block_reason_counts", {
            "BLOCK_BELOW_THRESHOLD": 0,
            "BLOCK_SESSION": 0,
            "BLOCK_VOL": 0,
            "BLOCK_REGIME": 0,
            "BLOCK_RISK": 0,
            "BLOCK_POSITION_LIMIT": 0,
            "BLOCK_COOLDOWN": 0,
            "BLOCK_UNKNOWN": 0,
        })
        object.__setattr__(self, "killchain_unknown_examples", [])
        object.__setattr__(self, "farm_diag", {})
        object.__setattr__(self, "stage0_reasons", defaultdict(int))
        object.__setattr__(self, "stage0_total_considered", 0)
        object.__setattr__(self, "veto_hard", {
            "veto_hard_warmup": 0,
            "veto_hard_session": 0,
            "veto_hard_spread": 0,
            "veto_hard_killswitch": 0,
        })
        object.__setattr__(self, "veto_soft", {"veto_soft_vol_regime_extreme": 0})
        object.__setattr__(self, "veto_pre", {
            "veto_pre_warmup": 0,
            "veto_pre_session": 0,
            "veto_pre_regime": 0,
            "veto_pre_spread": 0,
            "veto_pre_atr": 0,
            "veto_pre_killswitch": 0,
            "veto_pre_model_missing": 0,
            "veto_pre_nan_features": 0,
        })
        object.__setattr__(self, "veto_cand", {
            "veto_cand_threshold": 0,
            "veto_cand_risk_guard": 0,
            "veto_cand_max_trades": 0,
            "veto_cand_big_brain": 0,
        })
        object.__setattr__(self, "veto_counters", {})
        object.__setattr__(self, "threshold_used", None)
        object.__setattr__(self, "p_long_values", [])
        object.__setattr__(self, "cluster_guard_history", deque(maxlen=600))
        object.__setattr__(self, "cluster_guard_atr_median", None)
        object.__setattr__(self, "spread_history", deque(maxlen=600))
        object.__setattr__(self, "n_v10_calls", 0)
        object.__setattr__(self, "n_v10_pred_ok", 0)
        object.__setattr__(self, "n_v10_pred_none_or_nan", 0)
        object.__setattr__(self, "_v10_log_count", 0)
        object.__setattr__(self, "_sniper_risk_guard", None)
        object.__setattr__(self, "_shadow_thresholds", [])
        object.__setattr__(self, "_shadow_journal_path", None)
        object.__setattr__(self, "_entry_critic_model", None)
        object.__setattr__(self, "_entry_critic_meta", None)
        object.__setattr__(self, "_entry_critic_feature_order", None)
        # Lazy-set attributes (must live on manager, not runner)
        object.__setattr__(self, "_last_atr_proxy", None)
        object.__setattr__(self, "_last_spread_bps", None)
        object.__setattr__(self, "_last_policy_state", None)
        object.__setattr__(self, "_next_trade_id", 0)
        object.__setattr__(self, "_entry_v10_1_size_overlay", None)
        object.__setattr__(self, "_force_entry_start_time", None)
        object.__setattr__(self, "_force_entry_trade_count", 0)
        object.__setattr__(self, "parity_sample_counter", 0)
        object.__setattr__(self, "parity_sample_every_n", 10)

    def __getattr__(self, name: str):
        # only called if attribute not found on self
        return getattr(self._runner, name)

    def __setattr__(self, name: str, value):
        if name == "_runner":
            object.__setattr__(self, name, value)
            return

        # If EntryManager already owns this attribute (instance or class), keep it local
        if name in self.__dict__ or hasattr(type(self), name):
            object.__setattr__(self, name, value)
            return

        # Otherwise treat as runner passthrough (compat)
        setattr(self._runner, name, value)

    def _killchain_inc_reason(self, reason_code: str) -> None:
        if reason_code not in self.killchain_block_reason_counts:
            reason_code = "BLOCK_UNKNOWN"
        self.killchain_block_reason_counts[reason_code] += 1

    def _killchain_record_unknown(self, example: Dict[str, Any]) -> None:
        if len(self.killchain_unknown_examples) >= 5:
            return
        self.killchain_unknown_examples.append(example)

    @staticmethod
    def _percentile_from_history(history: deque, value: Optional[float]) -> Optional[float]:
        if value is None or not history:
            return None
        try:
            arr = np.array(history, dtype=float)
        except Exception:
            return None
        if arr.size == 0 or not np.isfinite(value):
            return None
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        sorted_arr = np.sort(arr)
        idx = np.searchsorted(sorted_arr, value, side="right")
        pct = (idx / float(sorted_arr.size)) * 100.0
        return max(0.0, min(100.0, float(pct)))

    @staticmethod
    def _compute_range_features(candles: pd.DataFrame, window: int = 96) -> tuple[float, float]:
        eps = 1e-12
        default_range_pos = 0.5
        default_distance = 0.5
        try:
            if len(candles) < window:
                return (default_range_pos, default_distance)
            has_direct = all(col in candles.columns for col in ['high', 'low', 'close'])
            has_bid_ask = all(col in candles.columns for col in ['bid_high', 'ask_high', 'bid_low', 'ask_low', 'bid_close', 'ask_close'])
            if not has_direct and not has_bid_ask:
                return (default_range_pos, default_distance)
            if len(candles) >= window + 1:
                recent = candles.iloc[-(window+1):-1]
            else:
                recent = candles.tail(window)
            if has_direct:
                high_vals = recent['high'].values
                low_vals = recent['low'].values
                close_vals = recent['close'].values
            else:
                high_vals = (recent['bid_high'].values + recent['ask_high'].values) / 2.0
                low_vals = (recent['bid_low'].values + recent['ask_low'].values) / 2.0
                close_vals = (recent['bid_close'].values + recent['ask_close'].values) / 2.0
            range_hi = float(np.max(high_vals))
            range_lo = float(np.min(low_vals))
            price_ref = float(close_vals[-1])
            denom = max(eps, range_hi - range_lo)
            range_pos_raw = (price_ref - range_lo) / denom
            range_pos = max(0.0, min(1.0, float(range_pos_raw)))
            dist_edge = min(range_pos, 1.0 - range_pos)
            distance_to_range = dist_edge * 2.0
            distance_to_range = max(0.0, min(1.0, float(distance_to_range)))
            return (range_pos, distance_to_range)
        except Exception:
            return (default_range_pos, default_distance)

    HARD_ELIGIBILITY_WARMUP = "HARD_WARMUP"
    HARD_ELIGIBILITY_SESSION_BLOCK = "HARD_SESSION_BLOCK"
    HARD_ELIGIBILITY_SPREAD_CAP = "HARD_SPREAD_CAP"
    HARD_ELIGIBILITY_KILLSWITCH = "HARD_KILLSWITCH"
    SOFT_ELIGIBILITY_VOL_REGIME_EXTREME = "SOFT_VOL_REGIME_EXTREME"

    def _check_hard_eligibility(
        self,
        candles: pd.DataFrame,
        policy_state: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        current_ts = candles.index[-1] if len(candles) > 0 else None
        warmup_bars = getattr(self, "warmup_bars", 288)
        if len(candles) < warmup_bars:
            return False, self.HARD_ELIGIBILITY_WARMUP
        policy_cfg = self.policy.get("entry_policy_sniper_v10_ctx", {})
        current_session = policy_state.get("session")
        if not current_session:
            from gx1.execution.live_features import infer_session_tag
            current_session = infer_session_tag(current_ts).upper()
            policy_state["session"] = current_session
        allowed_sessions = policy_cfg.get("allowed_sessions", ["EU", "OVERLAP", "US"])
        if current_session not in allowed_sessions:
            return False, self.HARD_ELIGIBILITY_SESSION_BLOCK
        spread_bps = self._get_spread_bps_before_features(candles)
        if spread_bps is not None:
            spread_hard_cap_bps = policy_cfg.get("spread_hard_cap_bps", 100.0)
            if spread_bps > spread_hard_cap_bps:
                return False, self.HARD_ELIGIBILITY_SPREAD_CAP
        if self._is_kill_switch_active():
            return False, self.HARD_ELIGIBILITY_KILLSWITCH
        return True, None

    def _get_spread_bps_before_features(self, candles: pd.DataFrame) -> Optional[float]:
        if candles.empty:
            return None
        try:
            if "bid_close" in candles.columns and "ask_close" in candles.columns:
                bid = candles["bid_close"].iloc[-1]
                ask = candles["ask_close"].iloc[-1]
                if pd.notna(bid) and pd.notna(ask) and bid > 0:
                    spread_price = ask - bid
                    spread_bps = (spread_price / bid) * 10000.0
                    return float(spread_bps)
            elif "spread" in candles.columns:
                spread_price = candles["spread"].iloc[-1]
                if pd.notna(spread_price):
                    close = candles.get("close", candles.index)
                    if len(close) > 0:
                        price_ref = float(close.iloc[-1])
                        if price_ref > 0:
                            spread_bps = (spread_price / price_ref) * 10000.0
                            return float(spread_bps)
        except Exception:
            pass
        return None

    def _is_kill_switch_active(self) -> bool:
        try:
            from pathlib import Path
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            kill_flag = project_root / "KILL_SWITCH_ON"
            return kill_flag.exists()
        except Exception:
            return False

    @staticmethod
    def _compute_range_edge_dist_atr(
        candles: pd.DataFrame,
        price_ref: float,
        range_hi: float,
        range_lo: float,
        atr_value: Optional[float],
        window: int = 96,
    ) -> float:
        eps = 1e-12
        default_value = 0.0
        try:
            if atr_value is None or not np.isfinite(atr_value) or atr_value <= 0:
                return default_value
            dist_to_low = max(0.0, price_ref - range_lo)
            dist_to_high = max(0.0, range_hi - price_ref)
            dist_edge_price = min(dist_to_low, dist_to_high)
            denom = max(eps, atr_value)
            range_edge_dist_atr_raw = dist_edge_price / denom
            return max(0.0, min(10.0, float(range_edge_dist_atr_raw)))
        except Exception:
            return default_value

    def _compute_cheap_atr_proxy(self, candles: pd.DataFrame, window: int = 14) -> Optional[float]:
        if candles.empty or len(candles) < window:
            return None
        try:
            high = candles.get("high", candles.get("high", None))
            low = candles.get("low", candles.get("low", None))
            close = candles.get("close", candles.get("close", None))
            if high is None or low is None or close is None:
                return None
            high_arr = high.iloc[-window:].values
            low_arr = low.iloc[-window:].values
            close_arr = close.iloc[-window:].values
            tr1 = high_arr - low_arr
            tr2 = np.abs(high_arr - np.roll(close_arr, 1))
            tr3 = np.abs(low_arr - np.roll(close_arr, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = np.mean(tr)
            return float(atr)
        except Exception:
            return None

    def _check_soft_eligibility(
        self,
        candles: pd.DataFrame,
        policy_state: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        atr_proxy = self._compute_cheap_atr_proxy(candles, window=14)
        self._last_atr_proxy = atr_proxy
        self._last_spread_bps = self._get_spread_bps_before_features(candles)
        if atr_proxy is not None:
            close = candles.get("close", None)
            if close is not None and len(close) > 0:
                current_price = float(close.iloc[-1])
                if current_price > 0:
                    atr_bps = (atr_proxy / current_price) * 10000.0
                    if atr_bps > 200.0:
                        return False, self.SOFT_ELIGIBILITY_VOL_REGIME_EXTREME
        return True, None

    def evaluate_entry(self, candles: pd.DataFrame) -> Optional[LiveTrade]:
        if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
            self.entry_feature_telemetry.reset_routing_for_next_bar()
        self.eval_calls_total += 1
        self.entry_telemetry["n_cycles"] += 1

        import os as os_module
        prebuilt_enabled = os_module.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1"
        is_replay = getattr(self._runner, "replay_mode", False)

        if is_replay and prebuilt_enabled:
            if not hasattr(self._runner, "lookup_attempts"):
                self._runner.lookup_attempts = 0
                self._runner.lookup_hits = 0
                self._runner.lookup_misses = 0
                self._runner.lookup_miss_details = []
            self._runner.lookup_attempts += 1

        policy_state = {}
        current_ts = candles.index[-1] if len(candles) > 0 else pd.Timestamp.now(tz="UTC")
        from gx1.execution.live_features import infer_session_tag
        current_session = infer_session_tag(current_ts).upper()
        policy_state["session"] = current_session

        eligible, eligibility_reason = self._check_hard_eligibility(candles, policy_state)
        if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
            if eligible:
                self.entry_feature_telemetry.record_gate(
                    gate_name="hard_eligibility", executed=True, blocked=False, passed=True, reason=None,
                )
            else:
                self.entry_feature_telemetry.record_gate(
                    gate_name="hard_eligibility", executed=True, blocked=True, passed=False, reason=eligibility_reason,
                )
        if not eligible:
            if is_replay and prebuilt_enabled and hasattr(self._runner, "lookup_misses"):
                self._runner.lookup_misses += 1
            reason_to_veto_key = {
                self.HARD_ELIGIBILITY_WARMUP: "veto_hard_warmup",
                self.HARD_ELIGIBILITY_SESSION_BLOCK: "veto_hard_session",
                self.HARD_ELIGIBILITY_SPREAD_CAP: "veto_hard_spread",
                self.HARD_ELIGIBILITY_KILLSWITCH: "veto_hard_killswitch",
            }
            veto_key = reason_to_veto_key.get(eligibility_reason)
            if veto_key and veto_key in self.veto_hard:
                self.veto_hard[veto_key] += 1
            return None

        if "n_eligible_hard" not in self.entry_telemetry:
            self.entry_telemetry["n_eligible_hard"] = 0
        self.entry_telemetry["n_eligible_hard"] += 1

        soft_eligible, soft_reason = self._check_soft_eligibility(candles, policy_state)
        if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
            if soft_eligible:
                self.entry_feature_telemetry.record_gate(
                    gate_name="soft_eligibility", executed=True, blocked=False, passed=True, reason=None,
                )
                self.entry_feature_telemetry.record_control_flow("SOFT_ELIGIBILITY_RETURN_TRUE")
            else:
                self.entry_feature_telemetry.record_gate(
                    gate_name="soft_eligibility", executed=True, blocked=True, passed=False, reason=soft_reason,
                )
                self.entry_feature_telemetry.record_control_flow("SOFT_ELIGIBILITY_RETURN_FALSE")
        if not soft_eligible:
            if soft_reason == self.SOFT_ELIGIBILITY_VOL_REGIME_EXTREME:
                self.veto_soft["veto_soft_vol_regime_extreme"] += 1
            return None

        self.entry_telemetry["n_eligible_cycles"] += 1
        if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
            self.entry_feature_telemetry.record_control_flow("AFTER_SOFT_ELIGIBILITY_PASSED")

        entry_context_features = None
        context_features_enabled = os.getenv("ENTRY_CONTEXT_FEATURES_ENABLED", "false").lower() == "true"
        if context_features_enabled:
            try:
                from gx1.execution.entry_context_features import build_entry_context_features
                atr_proxy = getattr(self, "_last_atr_proxy", None)
                spread_bps = getattr(self, "_last_spread_bps", None)
                entry_context_features = build_entry_context_features(
                    candles=candles,
                    policy_state=policy_state,
                    atr_proxy=atr_proxy,
                    spread_bps=spread_bps,
                    is_replay=is_replay,
                )
                if "n_context_built" not in self.entry_telemetry:
                    self.entry_telemetry["n_context_built"] = 0
                self.entry_telemetry["n_context_built"] += 1
            except Exception as e:
                if is_replay:
                    raise RuntimeError(
                        f"CONTEXT_FEATURES_BUILD_FAILED: Failed to build context features: {e}"
                    ) from e
                if "n_context_missing_or_invalid" not in self.entry_telemetry:
                    self.entry_telemetry["n_context_missing_or_invalid"] = 0
                self.entry_telemetry["n_context_missing_or_invalid"] += 1

        prebuilt_features_df_exists = hasattr(self._runner, "prebuilt_features_df")
        prebuilt_features_df_is_none = not prebuilt_features_df_exists or self._runner.prebuilt_features_df is None
        prebuilt_used_flag = getattr(self._runner, "prebuilt_used", False) if hasattr(self._runner, "prebuilt_used") else False
        prebuilt_available = (
            is_replay and prebuilt_enabled and prebuilt_features_df_exists
            and not prebuilt_features_df_is_none and hasattr(self._runner, "prebuilt_used") and prebuilt_used_flag
        )
        if prebuilt_available:
            self.eval_calls_prebuilt_gate_true += 1
        else:
            self.eval_calls_prebuilt_gate_false += 1
            if is_replay and prebuilt_enabled:
                if not hasattr(self._runner, "lookup_misses"):
                    self._runner.lookup_misses = 0
                self._runner.lookup_misses += 1

        if is_replay and prebuilt_enabled and not prebuilt_available:
            raise RuntimeError(
                "[PREBUILT_FAIL] PREBUILT mode enabled but prebuilt_available=False. "
                "Check logs for [PREBUILT_FAIL] errors during prebuilt loading."
            )

        if prebuilt_available:
            current_ts = candles.index[-1] if len(candles) > 0 else None
            if current_ts is None:
                raise RuntimeError("[PREBUILT_FAIL] Cannot get current timestamp from candles")
            try:
                features_row = self._runner.prebuilt_features_df.loc[current_ts]
                self._runner.lookup_hits += 1
            except KeyError:
                self._runner.lookup_misses += 1
                idx = self._runner.prebuilt_features_df.index
                raise RuntimeError(
                    f"[PREBUILT_LOOKUP_MISS] Timestamp {current_ts} not found in prebuilt features. "
                    f"Prebuilt range: {idx.min()} to {idx.max()}."
                )
            from gx1.execution.live_features import EntryFeatureBundle, compute_atr_bps, infer_vol_bucket
            from gx1.tuning.feature_manifest import align_features, load_manifest
            features_df = features_row.to_frame().T
            features_df.index = [current_ts]
            manifest = load_manifest()
            aligned = align_features(features_df, manifest=manifest, training_stats=manifest.get("training_stats"))
            aligned_last = aligned.tail(1).copy()
            atr_series = compute_atr_bps(candles[["high", "low", "close"]])
            atr_bps = float(atr_series.iloc[-1])
            from gx1.execution.live_features import ADR_WINDOW, PIPS_PER_PERCENT
            recent_window = candles.tail(ADR_WINDOW)
            adr = (recent_window["high"].max() - recent_window["low"].min()) if len(recent_window) >= 2 else np.nan
            adr_bps = (adr / recent_window["close"].iloc[-1]) * PIPS_PER_PERCENT if not np.isnan(adr) else np.nan
            atr_adr_ratio = float(atr_bps / adr_bps) if adr_bps and adr_bps > 0 else np.nan
            vol_bucket = infer_vol_bucket(atr_adr_ratio)
            close_price = float(candles["close"].iloc[-1])
            bid_open = float(candles["bid_open"].iloc[-1]) if "bid_open" in candles.columns else None
            bid_close = float(candles["bid_close"].iloc[-1]) if "bid_close" in candles.columns else None
            ask_open = float(candles["ask_open"].iloc[-1]) if "ask_open" in candles.columns else None
            ask_close = float(candles["ask_close"].iloc[-1]) if "ask_close" in candles.columns else None
            raw_row = candles.iloc[-1]
            entry_bundle = EntryFeatureBundle(
                features=aligned_last,
                raw_row=raw_row,
                close_price=close_price,
                atr_bps=atr_bps,
                vol_bucket=vol_bucket,
                bid_open=bid_open,
                bid_close=bid_close,
                ask_open=ask_open,
                ask_close=ask_close,
            )
            feat_time = 0.0
            if not hasattr(self._runner, "prebuilt_bypass_count"):
                self._runner.prebuilt_bypass_count = 0
            self._runner.prebuilt_bypass_count += 1
            if not self._runner.prebuilt_used:
                raise RuntimeError(
                    "[PREBUILT_FAIL] prebuilt_features_df is available but prebuilt_used=False."
                )
        else:
            feat_start = time.perf_counter()
            entry_bundle = build_live_entry_features(candles)
            feat_time = time.perf_counter() - feat_start

        if hasattr(self._runner, "perf_feat_time"):
            self._runner.perf_feat_time += feat_time
        else:
            self._runner.perf_feat_time = feat_time

        current_atr_bps: Optional[float] = None
        current_atr_pct: Optional[float] = None
        try:
            if entry_bundle.atr_bps is not None:
                current_atr_bps = float(entry_bundle.atr_bps)
        except (TypeError, ValueError):
            current_atr_bps = None
        current_spread_bps: Optional[float] = None
        if hasattr(entry_bundle, "features") and not entry_bundle.features.empty:
            try:
                feat_row = entry_bundle.features.iloc[-1]
                spread_bps_raw = feat_row.get("spread_bps") or feat_row.get("_v1_spread_bps")
                if spread_bps_raw is not None:
                    current_spread_bps = float(spread_bps_raw)
            except (TypeError, ValueError, KeyError):
                pass
        if current_atr_bps is not None and current_atr_bps > 0:
            self.cluster_guard_history.append(current_atr_bps)
            if len(self.cluster_guard_history) >= 25:
                try:
                    self.cluster_guard_atr_median = float(np.median(self.cluster_guard_history))
                except Exception:
                    pass
            current_atr_pct = self._percentile_from_history(self.cluster_guard_history, current_atr_bps)

        if hasattr(self._runner, "big_brain_v1") and self._runner.big_brain_v1 is not None:
            try:
                lookback_bars = getattr(self._runner.big_brain_v1, "lookback", 288)
                has_warmup = (
                    hasattr(self._runner.big_brain_v1, "_warmup_buffer")
                    and self._runner.big_brain_v1._warmup_buffer is not None
                    and not self._runner.big_brain_v1._warmup_buffer.empty
                )
                if has_warmup or len(candles) >= lookback_bars:
                    brain_candles = candles.copy() if has_warmup else candles.tail(lookback_bars).copy()
                    required = ["open", "high", "low", "close", "volume", "atr"]
                    if "atr" not in brain_candles.columns:
                        high_low = brain_candles["high"] - brain_candles["low"]
                        high_close = (brain_candles["high"] - brain_candles["close"].shift(1)).abs()
                        low_close = (brain_candles["low"] - brain_candles["close"].shift(1)).abs()
                        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                        brain_candles["atr"] = tr.rolling(window=14, min_periods=1).mean()
                    brain_out = self._runner.big_brain_v1.infer_from_df(brain_candles)
                    policy_state["brain_trend_regime"] = brain_out.get("trend_regime", "UNKNOWN")
                    policy_state["brain_vol_regime"] = brain_out.get("vol_regime", "UNKNOWN")
                    policy_state["brain_risk_score"] = brain_out.get("brain_risk_score", 0.0)
                else:
                    policy_state["brain_trend_regime"] = "UNKNOWN"
                    policy_state["brain_vol_regime"] = "UNKNOWN"
                    policy_state["brain_risk_score"] = 0.0
            except Exception:
                policy_state["brain_trend_regime"] = "UNKNOWN"
                policy_state["brain_vol_regime"] = "UNKNOWN"
                policy_state["brain_risk_score"] = 0.0
        else:
            policy_state["brain_trend_regime"] = "UNKNOWN"
            policy_state["brain_vol_regime"] = "UNKNOWN"
            policy_state["brain_risk_score"] = 0.0

        if hasattr(self._runner, "replay_mode") and self._runner.replay_mode and candles is not None:
            from gx1.execution.replay_features import ensure_replay_tags
            current_ts = candles.index[-1] if len(candles) > 0 else None
            dummy_row = pd.Series({"ts": current_ts} if current_ts else {})
            dummy_row, policy_state = ensure_replay_tags(
                dummy_row, candles, policy_state, current_ts=current_ts,
            )

        trend = policy_state.get("brain_trend_regime", "UNKNOWN")
        vol = policy_state.get("brain_vol_regime", "UNKNOWN")
        risk_score = policy_state.get("brain_risk_score", 0.0)
        if "session" not in policy_state:
            current_ts_for_session = candles.index[-1]
            policy_state["session"] = infer_session_tag(current_ts_for_session).upper()
        current_session = policy_state["session"]

        stage0_enabled = getattr(self, "stage0_enabled", True)
        debug_force_cfg = self.policy.get("debug_force", {})
        force_enabled = (
            debug_force_cfg.get("enabled", False)
            and self.policy.get("meta", {}).get("role") == "CANARY"
            and hasattr(self, "oanda_env") and self.oanda_env == "practice"
        )
        if force_enabled:
            if not hasattr(self, "_force_entry_start_time"):
                self._force_entry_start_time = time.time()
                self._force_entry_trade_count = 0
            max_trades = debug_force_cfg.get("max_trades", 1)
            if getattr(self, "_force_entry_trade_count", 0) >= max_trades:
                force_enabled = False
            else:
                timeout_minutes = debug_force_cfg.get("force_entry_if_no_trade_after_minutes", 30)
                elapsed = (time.time() - getattr(self, "_force_entry_start_time", time.time())) / 60.0
                if elapsed >= timeout_minutes and getattr(self, "_force_entry_trade_count", 0) == 0:
                    allowed_session = debug_force_cfg.get("allowed_session")
                    if allowed_session is None or allowed_session == "" or current_session.upper() == allowed_session.upper():
                        policy_state["force_entry"] = True
                        policy_state["force_entry_reason"] = f"timeout_after_{timeout_minutes}_minutes"
                        bypass = debug_force_cfg.get("bypass_trend_vol_unknown", False)
                        if bypass and (trend == "UNKNOWN" or vol == "UNKNOWN"):
                            if trend == "UNKNOWN":
                                policy_state["brain_trend_regime"] = "TREND_UP"
                                trend = "TREND_UP"
                            if vol == "UNKNOWN":
                                policy_state["brain_vol_regime"] = "MEDIUM"
                                vol = "MEDIUM"

        if not policy_state.get("force_entry", False) and stage0_enabled:
            if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
                self.entry_feature_telemetry.record_control_flow("BEFORE_STAGE0_CHECK")
            self.stage0_total_considered += 1
            should_consider = self.should_consider_entry(
                trend=trend, vol=vol, session=current_session, risk_score=risk_score
            )
            if not should_consider:
                reason = getattr(self._runner, "_last_stage0_reason", "stage0_unknown")
                self.stage0_reasons[reason] += 1
                if reason == "stage0_session_block":
                    self.veto_pre["veto_pre_session"] += 1
                elif reason == "stage0_vol_block":
                    self.veto_pre["veto_pre_atr"] += 1
                elif reason == "stage0_trend_vol_block":
                    self.veto_pre["veto_pre_regime"] += 1
                return None
            self.entry_telemetry["n_precheck_pass"] += 1
        else:
            if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
                self.entry_feature_telemetry.record_control_flow("BEFORE_STAGE0_CHECK")
            self.entry_telemetry["n_precheck_pass"] += 1

        policy_state["coverage_cutoff_base"] = float(
            self.policy.get("entry_gating", {}).get("coverage", {}).get("target", 0.20)
            or self.entry_params.get("gating", {}).get("coverage", {}).get("target", 0.20)
        )
        self._last_policy_state = policy_state

        if not (hasattr(self._runner, "entry_v10_enabled") and self._runner.entry_v10_enabled):
            self.veto_pre["veto_pre_model_missing"] += 1
            return None

        if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
            v10_enabled = hasattr(self._runner, "entry_v10_enabled") and self._runner.entry_v10_enabled
            self.entry_feature_telemetry.record_v10_enable_state(
                enabled=v10_enabled,
                reason="ENABLED" if v10_enabled else "POLICY_DISABLED",
            )

        v10_supported_sessions = {"EU", "US", "OVERLAP"}
        if current_session not in v10_supported_sessions:
            return None

        if hasattr(self, "entry_feature_telemetry") and self.entry_feature_telemetry:
            self.entry_feature_telemetry.record_entry_routing(
                selected_model="v10_hybrid", reason="V10_ENABLED_AND_SESSION_SUPPORTED"
            )
        self.n_v10_calls += 1

        model_start = time.perf_counter()
        entry_pred = self._runner._predict_entry_v10_hybrid(
            entry_bundle,
            candles,
            policy_state,
            entry_context_features=entry_context_features,
        )
        model_time = time.perf_counter() - model_start
        if hasattr(self._runner, "perf_model_time"):
            self._runner.perf_model_time += model_time
        else:
            self._runner.perf_model_time = model_time

        if entry_pred is not None and np.isfinite(entry_pred.prob_long) and np.isfinite(entry_pred.prob_short):
            self.n_v10_pred_ok += 1
            self.entry_telemetry["n_candidates"] += 1
            self.entry_telemetry["n_predictions"] += 1
            self.entry_telemetry["p_long_values"].append(float(entry_pred.prob_long))
            session_key = current_session
            self.entry_telemetry["candidate_sessions"][session_key] = (
                self.entry_telemetry["candidate_sessions"].get(session_key, 0) + 1
            )
        else:
            self.n_v10_pred_none_or_nan += 1

        if entry_pred is None:
            return None

        self.killchain_n_entry_pred_total += 1
        policy_cfg = self.policy.get("entry_policy_sniper_v10_ctx", {})
        killchain_min_prob_long = float(policy_cfg.get("min_prob_long", 0.67))
        killchain_min_prob_short = float(policy_cfg.get("min_prob_short", 0.72))
        killchain_allow_short = bool(policy_cfg.get("allow_short", False))
        if os_module.getenv("GX1_ANALYSIS_MODE") == "1" and os_module.getenv("GX1_ENTRY_THRESHOLD_OVERRIDE"):
            try:
                killchain_min_prob_long = float(os_module.getenv("GX1_ENTRY_THRESHOLD_OVERRIDE"))
                killchain_min_prob_short = killchain_min_prob_long
            except (ValueError, TypeError):
                pass
        killchain_above_threshold = bool(
            (float(entry_pred.prob_long) >= killchain_min_prob_long)
            or (killchain_allow_short and float(entry_pred.prob_short) >= killchain_min_prob_short)
        )
        if killchain_above_threshold:
            self.killchain_n_above_threshold += 1

        current_row = entry_bundle.features.iloc[-1:].copy()
        current_row["prob_long"] = entry_pred.prob_long
        current_row["prob_short"] = entry_pred.prob_short
        if "close" not in current_row.columns:
            current_row["close"] = entry_bundle.close_price
        if "_v1_atr14" not in current_row.columns and hasattr(entry_bundle.raw_row, "index") and "_v1_atr14" in entry_bundle.raw_row.index:
            current_row["_v1_atr14"] = entry_bundle.raw_row["_v1_atr14"]
        if isinstance(current_row.index, pd.DatetimeIndex):
            current_row = current_row.reset_index()
            if len(current_row.columns) > 0:
                current_row = current_row.rename(columns={current_row.columns[0]: "ts"})

        if hasattr(self, "replay_mode") and self.replay_mode and candles is not None:
            needs_tags = (
                policy_state.get("brain_trend_regime", "UNKNOWN") == "UNKNOWN"
                or policy_state.get("brain_vol_regime", "UNKNOWN") == "UNKNOWN"
                or policy_state.get("session") == "UNKNOWN"
            )
            if needs_tags:
                from gx1.execution.replay_features import ensure_replay_tags
                current_ts = candles.index[-1] if len(candles) > 0 else None
                current_row_for_tags = current_row.iloc[0] if isinstance(current_row, pd.DataFrame) and len(current_row) > 0 else current_row
                current_row_for_tags, policy_state = ensure_replay_tags(
                    current_row_for_tags, candles, policy_state, current_ts=current_ts,
                )
                if isinstance(current_row, pd.DataFrame) and len(current_row) > 0:
                    for col in ["session", "vol_regime", "atr_regime", "trend_regime", "_v1_atr_regime_id"]:
                        if col in current_row_for_tags.index:
                            current_row.loc[current_row.index[0], col] = current_row_for_tags[col]
                    if "session" in policy_state:
                        current_row.loc[current_row.index[0], "session"] = policy_state["session"]
        if isinstance(current_row, pd.DataFrame) and len(current_row) > 0:
            sess_val = current_row["session"].iloc[0] if "session" in current_row.columns else None
            if sess_val is None or (isinstance(sess_val, str) and sess_val == "UNKNOWN") or (not isinstance(sess_val, str) and pd.isna(sess_val)):
                if "session" in policy_state and policy_state["session"] != "UNKNOWN":
                    current_row.loc[current_row.index[0], "session"] = policy_state["session"]
                else:
                    current_ts = candles.index[-1] if candles is not None and len(candles) > 0 else None
                    if current_ts is not None:
                        tag = infer_session_tag(current_ts)
                        current_row.loc[current_row.index[0], "session"] = tag
                        policy_state["session"] = tag

        allow_high_vol = policy_cfg.get("allow_high_vol", True)
        allow_extreme_vol = policy_cfg.get("allow_extreme_vol", False)
        try:
            if killchain_above_threshold:
                from gx1.policy.farm_guards import _extract_session_vol_regime
                kc_session, kc_vol_regime = _extract_session_vol_regime(current_row.iloc[0])
                if kc_session in ["EU", "OVERLAP", "US"]:
                    self.killchain_n_after_session_guard += 1
                allowed_vol = ["LOW", "MEDIUM"]
                if allow_high_vol:
                    allowed_vol.append("HIGH")
                if allow_extreme_vol:
                    allowed_vol.append("EXTREME")
                if kc_vol_regime in allowed_vol:
                    self.killchain_n_after_vol_guard += 1
            from gx1.policy.farm_guards import sniper_guard_v1
            sniper_guard_v1(
                current_row.iloc[0],
                context="live_runner_pre_policy_sniper",
                allow_high_vol=allow_high_vol,
                allow_extreme_vol=allow_extreme_vol,
            )
        except AssertionError as e:
            error_str = str(e)
            if "vol_regime=UNKNOWN" in error_str:
                self.entry_telemetry["vol_regime_unknown_count"] += 1
            if "session=" in error_str and "not in" in error_str:
                self._killchain_inc_reason("BLOCK_SESSION")
            elif "vol_regime=" in error_str and "not in" in error_str:
                self._killchain_inc_reason("BLOCK_VOL")
            else:
                self._killchain_inc_reason("BLOCK_UNKNOWN")
                try:
                    current_ts = candles.index[-1] if candles is not None and len(candles) > 0 else None
                    self._killchain_record_unknown({
                        "where": "SESSION_VOL_GUARD",
                        "ts": current_ts.isoformat() if current_ts is not None else None,
                        "error": error_str[:300],
                    })
                except Exception:
                    pass
            return None

        risk_guard_blocked = False
        risk_guard_reason = None
        risk_guard_details = {}
        risk_guard_clamp = None
        if self._sniper_risk_guard is None:
            risk_guard_cfg_path = self.policy.get("risk_guard", {}).get("config_path")
            if risk_guard_cfg_path:
                try:
                    import yaml
                    from pathlib import Path
                    from gx1.policy.sniper_risk_guard import SniperRiskGuardV1
                    guard_cfg_path = Path(risk_guard_cfg_path)
                    if not guard_cfg_path.is_absolute():
                        guard_cfg_path = Path(__file__).resolve().parent.parent.parent / guard_cfg_path
                    if guard_cfg_path.exists():
                        with open(guard_cfg_path) as f:
                            self._sniper_risk_guard = SniperRiskGuardV1(yaml.safe_load(f))
                except Exception:
                    pass
        if self._sniper_risk_guard is not None and getattr(self._sniper_risk_guard, "enabled", False):
            entry_snapshot = {
                "session": current_row["session"].iloc[0] if "session" in current_row.columns and len(current_row) > 0 else None,
                "vol_regime": current_row["vol_regime"].iloc[0] if "vol_regime" in current_row.columns and len(current_row) > 0 else None,
                "spread_bps": current_row.get("spread_bps", [None])[0] if "spread_bps" in current_row.columns and len(current_row) > 0 else None,
                "atr_bps": current_row.get("atr_bps", [None])[0] if "atr_bps" in current_row.columns and len(current_row) > 0 else None,
            }
            feature_context = {"spread_bps": getattr(entry_bundle, "spread_bps", None), "atr_bps": current_atr_bps}
            should_block, reason_code, details = self._sniper_risk_guard.should_block(
                entry_snapshot, feature_context, policy_state, len(candles) if candles is not None else 0,
            )
            if should_block:
                risk_guard_blocked = True
                risk_guard_reason = reason_code
                risk_guard_details = details
                if "cooldown" in str(reason_code).lower():
                    self._killchain_inc_reason("BLOCK_COOLDOWN")
                else:
                    self._killchain_inc_reason("BLOCK_RISK")
                if hasattr(self, "trade_journal") and self.trade_journal and hasattr(self.trade_journal, "log_entry_attempt"):
                    try:
                        self.trade_journal.log_entry_attempt(
                            entry_time=candles.index[-1].isoformat() if candles is not None and len(candles) > 0 else None,
                            reason=reason_code,
                            details=details,
                        )
                    except Exception:
                        pass
                return None
            session = entry_snapshot.get("session") or policy_state.get("session")
            clamp = self._sniper_risk_guard.get_session_clamp(session)
            if clamp is not None and clamp > 0:
                risk_guard_clamp = clamp
                policy_state["risk_guard_min_prob_long_clamp"] = clamp
            policy_state["risk_guard_blocked"] = risk_guard_blocked
            policy_state["risk_guard_reason"] = risk_guard_reason
            policy_state["risk_guard_details"] = risk_guard_details
            policy_state["risk_guard_clamp"] = risk_guard_clamp

        meta_model = getattr(self, "farm_entry_meta_model", None)
        meta_feature_cols = getattr(self, "farm_entry_meta_feature_cols", None)
        replay_config = self.policy.get("replay_config", {})
        policy_module = replay_config.get("policy_module", "gx1.policy.entry_policy_sniper_v10_ctx")
        if hasattr(self._runner, "replay_mode") and self._runner.replay_mode:
            if not policy_module:
                raise RuntimeError(
                    "REPLAY_CONFIG_REQUIRED: replay_config.policy_module is missing in policy YAML."
                )
            if policy_module != "gx1.policy.entry_policy_sniper_v10_ctx":
                raise RuntimeError(
                    f"REPLAY_POLICY_FORBIDDEN: replay_config.policy_module must be "
                    f"'gx1.policy.entry_policy_sniper_v10_ctx', got '{policy_module}'."
                )
        from gx1.policy.entry_policy_sniper_v10_ctx import apply_entry_policy_sniper_v10_ctx
        apply_policy_fn = apply_entry_policy_sniper_v10_ctx
        policy_flag_col_name = replay_config.get("policy_id", "entry_policy_sniper_v10_ctx")

        min_prob_long = float(policy_cfg.get("min_prob_long", 0.67))
        min_prob_short = float(policy_cfg.get("min_prob_short", 0.72))
        allow_short = policy_cfg.get("allow_short", False)
        if os_module.getenv("GX1_ANALYSIS_MODE") == "1" and os_module.getenv("GX1_ENTRY_THRESHOLD_OVERRIDE"):
            try:
                min_prob_long = float(os_module.getenv("GX1_ENTRY_THRESHOLD_OVERRIDE"))
                min_prob_short = min_prob_long
            except (ValueError, TypeError):
                pass
        self.threshold_used = f"long={min_prob_long},short={min_prob_short}" if allow_short else f"long={min_prob_long}"

        df_policy = apply_policy_fn(
            current_row,
            self.policy,
            meta_model=meta_model,
            meta_feature_cols=meta_feature_cols,
        )
        policy_flag_col = policy_flag_col_name
        if policy_flag_col not in df_policy.columns and len(df_policy.columns) > 0:
            for c in df_policy.columns:
                if "policy" in c.lower() and "sniper" in c.lower():
                    policy_flag_col = c
                    break

        if len(df_policy) > 0 and policy_flag_col in df_policy.columns and df_policy[policy_flag_col].sum() > 0:
            accepted_row = df_policy[df_policy[policy_flag_col]].iloc[0]
            policy_state["p_long"] = accepted_row.get("p_long", entry_pred.prob_long)
            policy_state["p_profitable"] = accepted_row.get("p_profitable", None)
            if "_policy_side" in accepted_row:
                policy_state["_policy_side"] = accepted_row["_policy_side"]
        else:
            self.veto_cand["veto_cand_threshold"] += 1
            self._killchain_inc_reason("BLOCK_BELOW_THRESHOLD")
            policy_state["p_long"] = df_policy["p_long"].iloc[0] if len(df_policy) > 0 and "p_long" in df_policy.columns else entry_pred.prob_long
            policy_state["p_profitable"] = df_policy["p_profitable"].iloc[0] if len(df_policy) > 0 and "p_profitable" in df_policy.columns else None
            return None
        if df_policy[policy_flag_col].sum() == 0:
            self._killchain_inc_reason("BLOCK_BELOW_THRESHOLD")
            return None

        policy_state["policy_name"] = "V10_CTX"
        policy_state["entry_model_active"] = "ENTRY_V10"
        policy_state["p_long_v10_1"] = entry_pred.prob_long
        prediction = entry_pred
        self._last_policy_state = policy_state

        entry_gating = self.policy.get("entry_gating", None)
        gates_cfg = entry_gating or {}
        now_ts = candles.index[-1] if len(candles) > 0 else pd.Timestamp.now(tz="UTC")
        side = "long" if prediction.prob_long >= prediction.prob_short else "short"
        T = 1.0
        ratio_pre = float(prediction.prob_long / max(1e-8, prediction.prob_short)) if side == "long" else float(prediction.prob_short / max(1e-8, prediction.prob_long))
        side_pre = "long" if prediction.prob_long >= prediction.prob_short else "short"
        side_post = side.upper()

        if self.mode == "ENTRY_ONLY":
            self._log_entry_only_event(
                timestamp=now_ts,
                side=side,
                price=entry_bundle.close_price,
                prediction=prediction,
                policy_state=policy_state,
            )
            return None

        if not hasattr(self, "_next_trade_id"):
            self._next_trade_id = 0
        self._next_trade_id += 1
        trade_id = f"SIM-{int(time.time())}-{self._next_trade_id:06d}"
        run_id = getattr(self._runner, "run_id", "unknown")
        chunk_id = getattr(self._runner, "chunk_id", "single")
        local_seq = self._next_trade_id
        uuid_short = uuid.uuid4().hex[:12]
        trade_uid = f"{run_id}:{chunk_id}::{local_seq:06d}:{uuid_short}"
        if self.is_replay:
            expected_prefix = f"{run_id}:{chunk_id}::"
            if not trade_uid.startswith(expected_prefix):
                raise RuntimeError(
                    f"BAD_TRADE_UID_FORMAT_REPLAY: Generated trade_uid={trade_uid} does not start with "
                    f"expected prefix={expected_prefix}. run_id={run_id}, chunk_id={chunk_id}."
                )
        if hasattr(self, "_force_entry_trade_count"):
            self._force_entry_trade_count += 1

        current_bar = candles.iloc[-1]
        try:
            entry_bid_price = float(current_bar["bid_close"])
            entry_ask_price = float(current_bar["ask_close"])
        except KeyError as exc:
            raise ValueError(
                "Bid/ask required for replay, but missing in candles during entry creation."
            ) from exc
        entry_price = entry_ask_price if side == "long" else entry_bid_price
        base_units = self.exec.default_units if side == "long" else -self.exec.default_units

        policy_state_snapshot = self._last_policy_state or {}
        spread_bps_for_overlay = None
        spread_pct_for_overlay = None
        try:
            spread_raw = float(entry_ask_price) - float(entry_bid_price)
            if spread_raw > 0:
                spread_bps_for_overlay = spread_raw * 10000.0
                if hasattr(self, "spread_history") and len(self.spread_history) > 0:
                    try:
                        spread_pct_for_overlay = self._percentile_from_history(self.spread_history, spread_bps_for_overlay)
                    except Exception:
                        pass
        except (TypeError, ValueError):
            pass
        feature_context_dict = {}
        if current_atr_bps is not None:
            feature_context_dict["atr_bps"] = current_atr_bps
        if spread_bps_for_overlay is not None:
            feature_context_dict["spread_bps"] = spread_bps_for_overlay
        if spread_pct_for_overlay is not None:
            feature_context_dict["spread_pct"] = spread_pct_for_overlay
        if entry_bundle is not None:
            if hasattr(entry_bundle, "atr_bps") and entry_bundle.atr_bps is not None:
                feature_context_dict.setdefault("atr_bps", entry_bundle.atr_bps)
            if hasattr(entry_bundle, "spread_bps") and entry_bundle.spread_bps is not None:
                feature_context_dict.setdefault("spread_bps", entry_bundle.spread_bps)
        regime_inputs = get_runtime_regime_inputs(
            prediction=prediction,
            feature_context=feature_context_dict if feature_context_dict else None,
            spread_pct=spread_pct_for_overlay,
            current_atr_bps=current_atr_bps,
            entry_bundle=entry_bundle,
            policy_state=policy_state_snapshot,
            entry_time=now_ts,
        )
        _trend_regime = regime_inputs["trend_regime"]
        _vol_regime = regime_inputs["vol_regime"]
        _atr_bps = regime_inputs["atr_bps"]
        _spread_bps = regime_inputs["spread_bps"]
        _session = regime_inputs["session"]

        overlays_meta = []
        units_current = base_units
        regime_overlay_cfg = self.policy.get("sniper_regime_size_overlay", {}) or {}
        try:
            units_1, meta_1 = apply_size_overlay(
                base_units=units_current, entry_time=now_ts,
                trend_regime=_trend_regime, vol_regime=_vol_regime, atr_bps=_atr_bps, spread_bps=_spread_bps, session=_session, cfg=regime_overlay_cfg,
            )
        except Exception as e:
            units_1 = units_current
            meta_1 = {"overlay_applied": False, "overlay_name": "Q4_B_MIXED_SIZE", "reason": f"overlay_error:{type(e).__name__}", "error": str(e)}
        overlays_meta.append(meta_1)
        units_current = units_1
        for overlay_name, apply_fn, cfg_key in [
            ("Q4_C_CHOP_SESSION_SIZE", apply_q4_cchop_overlay, "sniper_q4_cchop_overlay"),
            ("Q4_A_TREND_SIZE", apply_q4_atrend_overlay, "sniper_q4_atrend_overlay"),
            ("EU_TIMING_SIZE", apply_q4_eu_timing_overlay, "sniper_q4_eu_timing_overlay"),
        ]:
            cfg = self.policy.get(cfg_key, {}) or {}
            try:
                u, meta = apply_fn(
                    base_units=units_current, entry_time=now_ts,
                    trend_regime=_trend_regime, vol_regime=_vol_regime, atr_bps=_atr_bps, spread_bps=_spread_bps, session=_session, cfg=cfg,
                )
            except Exception as e:
                u = units_current
                meta = {"overlay_applied": False, "overlay_name": overlay_name, "reason": f"overlay_error:{type(e).__name__}", "error": str(e)}
            overlays_meta.append(meta)
            units_current = u

        if ENTRY_V10_1_SIZE_OVERLAY_AVAILABLE:
            if not hasattr(self, "_entry_v10_1_size_overlay"):
                v10_1_cfg = {}
                entry_config_path = self.policy.get("entry_config")
                if entry_config_path:
                    try:
                        import yaml
                        from pathlib import Path
                        ec = yaml.safe_load(Path(entry_config_path).read_text()) or {}
                        v10_1_cfg = ec.get("entry_v10_1_size_overlay", {}) or {}
                    except Exception:
                        v10_1_cfg = {}
                if v10_1_cfg.get("enabled", False):
                    try:
                        self._entry_v10_1_size_overlay = load_entry_v10_1_size_overlay(v10_1_cfg)
                    except Exception:
                        self._entry_v10_1_size_overlay = None
                else:
                    self._entry_v10_1_size_overlay = None
            if getattr(self, "_entry_v10_1_size_overlay", None) is not None:
                try:
                    p_long_v10_1 = policy_state.get("p_long_v10_1") or policy_state.get("p_long")
                    if p_long_v10_1 is not None:
                        trend_name = "UP" if _trend_regime == "TREND_UP" else ("DOWN" if _trend_regime == "TREND_DOWN" else "NEUTRAL")
                        vol_name = _vol_regime.upper() if isinstance(_vol_regime, str) else "LOW"
                        regime_str = f"{trend_name}×{vol_name}"
                        sl_bps = abs(int(self.tick_cfg.get("sl_bps", 100)))
                        units_current, meta_5 = self._entry_v10_1_size_overlay.apply_overlay(
                            base_units=units_current, p_long_v10_1=float(p_long_v10_1), session=_session, regime=regime_str, sl_bps=sl_bps,
                        )
                        overlays_meta.append(meta_5)
                except Exception:
                    overlays_meta.append({"overlay_applied": False, "overlay_name": "ENTRY_V10_1_SIZE", "reason": "error"})
        units_out = units_current

        if units_out == 0:
            self._killchain_inc_reason("BLOCK_RISK")
            return None
        self.killchain_n_after_risk_sizing += 1

        self.entry_telemetry["n_candidate_pass"] += 1
        self.killchain_n_trade_create_attempts += 1
        from gx1.execution.oanda_demo_runner import LiveTrade
        trade = LiveTrade(
            trade_id=trade_id,
            trade_uid=trade_uid,
            entry_time=now_ts,
            side=side,
            units=units_out,
            entry_price=entry_price,
            entry_bid=entry_bid_price,
            entry_ask=entry_ask_price,
            atr_bps=entry_bundle.atr_bps,
            vol_bucket=entry_bundle.vol_bucket,
            entry_prob_long=prediction.prob_long,
            entry_prob_short=prediction.prob_short,
            dry_run=self.exec.dry_run,
        )
        self.entry_telemetry["n_trades_created"] += 1
        self.killchain_n_trade_created += 1
        session_key = policy_state.get("session") or infer_session_tag(trade.entry_time).upper()
        self.entry_telemetry["trade_sessions"][session_key] = self.entry_telemetry["trade_sessions"].get(session_key, 0) + 1

        if not hasattr(trade, "extra"):
            trade.extra = {}
        trade.extra["tp_bps"] = int(self.tick_cfg.get("tp_bps", 180))
        trade.extra["sl_bps"] = int(self.tick_cfg.get("sl_bps", 100))
        trade.extra["be_trigger_bps"] = int(self.tick_cfg.get("be_trigger_bps", 50))
        trade.extra["be_active"] = False
        trade.extra["be_price"] = None

        range_window = 96
        range_pos, distance_to_range = self._compute_range_features(candles, window=range_window)
        trade.extra["range_pos"] = float(range_pos)
        trade.extra["distance_to_range"] = float(distance_to_range)
        atr_value = (current_atr_bps / 10000.0) * entry_price if current_atr_bps and current_atr_bps > 0 and entry_price > 0 else None
        recent = candles.iloc[-(range_window+1):-1] if len(candles) >= range_window + 1 else candles.tail(range_window)
        has_direct = all(col in candles.columns for col in ["high", "low", "close"])
        has_bid_ask = all(col in candles.columns for col in ["bid_high", "ask_high", "bid_low", "ask_low", "bid_close", "ask_close"])
        range_hi = range_lo = price_ref = None
        if has_direct or has_bid_ask:
            if has_direct:
                high_vals, low_vals = recent["high"].values, recent["low"].values
                close_vals = recent["close"].values
            else:
                high_vals = (recent["bid_high"].values + recent["ask_high"].values) / 2.0
                low_vals = (recent["bid_low"].values + recent["ask_low"].values) / 2.0
                close_vals = (recent["bid_close"].values + recent["ask_close"].values) / 2.0
            range_hi = float(np.max(high_vals))
            range_lo = float(np.min(low_vals))
            price_ref = float(close_vals[-1])
        range_edge_dist_atr = self._compute_range_edge_dist_atr(
            candles=candles, price_ref=price_ref or entry_price, range_hi=range_hi or entry_price, range_lo=range_lo or 0.0, atr_value=atr_value, window=range_window,
        ) if range_hi is not None and range_lo is not None else 0.0
        trade.extra["range_edge_dist_atr"] = float(range_edge_dist_atr)

        spread_bps = None
        spread_pct = None
        try:
            spread_raw = float(entry_ask_price) - float(entry_bid_price)
            if spread_raw > 0:
                spread_bps = spread_raw * 10000.0
        except (TypeError, ValueError):
            pass
        if spread_bps is not None and np.isfinite(spread_bps):
            spread_pct = self._percentile_from_history(self.spread_history, spread_bps)
            self.spread_history.append(spread_bps)

        if getattr(self, "exit_hybrid_enabled", False) and getattr(self, "exit_mode_selector", None):
            session_for_exit = policy_state_snapshot.get("session") or infer_session_tag(trade.entry_time).upper()
            farm_regime = policy_state_snapshot.get("farm_regime") or policy_state_snapshot.get("_farm_regime") or "UNKNOWN"
            selected_profile = self.exit_mode_selector.choose_exit_profile(
                atr_bps=current_atr_bps or trade.atr_bps,
                atr_pct=current_atr_pct,
                spread_bps=spread_bps,
                spread_pct=spread_pct,
                session=session_for_exit,
                regime=farm_regime,
                range_pos=range_pos,
                distance_to_range=distance_to_range,
                range_edge_dist_atr=range_edge_dist_atr,
            )
            trade.extra["exit_profile"] = selected_profile
            trade.extra.setdefault("exit_hybrid", {})
            trade.extra["exit_hybrid"].update({
                "mode": getattr(self, "exit_hybrid_mode", "RULE5_RULE6A_ATR_SPREAD_V1"),
                "atr_bps": current_atr_bps, "atr_pct": current_atr_pct,
                "spread_bps": spread_bps, "spread_pct": spread_pct,
                "session": session_for_exit, "regime": farm_regime,
            })

        self._ensure_exit_profile(trade, context="entry_manager")
        if self.exit_config_name and not (getattr(trade, "extra", {}) or {}).get("exit_profile"):
            raise RuntimeError(
                f"[EXIT_PROFILE] Trade created without exit_profile under exit-config {self.exit_config_name}: {trade.trade_id}"
            )
        trade.client_order_id = self._generate_client_order_id(trade.entry_time, trade.entry_price, trade.side)

        if hasattr(self._runner, "trade_journal") and self._runner.trade_journal:
            try:
                from gx1.monitoring.trade_journal import EVENT_ENTRY_SIGNAL
                gates = {}
                if spread_bps is not None:
                    gates["spread_bps"] = {"value": spread_bps, "passed": True}
                if current_atr_bps is not None:
                    gates["atr_bps"] = {"value": current_atr_bps, "passed": True}
                if policy_state_snapshot:
                    gates["regime"] = {"value": policy_state_snapshot.get("farm_regime", "UNKNOWN"), "passed": True}
                    gates["session"] = {"value": policy_state_snapshot.get("session", "UNKNOWN"), "passed": True}
                features_snapshot = {
                    "atr_bps": current_atr_bps, "atr_pct": current_atr_pct,
                    "spread_bps": spread_bps, "spread_pct": spread_pct,
                    "range_pos": float(range_pos), "distance_to_range": float(distance_to_range),
                    "range_edge_dist_atr": float(range_edge_dist_atr),
                }
                self._runner.trade_journal.log(
                    EVENT_ENTRY_SIGNAL,
                    {
                        "entry_time": trade.entry_time.isoformat() if hasattr(trade.entry_time, "isoformat") else str(trade.entry_time),
                        "entry_price": trade.entry_price,
                        "side": trade.side,
                        "entry_model_outputs": {
                            "p_long": prediction.prob_long, "p_short": prediction.prob_short,
                            "p_hat": prediction.p_hat, "margin": prediction.margin, "session": prediction.session,
                        },
                        "gates": gates,
                        "features_snapshot": features_snapshot,
                        "warmup_state": {"bars_since_start": len(candles) if candles is not None else None},
                    },
                    trade_key={
                        "entry_time": trade.entry_time.isoformat() if hasattr(trade.entry_time, "isoformat") else str(trade.entry_time),
                        "entry_price": trade.entry_price,
                        "side": trade.side,
                    },
                    trade_id=trade.trade_id,
                )
            except Exception as e:
                log.warning("[TRADE_JOURNAL] Failed to log ENTRY_SIGNAL: %s", e)

        if hasattr(self._runner, "trade_journal") and self._runner.trade_journal:
            try:
                entry_filters_passed = []
                entry_filters_blocked = []
                if spread_bps is not None and spread_pct is not None:
                    entry_filters_passed.append("spread_ok")
                if current_atr_bps is not None and current_atr_pct is not None:
                    entry_filters_passed.append("atr_ok")
                if policy_state_snapshot:
                    if policy_state_snapshot.get("farm_regime"):
                        entry_filters_passed.append("regime_ok")
                    if policy_state_snapshot.get("session"):
                        entry_filters_passed.append("session_ok")
                entry_time_iso = trade.entry_time.isoformat() if hasattr(trade.entry_time, "isoformat") else str(trade.entry_time)
                instrument_val = getattr(self._runner, "instrument", "XAU_USD") if hasattr(self._runner, "instrument") else "XAU_USD"
                model_name_val = getattr(self._runner, "model_name", None) if hasattr(self._runner, "model_name") else None
                is_force_entry = policy_state_snapshot.get("force_entry", False) if policy_state_snapshot else False
                test_mode = is_force_entry
                reason = "FORCED_CANARY_TRADE" if is_force_entry else None
                warmup_degraded = getattr(self._runner, "_warmup_degraded", False)
                cached_bars_at_entry = getattr(self._runner, "_cached_bars_at_startup", None)
                warmup_bars_required = self._runner.policy.get("warmup_bars", 288) if hasattr(self._runner, "policy") else None
                vol_regime = (policy_state_snapshot.get("brain_vol_regime") or policy_state_snapshot.get("vol_regime")) if policy_state_snapshot else None
                trend_regime = (policy_state_snapshot.get("brain_trend_regime") or policy_state_snapshot.get("trend_regime")) if policy_state_snapshot else None
                journal_start = time.perf_counter()
                try:
                    self._runner.trade_journal.log_entry_snapshot(
                        entry_time=entry_time_iso,
                        trade_uid=trade.trade_uid,
                        trade_id=trade.trade_id,
                        instrument=instrument_val,
                        side=trade.side,
                        entry_price=trade.entry_price,
                        units=units_out,
                        base_units=base_units,
                        session=regime_inputs["session"],
                        regime=policy_state_snapshot.get("farm_regime") if policy_state_snapshot else None,
                        vol_regime=regime_inputs["vol_regime"],
                        trend_regime=regime_inputs["trend_regime"],
                        entry_model_version=model_name_val,
                        entry_score={
                            "p_long": prediction.prob_long, "p_short": prediction.prob_short,
                            "p_hat": prediction.p_hat, "margin": prediction.margin,
                        },
                        entry_filters_passed=entry_filters_passed,
                        entry_filters_blocked=entry_filters_blocked,
                        test_mode=test_mode,
                        reason=reason,
                        warmup_degraded=warmup_degraded,
                        cached_bars_at_entry=cached_bars_at_entry,
                        warmup_bars_required=warmup_bars_required,
                        risk_guard_blocked=risk_guard_blocked,
                        risk_guard_reason=risk_guard_reason,
                        risk_guard_details=risk_guard_details,
                        risk_guard_min_prob_long_clamp=risk_guard_clamp,
                        sniper_overlays=overlays_meta,
                        sniper_overlay=overlays_meta[-1] if overlays_meta else None,
                        atr_bps=regime_inputs["atr_bps"],
                        spread_bps=regime_inputs["spread_bps"],
                        entry_critic=None,
                    )
                    self.entry_telemetry["n_entry_snapshots_written"] += 1
                except Exception as e:
                    self.entry_telemetry["n_entry_snapshots_failed"] += 1
                    if self.is_replay:
                        raise RuntimeError(
                            f"ENTRY_SNAPSHOT_MISSING: Failed to log entry_snapshot for trade_uid={trade.trade_uid}, "
                            f"trade_id={trade.trade_id}. This is a hard contract violation in replay mode. Error: {e}"
                        ) from e
                    log.error(
                        "[TRADE_JOURNAL] Failed to log structured entry snapshot for trade_uid=%s, trade_id=%s: %s",
                        trade.trade_uid, trade.trade_id, e, exc_info=True,
                    )
                    if hasattr(self._runner, "data_integrity_degraded"):
                        self._runner.data_integrity_degraded = True
                atr_value_price = (current_atr_bps / 10000.0) * trade.entry_price if current_atr_bps and current_atr_bps > 0 and trade.entry_price > 0 else None
                candle_close = candle_high = candle_low = None
                if candles is not None and len(candles) > 0:
                    last_bar = candles.iloc[-1]
                    if "close" in candles.columns:
                        candle_close = float(last_bar["close"])
                        candle_high = float(last_bar["high"])
                        candle_low = float(last_bar["low"])
                    elif "bid_close" in candles.columns and "ask_close" in candles.columns:
                        candle_close = float((last_bar["bid_close"] + last_bar["ask_close"]) / 2.0)
                        candle_high = float((last_bar["bid_high"] + last_bar["ask_high"]) / 2.0) if "bid_high" in candles.columns else None
                        candle_low = float((last_bar["bid_low"] + last_bar["ask_low"]) / 2.0) if "bid_low" in candles.columns else None
                spread_value_price = float(entry_ask_price) - float(entry_bid_price) if entry_ask_price and entry_bid_price else None
                self._runner.trade_journal.log_feature_context(
                    trade_uid=trade.trade_uid,
                    trade_id=trade.trade_id,
                    atr_bps=current_atr_bps,
                    atr_price=atr_value_price,
                    atr_percentile=current_atr_pct,
                    range_pos=float(range_pos),
                    distance_to_range=float(distance_to_range),
                    range_edge_dist_atr=float(range_edge_dist_atr),
                    spread_price=spread_value_price,
                    spread_pct=spread_pct,
                    candle_close=candle_close,
                    candle_high=candle_high,
                    candle_low=candle_low,
                )
                if hasattr(self._runner, "perf_journal_time"):
                    self._runner.perf_journal_time += time.perf_counter() - journal_start
                else:
                    self._runner.perf_journal_time = time.perf_counter() - journal_start
            except Exception as e:
                log.warning("[TRADE_JOURNAL] Failed to log structured entry snapshot: %s", e)

        if hasattr(self, "eval_log_path") and self.eval_log_path and entry_pred is not None:
            try:
                now_ts_utc = now_ts
                r5 = r8 = atr_z = 0.0
                ema5_pct = ema20_pct = 0.0
                if hasattr(entry_bundle, "features") and not entry_bundle.features.empty:
                    feat_row = entry_bundle.features.iloc[-1]
                    if "_v1_r5" in entry_bundle.features.columns:
                        r5 = float(feat_row["_v1_r5"]) if not pd.isna(feat_row.get("_v1_r5")) else 0.0
                    if "_v1_r8" in entry_bundle.features.columns:
                        r8 = float(feat_row["_v1_r8"]) if not pd.isna(feat_row.get("_v1_r8")) else 0.0
                eval_record = {
                    "ts_utc": now_ts_utc.isoformat(),
                    "session": prediction.session,
                    "p_long": float(prediction.prob_long),
                    "p_short": float(prediction.prob_short),
                    "p_hat": float(prediction.p_hat),
                    "margin": float(prediction.margin),
                    "T": float(T),
                    "side_pre": side_pre,
                    "ratio_pre": float(ratio_pre),
                    "decision": side_post,
                    "r5": float(r5),
                    "r8": float(r8),
                    "atr_z": float(atr_z),
                    "ema5_pct": float(ema5_pct),
                    "ema20_pct": float(ema20_pct),
                    "ratio": float(ratio_pre),
                    "price": float(entry_bundle.close_price),
                    "units": int(self.exec.default_units if side == "long" else (-self.exec.default_units if side == "short" else 0)),
                }
                from gx1.execution.oanda_demo_runner import append_eval_log
                append_eval_log(self.eval_log_path, eval_record)
            except Exception as e:
                log.debug("Failed to append eval_log: %s", e)

        if getattr(self, "parity_enabled", False) and not getattr(self, "_parity_disabled", False):
            if not hasattr(self, "parity_sample_counter"):
                self.parity_sample_counter = 0
            if not hasattr(self, "parity_sample_every_n"):
                self.parity_sample_every_n = 10
            self.parity_sample_counter += 1
            if self.parity_sample_counter % self.parity_sample_every_n == 0 and hasattr(self, "parity_log_path") and self.parity_log_path:
                try:
                    dir_live = side.upper()
                    parity_record = {
                        "ts_utc": now_ts.isoformat(),
                        "session": prediction.session,
                        "p_hat_live": float(prediction.p_hat),
                        "p_hat_off": None,
                        "margin_live": float(prediction.margin),
                        "margin_off": None,
                        "dir_live": dir_live,
                        "dir_off": dir_live,
                        "abs_err_p_hat": None,
                        "abs_err_margin": None,
                        "dir_match": True,
                    }
                    from gx1.execution.oanda_demo_runner import append_eval_log
                    append_eval_log(self.parity_log_path, parity_record)
                except Exception as e:
                    log.debug("Parity log skipped: %s", e)

        trade.extra["atr_regime"] = policy_state_snapshot.get("brain_vol_regime") or "UNKNOWN"
        trade.extra["big_brain_v1"] = {
            "brain_trend_regime": policy_state_snapshot.get("brain_trend_regime", "UNKNOWN"),
            "brain_vol_regime": policy_state_snapshot.get("brain_vol_regime", "UNKNOWN"),
            "brain_risk_score": policy_state_snapshot.get("brain_risk_score", 0.0),
        }
        trade.extra["brain_trend_regime"] = policy_state_snapshot.get("brain_trend_regime", "UNKNOWN")
        trade.extra["brain_vol_regime"] = policy_state_snapshot.get("brain_vol_regime", "UNKNOWN")
        trade.extra["brain_risk_score"] = policy_state_snapshot.get("brain_risk_score", 0.0)
        trade.extra["vol_regime_entry"] = policy_state_snapshot.get("brain_vol_regime") or "UNKNOWN"
        trade.extra["vol_regime"] = trade.extra["vol_regime_entry"]
        session_entry_value = policy_state_snapshot.get("session") or infer_session_tag(trade.entry_time).upper()
        trade.extra["session"] = session_entry_value
        trade.extra["session_entry"] = session_entry_value
        trade.extra["trade_id"] = trade_id

        return trade
