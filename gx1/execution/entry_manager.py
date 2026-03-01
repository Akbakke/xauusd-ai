from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
from collections import deque, defaultdict

import logging
import os
import time
import uuid

import numpy as np
import pandas as pd

from gx1.execution.live_features import build_live_entry_features, infer_session_tag
from gx1.xgb.multihead.xgb_multihead_model_v1 import proba_to_signal_bridge_v1

if TYPE_CHECKING:
    from gx1.execution.oanda_demo_runner import GX1DemoRunner, LiveTrade


log = logging.getLogger(__name__)


class EntryManager:
    """
    Canonical ENTRY manager (V10_CTX-only).

    HARD RULES (TRUTH / replay hygiene):
      - No SNIPER references/imports/config keys.
      - No fallbacks to legacy policy modules.
      - Replay + prebuilt enabled => must be available, else hard-fail.
      - Replay contract: trade_uid format and entry snapshot writing are hard requirements.
    """

    HARD_ELIGIBILITY_WARMUP = "HARD_WARMUP"
    HARD_ELIGIBILITY_SESSION_BLOCK = "HARD_SESSION_BLOCK"
    HARD_ELIGIBILITY_SPREAD_CAP = "HARD_SPREAD_CAP"
    HARD_ELIGIBILITY_KILLSWITCH = "HARD_KILLSWITCH"

    SOFT_ELIGIBILITY_VOL_REGIME_EXTREME = "SOFT_VOL_REGIME_EXTREME"

    def __init__(self, runner: "GX1DemoRunner", exit_config_name: Optional[str] = None) -> None:
        object.__setattr__(self, "_runner", runner)
        object.__setattr__(self, "exit_config_name", exit_config_name)

        # Optional telemetry collector (only if explicitly required)
        object.__setattr__(self, "entry_feature_telemetry", None)
        require_telemetry = os.environ.get("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
        if require_telemetry:
            from gx1.execution.entry_feature_telemetry import EntryFeatureTelemetryCollector

            output_dir = getattr(runner, "output_dir", None)
            object.__setattr__(self, "entry_feature_telemetry", EntryFeatureTelemetryCollector(output_dir=output_dir))

        object.__setattr__(
            self,
            "entry_telemetry",
            {
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
            },
        )

        # Eval parity counters, etc. (kept because pipeline expects them)
        object.__setattr__(self, "eval_calls_total", 0)
        object.__setattr__(self, "eval_calls_prebuilt_gate_true", 0)
        object.__setattr__(self, "eval_calls_prebuilt_gate_false", 0)

        # Killchain / veto counters (kept for exporters/diags)
        object.__setattr__(self, "killchain_version", 1)
        object.__setattr__(self, "killchain_n_entry_pred_total", 0)
        object.__setattr__(self, "killchain_n_above_threshold", 0)
        object.__setattr__(self, "killchain_n_after_session_guard", 0)
        object.__setattr__(self, "killchain_n_after_vol_guard", 0)
        object.__setattr__(self, "killchain_n_after_risk_sizing", 0)
        object.__setattr__(self, "killchain_n_trade_create_attempts", 0)
        object.__setattr__(self, "killchain_n_trade_created", 0)
        object.__setattr__(
            self,
            "killchain_block_reason_counts",
            {
                "BLOCK_BELOW_THRESHOLD": 0,
                "BLOCK_SESSION": 0,
                "BLOCK_VOL": 0,
                "BLOCK_RISK": 0,
                "BLOCK_POSITION_LIMIT": 0,
                "BLOCK_COOLDOWN": 0,
                "BLOCK_UNKNOWN": 0,
            },
        )
        object.__setattr__(self, "killchain_unknown_examples", [])

        object.__setattr__(
            self,
            "veto_hard",
            {
                "veto_hard_warmup": 0,
                "veto_hard_session": 0,
                "veto_hard_spread": 0,
                "veto_hard_killswitch": 0,
            },
        )
        object.__setattr__(self, "veto_soft", {"veto_soft_vol_regime_extreme": 0})
        object.__setattr__(
            self,
            "veto_pre",
            {
                "veto_pre_warmup": 0,
                "veto_pre_session": 0,
                "veto_pre_regime": 0,
                "veto_pre_spread": 0,
                "veto_pre_atr": 0,
                "veto_pre_killswitch": 0,
                "veto_pre_model_missing": 0,
                "veto_pre_nan_features": 0,
            },
        )
        object.__setattr__(
            self,
            "veto_cand",
            {
                "veto_cand_threshold": 0,
                "veto_cand_risk_guard": 0,
                "veto_cand_max_trades": 0,
                "veto_cand_other": 0,
            },
        )

        # Misc state used by downstream logs/guards
        object.__setattr__(self, "threshold_used", None)
        object.__setattr__(self, "cluster_guard_history", deque(maxlen=600))
        object.__setattr__(self, "cluster_guard_atr_median", None)
        object.__setattr__(self, "spread_history", deque(maxlen=600))

        object.__setattr__(self, "n_v10_calls", 0)
        object.__setattr__(self, "n_v10_pred_ok", 0)
        object.__setattr__(self, "n_v10_pred_none_or_nan", 0)

        # Risk guard instance (canonical)
        object.__setattr__(self, "_risk_guard", None)
        object.__setattr__(self, "risk_guard_identity", None)

        # Lazy-set attributes (must live on manager, not runner)
        object.__setattr__(self, "_last_atr_proxy", None)
        object.__setattr__(self, "_last_spread_bps", None)
        object.__setattr__(self, "_last_policy_state", None)
        object.__setattr__(self, "_next_trade_id", 0)

        # parity (kept; used by existing tooling)
        object.__setattr__(self, "parity_sample_counter", 0)
        object.__setattr__(self, "parity_sample_every_n", 10)

        # Load risk guard immediately so identity is captured for capsules even if
        # no evaluation cycle runs (e.g., warmup-only windows).
        try:
            self._maybe_load_risk_guard()
        except Exception:
            # Let replay surface failures later when guard is required; identity
            # remains None if load failed here.
            pass

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

    # -------------------------
    # Small utilities
    # -------------------------

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

    def _is_kill_switch_active(self) -> bool:
        try:
            from pathlib import Path

            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            kill_flag = project_root / "KILL_SWITCH_ON"
            return kill_flag.exists()
        except Exception:
            return False

    def _get_spread_bps_before_features(self, candles: pd.DataFrame) -> Optional[float]:
        if candles.empty:
            return None
        try:
            if "bid_close" in candles.columns and "ask_close" in candles.columns:
                bid = candles["bid_close"].iloc[-1]
                ask = candles["ask_close"].iloc[-1]
                if pd.notna(bid) and pd.notna(ask) and bid > 0:
                    spread_price = ask - bid
                    return float((spread_price / bid) * 10000.0)
            if "spread" in candles.columns:
                spread_price = candles["spread"].iloc[-1]
                if pd.notna(spread_price):
                    close = candles.get("close", None)
                    if close is not None and len(close) > 0:
                        price_ref = float(close.iloc[-1])
                        if price_ref > 0:
                            return float((float(spread_price) / price_ref) * 10000.0)
        except Exception:
            pass
        return None

    def _compute_cheap_atr_proxy(self, candles: pd.DataFrame, window: int = 14) -> Optional[float]:
        if candles.empty or len(candles) < window:
            return None
        try:
            if not all(c in candles.columns for c in ("high", "low", "close")):
                return None
            high_arr = candles["high"].iloc[-window:].values
            low_arr = candles["low"].iloc[-window:].values
            close_arr = candles["close"].iloc[-window:].values
            tr1 = high_arr - low_arr
            tr2 = np.abs(high_arr - np.roll(close_arr, 1))
            tr3 = np.abs(low_arr - np.roll(close_arr, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = np.mean(tr)
            return float(atr)
        except Exception:
            return None

    # -------------------------
    # Eligibility gates
    # -------------------------

    def _check_hard_eligibility(self, candles: pd.DataFrame, policy_state: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        warmup_bars = getattr(self, "warmup_bars", 288)
        if len(candles) < warmup_bars:
            return False, self.HARD_ELIGIBILITY_WARMUP

        # Canonical key
        policy_cfg = self.policy.get("entry_policy_v10_ctx", {})

        # Strict legacy key ban (this file is supposed to be clean)
        if "entry_policy_sniper_v10_ctx" in self.policy:
            raise RuntimeError("[LEGACY_FORBIDDEN] Found legacy key 'entry_policy_sniper_v10_ctx' in policy dict.")

        current_ts = candles.index[-1] if len(candles) > 0 else None
        current_session = policy_state.get("session")
        if not current_session:
            current_session = infer_session_tag(current_ts).upper() if current_ts is not None else "UNKNOWN"
            policy_state["session"] = current_session

        allowed_sessions = policy_cfg.get("allowed_sessions", ["EU", "OVERLAP", "US"])
        if current_session not in allowed_sessions:
            return False, self.HARD_ELIGIBILITY_SESSION_BLOCK

        spread_bps = self._get_spread_bps_before_features(candles)
        if spread_bps is not None:
            spread_hard_cap_bps = float(policy_cfg.get("spread_hard_cap_bps", 100.0))
            if spread_bps > spread_hard_cap_bps:
                return False, self.HARD_ELIGIBILITY_SPREAD_CAP

        if self._is_kill_switch_active():
            return False, self.HARD_ELIGIBILITY_KILLSWITCH

        return True, None

    def _check_soft_eligibility(
        self, candles: pd.DataFrame, policy_state: Dict[str, Any], *, is_replay: bool = False
    ) -> Tuple[bool, Optional[str]]:
        atr_proxy = self._compute_cheap_atr_proxy(candles, window=14)
        self._last_atr_proxy = atr_proxy
        self._last_spread_bps = self._get_spread_bps_before_features(candles)

        # Cheap “extreme ATR” soft block (bps)
        atr_bps: Optional[float] = None
        current_ts = candles.index[-1] if len(candles) > 0 else None
        if atr_proxy is not None and "close" in candles.columns and len(candles) > 0:
            try:
                current_price = float(candles["close"].iloc[-1])
                if current_price > 0:
                    atr_bps = (float(atr_proxy) / current_price) * 10000.0
            except Exception:
                atr_bps = None

        # In TRUTH replay we bypass ATR pregate to force trades, but log proof once.
        if is_replay:
            if not getattr(self, "_atr_proof_logged", False):
                would_pass = True if atr_bps is None else atr_bps <= 200.0
                log.info(
                    "[ATR_PROOF] current_ts=%s atr_value=%s source_field=%s min_atr_bps=%s max_atr_bps=%s would_pass=%s (NOTE: ATR gate bypassed in REPLAY)",
                    current_ts,
                    atr_bps,
                    "cheap_atr_proxy",
                    "N/A",
                    "200.0",
                    would_pass,
                )
                self._atr_proof_logged = True
            return True, None

        if atr_bps is not None and atr_bps > 200.0:
            return False, self.SOFT_ELIGIBILITY_VOL_REGIME_EXTREME
        return True, None

    # -------------------------
    # Session/vol guard (local, deterministic, no legacy imports)
    # -------------------------

    @staticmethod
    def _extract_session_and_vol_regime(row: pd.Series) -> Tuple[str, str]:
        session = "UNKNOWN"
        vol_regime = "UNKNOWN"
        try:
            if "session" in row.index:
                session = str(row["session"])
            if "vol_regime" in row.index:
                vol_regime = str(row["vol_regime"])
        except Exception:
            pass
        return session, vol_regime

    @staticmethod
    def _assert_session_vol_allowed(
        row: pd.Series,
        *,
        allowed_sessions: Tuple[str, ...],
        allowed_vol_regimes: Tuple[str, ...],
    ) -> None:
        session, vol_regime = EntryManager._extract_session_and_vol_regime(row)
        if session not in allowed_sessions:
            raise AssertionError(f"session={session} not in {allowed_sessions}")
        if vol_regime not in allowed_vol_regimes:
            raise AssertionError(f"vol_regime={vol_regime} not in {allowed_vol_regimes}")

    # -------------------------
    # Risk guard (canonical)
    # -------------------------

    def _maybe_load_risk_guard(self) -> None:
        if self._risk_guard is not None:
            return

        cfg = (self.policy.get("risk_guard") or {}) if hasattr(self, "policy") else {}
        risk_guard_cfg_path = cfg.get("config_path")
        if not risk_guard_cfg_path:
            self._risk_guard = None
            self.risk_guard_identity = None
            return

        try:
            import yaml
            from pathlib import Path
            from gx1.policy.risk_guard_v1 import RiskGuardV1

            guard_cfg_path = Path(risk_guard_cfg_path)
            if not guard_cfg_path.is_absolute():
                guard_cfg_path = Path(__file__).resolve().parent.parent.parent / guard_cfg_path

            if not guard_cfg_path.exists():
                raise FileNotFoundError(str(guard_cfg_path))

            with open(guard_cfg_path, "r", encoding="utf-8") as f:
                guard_cfg = yaml.safe_load(f) or {}

            self._risk_guard = RiskGuardV1(guard_cfg)

            # Guard identity validation (optional guard_id in config)
            try:
                self.risk_guard_identity = self._risk_guard.get_guard_id()
            except Exception:
                self.risk_guard_identity = None

        except Exception as e:
            # No silent fallback: if path is specified but cannot load, fail in replay; warn in live.
            if getattr(self._runner, "replay_mode", False):
                raise RuntimeError(f"[RISK_GUARD_LOAD_FAILED] {type(e).__name__}: {e}") from e
            log.warning("[RISK_GUARD_LOAD_FAILED] %s: %s", type(e).__name__, e)
            self._risk_guard = None
            self.risk_guard_identity = None

    # -------------------------
    # Main entry evaluation
    # -------------------------

    def evaluate_entry(self, candles: pd.DataFrame) -> Optional["LiveTrade"]:
        # Reset routing for telemetry collector
        if getattr(self, "entry_feature_telemetry", None):
            self.entry_feature_telemetry.reset_routing_for_next_bar()

        # Canonical hard-fail: legacy V9 policy key must not be present
        if hasattr(self, "policy") and isinstance(self.policy, dict):
            if "entry_v9_policy_sniper" in self.policy:
                raise RuntimeError(
                    "[FORBIDDEN_CONFIG] entry_v9_policy_sniper is deprecated; use entry_policy_v10_ctx"
                )

        # Always load risk guard early so identity is available for capsules, even if
        # later gates short-circuit the evaluation in replay.
        self._maybe_load_risk_guard()

        self.eval_calls_total += 1
        self.entry_telemetry["n_cycles"] += 1

        prebuilt_enabled = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1"
        is_replay = bool(getattr(self._runner, "replay_mode", False))

        if is_replay and prebuilt_enabled:
            if not hasattr(self._runner, "lookup_attempts"):
                self._runner.lookup_attempts = 0
                self._runner.lookup_hits = 0
                self._runner.lookup_misses = 0
                self._runner.lookup_miss_details = []
            self._runner.lookup_attempts += 1

        policy_state: Dict[str, Any] = {}
        current_ts = candles.index[-1] if len(candles) > 0 else pd.Timestamp.now(tz="UTC")
        current_session = infer_session_tag(current_ts).upper()
        policy_state["session"] = current_session

        def _inc_gate(reason: str) -> None:
            try:
                counters = getattr(self._runner, "entry_gate_counters", None)
                if counters is None:
                    counters = {}
                    setattr(self._runner, "entry_gate_counters", counters)
                counters[reason] = counters.get(reason, 0) + 1
            except Exception:
                pass

        # Hard eligibility gate
        eligible, eligibility_reason = self._check_hard_eligibility(candles, policy_state)
        if getattr(self, "entry_feature_telemetry", None):
            self.entry_feature_telemetry.record_gate(
                gate_name="hard_eligibility",
                executed=True,
                blocked=not eligible,
                passed=bool(eligible),
                reason=None if eligible else eligibility_reason,
            )
        if not eligible:
            if eligibility_reason == self.HARD_ELIGIBILITY_WARMUP:
                _inc_gate("warmup_not_ready")
            elif eligibility_reason == self.HARD_ELIGIBILITY_SESSION_BLOCK:
                _inc_gate("pregate_session")
            elif eligibility_reason == self.HARD_ELIGIBILITY_SPREAD_CAP:
                _inc_gate("pregate_spread")
            elif eligibility_reason == self.HARD_ELIGIBILITY_KILLSWITCH:
                _inc_gate("guard_veto")
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

        self.entry_telemetry["n_eligible_hard"] += 1

        # Soft eligibility gate
        soft_eligible, soft_reason = self._check_soft_eligibility(candles, policy_state, is_replay=is_replay)
        if getattr(self, "entry_feature_telemetry", None):
            self.entry_feature_telemetry.record_gate(
                gate_name="soft_eligibility",
                executed=True,
                blocked=not soft_eligible,
                passed=bool(soft_eligible),
                reason=None if soft_eligible else soft_reason,
            )
        if not soft_eligible:
            if soft_reason == self.SOFT_ELIGIBILITY_VOL_REGIME_EXTREME:
                self.veto_soft["veto_soft_vol_regime_extreme"] += 1
                _inc_gate("pregate_atr")
            return None

        self.entry_telemetry["n_eligible_cycles"] += 1

        # Optional context features (STRICT in replay if enabled and builder fails)
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
                self.entry_telemetry["n_context_built"] += 1
            except Exception as e:
                if is_replay:
                    raise RuntimeError(f"CONTEXT_FEATURES_BUILD_FAILED: {e}") from e
                self.entry_telemetry["n_context_missing_or_invalid"] += 1

        # Prebuilt strictness (replay)
        prebuilt_features_df_exists = hasattr(self._runner, "prebuilt_features_df")
        prebuilt_features_df_is_none = (not prebuilt_features_df_exists) or (self._runner.prebuilt_features_df is None)
        prebuilt_used_flag = bool(getattr(self._runner, "prebuilt_used", False)) if hasattr(self._runner, "prebuilt_used") else False
        prebuilt_available = (
            is_replay
            and prebuilt_enabled
            and prebuilt_features_df_exists
            and (not prebuilt_features_df_is_none)
            and hasattr(self._runner, "prebuilt_used")
            and prebuilt_used_flag
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
                "Check logs for prebuilt loading errors."
            )

        # Build entry features (prebuilt or live)
        if prebuilt_available:
            if len(candles) == 0:
                raise RuntimeError("[PREBUILT_FAIL] candles is empty; cannot get timestamp for prebuilt lookup.")
            ts = candles.index[-1]
            try:
                features_row = self._runner.prebuilt_features_df.loc[ts]
                self._runner.lookup_hits += 1
            except KeyError:
                self._runner.lookup_misses += 1
                idx = self._runner.prebuilt_features_df.index
                raise RuntimeError(
                    f"[PREBUILT_LOOKUP_MISS] Timestamp {ts} not found in prebuilt features. "
                    f"Prebuilt range: {idx.min()} to {idx.max()}."
                )

            from gx1.execution.live_features import EntryFeatureBundle, compute_atr_bps, infer_vol_bucket
            from gx1.tuning.feature_manifest import align_features, load_manifest
            from gx1.execution.live_features import ADR_WINDOW, PIPS_PER_PERCENT

            features_df = features_row.to_frame().T
            features_df.index = [ts]

            manifest = load_manifest()
            aligned = align_features(features_df, manifest=manifest, training_stats=manifest.get("training_stats"))
            aligned_last = aligned.tail(1).copy()

            atr_series = compute_atr_bps(candles[["high", "low", "close"]])
            atr_bps = float(atr_series.iloc[-1])

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

            if not getattr(self._runner, "prebuilt_used", False):
                raise RuntimeError("[PREBUILT_FAIL] prebuilt_features_df is available but prebuilt_used=False.")
        else:
            entry_bundle = build_live_entry_features(candles)

        # Cache ATR/spread diags
        current_atr_bps: Optional[float] = None
        current_atr_pct: Optional[float] = None
        try:
            if entry_bundle.atr_bps is not None:
                current_atr_bps = float(entry_bundle.atr_bps)
        except Exception:
            current_atr_bps = None

        if current_atr_bps is not None and current_atr_bps > 0:
            self.cluster_guard_history.append(current_atr_bps)
            if len(self.cluster_guard_history) >= 25:
                try:
                    self.cluster_guard_atr_median = float(np.median(self.cluster_guard_history))
                except Exception:
                    pass
            current_atr_pct = self._percentile_from_history(self.cluster_guard_history, current_atr_bps)

        # Require model enabled
        if not (hasattr(self._runner, "entry_v10_enabled") and self._runner.entry_v10_enabled):
            self.veto_pre["veto_pre_model_missing"] += 1
            return None

        # V10 supported sessions only
        v10_supported_sessions = {"EU", "US", "OVERLAP"}
        if current_session not in v10_supported_sessions:
            return None

        # Model predict (canonical)
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

        if entry_pred is None:
            self.n_v10_pred_none_or_nan += 1
            return None

        if np.isfinite(entry_pred.prob_long) and np.isfinite(entry_pred.prob_short):
            self.n_v10_pred_ok += 1
            self.entry_telemetry["n_candidates"] += 1
            self.entry_telemetry["n_predictions"] += 1
            self.entry_telemetry["p_long_values"].append(float(entry_pred.prob_long))
            self.entry_telemetry["candidate_sessions"][current_session] = (
                self.entry_telemetry["candidate_sessions"].get(current_session, 0) + 1
            )
        else:
            self.n_v10_pred_none_or_nan += 1
            return None

        # EXIT IO signal7 ringbuffer (per bar)
        try:
            proba_row = np.asarray(
                [[float(entry_pred.prob_long), float(entry_pred.prob_short), float(entry_pred.prob_neutral)]],
                dtype=float,
            )
            bridge = proba_to_signal_bridge_v1(proba_row)[0].tolist()
            signal7_now = {
                "p_long": float(bridge[0]),
                "p_short": float(bridge[1]),
                "p_flat": float(bridge[2]),
                "p_hat": float(bridge[3]),
                "uncertainty_score": float(bridge[4]),
                "margin_top1_top2": float(bridge[5]),
                "entropy": float(bridge[6]),
            }
        except Exception as e:
            raise RuntimeError(f"[EXIT_IO_CONTRACT_VIOLATION] signal7 unavailable: {e}") from e

        # Threshold gate (canonical policy key)
        policy_cfg = self.policy.get("entry_policy_v10_ctx", {})
        entry_gating_cfg = self.policy.get("entry_gating", {}) if getattr(self._runner, "guard_enabled", True) else {}
        min_prob_long = float(policy_cfg.get("min_prob_long", 0.67))
        min_prob_short = float(policy_cfg.get("min_prob_short", 0.72))
        allow_short = bool(policy_cfg.get("allow_short", False))
        # Use side-aware probability for threshold (p_side) rather than always prob_long
        side = "long" if float(entry_pred.prob_long) >= float(entry_pred.prob_short) else "short"
        try:
            if not hasattr(self._runner, "entry_attempt_long"):
                self._runner.entry_attempt_long = 0
                self._runner.entry_attempt_short = 0
                self._runner.entry_accept_long = 0
                self._runner.entry_accept_short = 0
            if side == "long":
                self._runner.entry_attempt_long += 1
            else:
                self._runner.entry_attempt_short += 1
        except Exception:
            pass
        p_side = float(entry_pred.prob_long) if side == "long" else float(entry_pred.prob_short)
        p_other = float(entry_pred.prob_short) if side == "long" else float(entry_pred.prob_long)

        # Analysis override (explicit and opt-in)
        if os.getenv("GX1_ANALYSIS_MODE") == "1" and os.getenv("GX1_ENTRY_THRESHOLD_OVERRIDE"):
            try:
                override = float(os.getenv("GX1_ENTRY_THRESHOLD_OVERRIDE"))
                min_prob_long = override
                min_prob_short = override
            except Exception:
                pass

        self.threshold_used = f"long={min_prob_long},short={min_prob_short}"

        self.killchain_n_entry_pred_total += 1
        p_side_min_cfg = entry_gating_cfg.get("p_side_min", {}) if isinstance(entry_gating_cfg, dict) else {}
        threshold_used_val = (
            float(p_side_min_cfg.get(side, min_prob_long if side == "long" else min_prob_short))
            if p_side_min_cfg
            else (min_prob_long if side == "long" else min_prob_short)
        )
        above_threshold = bool(p_side >= threshold_used_val)
        if above_threshold:
            self.killchain_n_above_threshold += 1
        else:
            self.veto_cand["veto_cand_threshold"] += 1
            self._killchain_inc_reason("BLOCK_BELOW_THRESHOLD")
            _inc_gate("p_threshold")
            if not getattr(self, "_p_threshold_proof_logged", False):
                try:
                    log.info(
                        "[P_THRESHOLD_PROOF] prob_long=%.4f prob_short=%.4f side=%s p_used=%.4f min_prob=%.4f pass=%s",
                        float(entry_pred.prob_long),
                        float(entry_pred.prob_short),
                        side.upper(),
                        p_side,
                        threshold_used_val,
                        False,
                    )
                finally:
                    self._p_threshold_proof_logged = True
            return None

        # Optional entry_gating (policy-driven; TRUTH counters)
        entry_gating_cfg = self.policy.get("entry_gating", {}) if getattr(self._runner, "guard_enabled", True) else {}
        if entry_gating_cfg:
            side = "long" if float(entry_pred.prob_long) >= float(entry_pred.prob_short) else "short"
            p_side = float(entry_pred.prob_long) if side == "long" else float(entry_pred.prob_short)
            p_other = float(entry_pred.prob_short) if side == "long" else float(entry_pred.prob_long)
            margin_val = float(getattr(entry_pred, "margin", np.nan))

            p_side_min_cfg = entry_gating_cfg.get("p_side_min", {})
            margin_min_cfg = entry_gating_cfg.get("margin_min", {})
            side_ratio_min = float(entry_gating_cfg.get("side_ratio_min", 1.25))

            p_side_min = float(p_side_min_cfg.get(side, 0.55))
            margin_min = float(margin_min_cfg.get(side, 0.08))

            if p_side < p_side_min:
                _inc_gate("p_threshold")
                return None

            if np.isfinite(margin_val) and margin_val < margin_min:
                _inc_gate("margin_threshold")
                return None

            ratio_val = p_side / max(p_other, 1e-6)
            if ratio_val < side_ratio_min:
                _inc_gate("ratio_threshold")
                return None

        # Build current_row for guards/journal (deterministic)
        current_row = entry_bundle.features.iloc[-1:].copy()
        current_row["prob_long"] = float(entry_pred.prob_long)
        current_row["prob_short"] = float(entry_pred.prob_short)
        if "close" not in current_row.columns:
            current_row["close"] = float(entry_bundle.close_price)

        # Ensure session tag exists on row (prefer policy_state)
        if len(current_row) > 0:
            if "session" not in current_row.columns or str(current_row["session"].iloc[0]) in ("", "UNKNOWN", "nan"):
                current_row.loc[current_row.index[0], "session"] = policy_state.get("session", current_session)

        # Session/vol guard (local)
        allow_high_vol = bool(policy_cfg.get("allow_high_vol", True))
        allow_extreme_vol = bool(policy_cfg.get("allow_extreme_vol", False))
        allowed_vol = ("LOW", "MEDIUM") + (("HIGH",) if allow_high_vol else tuple()) + (("EXTREME",) if allow_extreme_vol else tuple())
        if is_replay:
            allowed_vol = ("LOW", "MEDIUM", "HIGH", "EXTREME", "UNKNOWN")

        try:
            if above_threshold:
                sess, vol_reg = self._extract_session_and_vol_regime(current_row.iloc[0])
                if sess in ("EU", "OVERLAP", "US"):
                    self.killchain_n_after_session_guard += 1
                if vol_reg in allowed_vol:
                    self.killchain_n_after_vol_guard += 1
            self._assert_session_vol_allowed(
                current_row.iloc[0],
                allowed_sessions=("EU", "OVERLAP", "US"),
                allowed_vol_regimes=allowed_vol,
            )
        except AssertionError as e:
            err = str(e)
            if "vol_regime=UNKNOWN" in err:
                self.entry_telemetry["vol_regime_unknown_count"] += 1
            if "session=" in err:
                self._killchain_inc_reason("BLOCK_SESSION")
                _inc_gate("pregate_session")
            elif "vol_regime=" in err:
                self._killchain_inc_reason("BLOCK_VOL")
                _inc_gate("pregate_atr")
            else:
                self._killchain_inc_reason("BLOCK_UNKNOWN")
                try:
                    self._killchain_record_unknown(
                        {"where": "SESSION_VOL_GUARD", "ts": current_ts.isoformat(), "error": err[:300]}
                    )
                except Exception:
                    pass
            return None

        # Risk guard (canonical)
        self._maybe_load_risk_guard()
        risk_guard_blocked = False
        risk_guard_reason = None
        risk_guard_details: Dict[str, Any] = {}
        risk_guard_clamp = None

        if self._risk_guard is not None and getattr(self._risk_guard, "enabled", False):
            entry_snapshot = {
                "session": str(current_row["session"].iloc[0]) if "session" in current_row.columns and len(current_row) > 0 else None,
                "vol_regime": str(current_row["vol_regime"].iloc[0]) if "vol_regime" in current_row.columns and len(current_row) > 0 else None,
                "spread_bps": float(current_row["spread_bps"].iloc[0]) if "spread_bps" in current_row.columns and len(current_row) > 0 else None,
                "atr_bps": float(current_row["atr_bps"].iloc[0]) if "atr_bps" in current_row.columns and len(current_row) > 0 else None,
            }
            feature_context = {"atr_bps": current_atr_bps, "spread_bps": getattr(entry_bundle, "spread_bps", None)}
            should_block, reason_code, details = self._risk_guard.should_block(
                entry_snapshot, feature_context, policy_state, len(candles) if candles is not None else 0
            )
            if should_block:
                risk_guard_blocked = True
                risk_guard_reason = reason_code
                risk_guard_details = details or {}
                if "cooldown" in str(reason_code).lower():
                    self._killchain_inc_reason("BLOCK_COOLDOWN")
                else:
                    self._killchain_inc_reason("BLOCK_RISK")
                _inc_gate("guard_veto")
                return None

            session_for_clamp = entry_snapshot.get("session") or policy_state.get("session")
            clamp = self._risk_guard.get_session_clamp(session_for_clamp)
            if clamp is not None and clamp > 0:
                risk_guard_clamp = clamp
                policy_state["risk_guard_min_prob_long_clamp"] = clamp

        policy_state["risk_guard_blocked"] = risk_guard_blocked
        policy_state["risk_guard_reason"] = risk_guard_reason
        policy_state["risk_guard_details"] = risk_guard_details
        policy_state["risk_guard_clamp"] = risk_guard_clamp

        # If a clamp exists, apply it deterministically to min_prob_long
        if risk_guard_clamp is not None:
            try:
                min_prob_long = max(min_prob_long, float(risk_guard_clamp))
                self.threshold_used = f"long={min_prob_long},short={min_prob_short}" if allow_short else f"long={min_prob_long}"
                if not (
                    (float(entry_pred.prob_long) >= min_prob_long)
                    or (allow_short and float(entry_pred.prob_short) >= min_prob_short)
                ):
                    self.veto_cand["veto_cand_risk_guard"] += 1
                    self._killchain_inc_reason("BLOCK_RISK")
                    return None
            except Exception:
                pass

        # Units (canonical: keep base units; no sniper overlays)
        side = "long" if entry_pred.prob_long >= entry_pred.prob_short else "short"
        base_units = self.exec.default_units if side == "long" else -self.exec.default_units
        units_out = int(base_units)

        if units_out == 0:
            self._killchain_inc_reason("BLOCK_RISK")
            return None

        self.killchain_n_after_risk_sizing += 1
        self.entry_telemetry["n_candidate_pass"] += 1

        # ENTRY_ONLY mode: log and stop
        now_ts = current_ts
        if getattr(self, "mode", None) == "ENTRY_ONLY":
            try:
                self._log_entry_only_event(
                    timestamp=now_ts,
                    side=side,
                    price=float(entry_bundle.close_price),
                    prediction=entry_pred,
                    policy_state=policy_state,
                )
            except Exception:
                pass
            return None

        # Trade creation
        self._next_trade_id += 1
        trade_id = f"SIM-{int(time.time())}-{self._next_trade_id:06d}"

        run_id = getattr(self._runner, "run_id", "unknown")
        chunk_id = getattr(self._runner, "chunk_id", "single")
        local_seq = self._next_trade_id
        uuid_short = uuid.uuid4().hex[:12]
        trade_uid = f"{run_id}:{chunk_id}::{local_seq:06d}:{uuid_short}"

        if getattr(self, "is_replay", False):
            expected_prefix = f"{run_id}:{chunk_id}::"
            if not trade_uid.startswith(expected_prefix):
                raise RuntimeError(
                    f"BAD_TRADE_UID_FORMAT_REPLAY: Generated trade_uid={trade_uid} does not start with "
                    f"expected prefix={expected_prefix}. run_id={run_id}, chunk_id={chunk_id}."
                )

        # Bid/ask required for replay correctness
        try:
            current_bar = candles.iloc[-1]
            entry_bid_price = float(current_bar["bid_close"])
            entry_ask_price = float(current_bar["ask_close"])
        except KeyError as exc:
            raise ValueError("Bid/ask required for replay, but missing in candles during entry creation.") from exc

        entry_price = entry_ask_price if side == "long" else entry_bid_price

        # Construct LiveTrade
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
            entry_prob_long=float(entry_pred.prob_long),
            entry_prob_short=float(entry_pred.prob_short),
            dry_run=self.exec.dry_run,
        )
        # Store entry snapshot fields for exit IO (CTX19)
        trade.p_long_entry = signal7_now["p_long"]
        trade.p_hat_entry = signal7_now["p_hat"]
        trade.uncertainty_entry = signal7_now["uncertainty_score"]
        trade.entropy_entry = signal7_now["entropy"]
        trade.margin_entry = signal7_now["margin_top1_top2"]
        # Initialize trade-state metrics for exit IO
        trade.mfe_bps = 0.0
        trade.mae_bps = 0.0
        trade._mfe_last_bar = 0

        self.entry_telemetry["n_trades_created"] += 1
        self.killchain_n_trade_create_attempts += 1
        self.killchain_n_trade_created += 1

        session_key = policy_state.get("session") or infer_session_tag(trade.entry_time).upper()
        self.entry_telemetry["trade_sessions"][session_key] = self.entry_telemetry["trade_sessions"].get(session_key, 0) + 1

        # Ensure extra exists
        if not hasattr(trade, "extra"):
            trade.extra = {}
        trade.extra["tp_bps"] = int(self.tick_cfg.get("tp_bps", 180))
        trade.extra["sl_bps"] = int(self.tick_cfg.get("sl_bps", 100))
        trade.extra["be_trigger_bps"] = int(self.tick_cfg.get("be_trigger_bps", 50))
        trade.extra["be_active"] = False
        trade.extra["be_price"] = None

        try:
            if side == "long":
                self._runner.entry_accept_long += 1
            else:
                self._runner.entry_accept_short += 1
        except Exception:
            pass

        # Exit profile selection (existing mechanism)
        self._ensure_exit_profile(trade, context="entry_manager")
        if self.exit_config_name and not (getattr(trade, "extra", {}) or {}).get("exit_profile"):
            raise RuntimeError(
                f"[EXIT_PROFILE] Trade created without exit_profile under exit-config {self.exit_config_name}: {trade.trade_id}"
            )

        trade.client_order_id = self._generate_client_order_id(trade.entry_time, trade.entry_price, trade.side)

        # Record risk guard entry (cooldown)
        if getattr(self, "_risk_guard", None) is not None and getattr(self._risk_guard, "enabled", False):
            is_replay = bool(getattr(self._runner, "replay_mode", False))
            if is_replay and (candles is None or len(candles) == 0):
                raise RuntimeError("[RISK_GUARD_COOLDOWN] candles missing/empty in replay when recording guard entry")
            current_bar_index = int(len(candles)) if candles is not None else 0
            self._risk_guard.record_entry(current_bar_index)
            try:
                policy_state["risk_guard_last_entry_bar"] = current_bar_index
            except Exception:
                pass

        # Journal / snapshot logging (kept; replay hard contract)
        risk_guard_id = getattr(self, "risk_guard_identity", None)
        if hasattr(self._runner, "trade_journal") and self._runner.trade_journal:
            try:
                from gx1.monitoring.trade_journal import EVENT_ENTRY_SIGNAL

                self._runner.trade_journal.log(
                    EVENT_ENTRY_SIGNAL,
                    {
                        "entry_time": trade.entry_time.isoformat(),
                        "entry_price": trade.entry_price,
                        "side": trade.side,
                        "entry_model_outputs": {
                            "p_long": float(entry_pred.prob_long),
                            "p_short": float(entry_pred.prob_short),
                            "p_hat": float(getattr(entry_pred, "p_hat", np.nan)),
                            "margin": float(getattr(entry_pred, "margin", np.nan)),
                            "session": str(getattr(entry_pred, "session", policy_state.get("session", "UNKNOWN"))),
                        },
                        "risk_guard": {
                            "identity": risk_guard_id,
                            "blocked": risk_guard_blocked,
                            "reason": risk_guard_reason,
                            "details": risk_guard_details,
                            "clamp": risk_guard_clamp,
                        },
                        "warmup_state": {"bars_since_start": len(candles) if candles is not None else None},
                    },
                    trade_key={
                        "entry_time": trade.entry_time.isoformat(),
                        "entry_price": trade.entry_price,
                        "side": trade.side,
                    },
                    trade_id=trade.trade_id,
                )
            except Exception as e:
                log.warning("[TRADE_JOURNAL] Failed to log ENTRY_SIGNAL: %s", e)

        if hasattr(self._runner, "trade_journal") and self._runner.trade_journal:
            try:
                # Minimal feature context snapshot
                spread_bps = None
                try:
                    spread_raw = float(entry_ask_price) - float(entry_bid_price)
                    if spread_raw > 0:
                        spread_bps = float(spread_raw * 10000.0)
                except Exception:
                    spread_bps = None

                spread_pct = None
                if spread_bps is not None and np.isfinite(spread_bps):
                    spread_pct = self._percentile_from_history(self.spread_history, spread_bps)
                    self.spread_history.append(spread_bps)

                entry_time_iso = trade.entry_time.isoformat()
                instrument_val = getattr(self._runner, "instrument", "XAU_USD")
                model_name_val = getattr(self._runner, "model_name", None)

                self._runner.trade_journal.log_entry_snapshot(
                    entry_time=entry_time_iso,
                    trade_uid=trade.trade_uid,
                    trade_id=trade.trade_id,
                    instrument=instrument_val,
                    side=trade.side,
                    entry_price=trade.entry_price,
                    units=units_out,
                    base_units=base_units,
                    session=policy_state.get("session", "UNKNOWN"),
                    regime=(policy_state.get("farm_regime") or "UNKNOWN"),
                    vol_regime=(policy_state.get("vol_regime") or policy_state.get("brain_vol_regime") or "UNKNOWN"),
                    trend_regime=(policy_state.get("trend_regime") or policy_state.get("brain_trend_regime") or "UNKNOWN"),
                    entry_model_version=model_name_val,
                    entry_score={
                        "p_long": float(entry_pred.prob_long),
                        "p_short": float(entry_pred.prob_short),
                        "p_hat": float(getattr(entry_pred, "p_hat", np.nan)),
                        "margin": float(getattr(entry_pred, "margin", np.nan)),
                    },
                    entry_filters_passed=[],
                    entry_filters_blocked=[],
                    test_mode=bool(policy_state.get("force_entry", False)),
                    reason=("FORCED_CANARY_TRADE" if policy_state.get("force_entry", False) else None),
                    warmup_degraded=getattr(self._runner, "_warmup_degraded", False),
                    cached_bars_at_entry=getattr(self._runner, "_cached_bars_at_startup", None),
                    warmup_bars_required=(self._runner.policy.get("warmup_bars", 288) if hasattr(self._runner, "policy") else None),
                    risk_guard_blocked=risk_guard_blocked,
                    risk_guard_reason=risk_guard_reason,
                    risk_guard_details=risk_guard_details,
                    risk_guard_min_prob_long_clamp=risk_guard_clamp,
                    atr_bps=current_atr_bps,
                    spread_bps=spread_bps,
                    entry_critic=None,
                )
                self.entry_telemetry["n_entry_snapshots_written"] += 1

                # Optional extra context record
                self._runner.trade_journal.log_feature_context(
                    trade_uid=trade.trade_uid,
                    trade_id=trade.trade_id,
                    atr_bps=current_atr_bps,
                    atr_percentile=current_atr_pct,
                    spread_price=(float(entry_ask_price) - float(entry_bid_price)) if entry_ask_price and entry_bid_price else None,
                    spread_pct=spread_pct,
                    candle_close=float(candles["close"].iloc[-1]) if candles is not None and "close" in candles.columns and len(candles) > 0 else None,
                    candle_high=float(candles["high"].iloc[-1]) if candles is not None and "high" in candles.columns and len(candles) > 0 else None,
                    candle_low=float(candles["low"].iloc[-1]) if candles is not None and "low" in candles.columns and len(candles) > 0 else None,
                )
            except Exception as e:
                self.entry_telemetry["n_entry_snapshots_failed"] += 1
                if getattr(self, "is_replay", False):
                    raise RuntimeError(
                        f"ENTRY_SNAPSHOT_MISSING: Failed to log entry_snapshot for trade_uid={trade.trade_uid}, "
                        f"trade_id={trade.trade_id}. Hard contract violation in replay mode. Error: {e}"
                    ) from e
                log.warning("[TRADE_JOURNAL] Failed to log structured entry snapshot: %s", e)

        # Eval log append (kept)
        if hasattr(self, "eval_log_path") and self.eval_log_path:
            try:
                from gx1.execution.oanda_demo_runner import append_eval_log

                eval_record = {
                    "ts_utc": now_ts.isoformat(),
                    "session": str(getattr(entry_pred, "session", policy_state.get("session", "UNKNOWN"))),
                    "p_long": float(entry_pred.prob_long),
                    "p_short": float(entry_pred.prob_short),
                    "p_hat": float(getattr(entry_pred, "p_hat", np.nan)),
                    "margin": float(getattr(entry_pred, "margin", np.nan)),
                    "decision": side.upper(),
                    "price": float(entry_bundle.close_price),
                    "units": int(base_units),
                }
                append_eval_log(self.eval_log_path, eval_record)
            except Exception:
                pass

        # Minimal extra annotations (kept)
        trade.extra["session"] = session_key
        trade.extra["session_entry"] = session_key
        trade.extra["trade_id"] = trade.trade_id
        trade.extra["risk_guard_identity"] = self.risk_guard_identity

        return trade