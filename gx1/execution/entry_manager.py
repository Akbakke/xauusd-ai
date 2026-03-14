from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
from collections import deque, defaultdict
import json
from pathlib import Path

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
        require_telemetry = (
            os.environ.get("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
            or bool(getattr(runner, "replay_mode", False))
            or os.environ.get("GX1_TRUTH_TELEMETRY", "0") == "1"
        )
        if require_telemetry:
            from gx1.execution.entry_feature_telemetry import EntryFeatureTelemetryCollector

            output_dir = getattr(runner, "output_dir", None)
            object.__setattr__(self, "entry_feature_telemetry", EntryFeatureTelemetryCollector(output_dir=output_dir))

        # Wire eval_log_path from runner for deterministic eval logging
        if hasattr(runner, "eval_log_path"):
            object.__setattr__(self, "eval_log_path", getattr(runner, "eval_log_path"))

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
                "entry_persistence_pass": 0,
                "entry_persistence_blocked": 0,
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

        # Persistence gate (optional; default 1 => disabled)
        try:
            persistence_bars = int(os.environ.get("GX1_ENTRY_PERSISTENCE_BARS", "1"))
        except Exception:
            persistence_bars = 1
        if persistence_bars < 1:
            persistence_bars = 1
        object.__setattr__(self, "entry_persistence_bars", int(persistence_bars))
        object.__setattr__(self, "_entry_persistence_hist", deque(maxlen=max(0, int(persistence_bars) - 1)))
        object.__setattr__(self, "_entry_persistence_log_count", 0)
        log.info(
            "[ENTRY_PERSISTENCE_PROOF] persistence_bars=%d pass=0 blocked=0",
            int(persistence_bars),
        )

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
        return None

    # -------------------------
    # Closed window (hard no-trade)
    # -------------------------

    @staticmethod
    def _is_closed_window_A(ts: pd.Timestamp) -> bool:
        """
        Closed window A: 21:55–23:00 UTC.
        """
        if ts is None or ts.tzinfo is None:
            return False
        h = ts.hour
        m = ts.minute
        return (h == 21 and m >= 55) or (h == 22)

    @staticmethod
    def _is_closed_window_B(ts: pd.Timestamp) -> bool:
        """
        Closed window B: 20:55–22:00 UTC.
        """
        if ts is None or ts.tzinfo is None:
            return False
        h = ts.hour
        m = ts.minute
        return (h == 20 and m >= 55) or (h == 21)

    @classmethod
    def _closed_window_kind(cls, ts: pd.Timestamp) -> Optional[str]:
        if cls._is_closed_window_A(ts):
            return "A"
        if cls._is_closed_window_B(ts):
            return "B"
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
        allow_all_sessions = os.getenv("GX1_ENTRY_ALLOW_ALL_SESSIONS", "0") == "1"
        if (not allow_all_sessions) and current_session not in allowed_sessions:
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

        # One-shot gap config log
        if not getattr(self, "_entry_gap_cfg_logged", False):
            cooldown_env = os.getenv("GX1_ENTRY_GAP_COOLDOWN_BARS")
            try:
                self._entry_gap_cooldown_bars = int(cooldown_env) if cooldown_env is not None else 3
            except Exception:
                self._entry_gap_cooldown_bars = 3
            self._gap_spacing_threshold_sec = 600
            log.info(
                "[ENTRY_GAP_CFG] cooldown_bars=%d gap_threshold_sec=%d",
                self._entry_gap_cooldown_bars,
                self._gap_spacing_threshold_sec,
            )
            self._entry_gap_cfg_logged = True
            if not hasattr(self, "_last_gap_bar_index"):
                self._last_gap_bar_index = None

        # Initialize attempt counter
        try:
            if not hasattr(self._runner, "entry_attempts_total"):
                self._runner.entry_attempts_total = 0
            self._runner.entry_attempts_total += 1
        except Exception:
            pass

        # Canonical hard-fail: legacy V9 policy key must not be present
        if hasattr(self, "policy") and isinstance(self.policy, dict):
            if "entry_v9_policy_sniper" in self.policy:
                raise RuntimeError(
                    "[FORBIDDEN_CONFIG] entry_v9_policy_sniper is deprecated; use entry_policy_v10_ctx"
                )

        # Always load risk guard early so identity is available for capsules, even if
        # later gates short-circuit the evaluation in replay.
        self._maybe_load_risk_guard()

        # Replay hard-fail: legacy/debug entry overrides are forbidden
        if getattr(self._runner, "replay_mode", False):
            legacy_envs = [
                "GX1_ENTRY_MINIMAL_POLICY",
                "GX1_ENTRY_MINIMAL_CONFIDENCE_MIN",
                "GX1_ENTRY_PFLAT_GATE",
                "GX1_ENTRY_PFLAT_MARGIN",
                "GX1_ENTRY_RUNNER_UP_MARGIN",
                "GX1_ENTRY_GATING_P_SIDE_MIN_LONG",
                "GX1_ENTRY_GATING_P_SIDE_MIN_SHORT",
                "GX1_ENTRY_GATING_SIDE_RATIO_MIN",
                "GX1_ENTRY_GATING_DISABLE_RATIO",
                "GX1_ENTRY_THRESHOLD_OVERRIDE",
                "GX1_ENTRY_MARGIN_MIN",
            ]
            for key in legacy_envs:
                if key in os.environ and str(os.environ.get(key, "")).strip() not in ("", "0", "false", "False"):
                    raise RuntimeError(f"[LEGACY_ENTRY_OVERRIDE_FORBIDDEN] {key} is set in replay")

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
        current_bar_index = len(candles)

        # Hard no-trade gate for expected closed market windows (A/B)
        closed_kind = self._closed_window_kind(current_ts)
        if closed_kind is not None:
            window = "21:55-23:00" if closed_kind == "A" else "20:55-22:00"
            log.info(
                "[ENTRY_NO_TRADE_CLOSED_WINDOW_PROOF] ts=%s window=%s pattern=%s reason=closed_market",
                current_ts.isoformat(),
                window,
                closed_kind,
            )
            return None

        # Gap guard (policy-level, deterministic; no synthetic bars)
        if len(candles) > 0:
            if "session_id" not in candles.columns:
                raise RuntimeError("[ENTRY_GAP_GUARD] session_id column missing in candles")
            if len(candles) >= 2:
                spacing_sec = (candles.index[-1] - candles.index[-2]).total_seconds()
                if spacing_sec > self._gap_spacing_threshold_sec:
                    self._last_gap_bar_index = current_bar_index
                    try:
                        if not hasattr(self._runner, "entry_gap_guard_hits"):
                            self._runner.entry_gap_guard_hits = 0
                        self._runner.entry_gap_guard_hits += 1
                    except Exception:
                        pass
                    log.info(
                        "[ENTRY_GAP_GUARD] spacing_sec=%.1f bars_since_gap=0 cooldown=%d ts=%s",
                        spacing_sec,
                        self._entry_gap_cooldown_bars,
                        current_ts,
                    )
                    return None
            if getattr(self, "_last_gap_bar_index", None) is not None:
                bars_since_gap = current_bar_index - int(self._last_gap_bar_index)
                if bars_since_gap < self._entry_gap_cooldown_bars:
                    try:
                        if not hasattr(self._runner, "entry_gap_guard_hits"):
                            self._runner.entry_gap_guard_hits = 0
                        self._runner.entry_gap_guard_hits += 1
                    except Exception:
                        pass
                    log.info(
                        "[ENTRY_GAP_GUARD] spacing_sec=%.1f bars_since_gap=%d cooldown=%d ts=%s",
                        (candles.index[-1] - candles.index[-2]).total_seconds() if len(candles) >= 2 else -1.0,
                        bars_since_gap,
                        self._entry_gap_cooldown_bars,
                        current_ts,
                    )
                    return None

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

        # Soft eligibility gate (legacy; disabled in replay for slim policy)
        if not is_replay:
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
        else:
            soft_eligible = True

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

            # Replay-only regime filter (explicit; configured in policy)
            if is_replay:
                try:
                    regime_cfg = self.policy.get("replay_regime_filter", {}) if hasattr(self, "policy") else {}
                    if isinstance(regime_cfg, dict) and regime_cfg.get("enabled", False):
                        # Optional session allowlist for replay-only regime filters
                        session_allow = regime_cfg.get("sessions", None)
                        current_session = infer_session_tag(ts).upper() if ts is not None else "UNKNOWN"
                        if session_allow and current_session not in set(session_allow):
                            # Session not in scope for this filter
                            pass
                        else:
                            triggered = []
                            # H1 compression: block when inside configured bucket
                            h1_cfg = regime_cfg.get("h1_range_compression_ratio", {}) or {}
                            if isinstance(h1_cfg, dict) and h1_cfg.get("enabled", False):
                                h1_val = features_row.get("H1_range_compression_ratio", None)
                                h1_min = h1_cfg.get("min", None)
                                h1_max = h1_cfg.get("max", None)
                                if h1_val is not None and np.isfinite(h1_val):
                                    in_bucket = True
                                    if h1_min is not None:
                                        in_bucket = in_bucket and (h1_val >= float(h1_min))
                                    if h1_max is not None:
                                        in_bucket = in_bucket and (h1_val <= float(h1_max))
                                    if in_bucket:
                                        triggered.append(("H1_RANGE_COMPRESSION_Q0_20", float(h1_val), h1_min, h1_max))
                                        _inc_gate("pregate_h1_compression")

                            # M15 compression: block when inside configured bucket
                            m15_cfg = regime_cfg.get("m15_range_compression_ratio", {}) or {}
                            if isinstance(m15_cfg, dict) and m15_cfg.get("enabled", False):
                                m15_val = features_row.get("M15_range_compression_ratio", None)
                                m15_min = m15_cfg.get("min", None)
                                m15_max = m15_cfg.get("max", None)
                                if m15_val is not None and np.isfinite(m15_val):
                                    in_bucket = True
                                    if m15_min is not None:
                                        in_bucket = in_bucket and (m15_val >= float(m15_min))
                                    if m15_max is not None:
                                        in_bucket = in_bucket and (m15_val <= float(m15_max))
                                    if in_bucket:
                                        triggered.append(("M15_RANGE_COMPRESSION_Q0_20", float(m15_val), m15_min, m15_max))
                                        _inc_gate("pregate_m15_compression")

                            # D1 ATR percentile: block when inside configured bucket (session-scoped)
                            d1_cfg = regime_cfg.get("d1_atr_percentile_252", {}) or {}
                            if isinstance(d1_cfg, dict) and d1_cfg.get("enabled", False):
                                d1_val = features_row.get("D1_atr_percentile_252", None)
                                d1_min = d1_cfg.get("min", None)
                                d1_max = d1_cfg.get("max", None)
                                if d1_val is not None and np.isfinite(d1_val):
                                    in_bucket = True
                                    if d1_min is not None:
                                        in_bucket = in_bucket and (d1_val >= float(d1_min))
                                    if d1_max is not None:
                                        in_bucket = in_bucket and (d1_val <= float(d1_max))
                                    if in_bucket:
                                        triggered.append(("D1_ATR_PERCENTILE_Q80_100", float(d1_val), d1_min, d1_max))
                                        _inc_gate("pregate_d1_atr_eu")

                            if triggered:
                                _inc_gate("pregate_regime_filter")
                                try:
                                    if not hasattr(self._runner, "entry_regime_filter_blocked"):
                                        self._runner.entry_regime_filter_blocked = 0
                                    self._runner.entry_regime_filter_blocked += 1
                                except Exception:
                                    pass
                                details = " | ".join(
                                    f"{name}:value={val:.6f} min={mn} max={mx}"
                                    for name, val, mn, mx in triggered
                                )
                                log.info(
                                    "[ENTRY_REGIME_FILTER_BLOCK] reason=REGIME_FILTER %s session=%s ts=%s",
                                    details,
                                    current_session,
                                    ts,
                                )
                                return None
                except Exception as e:
                    raise RuntimeError(f"[REPLAY_REGIME_FILTER_FAIL] {e}") from e

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

        # Replay: enforce a single confidence source and forbid legacy entry_gating
        if is_replay and abs(min_prob_long - min_prob_short) > 1e-9:
            raise RuntimeError(
                f"[REPLAY_THRESHOLD_MISMATCH] min_prob_long={min_prob_long} min_prob_short={min_prob_short} must match"
            )
        if is_replay and isinstance(entry_gating_cfg, dict) and entry_gating_cfg:
            raise RuntimeError("[LEGACY_ENTRY_GATING_FORBIDDEN] entry_gating is not allowed in replay")

        p_long = float(entry_pred.prob_long)
        p_short = float(entry_pred.prob_short)
        p_flat = float(entry_pred.prob_neutral)
        margin_top1_top2 = float(signal7_now.get("margin_top1_top2", np.nan))
        # Flat gate disabled in slim policy
        flat_gate = 0.0
        flat_margin = 0.0

        threshold_long = float(min_prob_long)
        threshold_short = float(min_prob_short)
        # Proof: effective candidate thresholds
        try:
            if not hasattr(self, "_threshold_proof_log_count"):
                self._threshold_proof_log_count = 0
            self._threshold_proof_log_count += 1
            if self._threshold_proof_log_count <= 5 or (self._threshold_proof_log_count % 5000) == 0:
                log.info(
                    "[ENTRY_THRESHOLD_EFFECTIVE_PROOF] threshold_long=%.4f threshold_short=%.4f env_override=%s policy_min_long=%.4f policy_min_short=%.4f",
                    threshold_long,
                    threshold_short,
                    None,
                    float(policy_cfg.get("min_prob_long", 0.67)),
                    float(policy_cfg.get("min_prob_short", 0.72)),
                )
        except Exception:
            pass

        # Snapshot gate config (for replay proof)
        try:
            p_side_min_long = None
            p_side_min_short = None
            if not hasattr(self._runner, "entry_gate_config_snapshot") or not isinstance(
                self._runner.entry_gate_config_snapshot, dict
            ):
                self._runner.entry_gate_config_snapshot = {}
            self._runner.entry_gate_config_snapshot.update(
                {
                    "p_side_min_long": p_side_min_long,
                    "p_side_min_short": p_side_min_short,
                    "p_flat_gate": flat_gate,
                    "p_flat_margin": flat_margin,
                    "candidate_threshold_long": threshold_long,
                    "candidate_threshold_short": threshold_short,
                    "runner_up_margin": None,
                }
            )
        except Exception:
            pass

        # Pre-gate preference (argmax long/short only)
        if p_long >= p_short:
            pre_gate_pref = "long"
        else:
            pre_gate_pref = "short"

        long_candidate = bool(p_long >= threshold_long)
        short_candidate = bool(p_short >= threshold_short)
        chosen_side = None
        reason = "none"
        winner = pre_gate_pref
        pass_for_winner = long_candidate if winner == "long" else short_candidate

        if winner == "short" and not allow_short:
            chosen_side = None
            reason = "short_disabled"
        elif winner == "long":
            chosen_side = "long" if long_candidate else None
            reason = "confidence_gate_pass" if chosen_side else "below_threshold"
        else:
            chosen_side = "short" if short_candidate else None
            reason = "confidence_gate_pass" if chosen_side else "below_threshold"

        # Persistence gate: require same-side winner + threshold pass across N consecutive bars
        persistence_bars = int(getattr(self, "entry_persistence_bars", 1))
        if persistence_bars > 1:
            hist = getattr(self, "_entry_persistence_hist", deque())
            if len(hist) >= (persistence_bars - 1):
                prev_ok = all((h.get("winner") == winner and h.get("pass") is True) for h in hist)
            else:
                prev_ok = False
            persistence_ok = bool(prev_ok and pass_for_winner)
            if chosen_side is not None and not persistence_ok:
                chosen_side = None
                reason = "persistence_gate"
                if hasattr(self, "entry_telemetry"):
                    self.entry_telemetry["entry_persistence_blocked"] += 1
                _inc_gate("candidate_persistence_blocked")
            elif chosen_side is not None and persistence_ok:
                if hasattr(self, "entry_telemetry"):
                    self.entry_telemetry["entry_persistence_pass"] += 1
            try:
                # log proof (rate-limited)
                self._entry_persistence_log_count += 1
                if self._entry_persistence_log_count <= 5 or (self._entry_persistence_log_count % 5000) == 0:
                    log.info(
                        "[ENTRY_PERSISTENCE_PROOF] persistence_bars=%d pass=%d blocked=%d",
                        int(persistence_bars),
                        int(self.entry_telemetry.get("entry_persistence_pass", 0)),
                        int(self.entry_telemetry.get("entry_persistence_blocked", 0)),
                    )
            except Exception:
                pass
            # Update history after computing gate
            try:
                hist.append({"winner": winner, "pass": bool(pass_for_winner)})
            except Exception:
                pass
        else:
            # Keep history in sync even when disabled
            try:
                hist = getattr(self, "_entry_persistence_hist", deque())
                hist.append({"winner": winner, "pass": bool(pass_for_winner)})
            except Exception:
                pass

        try:
            if not hasattr(self._runner, "signal_candidate_long"):
                self._runner.signal_candidate_long = 0
                self._runner.signal_candidate_short = 0
                self._runner.signal_candidate_none = 0
            if long_candidate:
                self._runner.signal_candidate_long += 1
            if short_candidate:
                self._runner.signal_candidate_short += 1
            if not long_candidate and not short_candidate:
                self._runner.signal_candidate_none += 1
                _inc_gate("candidate_below_threshold")
        except Exception:
            pass

        try:
            if not hasattr(self._runner, "entry_pref_pre_long"):
                self._runner.entry_pref_pre_long = 0
                self._runner.entry_pref_pre_short = 0
                self._runner.entry_pref_pre_flat = 0
            if pre_gate_pref == "long":
                self._runner.entry_pref_pre_long += 1
            elif pre_gate_pref == "short":
                self._runner.entry_pref_pre_short += 1
            else:
                self._runner.entry_pref_pre_flat += 1
        except Exception:
            pass

        try:
            if not hasattr(self._runner, "entry_pref_post_long"):
                self._runner.entry_pref_post_long = 0
                self._runner.entry_pref_post_short = 0
                self._runner.entry_pref_post_none = 0
            if chosen_side == "long":
                self._runner.entry_pref_post_long += 1
            elif chosen_side == "short":
                self._runner.entry_pref_post_short += 1
            else:
                self._runner.entry_pref_post_none += 1
        except Exception:
            pass

        try:
            if not hasattr(self, "_signal_side_log_count"):
                self._signal_side_log_count = 0
            self._signal_side_log_count += 1
            if self._signal_side_log_count <= 5 or (self._signal_side_log_count % 5000) == 0:
                log.warning(
                    "[SIGNAL_SIDE_DECISION] p_long=%.6f p_short=%.6f p_flat=%.6f margin_top1_top2=%.6f "
                    "chosen_side=%s threshold_long=%.4f threshold_short=%.4f long_candidate=%s short_candidate=%s reason=%s",
                    p_long,
                    p_short,
                    p_flat,
                    margin_top1_top2,
                    (chosen_side or "NONE"),
                    threshold_long,
                    threshold_short,
                    long_candidate,
                    short_candidate,
                    reason,
                )
        except Exception:
            pass

        # Flat gate disabled: no config log

        # Ensure timestamp is defined for audit logging even if no trade is taken.
        now_ts = current_ts
        if chosen_side is None:
            # Even if no trade is taken, emit eval_log for audit (decision=NONE)
            eval_log_path = getattr(self, "eval_log_path", None) or getattr(self._runner, "eval_log_path", None)
            if eval_log_path:
                try:
                    from gx1.execution.oanda_demo_runner import append_eval_log

                    xgb_signal7 = getattr(self._runner, "_last_xgb_signal7", None)
                    if not isinstance(xgb_signal7, dict):
                        xgb_signal7 = {}

                    safe_price = np.nan
                    try:
                        safe_price = float(getattr(entry_bundle, "close_price", np.nan))
                    except Exception:
                        safe_price = np.nan

                    eval_record = {
                        "ts_utc": now_ts.isoformat(),
                        "session": str(getattr(entry_pred, "session", policy_state.get("session", "UNKNOWN"))),
                        "p_long": float(entry_pred.prob_long),
                        "p_short": float(entry_pred.prob_short),
                        "p_flat": float(getattr(entry_pred, "prob_neutral", np.nan)),
                        "p_hat": float(getattr(entry_pred, "p_hat", np.nan)),
                        "uncertainty_score": float(signal7_now.get("uncertainty_score", np.nan)),
                        "margin": float(getattr(entry_pred, "margin", np.nan)),
                        "margin_top1_top2": float(signal7_now.get("margin_top1_top2", np.nan)),
                        "entropy": float(signal7_now.get("entropy", np.nan)),
                        # Raw XGB signal7 (pre-ENTRY) for audit
                        "xgb_p_long": float(xgb_signal7.get("p_long", np.nan)),
                        "xgb_p_short": float(xgb_signal7.get("p_short", np.nan)),
                        "xgb_p_flat": float(xgb_signal7.get("p_flat", np.nan)),
                        "xgb_p_hat": float(xgb_signal7.get("p_hat", np.nan)),
                        "xgb_uncertainty_score": float(xgb_signal7.get("uncertainty_score", np.nan)),
                        "xgb_margin_top1_top2": float(xgb_signal7.get("margin_top1_top2", np.nan)),
                        "xgb_entropy": float(xgb_signal7.get("entropy", np.nan)),
                        # Entry signal7 (post-ENTRY transformer) for audit
                        "entry_p_long": float(signal7_now.get("p_long", np.nan)),
                        "entry_p_short": float(signal7_now.get("p_short", np.nan)),
                        "entry_p_flat": float(signal7_now.get("p_flat", np.nan)),
                        "entry_p_hat": float(signal7_now.get("p_hat", np.nan)),
                        "entry_uncertainty_score": float(signal7_now.get("uncertainty_score", np.nan)),
                        "entry_margin_top1_top2": float(signal7_now.get("margin_top1_top2", np.nan)),
                        "entry_entropy": float(signal7_now.get("entropy", np.nan)),
                        "pre_gate_pref": pre_gate_pref,
                        "decision_reason": reason,
                        "flat_gate": flat_gate,
                        "flat_margin": flat_margin,
                        "decision": "NONE",
                        "price": safe_price,
                        "units": 0,
                    }
                    append_eval_log(eval_log_path, eval_record)
                except Exception as e:
                    if getattr(self, "is_replay", False):
                        raise RuntimeError(f"EVAL_LOG_APPEND_FAILED(no_trade): {e}") from e
                    log.warning("[EVAL_LOG_APPEND_FAILED] no_trade: %s", e)
            return None

        # Use side-aware probability for threshold (p_side) rather than always prob_long
        side = chosen_side
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
        p_side = p_long if side == "long" else p_short
        p_other = p_short if side == "long" else p_long

        # Slim policy: single confidence source only
        self.threshold_used = f"long={min_prob_long},short={min_prob_short}"

        def _append_eval_log_block(decision: str, reason_str: str) -> None:
            eval_log_path = getattr(self, "eval_log_path", None) or getattr(self._runner, "eval_log_path", None)
            if not eval_log_path:
                return
            try:
                from gx1.execution.oanda_demo_runner import append_eval_log

                xgb_signal7 = getattr(self._runner, "_last_xgb_signal7", None)
                if not isinstance(xgb_signal7, dict):
                    xgb_signal7 = {}

                safe_price = np.nan
                try:
                    safe_price = float(getattr(entry_bundle, "close_price", np.nan))
                except Exception:
                    safe_price = np.nan

                eval_record = {
                    "ts_utc": now_ts.isoformat(),
                    "session": str(getattr(entry_pred, "session", policy_state.get("session", "UNKNOWN"))),
                    "p_long": float(entry_pred.prob_long),
                    "p_short": float(entry_pred.prob_short),
                    "p_flat": float(signal7_now.get("p_flat", np.nan)),
                    "p_hat": float(getattr(entry_pred, "p_hat", np.nan)),
                    "uncertainty_score": float(signal7_now.get("uncertainty_score", np.nan)),
                    "margin": float(getattr(entry_pred, "margin", np.nan)),
                    "margin_top1_top2": float(signal7_now.get("margin_top1_top2", np.nan)),
                    "entropy": float(signal7_now.get("entropy", np.nan)),
                    "xgb_p_long": float(xgb_signal7.get("p_long", np.nan)),
                    "xgb_p_short": float(xgb_signal7.get("p_short", np.nan)),
                    "xgb_p_flat": float(xgb_signal7.get("p_flat", np.nan)),
                    "xgb_p_hat": float(xgb_signal7.get("p_hat", np.nan)),
                    "xgb_uncertainty_score": float(xgb_signal7.get("uncertainty_score", np.nan)),
                    "xgb_margin_top1_top2": float(xgb_signal7.get("margin_top1_top2", np.nan)),
                    "xgb_entropy": float(xgb_signal7.get("entropy", np.nan)),
                    "entry_p_long": float(signal7_now.get("p_long", np.nan)),
                    "entry_p_short": float(signal7_now.get("p_short", np.nan)),
                    "entry_p_flat": float(signal7_now.get("p_flat", np.nan)),
                    "entry_p_hat": float(signal7_now.get("p_hat", np.nan)),
                    "entry_uncertainty_score": float(signal7_now.get("uncertainty_score", np.nan)),
                    "entry_margin_top1_top2": float(signal7_now.get("margin_top1_top2", np.nan)),
                    "entry_entropy": float(signal7_now.get("entropy", np.nan)),
                    "pre_gate_pref": pre_gate_pref,
                    "decision_reason": reason_str,
                    "flat_gate": flat_gate,
                    "flat_margin": flat_margin,
                    "decision": decision,
                    "price": safe_price,
                    "units": 0,
                }
                append_eval_log(eval_log_path, eval_record)
            except Exception as e:
                if getattr(self, "is_replay", False):
                    raise RuntimeError(f"EVAL_LOG_APPEND_FAILED(block): {e}") from e
                log.warning("[EVAL_LOG_APPEND_FAILED] block: %s", e)

        self.killchain_n_entry_pred_total += 1
        self.killchain_n_above_threshold += 1

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

        # Session/vol guard (legacy; disabled in replay for slim policy)
        if not is_replay:
            allow_high_vol = bool(policy_cfg.get("allow_high_vol", True))
            allow_extreme_vol = bool(policy_cfg.get("allow_extreme_vol", False))
            allowed_vol = ("LOW", "MEDIUM") + (("HIGH",) if allow_high_vol else tuple()) + (("EXTREME",) if allow_extreme_vol else tuple())
            try:
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
                _inc_gate("candidate_risk_guard")
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
                    _inc_gate("candidate_risk_guard")
                    return None
            except Exception:
                pass

        # Units (canonical: keep base units; no sniper overlays)
        # Use side selection (chosen_side)
        side = chosen_side
        base_units = self.exec.default_units if side == "long" else -self.exec.default_units
        units_out = int(base_units)

        if units_out == 0:
            self._killchain_inc_reason("BLOCK_RISK")
            return None

        # Proof: side selection
        try:
            if not hasattr(self, "_side_selection_log_count"):
                self._side_selection_log_count = 0
            self._side_selection_log_count += 1
            if self._side_selection_log_count <= 5 or (self._side_selection_log_count % 5000) == 0:
                log.info(
                    "[ENTRY_SIDE_SELECTION_PROOF] p_long=%.6f p_short=%.6f p_flat=%.6f winner=%s selected_side=%s",
                    float(entry_pred.prob_long),
                    float(entry_pred.prob_short),
                    float(entry_pred.prob_flat),
                    winner,
                    side,
                )
        except Exception:
            pass

        self.killchain_n_after_risk_sizing += 1
        self.entry_telemetry["n_candidate_pass"] += 1

        # ENTRY_ONLY mode: log and stop
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

        current_bar_index = int(len(candles)) if candles is not None else 0
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
        # Bar-based entry index for exit timing (bar-count, not wall clock)
        trade.entry_bar_index = current_bar_index

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
                    entry_bid=float(entry_bid_price) if entry_bid_price is not None else None,
                    entry_ask=float(entry_ask_price) if entry_ask_price is not None else None,
                    entry_spread_bps=spread_bps,
                    entry_price_used=trade.entry_price,
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
        eval_log_path = getattr(self, "eval_log_path", None) or getattr(self._runner, "eval_log_path", None)
        if eval_log_path:
            try:
                from gx1.execution.oanda_demo_runner import append_eval_log

                xgb_signal7 = getattr(self._runner, "_last_xgb_signal7", None)
                if not isinstance(xgb_signal7, dict):
                    xgb_signal7 = {}

                safe_price = np.nan
                try:
                    safe_price = float(getattr(entry_bundle, "close_price", np.nan))
                except Exception:
                    safe_price = np.nan

                eval_record = {
                    "ts_utc": now_ts.isoformat(),
                    "session": str(getattr(entry_pred, "session", policy_state.get("session", "UNKNOWN"))),
                    "p_long": float(entry_pred.prob_long),
                    "p_short": float(entry_pred.prob_short),
                    "p_flat": float(signal7_now.get("p_flat", np.nan)),
                    "p_hat": float(getattr(entry_pred, "p_hat", np.nan)),
                    "uncertainty_score": float(signal7_now.get("uncertainty_score", np.nan)),
                    "margin": float(getattr(entry_pred, "margin", np.nan)),
                    "margin_top1_top2": float(signal7_now.get("margin_top1_top2", np.nan)),
                    "entropy": float(signal7_now.get("entropy", np.nan)),
                    # Raw XGB signal7 (pre-ENTRY) for audit
                    "xgb_p_long": float(xgb_signal7.get("p_long", np.nan)),
                    "xgb_p_short": float(xgb_signal7.get("p_short", np.nan)),
                    "xgb_p_flat": float(xgb_signal7.get("p_flat", np.nan)),
                    "xgb_p_hat": float(xgb_signal7.get("p_hat", np.nan)),
                    "xgb_uncertainty_score": float(xgb_signal7.get("uncertainty_score", np.nan)),
                    "xgb_margin_top1_top2": float(xgb_signal7.get("margin_top1_top2", np.nan)),
                    "xgb_entropy": float(xgb_signal7.get("entropy", np.nan)),
                    # Entry signal7 (post-ENTRY transformer) for audit
                    "entry_p_long": float(signal7_now.get("p_long", np.nan)),
                    "entry_p_short": float(signal7_now.get("p_short", np.nan)),
                    "entry_p_flat": float(signal7_now.get("p_flat", np.nan)),
                    "entry_p_hat": float(signal7_now.get("p_hat", np.nan)),
                    "entry_uncertainty_score": float(signal7_now.get("uncertainty_score", np.nan)),
                    "entry_margin_top1_top2": float(signal7_now.get("margin_top1_top2", np.nan)),
                    "entry_entropy": float(signal7_now.get("entropy", np.nan)),
                    "pre_gate_pref": pre_gate_pref,
                    "decision_reason": reason,
                    "flat_gate": flat_gate,
                    "flat_margin": flat_margin,
                    "decision": side.upper(),
                    "price": safe_price,
                    "units": int(base_units),
                }
                append_eval_log(eval_log_path, eval_record)

                # Proof line (first 3 samples) to ensure xgb_* and entry_* are both present and differ
                if not hasattr(self, "_xgb_entry_signal7_log_count"):
                    self._xgb_entry_signal7_log_count = 0
                if self._xgb_entry_signal7_log_count < 3:
                    self._xgb_entry_signal7_log_count += 1
                    log.info(
                        "[XGB_SIGNAL7_LOGGED] xgb_p_long=%.6f entry_p_long=%.6f xgb_p_flat=%.6f entry_p_flat=%.6f",
                        eval_record.get("xgb_p_long", float("nan")),
                        eval_record.get("entry_p_long", float("nan")),
                        eval_record.get("xgb_p_flat", float("nan")),
                        eval_record.get("entry_p_flat", float("nan")),
                    )
                    log.info(
                        "[ENTRY_SIGNAL7_CHAIN] xgb_margin=%.6f entry_margin=%.6f entry_prob_neutral=%.6f",
                        eval_record.get("xgb_margin_top1_top2", float("nan")),
                        eval_record.get("entry_margin_top1_top2", float("nan")),
                        float(getattr(entry_pred, "prob_neutral", np.nan)),
                    )

                pred_trace_path_str = os.environ.get("GX1_PRED_TRACE_PATH", "")
                if pred_trace_path_str:
                    head_val = os.environ.get("GX1_PRED_TRACE_HEAD", "").strip()
                    horizon_val = os.environ.get("GX1_PRED_TRACE_HORIZON", "").strip()
                    if not head_val or not horizon_val:
                        raise RuntimeError("[PRED_TRACE] missing GX1_PRED_TRACE_HEAD or GX1_PRED_TRACE_HORIZON env")
                    try:
                        horizon_int = int(horizon_val)
                    except Exception:
                        raise RuntimeError(f"[PRED_TRACE] invalid GX1_PRED_TRACE_HORIZON={horizon_val!r}")

                    p_long = float(entry_pred.prob_long)
                    p_short = float(entry_pred.prob_short)
                    p_flat = float(
                        getattr(
                            entry_pred,
                            "prob_neutral",
                            getattr(entry_pred, "prob_flat", max(0.0, 1.0 - p_long - p_short)),
                        )
                    )

                    trace_row = {
                        "ts_utc": eval_record["ts_utc"],
                        "head": head_val,
                        "horizon_bars": horizon_int,
                        "session": eval_record["session"],
                        "p_long": p_long,
                        "p_short": p_short,
                        "p_flat": p_flat,
                        "xgb_p_long": float(eval_record.get("xgb_p_long", float("nan"))),
                        "xgb_p_short": float(eval_record.get("xgb_p_short", float("nan"))),
                        "xgb_p_flat": float(eval_record.get("xgb_p_flat", float("nan"))),
                        "entry_p_long": float(eval_record.get("entry_p_long", float("nan"))),
                        "entry_p_short": float(eval_record.get("entry_p_short", float("nan"))),
                        "entry_p_flat": float(eval_record.get("entry_p_flat", float("nan"))),
                        "p_hat": eval_record["p_hat"],
                        "uncertainty_score": eval_record["uncertainty_score"],
                        "margin": eval_record["margin"],
                        "margin_top1_top2": eval_record["margin_top1_top2"],
                        "entropy": eval_record["entropy"],
                        "xgb_short_minus_long": float(eval_record.get("xgb_p_short", float("nan")))
                        - float(eval_record.get("xgb_p_long", float("nan"))),
                        "entry_short_minus_long": float(eval_record.get("entry_p_short", float("nan")))
                        - float(eval_record.get("entry_p_long", float("nan"))),
                        "decision": eval_record["decision"],
                    }
                    pred_trace_path = Path(pred_trace_path_str)
                    pred_trace_path.parent.mkdir(parents=True, exist_ok=True)
                    with pred_trace_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(trace_row, separators=(",", ":")) + "\n")
            except Exception:
                pass

        # Minimal extra annotations (kept)
        trade.extra["session"] = session_key
        trade.extra["session_entry"] = session_key
        trade.extra["trade_id"] = trade.trade_id
        trade.extra["risk_guard_identity"] = self.risk_guard_identity

        return trade
