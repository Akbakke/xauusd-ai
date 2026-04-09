from __future__ import annotations

import os
import json
import math
import csv
import hashlib
import time
from collections import deque
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pathlib import Path
from datetime import datetime, timezone, date, time as dtime

import logging
import pandas as pd
import numpy as np
from gx1.exits.contracts.exit_io_v1_ctx36 import (
    EXIT_IO_V1_CTX36_FEATURES,
    assert_exit_io_v1_ctx36_contract,
    compute_feature_names_hash,
    exit_feature_index_v1_ctx36,
)
from gx1.exits.contracts.exit_io_v3_ctx36_m1l512_phase5 import (
    EXIT_IO_V3_CTX36_M1L512_PHASE5_EXTRA_FEATURES,
    compute_m5_phase_onehot,
)
from gx1.exits.contracts.registry import get_exit_io_contract

# DEL 3: PREBUILT mode fix - move live_features imports to lazy imports
# live_features is forbidden in PREBUILT mode, so we only import it when needed (live mode only)
# These functions are used at runtime, so we import them locally where needed
from gx1.utils.pnl import compute_pnl_bps
from gx1.execution.live_features import infer_session_tag
if TYPE_CHECKING:
    from gx1.execution.oanda_demo_runner import GX1DemoRunner

log = logging.getLogger(__name__)

EXIT_FEATURE_GROUP_SIGNAL = EXIT_IO_V1_CTX36_FEATURES[:7]
EXIT_FEATURE_GROUP_ENTRY_SNAPSHOT = EXIT_IO_V1_CTX36_FEATURES[7:12]
EXIT_FEATURE_GROUP_TRADE_STATE = [
    "pnl_bps_now",
    "mfe_bps",
    "mae_bps",
    "dd_from_mfe_bps",
    "distance_from_peak_mfe_bps",
    "bars_held",
    "time_since_mfe_bars",
    "mfe_decay_rate",
    "pnl_velocity",
    "pnl_acceleration",
    "rolling_slope_since_entry",
    "atr_bps_now",
]
EXIT_FEATURE_GROUP_EXIT_SPECIFIC = ["giveback_ratio", "giveback_acceleration"]
EXIT_FEATURE_GROUP_CTX_CAT = [
    "session_id",
    "trend_regime_id",
    "vol_regime_id",
    "atr_bucket",
    "spread_bucket",
    "H4_trend_sign_cat",
]
_ctx_start = (
    len(EXIT_FEATURE_GROUP_SIGNAL)
    + len(EXIT_FEATURE_GROUP_ENTRY_SNAPSHOT)
    + len(EXIT_FEATURE_GROUP_TRADE_STATE)
    + len(EXIT_FEATURE_GROUP_EXIT_SPECIFIC)
)
_ctx_end = len(EXIT_IO_V1_CTX36_FEATURES) - len(EXIT_FEATURE_GROUP_CTX_CAT)
EXIT_FEATURE_GROUP_CTX_CONT = EXIT_IO_V1_CTX36_FEATURES[
    _ctx_start:_ctx_end
]
_EXIT_FEATURE_GROUPS = {
    "signal": EXIT_FEATURE_GROUP_SIGNAL,
    "entry_snapshot": EXIT_FEATURE_GROUP_ENTRY_SNAPSHOT,
    "trade_state": EXIT_FEATURE_GROUP_TRADE_STATE,
    "exit_specific": EXIT_FEATURE_GROUP_EXIT_SPECIFIC,
    "ctx_cont": EXIT_FEATURE_GROUP_CTX_CONT,
    "ctx_cat": EXIT_FEATURE_GROUP_CTX_CAT,
}
_EXIT_FEATURE_GROUP_LOOKUP = {name: grp for grp, names in _EXIT_FEATURE_GROUPS.items() for name in names}
_EXIT_FEATURE_GROUP_COUNTS = {grp: len(names) for grp, names in _EXIT_FEATURE_GROUPS.items()}
_EXIT_FEATURE_GROUP_COUNTS["input_dim"] = len(EXIT_IO_V1_CTX36_FEATURES)


def _build_active_exit_feature_groups(feature_names: List[str]) -> Dict[str, List[str]]:
    groups = {
        "signal": list(EXIT_FEATURE_GROUP_SIGNAL),
        "entry_snapshot": list(EXIT_FEATURE_GROUP_ENTRY_SNAPSHOT),
        "trade_state": list(EXIT_FEATURE_GROUP_TRADE_STATE),
        "exit_specific": list(EXIT_FEATURE_GROUP_EXIT_SPECIFIC),
        "ctx_cont": list(EXIT_FEATURE_GROUP_CTX_CONT),
        "ctx_cat": list(EXIT_FEATURE_GROUP_CTX_CAT),
    }
    extras = [name for name in feature_names if name not in _EXIT_FEATURE_GROUP_LOOKUP]
    if extras:
        if all(name in EXIT_IO_V3_CTX36_M1L512_PHASE5_EXTRA_FEATURES for name in extras):
            groups["m5_phase"] = extras
        else:
            groups["extra"] = extras
    return groups


def _is_closed_window_utc(ts: pd.Timestamp) -> bool:
    h = ts.hour
    m = ts.minute
    return (h == 21 and m >= 55) or (h == 22)


_M1_EXPECTED_GAP_WINDOWS = {
    # Early-close / holiday reopen windows observed in the raw M1 lane
    date(2025, 1, 20): [(dtime(18, 25), dtime(23, 5), 12_000.0, 13_100.0, "EARLY_CLOSE_2025_01_20")],
    date(2025, 2, 17): [(dtime(18, 25), dtime(23, 5), 12_000.0, 13_100.0, "EARLY_CLOSE_2025_02_17")],
    date(2025, 5, 26): [(dtime(18, 25), dtime(22, 5), 12_000.0, 13_100.0, "EARLY_CLOSE_2025_05_26")],
    date(2025, 6, 19): [(dtime(18, 25), dtime(22, 5), 12_000.0, 13_100.0, "EARLY_CLOSE_2025_06_19")],
    date(2025, 9, 1): [(dtime(18, 25), dtime(22, 5), 12_000.0, 13_100.0, "EARLY_CLOSE_2025_09_01")],
    date(2026, 1, 19): [(dtime(18, 25), dtime(23, 5), 12_000.0, 13_100.0, "EARLY_CLOSE_2026_01_19")],
    date(2026, 2, 16): [(dtime(18, 25), dtime(23, 5), 12_000.0, 13_100.0, "EARLY_CLOSE_2026_02_16")],
    # Maintenance windows / micro-holes seen in raw M1
    date(2025, 3, 20): [(dtime(22, 35), dtime(22, 45), 60.0, 600.0, "MAINT_2025_03_20_2237")],
    date(2025, 4, 6): [(dtime(22, 5), dtime(22, 15), 60.0, 600.0, "MAINT_2025_04_06_2207")],
    date(2025, 4, 24): [(dtime(22, 0), dtime(22, 20), 60.0, 1_200.0, "MAINT_2025_04_24_2200")],
    date(2025, 5, 11): [(dtime(22, 0), dtime(22, 10), 60.0, 600.0, "MAINT_2025_05_11_2205")],
    date(2025, 7, 3): [(dtime(20, 15), dtime(20, 30), 60.0, 600.0, "MAINT_2025_07_03_2021")],
    date(2025, 8, 27): [(dtime(22, 50), dtime(23, 0), 60.0, 600.0, "MAINT_2025_08_27_2253")],
    date(2025, 8, 28): [(dtime(23, 20), dtime(23, 30), 60.0, 600.0, "MAINT_2025_08_28_2323")],
    date(2025, 11, 27): [
        (dtime(18, 45), dtime(22, 0), 60.0, 2_400.0, "MAINT_2025_11_27_EVENING"),
    ],
    date(2025, 11, 28): [
        (dtime(3, 0), dtime(10, 0), 60.0, 900.0, "MAINT_2025_11_28_MORNING"),
    ],
    date(2025, 12, 7): [(dtime(23, 30), dtime(23, 45), 60.0, 600.0, "MAINT_2025_12_07")],
    date(2026, 3, 27): [(dtime(16, 40), dtime(17, 10), 60.0, 1_800.0, "MAINT_2026_03_27")],
}


def _is_expected_m1_calendar_gap(prev_ts: pd.Timestamp, now_ts: pd.Timestamp, spacing_sec: float) -> Optional[str]:
    """
    Raw M1 expected gaps that should behave like the historical M5 allowance:
    daily close/reopen windows, early closes, and measured maintenance holes.
    """
    if prev_ts is None or now_ts is None:
        return None
    prev_d = prev_ts.date()
    if prev_d in _M1_EXPECTED_GAP_WINDOWS:
        prev_t = prev_ts.time()
        now_t = now_ts.time()
        for start_t, end_t, min_spacing, max_spacing, label in _M1_EXPECTED_GAP_WINDOWS[prev_d]:
            if min_spacing <= float(spacing_sec) <= max_spacing and start_t <= prev_t <= end_t and start_t <= now_t <= end_t:
                return label
    return None


def _is_expected_closed_gap(prev_ts: pd.Timestamp, now_ts: pd.Timestamp, spacing_sec: float) -> Optional[str]:
    """
    Expected daily closed-window gaps. Pattern B (20:55-22:00 UTC) often has no
    candles for 20:55 and 21:00, so raw/prebuilt can show 20:50 -> 21:05 (900s)
    instead of the full 20:55 -> 22:00 (3900s). Both are treated as expected.

    The raw M1 lane can surface slightly different close/reopen stamps than the
    canonical M5 view, so we also allow a small M1-compatible closed-window band
    around the same market-close windows.
    """
    if prev_ts is None or now_ts is None:
        return None
    m1_calendar = _is_expected_m1_calendar_gap(prev_ts, now_ts, spacing_sec)
    if m1_calendar:
        return m1_calendar
    if 3500 <= spacing_sec <= 4300:
        if prev_ts.hour == 21 and prev_ts.minute >= 55 and now_ts.hour == 23 and now_ts.minute <= 5:
            return "A_M1"
        if prev_ts.hour == 20 and prev_ts.minute >= 55 and now_ts.hour == 22 and now_ts.minute <= 5:
            return "B_M1"
    if spacing_sec == 3900 and prev_ts.hour == 21 and prev_ts.minute == 55 and now_ts.hour == 23 and now_ts.minute == 0:
        return "A"
    if spacing_sec == 3900 and prev_ts.hour == 20 and prev_ts.minute == 55 and now_ts.hour == 22 and now_ts.minute == 0:
        return "B"
    # Partial B: missing 20:55 and 21:00 bars only (data omits first 2 bars of closed window)
    if float(spacing_sec) == 900.0:
        prev_u = prev_ts.tz_convert("UTC") if prev_ts.tzinfo else prev_ts.tz_localize("UTC")
        now_u = now_ts.tz_convert("UTC") if now_ts.tzinfo else now_ts.tz_localize("UTC")
        if (
            prev_u.date() == now_u.date()
            and prev_u.hour == 20
            and prev_u.minute == 50
            and now_u.hour == 21
            and now_u.minute == 5
        ):
            return "B"
    return None


def _is_weekend_gap(prev_ts: pd.Timestamp, now_ts: pd.Timestamp, spacing_sec: float) -> bool:
    if prev_ts is None or now_ts is None:
        return False
    if spacing_sec < 24 * 3600:
        return False
    pw = prev_ts.weekday()
    nw = now_ts.weekday()
    if pw == 4 and (nw == 6 or nw == 0):
        return True
    if pw == 5 and (nw == 6 or nw == 0):
        return True
    return False


def _is_holiday_gap(prev_ts: pd.Timestamp, now_ts: pd.Timestamp, spacing_sec: float) -> Optional[str]:
    if prev_ts is None or now_ts is None:
        return None
    if (
        prev_ts.year == 2025
        and prev_ts.month == 4
        and prev_ts.day == 17
        and now_ts.year == 2025
        and now_ts.month == 4
        and now_ts.day == 20
        and spacing_sec >= 48 * 3600
    ):
        return "EASTER_2025"
    return None


def _holiday_or_maint_gap_name(prev_ts: pd.Timestamp, now_ts: pd.Timestamp) -> Optional[str]:
    """
    Allowlist of measured holiday/early-close/maintenance gaps in BASE28 2025.
    Matches exact prev/now timestamps (UTC). Returns name if matched else None.
    """
    if prev_ts is None or now_ts is None:
        return None
    key = (
        prev_ts.year,
        prev_ts.month,
        prev_ts.day,
        prev_ts.hour,
        prev_ts.minute,
        now_ts.year,
        now_ts.month,
        now_ts.day,
        now_ts.hour,
        now_ts.minute,
    )
    allow = {
        (2025, 12, 24, 18, 40, 2025, 12, 25, 23, 0): "XMAS_2025",
        (2025, 1, 20, 19, 25, 2025, 1, 20, 23, 0): "EARLY_CLOSE_2025_01_20",
        (2025, 2, 17, 19, 25, 2025, 2, 17, 23, 0): "EARLY_CLOSE_2025_02_17",
        (2025, 3, 7, 21, 55, 2025, 3, 9, 22, 0): "WEEKEND_2025_03_07",
        (2025, 4, 17, 20, 55, 2025, 4, 20, 22, 0): "EASTER_2025",
        (2025, 4, 24, 22, 0, 2025, 4, 24, 22, 10): "MAINT_2025_04_24_2200",
        (2025, 5, 26, 18, 25, 2025, 5, 26, 22, 0): "EARLY_CLOSE_2025_05_26",
        (2025, 6, 19, 18, 25, 2025, 6, 19, 22, 0): "EARLY_CLOSE_2025_06_19",
        (2025, 7, 4, 16, 55, 2025, 7, 6, 22, 0): "HOLIDAY_2025_07_04",
        (2025, 9, 1, 18, 25, 2025, 9, 1, 22, 0): "EARLY_CLOSE_2025_09_01",
        (2025, 10, 10, 20, 55, 2025, 10, 12, 22, 5): "WEEKEND_2025_10_10",
        (2025, 10, 31, 20, 55, 2025, 11, 2, 23, 0): "WEEKEND_2025_10_31",
        (2025, 11, 27, 19, 25, 2025, 11, 27, 19, 35): "MAINT_2025_11_27_1925",
        (2025, 11, 27, 20, 50, 2025, 11, 27, 21, 5): "MAINT_2025_11_27_2050",
        (2025, 11, 27, 21, 10, 2025, 11, 27, 21, 20): "MAINT_2025_11_27_2110",
        (2025, 11, 27, 21, 20, 2025, 11, 27, 21, 55): "MAINT_2025_11_27",
        (2025, 12, 7, 23, 0, 2025, 12, 7, 23, 35): "MAINT_2025_12_07",
        (2025, 11, 28, 8, 10, 2025, 11, 28, 8, 25): "MAINT_2025_11_28_AM_GAP",
        (2025, 11, 28, 8, 25, 2025, 11, 28, 8, 35): "MAINT_2025_11_28_AM_GAP_0825",
        (2025, 11, 28, 8, 50, 2025, 11, 28, 9, 0): "MAINT_2025_11_28_AM_GAP_0850",
        (2025, 11, 28, 9, 30, 2025, 11, 28, 9, 40): "MAINT_2025_11_28_AM_GAP_0930",
        (2025, 11, 28, 19, 45, 2025, 11, 30, 23, 0): "WEEKEND_2025_11_28",
        (2026, 1, 19, 19, 25, 2026, 1, 19, 23, 0): "EARLY_CLOSE_2026_01_19",
        (2026, 2, 16, 19, 25, 2026, 2, 16, 23, 0): "EARLY_CLOSE_2026_02_16",
        (2025, 12, 29, 9, 10, 2026, 1, 1, 23, 0): "NEW_YEAR_2025_2026",
    }
    return allow.get(key)

class ExitManager:
    def __init__(self, runner: "GX1DemoRunner") -> None:
        object.__setattr__(self, "_runner", runner)
        try:
            assert_exit_io_v1_ctx36_contract()
        except Exception:
            # Propagate to fail fast; ensures contract checked even if import guard skipped
            raise
        exit_contract = get_exit_io_contract(getattr(self._runner, "exit_transformer_io_version", None))
        self._exit_io_version = str(exit_contract["io_version"])
        self._exit_feature_names = list(exit_contract["feature_names"])
        self._exit_feature_to_index = {name: idx for idx, name in enumerate(self._exit_feature_names)}
        self._exit_feature_groups = _build_active_exit_feature_groups(self._exit_feature_names)
        self._exit_feature_group_lookup = {
            name: grp for grp, names in self._exit_feature_groups.items() for name in names
        }
        self._exit_feature_group_counts = {
            grp: len(names) for grp, names in self._exit_feature_groups.items()
        }
        self._exit_feature_group_counts["input_dim"] = len(self._exit_feature_names)
        self._exit_input_dim = int(exit_contract["feature_count"])
        self._exit_feature_hash = str(exit_contract["feature_hash"])
        self._exit_window_len = int(getattr(self._runner, "exit_transformer_window_len", exit_contract["default_window_len"]))
        if not getattr(self, "_exit_io_contract_proof_logged", False):
            computed_hash = compute_feature_names_hash(self._exit_feature_names)
            log.info(
                "[EXIT_IO_CONTRACT_PROOF] io_version=%s input_dim=%d feature_hash=%s window_len=%d",
                self._exit_io_version,
                self._exit_input_dim,
                computed_hash,
                self._exit_window_len,
            )
            self._exit_io_contract_proof_logged = True
        self._exit_input_audit_logged_once = False
        self._exit_input_audit_samples_count = 0
        self._exit_ctx_audit_logged_once = False
        self._exit_ctx_event_counts = {"exit_ml_event": 0, "exit_io_event": 0}
        self._exit_ctx_event_dims_logged = False
        self._exit_effective_cfg_logged = False
        self._exit_prob_n = 0
        self._exit_prob_ge_thr = 0
        self._exit_prob_min = float("inf")
        self._exit_prob_max = float("-inf")
        self._exit_prob_buckets = {
            "<0.001": 0,
            "<0.01": 0,
            "<0.05": 0,
            "<0.1": 0,
            "<0.2": 0,
            ">=0.2": 0,
        }
        # Exit prob ranking audit (per-trade, per-bar)
        self._exit_prob_all: list[float] = []
        self._exit_logit_all: list[float] = []
        self._exit_prob_inc = 0
        self._exit_prob_dec = 0
        self._exit_prob_flat = 0
        self._exit_prob_last_by_trade: dict[str, float] = {}
        self._exit_prob_trade_states: dict[str, list[tuple[int, float, float]]] = {}
        self._exit_prob_at_entry: list[float] = []
        self._exit_prob_at_exit: list[float] = []
        self._exit_prob_before_exit: dict[int, list[float]] = {1: [], 3: [], 5: []}
        self._exit_prob_at_mfe_peak: list[float] = []
        self._exit_prob_at_mae_peak: list[float] = []
        self._exit_prob_group_a: list[float] = []
        self._exit_prob_group_b: list[float] = []
        self._exit_prob_ranking_audit_written = False
        self._exit_prob_score_audit_written = False
        self._exit_runtime_sample_proof_logged = False
        self._exit_prob_sample_proof_logged = False
        self._exit_eval_trace_header_written = False
        self._exit_eval_trace_path: Optional[Path] = None
        self._exit_prob_bars_before_exit: dict[str, list[float]] = {
            "b1": [],
            "b2": [],
            "b3": [],
            "b4_5": [],
            "b_gt5": [],
        }
        self._exit_feature_vectors: list[list[float]] = []
        self._exit_feature_sample_trades: set[str] = set()
        # Exit eval flow counters (replay proof, not per-bar spam)
        self._exit_eval_flow = {
            "trade_active_count": 0,
            "eligible_count": 0,
            "blocked_pre_eval_count": 0,
            "model_eval_attempt_count": 0,
            "model_eval_complete_count": 0,
            "threshold_check_count": 0,
        }
        self._exit_eval_blocked_reasons: dict[str, int] = {}
        # Time-stop is explicitly disabled in active EXIT path: exits are edge/event driven.
        self._exit_max_bars_held = 0
        # Catastrophic-loss guard (runtime safety exception) is config-owned in the
        # active canonical lane; keep the same numeric defaults as final fallback.
        self._exit_cat_guard_enabled = bool(getattr(self._runner, "exit_cat_guard_enabled", True))
        # MFE in exit scalars is often near-zero on catastrophic givebacks; keep this permissive.
        self._exit_cat_guard_mfe_bps = float(getattr(self._runner, "exit_cat_guard_mfe_bps", 1.0))
        self._exit_cat_guard_dd_bps = float(getattr(self._runner, "exit_cat_guard_dd_bps", 60.0))
        self._exit_cat_guard_giveback_ratio = float(getattr(self._runner, "exit_cat_guard_giveback_ratio", 1.0))
        self._exit_cat_guard_pnl_max = float(getattr(self._runner, "exit_cat_guard_pnl_max", 0.0))
        self._exit_cat_guard_pnl_frac_mfe = float(getattr(self._runner, "exit_cat_guard_pnl_frac_mfe", 0.0))
        self._exit_cat_guard_triggers = 0
        # Threshold event-arm knobs (used for observability + event arming only).
        self._exit_edge_death_mfe_min_short = float(getattr(self._runner, "exit_edge_death_mfe_min_short", 6.0))
        self._exit_edge_death_mfe_min_long = float(getattr(self._runner, "exit_edge_death_mfe_min_long", 8.0))
        self._exit_edge_death_dd_min_short = float(getattr(self._runner, "exit_edge_death_dd_min_short", 8.0))
        self._exit_edge_death_dd_min_long = float(getattr(self._runner, "exit_edge_death_dd_min_long", 10.0))
        self._exit_edge_death_gb_min_short = float(getattr(self._runner, "exit_edge_death_gb_min_short", 0.45))
        self._exit_edge_death_gb_min_long = float(getattr(self._runner, "exit_edge_death_gb_min_long", 0.50))
        exit_cfg = getattr(self, "exit_params", {}) or {}
        exit_tf_cfg = (
            (exit_cfg.get("exit", {}) or {})
            .get("params", {})
            .get("exit_ml", {})
            .get("exit_transformer", {})
        ) if isinstance(exit_cfg, dict) else {}
        post_edge_cfg = (exit_tf_cfg.get("selective_post_edge_separator", {}) or {}) if isinstance(exit_tf_cfg, dict) else {}
        self._exit_post_edge_enabled = bool(post_edge_cfg.get("enabled", False))
        self._exit_post_edge_score_source = str(post_edge_cfg.get("score_source", "exit_prob") or "exit_prob").strip()
        self._exit_post_edge_score_min = float(post_edge_cfg.get("score_min", 0.70))
        self._exit_post_edge_require_model_below_main_threshold = bool(
            post_edge_cfg.get("require_model_below_main_threshold", True)
        )
        self._exit_post_edge_require_threshold_event = bool(post_edge_cfg.get("require_threshold_event", True))
        self._exit_post_edge_allowed_sides = {
            str(x).strip().lower() for x in (post_edge_cfg.get("allowed_sides", ["long"]) or ["long"])
        }
        self._exit_post_edge_allowed_sessions = {
            str(x).strip().upper() for x in (post_edge_cfg.get("allowed_sessions", ["OVERLAP"]) or ["OVERLAP"])
        }
        self._exit_post_edge_mfe_min = float(post_edge_cfg.get("mfe_bps_min", 60.0))
        self._exit_post_edge_dd_min = float(post_edge_cfg.get("dd_from_mfe_bps_min", 55.0))
        self._exit_post_edge_gb_min = float(post_edge_cfg.get("giveback_ratio_min", 0.75))
        self._exit_post_edge_pnl_min = float(post_edge_cfg.get("pnl_bps_min", 2.0))
        self._exit_post_edge_pnl_max = float(post_edge_cfg.get("pnl_bps_max", 10.0))
        self._exit_post_edge_pnl_frac_mfe_max = float(post_edge_cfg.get("pnl_frac_mfe_max", 0.45))
        self._exit_post_edge_time_since_mfe_min = float(post_edge_cfg.get("time_since_mfe_bars_min", 8.0))
        self._exit_post_edge_directional_edge_max = float(post_edge_cfg.get("directional_edge_max", 0.02))
        self._exit_post_edge_triggers = 0
        self._exit_protected_profit_close_triggers = 0

    def __getattr__(self, name: str):
        return getattr(self._runner, name)

    def __setattr__(self, name: str, value):
        # ExitManager owns all private attrs (including _runner); public attrs proxy to runner
        if name == "_runner" or name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        setattr(self._runner, name, value)

    def _get_expected_ctx_dims(self) -> tuple[int, int]:
        runner = self._runner
        cont = getattr(runner, "ctx_cont_dim", None)
        cat = getattr(runner, "ctx_cat_dim", None)
        if cont is None or cat is None:
            try:
                bundle = getattr(runner, "entry_v10_bundle", None)
                meta = getattr(bundle, "metadata", None) if bundle is not None else None
                if meta:
                    cont = meta.get("ctx_cont_dim") or meta.get("expected_ctx_cont_dim")
                    cat = meta.get("ctx_cat_dim") or meta.get("expected_ctx_cat_dim")
            except Exception:
                pass
        if cont is None or cat is None:
            if bool(getattr(runner, "replay_mode", False)) or os.getenv("GX1_RUN_MODE", "").upper() == "TRUTH":
                raise RuntimeError(
                    "[EXIT_CTX_DIM_RESOLVE_FAIL] missing ctx dims from runner/bundle metadata in active replay/truth path"
                )
            from gx1.contracts.signal_bridge_v1 import get_canonical_ctx_contract
            canonical = get_canonical_ctx_contract()
            cont = len(canonical.get("ctx_cont_names") or [])
            cat = len(canonical.get("ctx_cat_names") or [])
        return int(cont), int(cat)

    def _exit_feature_index(self, name: str) -> int:
        try:
            return int(self._exit_feature_to_index[name])
        except KeyError as exc:
            raise KeyError(f"[EXIT_CONTRACT] unknown active exit feature: {name}") from exc

    def _log_exit_ctx_event_proof(self, event_type: str, ctx_cont: List[float], ctx_cat: List[int]) -> None:
        try:
            if event_type in self._exit_ctx_event_counts:
                self._exit_ctx_event_counts[event_type] += 1
            if not self._exit_ctx_event_dims_logged:
                log.info(
                    "[EXIT_CTX_EVENT_PROOF] event_type=%s ctx_cont_dim=%d ctx_cat_dim=%d count=%d",
                    event_type,
                    len(ctx_cont),
                    len(ctx_cat),
                    int(self._exit_ctx_event_counts.get(event_type, 0)),
                )
                self._exit_ctx_event_dims_logged = True
        except Exception:
            return
    

    def evaluate_and_close_trades(self, candles: pd.DataFrame) -> None:
        """
        Evaluate and close trades using the canonical exit transformer path.

        Raises if the configured exit_mode deviates from exit_transformer_v0 to
        enforce single-universe execution without fallbacks.
        """
        # Skip exit evaluation in ENTRY_ONLY mode (no trades to close)
        if self.mode == "ENTRY_ONLY":
            return
        
        if not self._exit_effective_cfg_logged:
            runner = self._runner
            arb = getattr(runner, "exit_control", None) or {}
            arb_allow = getattr(arb, "allow_model_exit_when", {}) if hasattr(arb, "allow_model_exit_when") else arb.get("allow_model_exit_when", {}) if isinstance(arb, dict) else {}
            protected_profit_floor_bps = None
            if isinstance(arb_allow, dict):
                protected_profit_floor_bps = arb_allow.get("min_realized_pnl_bps", arb_allow.get("min_pnl_bps"))
            log.info(
                "[EXIT_EFFECTIVE_CFG] threshold=%.4f require_consecutive=%d arb_min_pnl_bps=%s arb_min_realized_pnl_bps=%s be_plus_floor_bps=%s arb_min_exit_prob=%s arb_exit_prob_hysteresis=%s arb_min_bars=%s",
                float(getattr(runner, "exit_threshold", 0.0)),
                int(getattr(runner, "exit_require_consecutive", 1)),
                arb_allow.get("min_pnl_bps"),
                arb_allow.get("min_realized_pnl_bps"),
                protected_profit_floor_bps,
                arb_allow.get("min_exit_prob"),
                arb_allow.get("exit_prob_hysteresis"),
                arb_allow.get("min_bars"),
            )
            log.info(
                "[EXIT_GUARD_CFG] cata={mfe>=%.1f dd>=%.1f gb>=%.2f pnl_max<=%.2f pnl_frac<=%.2f} threshold_event={long:mfe>=%.1f dd>=%.1f gb>=%.2f,short:mfe>=%.1f dd>=%.1f gb>=%.2f}",
                float(self._exit_cat_guard_mfe_bps),
                float(self._exit_cat_guard_dd_bps),
                float(self._exit_cat_guard_giveback_ratio),
                float(self._exit_cat_guard_pnl_max),
                float(self._exit_cat_guard_pnl_frac_mfe),
                float(self._exit_edge_death_mfe_min_long),
                float(self._exit_edge_death_dd_min_long),
                float(self._exit_edge_death_gb_min_long),
                float(self._exit_edge_death_mfe_min_short),
                float(self._exit_edge_death_dd_min_short),
                float(self._exit_edge_death_gb_min_short),
            )
            log.info(
                "[EXIT_GUARD_MODE] model_owned_normal_exits=1 runtime_safety_only=1 sideguards_active=0 cata_active=%s",
                bool(getattr(self, "_exit_cat_guard_enabled", False)),
            )
            weekly_mgmt = getattr(runner, "exit_weekly_management", None)
            log.info(
                "[EXIT_POLICY_MGMT_CFG] friday_flat_enabled=%s friday_flat_weekdays=%s friday_flat_cutoff_hhmm_utc=%s friday_flat_reason=%s",
                bool(getattr(weekly_mgmt, "friday_flat_enabled", False)) if weekly_mgmt is not None else False,
                list(getattr(weekly_mgmt, "friday_flat_weekdays", ()) or ()) if weekly_mgmt is not None else [],
                None
                if weekly_mgmt is None or getattr(weekly_mgmt, "friday_flat_cutoff_hhmm_utc", None) is None
                else f"{weekly_mgmt.friday_flat_cutoff_hhmm_utc[0]:02d}:{weekly_mgmt.friday_flat_cutoff_hhmm_utc[1]:02d}",
                None if weekly_mgmt is None else getattr(weekly_mgmt, "friday_flat_reason", None),
            )
            log.info(
                "[EXIT_POST_EDGE_CFG] enabled=%s score_source=%s score_min=%.3f require_model_below_main_threshold=%s require_threshold_event=%s allowed_sides=%s allowed_sessions=%s mfe>=%.1f dd>=%.1f gb>=%.2f pnl_range=%.1f..%.1f pnl_frac_mfe<=%.2f tsm>=%.1f directional_edge<=%.3f be_plus_floor_bps=%s",
                bool(getattr(self, "_exit_post_edge_enabled", False)),
                str(getattr(self, "_exit_post_edge_score_source", "exit_prob")),
                float(getattr(self, "_exit_post_edge_score_min", 0.70)),
                bool(getattr(self, "_exit_post_edge_require_model_below_main_threshold", True)),
                bool(getattr(self, "_exit_post_edge_require_threshold_event", True)),
                sorted(getattr(self, "_exit_post_edge_allowed_sides", {"long"})),
                sorted(getattr(self, "_exit_post_edge_allowed_sessions", {"OVERLAP"})),
                float(getattr(self, "_exit_post_edge_mfe_min", 60.0)),
                float(getattr(self, "_exit_post_edge_dd_min", 55.0)),
                float(getattr(self, "_exit_post_edge_gb_min", 0.75)),
                float(getattr(self, "_exit_post_edge_pnl_min", 2.0)),
                float(getattr(self, "_exit_post_edge_pnl_max", 10.0)),
                float(getattr(self, "_exit_post_edge_pnl_frac_mfe_max", 0.45)),
                float(getattr(self, "_exit_post_edge_time_since_mfe_min", 8.0)),
                float(getattr(self, "_exit_post_edge_directional_edge_max", 0.02)),
                protected_profit_floor_bps,
            )
            self._exit_effective_cfg_logged = True
        
        exit_mode = (
            (getattr(self, "exit_params", {}) or {})
            .get("exit", {})
            .get("params", {})
            .get("exit_ml", {})
            .get("mode")
            if isinstance(getattr(self, "exit_params", {}), dict)
            else None
        )
        if exit_mode != "exit_transformer_v0":
            raise RuntimeError("[EXIT_CONTRACT] only exit_transformer_v0 supported in canonical truth")
        
        # Tick-exit handling (no FARM paths)
        if self.tick_cfg.get("enabled", False) and not self.replay_mode:
            # Normal mode: tick-exit is handled by TickWatcher thread (live mode only)
            # In replay mode, tick-exit is evaluated here if enabled
            pass
        
        # Get snapshot of open trades
        if not self.open_trades:
            return
        open_trades_copy = list(self.open_trades)
        runner_obj = getattr(self, "_runner", None)

        def _accum_runner_metric(sec_name: str, count_name: str, dt: float) -> None:
            try:
                if runner_obj is None:
                    return
                setattr(runner_obj, sec_name, float(getattr(runner_obj, sec_name, 0.0) + float(dt)))
                setattr(runner_obj, count_name, int(getattr(runner_obj, count_name, 0) + 1))
            except Exception:
                return
        
        # One-shot exit window_len report (observability only; no behavior change)
        def _maybe_log_exit_windowlen_once() -> None:
            if getattr(self, "_exit_windowlen_reported", False):
                return
            try:
                policy_window_len = None
                model_window_len = None
                model_input_dim = None
                model_io_version = None
                model_feat_hash = None
                bundle_dir = None

                # Policy hint
                try:
                    exit_cfg = getattr(self, "exit_params", {}) or {}
                    policy_window_len = (
                        exit_cfg.get("exit", {})
                        .get("params", {})
                        .get("exit_ml", {})
                        .get("exit_transformer", {})
                        .get("window_len")
                    )
                except Exception:
                    policy_window_len = None

                # Resolve bundle dir from policy model_path
                try:
                    exit_cfg = getattr(self, "exit_params", {}) or {}
                    model_path_raw = (
                        exit_cfg.get("exit", {})
                        .get("params", {})
                        .get("exit_ml", {})
                        .get("exit_transformer", {})
                        .get("model_path")
                    )
                    if model_path_raw:
                        model_path = Path(model_path_raw)
                        if not model_path.is_absolute():
                            gx1_data_root = Path(os.environ.get("GX1_DATA", "")).expanduser()
                            model_path = gx1_data_root / model_path
                        bundle_dir = model_path
                        config_path = bundle_dir / "exit_transformer_config.json"
                        with open(config_path, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                        model_window_len = cfg.get("window_len")
                        model_input_dim = cfg.get("input_dim")
                        model_io_version = cfg.get("exit_ml_io_version")
                        model_feat_hash = cfg.get("feature_names_hash")
                except Exception:
                    pass

                # Effective window_len: prefer model config if available, else policy hint
                effective_window_len = model_window_len or policy_window_len

                # Mismatch notice (no crash)
                if policy_window_len is not None and model_window_len is not None and policy_window_len != model_window_len:
                    log.info(
                        "[EXIT_WINDOWLEN_MISMATCH] policy=%s model=%s using=%s",
                        policy_window_len,
                        model_window_len,
                        model_window_len,
                    )

                # Build tensor shape guess (B, T, D) using current open trades count
                built_tensor_shape = None
                try:
                    T = int(effective_window_len) if effective_window_len is not None else None
                    D = int(model_input_dim) if model_input_dim is not None else None
                    if T and D:
                        built_tensor_shape = (1, T, D)
                except Exception:
                    built_tensor_shape = None

                # Write report to run root if available
                out_dir = getattr(self, "explicit_output_dir", None) or getattr(self, "output_dir", None)
                if out_dir:
                    report_path = Path(out_dir) / "EXIT_WINDOWLEN_REPORT.json"
                    payload = {
                        "run_id": getattr(self, "run_id", None),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "policy_window_len": policy_window_len,
                        "model_config_window_len": model_window_len,
                        "effective_window_len": effective_window_len,
                        "model_input_dim": model_input_dim,
                        "model_io_version": model_io_version,
                        "model_feature_names_hash": model_feat_hash,
                        "bundle_dir": str(bundle_dir) if bundle_dir else None,
                        "built_tensor_shape": built_tensor_shape,
                    }
                    try:
                        from gx1.utils.atomic_json import atomic_write_json

                        atomic_write_json(report_path, payload)
                    except Exception:
                        try:
                            with open(report_path, "w", encoding="utf-8") as f:
                                json.dump(payload, f, indent=2)
                        except Exception:
                            pass

                log.info(
                    "[EXIT_WINDOWLEN_REPORT] policy_window_len=%s model_config_window_len=%s effective_window_len=%s model_input_dim=%s model_io_version=%s model_feature_names_hash=%s built_tensor_shape=%s",
                    policy_window_len,
                    model_window_len,
                    effective_window_len,
                    model_input_dim,
                    model_io_version,
                    model_feat_hash,
                    built_tensor_shape,
                )
                # Additional boot proof when transformer mode declared
                exit_mode = (
                    (getattr(self, "exit_params", {}) or {})
                    .get("exit", {})
                    .get("params", {})
                    .get("exit_ml", {})
                    .get("mode")
                    if isinstance(getattr(self, "exit_params", {}), dict)
                    else None
                )
                if exit_mode == "exit_transformer_v0":
                    log.info(
                        "[EXIT_T8_PROOF_BOOT] mode=%s model_path=%s model_window_len=%s input_dim=%s feature_hash=%s",
                        exit_mode,
                        bundle_dir,
                        model_window_len,
                        model_input_dim,
                        model_feat_hash,
                    )
            finally:
                setattr(self, "_exit_windowlen_reported", True)

        _maybe_log_exit_windowlen_once()

        self._ensure_bid_ask_columns(candles, context="exit_manager")
        now_ts = candles.index[-1]
        current_bid = float(candles["bid_close"].iloc[-1])
        current_ask = float(candles["ask_close"].iloc[-1])
        current_bar_index = len(candles)
        if len(candles) >= 2:
            prev_ts = candles.index[-2]
            spacing_sec = (now_ts - prev_ts).total_seconds()
            if getattr(self, "replay_mode", False) and spacing_sec > 600:
                pattern = _is_expected_closed_gap(prev_ts, now_ts, spacing_sec)
                if pattern:
                    window = "21:55-23:00" if pattern == "A" else "20:55-22:00"
                    log.info(
                        "[CANDLE_GAP_EXPECTED_CLOSED_WINDOW_PROOF] prev_ts=%s now_ts=%s spacing_sec=%.1f window=%s pattern=%s",
                        prev_ts,
                        now_ts,
                        spacing_sec,
                        window,
                        pattern,
                    )
                else:
                    holiday_name = _is_holiday_gap(prev_ts, now_ts, spacing_sec)
                    allow_name = _holiday_or_maint_gap_name(prev_ts, now_ts)
                    if holiday_name:
                        log.info(
                            "[CANDLE_GAP_EXPECTED_HOLIDAY_PROOF] name=%s prev_ts=%s now_ts=%s spacing_sec=%.1f",
                            holiday_name,
                            prev_ts,
                            now_ts,
                            spacing_sec,
                        )
                    elif allow_name:
                        log.info(
                            "[CANDLE_GAP_EXPECTED_HOLIDAY_PROOF] name=%s prev_ts=%s now_ts=%s spacing_sec=%.1f",
                            allow_name,
                            prev_ts,
                            now_ts,
                            spacing_sec,
                        )
                    elif _is_weekend_gap(prev_ts, now_ts, spacing_sec):
                        log.info(
                            "[CANDLE_GAP_EXPECTED_WEEKEND_PROOF] prev_ts=%s now_ts=%s spacing_sec=%.1f prev_wd=%d now_wd=%d",
                            prev_ts,
                            now_ts,
                            spacing_sec,
                            prev_ts.weekday(),
                            now_ts.weekday(),
                        )
                    else:
                        log.error(
                            "[CANDLE_GAP_DETECTED] spacing_sec=%.1f prev_ts=%s now_ts=%s",
                            spacing_sec,
                            prev_ts,
                            now_ts,
                        )
                        raise RuntimeError(
                            f"[CANDLE_GAP] spacing_sec={spacing_sec} prev_ts={prev_ts} now_ts={now_ts}"
                        )
        # Update per-bar extrema once per bar for determinism
        self._update_trade_extremes_current_bar(open_trades_copy, now_ts, current_bid, current_ask)
        # Snapshot ctx once per bar using the same prebuilt row as entry ctx (if available)
        ctx_cont_snapshot: Optional[List[float]] = None
        ctx_cat_snapshot: Optional[List[int]] = None
        audit_strict = os.environ.get("GX1_EXIT_AUDIT_STRICT") == "1"
        try:
            prebuilt_df = getattr(self._runner, "prebuilt_features_df", None)
            ctx_cont_cols = getattr(self._runner, "ctx_cont_required_columns", None)
            ctx_cat_cols = getattr(self._runner, "ctx_cat_required_columns", None)
            if prebuilt_df is not None:
                if ctx_cont_cols is None or ctx_cat_cols is None:
                    from gx1.contracts.signal_bridge_v1 import get_canonical_ctx_contract
                    ctx_contract = get_canonical_ctx_contract()
                    ctx_cont_cols = ctx_contract.get("ctx_cont_names")
                    ctx_cat_cols = ctx_contract.get("ctx_cat_names")
                ctx_cont_cols = list(ctx_cont_cols or [])
                ctx_cat_cols = list(ctx_cat_cols or [])
                expected_cont_dim, expected_cat_dim = self._get_expected_ctx_dims()
                ctx_contract_mode = os.environ.get("GX1_CTX_CONTRACT", "V_NEXT").upper()
                if ctx_contract_mode == "V_NEXT":
                    extra_ctx_cont = [
                        "is_ASIA",
                        "minutes_since_session_open",
                        "minutes_to_next_session_boundary",
                        "session_change_flag",
                        "session_tradable",
                    ]
                    if all(name in ctx_cont_cols for name in extra_ctx_cont):
                        ctx_cont_cols = list(ctx_cont_cols)
                    else:
                        ctx_cont_cols = list(ctx_cont_cols) + extra_ctx_cont
                    if not getattr(self, "_ctx_vnext_audit_logged", False):
                        import hashlib
                        fp_payload = {
                            "ctx_cat_keys": list(ctx_cat_cols),
                            "ctx_cont_keys": list(ctx_cont_cols),
                            "ctx_cat_dim": int(len(ctx_cat_cols)),
                            "ctx_cont_dim": int(len(ctx_cont_cols)),
                        }
                        fp_str = json.dumps(fp_payload, sort_keys=True)
                        fp_hash = hashlib.sha256(fp_str.encode("utf-8")).hexdigest()
                        log.info(
                            "[CTX_CONTRACT_V_NEXT] ctx_cat_keys=%s ctx_cont_keys=%s ctx_cat_dim=%d ctx_cont_dim=%d fingerprint=%s",
                            list(ctx_cat_cols),
                            list(ctx_cont_cols),
                            int(len(ctx_cat_cols)),
                            int(len(ctx_cont_cols)),
                            fp_hash,
                        )
                        self._ctx_vnext_audit_logged = True
                    if len(ctx_cont_cols) != expected_cont_dim:
                        raise RuntimeError(
                            f"[CTX_CONTRACT_V_NEXT_MISMATCH] ctx_cont_dim={len(ctx_cont_cols)} expected={expected_cont_dim}"
                        )
                if os.getenv("GX1_CTX_KEYS_AUDIT", "0") == "1" and not getattr(self, "_exit_ctx_keys_audit_logged", False):
                    log.info(
                        "[EXIT_CTX_KEYS_AUDIT] ctx_cat_keys=%s ctx_cont_keys=%s ctx_cat_dim=%s ctx_cont_dim=%s",
                        list(ctx_cat_cols),
                        list(ctx_cont_cols),
                        int(expected_cat_dim),
                        int(expected_cont_dim),
                    )
                    self._exit_ctx_keys_audit_logged = True
                if (
                    len(ctx_cont_cols) == expected_cont_dim
                    and len(ctx_cat_cols) == expected_cat_dim
                    and now_ts in prebuilt_df.index
                ):
                    row = prebuilt_df.loc[now_ts]
                    session_vals = {}
                    if ctx_contract_mode == "V_NEXT":
                        try:
                            from gx1.time.session_detector import (
                                get_session_id_vectorized,
                                get_session_minutes_since_open_vectorized,
                                get_session_minutes_to_next_boundary_vectorized,
                            )
                            ts_pair = pd.Series([now_ts])
                            sess_id = int(get_session_id_vectorized(ts_pair).iloc[0])
                            prev_ts = candles.index[-2] if len(candles) >= 2 else None
                            prev_sess = None
                            if prev_ts is not None:
                                prev_sess = int(get_session_id_vectorized(pd.Series([prev_ts])).iloc[0])
                            session_vals = {
                                "is_ASIA": float(1 if sess_id == 0 else 0),
                                "minutes_since_session_open": float(get_session_minutes_since_open_vectorized(ts_pair).iloc[0]),
                                "minutes_to_next_session_boundary": float(get_session_minutes_to_next_boundary_vectorized(ts_pair).iloc[0]),
                                "session_change_flag": float(1 if (prev_sess is not None and sess_id != prev_sess) else 0),
                                "session_tradable": float(1 if sess_id != 0 else 0),
                            }
                        except Exception:
                            session_vals = {}
                    def _extract(cols):
                        out = []
                        for c in cols:
                            if c in row.index:
                                out.append(float(row[c]))
                                continue
                            if c in session_vals:
                                out.append(float(session_vals[c]))
                                continue
                            return None
                        return out
                    ctx_cont_snapshot = _extract(ctx_cont_cols)
                    ctx_cat_raw = _extract(ctx_cat_cols)
                    if (
                        ctx_cont_snapshot is not None
                        and ctx_cat_raw is not None
                        and len(ctx_cont_snapshot) == expected_cont_dim
                        and len(ctx_cat_raw) == expected_cat_dim
                    ):
                        ctx_cat_snapshot = [int(x) for x in ctx_cat_raw]
                    elif audit_strict:
                        raise RuntimeError(
                            f"[EXIT_CTX_AUDIT] ctx {expected_cont_dim}/{expected_cat_dim} missing for ts with prebuilt row present"
                        )
        except Exception:
            ctx_cont_snapshot = None
            ctx_cat_snapshot = None
            if audit_strict:
                raise
        if not getattr(self, "_exit_ctx_audit_logged_once", False):
            expected_cont_dim, expected_cat_dim = self._get_expected_ctx_dims()
            log.info(
                "[EXIT_CTX_AUDIT] have_ctx_cont=%s have_ctx_cat=%s cont_dim=%s cat_dim=%s ts=%s",
                bool(ctx_cont_snapshot and len(ctx_cont_snapshot) == expected_cont_dim),
                bool(ctx_cat_snapshot and len(ctx_cat_snapshot) == expected_cat_dim),
                expected_cont_dim,
                expected_cat_dim,
                now_ts,
            )
            self._exit_ctx_audit_logged_once = True
        if self.exit_verbose_logging:
            log.info(
                "[EXIT] Evaluating %d open trades on bar %s (replay=%s)",
                len(open_trades_copy),
                now_ts,
                self.replay_mode,
            )
        else:
            log.debug(
                "[EXIT] Evaluating %d open trades on bar %s (replay=%s)",
                len(open_trades_copy),
                now_ts,
                self.replay_mode,
            )
        closes_requested = 0
        closes_accepted = 0
        
        # Collect price trace for open trades (for intratrade metrics)
        self._collect_price_trace(candles, now_ts, open_trades_copy)

        # Legacy FARM/critic paths are intentionally removed from canonical EXIT flow.
        
        # Exit transformer path (single supported mode)
        exit_mode = (
            (getattr(self, "exit_params", {}) or {})
            .get("exit", {})
            .get("params", {})
            .get("exit_ml", {})
            .get("mode")
            if isinstance(getattr(self, "exit_params", {}), dict)
            else None
        )
        if exit_mode != "exit_transformer_v0":
            raise RuntimeError(
                "[EXIT_CONTRACT] only exit_transformer_v0 supported in canonical truth"
            )
        if exit_mode == "exit_transformer_v0":
            io_only = bool(getattr(self._runner, "exit_io_only_replay", False))
            if io_only:
                log.info(
                    "[EXIT_IO_ONLY_REPLAY_PROOF] enabled=1 io_version=%s input_dim=%d",
                    self._exit_io_version,
                    self._exit_input_dim,
                )
            decider = getattr(self, "exit_transformer_decider", None)
            if decider is None and not io_only:
                raise RuntimeError(
                    "[EXIT_MODEL_REQUIRED] Exit transformer model is required for configured exit policy; decider missing."
                )
            if (
                decider is not None
                and bool(getattr(self, "_exit_post_edge_enabled", False))
                and str(getattr(self, "_exit_post_edge_score_source", "exit_prob")) == "profit_protect_prob"
                and not bool(getattr(decider, "profit_protect_head_available", False))
            ):
                raise RuntimeError(
                    "[EXIT_POST_EDGE_HEAD_MISSING] score_source=profit_protect_prob requested but active exit bundle has no trained profit_protect head"
                )
            window_len = int(getattr(decider, "window_len", -1))
            input_dim = int(getattr(decider, "input_dim", -1))
            if io_only:
                window_len = self._exit_window_len
                input_dim = self._exit_input_dim
            if window_len != self._exit_window_len or input_dim != self._exit_input_dim:
                if bool(getattr(self._runner, "exit_hash_guard_bypass_enabled", False)):
                    log.warning(
                        "[EXIT_IO_CONTRACT_GUARD_BYPASS] expected io_version=%s window_len=%d,input_dim=%d got window_len=%d,input_dim=%d",
                        self._exit_io_version,
                        self._exit_window_len,
                        self._exit_input_dim,
                        window_len,
                        input_dim,
                    )
                else:
                    raise RuntimeError(
                        f"[EXIT_IO_CONTRACT_VIOLATION] expected io_version={self._exit_io_version} window_len={self._exit_window_len},input_dim={self._exit_input_dim} got window_len={window_len},input_dim={input_dim}"
                    )
            if not getattr(self, "_exit_input_dim_proof_logged", False):
                log.info(
                    "[EXIT_INPUT_DIM_PROOF] window_len=%d input_dim=%d",
                    window_len,
                    input_dim,
                )
                self._exit_input_dim_proof_logged = True
            if not getattr(self, "_exit_feature_vector_proof_logged", False):
                computed_hash = compute_feature_names_hash(self._exit_feature_names)
                log.info(
                    "[EXIT_FEATURE_VECTOR_PROOF] io_version=%s input_dim=%d feature_hash=%s feature_names=%s expected_hash=%s signal_count=%d ctx_cont_count=%d ctx_cat_count=%d trade_state_count=%d entry_snapshot_count=%d exit_specific_count=%d first3=%s last3=%s",
                    self._exit_io_version,
                    self._exit_input_dim,
                    computed_hash,
                    self._exit_feature_names,
                    self._exit_feature_hash,
                    self._exit_feature_group_counts.get("signal", 0),
                    self._exit_feature_group_counts.get("ctx_cont", 0),
                    self._exit_feature_group_counts.get("ctx_cat", 0),
                    self._exit_feature_group_counts.get("trade_state", 0),
                    self._exit_feature_group_counts.get("entry_snapshot", 0),
                    self._exit_feature_group_counts.get("exit_specific", 0),
                    self._exit_feature_names[:3],
                    self._exit_feature_names[-3:],
                )
                self._exit_feature_vector_proof_logged = True
            if not getattr(self, "_exit_t8_forward_logged", False):
                log.info(
                    "[EXIT_T8_PROOF_FWD] built_tensor_shape=%s n_open_trades=%s",
                    (1, window_len, input_dim),
                    len(open_trades_copy),
                )
                self._exit_t8_forward_logged = True

            for trade in open_trades_copy:
                if trade not in self.open_trades:
                    self._exit_eval_flow["blocked_pre_eval_count"] += 1
                    self._exit_eval_blocked_reasons["trade_not_open"] = (
                        self._exit_eval_blocked_reasons.get("trade_not_open", 0) + 1
                    )
                    continue
                self._exit_eval_flow["trade_active_count"] += 1
                entry_bar_index = int(getattr(trade, "entry_bar_index", current_bar_index))
                bars_in_trade_min = max(0, int(current_bar_index) - entry_bar_index)
                entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
                entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
                pnl_bps_now = compute_pnl_bps(entry_bid, entry_ask, current_bid, current_ask, trade.side)
                entry_ts = (
                    getattr(trade, "entry_time", None)
                    or getattr(trade, "open_time", None)
                    or getattr(trade, "open_ts_utc", None)
                )
                session_entry = infer_session_tag(entry_ts).upper() if entry_ts is not None else "UNKNOWN"
                session_current = infer_session_tag(now_ts).upper() if now_ts is not None else "UNKNOWN"

                def _emit_exit_eval_trace(exit_model_evaluated: bool, exit_prob: float, exit_decision: str) -> None:
                    mfe_bps = float(getattr(trade, "mfe_bps", 0.0))
                    mae_bps = float(getattr(trade, "mae_bps", 0.0))
                    mfe_last_bar = float(getattr(trade, "_mfe_last_bar", 0.0))
                    distance_from_peak_mfe_bps = max(0.0, mfe_bps - float(pnl_bps_now))
                    time_since_mfe_bars = max(0.0, float(bars_in_trade_min) - float(mfe_last_bar))
                    giveback_ratio = (distance_from_peak_mfe_bps / mfe_bps) if mfe_bps > 0 else 0.0
                    self._append_exit_eval_trace(
                        {
                            "trade_id": getattr(trade, "trade_id", None),
                            "timestamp": str(now_ts),
                            "bars_held": int(bars_in_trade_min),
                            "session_current": session_current,
                            "session_entry": session_entry,
                            "exit_model_evaluated": int(bool(exit_model_evaluated)),
                            "exit_prob": float(exit_prob) if exit_model_evaluated else float("nan"),
                            "exit_threshold": float(self.exit_threshold),
                            "exit_decision": exit_decision,
                            "pnl_bps": float(pnl_bps_now),
                            "mfe_bps": float(mfe_bps),
                            "mae_bps": float(mae_bps),
                            "distance_from_peak_mfe_bps": float(distance_from_peak_mfe_bps),
                            "time_since_mfe_bars": float(time_since_mfe_bars),
                            "giveback_ratio": float(giveback_ratio),
                        }
                    )
                if io_only:
                    self._exit_eval_flow["blocked_pre_eval_count"] += 1
                    self._exit_eval_blocked_reasons["io_only_replay"] = (
                        self._exit_eval_blocked_reasons.get("io_only_replay", 0) + 1
                    )
                    t_exit_input_start = time.perf_counter()
                    window_arr = self._build_exit_ctx19_window(trade, candles, window_len)
                    _accum_runner_metric(
                        "t_exit_input_prep_sec",
                        "n_exit_input_prep_calls",
                        time.perf_counter() - t_exit_input_start,
                    )
                    if window_arr is None:
                        self._exit_eval_blocked_reasons["window_build_failed"] = (
                            self._exit_eval_blocked_reasons.get("window_build_failed", 0) + 1
                        )
                        _emit_exit_eval_trace(False, float("nan"), "none")
                        continue
                    self._append_exit_io_record(
                        event_ts=now_ts,
                        trade=trade,
                        prob_close=0.0,
                        window_arr=window_arr,
                        ctx_cont=ctx_cont_snapshot,
                        ctx_cat=ctx_cat_snapshot,
                    )
                    _emit_exit_eval_trace(False, float("nan"), "none")
                    continue
                # No min-hold/time-stop guards in active path; decisions are edge/event-driven.
                t_exit_input_start = time.perf_counter()
                window_arr = self._build_exit_ctx19_window(trade, candles, window_len)
                _accum_runner_metric(
                    "t_exit_input_prep_sec",
                    "n_exit_input_prep_calls",
                    time.perf_counter() - t_exit_input_start,
                )
                if window_arr is None:
                    self._exit_eval_flow["blocked_pre_eval_count"] += 1
                    self._exit_eval_blocked_reasons["window_build_failed"] = (
                        self._exit_eval_blocked_reasons.get("window_build_failed", 0) + 1
                    )
                    _emit_exit_eval_trace(False, float("nan"), "none")
                    continue
                self._exit_eval_flow["eligible_count"] += 1
                # Sideguards are intentionally disabled in active canonical path.
                # Keep event-arm logs below for observability; runtime safety remains CATA-only.
                guard_payload = self._should_trigger_cat_guard(
                    window_arr=window_arr,
                    pnl_bps_now=float(pnl_bps_now),
                )
                if guard_payload is not None:
                    log.info(
                        "[THRESHOLD_SKIPPED_BECAUSE_PRIOR_GUARD] trade_id=%s side=%s bars_held=%d guard=CATASTROPHIC_GUARD",
                        getattr(trade, "trade_id", None),
                        getattr(trade, "side", None),
                        bars_in_trade_min,
                    )
                    mark_price = current_bid if trade.side == "long" else current_ask
                    self._exit_cat_guard_triggers += 1
                    _emit_exit_eval_trace(False, float("nan"), "guard")
                    log.info(
                        "[EXIT_CATA_GUARD_TRIGGER] trade_uid=%s trade_id=%s side=%s bars_held=%d pnl_bps=%.2f mfe_bps=%.2f dd_from_mfe_bps=%.2f giveback_ratio=%.3f mfe_decay_rate=%.3f pnl_limit=%.2f thresholds={mfe>=%.1f dd>=%.1f gb>=%.2f pnl_max<=%.2f pnl_frac<=%.2f}",
                        getattr(trade, "trade_uid", None),
                        getattr(trade, "trade_id", None),
                        getattr(trade, "side", None),
                        bars_in_trade_min,
                        float(pnl_bps_now),
                        float(guard_payload["mfe_bps"]),
                        float(guard_payload["dd_from_mfe_bps"]),
                        float(guard_payload["giveback_ratio"]),
                        float(guard_payload["mfe_decay_rate"]),
                        float(guard_payload["pnl_limit"]),
                        float(self._exit_cat_guard_mfe_bps),
                        float(self._exit_cat_guard_dd_bps),
                        float(self._exit_cat_guard_giveback_ratio),
                        float(self._exit_cat_guard_pnl_max),
                        float(self._exit_cat_guard_pnl_frac_mfe),
                    )
                    log.info(
                        "[EXIT_CATA_GUARD_SIGNAL] trade_id=%s bars_held=%d pnl_bps=%.2f dd_from_mfe_bps=%.2f giveback_ratio=%.3f",
                        getattr(trade, "trade_id", None),
                        bars_in_trade_min,
                        float(pnl_bps_now),
                        float(guard_payload["dd_from_mfe_bps"]),
                        float(guard_payload["giveback_ratio"]),
                    )
                    log.info(
                        "[EXIT_CATA_GUARD_TO_ARBITER] trade_id=%s reason=CATASTROPHIC_GUARD pnl_bps=%.2f bars_held=%d",
                        getattr(trade, "trade_id", None),
                        float(pnl_bps_now),
                        bars_in_trade_min,
                    )
                    log.info(
                        "[EXIT_RUNTIME_SAFETY_EXIT] trade_id=%s side=%s reason=CATASTROPHIC_GUARD pnl_bps=%.2f bars_held=%d",
                        getattr(trade, "trade_id", None),
                        getattr(trade, "side", None),
                        float(pnl_bps_now),
                        bars_in_trade_min,
                    )
                    accepted = self.request_close(
                        trade_id=trade.trade_id,
                        source="EXIT_CATA_GUARD",
                        reason="CATASTROPHIC_GUARD",
                        px=mark_price,
                        pnl_bps=pnl_bps_now,
                        exit_bid=current_bid,
                        exit_ask=current_ask,
                        bars_in_trade=bars_in_trade_min,
                    )
                    closes_requested += 1
                    if accepted:
                        closes_accepted += 1
                        log.info(
                            "[EXIT_CATA_GUARD_CLOSE_EXECUTED] trade_id=%s pnl_bps=%.2f bars_held=%d",
                            getattr(trade, "trade_id", None),
                            float(pnl_bps_now),
                            bars_in_trade_min,
                        )
                        self._record_exit_prob_on_close(trade, bars_in_trade_min, prob_close=None, exit_reason="CATASTROPHIC_GUARD")
                        if trade in self.open_trades:
                            self.open_trades.remove(trade)
                        self.record_realized_pnl(now_ts, pnl_bps_now)
                        self._log_trade_close_with_metrics(
                            trade=trade,
                            exit_time=now_ts,
                            exit_price=mark_price,
                            exit_reason="CATASTROPHIC_GUARD",
                            realized_pnl_bps=pnl_bps_now,
                            bars_held=bars_in_trade_min,
                        )
                        self._update_trade_log_on_close(
                            trade.trade_id,
                            mark_price,
                            pnl_bps_now,
                            "CATASTROPHIC_GUARD",
                            now_ts,
                            bars_in_trade=bars_in_trade_min,
                        )
                    else:
                        log.error(
                            "[EXIT_CATA_GUARD_ARBITER_REJECT] trade_id=%s pnl_bps=%.2f bars_held=%d",
                            getattr(trade, "trade_id", None),
                            float(pnl_bps_now),
                            bars_in_trade_min,
                        )
                    continue
                policy_flat_state = self._policy_friday_flat_state(
                    now_ts=now_ts,
                    window_arr=window_arr,
                    pnl_bps_now=float(pnl_bps_now),
                )
                if policy_flat_state is not None:
                    mark_price = current_bid if trade.side == "long" else current_ask
                    _emit_exit_eval_trace(False, float("nan"), "policy_friday_flat")
                    log.info(
                        "[EXIT_POLICY_FRIDAY_FLAT_TRIGGER] trade_id=%s side=%s bars_held=%d pnl_bps=%.2f mfe_bps=%s dd_from_mfe_bps=%s giveback_ratio=%s time_since_mfe_bars=%s weekday=%s cutoff_hhmm_utc=%s reason=%s",
                        getattr(trade, "trade_id", None),
                        getattr(trade, "side", None),
                        bars_in_trade_min,
                        float(pnl_bps_now),
                        policy_flat_state["mfe_bps"],
                        policy_flat_state["dd_from_mfe_bps"],
                        policy_flat_state["giveback_ratio"],
                        policy_flat_state["time_since_mfe_bars"],
                        policy_flat_state["weekday_name"],
                        policy_flat_state["cutoff_hhmm_utc"],
                        policy_flat_state["reason"],
                    )
                    accepted = self.request_close(
                        trade_id=trade.trade_id,
                        source="POLICY_FLAT",
                        reason=str(policy_flat_state["reason"]),
                        px=mark_price,
                        pnl_bps=pnl_bps_now,
                        exit_bid=current_bid,
                        exit_ask=current_ask,
                        bars_in_trade=bars_in_trade_min,
                    )
                    closes_requested += 1
                    if accepted:
                        closes_accepted += 1
                        self._record_exit_prob_on_close(
                            trade,
                            bars_in_trade_min,
                            prob_close=None,
                            exit_reason=str(policy_flat_state["reason"]),
                        )
                        if trade in self.open_trades:
                            self.open_trades.remove(trade)
                        self.record_realized_pnl(now_ts, pnl_bps_now)
                        self._log_trade_close_with_metrics(
                            trade=trade,
                            exit_time=now_ts,
                            exit_price=mark_price,
                            exit_reason=str(policy_flat_state["reason"]),
                            realized_pnl_bps=pnl_bps_now,
                            bars_held=bars_in_trade_min,
                        )
                        self._update_trade_log_on_close(
                            trade.trade_id,
                            mark_price,
                            pnl_bps_now,
                            str(policy_flat_state["reason"]),
                            now_ts,
                            bars_in_trade=bars_in_trade_min,
                        )
                    else:
                        log.error(
                            "[EXIT_POLICY_FRIDAY_FLAT_REJECT] trade_id=%s pnl_bps=%.2f bars_held=%d reason=%s",
                            getattr(trade, "trade_id", None),
                            float(pnl_bps_now),
                            bars_in_trade_min,
                            policy_flat_state["reason"],
                        )
                    continue
                if decider is not None and bool(getattr(self._runner, "exit_hash_guard_bypass_enabled", False)):
                    expected_dim = getattr(decider, "input_dim", None)
                    if expected_dim is not None:
                        try:
                            arr = np.asarray(window_arr, dtype=np.float32)
                            if arr.ndim == 2 and arr.shape[1] != int(expected_dim):
                                got_dim = int(arr.shape[1])
                                if got_dim > int(expected_dim):
                                    arr = arr[:, : int(expected_dim)]
                                    window_arr = arr
                                    log.warning(
                                        "[EXIT_IO_CONTRACT_TRUNCATE] expected_input_dim=%d got_input_dim=%d action=truncate",
                                        int(expected_dim),
                                        got_dim,
                                    )
                                else:
                                    log.warning(
                                        "[EXIT_IO_CONTRACT_TRUNCATE] expected_input_dim=%d got_input_dim=%d action=none",
                                        int(expected_dim),
                                        got_dim,
                                    )
                        except Exception:
                            pass
                self._exit_eval_flow["model_eval_attempt_count"] += 1
                t_exit_model_start = time.perf_counter()
                pred_details = decider.predict_details(window_arr)
                _accum_runner_metric(
                    "t_exit_model_infer_sec",
                    "n_exit_model_infer_calls",
                    time.perf_counter() - t_exit_model_start,
                )
                self._exit_eval_flow["model_eval_complete_count"] += 1
                prob_close = float(pred_details.get("prob_close") or 0.0)
                logit_val = pred_details.get("logit")
                profit_protect_prob = pred_details.get("profit_protect_prob")
                family_top_name = pred_details.get("family_top_name")
                try:
                    last_row = np.asarray(window_arr, dtype=np.float32)[-1].tolist()
                    if isinstance(last_row, list):
                        self._exit_feature_vectors.append(last_row)
                    tid = str(getattr(trade, "trade_id", "")) or str(getattr(trade, "trade_uid", ""))
                    if tid:
                        self._exit_feature_sample_trades.add(tid)
                except Exception:
                    pass
                entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
                entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
                pnl_bps_now = float(pnl_bps_now)
                self._record_exit_prob_state(trade, bars_in_trade_min, prob_close, pnl_bps_now, logit=logit_val)
                threshold_event = self._threshold_event_state(
                    window_arr=window_arr,
                    side=trade.side,
                    family_top_name=family_top_name,
                )
                threshold_event_armed = threshold_event is not None
                if not hasattr(self, "_threshold_eval_reached_log_count"):
                    self._threshold_eval_reached_log_count = 0
                self._threshold_eval_reached_log_count += 1
                if self._threshold_eval_reached_log_count <= 5 or (self._threshold_eval_reached_log_count % 5000) == 0:
                    log.info(
                        "[THRESHOLD_EVAL_REACHED] trade_id=%s side=%s bars_held=%d event_armed=%s",
                        getattr(trade, "trade_id", None),
                        getattr(trade, "side", None),
                        bars_in_trade_min,
                        bool(threshold_event_armed),
                    )
                if threshold_event_armed:
                    if not hasattr(self, "_threshold_event_armed_log_count"):
                        self._threshold_event_armed_log_count = 0
                    self._threshold_event_armed_log_count += 1
                    if (
                        self._threshold_event_armed_log_count <= 5
                        or (self._threshold_event_armed_log_count % 5000) == 0
                    ):
                        log.info(
                            "[THRESHOLD_EVENT_ARMED] trade_id=%s side=%s bars_held=%d mfe_bps=%.2f dd_from_mfe_bps=%.2f giveback_ratio=%.3f directional_edge=%.4f family_top_name=%s",
                            getattr(trade, "trade_id", None),
                            getattr(trade, "side", None),
                            bars_in_trade_min,
                            float(threshold_event["mfe_bps"]),
                            float(threshold_event["dd_from_mfe_bps"]),
                            float(threshold_event["giveback_ratio"]),
                            float(threshold_event["directional_edge"]),
                            threshold_event.get("family_top_name"),
                        )
                # Track probability stats
                self._exit_prob_n += 1
                self._exit_eval_flow["threshold_check_count"] += 1
                if prob_close >= self.exit_threshold:
                    self._exit_prob_ge_thr += 1
                self._exit_prob_min = min(self._exit_prob_min, prob_close)
                self._exit_prob_max = max(self._exit_prob_max, prob_close)
                b = self._exit_prob_buckets
                if prob_close < 0.001:
                    b["<0.001"] += 1
                elif prob_close < 0.01:
                    b["<0.01"] += 1
                elif prob_close < 0.05:
                    b["<0.05"] += 1
                elif prob_close < 0.1:
                    b["<0.1"] += 1
                elif prob_close < 0.2:
                    b["<0.2"] += 1
                else:
                    b[">=0.2"] += 1
                try:
                    should_log_io = self.replay_mode
                    if not should_log_io:
                        try:
                            exit_cfg = getattr(self, "exit_params", {}) or {}
                            log_flag = (
                                exit_cfg.get("exit", {})
                                .get("params", {})
                                .get("exit_ml", {})
                                .get("exit_transformer", {})
                                .get("log_io_features", False)
                            )
                            should_log_io = bool(log_flag)
                        except Exception:
                            should_log_io = False
                    if should_log_io:
                        self._append_exit_io_record(
                            event_ts=now_ts,
                            trade=trade,
                            prob_close=prob_close,
                            window_arr=window_arr,
                            ctx_cont=ctx_cont_snapshot,
                            ctx_cat=ctx_cat_snapshot,
                        )
                except Exception:
                    pass
                if not hasattr(self, "_exit_runtime_gt_count"):
                    self._exit_runtime_gt_count = 0
                if not hasattr(self, "_exit_hysteresis_dbg_count"):
                    self._exit_hysteresis_dbg_count = 0
                hist = self._ensure_prob_history(trade)
                hist_before = list(hist)
                try:
                    if trade.side == "long":
                        self._runner.exit_eval_long += 1
                    else:
                        self._runner.exit_eval_short += 1
                except Exception:
                    pass
                len_before = len(hist)
                hist.append(float(prob_close))
                len_after = len(hist)
                if self._exit_hysteresis_dbg_count < 30:
                    recent = list(hist)[-self.exit_require_consecutive:]
                    recent_f = [float(x) for x in recent]
                    model_decided_exit_dbg = (
                        len_after >= self.exit_require_consecutive
                        and all(x >= float(self.exit_threshold) for x in recent_f)
                    )
                    log.info(
                        "[EXIT_HYSTERESIS_DBG] trade_uid=%s trade_id=%s side=%s trade_obj_id=%s hist_type=%s maxlen=%s len_before=%d len_after=%d thr=%.6f req=%d history=%s recent=%s recent_f=%s should_exit=%s",
                        getattr(trade, "trade_uid", None) or getattr(trade, "trade_id", None),
                        getattr(trade, "trade_id", None),
                        getattr(trade, "side", None),
                        id(trade),
                        type(hist).__name__,
                        getattr(hist, "maxlen", None),
                        len_before,
                        len_after,
                        float(self.exit_threshold),
                        int(self.exit_require_consecutive),
                        list(hist),
                        recent,
                        recent_f,
                        model_decided_exit_dbg,
                    )
                    self._exit_hysteresis_dbg_count += 1
                if self._exit_runtime_gt_count < 5:
                    hist_after = list(hist)
                    log.info(
                        "[EXIT_RUNTIME_GROUND_TRUTH] trade_uid=%s trade_id=%s side=%s prob_close=%.6f threshold=%.6f require_consecutive=%d hist_before=%s hist_after=%s model_path=%s model_sha=%s window_len=%s input_dim=%s bars_held=%s",
                        getattr(trade, "trade_uid", None) or getattr(trade, "trade_id", None),
                        getattr(trade, "trade_id", None),
                        getattr(trade, "side", None),
                        prob_close,
                        self.exit_threshold,
                        self.exit_require_consecutive,
                        [f"{p:.6f}" for p in hist_before],
                        [f"{p:.6f}" for p in hist_after],
                        getattr(decider, "model_path", getattr(decider, "bundle_dir", None)),
                        getattr(decider, "model_sha", None),
                        getattr(decider, "window_len", None),
                        getattr(decider, "input_dim", None),
                        bars_in_trade_min,
                    )
                    self._exit_runtime_gt_count += 1

                # Best-effort: log one context-bearing exit ML event (ensures exits jsonl has ctx 6/6)
                if not getattr(self, "_exit_ml_ctx_logged_once", False) and ctx_cont_snapshot and ctx_cat_snapshot:
                    self._maybe_log_exit_ml_event(
                        event_ts=now_ts,
                        prob_close=prob_close,
                        trade=trade,
                        ctx_cont=ctx_cont_snapshot,
                        ctx_cat=ctx_cat_snapshot,
                    )
                    self._exit_ml_ctx_logged_once = True

                log.debug(
                    "[EXIT] Trade %s: bars_in_trade=%d prob_close=%.4f threshold=%.4f history=%s (require_consecutive=%d)",
                    trade.trade_id,
                    bars_in_trade_min,
                    prob_close,
                    self.exit_threshold,
                    list(trade.prob_close_history),
                    self.exit_require_consecutive,
                )

                event_armed = bool(threshold_event_armed)
                recent_req = list(hist)[-self.exit_require_consecutive:]
                recent_req_f = [float(x) for x in recent_req]
                model_decided_exit = bool(
                    len_after >= self.exit_require_consecutive
                    and all(x >= float(self.exit_threshold) for x in recent_req_f)
                )
                residual_state = self._canonical_residual_exit_state(window_arr=window_arr)
                residual_should_exit = False
                if (
                    bool(getattr(self._runner, "exit_residual_enabled", False))
                    and not model_decided_exit
                    and residual_state is not None
                ):
                    residual_prob_gate_ok = True
                    if bool(getattr(self._runner, "exit_residual_require_prob_below_threshold", True)):
                        residual_prob_gate_ok = float(prob_close) < float(self.exit_threshold)
                    residual_pnl_gate_ok = True
                    if bool(getattr(self._runner, "exit_residual_require_nonpositive_pnl", False)):
                        residual_pnl_gate_ok = float(pnl_bps_now) <= 0.0
                    residual_late_or_boundary = (
                        float(residual_state["minutes_since_session_open"])
                        >= float(getattr(self._runner, "exit_residual_minutes_since_session_open_min", 120.0))
                    ) or (
                        float(residual_state["minutes_to_next_session_boundary"])
                        <= float(getattr(self._runner, "exit_residual_minutes_to_next_session_boundary_max", 120.0))
                    )
                    residual_deterioration_ok = (
                        float(residual_state["distance_from_peak_mfe_bps"])
                        >= float(getattr(self._runner, "exit_residual_distance_from_peak_mfe_bps_min", 40.0))
                    )
                    residual_should_exit = bool(
                        residual_prob_gate_ok
                        and residual_pnl_gate_ok
                        and residual_late_or_boundary
                        and residual_deterioration_ok
                    )
                post_edge_state = self._selective_post_edge_separator_state(
                    window_arr=window_arr,
                    side=getattr(trade, "side", ""),
                    session_current=session_current,
                    pnl_bps_now=float(pnl_bps_now),
                    threshold_event=threshold_event,
                    threshold_event_armed=threshold_event_armed,
                    exit_prob=float(prob_close),
                    profit_protect_prob=profit_protect_prob,
                )
                protected_profit_state = self._protected_profit_state(
                    trade=trade,
                    window_arr=window_arr,
                    side=getattr(trade, "side", ""),
                    session_current=session_current,
                    pnl_bps_now=float(pnl_bps_now),
                    threshold_event=threshold_event,
                    threshold_event_armed=threshold_event_armed,
                    post_edge_state=post_edge_state,
                    profit_protect_prob=profit_protect_prob,
                    bars_in_trade_min=bars_in_trade_min,
                )
                if protected_profit_state is not None:
                    if bool(protected_profit_state.get("newly_armed", False)):
                        if not hasattr(self, "_exit_protected_profit_armed_count"):
                            self._exit_protected_profit_armed_count = 0
                        self._exit_protected_profit_armed_count += 1
                        if (
                            self._exit_protected_profit_armed_count <= 5
                            or (self._exit_protected_profit_armed_count % 5000) == 0
                        ):
                            log.info(
                                "[EXIT_PROTECTED_PROFIT_ARMED] trade_id=%s side=%s bars_held=%d floor_bps=%.2f pnl_bps_now=%.2f mfe_bps=%.2f dd_from_mfe_bps=%.2f giveback_ratio=%.3f time_since_mfe_bars=%.2f score_source=%s score_value=%.6f",
                                getattr(trade, "trade_id", None),
                                getattr(trade, "side", None),
                                bars_in_trade_min,
                                float(protected_profit_state["be_plus_floor_bps"]),
                                float(pnl_bps_now),
                                float(protected_profit_state["mfe_bps"]),
                                float(protected_profit_state["dd_from_mfe_bps"]),
                                float(protected_profit_state["giveback_ratio"]),
                                float(protected_profit_state["time_since_mfe_bars"]),
                                str(protected_profit_state["score_source"]),
                                float(protected_profit_state["score_value"]),
                            )
                protected_profit_should_exit = bool(protected_profit_state and protected_profit_state.get("should_exit", False))
                if protected_profit_should_exit:
                    if not hasattr(self, "_exit_protected_profit_close_count"):
                        self._exit_protected_profit_close_count = 0
                    self._exit_protected_profit_close_count += 1
                    if (
                        self._exit_protected_profit_close_count <= 5
                        or (self._exit_protected_profit_close_count % 5000) == 0
                    ):
                        log.info(
                            "[EXIT_PROTECTED_PROFIT_CLOSE_TRIGGER] trade_id=%s side=%s bars_held=%d floor_bps=%.2f pnl_bps_now=%.2f mfe_bps=%.2f dd_from_mfe_bps=%.2f giveback_ratio=%.3f time_since_mfe_bars=%.2f score_source=%s score_value=%.6f",
                            getattr(trade, "trade_id", None),
                            getattr(trade, "side", None),
                            bars_in_trade_min,
                            float(protected_profit_state["be_plus_floor_bps"]),
                            float(pnl_bps_now),
                            float(protected_profit_state["mfe_bps"]),
                            float(protected_profit_state["dd_from_mfe_bps"]),
                            float(protected_profit_state["giveback_ratio"]),
                            float(protected_profit_state["time_since_mfe_bars"]),
                            str(protected_profit_state["score_source"]),
                            float(protected_profit_state["score_value"]),
                        )
                post_edge_should_exit = False
                if (
                    post_edge_state is not None
                    and not model_decided_exit
                    and not residual_should_exit
                ):
                    if (
                        post_edge_state["score_source"] == "profit_protect_prob"
                        and profit_protect_prob is None
                        and (bool(getattr(self, "replay_mode", False)) or os.getenv("GX1_RUN_MODE", "").upper() == "TRUTH")
                    ):
                        raise RuntimeError(
                            "[EXIT_POST_EDGE_HEAD_MISSING] score_source=profit_protect_prob requested but active exit bundle has no trained profit_protect head"
                        )
                    below_main_threshold_ok = True
                    if bool(getattr(self, "_exit_post_edge_require_model_below_main_threshold", True)):
                        below_main_threshold_ok = float(prob_close) < float(self.exit_threshold)
                    post_edge_should_exit = bool(below_main_threshold_ok)
                if not hasattr(self, "_exit_model_decision_log_count"):
                    self._exit_model_decision_log_count = 0
                self._exit_model_decision_log_count += 1
                if self._exit_model_decision_log_count <= 5 or (self._exit_model_decision_log_count % 5000) == 0:
                    log.info(
                        "[EXIT_MODEL_DECISION] trade_id=%s side=%s bars_held=%d model_decided_exit=%s residual_decided_exit=%s post_edge_decided_exit=%s event_armed=%s protected_profit_state=%s prob_close=%.6f threshold=%.6f require_consecutive=%d pnl_bps_now=%.2f recent=%s residual_state=%s post_edge_state=%s family_top_name=%s",
                        getattr(trade, "trade_id", None),
                        getattr(trade, "side", None),
                        bars_in_trade_min,
                        bool(model_decided_exit),
                        bool(residual_should_exit),
                        bool(post_edge_should_exit),
                        bool(event_armed),
                        bool(protected_profit_state),
                        float(prob_close),
                        float(self.exit_threshold),
                        int(self.exit_require_consecutive),
                        float(pnl_bps_now),
                        [f"{p:.6f}" for p in recent_req_f],
                        residual_state,
                        post_edge_state,
                        family_top_name,
                    )

                should_exit = bool(model_decided_exit or residual_should_exit or post_edge_should_exit or protected_profit_should_exit)
                exit_reason = (
                    "THRESHOLD" if bool(model_decided_exit)
                    else "RESIDUAL_CANONICAL" if bool(residual_should_exit)
                    else "PROTECTED_PROFIT" if bool(protected_profit_should_exit)
                    else "POST_EDGE_SEPARATOR" if bool(post_edge_should_exit)
                    else None
                )
                if should_exit:
                    ev = threshold_event or {}
                    log.info(
                        "[EXIT_MODEL_DECIDED_EXIT] trade_id=%s side=%s bars_held=%d exit_reason=%s prob_close=%.6f profit_protect_prob=%s family_top_name=%s event_armed=%s protected_profit_state=%s mfe_bps=%.2f dd_from_mfe_bps=%.2f giveback_ratio=%.3f directional_edge=%.4f p_hat=%.4f p_flat=%.4f margin=%.4f uncertainty=%.4f residual_distance_from_peak_mfe_bps=%.2f residual_minutes_since_session_open=%.2f residual_minutes_to_next_session_boundary=%.2f post_edge_state=%s",
                        trade.trade_id,
                        trade.side,
                        bars_in_trade_min,
                        exit_reason,
                        float(prob_close),
                        None if profit_protect_prob is None else round(float(profit_protect_prob), 6),
                        family_top_name,
                        bool(event_armed),
                        bool(protected_profit_state),
                        float(ev.get("mfe_bps", float("nan"))),
                        float(ev.get("dd_from_mfe_bps", float("nan"))),
                        float(ev.get("giveback_ratio", float("nan"))),
                        float(ev.get("directional_edge", float("nan"))),
                        float(ev.get("p_hat", float("nan"))),
                        float(ev.get("p_flat", float("nan"))),
                        float(ev.get("margin_top1_top2", float("nan"))),
                        float(ev.get("uncertainty_score", float("nan"))),
                        float((residual_state or {}).get("distance_from_peak_mfe_bps", float("nan"))),
                        float((residual_state or {}).get("minutes_since_session_open", float("nan"))),
                        float((residual_state or {}).get("minutes_to_next_session_boundary", float("nan"))),
                        post_edge_state,
                    )
                    if bool(post_edge_should_exit):
                        self._exit_post_edge_triggers += 1
                    if bool(protected_profit_should_exit):
                        self._exit_protected_profit_close_triggers += 1
                    try:
                        if trade.side == "long":
                            self._runner.exit_close_long += 1
                        else:
                            self._runner.exit_close_short += 1
                    except Exception:
                        pass
                else:
                    log.debug(
                        "[EXIT] Trade %s: model_blocked model_decided_exit=%s event_armed=%s prob_close=%.4f",
                        trade.trade_id,
                        bool(model_decided_exit),
                        bool(event_armed),
                        float(prob_close),
                    )

                _emit_exit_eval_trace(
                    True,
                    float(prob_close),
                    "threshold" if bool(model_decided_exit) else "residual" if bool(residual_should_exit) else "post_edge_separator" if bool(post_edge_should_exit) else "none",
                )

                if should_exit:
                    entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
                    entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
                    pnl_bps = compute_pnl_bps(entry_bid, entry_ask, current_bid, current_ask, trade.side)
                    mark_price = current_bid if trade.side == "long" else current_ask
                    delta_minutes = (now_ts - trade.entry_time).total_seconds() / 60.0
                    bars_in_trade = int(round(delta_minutes / 5.0))
                    did_eval_any = True
                    if not hasattr(self, "_exit_decision_logged"):
                        self._exit_decision_logged = set()
                    tuid = getattr(trade, "trade_uid", None) or getattr(trade, "trade_id", None)
                    if tuid not in self._exit_decision_logged:
                        arb_allow = getattr(getattr(self._runner, "exit_control", None), "allow_model_exit_when", {}) or {}
                        log.info(
                            "[EXIT_DECISION_PROOF] trade_uid=%s trade_id=%s side=%s exit_reason=%s prob_close=%.6f threshold=%.6f require_consecutive=%d prob_history=%s arb_cfg={min_exit_prob=%s, exit_prob_hysteresis=%s, min_pnl_bps=%s, min_realized_pnl_bps=%s, require_positive_mfe_before_model_exit=%s, min_mfe_bps_for_model_exit=%s, max_giveback_frac_from_mfe=%s, max_giveback_bps_from_mfe=%s} bars_held=%s pnl_bps=%s will_call_request_close=true",
                            tuid,
                            getattr(trade, "trade_id", None),
                            getattr(trade, "side", None),
                            exit_reason,
                            prob_close,
                            self.exit_threshold,
                            self.exit_require_consecutive,
                            [f"{p:.6f}" for p in list(trade.prob_close_history)],
                            arb_allow.get("min_exit_prob"),
                            arb_allow.get("exit_prob_hysteresis"),
                            arb_allow.get("min_pnl_bps"),
                            arb_allow.get("min_realized_pnl_bps"),
                            arb_allow.get("require_positive_mfe_before_model_exit"),
                            arb_allow.get("min_mfe_bps_for_model_exit"),
                            arb_allow.get("max_giveback_frac_from_mfe"),
                            arb_allow.get("max_giveback_bps_from_mfe"),
                            bars_in_trade,
                            pnl_bps,
                        )
                        self._exit_decision_logged.add(tuid)
                    if trade not in self.open_trades:
                        continue
                    if not hasattr(self, "_exit_mgr_pnl_log_count"):
                        self._exit_mgr_pnl_log_count = 0
                    if self._exit_mgr_pnl_log_count < 5:
                        log.debug(
                            "[PNL] ExitManager PnL: entry_bid=%.5f entry_ask=%.5f exit_bid=%.5f exit_ask=%.5f side=%s",
                            entry_bid,
                            entry_ask,
                            current_bid,
                            current_ask,
                            trade.side,
                        )
                        self._exit_mgr_pnl_log_count += 1
                    log.info("[EXIT] propose close reason=%s pnl=%.1f bars_in_trade=%d", exit_reason, pnl_bps, bars_in_trade)
                    close_source = "MODEL_EXIT"
                    close_reason = str(exit_reason)
                    if bool(protected_profit_should_exit):
                        close_source = "PROTECTED_PROFIT"
                        close_reason = "BE_PLUS_FLOOR"
                    accepted = self.request_close(
                        trade_id=trade.trade_id,
                        source=close_source,
                        reason=close_reason,
                        px=mark_price,
                        pnl_bps=pnl_bps,
                        exit_bid=current_bid,
                        exit_ask=current_ask,
                        bars_in_trade=bars_in_trade,
                    )
                    closes_requested += 1
                    if not accepted:
                        log.warning("[EXIT] close rejected by ExitArbiter for trade %s", trade.trade_id)
                        log.error("Exit signaled but not applied for trade %s", trade.trade_id)
                        log.error(
                            "[EXIT_DECISION_RESULT] trade_uid=%s trade_id=%s accepted=False reject_reason=arbiter_reject still_in_open_trades=%s",
                            tuid,
                            getattr(trade, "trade_id", None),
                            trade in self.open_trades,
                        )
                        continue
                    closes_accepted += 1
                    if str(exit_reason) == "THRESHOLD":
                        self._record_exit_prob_on_close(trade, bars_in_trade, prob_close, exit_reason="THRESHOLD")
                    if trade in self.open_trades:
                        self.open_trades.remove(trade)
                    log.info(
                        "[EXIT_DECISION_RESULT] trade_uid=%s trade_id=%s accepted=True reject_reason=None still_in_open_trades=%s",
                        tuid,
                        getattr(trade, "trade_id", None),
                        trade in self.open_trades,
                    )
                    self.record_realized_pnl(now_ts, pnl_bps)
                    self._log_trade_close_with_metrics(
                        trade=trade,
                        exit_time=now_ts,
                        exit_price=mark_price,
                        exit_reason=str(exit_reason),
                        realized_pnl_bps=pnl_bps,
                        bars_held=bars_in_trade,
                    )
            self._maybe_log_exit_prob_audit(decider=decider)
            return

        if closes_requested:
            log.debug(
                "[EXIT] Close summary: requested=%d accepted=%d remaining_open=%d",
                closes_requested,
                closes_accepted,
                len(self.open_trades),
            )
        
        # Update tick watcher (stop if no more open trades)
        self._maybe_update_tick_watcher()
        self._maybe_log_exit_prob_audit(decider=decider)

    def _ensure_prob_history(self, trade: Any) -> deque:
        """
        Ensure trade.prob_close_history exists and has sufficient maxlen for hysteresis.
        """
        req = int(getattr(self, "exit_require_consecutive", 1))
        target = max(req * 4, req, 8)
        h = getattr(trade, "prob_close_history", None)
        if h is None:
            h = deque(maxlen=target)
            trade.prob_close_history = h
            return h
        cur_maxlen = getattr(h, "maxlen", None)
        if cur_maxlen is None or int(cur_maxlen) < target:
            old = []
            try:
                old = list(h)
            except Exception:
                old = []
            new = deque(old[-target:], maxlen=target)
            trade.prob_close_history = new
            return new
        return h

    def _record_exit_prob_state(
        self,
        trade: Any,
        bars_in_trade: int,
        prob_close: float,
        pnl_bps_now: float,
        logit: Optional[float] = None,
    ) -> None:
        try:
            tid = str(getattr(trade, "trade_id", ""))
            if not tid:
                return
            states = self._exit_prob_trade_states.setdefault(tid, [])
            if not states:
                self._exit_prob_at_entry.append(float(prob_close))
            states.append((int(bars_in_trade), float(prob_close), float(pnl_bps_now)))

            last = self._exit_prob_last_by_trade.get(tid)
            if last is not None:
                if prob_close > last:
                    self._exit_prob_inc += 1
                elif prob_close < last:
                    self._exit_prob_dec += 1
                else:
                    self._exit_prob_flat += 1
            self._exit_prob_last_by_trade[tid] = float(prob_close)
            self._exit_prob_all.append(float(prob_close))
            if logit is not None:
                self._exit_logit_all.append(float(logit))
        except Exception:
            return

    def _record_exit_prob_on_close(
        self,
        trade: Any,
        bars_in_trade: int,
        prob_close: Optional[float],
        exit_reason: Optional[str] = None,
    ) -> None:
        try:
            tid = str(getattr(trade, "trade_id", ""))
            states = self._exit_prob_trade_states.get(tid) or []
            if not states:
                return
            if prob_close is None:
                try:
                    prob_close = float(states[-1][1])
                except Exception:
                    return
            total_bars = int(bars_in_trade)
            self._exit_prob_at_exit.append(float(prob_close))

            # N bars before exit
            for n in (1, 3, 5):
                target = total_bars - n
                if target < 0:
                    continue
                for b, p, _ in reversed(states):
                    if b == target:
                        self._exit_prob_before_exit[n].append(float(p))
                        break

            # MFE/MAE peaks by pnl_bps_now
            pnl_series = [s[2] for s in states]
            if pnl_series:
                max_idx = int(max(range(len(pnl_series)), key=lambda i: pnl_series[i]))
                min_idx = int(min(range(len(pnl_series)), key=lambda i: pnl_series[i]))
                self._exit_prob_at_mfe_peak.append(float(states[max_idx][1]))
                self._exit_prob_at_mae_peak.append(float(states[min_idx][1]))

            # Ranking groups: A (<=3 bars before exit), B (>10 bars before exit)
            for b, p, _ in states:
                bars_to_exit = total_bars - int(b)
                if bars_to_exit <= 3:
                    self._exit_prob_group_a.append(float(p))
                elif bars_to_exit > 10:
                    self._exit_prob_group_b.append(float(p))
                if bars_to_exit == 1:
                    self._exit_prob_bars_before_exit["b1"].append(float(p))
                elif bars_to_exit == 2:
                    self._exit_prob_bars_before_exit["b2"].append(float(p))
                elif bars_to_exit == 3:
                    self._exit_prob_bars_before_exit["b3"].append(float(p))
                elif bars_to_exit in (4, 5):
                    self._exit_prob_bars_before_exit["b4_5"].append(float(p))
                elif bars_to_exit > 5:
                    self._exit_prob_bars_before_exit["b_gt5"].append(float(p))
        except Exception:
            return

    def _resolve_exit_eval_trace_path(self) -> Optional[Path]:
        if not bool(getattr(self, "replay_mode", False)):
            return None
        if self._exit_eval_trace_path:
            return self._exit_eval_trace_path
        out_dir = getattr(self, "explicit_output_dir", None) or getattr(self, "output_dir", None)
        if not out_dir:
            return None
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._exit_eval_trace_path = out_dir / "EXIT_EVAL_TRACE.csv"
        return self._exit_eval_trace_path

    def _append_exit_eval_trace(self, row: dict) -> None:
        path = self._resolve_exit_eval_trace_path()
        if not path:
            return
        write_header = False
        if not self._exit_eval_trace_header_written:
            if not path.exists():
                write_header = True
            self._exit_eval_trace_header_written = True
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        try:
            runner = getattr(self, "_runner", None)
            if runner is not None:
                runner.perf_exit_eval_trace_rows = int(getattr(runner, "perf_exit_eval_trace_rows", 0) + 1)
        except Exception:
            pass

    def _maybe_log_exit_prob_audit(self, decider: Any = None, force: bool = False) -> None:
        """
        One-shot probability audit log for exit transformer evaluations.
        """
        try:
            if getattr(self, "_exit_prob_audit_logged", False) and not force:
                return
            n = int(getattr(self, "_exit_prob_n", 0))
            runner = getattr(self, "_runner", None)
            model_path = getattr(runner, "exit_transformer_model_path", None)
            model_dir = getattr(runner, "exit_transformer_model_dir", None)
            config_path = getattr(runner, "exit_transformer_config_path", None)
            io_version = getattr(runner, "exit_transformer_io_version", None) or getattr(runner, "exit_ml_io_version", None)
            input_dim = getattr(runner, "exit_transformer_input_dim", None) or getattr(runner, "exit_ml_input_dim", None)
            feature_hash = getattr(runner, "exit_transformer_feature_hash", None) or getattr(runner, "exit_ml_config_hash", None)
            uses_logits_path = int(getattr(runner, "exit_transformer_uses_logits_path", 0))
            if decider is not None:
                model_path = model_path or getattr(decider, "model_path", None)
                model_dir = model_dir or getattr(decider, "bundle_dir", None) or getattr(decider, "model_path", None)
                cfg = getattr(decider, "config", {}) or {}
                io_version = io_version or cfg.get("exit_ml_io_version")
                input_dim = input_dim or getattr(decider, "input_dim", None)
                feature_hash = feature_hash or cfg.get("feature_names_hash")
                if getattr(decider, "model", None) is not None and hasattr(decider.model, "forward_logits"):
                    uses_logits_path = 1
            if not hasattr(self, "_exit_runtime_model_proof_log_count"):
                self._exit_runtime_model_proof_log_count = 0
            self._exit_runtime_model_proof_log_count += 1
            if self._exit_runtime_model_proof_log_count <= 5 or (self._exit_runtime_model_proof_log_count % 5000) == 0:
                log.info(
                    "[EXIT_RUNTIME_MODEL_PROOF] model_path=%s model_dir=%s config_path=%s io_version=%s input_dim=%s feature_hash=%s uses_logits_path=%d",
                    model_path,
                    model_dir,
                    config_path,
                    io_version,
                    input_dim,
                    feature_hash,
                    uses_logits_path,
                )
            if not model_path or not model_dir:
                log.warning(
                    "[EXIT_RUNTIME_MODEL_PROOF_WARN] model_missing=1 model_path=%s model_dir=%s",
                    model_path,
                    model_dir,
                )
            if not force:
                try:
                    if runner is not None and bool(getattr(runner, "replay_mode", False)):
                        replay_end_ts = getattr(runner, "replay_end_ts", None)
                        replay_cur_ts = getattr(runner, "replay_current_ts", None) or getattr(runner, "_replay_current_ts", None)
                        if replay_end_ts is not None and replay_cur_ts is not None:
                            if pd.Timestamp(replay_cur_ts) < pd.Timestamp(replay_end_ts):
                                return
                except Exception:
                    pass
            # Compute runtime score stats even for small samples (write at replay end)
            arr = np.asarray(self._exit_prob_all, dtype=np.float32)
            if arr.size:
                dist = {
                    "count": int(arr.size),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "p50": float(np.percentile(arr, 50)),
                    "p75": float(np.percentile(arr, 75)),
                    "p90": float(np.percentile(arr, 90)),
                    "p95": float(np.percentile(arr, 95)),
                    "p99": float(np.percentile(arr, 99)),
                }
            else:
                dist = {
                    "count": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "p50": 0.0,
                    "p75": 0.0,
                    "p90": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                }
            logits_arr = np.asarray(self._exit_logit_all, dtype=np.float32)
            if logits_arr.size:
                logit_dist = {
                    "count": int(logits_arr.size),
                    "mean": float(np.mean(logits_arr)),
                    "std": float(np.std(logits_arr)),
                    "min": float(np.min(logits_arr)),
                    "max": float(np.max(logits_arr)),
                }
            else:
                logit_dist = {
                    "count": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                }
            if not self._exit_prob_score_audit_written:
                log.info(
                    "[EXIT_RUNTIME_SCORE_STATS] logit_mean=%.6f logit_std=%.6f prob_mean=%.6f prob_std=%.6f prob_min=%.6f prob_max=%.6f n=%d",
                    logit_dist["mean"],
                    logit_dist["std"],
                    dist["mean"],
                    dist["std"],
                    dist["min"],
                    dist["max"],
                    dist["count"],
                )
            feat_arr = np.asarray(self._exit_feature_vectors, dtype=np.float32)
            if feat_arr.size and feat_arr.ndim == 2:
                feat_mean = np.mean(feat_arr, axis=0)
                feat_std = np.std(feat_arr, axis=0)
                feat_min = np.min(feat_arr, axis=0)
                feat_max = np.max(feat_arr, axis=0)
                zero_var = int(np.sum(feat_std <= 0.0))
                low_var = int(np.sum(feat_std <= 1e-6))
                if not self._exit_prob_score_audit_written:
                    log.info(
                        "[EXIT_RUNTIME_FEATURE_STATS] feature_mean=%s feature_std=%s feature_min=%s feature_max=%s n=%d features_with_zero_variance=%d features_with_low_variance=%d",
                        np.array2string(feat_mean, precision=6, separator=","),
                        np.array2string(feat_std, precision=6, separator=","),
                        np.array2string(feat_min, precision=6, separator=","),
                        np.array2string(feat_max, precision=6, separator=","),
                        int(feat_arr.shape[0]),
                        zero_var,
                        low_var,
                    )
            n_model_evals_sampled = int(dist.get("count", 0))
            n_unique_trades_sampled = int(len(self._exit_feature_sample_trades))
            if not self._exit_runtime_sample_proof_logged:
                log.info(
                    "[EXIT_RUNTIME_SAMPLE_PROOF] n_model_evals_sampled=%d n_unique_trades_sampled=%d",
                    n_model_evals_sampled,
                    n_unique_trades_sampled,
                )
                self._exit_runtime_sample_proof_logged = True
            flow = getattr(self, "_exit_eval_flow", {}) or {}
            blocked = getattr(self, "_exit_eval_blocked_reasons", {}) or {}
            log.info(
                "[EXIT_EVAL_FLOW_PROOF] trade_active_count=%d eligible_count=%d blocked_pre_eval_count=%d model_eval_attempt_count=%d model_eval_complete_count=%d threshold_check_count=%d blocked_reasons=%s",
                int(flow.get("trade_active_count", 0)),
                int(flow.get("eligible_count", 0)),
                int(flow.get("blocked_pre_eval_count", 0)),
                int(flow.get("model_eval_attempt_count", 0)),
                int(flow.get("model_eval_complete_count", 0)),
                int(flow.get("threshold_check_count", 0)),
                blocked,
            )
            if int(flow.get("model_eval_attempt_count", 0)) == 0:
                primary_reason = None
                try:
                    if blocked:
                        primary_reason = max(blocked.items(), key=lambda kv: kv[1])[0]
                except Exception:
                    primary_reason = None
                log.warning(
                    "[EXIT_EVAL_FLOW_BLOCKED_PROOF] model_eval_attempt_count=0 reason=%s io_only=%s min_hold_bars=%s exit_max_bars_held=%s",
                    primary_reason or "unknown",
                    1 if bool(getattr(self._runner, "exit_io_only_replay", False)) else 0,
                    2,
                    int(getattr(self, "_exit_max_bars_held", 0)),
                )
            try:
                log_dir = Path(getattr(runner, "log_dir", "")) if runner is not None else None
                out_dir = Path(getattr(runner, "output_dir", "")) if runner is not None else None
                if log_dir and log_dir.name == "logs":
                    out_dir = log_dir.parent
                if out_dir and str(out_dir):
                    out_dir.mkdir(parents=True, exist_ok=True)
                    model_proof_path = out_dir / "EXIT_RUNTIME_MODEL_PROOF.json"
                    feature_vector_path = out_dir / "EXIT_FEATURE_VECTOR_PROOF.json"
                    feature_groups_payload = {
                        grp: {"count": len(names), "features": list(names)}
                        for grp, names in self._exit_feature_groups.items()
                    }
                    feature_list_payload = [
                        {"name": name, "group": self._exit_feature_group_lookup.get(name, "unknown")}
                        for name in self._exit_feature_names
                    ]
                    feature_vector_proof = {
                        "io_version": self._exit_io_version,
                        "input_dim": self._exit_input_dim,
                        "feature_hash": compute_feature_names_hash(self._exit_feature_names),
                        "expected_hash": self._exit_feature_hash,
                        "feature_names": list(self._exit_feature_names),
                        "feature_groups": feature_groups_payload,
                        "feature_group_counts": self._exit_feature_group_counts,
                        "feature_list": feature_list_payload,
                    }
                    with model_proof_path.open("w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "model_path": model_path,
                                "model_dir": model_dir,
                                "config_path": config_path,
                                "io_version": io_version,
                                "input_dim": input_dim,
                                "feature_hash": feature_hash,
                                "uses_logits_path": int(uses_logits_path),
                                "feature_group_counts": self._exit_feature_group_counts,
                            },
                            f,
                            indent=2,
                        )
                    with feature_vector_path.open("w", encoding="utf-8") as f:
                        json.dump(feature_vector_proof, f, indent=2)
                    audit_path = out_dir / "EXIT_PROB_RANKING_AUDIT.json"
                    audit = {}
                    if audit_path.exists():
                        try:
                            with audit_path.open("r", encoding="utf-8") as f:
                                audit = json.load(f) or {}
                        except Exception:
                            audit = {}
                    audit["runtime_model_proof"] = {
                        "model_path": model_path,
                        "model_dir": model_dir,
                        "config_path": config_path,
                        "io_version": io_version,
                        "input_dim": input_dim,
                        "feature_hash": feature_hash,
                        "uses_logits_path": int(uses_logits_path),
                        "feature_group_counts": self._exit_feature_group_counts,
                    }
                    audit["runtime_feature_vector_proof"] = feature_vector_proof
                    audit["runtime_score_stats"] = {
                        "logit_mean": logit_dist["mean"],
                        "logit_std": logit_dist["std"],
                        "prob_mean": dist["mean"],
                        "prob_std": dist["std"],
                        "prob_min": dist["min"],
                        "prob_max": dist["max"],
                        "n": dist["count"],
                    }
                    if feat_arr.size and feat_arr.ndim == 2:
                        audit["runtime_feature_stats"] = {
                            "feature_mean": feat_mean.tolist(),
                            "feature_std": feat_std.tolist(),
                            "feature_min": feat_min.tolist(),
                            "feature_max": feat_max.tolist(),
                            "n": int(feat_arr.shape[0]),
                            "features_with_zero_variance": zero_var,
                            "features_with_low_variance": low_var,
                        }
                    audit["runtime_sample_proof"] = {
                        "n_model_evals_sampled": n_model_evals_sampled,
                        "n_unique_trades_sampled": n_unique_trades_sampled,
                    }
                    audit["eval_flow_proof"] = {
                        "trade_active_count": int(flow.get("trade_active_count", 0)),
                        "eligible_count": int(flow.get("eligible_count", 0)),
                        "blocked_pre_eval_count": int(flow.get("blocked_pre_eval_count", 0)),
                        "model_eval_attempt_count": int(flow.get("model_eval_attempt_count", 0)),
                        "model_eval_complete_count": int(flow.get("model_eval_complete_count", 0)),
                        "threshold_check_count": int(flow.get("threshold_check_count", 0)),
                        "blocked_reasons": blocked,
                    }
                    if int(flow.get("model_eval_attempt_count", 0)) == 0:
                        audit["eval_flow_blocked_proof"] = {
                            "reason": primary_reason or "unknown",
                            "io_only": 1 if bool(getattr(self._runner, "exit_io_only_replay", False)) else 0,
                            "min_hold_bars": 2,
                            "exit_max_bars_held": int(getattr(self, "_exit_max_bars_held", 0)),
                        }
                    with audit_path.open("w", encoding="utf-8") as f:
                        json.dump(audit, f, indent=2)
                    self._exit_prob_score_audit_written = True
            except Exception:
                pass
            if n <= 0:
                # Still allow runtime stats/logging; skip ranking section below.
                pass
            trade_states = getattr(self, "_exit_prob_trade_states", {}) or {}
            per_trade_counts = [len(v) for v in trade_states.values() if v]
            n_trades = int(len(per_trade_counts))
            active_states_total = int(sum(per_trade_counts))
            min_states = int(min(per_trade_counts)) if per_trade_counts else 0
            max_states = int(max(per_trade_counts)) if per_trade_counts else 0
            mean_states = float(sum(per_trade_counts) / n_trades) if n_trades > 0 else 0.0
            trades_ge_2 = int(sum(1 for c in per_trade_counts if c >= 2))
            trades_ge_5 = int(sum(1 for c in per_trade_counts if c >= 5))
            if runner is not None and bool(getattr(runner, "replay_mode", False)):
                replay_end_ts = getattr(runner, "replay_end_ts", None)
                replay_cur_ts = getattr(runner, "replay_current_ts", None) or getattr(runner, "_replay_current_ts", None)
                try:
                    if replay_end_ts is not None and replay_cur_ts is not None:
                        if pd.Timestamp(replay_cur_ts) < pd.Timestamp(replay_end_ts) and active_states_total < 20:
                            return
                except Exception:
                    pass
            b = getattr(self, "_exit_prob_buckets", {})
            log.info(
                "[EXIT_PROB_AUDIT] n=%d ge_thr=%d thr=%.4f min=%.4f max=%.4f buckets=%s model_dir=%s",
                n,
                int(getattr(self, "_exit_prob_ge_thr", 0)),
                float(getattr(self._runner, "exit_threshold", getattr(self, "exit_threshold", 0.0))),
                float(self._exit_prob_min if getattr(self, "_exit_prob_min", float("inf")) != float("inf") else 0.0),
                float(self._exit_prob_max if getattr(self, "_exit_prob_max", float("-inf")) != float("-inf") else 0.0),
                b,
                model_dir,
            )
            # Ranking audit (per-trade, per-bar) only if we have samples
            if n > 0 and not self._exit_prob_ranking_audit_written:
                if not self._exit_prob_sample_proof_logged:
                    log.info(
                        "[EXIT_PROB_RANKING_SAMPLE_PROOF] n_trades=%d n_active_bar_states=%d min_states_per_trade=%d max_states_per_trade=%d mean_states_per_trade=%.2f",
                        n_trades,
                        active_states_total,
                        min_states,
                        max_states,
                        mean_states,
                    )
                    self._exit_prob_sample_proof_logged = True
                def _mean_or_zero(vals: list[float]) -> float:
                    return float(np.mean(vals)) if vals else 0.0

                # Build full-series ranking buckets from all model evals per trade
                full_group_a: list[float] = []
                full_group_b: list[float] = []
                full_bars_before = {"b1": [], "b2": [], "b3": [], "b4_5": [], "b_gt5": []}
                for states in trade_states.values():
                    if not states:
                        continue
                    try:
                        total_bars = max(int(s[0]) for s in states)
                    except Exception:
                        continue
                    for b, p, _ in states:
                        bars_to_exit = total_bars - int(b)
                        if bars_to_exit <= 3:
                            full_group_a.append(float(p))
                        elif bars_to_exit > 10:
                            full_group_b.append(float(p))
                        if bars_to_exit == 1:
                            full_bars_before["b1"].append(float(p))
                        elif bars_to_exit == 2:
                            full_bars_before["b2"].append(float(p))
                        elif bars_to_exit == 3:
                            full_bars_before["b3"].append(float(p))
                        elif bars_to_exit in (4, 5):
                            full_bars_before["b4_5"].append(float(p))
                        elif bars_to_exit > 5:
                            full_bars_before["b_gt5"].append(float(p))

                group_a_mean = _mean_or_zero(full_group_a)
                group_b_mean = _mean_or_zero(full_group_b)
                late_mean = group_a_mean
                early_mean = group_b_mean
                delta = late_mean - early_mean
                b1_mean = _mean_or_zero(full_bars_before.get("b1", []))
                b2_mean = _mean_or_zero(full_bars_before.get("b2", []))
                b3_mean = _mean_or_zero(full_bars_before.get("b3", []))
                b4_5_mean = _mean_or_zero(full_bars_before.get("b4_5", []))
                b_gt5_mean = _mean_or_zero(full_bars_before.get("b_gt5", []))
                is_monotonic = bool(b1_mean > b3_mean > b_gt5_mean)
                insufficient_sample = bool(active_states_total < 20)
                if insufficient_sample:
                    log.warning(
                        "[EXIT_PROB_RANKING_PROOF] insufficient_sample=1 n_active_bar_states=%d",
                        active_states_total,
                    )
                log.info(
                    "[EXIT_RUNTIME_RANKING_DISTANCE_PROOF] b1=%.6f b2=%.6f b3=%.6f b4_5=%.6f b_gt5=%.6f counts=%s",
                    b1_mean,
                    b2_mean,
                    b3_mean,
                    b4_5_mean,
                    b_gt5_mean,
                    {
                        "b1": int(len(full_bars_before.get("b1", []))),
                        "b2": int(len(full_bars_before.get("b2", []))),
                        "b3": int(len(full_bars_before.get("b3", []))),
                        "b4_5": int(len(full_bars_before.get("b4_5", []))),
                        "b_gt5": int(len(full_bars_before.get("b_gt5", []))),
                    },
                )
                log.info(
                    "[EXIT_RUNTIME_RANKING_SEPARATION_PROOF] late_mean=%.6f early_mean=%.6f delta=%.6f is_monotonic=%d",
                    late_mean,
                    early_mean,
                    delta,
                    1 if is_monotonic else 0,
                )
                log.info(
                    "[EXIT_PROB_RANKING_PROOF] mean_A=%.6f mean_B=%.6f delta=%.6f n_A=%d n_B=%d",
                    late_mean,
                    early_mean,
                    delta,
                    len(full_group_a),
                    len(full_group_b),
                )

                audit = {
                    "runtime_model_proof": {
                        "model_path": model_path,
                        "model_dir": model_dir,
                        "config_path": config_path,
                        "io_version": io_version,
                        "input_dim": input_dim,
                        "feature_hash": feature_hash,
                        "uses_logits_path": int(uses_logits_path),
                    },
                    "dist_all_bars": dist,
                    "logit_stats": logit_dist,
                    "delta_counts": {
                        "increases": int(self._exit_prob_inc),
                        "decreases": int(self._exit_prob_dec),
                        "flat": int(self._exit_prob_flat),
                    },
                    "prob_at_entry": {
                        "count": int(len(self._exit_prob_at_entry)),
                        "mean": _mean_or_zero(self._exit_prob_at_entry),
                    },
                    "prob_at_exit": {
                        "count": int(len(self._exit_prob_at_exit)),
                        "mean": _mean_or_zero(self._exit_prob_at_exit),
                    },
                    "prob_before_exit": {
                        "1": {"count": int(len(self._exit_prob_before_exit[1])), "mean": _mean_or_zero(self._exit_prob_before_exit[1])},
                        "3": {"count": int(len(self._exit_prob_before_exit[3])), "mean": _mean_or_zero(self._exit_prob_before_exit[3])},
                        "5": {"count": int(len(self._exit_prob_before_exit[5])), "mean": _mean_or_zero(self._exit_prob_before_exit[5])},
                    },
                    "prob_at_mfe_peak": {
                        "count": int(len(self._exit_prob_at_mfe_peak)),
                        "mean": _mean_or_zero(self._exit_prob_at_mfe_peak),
                    },
                    "prob_at_mae_peak": {
                        "count": int(len(self._exit_prob_at_mae_peak)),
                        "mean": _mean_or_zero(self._exit_prob_at_mae_peak),
                    },
                    "ranking_groups": {
                        "mean_A": late_mean,
                        "mean_B": early_mean,
                        "delta": delta,
                        "n_A": int(len(full_group_a)),
                        "n_B": int(len(full_group_b)),
                        "insufficient_sample": bool(active_states_total < 20),
                    },
                    "ranking_distance": {
                        "b1": b1_mean,
                        "b2": b2_mean,
                        "b3": b3_mean,
                        "b4_5": b4_5_mean,
                        "b_gt5": b_gt5_mean,
                        "is_monotonic": bool(is_monotonic),
                        "counts": {
                            "b1": int(len(self._exit_prob_bars_before_exit.get("b1", []))),
                            "b2": int(len(self._exit_prob_bars_before_exit.get("b2", []))),
                            "b3": int(len(self._exit_prob_bars_before_exit.get("b3", []))),
                            "b4_5": int(len(self._exit_prob_bars_before_exit.get("b4_5", []))),
                            "b_gt5": int(len(self._exit_prob_bars_before_exit.get("b_gt5", []))),
                        },
                    },
                    "ranking_separation": {
                        "late_mean": late_mean,
                        "early_mean": early_mean,
                        "delta": delta,
                        "is_monotonic": bool(is_monotonic),
                    },
                    "active_states_total": active_states_total,
                    "active_states_per_trade": {
                        "count": n_trades,
                        "min": min_states,
                        "max": max_states,
                        "mean": mean_states,
                        "trades_with_ge_2_states": trades_ge_2,
                        "trades_with_ge_5_states": trades_ge_5,
                    },
                }

                try:
                    runner = getattr(self, "_runner", None)
                    log_dir = Path(getattr(runner, "log_dir", "")) if runner is not None else None
                    out_dir = Path(getattr(runner, "output_dir", "")) if runner is not None else None
                    if log_dir and log_dir.name == "logs":
                        out_dir = log_dir.parent
                    if out_dir and str(out_dir):
                        out_dir.mkdir(parents=True, exist_ok=True)
                        audit_path = out_dir / "EXIT_PROB_RANKING_AUDIT.json"
                        existing = {}
                        if audit_path.exists():
                            try:
                                with audit_path.open("r", encoding="utf-8") as f:
                                    existing = json.load(f) or {}
                            except Exception:
                                existing = {}
                        # Preserve runtime stats written by the earlier score-audit pass.
                        if "runtime_score_stats" in existing and "runtime_score_stats" not in audit:
                            audit["runtime_score_stats"] = existing["runtime_score_stats"]
                        if "eval_flow_proof" in existing and "eval_flow_proof" not in audit:
                            audit["eval_flow_proof"] = existing["eval_flow_proof"]
                        # Preserve runtime feature/sample stats written earlier.
                        if "runtime_feature_stats" in existing and "runtime_feature_stats" not in audit:
                            audit["runtime_feature_stats"] = existing["runtime_feature_stats"]
                        if "runtime_sample_proof" in existing and "runtime_sample_proof" not in audit:
                            audit["runtime_sample_proof"] = existing["runtime_sample_proof"]
                        with audit_path.open("w", encoding="utf-8") as f:
                            json.dump(audit, f, indent=2)
                        model_proof_path = out_dir / "EXIT_RUNTIME_MODEL_PROOF.json"
                        with model_proof_path.open("w", encoding="utf-8") as f:
                            json.dump(audit.get("runtime_model_proof", {}), f, indent=2)
                        try:
                            if model_dir:
                                train_audit_path = Path(model_dir) / "SCORE_COMPRESSION_AUDIT.json"
                                if train_audit_path.exists():
                                    with train_audit_path.open("r", encoding="utf-8") as tf:
                                        train_audit = json.load(tf) or {}
                                    train_scope = str(train_audit.get("scope") or "").strip()
                                    runtime_scope = "replay_open_trade_eval_windows"
                                    n_a = int(len(full_group_a))
                                    n_b = int(len(full_group_b))
                                    if not train_scope:
                                        log.info(
                                            "[EXIT_RUNTIME_PROB_COMPARISON_SKIPPED] "
                                            "reason=train_audit_scope_missing runtime_scope=%s n_A=%d n_B=%d model_dir=%s",
                                            runtime_scope,
                                            n_a,
                                            n_b,
                                            model_dir,
                                        )
                                    elif train_scope != runtime_scope:
                                        log.info(
                                            "[EXIT_RUNTIME_PROB_COMPARISON_SKIPPED] "
                                            "reason=scope_mismatch train_scope=%s runtime_scope=%s n_A=%d n_B=%d model_dir=%s",
                                            train_scope,
                                            runtime_scope,
                                            n_a,
                                            n_b,
                                            model_dir,
                                        )
                                    elif n_a <= 0 or n_b <= 0:
                                        log.info(
                                            "[EXIT_RUNTIME_PROB_COMPARISON_SKIPPED] "
                                            "reason=runtime_sample_not_bi_modal runtime_scope=%s n_A=%d n_B=%d model_dir=%s",
                                            runtime_scope,
                                            n_a,
                                            n_b,
                                            model_dir,
                                        )
                                    else:
                                        train_prob_mean = float(train_audit.get("prob_mean", 0.0))
                                        runtime_prob_mean = float(dist.get("mean", 0.0))
                                        delta_prob = abs(runtime_prob_mean - train_prob_mean)
                                        if delta_prob > 0.1:
                                            log.warning(
                                                "[EXIT_RUNTIME_PROB_MISMATCH_WARN] train_prob_mean=%.6f runtime_prob_mean=%.6f delta=%.6f",
                                                train_prob_mean,
                                                runtime_prob_mean,
                                                delta_prob,
                                            )
                        except Exception:
                            pass
                except Exception:
                    pass

                self._exit_prob_ranking_audit_written = True
            self._exit_prob_audit_logged = True
        except Exception:
            return

    def _should_trigger_cat_guard(
        self,
        *,
        window_arr: Any,
        pnl_bps_now: float,
    ) -> Optional[dict]:
        if not getattr(self, "_exit_cat_guard_enabled", False):
            return None
        try:
            arr = np.asarray(window_arr, dtype=np.float32)
            idx = dict(self._exit_feature_to_index)
            last = arr[-1]
            mfe_bps = float(last[idx["mfe_bps"]])
            dd_from_mfe_bps = float(last[idx["dd_from_mfe_bps"]])
            giveback_ratio = float(last[idx["giveback_ratio"]])
            mfe_decay_rate = float(last[idx["mfe_decay_rate"]])
        except Exception:
            return None
        if not all(map(math.isfinite, [mfe_bps, dd_from_mfe_bps, giveback_ratio])):
            return None
        if mfe_bps < float(self._exit_cat_guard_mfe_bps):
            return None
        if dd_from_mfe_bps < float(self._exit_cat_guard_dd_bps):
            return None
        if giveback_ratio < float(self._exit_cat_guard_giveback_ratio):
            return None
        pnl_limit = min(float(self._exit_cat_guard_pnl_max), float(self._exit_cat_guard_pnl_frac_mfe) * mfe_bps)
        if pnl_bps_now > pnl_limit:
            return None
        return {
            "mfe_bps": mfe_bps,
            "dd_from_mfe_bps": dd_from_mfe_bps,
            "giveback_ratio": giveback_ratio,
            "mfe_decay_rate": mfe_decay_rate,
            "pnl_limit": pnl_limit,
        }

    def _threshold_event_state(
        self,
        *,
        window_arr: Any,
        side: str,
        family_top_name: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Event-driven threshold arm:
        - proven edge (MFE)
        - clear deterioration from peak (DD/giveback)
        - signal direction no longer supports holding
        """
        side_norm = str(side or "").lower()
        if side_norm not in ("short", "long"):
            return None
        try:
            arr = np.asarray(window_arr, dtype=np.float32)
            idx = dict(self._exit_feature_to_index)
            last = arr[-1]
            mfe_bps = float(last[idx["mfe_bps"]])
            dd_from_mfe_bps = float(last[idx["dd_from_mfe_bps"]])
            giveback_ratio = float(last[idx["giveback_ratio"]])
            p_long = float(last[idx["p_long"]])
            p_short = float(last[idx["p_short"]])
            p_flat = float(last[idx["p_flat"]])
            p_hat = float(last[idx["p_hat"]])
            margin_top1_top2 = float(last[idx["margin_top1_top2"]])
            uncertainty_score = float(last[idx["uncertainty_score"]])
        except Exception:
            return None
        values = [
            mfe_bps,
            dd_from_mfe_bps,
            giveback_ratio,
            p_long,
            p_short,
            p_flat,
            p_hat,
            margin_top1_top2,
            uncertainty_score,
        ]
        if not all(map(math.isfinite, values)):
            return None
        is_short = side_norm == "short"
        mfe_min = float(self._exit_edge_death_mfe_min_short if is_short else self._exit_edge_death_mfe_min_long)
        dd_min = float(self._exit_edge_death_dd_min_short if is_short else self._exit_edge_death_dd_min_long)
        gb_min = float(self._exit_edge_death_gb_min_short if is_short else self._exit_edge_death_gb_min_long)
        if mfe_bps < mfe_min:
            return None
        if dd_from_mfe_bps < dd_min and giveback_ratio < gb_min:
            return None
        # Use deterioration family as a support signal; do not let direction alone
        # suppress protected-profit once edge death is already evident.
        directional_edge = (p_long - p_short) if side_norm == "long" else (p_short - p_long)
        family_norm = str(family_top_name or "").strip().upper()
        family_supports_protect = any(
            token in family_norm
            for token in (
                "DETERIORATION",
                "DEATH",
                "COLLAPSE",
                "ROT",
            )
        )
        if directional_edge > 0.02 and not family_supports_protect:
            return None
        return {
            "mfe_bps": mfe_bps,
            "dd_from_mfe_bps": dd_from_mfe_bps,
            "giveback_ratio": giveback_ratio,
            "p_long": p_long,
            "p_short": p_short,
            "p_flat": p_flat,
            "p_hat": p_hat,
            "margin_top1_top2": margin_top1_top2,
            "uncertainty_score": uncertainty_score,
            "directional_edge": directional_edge,
            "family_top_name": family_norm or None,
        }

    def _selective_post_edge_separator_state(
        self,
        *,
        window_arr: Any,
        side: str,
        session_current: str,
        pnl_bps_now: float,
        threshold_event: Optional[dict],
        threshold_event_armed: bool,
        exit_prob: float,
        profit_protect_prob: Optional[float] = None,
    ) -> Optional[dict]:
        if not bool(getattr(self, "_exit_post_edge_enabled", False)):
            return None
        side_norm = str(side or "").strip().lower()
        if side_norm not in getattr(self, "_exit_post_edge_allowed_sides", {"long"}):
            return None
        if str(session_current or "").strip().upper() not in getattr(self, "_exit_post_edge_allowed_sessions", {"OVERLAP"}):
            return None
        if bool(getattr(self, "_exit_post_edge_require_threshold_event", True)) and not bool(threshold_event_armed):
            return None
        arr = np.asarray(window_arr, dtype=np.float32)
        last = arr[-1]
        try:
            mfe_bps = float(last[self._exit_feature_index("mfe_bps")])
            dd_from_mfe_bps = float(last[self._exit_feature_index("dd_from_mfe_bps")])
            giveback_ratio = float(last[self._exit_feature_index("giveback_ratio")])
            time_since_mfe_bars = float(last[self._exit_feature_index("time_since_mfe_bars")])
        except Exception:
            return None
        directional_edge = float((threshold_event or {}).get("directional_edge", 0.0))
        values = [
            mfe_bps,
            dd_from_mfe_bps,
            giveback_ratio,
            time_since_mfe_bars,
            pnl_bps_now,
            directional_edge,
            exit_prob,
        ]
        if not all(map(math.isfinite, values)):
            return None
        if mfe_bps < float(getattr(self, "_exit_post_edge_mfe_min", 60.0)):
            return None
        if dd_from_mfe_bps < float(getattr(self, "_exit_post_edge_dd_min", 55.0)):
            return None
        if giveback_ratio < float(getattr(self, "_exit_post_edge_gb_min", 0.75)):
            return None
        if time_since_mfe_bars < float(getattr(self, "_exit_post_edge_time_since_mfe_min", 8.0)):
            return None
        pnl_cap = min(
            float(getattr(self, "_exit_post_edge_pnl_max", 10.0)),
            float(getattr(self, "_exit_post_edge_pnl_frac_mfe_max", 0.45)) * float(mfe_bps),
        )
        if pnl_bps_now < float(getattr(self, "_exit_post_edge_pnl_min", 2.0)):
            return None
        if pnl_bps_now > pnl_cap:
            return None
        if directional_edge > float(getattr(self, "_exit_post_edge_directional_edge_max", 0.02)):
            return None
        score_source = str(getattr(self, "_exit_post_edge_score_source", "exit_prob"))
        score_value: Optional[float]
        if score_source == "profit_protect_prob":
            score_value = profit_protect_prob
        else:
            score_value = exit_prob
        if score_value is None or not math.isfinite(float(score_value)):
            return None
        if float(score_value) < float(getattr(self, "_exit_post_edge_score_min", 0.70)):
            return None
        return {
            "score_source": score_source,
            "score_value": float(score_value),
            "mfe_bps": mfe_bps,
            "dd_from_mfe_bps": dd_from_mfe_bps,
            "giveback_ratio": giveback_ratio,
            "time_since_mfe_bars": time_since_mfe_bars,
            "pnl_cap": float(pnl_cap),
            "directional_edge": directional_edge,
            "profit_protect_prob": None if profit_protect_prob is None else float(profit_protect_prob),
        }

    def _protected_profit_floor_bps(self) -> float:
        runner = getattr(self, "_runner", None)
        exit_control = getattr(runner, "exit_control", None) if runner is not None else None
        allow_model_exit_when = getattr(exit_control, "allow_model_exit_when", None) if exit_control is not None else None
        floor_bps = None
        if isinstance(allow_model_exit_when, dict):
            floor_bps = allow_model_exit_when.get("min_realized_pnl_bps")
            if floor_bps is None:
                floor_bps = allow_model_exit_when.get("min_pnl_bps")
        elif allow_model_exit_when is not None:
            floor_bps = getattr(allow_model_exit_when, "min_realized_pnl_bps", None)
            if floor_bps is None:
                floor_bps = getattr(allow_model_exit_when, "min_pnl_bps", None)
        try:
            return float(floor_bps)
        except Exception:
            return float("nan")

    def _get_protected_profit_state(self, trade: Any) -> Optional[dict]:
        try:
            extra = getattr(trade, "extra", None)
            if not isinstance(extra, dict):
                return None
            state = extra.get("_protected_profit_state")
            if isinstance(state, dict) and bool(state.get("armed", False)):
                return dict(state)
        except Exception:
            return None
        return None

    def _set_protected_profit_state(self, trade: Any, state: dict) -> None:
        try:
            if not hasattr(trade, "extra") or trade.extra is None:
                trade.extra = {}
            trade.extra["_protected_profit_state"] = dict(state)
        except Exception:
            return

    def _protected_profit_state(
        self,
        *,
        trade: Any,
        window_arr: Any,
        side: str,
        session_current: str,
        pnl_bps_now: float,
        threshold_event: Optional[dict],
        threshold_event_armed: bool,
        post_edge_state: Optional[dict],
        profit_protect_prob: Optional[float],
        bars_in_trade_min: int,
    ) -> Optional[dict]:
        floor_bps = float(self._protected_profit_floor_bps())
        if not math.isfinite(floor_bps):
            return None

        current_state = self._get_protected_profit_state(trade)
        armed_state: Optional[dict] = dict(current_state) if current_state is not None else None
        newly_armed = False

        side_norm = str(side or "").strip().lower()
        if side_norm not in getattr(self, "_exit_post_edge_allowed_sides", {"long"}):
            return armed_state
        if armed_state is None and str(session_current or "").strip().upper() not in getattr(self, "_exit_post_edge_allowed_sessions", {"OVERLAP"}):
            return None

        # Arm only on proven edge + deterioration signals. The threshold_event is the
        # earlier arming signal; the strict post-edge separator can still be used as a
        # later fallback, but protected-profit must not depend on it exclusively.
        score_value = None
        score_source = None
        if profit_protect_prob is not None and math.isfinite(float(profit_protect_prob)):
            score_value = float(profit_protect_prob)
            score_source = "profit_protect_prob"
        elif post_edge_state is not None:
            score_value = float(post_edge_state.get("score_value", float("nan")))
            score_source = str(post_edge_state.get("score_source", "exit_prob"))

        if (
            armed_state is None
            and bool(threshold_event_armed)
            and threshold_event is not None
            and score_value is not None
            and math.isfinite(score_value)
            and score_value >= float(getattr(self, "_exit_post_edge_score_min", 0.70))
            and float(pnl_bps_now) >= floor_bps
        ):
            armed_state = {
                "armed": True,
                "armed_ts": None,
                "armed_bars_held": int(bars_in_trade_min),
                "armed_pnl_bps_now": float(pnl_bps_now),
                "be_plus_floor_bps": float(floor_bps),
                "score_source": str(score_source or "profit_protect_prob"),
                "score_value": float(score_value),
                "mfe_bps": float(threshold_event.get("mfe_bps", float("nan"))),
                "dd_from_mfe_bps": float(threshold_event.get("dd_from_mfe_bps", float("nan"))),
                "giveback_ratio": float(threshold_event.get("giveback_ratio", float("nan"))),
                "time_since_mfe_bars": float(threshold_event.get("time_since_mfe_bars", float("nan"))),
                "side": str(side),
                "session_current": str(session_current),
                "source": "THRESHOLD_EVENT",
                "should_exit": False,
                "newly_armed": True,
            }
            self._set_protected_profit_state(trade, armed_state)
            newly_armed = True

        if armed_state is None:
            return None

        try:
            arr = np.asarray(window_arr, dtype=np.float32)
            last = arr[-1]
            mfe_bps = float(last[self._exit_feature_index("mfe_bps")])
            dd_from_mfe_bps = float(last[self._exit_feature_index("dd_from_mfe_bps")])
            giveback_ratio = float(last[self._exit_feature_index("giveback_ratio")])
            time_since_mfe_bars = float(last[self._exit_feature_index("time_since_mfe_bars")])
        except Exception:
            mfe_bps = float(armed_state.get("mfe_bps", float("nan")))
            dd_from_mfe_bps = float(armed_state.get("dd_from_mfe_bps", float("nan")))
            giveback_ratio = float(armed_state.get("giveback_ratio", float("nan")))
            time_since_mfe_bars = float(armed_state.get("time_since_mfe_bars", float("nan")))

        if math.isfinite(mfe_bps):
            armed_state["mfe_bps"] = float(mfe_bps)
        if math.isfinite(dd_from_mfe_bps):
            armed_state["dd_from_mfe_bps"] = float(dd_from_mfe_bps)
        if math.isfinite(giveback_ratio):
            armed_state["giveback_ratio"] = float(giveback_ratio)
        if math.isfinite(time_since_mfe_bars):
            armed_state["time_since_mfe_bars"] = float(time_since_mfe_bars)
        armed_state["pnl_bps_now"] = float(pnl_bps_now)
        armed_state["be_plus_floor_bps"] = float(floor_bps)
        armed_state["score_value"] = float(score_value) if score_value is not None and math.isfinite(float(score_value)) else float(armed_state.get("score_value", float("nan")))
        armed_state["score_source"] = str(score_source or armed_state.get("score_source", "profit_protect_prob"))
        armed_state["armed"] = True
        armed_state["newly_armed"] = bool(newly_armed)
        armed_state["should_exit"] = bool(float(pnl_bps_now) <= float(floor_bps))
        if armed_state["should_exit"]:
            armed_state["exit_reason"] = "BE_PLUS_FLOOR"
        self._set_protected_profit_state(trade, armed_state)
        return armed_state

    def _canonical_residual_exit_state(self, *, window_arr: Any) -> Optional[dict]:
        try:
            arr = np.asarray(window_arr, dtype=np.float32)
            idx = dict(self._exit_feature_to_index)
            last = arr[-1]
            distance_from_peak_mfe_bps = float(last[idx["distance_from_peak_mfe_bps"]])
            minutes_since_session_open = float(last[idx["minutes_since_session_open"]])
            minutes_to_next_session_boundary = float(last[idx["minutes_to_next_session_boundary"]])
        except Exception:
            return None
        values = [
            distance_from_peak_mfe_bps,
            minutes_since_session_open,
            minutes_to_next_session_boundary,
        ]
        if not all(map(math.isfinite, values)):
            return None
        return {
            "distance_from_peak_mfe_bps": distance_from_peak_mfe_bps,
            "minutes_since_session_open": minutes_since_session_open,
            "minutes_to_next_session_boundary": minutes_to_next_session_boundary,
        }

    def _policy_friday_flat_state(
        self,
        *,
        now_ts: pd.Timestamp,
        window_arr: Any,
        pnl_bps_now: float,
    ) -> Optional[dict]:
        cfg = getattr(self._runner, "exit_weekly_management", None)
        if cfg is None or not bool(getattr(cfg, "friday_flat_enabled", False)):
            return None
        ts_utc = pd.Timestamp(now_ts)
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.tz_localize("UTC")
        else:
            ts_utc = ts_utc.tz_convert("UTC")
        weekday = int(ts_utc.weekday())
        if weekday not in tuple(getattr(cfg, "friday_flat_weekdays", ()) or ()):
            return None
        cutoff = getattr(cfg, "friday_flat_cutoff_hhmm_utc", None)
        if cutoff is None:
            return None
        cutoff_hour, cutoff_minute = int(cutoff[0]), int(cutoff[1])
        if (ts_utc.hour, ts_utc.minute) < (cutoff_hour, cutoff_minute):
            return None
        try:
            arr = np.asarray(window_arr, dtype=np.float32)
            last = arr[-1]
            mfe_bps = float(last[self._exit_feature_index("mfe_bps")])
            dd_from_mfe_bps = float(last[self._exit_feature_index("dd_from_mfe_bps")])
            giveback_ratio = float(last[self._exit_feature_index("giveback_ratio")])
            time_since_mfe_bars = float(last[self._exit_feature_index("time_since_mfe_bars")])
        except Exception:
            mfe_bps = float("nan")
            dd_from_mfe_bps = float("nan")
            giveback_ratio = float("nan")
            time_since_mfe_bars = float("nan")
        return {
            "reason": str(getattr(cfg, "friday_flat_reason", "POLICY_FRIDAY_FLAT")),
            "weekday": weekday,
            "weekday_name": ts_utc.day_name().upper(),
            "cutoff_hhmm_utc": f"{cutoff_hour:02d}:{cutoff_minute:02d}",
            "pnl_bps_now": float(pnl_bps_now),
            "mfe_bps": mfe_bps,
            "dd_from_mfe_bps": dd_from_mfe_bps,
            "giveback_ratio": giveback_ratio,
            "time_since_mfe_bars": time_since_mfe_bars,
        }

    def _maybe_log_exit_ml_event(
        self,
        event_ts: pd.Timestamp,
        prob_close: float,
        trade: Any,
        ctx_cont: Optional[List[float]] = None,
        ctx_cat: Optional[List[int]] = None,
    ) -> None:
        """
        Append one exit ML event to exits_<run_id>.jsonl with ctx_cont/ctx_cat (bundle-driven dims).
        Does nothing if required context columns are missing or dims mismatch.
        """
        try:
            runner = self._runner
            log_dir = getattr(runner, "log_dir", None)
            run_id = getattr(runner, "run_id", None)
            if log_dir is None or run_id is None:
                return
            if ctx_cont is None or ctx_cat is None:
                return
            expected_cont_dim, expected_cat_dim = self._get_expected_ctx_dims()
            if len(ctx_cont) != expected_cont_dim or len(ctx_cat) != expected_cat_dim:
                return
            exits_path = Path(log_dir) / "exits" / f"exits_{run_id}.jsonl"
            exits_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": event_ts.isoformat() if hasattr(event_ts, "isoformat") else str(event_ts),
                "trade_id": getattr(trade, "trade_id", None),
                "computed": {"mode": "exit_transformer_v0", "prob_close": float(prob_close)},
                "context": {"ctx_cont": ctx_cont, "ctx_cat": ctx_cat},
            }
            with exits_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
            try:
                if runner is not None:
                    runner.perf_exit_ml_event_rows = int(getattr(runner, "perf_exit_ml_event_rows", 0) + 1)
            except Exception:
                pass
            self._log_exit_ctx_event_proof("exit_ml_event", ctx_cont, ctx_cat)
        except Exception:
            return

    def _append_exit_io_record(
        self,
        *,
        event_ts: pd.Timestamp,
        trade: Any,
        prob_close: float,
        window_arr: Any,
        ctx_cont: Optional[List[float]],
        ctx_cat: Optional[List[int]],
    ) -> None:
        """
        Append full IO (T,D) for exit transformer evaluation to exits_<run_id>.jsonl.
        """
        try:
            runner = self._runner
            log_dir = getattr(runner, "log_dir", None)
            run_id = getattr(runner, "run_id", None)
            if log_dir is None or run_id is None:
                return
            window_len = int(self._exit_window_len)
            feature_dim = int(self._exit_input_dim)

            exits_path = Path(log_dir) / "exits" / f"exits_{run_id}.jsonl"
            exits_path.parent.mkdir(parents=True, exist_ok=True)

            # Normalize window to float list
            arr = np.asarray(window_arr, dtype=np.float32)
            if arr.shape != (window_len, feature_dim):
                raise RuntimeError(f"[EXIT_IO_SHAPE] expected ({window_len},{feature_dim}), got {arr.shape}")

            # Index map for scalars
            idx = dict(self._exit_feature_to_index)
            last = arr[-1]
            try:
                scalars = {
                    "pnl_bps_now": float(last[idx["pnl_bps_now"]]),
                    "mfe_bps": float(last[idx["mfe_bps"]]),
                    "mae_bps": float(last[idx["mae_bps"]]),
                    "dd_from_mfe_bps": float(last[idx["dd_from_mfe_bps"]]),
                    "giveback_ratio": float(last[idx["giveback_ratio"]]),
                    "bars_held": float(last[idx["bars_held"]]),
                    "time_since_mfe_bars": float(last[idx["time_since_mfe_bars"]]),
                    "atr_bps_now": float(last[idx["atr_bps_now"]]),
                }
            except KeyError as e:
                raise RuntimeError(f"[EXIT_IO_INDEX_MISSING] {e}")

            ctx_payload = None
            if ctx_cont and ctx_cat:
                expected_cont_dim, expected_cat_dim = self._get_expected_ctx_dims()
                if len(ctx_cont) == expected_cont_dim and len(ctx_cat) == expected_cat_dim:
                    ctx_payload = {"ctx_cont": ctx_cont, "ctx_cat": ctx_cat}

            entry_time = getattr(trade, "entry_time", None)
            entry_time_iso = entry_time.isoformat() if hasattr(entry_time, "isoformat") else (str(entry_time) if entry_time is not None else None)
            record = {
                "ts": event_ts.isoformat() if hasattr(event_ts, "isoformat") else str(event_ts),
                "run_id": run_id,
                "trade_uid": getattr(trade, "trade_uid", None),
                "trade_id": getattr(trade, "trade_id", None),
                "side": getattr(trade, "side", None),
                "entry_time": entry_time_iso,
                "open_ts_utc": entry_time_iso,
                "computed": {
                    "mode": "exit_transformer_v0",
                    "prob_close": float(prob_close),
                    "threshold": float(self.exit_threshold),
                },
                "context": ctx_payload,
                "io": {
                    "io_version": self._exit_io_version,
                    "feature_names_hash": self._exit_feature_hash,
                    "window_len": window_len,
                    "input_dim": feature_dim,
                    "io_features": arr.tolist(),
                },
                "scalars": scalars,
            }

            with exits_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
            try:
                if runner is not None:
                    runner.perf_exit_io_rows = int(getattr(runner, "perf_exit_io_rows", 0) + 1)
            except Exception:
                pass
            if ctx_payload is not None:
                self._log_exit_ctx_event_proof("exit_io_event", ctx_payload["ctx_cont"], ctx_payload["ctx_cat"])
        except Exception:
            return

    def _build_exit_ctx19_window(self, trade: Any, candles: pd.DataFrame, window_len: int) -> Optional[np.ndarray]:
        """
        Build (T,D) exit transformer input window for a single trade.
        """
        runner_obj = getattr(self, "_runner", None)

        def _accum_input_prep_metric(sec_name: str, dt: float) -> None:
            try:
                if runner_obj is None:
                    return
                setattr(runner_obj, sec_name, float(getattr(runner_obj, sec_name, 0.0) + float(dt)))
            except Exception:
                return

        t_hist_start = time.perf_counter()
        signal_history = list(getattr(self._runner, "exit_signal7_history", []))
        if len(signal_history) < window_len:
            if not getattr(trade, "_exit_warmup_logged", False):
                log.info(
                    "[EXIT_WARMUP_SKIP] insufficient signal7 history: have=%d need=%d trade_uid=%s trade_id=%s",
                    len(signal_history),
                    window_len,
                    getattr(trade, "trade_uid", None),
                    getattr(trade, "trade_id", None),
                )
                trade._exit_warmup_logged = True
            return None
        candle_tail = candles.tail(window_len)
        if len(candle_tail) < window_len:
            raise RuntimeError(
                f"[EXIT_TRANSFORMER_INPUT] insufficient bars for window_len={window_len}, have {len(candle_tail)}"
            )
        candle_rows = list(candle_tail.itertuples(index=True, name="CandleRow"))
        signal_slice = signal_history[-window_len:]
        _accum_input_prep_metric("t_exit_input_hist_select_sec", time.perf_counter() - t_hist_start)

        t_atr_start = time.perf_counter()
        runtime_atr_bps = self._compute_runtime_atr_bps(candles, period=5)
        if runtime_atr_bps is None:
            raise RuntimeError("[EXIT_IO_CONTRACT_VIOLATION] atr_bps_now unavailable")
        _accum_input_prep_metric("t_exit_input_runtime_atr_sec", time.perf_counter() - t_atr_start)

        # ctx_cont + ctx_cat per-bar snapshot (prebuilt SSoT, ENTRY-parity context base)
        t_ctx_contract_start = time.perf_counter()
        prebuilt_df = getattr(self._runner, "prebuilt_features_df", None)
        expected_ctx_cont = list(EXIT_FEATURE_GROUP_CTX_CONT)
        expected_ctx_cat = list(EXIT_FEATURE_GROUP_CTX_CAT)
        expected_cont_dim = len(expected_ctx_cont)
        expected_cat_dim = len(expected_ctx_cat)
        if prebuilt_df is None:
            raise RuntimeError("[EXIT_IO_CONTRACT_VIOLATION] prebuilt_features_df missing for ctx_cont/ctx_cat")
        session_keys = {
            "is_ASIA",
            "minutes_since_session_open",
            "minutes_to_next_session_boundary",
            "session_change_flag",
            "session_tradable",
        }
        missing_prebuilt = [c for c in expected_ctx_cont if c not in session_keys and c not in prebuilt_df.columns]
        missing_prebuilt.extend([c for c in expected_ctx_cat if c not in prebuilt_df.columns])
        if missing_prebuilt:
            raise RuntimeError(
                f"[EXIT_IO_CONTRACT_VIOLATION] ctx_cont/ctx_cat missing in prebuilt: {missing_prebuilt}"
            )
        _accum_input_prep_metric("t_exit_input_ctx_contract_sec", time.perf_counter() - t_ctx_contract_start)

        t_prebuilt_window_start = time.perf_counter()
        ts_index = pd.Index([row.Index for row in candle_rows])
        missing_ts = ts_index.difference(prebuilt_df.index)
        if len(missing_ts) > 0:
            raise RuntimeError(f"[EXIT_IO_CONTRACT_VIOLATION] ctx_cont/ctx_cat missing for ts={missing_ts[0]}")
        prebuilt_window_df = prebuilt_df.reindex(ts_index)
        _accum_input_prep_metric("t_exit_input_prebuilt_window_resolve_sec", time.perf_counter() - t_prebuilt_window_start)

        # Precompute session features for window (vectorized)
        t_session_start = time.perf_counter()
        try:
            from gx1.time.session_detector import (
                get_session_id_vectorized,
                get_session_minutes_since_open_vectorized,
                get_session_minutes_to_next_boundary_vectorized,
            )
            ts_series = pd.Series(ts_index)
            sess_ids = get_session_id_vectorized(ts_series)
            minutes_since = get_session_minutes_since_open_vectorized(ts_series)
            minutes_to = get_session_minutes_to_next_boundary_vectorized(ts_series)
            sess_change = (sess_ids.diff().fillna(0) != 0).astype(int)
            sess_tradable = (sess_ids != 0).astype(int)
        except Exception:
            sess_ids = None
            minutes_since = None
            minutes_to = None
            sess_change = None
            sess_tradable = None
        _accum_input_prep_metric("t_exit_input_session_features_sec", time.perf_counter() - t_session_start)
        for attr in ("p_long_entry", "p_hat_entry", "uncertainty_entry", "entropy_entry", "margin_entry"):
            if not hasattr(trade, attr):
                raise RuntimeError(f"[EXIT_IO_CONTRACT_VIOLATION] trade missing {attr} snapshot")

        entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
        entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
        local_mfe = float(getattr(trade, "mfe_bps", 0.0))
        local_mae = float(getattr(trade, "mae_bps", 0.0))
        local_mfe_last_bar = int(getattr(trade, "_mfe_last_bar", 0))
        current_bar_index = len(candles)
        entry_bar_index = int(getattr(trade, "entry_bar_index", current_bar_index))
        base_bar_index = current_bar_index - window_len

        slope_window = min(window_len, 8)

        def _linear_regression_slope(values: List[float], max_points: int) -> float:
            if not values:
                return 0.0
            if max_points > 0 and len(values) > max_points:
                values = values[-max_points:]
            if len(values) < 2:
                return 0.0
            y = np.asarray(values, dtype=np.float32)
            x = np.arange(len(y), dtype=np.float32)
            x_mean = float(np.mean(x))
            y_mean = float(np.mean(y))
            denom = float(np.sum((x - x_mean) ** 2))
            if denom <= 0.0:
                return 0.0
            slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
            if not np.isfinite(slope):
                return 0.0
            return slope

        window_rows: List[List[float]] = []
        pre_entry_count = 0
        prev_pnl: Optional[float] = None
        prev_velocity: Optional[float] = None
        prev_giveback: Optional[float] = None
        pnl_history_post_entry: List[float] = []
        t_row_loop_start = time.perf_counter()
        for idx, signal_dict in enumerate(signal_slice):
            candle_row = candle_rows[idx]
            try:
                bid_now = float(getattr(candle_row, "bid_close"))
                ask_now = float(getattr(candle_row, "ask_close"))
                bar_ts = pd.Timestamp(candle_row.Index)
            except Exception as e:
                raise RuntimeError(f"[EXIT_IO_CONTRACT_VIOLATION] missing bid/ask or ts in candle row: {e}") from e

            t_lookup_start = time.perf_counter()
            prebuilt_row = prebuilt_window_df.iloc[idx]
            _accum_input_prep_metric("t_exit_input_prebuilt_lookup_sec", time.perf_counter() - t_lookup_start)
            ctx_cont_row = []
            ctx_cat_row = []
            t_ctx_pack_start = time.perf_counter()
            for c in expected_ctx_cont:
                if c not in session_keys:
                    ctx_cont_row.append(float(prebuilt_row[c]))
                    continue
                if c in session_keys:
                    if sess_ids is None:
                        try:
                            from gx1.time.session_detector import (
                                get_session_id_vectorized,
                                get_session_minutes_since_open_vectorized,
                                get_session_minutes_to_next_boundary_vectorized,
                            )
                            ts_one = pd.Series([bar_ts])
                            sess_id_one = int(get_session_id_vectorized(ts_one).iloc[0])
                            minutes_since_one = float(get_session_minutes_since_open_vectorized(ts_one).iloc[0])
                            minutes_to_one = float(get_session_minutes_to_next_boundary_vectorized(ts_one).iloc[0])
                            prev_ts = ts_index[idx - 1] if idx > 0 else None
                            prev_sess = None
                            if prev_ts is not None:
                                prev_sess = int(get_session_id_vectorized(pd.Series([prev_ts])).iloc[0])
                            sess_change_one = float(1 if (prev_sess is not None and sess_id_one != prev_sess) else 0)
                            sess_tradable_one = float(1 if sess_id_one != 0 else 0)
                        except Exception:
                            raise RuntimeError(f"[EXIT_IO_CONTRACT_VIOLATION] session features unavailable for {c}")
                        if c == "is_ASIA":
                            ctx_cont_row.append(float(1 if sess_id_one == 0 else 0))
                        elif c == "minutes_since_session_open":
                            ctx_cont_row.append(float(minutes_since_one))
                        elif c == "minutes_to_next_session_boundary":
                            ctx_cont_row.append(float(minutes_to_one))
                        elif c == "session_change_flag":
                            ctx_cont_row.append(float(sess_change_one))
                        elif c == "session_tradable":
                            ctx_cont_row.append(float(sess_tradable_one))
                        else:
                            raise RuntimeError(f"[EXIT_IO_CONTRACT_VIOLATION] session key unknown: {c}")
                        continue
                    if c == "is_ASIA":
                        ctx_cont_row.append(float(1 if int(sess_ids.iloc[idx]) == 0 else 0))
                    elif c == "minutes_since_session_open":
                        ctx_cont_row.append(float(minutes_since.iloc[idx]))
                    elif c == "minutes_to_next_session_boundary":
                        ctx_cont_row.append(float(minutes_to.iloc[idx]))
                    elif c == "session_change_flag":
                        ctx_cont_row.append(float(sess_change.iloc[idx]))
                    elif c == "session_tradable":
                        ctx_cont_row.append(float(sess_tradable.iloc[idx]))
                    else:
                        raise RuntimeError(f"[EXIT_IO_CONTRACT_VIOLATION] session key unknown: {c}")
                    continue
                raise RuntimeError(f"[EXIT_IO_CONTRACT_VIOLATION] ctx_cont col missing: {c}")
            for c in expected_ctx_cat:
                try:
                    ctx_cat_row.append(float(prebuilt_row[c]))
                except Exception as e:
                    raise RuntimeError(f"[EXIT_IO_CONTRACT_VIOLATION] ctx_cat col missing: {c}") from e
            if len(ctx_cont_row) != expected_cont_dim:
                raise RuntimeError(
                    f"[EXIT_IO_CONTRACT_VIOLATION] ctx_cont dim mismatch: got={len(ctx_cont_row)} expected={expected_cont_dim}"
                )
            if len(ctx_cat_row) != expected_cat_dim:
                raise RuntimeError(
                    f"[EXIT_IO_CONTRACT_VIOLATION] ctx_cat dim mismatch: got={len(ctx_cat_row)} expected={expected_cat_dim}"
                )
            _accum_input_prep_metric("t_exit_input_ctx_pack_sec", time.perf_counter() - t_ctx_pack_start)

            # Prefer absolute bar index from candles index (stable across windows)
            bar_index_for_row = base_bar_index + idx + 1
            is_post_entry = bar_index_for_row >= entry_bar_index
            t_state_start = time.perf_counter()
            if is_post_entry:
                pnl_bps_now = compute_pnl_bps(entry_bid, entry_ask, bid_now, ask_now, trade.side)
                bars_held = max(0, bar_index_for_row - entry_bar_index)
                if pnl_bps_now > local_mfe:
                    local_mfe = pnl_bps_now
                    local_mfe_last_bar = bars_held
                if pnl_bps_now < local_mae:
                    local_mae = pnl_bps_now
                dd_from_mfe = max(0.0, local_mfe - pnl_bps_now)
                if local_mfe > 0:
                    giveback_ratio = dd_from_mfe / local_mfe
                else:
                    giveback_ratio = 0.0
                giveback_ratio = float(np.clip(giveback_ratio, 0.0, 2.0))
                time_since_mfe_bars = max(0.0, bars_held - local_mfe_last_bar)
                pnl_velocity = 0.0 if prev_pnl is None else float(pnl_bps_now - prev_pnl)
                pnl_acceleration = 0.0 if prev_velocity is None else float(pnl_velocity - prev_velocity)
                giveback_acceleration = 0.0 if prev_giveback is None else float(giveback_ratio - prev_giveback)
                distance_from_peak_mfe_bps = float(dd_from_mfe)
                if time_since_mfe_bars > 0.0:
                    mfe_decay_rate = float(distance_from_peak_mfe_bps / time_since_mfe_bars)
                else:
                    mfe_decay_rate = 0.0
                pnl_history_post_entry.append(float(pnl_bps_now))
                rolling_slope_since_entry = _linear_regression_slope(pnl_history_post_entry, slope_window)
                local_mfe_for_row = local_mfe
                local_mae_for_row = local_mae
                bars_held_for_row = float(bars_held)
            else:
                pre_entry_count += 1
                pnl_bps_now = 0.0
                local_mfe_for_row = 0.0
                local_mae_for_row = 0.0
                dd_from_mfe = 0.0
                giveback_ratio = 0.0
                bars_held_for_row = 0.0
                time_since_mfe_bars = 0.0
                pnl_velocity = 0.0
                pnl_acceleration = 0.0
                giveback_acceleration = 0.0
                distance_from_peak_mfe_bps = 0.0
                mfe_decay_rate = 0.0
                rolling_slope_since_entry = 0.0
            _accum_input_prep_metric("t_exit_input_trade_state_compute_sec", time.perf_counter() - t_state_start)

            # MODEL OBSERVABILITY AUDIT (EXIT trade-state) - gated, first pre/post per trade
            if os.getenv("GX1_MODEL_OBS_AUDIT", "0") == "1":
                try:
                    if not hasattr(self, "_exit_trade_state_audit_seen_pre"):
                        self._exit_trade_state_audit_seen_pre = set()
                    if not hasattr(self, "_exit_trade_state_audit_seen_post"):
                        self._exit_trade_state_audit_seen_post = set()
                    if not hasattr(self, "_exit_pnl_hist_audit_counts"):
                        self._exit_pnl_hist_audit_counts = {}

                    trade_uid = getattr(trade, "trade_uid", None) or getattr(trade, "trade_id", None)
                    phase = "post_entry" if is_post_entry else "pre_entry"
                    seen_set = self._exit_trade_state_audit_seen_post if is_post_entry else self._exit_trade_state_audit_seen_pre
                    if trade_uid is not None and trade_uid not in seen_set:
                        # For post-entry, wait for first row with bars_held > 0 if possible
                        if is_post_entry and float(bars_held_for_row) <= 0.0:
                            pass
                        else:
                            raw_sources = {
                                "trade.mfe_bps": float(getattr(trade, "mfe_bps", 0.0)),
                                "trade.mae_bps": float(getattr(trade, "mae_bps", 0.0)),
                                "trade._mfe_last_bar": float(getattr(trade, "_mfe_last_bar", 0.0)),
                                "entry_bar_index": int(entry_bar_index),
                                "current_bar_index": int(current_bar_index),
                                "bar_index_for_row": int(bar_index_for_row),
                            }
                            computed = {
                                "pnl_bps_now": float(pnl_bps_now),
                                "bars_held": float(bars_held_for_row),
                                "local_mfe_bps": float(local_mfe_for_row),
                                "local_mae_bps": float(local_mae_for_row),
                                "dd_from_mfe_bps": float(dd_from_mfe),
                                "distance_from_peak_mfe_bps": float(distance_from_peak_mfe_bps),
                                "time_since_mfe_bars": float(time_since_mfe_bars),
                            }
                            packed = {
                                "mfe_bps": float(local_mfe_for_row),
                                "mae_bps": float(local_mae_for_row),
                                "dd_from_mfe_bps": float(dd_from_mfe),
                                "distance_from_peak_mfe_bps": float(distance_from_peak_mfe_bps),
                                "bars_held": float(bars_held_for_row),
                                "time_since_mfe_bars": float(time_since_mfe_bars),
                            }
                            log.info(
                                "[EXIT_TRADE_STATE_AUDIT] phase=%s trade_uid=%s trade_id=%s ts=%s raw=%s computed=%s packed=%s",
                                phase,
                                trade_uid,
                                getattr(trade, "trade_id", None),
                                str(bar_ts),
                                json.dumps(raw_sources, sort_keys=True),
                                json.dumps(computed, sort_keys=True),
                                json.dumps(packed, sort_keys=True),
                            )
                            seen_set.add(trade_uid)

                    # Additional audit: raw inputs for pnl_velocity/pnl_acceleration/rolling_slope_since_entry
                    # Log first 3 post-entry rows for a trade to confirm pnl history evolves.
                    if is_post_entry and trade_uid is not None:
                        count = self._exit_pnl_hist_audit_counts.get(trade_uid, 0)
                        if count < 3:
                            hist_tail = pnl_history_post_entry[-5:] if len(pnl_history_post_entry) > 0 else []
                            audit_payload = {
                                "bar_index_for_row": int(bar_index_for_row),
                                "entry_bar_index": int(entry_bar_index),
                                "current_bar_index": int(current_bar_index),
                                "bars_held": float(bars_held_for_row),
                                "pnl_bps_now": float(pnl_bps_now),
                                "prev_pnl_bps": None if prev_pnl is None else float(prev_pnl),
                                "pnl_velocity": float(pnl_velocity),
                                "prev_velocity": None if prev_velocity is None else float(prev_velocity),
                                "pnl_acceleration": float(pnl_acceleration),
                                "slope_window": int(slope_window),
                                "pnl_history_len": int(len(pnl_history_post_entry)),
                                "pnl_history_tail": [float(x) for x in hist_tail],
                                "rolling_slope_since_entry": float(rolling_slope_since_entry),
                            }
                            log.info(
                                "[EXIT_PNL_HISTORY_AUDIT] trade_uid=%s trade_id=%s ts=%s payload=%s",
                                trade_uid,
                                getattr(trade, "trade_id", None),
                                str(bar_ts),
                                json.dumps(audit_payload, sort_keys=True),
                            )
                            self._exit_pnl_hist_audit_counts[trade_uid] = count + 1
                except Exception as e:
                    log.warning("[EXIT_TRADE_STATE_AUDIT] failed: %s", e)

            t_pack_start = time.perf_counter()
            row = [
                float(signal_dict["p_long"]),
                float(signal_dict["p_short"]),
                float(signal_dict["p_flat"]),
                float(signal_dict["p_hat"]),
                float(signal_dict["uncertainty_score"]),
                float(signal_dict["margin_top1_top2"]),
                float(signal_dict["entropy"]),
                float(trade.p_long_entry),
                float(trade.p_hat_entry),
                float(trade.uncertainty_entry),
                float(trade.entropy_entry),
                float(trade.margin_entry),
                float(pnl_bps_now),
                float(local_mfe_for_row),
                float(local_mae_for_row),
                float(dd_from_mfe),
                float(distance_from_peak_mfe_bps),
                float(bars_held_for_row),
                float(time_since_mfe_bars),
                float(mfe_decay_rate),
                float(pnl_velocity),
                float(pnl_acceleration),
                float(rolling_slope_since_entry),
                float(runtime_atr_bps),
                float(giveback_ratio),
                float(giveback_acceleration),
            ] + ctx_cont_row + ctx_cat_row
            extra_feature_names = self._exit_feature_names[len(EXIT_IO_V1_CTX36_FEATURES):]
            if extra_feature_names:
                phase_onehot = compute_m5_phase_onehot(bar_ts)
                for feat_name in extra_feature_names:
                    if feat_name in EXIT_IO_V3_CTX36_M1L512_PHASE5_EXTRA_FEATURES:
                        phase_idx = int(feat_name.rsplit("_", 1)[1])
                        row.append(float(phase_onehot[phase_idx]))
                    else:
                        raise RuntimeError(
                            f"[EXIT_IO_CONTRACT_VIOLATION] unsupported extra feature in active contract: {feat_name}"
                        )
            window_rows.append(row)
            _accum_input_prep_metric("t_exit_input_feature_pack_sec", time.perf_counter() - t_pack_start)

            prev_pnl = float(pnl_bps_now)
            prev_velocity = float(pnl_velocity)
            prev_giveback = float(giveback_ratio)
        _accum_input_prep_metric("t_exit_input_row_loop_sec", time.perf_counter() - t_row_loop_start)

        t_numpy_start = time.perf_counter()
        window_arr = np.asarray(window_rows, dtype=np.float32)
        expected_shape = (window_len, self._exit_input_dim)
        if window_arr.shape != expected_shape:
            raise RuntimeError(
                f"[EXIT_TRANSFORMER_INPUT] bad shape built: {window_arr.shape}, expected {expected_shape}"
            )
        if not np.isfinite(window_arr).all():
            raise RuntimeError("[EXIT_IO_CONTRACT_VIOLATION] non-finite values in exit transformer window")
        _accum_input_prep_metric("t_exit_input_numpy_finalize_sec", time.perf_counter() - t_numpy_start)

        t_contract_checks_start = time.perf_counter()
        idx = dict(self._exit_feature_to_index)
        strict = os.environ.get("GX1_EXIT_AUDIT_STRICT") == "1"
        eps = 1e-6

        # MODEL OBSERVABILITY AUDIT (EXIT) - gated, one-shot per run
        if os.getenv("GX1_MODEL_OBS_AUDIT", "0") == "1" and not getattr(self, "_exit_model_obs_audit_done", False):
            try:
                exit_feature_keys = list(self._exit_feature_names)
                io_version = self._exit_io_version
                feature_hash = compute_feature_names_hash(exit_feature_keys)

                def _classify(name: str) -> str:
                    lname = name.lower()
                    if lname.startswith("ctx_"):
                        return "ctx"
                    trade_markers = (
                        "pnl",
                        "mae",
                        "mfe",
                        "bars_held",
                        "time_since",
                        "giveback",
                        "dd_from_mfe",
                        "distance_from_peak",
                        "mfe_decay",
                        "pnl_velocity",
                        "pnl_acceleration",
                        "rolling_slope",
                        "entry_price",
                        "atr_bps_now",
                    )
                    if any(tok in lname for tok in trade_markers):
                        return "trade_state"
                    return "market"

                grouped = {"market": [], "ctx": [], "trade_state": []}
                for name in exit_feature_keys:
                    grouped[_classify(name)].append(name)

                if len(grouped["trade_state"]) == 0:
                    log.warning("[EXIT_HEALTH] TRADE_STATE_FEATURES_MISSING -> failfast kan ikke læres")

                # Feature stats on first eval batch
                feat_min = np.min(window_arr, axis=0)
                feat_max = np.max(window_arr, axis=0)
                feat_mean = np.mean(window_arr, axis=0)
                feat_std = np.std(window_arr, axis=0)
                constant_features = [name for name, s in zip(exit_feature_keys, feat_std) if float(s) == 0.0]

                # Cap-binding candidates: >=90% values at min or max (small window, heuristic)
                cap_binding_candidates: List[str] = []
                n_rows = int(window_arr.shape[0])
                for i, name in enumerate(exit_feature_keys):
                    col = window_arr[:, i]
                    at_min = np.sum(col == feat_min[i])
                    at_max = np.sum(col == feat_max[i])
                    if max(at_min, at_max) >= max(1, int(0.9 * n_rows)):
                        cap_binding_candidates.append(name)

                # Exit behavior diag (best-effort)
                last_exit_score = None
                try:
                    if getattr(self, "_exit_prob_all", None):
                        last_exit_score = float(self._exit_prob_all[-1])
                except Exception:
                    last_exit_score = None
                threshold = float(getattr(self, "exit_threshold", 0.0))

                # Contract fingerprint
                fp_payload = "|".join(exit_feature_keys) + f"|{io_version}|{feature_hash}"
                fp_hash = hashlib.sha256(fp_payload.encode("utf-8")).hexdigest()

                audit_payload = {
                    "model_kind": "EXIT",
                    "bundle_dir": None,
                    "io_version": io_version,
                    "feature_hash": feature_hash,
                    "exit_feature_keys": exit_feature_keys,
                    "feature_classification": grouped,
                    "input_health": {
                        "any_nan": bool(np.isnan(window_arr).any()),
                        "any_inf": bool(np.isinf(window_arr).any()),
                        "feature_stats": {
                            name: {
                                "min": float(mn),
                                "max": float(mx),
                                "mean": float(mu),
                                "std": float(sd),
                            }
                            for name, mn, mx, mu, sd in zip(exit_feature_keys, feat_min, feat_max, feat_mean, feat_std)
                        },
                        "constant_features": constant_features,
                        "cap_binding_candidates": cap_binding_candidates,
                    },
                    "exit_behavior_diag": {
                        "last_exit_score": last_exit_score,
                        "threshold": threshold,
                    },
                    "contract_fingerprint_sha256": fp_hash,
                }

                # Resolve bundle_dir if possible
                decider = getattr(self, "exit_transformer_decider", None)
                runner = getattr(self, "_runner", None)
                bundle_dir = None
                if decider is not None:
                    bundle_dir = getattr(decider, "bundle_dir", None) or getattr(decider, "model_path", None)
                if not bundle_dir and runner is not None:
                    bundle_dir = getattr(runner, "exit_transformer_model_dir", None) or getattr(runner, "exit_transformer_model_path", None)
                audit_payload["bundle_dir"] = bundle_dir

                # Hard-fail on NaN/Inf in audit mode
                if audit_payload["input_health"]["any_nan"] or audit_payload["input_health"]["any_inf"]:
                    raise RuntimeError("[EXIT_MODEL_OBS_AUDIT_FAIL] non-finite detected in exit input window")

                # Write audit artifacts
                out_dir = None
                if runner is not None and getattr(runner, "explicit_output_dir", None):
                    out_dir = Path(getattr(runner, "explicit_output_dir"))
                elif runner is not None and getattr(runner, "output_dir", None):
                    out_dir = Path(getattr(runner, "output_dir"))
                elif getattr(self, "explicit_output_dir", None):
                    out_dir = Path(getattr(self, "explicit_output_dir"))
                elif getattr(self, "output_dir", None):
                    out_dir = Path(getattr(self, "output_dir"))
                if out_dir is not None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    audit_path = out_dir / "MODEL_OBS_AUDIT_EXIT.json"
                    with audit_path.open("w", encoding="utf-8") as f:
                        json.dump(audit_payload, f, indent=2, sort_keys=True)

                    # Optional CSV stats
                    csv_path = out_dir / "EXIT_FEATURE_STATS.csv"
                    with csv_path.open("w", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["feature", "min", "max", "mean", "std", "constant", "cap_binding_candidate"])
                        for name, mn, mx, mu, sd in zip(exit_feature_keys, feat_min, feat_max, feat_mean, feat_std):
                            writer.writerow([
                                name,
                                float(mn),
                                float(mx),
                                float(mu),
                                float(sd),
                                1 if name in constant_features else 0,
                                1 if name in cap_binding_candidates else 0,
                            ])

                    log.info(
                        "[EXIT_MODEL_OBS_AUDIT] pass io_version=%s feature_hash=%s trade_state_features_count=%d fingerprint=%s path=%s",
                        io_version,
                        feature_hash,
                        len(grouped["trade_state"]),
                        fp_hash,
                        audit_path,
                    )
                else:
                    log.info(
                        "[EXIT_MODEL_OBS_AUDIT] pass io_version=%s feature_hash=%s trade_state_features_count=%d fingerprint=%s path=None",
                        io_version,
                        feature_hash,
                        len(grouped["trade_state"]),
                        fp_hash,
                    )

                self._exit_model_obs_audit_done = True
            except Exception as e:
                log.error("[EXIT_MODEL_OBS_AUDIT] failed: %s", e, exc_info=True)
                raise

        # One-shot audit log with snapshots
        if not getattr(self, "_exit_input_audit_logged_once", False):
            first_row_first7 = window_arr[0, :7].tolist()
            last_row_first7 = window_arr[-1, :7].tolist()
            entry_snapshots = {
                "p_long_entry": float(window_arr[0, idx["p_long_entry"]]),
                "p_hat_entry": float(window_arr[0, idx["p_hat_entry"]]),
                "uncertainty_entry": float(window_arr[0, idx["uncertainty_entry"]]),
                "entropy_entry": float(window_arr[0, idx["entropy_entry"]]),
                "margin_entry": float(window_arr[0, idx["margin_entry"]]),
            }
            last_row_scalars = {
                "pnl_bps_now": float(window_arr[-1, idx["pnl_bps_now"]]),
                "mfe_bps": float(window_arr[-1, idx["mfe_bps"]]),
                "mae_bps": float(window_arr[-1, idx["mae_bps"]]),
                "dd_from_mfe_bps": float(window_arr[-1, idx["dd_from_mfe_bps"]]),
                "giveback_ratio": float(window_arr[-1, idx["giveback_ratio"]]),
                "bars_held": float(window_arr[-1, idx["bars_held"]]),
                "time_since_mfe_bars": float(window_arr[-1, idx["time_since_mfe_bars"]]),
                "atr_bps_now": float(window_arr[-1, idx["atr_bps_now"]]),
            }
            log.info(
                "[EXIT_INPUT_AUDIT_ONESHOT] io_version=%s feature_hash=%s window_len=%d input_dim=%d first_row_first7=%s last_row_first7=%s entry_snapshots=%s last_row_scalars=%s pre_entry_count=%d",
                self._exit_io_version,
                self._exit_feature_hash,
                window_len,
                window_arr.shape[1],
                first_row_first7,
                last_row_first7,
                entry_snapshots,
                last_row_scalars,
                pre_entry_count,
            )
            self._exit_input_audit_logged_once = True

        if not getattr(self, "_exit_feature_vector_logged_once", False):
            log.info(
                "[EXIT_FEATURE_VECTOR_PROOF] input_dim=%d first_row_first7=%s last_row_last5=%s",
                window_arr.shape[1],
                window_arr[0, :7].tolist(),
                window_arr[-1, -5:].tolist(),
            )
            self._exit_feature_vector_logged_once = True

        if not getattr(self, "_exit_giveback_ratio_logged_once", False):
            series = window_arr[:, idx["giveback_ratio"]]
            mean_val = float(np.mean(series))
            p90_val = float(np.percentile(series, 90))
            log.info(
                "[EXIT_GIVEBACK_RATIO_PROOF] mean=%.6f p90=%.6f",
                mean_val,
                p90_val,
            )
            self._exit_giveback_ratio_logged_once = True

        # Sample a few windows for additional visibility
        if getattr(self, "_exit_input_audit_samples_count", 0) < 5:
            bars_series = window_arr[:, idx["bars_held"]]
            pnl_series = window_arr[:, idx["pnl_bps_now"]]
            mfe_series = window_arr[:, idx["mfe_bps"]]
            dd_series = window_arr[:, idx["dd_from_mfe_bps"]]
            ts_mfe_series = window_arr[:, idx["time_since_mfe_bars"]]
            atr_series = window_arr[:, idx["atr_bps_now"]]
            log.info(
                "[EXIT_INPUT_AUDIT_SAMPLE] trade_uid=%s side=%s bars_last=%.3f pnl_last=%.3f mfe_last=%.3f dd_last=%.3f ts_mfe_last=%.3f atr_last=%.3f pnl_min_max=%.3f/%.3f mfe_min_max=%.3f/%.3f dd_min_max=%.3f/%.3f bars_min_max=%.3f/%.3f",
                getattr(trade, "trade_uid", None),
                getattr(trade, "side", None),
                float(bars_series[-1]),
                float(pnl_series[-1]),
                float(mfe_series[-1]),
                float(dd_series[-1]),
                float(ts_mfe_series[-1]),
                float(atr_series[-1]),
                float(pnl_series.min()),
                float(pnl_series.max()),
                float(mfe_series.min()),
                float(mfe_series.max()),
                float(dd_series.min()),
                float(dd_series.max()),
                float(bars_series.min()),
                float(bars_series.max()),
            )
            self._exit_input_audit_samples_count += 1

        if strict:
            bars_series = window_arr[:, idx["bars_held"]]
            if not np.all(np.diff(bars_series) >= -eps):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] bars_held not non-decreasing")
            ts_mfe_series = window_arr[:, idx["time_since_mfe_bars"]]
            if not np.all(ts_mfe_series >= -eps):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] time_since_mfe_bars negative")
            pnl_series = window_arr[:, idx["pnl_bps_now"]]
            mfe_series = window_arr[:, idx["mfe_bps"]]
            dd_series = window_arr[:, idx["dd_from_mfe_bps"]]
            if not np.all(mfe_series + eps >= pnl_series):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] mfe_bps < pnl_bps_now")
            if not np.all(dd_series + eps >= 0):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] dd_from_mfe_bps negative beyond eps")
            atr_series = window_arr[:, idx["atr_bps_now"]]
            if not np.all(atr_series >= 0):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] atr_bps_now negative")
            # signal7 sanity: probabilities in [0,1] and sum≈1; other channels finite & >=0
            sig = window_arr[:, :7]
            if not np.isfinite(sig).all():
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] non-finite signal7")
            tol = 1e-3
            proba = sig[:, :3]
            if not np.all((proba >= -tol) & (proba <= 1 + tol)):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] p_long/p_short/p_flat out of [0,1]")
            sums = proba.sum(axis=1)
            if not np.all(np.abs(sums - 1.0) <= tol):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] p_long+p_short+p_flat != 1")
            p_hat = sig[:, 3]
            if not np.all((p_hat >= -tol) & (p_hat <= 1 + tol)):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] p_hat out of [0,1]")
            others = sig[:, 4:]
            if not np.all(others >= -tol):
                raise RuntimeError("[EXIT_INPUT_AUDIT_ASSERT] signal7 other channels negative")
        _accum_input_prep_metric("t_exit_input_contract_checks_sec", time.perf_counter() - t_contract_checks_start)

        try:
            if runner_obj is not None:
                runner_obj.n_exit_input_windows_built = int(getattr(runner_obj, "n_exit_input_windows_built", 0) + 1)
                n_windows = int(getattr(runner_obj, "n_exit_input_windows_built", 0))
                if n_windows == 1 or (n_windows % 2000) == 0:
                    breakdown = {
                        "history_selection_sec": float(getattr(runner_obj, "t_exit_input_hist_select_sec", 0.0)),
                        "runtime_atr_sec": float(getattr(runner_obj, "t_exit_input_runtime_atr_sec", 0.0)),
                        "ctx_contract_sec": float(getattr(runner_obj, "t_exit_input_ctx_contract_sec", 0.0)),
                        "prebuilt_window_resolve_sec": float(getattr(runner_obj, "t_exit_input_prebuilt_window_resolve_sec", 0.0)),
                        "session_features_sec": float(getattr(runner_obj, "t_exit_input_session_features_sec", 0.0)),
                        "row_loop_sec": float(getattr(runner_obj, "t_exit_input_row_loop_sec", 0.0)),
                        "prebuilt_lookup_sec": float(getattr(runner_obj, "t_exit_input_prebuilt_lookup_sec", 0.0)),
                        "ctx_pack_sec": float(getattr(runner_obj, "t_exit_input_ctx_pack_sec", 0.0)),
                        "trade_state_compute_sec": float(getattr(runner_obj, "t_exit_input_trade_state_compute_sec", 0.0)),
                        "feature_pack_sec": float(getattr(runner_obj, "t_exit_input_feature_pack_sec", 0.0)),
                        "numpy_finalize_sec": float(getattr(runner_obj, "t_exit_input_numpy_finalize_sec", 0.0)),
                        "contract_checks_sec": float(getattr(runner_obj, "t_exit_input_contract_checks_sec", 0.0)),
                    }
                    log.info(
                        "[EXIT_INPUT_PREP_BREAKDOWN] windows=%d total_sec=%.6f breakdown=%s",
                        n_windows,
                        float(getattr(runner_obj, "t_exit_input_prep_sec", 0.0)),
                        breakdown,
                    )
        except Exception:
            pass

        return window_arr

    def _collect_price_trace(self, candles: pd.DataFrame, now_ts: pd.Timestamp, open_trades: List[Any]) -> None:
        """
        Collect price trace for open trades (for intratrade metrics calculation).
        
        Stores high/low/close per bar in trade.extra["_price_trace"].
        Limited to last 5000 points to prevent memory bloat.
        """
        if len(candles) == 0:
            return
        
        try:
            # Get last closed bar prices
            last_bar = candles.iloc[-1]
            
            # Determine price source (prefer direct OHLC, fallback to bid/ask mid)
            if "high" in candles.columns and "low" in candles.columns and "close" in candles.columns:
                bar_high = float(last_bar["high"])
                bar_low = float(last_bar["low"])
                bar_close = float(last_bar["close"])
            elif "bid_high" in candles.columns and "ask_high" in candles.columns:
                bar_high = float((last_bar["bid_high"] + last_bar["ask_high"]) / 2.0)
                bar_low = float((last_bar["bid_low"] + last_bar["ask_low"]) / 2.0)
                bar_close = float((last_bar["bid_close"] + last_bar["ask_close"]) / 2.0)
            else:
                # Fallback: use close price for all
                if "close" in candles.columns:
                    bar_close = float(last_bar["close"])
                elif "bid_close" in candles.columns and "ask_close" in candles.columns:
                    bar_close = float((last_bar["bid_close"] + last_bar["ask_close"]) / 2.0)
                else:
                    return  # Cannot collect trace without price data
                bar_high = bar_close
                bar_low = bar_close
            
            # Append to price trace for each open trade
            for trade in open_trades:
                if not hasattr(trade, "extra") or trade.extra is None:
                    trade.extra = {}
                
                if "_price_trace" not in trade.extra:
                    trade.extra["_price_trace"] = []
                
                trace_point = {
                    "ts": now_ts.isoformat() if hasattr(now_ts, "isoformat") else str(now_ts),
                    "high": bar_high,
                    "low": bar_low,
                    "close": bar_close,
                }
                
                trade.extra["_price_trace"].append(trace_point)
                
                # Limit trace size (keep last 5000 points)
                max_trace_size = 5000
                if len(trade.extra["_price_trace"]) > max_trace_size:
                    trade.extra["_price_trace"] = trade.extra["_price_trace"][-max_trace_size:]
        
        except Exception as e:
            # Never break trading due to trace collection failure
            log.debug("[EXIT] Failed to collect price trace: %s", e)

    def _update_trade_extremes_current_bar(
        self, open_trades: List[Any], now_ts: pd.Timestamp, current_bid: float, current_ask: float
    ) -> None:
        """
        Update MFE/MAE per trade once per bar using current bid/ask.
        """
        for trade in open_trades:
            entry_bid = float(getattr(trade, "entry_bid", trade.entry_price))
            entry_ask = float(getattr(trade, "entry_ask", trade.entry_price))
            pnl_now = compute_pnl_bps(entry_bid, entry_ask, current_bid, current_ask, trade.side)
            bars_held = int(round((now_ts - trade.entry_time).total_seconds() / 300.0))
            if not hasattr(trade, "mfe_bps"):
                trade.mfe_bps = pnl_now
            if not hasattr(trade, "mae_bps"):
                trade.mae_bps = pnl_now
            if not hasattr(trade, "_mfe_last_bar"):
                trade._mfe_last_bar = bars_held
            if pnl_now > trade.mfe_bps:
                trade.mfe_bps = pnl_now
                trade._mfe_last_bar = bars_held
            if pnl_now < trade.mae_bps:
                trade.mae_bps = pnl_now
    
    def _log_trade_close_with_metrics(
        self,
        trade: Any,
        exit_time: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
        realized_pnl_bps: float,
        bars_held: int,
    ) -> None:
        """
        Log trade close to journal with intratrade metrics calculation.
        
        Helper function to ensure consistent logging across all exit paths.
        """
        # Record ranking stats for all non-threshold exits using last prob state
        if exit_reason != "THRESHOLD":
            self._record_exit_prob_on_close(
                trade=trade,
                bars_in_trade=bars_held,
                prob_close=None,
                exit_reason=exit_reason,
            )
        if not hasattr(self._runner, "trade_journal") or not self._runner.trade_journal:
            return
        
        try:
            from gx1.monitoring.trade_journal import EVENT_TRADE_CLOSED
            
            exit_time_iso = exit_time.isoformat() if hasattr(exit_time, "isoformat") else str(exit_time)

            # Calculate intratrade metrics from price trace
            intratrade_metrics = self._compute_intratrade_metrics(trade, exit_price)
            
            # Validate invariants with realized_pnl
            self._validate_intratrade_metrics(trade.trade_id, intratrade_metrics, realized_pnl_bps)
            
            # Remove _price_trace from trade.extra to prevent bloating trade logs
            # (metrics are already calculated and logged)
            if "_price_trace" in trade.extra:
                del trade.extra["_price_trace"]
            
            # Log exit summary (structured)
            self._runner.trade_journal.log_exit_summary(
                exit_time=exit_time_iso,
                trade_uid=trade.trade_uid,  # Primary key (COMMIT C)
                trade_id=trade.trade_id,  # Display ID (backward compatibility)
                exit_price=exit_price,
                exit_reason=exit_reason,
                realized_pnl_bps=realized_pnl_bps,
                max_mfe_bps=intratrade_metrics.get("max_mfe_bps"),
                max_mae_bps=intratrade_metrics.get("max_mae_bps"),
                intratrade_drawdown_bps=intratrade_metrics.get("intratrade_drawdown_bps"),
            )
            
            # Log trade closed (backward compatibility JSONL)
            self._runner.trade_journal.log(
                EVENT_TRADE_CLOSED,
                {
                    "exit_time": exit_time_iso,
                    "exit_price": exit_price,
                    "exit_profile": trade.extra.get("exit_profile", "UNKNOWN"),
                    "exit_reason": exit_reason,
                    "bars_held": bars_held,
                    "pnl_bps": realized_pnl_bps,
                    "final_status": "CLOSED",
                },
                trade_key={
                    "entry_time": trade.entry_time.isoformat() if hasattr(trade.entry_time, "isoformat") else str(trade.entry_time),
                    "entry_price": trade.entry_price,
                    "side": trade.side,
                },
                trade_id=trade.trade_id,
            )
        except Exception as e:
            log.warning("[TRADE_JOURNAL] Failed to log TRADE_CLOSED: %s", e)
    
    def _compute_intratrade_metrics(
        self,
        trade: Any,
        exit_price: float,
    ) -> Dict[str, Optional[float]]:
        """
        Compute intratrade metrics (MFE, MAE, intratrade drawdown) from price trace.
        
        Args:
            trade: Trade object with entry_price, side, and _price_trace
            exit_price: Final exit price
            
        Returns:
            Dict with max_mfe_bps, max_mae_bps, intratrade_drawdown_bps
        """
        try:
            price_trace = trade.extra.get("_price_trace", [])
            if not price_trace:
                return {
                    "max_mfe_bps": None,
                    "max_mae_bps": None,
                    "intratrade_drawdown_bps": None,
                }
            
            entry_price = trade.entry_price
            side = trade.side.lower()
            
            # Convert price trace to unrealized PnL curve (in bps)
            pnl_curve = []
            for point in price_trace:
                # Use high for favorable (long), low for adverse (long)
                # Use low for favorable (short), high for adverse (short)
                high_price = point["high"]
                low_price = point["low"]
                close_price = point["close"]
                
                if side == "long":
                    # Favorable: use high (best case)
                    favorable_pnl_bps = (high_price - entry_price) / entry_price * 10000.0
                    # Adverse: use low (worst case)
                    adverse_pnl_bps = (low_price - entry_price) / entry_price * 10000.0
                    # Current unrealized: use close
                    current_pnl_bps = (close_price - entry_price) / entry_price * 10000.0
                else:  # short
                    # Favorable: use low (best case for short)
                    favorable_pnl_bps = (entry_price - low_price) / entry_price * 10000.0
                    # Adverse: use high (worst case for short)
                    adverse_pnl_bps = (entry_price - high_price) / entry_price * 10000.0
                    # Current unrealized: use close
                    current_pnl_bps = (entry_price - close_price) / entry_price * 10000.0
                
                pnl_curve.append({
                    "favorable": favorable_pnl_bps,
                    "adverse": adverse_pnl_bps,
                    "current": current_pnl_bps,
                })
            
            # Add exit price to curve
            if side == "long":
                exit_pnl_bps = (exit_price - entry_price) / entry_price * 10000.0
            else:  # short
                exit_pnl_bps = (entry_price - exit_price) / entry_price * 10000.0
            
            pnl_curve.append({
                "favorable": exit_pnl_bps,
                "adverse": exit_pnl_bps,
                "current": exit_pnl_bps,
            })
            
            # Calculate MFE (max favorable excursion)
            max_mfe_bps = max(p["favorable"] for p in pnl_curve)
            
            # Calculate MAE (max adverse excursion) - convert to positive magnitude
            max_mae_bps_raw = min(p["adverse"] for p in pnl_curve)
            max_mae_bps = abs(max_mae_bps_raw)  # Positive magnitude (>=0)
            
            # Calculate intratrade drawdown (largest peak-to-trough drawdown)
            # Build unrealized PnL curve from current values
            unrealized_curve = [p["current"] for p in pnl_curve]
            
            max_drawdown_bps = 0.0
            peak = unrealized_curve[0] if unrealized_curve else 0.0
            
            for pnl in unrealized_curve:
                if pnl > peak:
                    peak = pnl
                drawdown = peak - pnl
                if drawdown > max_drawdown_bps:
                    max_drawdown_bps = drawdown
            
            # Ensure MFE is positive (should already be, but enforce)
            max_mfe_bps_abs = abs(float(max_mfe_bps)) if max_mfe_bps < 0 else float(max_mfe_bps)
            max_drawdown_bps_abs = abs(float(max_drawdown_bps)) if max_drawdown_bps < 0 else float(max_drawdown_bps)
            
            metrics = {
                "max_mfe_bps": max_mfe_bps_abs,
                "max_mae_bps": float(max_mae_bps),  # Already positive magnitude from line 1031
                "intratrade_drawdown_bps": max_drawdown_bps_abs,
            }
            
            # Note: Invariants validated in _log_trade_close_with_metrics() with realized_pnl
            
            return metrics
        
        except Exception as e:
            log.warning("[EXIT] Failed to compute intratrade metrics for trade %s: %s", trade.trade_id, e)
            return {
                "max_mfe_bps": None,
                "max_mae_bps": None,
                "intratrade_drawdown_bps": None,
            }
    
    def _validate_intratrade_metrics(
        self,
        trade_id: str,
        metrics: Dict[str, Optional[float]],
        realized_pnl_bps: Optional[float] = None,
    ) -> None:
        """
        Validate intratrade metrics invariants (soft warnings in prod).
        
        Invariants:
        - MFE >= 0 (favorable excursion magnitude)
        - MAE >= 0 (adverse excursion magnitude)
        - Intratrade DD >= 0 (drawdown magnitude)
        - If realized_pnl > 0, MFE should be >= realized_pnl (MFE is best case)
        
        On violation: log WARNING but do not block trading.
        """
        eps = 1e-6
        
        mfe_bps = metrics.get("max_mfe_bps")
        mae_bps = metrics.get("max_mae_bps")
        dd_bps = metrics.get("intratrade_drawdown_bps")
        
        # Validate MFE >= 0
        if mfe_bps is not None and mfe_bps < -eps:
            log.warning(
                "[INTRATRADE_INVARIANT] Trade %s: MFE violation (MFE=%.2f < 0). "
                "Metrics snapshot: MFE=%.2f, MAE=%.2f, DD=%.2f",
                trade_id, mfe_bps, mfe_bps, mae_bps, dd_bps
            )
        
        # Validate MAE magnitude >= 0
        if mae_bps is not None and mae_bps < -eps:
            log.warning(
                "[INTRATRADE_INVARIANT] Trade %s: MAE violation (MAE=%.2f < 0). "
                "Metrics snapshot: MFE=%.2f, MAE=%.2f, DD=%.2f",
                trade_id, mae_bps, mfe_bps, mae_bps, dd_bps
            )
        
        # Validate DD magnitude >= 0
        if dd_bps is not None and dd_bps < -eps:
            log.warning(
                "[INTRATRADE_INVARIANT] Trade %s: Drawdown violation (DD=%.2f < 0). "
                "Metrics snapshot: MFE=%.2f, MAE=%.2f, DD=%.2f",
                trade_id, dd_bps, mfe_bps, mae_bps, dd_bps
            )
        
        # Validate MFE >= realized_pnl (if realized_pnl > 0)
        if realized_pnl_bps is not None and realized_pnl_bps > eps:
            if mfe_bps is not None and mfe_bps + eps < realized_pnl_bps:
                log.warning(
                    "[INTRATRADE_INVARIANT] Trade %s: MFE < realized_pnl (MFE=%.2f < realized=%.2f). "
                    "Metrics snapshot: MFE=%.2f, MAE=%.2f, DD=%.2f, realized=%.2f",
                    trade_id, mfe_bps, realized_pnl_bps, mfe_bps, mae_bps, dd_bps, realized_pnl_bps
                )

    def _compute_runtime_atr_bps(self, candles: pd.DataFrame, period: int = 5) -> Optional[float]:
        """Compute mid-price ATR(5) in bps for adaptive exit overlays."""
        required_cols = [
            "bid_high",
            "bid_low",
            "bid_close",
            "ask_high",
            "ask_low",
            "ask_close",
        ]
        if not all(col in candles.columns for col in required_cols):
            return None
        window = candles[required_cols].tail(period + 2)
        if len(window) < period:
            return None
        mid_df = pd.DataFrame(
            {
                "high": (window["bid_high"] + window["ask_high"]) * 0.5,
                "low": (window["bid_low"] + window["ask_low"]) * 0.5,
                "close": (window["bid_close"] + window["ask_close"]) * 0.5,
            },
            index=window.index,
        )
        # DEL 3: Lazy import - only import when actually needed (baseline mode only)
        # live_features is forbidden in PREBUILT mode, so we only import it when needed
        from gx1.execution.live_features import compute_atr_bps
        atr_series = compute_atr_bps(mid_df, period=period)
        if atr_series.empty:
            return None
        latest = atr_series.iloc[-1]
        if pd.isna(latest):
            return None
        return float(latest)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #


if __name__ == "__main__":
    print("ExitManager smoke test: module import succeeded.")
