"""Observed-market Entry targets on trading-day horizons (operator decision 11.10.2026).

Explicitly bound successor to ``entry_observed_market_v1`` for the native D1
wave: the same observed, executable, teacher-free cash outcome, measured at
1, 5 and 20 trading days instead of a fixed count of M5 bars. The decision
authority is the 20-day horizon's unique argmax over LONG/SHORT/FLAT values in
bps; the shorter horizons are retained observation targets, never holding
limits.

Trading days are sessions on the one UTC session clock (boundary 22:00 UTC).
A horizon of ``h`` trading days ends in the session that is ``h`` observed
sessions after the entry session; sessions without any M1 bar (weekends,
holidays, source absence) are skipped, never counted. The exit is the exact
M1 quote at the same offset in the target observed session; a delayed quote
is not substituted. A row whose exact terminal quote is absent is censored
and excluded by the label owner, never filled.

Costs reuse the v1 economics owner: the round trip is charged once and
financing accrues over the actual wall-clock seconds between the entry
decision and the exit bar. Nothing here queries a model, clips an outcome or
grants serving authority.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch

from gx1.contracts.entry_causal_m1_outcomes_v1 import (
    CausalM1QuoteSource, _entry_notional_pnl_bps, _m1_arrays, _require_surface, causal_m1_target_contract,
)
from gx1.contracts.entry_fitted_q_v1 import ENTRY_FITTED_Q_ACTION_ORDER
from gx1.contracts.entry_observed_market_v1 import (
    SCHEMA_VERSION as PREDECESSOR_SCHEMA_VERSION, build_entry_observed_market_targets,
)
from gx1.time.session_detector import (
    TRADING_SESSION_BOUNDARY_OFFSET, TRADING_SESSION_CLOCK_SCHEMA_VERSION, TRADING_SESSION_DURATION,
)

SCHEMA_VERSION = "gx1_entry_observed_market_horizons_v2"
OBSERVATION_HORIZONS_TRADING_DAYS = (1, 5, 20)
DECISION_HORIZON_TRADING_DAYS = 20
GROSS_TARGET_COLUMNS_BY_HORIZON = {
    h: (f"y_long_gross_at_{h}d_bps", f"y_short_gross_at_{h}d_bps") for h in OBSERVATION_HORIZONS_TRADING_DAYS
}
ELAPSED_COLUMN_BY_HORIZON = {h: f"elapsed_seconds_at_{h}d" for h in OBSERVATION_HORIZONS_TRADING_DAYS}
BATCH_TARGET_KEY_BY_HORIZON = {h: f"entry_observed_market_target_bps_h{h}d" for h in OBSERVATION_HORIZONS_TRADING_DAYS}
DECISION_BATCH_TARGET_KEY = BATCH_TARGET_KEY_BY_HORIZON[DECISION_HORIZON_TRADING_DAYS]
# Operator decision 12.10.2026 (minimum native learning): the first planned learning run on the horizon
# target is the fit phase (1024 steps on TRAIN256); the V38 generation's frozen designs planned the
# 256-step native smoke. A frozen design's budget.planned_optimizer_steps must equal the value its own
# declared scope selects; the learning study's fixed fit budget reads the same constant.
MINIMUM_PLAN_FIT_OPTIMIZER_STEPS = 1024
# Operator decision 14.10.2026 (GX1_RULES): a smaller chronological window before any full epoch. The
# phase trains the eligible TRAIN rows whose decision lies in the last RECENT_WINDOW_DAYS before the
# measured eligibility cutoff, in the frozen epoch-0 order restricted to that window; its step budget is
# derived from the window size (one origin), never a literal.
RECENT_WINDOW_PHASE = "recent_year"
RECENT_WINDOW_DAYS = 365
# Operator decision 15.10.2026 («Ja, 1 og 2»): seed variants for the window phases, named here with the same
# date convention as the design seed 20260911; a window phase may declare exactly one of these or the design
# seed. Nothing else in the chain (normalization, targets, frozen order) depends on the seed; it changes only
# the fresh initialization and the training RNG stream.
RECENT_WINDOW_SEED_VARIANTS = (20261015, 20261016)


def allowed_prefix_training_seeds(design_seed: int) -> tuple[int, ...]:
    return (int(design_seed), *RECENT_WINDOW_SEED_VARIANTS)


RECENT_WINDOW_BATCH_SIZE = 16
RECENT_WINDOW_TRAIN_PROBE_ROWS = 4096
RECENT_WINDOW_TRAIN_OBSERVATION_STEPS = (0, 256, 1024)
V38_SMOKE_PLANNED_OPTIMIZER_STEPS = 256


def planned_prefix_optimizer_steps(design: Mapping[str, Any]) -> int:
    """The optimizer budget a frozen chronological design must plan: the minimum plan's fit budget when
    its scope declares the entry-only horizon phase, otherwise the V38 smoke convention."""
    scope = design.get("scope") if isinstance(design, Mapping) else None
    if isinstance(scope, Mapping) and scope.get("entry_only_training_phase") == ENTRY_ONLY_HORIZON_TRAINING_PHASE:
        return MINIMUM_PLAN_FIT_OPTIMIZER_STEPS
    return V38_SMOKE_PLANNED_OPTIMIZER_STEPS


def recent_window_rows(decision_ns: np.ndarray, *, cutoff_ns: int, window_days: int) -> np.ndarray:
    """Positions (into the given eligible decision clock) whose decision lies in (cutoff - window, cutoff]."""
    decision_ns = np.asarray(decision_ns, dtype=np.int64)
    if decision_ns.ndim != 1 or len(decision_ns) < 1 or type(window_days) is not int or window_days < 1:
        raise RuntimeError("ENTRY_STUDY_RECENT_WINDOW_INPUT_INVALID")
    start_ns = int(cutoff_ns) - int(window_days) * 86_400 * 10**9
    return np.flatnonzero((decision_ns > start_ns) & (decision_ns <= int(cutoff_ns))).astype(np.int64)


def recent_window_phase_spec(window_rows: int, *, window_days: int = RECENT_WINDOW_DAYS) -> dict[str, Any]:
    """The fixed phase derived from the window size: floor((n-1)/16) steps so the loader never completes
    an epoch (no VAL phase boundary), unique rows = steps x batch, TRAIN probe/CONTROL observed at the
    declared steps and at the end."""
    if type(window_rows) is not int or window_rows <= RECENT_WINDOW_TRAIN_PROBE_ROWS:
        raise RuntimeError("ENTRY_STUDY_RECENT_WINDOW_TOO_SMALL")
    steps = (window_rows - 1) // RECENT_WINDOW_BATCH_SIZE
    if steps <= max(RECENT_WINDOW_TRAIN_OBSERVATION_STEPS):
        raise RuntimeError("ENTRY_STUDY_RECENT_WINDOW_TOO_SMALL")
    return {"optimizer_steps": steps, "window_days": int(window_days), "window_rows": int(window_rows),
            "train_observation_steps": [*RECENT_WINDOW_TRAIN_OBSERVATION_STEPS, steps],
            "control_observation_steps": [0, steps], "max_invocations": 1,
            "unique_train_rows": steps * RECENT_WINDOW_BATCH_SIZE}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def entry_observed_market_horizons_contract() -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "predecessor_schema_version": PREDECESSOR_SCHEMA_VERSION,
        "action_order": list(ENTRY_FITTED_Q_ACTION_ORDER),
        "quote_contract": causal_m1_target_contract(),
        "session_clock": TRADING_SESSION_CLOCK_SCHEMA_VERSION,
        "session_boundary_utc_offset_seconds": int(TRADING_SESSION_BOUNDARY_OFFSET.total_seconds()),
        "observation_horizons_trading_days": list(OBSERVATION_HORIZONS_TRADING_DAYS),
        "decision_horizon_trading_days": DECISION_HORIZON_TRADING_DAYS,
        "horizon_rule": "h observed sessions after the entry session; sessions without M1 bars are skipped, never counted",
        "exit_rule": "exact M1 quote at the same offset in the target observed UTC session; absent or delayed terminal is censored",
        "historical_exchange_calendar_qualified": False,
        "calendar_semantics": "existing observed UTC22 session clock retained; no current NY-hours template backdated as historical truth",
        "elapsed_source": "actual wall-clock seconds from the entry decision to the exit bar (financing)",
        "gross_target_columns": {str(h): list(c) for h, c in GROSS_TARGET_COLUMNS_BY_HORIZON.items()},
        "elapsed_columns": {str(h): c for h, c in ELAPSED_COLUMN_BY_HORIZON.items()},
        "batch_target_keys": {str(h): k for h, k in BATCH_TARGET_KEY_BY_HORIZON.items()},
        "decision_batch_target_key": DECISION_BATCH_TARGET_KEY,
        "decision_authority": "unique argmax over the decision horizon's LONG/SHORT/FLAT values in bps; exact ties fail closed",
        "target": "observed_executable_terminal_cash_return_net_of_declared_costs",
        "target_unit": "raw_bps",
        "flat_target_bps": 0.0,
        "exit_model_used": False,
        "future_outcomes_used_as_model_inputs": False,
        "hindsight_best_exit_used": False,
        "negative_outcomes_clipped": False,
        "censored_rows_filled": False,
        "holding_time_limit": False,
        "horizon_search_allowed": False,
        "score_is_calibrated_probability": False,
        "serving_authority_granted": False,
    }
    return {**value, "contract_sha256": _sha(value)}


def session_day_index(times: pd.DatetimeIndex) -> np.ndarray:
    """Integer index of the UTC session (boundary 22:00) that contains each time."""
    index = pd.DatetimeIndex(times)
    if index.tz is None:
        raise RuntimeError("ENTRY_OBSERVED_HORIZONS_CLOCK_NAIVE")
    shifted = index.tz_convert("UTC") - TRADING_SESSION_BOUNDARY_OFFSET
    return (shifted.as_unit("ns").asi8 // int(TRADING_SESSION_DURATION.value)).astype(np.int64)


def causal_m1_terminal_outcomes_at_trading_days(
    *, fill_surface: pd.DataFrame, closed_m1: pd.DataFrame | CausalM1QuoteSource, horizon_trading_days: int,
) -> pd.DataFrame:
    """Exact executable terminal PnL ``horizon_trading_days`` observed sessions after entry."""
    if type(horizon_trading_days) is not int or horizon_trading_days < 1:
        raise RuntimeError("ENTRY_OBSERVED_HORIZONS_HORIZON_INVALID")
    surface = _require_surface(fill_surface)
    m1_times, values = _m1_arrays(closed_m1)
    m1_session = session_day_index(m1_times)
    sessions, first_row = np.unique(m1_session, return_index=True)  # observed sessions and their first M1 row
    size = len(surface)
    start = surface["entry_m1_row"].to_numpy(dtype=np.int64)
    fill_bound = surface["entry_fill_bound"].to_numpy(dtype=np.bool_)
    decision_at = pd.DatetimeIndex(surface["entry_decision_at"])
    valid = fill_bound & (start >= 0) & (start < len(m1_times))
    positions = np.flatnonzero(valid)
    valid[positions] &= m1_times.asi8[start[positions]] == decision_at.asi8[positions]
    exit_row = np.full(size, -1, dtype=np.int64)
    positions = np.flatnonzero(valid)
    if len(positions):
        entry_session = session_day_index(decision_at[positions])
        entry_pos = np.searchsorted(sessions, entry_session)
        in_range = (entry_pos < len(sessions)) & (entry_pos + horizon_trading_days < len(sessions))
        in_range &= sessions[np.minimum(entry_pos, len(sessions) - 1)] == entry_session
        target_pos = np.minimum(entry_pos + horizon_trading_days, len(sessions) - 1)
        session_start_ns = sessions * int(TRADING_SESSION_DURATION.value) + int(TRADING_SESSION_BOUNDARY_OFFSET.value)
        offset_ns = decision_at.asi8[positions] - session_start_ns[np.minimum(entry_pos, len(sessions) - 1)]
        candidate_ns = session_start_ns[target_pos] + offset_ns
        exit_candidate = np.searchsorted(m1_times.asi8, candidate_ns, side="left")
        ok = in_range & (exit_candidate < len(m1_times))
        same_session = np.zeros(len(positions), dtype=bool)
        same_session[ok] = m1_session[exit_candidate[ok]] == sessions[target_pos[ok]]
        ok &= same_session
        exact = np.zeros(len(positions), dtype=bool)
        exact[ok] = m1_times.asi8[exit_candidate[ok]] == candidate_ns[ok]
        ok &= exact
        exit_row[positions[ok]] = exit_candidate[ok]
        valid[positions[~ok]] = False
    exit_bid = np.full(size, np.nan)
    exit_ask = np.full(size, np.nan)
    exit_bid[valid] = values["bid_open"][exit_row[valid]]
    exit_ask[valid] = values["ask_open"][exit_row[valid]]
    entry_bid = surface["entry_bid"].to_numpy(dtype=np.float64)
    entry_ask = surface["entry_ask"].to_numpy(dtype=np.float64)
    long_pnl = np.full(size, np.nan)
    short_pnl = np.full(size, np.nan)
    if valid.any():
        long_pnl[valid], short_pnl[valid] = _entry_notional_pnl_bps(
            entry_bid=entry_bid[valid], entry_ask=entry_ask[valid], exit_bid=exit_bid[valid], exit_ask=exit_ask[valid])
        if not np.isfinite(long_pnl[valid]).all() or not np.isfinite(short_pnl[valid]).all():
            raise RuntimeError("ENTRY_OBSERVED_HORIZONS_OUTCOME_VALUES_INVALID")
    exit_time = pd.Series(pd.NaT, index=np.arange(size), dtype="datetime64[ns, UTC]")
    exit_time.iloc[valid] = m1_times[exit_row[valid]]
    elapsed = np.full(size, -1, dtype=np.int64)
    elapsed[valid] = (m1_times.asi8[exit_row[valid]] - decision_at.asi8[valid]) // 10**9
    if valid.any() and not bool((elapsed[valid] > 0).all()):
        raise RuntimeError("ENTRY_OBSERVED_HORIZONS_ELAPSED_INVALID")
    long_col, short_col = GROSS_TARGET_COLUMNS_BY_HORIZON.get(horizon_trading_days, (f"y_long_gross_at_{horizon_trading_days}d_bps", f"y_short_gross_at_{horizon_trading_days}d_bps"))
    return pd.DataFrame({
        "time": surface["time"].to_numpy(),
        "entry_decision_at": surface["entry_decision_at"].to_numpy(),
        "exit_decision_at": exit_time.to_numpy(),
        "horizon_trading_days": np.full(size, horizon_trading_days, dtype=np.int32),
        ELAPSED_COLUMN_BY_HORIZON.get(horizon_trading_days, f"elapsed_seconds_at_{horizon_trading_days}d"): elapsed,
        "entry_bid": entry_bid, "entry_ask": entry_ask, "exit_bid": exit_bid, "exit_ask": exit_ask,
        long_col: long_pnl, short_col: short_pnl, "outcome_valid": valid,
    })


@torch.no_grad()
def build_entry_observed_market_horizon_targets(
    *, gross_by_horizon: Mapping[int, torch.Tensor], elapsed_by_horizon: Mapping[int, torch.Tensor],
    economics: Mapping[str, Any],
) -> dict[int, dict[str, torch.Tensor]]:
    """Net LONG/SHORT/FLAT=0 per horizon through the predecessor's cost arithmetic; all horizons required."""
    if set(gross_by_horizon) != set(OBSERVATION_HORIZONS_TRADING_DAYS) or set(elapsed_by_horizon) != set(OBSERVATION_HORIZONS_TRADING_DAYS):
        raise RuntimeError("ENTRY_OBSERVED_HORIZONS_TARGET_INPUT_INVALID")
    rows = None
    out = {}
    for h in OBSERVATION_HORIZONS_TRADING_DAYS:
        built = build_entry_observed_market_targets(
            gross_return_bps=gross_by_horizon[h], elapsed_wall_clock_seconds=elapsed_by_horizon[h], economics=economics)
        if rows is None:
            rows = built["target_bps"].shape[0]
        elif built["target_bps"].shape[0] != rows:
            raise RuntimeError("ENTRY_OBSERVED_HORIZONS_TARGET_ROWS_DIFFER")
        out[h] = built
    return out


def require_entry_observed_market_horizon_batch(
    batch: Mapping[str, Any], *, device: torch.device,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Exact per-horizon batch keys (rule 17): every horizon present, finite, FLAT exactly zero."""
    out = {}
    rows = None
    for h, key in BATCH_TARGET_KEY_BY_HORIZON.items():
        targets = batch.get(key)
        if (
            not isinstance(targets, torch.Tensor) or targets.dtype != torch.float32 or targets.ndim != 2
            or targets.shape[1] != 3 or targets.shape[0] < 1 or not bool(torch.isfinite(targets).all())
            or not bool((targets[:, 2] == 0.0).all()) or (rows is not None and targets.shape[0] != rows)
        ):
            raise RuntimeError(f"ENTRY_OBSERVED_HORIZONS_BATCH_TARGET_INVALID:{key}")
        rows = targets.shape[0]
        targets = targets.detach().to(device)
        out[h] = (targets, torch.ones_like(targets, dtype=torch.bool))
    return out


# ---------------------------------------------------------------------------
# Learning design, dataset binding and batch dispatch (operator decision 12.10.2026:
# minimum Entry-only learning on the 20-trading-day target; 1 and 5 days observed only).
# ---------------------------------------------------------------------------
RECIPE_KEY = "entry_observed_market_horizons"
LEARNING_DESIGN_SCHEMA_VERSION = SCHEMA_VERSION + "_learning_design"
ENTRY_ONLY_HORIZON_TRAINING_PHASE = "entry_only_observed_horizon_20d"
OBSERVED_ONLY_BATCH_TARGET_KEYS = [
    BATCH_TARGET_KEY_BY_HORIZON[h] for h in OBSERVATION_HORIZONS_TRADING_DAYS if h != DECISION_HORIZON_TRADING_DAYS
]
HORIZON_LABEL_COLUMNS = [
    column for h in OBSERVATION_HORIZONS_TRADING_DAYS
    for column in (*GROSS_TARGET_COLUMNS_BY_HORIZON[h], ELAPSED_COLUMN_BY_HORIZON[h])
]


def require_entry_observed_market_horizons_scope(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the trading-day-horizon successor design: the v1 prefix/source/policy/cost checks,
    plus the exact decision horizon, loss key, observed-only keys, measured entry-eligibility
    cutoffs and the physical manifests' horizon-target block."""
    from pathlib import Path
    from gx1.contracts.entry_observed_market_v1 import _require_observed_market_plan
    from gx1.contracts.local_random_access_campaign_v2 import read_bound_json, require_binding

    base = _require_observed_market_plan(
        recipe, recipe_key=RECIPE_KEY, expected_schema=LEARNING_DESIGN_SCHEMA_VERSION,
        expected_target_contract=entry_observed_market_horizons_contract(),
        expected_training_phase=ENTRY_ONLY_HORIZON_TRAINING_PHASE,
        extra_plan_keys=frozenset({"decision_horizon_trading_days", "loss_target_batch_key",
                                   "observed_only_batch_keys", "entry_eligibility_cutoff_utc"}),
    )
    plan = base["plan"]
    cutoffs = plan["entry_eligibility_cutoff_utc"]
    if (
        plan["decision_horizon_trading_days"] != DECISION_HORIZON_TRADING_DAYS
        or plan["loss_target_batch_key"] != DECISION_BATCH_TARGET_KEY
        or plan["observed_only_batch_keys"] != OBSERVED_ONLY_BATCH_TARGET_KEYS
        or not isinstance(cutoffs, Mapping) or set(cutoffs) != {"train", "val"}
        or pd.Timestamp(cutoffs["train"]) > pd.Timestamp(base["train_control_cutoff"])
        or pd.Timestamp(cutoffs["val"]) > pd.Timestamp(base["development_control_entry_end_exclusive"])
    ):
        raise RuntimeError("ENTRY_OBSERVED_HORIZONS_DESIGN_INVALID")
    for split in ("train", "val"):
        checked = require_binding(base["physical_sources"][split]["manifest"],
                                  label="Entry horizon physical source manifest", verify_file=True)
        block = read_bound_json(Path(checked["path"]), checked["sha256"]).get("extra", {}).get("observed_market_horizons")
        if (
            not isinstance(block, Mapping) or block.get("contract") != entry_observed_market_horizons_contract()
            or block.get("censored_or_boundary_rows_are_nan") is not True
            or block.get("label_columns") != HORIZON_LABEL_COLUMNS
        ):
            raise RuntimeError("ENTRY_OBSERVED_HORIZONS_SOURCE_MANIFEST_INVALID")
    return {
        **base, "decision_horizon_trading_days": DECISION_HORIZON_TRADING_DAYS,
        "loss_target_batch_key": DECISION_BATCH_TARGET_KEY, "batch_target_keys": dict(BATCH_TARGET_KEY_BY_HORIZON),
        "entry_eligibility_cutoff_utc": {split: str(pd.Timestamp(cutoffs[split])) for split in ("train", "val")},
    }


def entry_observed_market_horizons_identity(scope: Mapping[str, Any]) -> dict[str, Any]:
    """v1 identity plus the horizon decision; bound in recipe, checkpoint and every observation."""
    from gx1.contracts.entry_observed_market_v1 import entry_observed_market_identity
    return {
        **entry_observed_market_identity(scope),
        "decision_horizon_trading_days": scope["decision_horizon_trading_days"],
        "loss_target_batch_key": scope["loss_target_batch_key"],
        "batch_target_keys": {str(h): key for h, key in scope["batch_target_keys"].items()},
        "entry_eligibility_cutoff_utc": dict(scope["entry_eligibility_cutoff_utc"]),
    }


def bind_entry_observed_market_horizons_dataset(dataset: Any, scope: Mapping[str, Any]) -> None:
    """Read the nine horizon label columns once; admit exactly the auxiliary-bound rows that carry
    complete outcomes at all three horizons and whose decision lies at or before the measured
    eligibility cutoff. Censoring (no closed M1 bar at the entry's session offset in the target
    session) is scattered through the whole period, so censored rows are excluded and counted,
    never filled; a complete row after the cutoff contradicts the measured design and fails closed."""
    import pyarrow.parquet as pq
    from pathlib import Path
    from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_DECISION_BAR_SECONDS
    from gx1.contracts.local_random_access_campaign_v2 import require_binding

    if getattr(dataset, "_entry_observed_market_binding", None) is not None:
        raise RuntimeError("ENTRY_OBSERVED_MARKET_ALREADY_BOUND")
    auxiliary = getattr(dataset, "_policy_dependent_auxiliary_binding", None)
    mask = getattr(dataset, "_policy_dependent_auxiliary_bound_rows", None)
    if (
        not isinstance(auxiliary, Mapping) or auxiliary.get("mode") != "original_physical_split_targets"
        or auxiliary.get("role") not in ("TRAIN", "CONTROL256", "CONTROL_STUDY")
        or (auxiliary.get("role") == "CONTROL_STUDY" and not isinstance(auxiliary.get("entry_learning_study"), Mapping))
        or not isinstance(mask, np.ndarray) or mask.dtype != np.bool_
        or mask.shape != (len(dataset.df),) or not bool(mask.any())
        or scope["training_phase"] != ENTRY_ONLY_HORIZON_TRAINING_PHASE
        or scope["target_contract"] != entry_observed_market_horizons_contract()
    ):
        raise RuntimeError("ENTRY_OBSERVED_HORIZONS_DATASET_SCOPE_INVALID")
    split = "train" if auxiliary["role"] == "TRAIN" else "val"
    source = scope["physical_sources"][split]
    if (
        Path(dataset.parquet_path) != Path(source["parquet"]["path"])
        or auxiliary.get("parent_entry_parquet") != source["parquet"]
        or auxiliary.get("parent_entry_manifest") != source["manifest"]
        or auxiliary.get("policies", {}).get("direction_policy", {}).get("policy_sha256") != scope["direction_policy_sha256"]
        or len(dataset.df) != source["physical_rows"]
    ):
        raise RuntimeError("ENTRY_OBSERVED_HORIZONS_DATASET_SOURCE_INVALID")
    require_binding(source["parquet"], label="observed Entry horizon physical targets", verify_file=True)
    frame = pq.read_table(dataset.parquet_path, columns=["time", *HORIZON_LABEL_COLUMNS]).to_pandas()
    clock = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True)).as_unit("ns").asi8
    input_clock = pd.DatetimeIndex(pd.to_datetime(dataset.df["time"], utc=True)).as_unit("ns").asi8
    if (
        len(frame) != len(dataset.df) or not np.array_equal(clock, input_clock)
        or hashlib.sha256(np.asarray(clock, dtype="<i8").tobytes()).hexdigest() != source["clock_sha256"]
    ):
        raise RuntimeError("ENTRY_OBSERVED_HORIZONS_DATASET_CLOCK_INVALID")
    rows = np.flatnonzero(mask)
    decision_ns = clock[rows] + ENTRY_DECISION_BAR_SECONDS * 10**9
    eligible = decision_ns <= pd.Timestamp(scope["entry_eligibility_cutoff_utc"][split]).value
    values = frame.iloc[rows][HORIZON_LABEL_COLUMNS].to_numpy(dtype=np.float64)
    complete = np.isfinite(values).all(axis=1)
    if bool((complete & ~eligible).any()):
        raise RuntimeError("ENTRY_OBSERVED_HORIZONS_COMPLETE_ROW_AFTER_ELIGIBILITY_CUTOFF")
    admitted = rows[eligible & complete]
    censored_inside_cutoff = int((eligible & ~complete).sum())
    if len(admitted) < 1:
        raise RuntimeError("ENTRY_OBSERVED_HORIZONS_NO_ADMITTED_ROWS")
    boundary = pd.Timestamp(scope["train_control_cutoff" if split == "train" else "development_control_entry_end_exclusive"]).value
    gross_by_horizon, elapsed_by_horizon = {}, {}
    for h in OBSERVATION_HORIZONS_TRADING_DAYS:
        gross = frame.iloc[admitted][list(GROSS_TARGET_COLUMNS_BY_HORIZON[h])].to_numpy(dtype=np.float32, copy=True)
        elapsed = frame.iloc[admitted][ELAPSED_COLUMN_BY_HORIZON[h]].to_numpy(dtype=np.float64)
        if bool((elapsed <= 0).any()) or bool((elapsed != np.floor(elapsed)).any()):
            raise RuntimeError("ENTRY_OBSERVED_HORIZONS_ELAPSED_INVALID")
        if bool((clock[admitted] + (ENTRY_DECISION_BAR_SECONDS + elapsed.astype(np.int64)) * 10**9 > boundary).any()):
            raise RuntimeError("ENTRY_OBSERVED_HORIZONS_OUTCOME_CROSSES_SPLIT")
        gross_by_horizon[h] = torch.from_numpy(gross)
        elapsed_by_horizon[h] = torch.from_numpy(elapsed.astype(np.int64))
    built = build_entry_observed_market_horizon_targets(
        gross_by_horizon=gross_by_horizon, elapsed_by_horizon=elapsed_by_horizon, economics=scope["economics"])
    targets = {}
    digests = {}
    for h in OBSERVATION_HORIZONS_TRADING_DAYS:
        observed = built[h]["target_bps"].numpy()
        full = np.full((len(frame), 3), np.nan, dtype=np.float32)
        full[admitted] = observed
        full.setflags(write=False)
        targets[h] = full
        digests[str(h)] = hashlib.sha256(np.asarray(observed, dtype="<f4").tobytes()).hexdigest()
    admitted_mask = np.zeros(len(frame), dtype=bool)
    admitted_mask[admitted] = True
    admitted_mask.setflags(write=False)
    binding = {
        "schema_version": SCHEMA_VERSION + "_dataset",
        "identity": entry_observed_market_horizons_identity(scope),
        "role": auxiliary["role"], "physical_source": source,
        "admitted_rows": int(len(admitted)), "auxiliary_bound_rows": int(len(rows)),
        "censored_rows_inside_cutoff": censored_inside_cutoff,
        "rows_after_cutoff": int((~eligible).sum()),
        "admitted_row_indices_sha256": hashlib.sha256(np.asarray(admitted, dtype="<i8").tobytes()).hexdigest(),
        "admitted_target_bps_sha256_by_horizon": digests,
    }
    dataset._entry_observed_market_horizon_targets = targets
    dataset._entry_observed_market_admitted_rows = admitted_mask
    dataset._entry_observed_market_binding = {**binding, "binding_sha256": _sha(binding)}


def observed_entry_batch_items(dataset: Any, row: int) -> dict[str, torch.Tensor]:
    """The exact supervision keys the Dataset emits for one row (rule 17): the v1 key when the
    v1 target is bound, the three horizon keys when the horizon targets are bound, never both."""
    v1 = getattr(dataset, "_entry_observed_market_targets", None)
    horizons = getattr(dataset, "_entry_observed_market_horizon_targets", None)
    if v1 is not None and horizons is not None:
        raise RuntimeError("ENTRY_OBSERVED_MARKET_TARGET_KEYS_AMBIGUOUS")
    if v1 is not None:
        # Output supervision only; never append future outcomes to inputs.
        return {"entry_observed_market_target_bps": torch.tensor(v1[row])}
    if horizons is not None:
        return {BATCH_TARGET_KEY_BY_HORIZON[h]: torch.tensor(horizons[h][row]) for h in OBSERVATION_HORIZONS_TRADING_DAYS}
    return {}


def observed_entry_batch_targets(
    batch: Mapping[str, Any], binding: Mapping[str, Any], *, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The loss target for an Entry-only batch: the bound design decides which exact key."""
    from gx1.contracts.entry_observed_market_v1 import entry_observed_market_batch_targets
    identity = binding["identity"]
    if identity.get("loss_target_batch_key") is None:
        return entry_observed_market_batch_targets(batch, device=device)
    if identity["loss_target_batch_key"] != DECISION_BATCH_TARGET_KEY:
        raise RuntimeError("ENTRY_OBSERVED_HORIZONS_LOSS_KEY_INVALID")
    return require_entry_observed_market_horizon_batch(batch, device=device)[DECISION_HORIZON_TRADING_DAYS]


def observed_entry_batch_targets_by_horizon(
    batch: Mapping[str, Any], binding: Mapping[str, Any], *, device: torch.device,
) -> dict[int, torch.Tensor]:
    """Every observed horizon target of a horizon-bound batch (observation only), else empty."""
    if binding["identity"].get("loss_target_batch_key") is None:
        return {}
    return {h: pair[0] for h, pair in require_entry_observed_market_horizon_batch(batch, device=device).items()}


def require_observed_entry_scope(recipe: Mapping[str, Any]) -> dict[str, Any] | None:
    """Exactly one explicit Entry-only design per recipe: v1 or the horizon successor."""
    from gx1.contracts.entry_observed_market_v1 import require_entry_observed_market_scope
    present = [key for key in ("entry_observed_market", RECIPE_KEY) if key in recipe]
    if not present:
        return None
    if len(present) != 1:
        raise RuntimeError("ENTRY_OBSERVED_MARKET_DESIGNS_AMBIGUOUS")
    if present[0] == RECIPE_KEY:
        return require_entry_observed_market_horizons_scope(recipe)
    return require_entry_observed_market_scope(recipe)


def observed_entry_identity(scope: Mapping[str, Any]) -> dict[str, Any]:
    from gx1.contracts.entry_observed_market_v1 import entry_observed_market_identity
    if "decision_horizon_trading_days" in scope:
        return entry_observed_market_horizons_identity(scope)
    return entry_observed_market_identity(scope)


def bind_observed_entry_dataset(dataset: Any, scope: Mapping[str, Any]) -> None:
    from gx1.contracts.entry_observed_market_v1 import bind_entry_observed_market_dataset
    if "decision_horizon_trading_days" in scope:
        bind_entry_observed_market_horizons_dataset(dataset, scope)
    else:
        bind_entry_observed_market_dataset(dataset, scope)
