"""One-use TRAIN-only hysteresis episode semantic audit; no target or model fit."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import time

REPO = Path("/home/andre2/src/GX1_CURRENT")
STAGE = "dynamic_mtf_state_hysteresis_semantics_20261106_001"
PLAN_PATH = REPO / "configs/research/DYNAMIC_MTF_STATE_HYSTERESIS_SEMANTICS_20261106_001.json"
ROOT = Path("/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_HYSTERESIS_SEMANTICS_20261106_001")
BUDGET = {"memory_max": 4294967296, "swap": 536870912, "pids": 64,
          "numerical_threads": 1, "max_wall_seconds": 1800,
          "max_invocations": 1, "max_json_bytes": 16777216}
MINUTE_NS = 60000000000
UNKNOWN, FLAT_ACTIVITY, ACTIVE = -1, 0, 1


def need(ok, key):
    if not ok:
        raise RuntimeError("HYSTERESIS_SEMANTICS_" + key)


def quantiles(values, np):
    values = np.asarray(values, np.float64)
    need(np.isfinite(values).all(), "FINITE_QUANTILES")
    if not len(values):
        return {"count": 0, "p10": None, "median": None, "p90": None}
    q = np.percentile(values, (10, 50, 90))
    return {"count": int(len(values)), "p10": float(q[0]),
            "median": float(q[1]), "p90": float(q[2])}


def latch_step(previous, complete, local_range, enter, release):
    need(previous in (UNKNOWN, FLAT_ACTIVITY, ACTIVE)
         and enter > release > 0 and local_range >= 0, "LATCH_DOMAIN")
    if not complete:
        return UNKNOWN
    if previous == ACTIVE:
        return FLAT_ACTIVITY if local_range < release else ACTIVE
    if previous == FLAT_ACTIVITY:
        return ACTIVE if local_range >= enter else FLAT_ACTIVITY
    if local_range >= enter:
        return ACTIVE
    if local_range < release:
        return FLAT_ACTIVITY
    return UNKNOWN


def candidate_path(axis, opens, highs, lows, closes, enter, local_range,
                   unknown_gap, window, step, np):
    """Source-validity tape; only a full post-reset W supports activity."""
    n = len(axis)
    need(n > 0 and window > 0 and all(len(x) == n for x in
         (opens, highs, lows, closes, enter, local_range, unknown_gap)), "PATH_SHAPE")
    need(np.all(axis[1:] > axis[:-1]) and np.all(enter >= 0)
         and np.all(local_range >= 0) and np.isfinite(enter).all()
         and np.isfinite(local_range).all(), "PATH_DOMAIN")
    activity = np.full(n, UNKNOWN, np.int8)
    direction = np.full(n, UNKNOWN, np.int8)
    state = np.full(n, UNKNOWN, np.int8)
    path = np.zeros(n, bool)
    complete_mask = np.zeros(n, bool)
    release = np.zeros(n, bool)
    quiet_resolution = np.zeros(n, bool)
    ambiguous_step = np.zeros(n, bool)
    count, latch, side, lo, hi, last_ambiguous = 0, UNKNOWN, UNKNOWN, float(opens[0]), float(opens[0]), -1
    for row in range(n):
        if unknown_gap[row] or enter[row] <= 0:
            count, latch, side = 0, UNKNOWN, UNKNOWN
            lo = hi = float(opens[row])
        if enter[row] <= 0:
            continue
        count += 1
        side, lo, hi, ambiguous = step(side, lo, hi,
            float(highs[row]), float(lows[row]), float(closes[row]), float(enter[row]))
        direction[row] = side
        ambiguous_step[row] = ambiguous
        if ambiguous:
            last_ambiguous = row
        complete = count >= window
        complete_mask[row] = complete
        path[row] = complete and (last_ambiguous < 0 or row - last_ambiguous >= window)
        prior = latch
        latch = latch_step(latch, complete, float(local_range[row]),
                           float(enter[row]), float(enter[row]) * 0.5)
        activity[row] = latch
        if latch == FLAT_ACTIVITY:
            state[row] = 1  # Existing DOWN/FLAT/UP numeric encoding.
        elif latch == ACTIVE and side in (0, 2):
            state[row] = side
        if latch == FLAT_ACTIVITY and local_range[row] < enter[row] * 0.5:
            if prior != FLAT_ACTIVITY:
                release[row] = prior == ACTIVE
                quiet_resolution[row] = prior == UNKNOWN
            # Every completed quiet bar seeds the next directional episode at
            # its close. A latched FLAT row in the middle band does not reset.
            side, lo, hi = UNKNOWN, float(closes[row]), float(closes[row])
    return {"activity": activity, "direction": direction, "state": state,
            "path_observed": path, "complete": complete_mask,
            "release": release, "quiet_resolution": quiet_resolution,
            "ambiguous_step": ambiguous_step}


def episode_report(axis, state, path, start, stop, horizon, np):
    lo, hi = np.searchsorted(axis, [start, stop])
    need(hi > lo and horizon > 0, "EPISODE_DOMAIN")
    code = np.where(path[lo:hi], state[lo:hi], UNKNOWN)
    clock = axis[lo:hi]
    result = {}
    for value, name in enumerate(("DOWN", "FLAT", "UP")):
        lengths, left, right = [], [], []
        i = 0
        while i < len(code):
            if code[i] != value:
                i += 1
                continue
            j = i
            while (j + 1 < len(code) and code[j + 1] == value
                   and clock[j + 1] - clock[j] == MINUTE_NS):
                j += 1
            confirmed_left = (i > 0 and clock[i] - clock[i - 1] == MINUTE_NS
                              and code[i - 1] >= 0 and code[i - 1] != value)
            confirmed_right = (j + 1 < len(code)
                               and clock[j + 1] - clock[j] == MINUTE_NS
                               and code[j + 1] >= 0 and code[j + 1] != value)
            lengths.append(j - i + 1)
            left.append(bool(confirmed_left))
            right.append(bool(confirmed_right))
            i = j + 1
        lengths = np.asarray(lengths, np.int64)
        left = np.asarray(left, bool)
        right = np.asarray(right, bool)
        n = int(left.sum())
        survived = int(np.count_nonzero(left & (lengths > horizon)))
        failed = int(np.count_nonzero(left & (lengths <= horizon) & right))
        censored = int(np.count_nonzero(left & (lengths <= horizon) & ~right))
        need(survived + failed + censored == n, "EPISODE_PARTITION")
        result[name] = {
            "segments": int(len(lengths)), "confirmed_starts": n,
            "left_truncated": int((~left).sum()),
            "confirmed_ends": int(right.sum()), "right_censored": int((~right).sum()),
            "both_bounds_confirmed": int((left & right).sum()),
            "observed_m1_minutes_lower_bound": quantiles(lengths, np),
            "completed_observed_m1_minutes": quantiles(lengths[left & right], np),
            "wall_elapsed_first_to_last_validity_minutes": quantiles(np.maximum(lengths - 1, 0), np),
            "exact_plus_own_tf_utc": {
                "horizon_minutes": horizon, "denominator_confirmed_starts": n,
                "survived_proven": survived, "ended_by_horizon_confirmed": failed,
                "censored_before_horizon": censored,
                "survival_lower_bound": survived / n if n else None,
                "survival_upper_bound": (survived + censored) / n if n else None}}
    return result


def synthetic_checks(np, target):
    need(latch_step(UNKNOWN, True, 0.5, 1.0, 0.5) == UNKNOWN
         and latch_step(UNKNOWN, True, 0.499, 1.0, 0.5) == FLAT_ACTIVITY
         and latch_step(UNKNOWN, True, 1.0, 1.0, 0.5) == ACTIVE
         and latch_step(ACTIVE, True, 0.5, 1.0, 0.5) == ACTIVE
         and latch_step(ACTIVE, True, 0.499, 1.0, 0.5) == FLAT_ACTIVITY
         and latch_step(FLAT_ACTIVITY, True, 0.999, 1.0, 0.5) == FLAT_ACTIVITY
         and latch_step(FLAT_ACTIVITY, True, 1.0, 1.0, 0.5) == ACTIVE,
         "SYNTHETIC_BOUNDARIES")
    axis = np.arange(7, dtype=np.int64) * MINUTE_NS
    o = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    h = np.array([100.2, 100.3, 100.3, 101.2, 101.2, 101.2, 101.2])
    l = np.array([99.8, 99.7, 99.7, 100.0, 100.0, 100.0, 100.0])
    c = np.array([100.0, 100.0, 100.0, 101.2, 101.2, 101.2, 101.2])
    theta = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    span = np.array([0.4, 0.4, 0.4, 1.2, 1.2, 1.2, 1.2])
    x = candidate_path(axis, o, h, l, c, theta, span,
                       np.zeros(7, bool), 3, target._step_direction, np)
    need(np.array_equal(x["complete"], [False, False, False, False, False, True, True])
         and x["activity"][3] == UNKNOWN and x["activity"][5] == ACTIVE,
         "SYNTHETIC_ATR_RESET")
    drift_axis = np.arange(8, dtype=np.int64) * MINUTE_NS
    drift_open = np.array([100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.8])
    drift_close = np.array([100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.8, 102.3])
    drift = candidate_path(drift_axis, drift_open, drift_close, drift_open,
        drift_close, np.ones(8), np.array([0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 1.1]),
        np.zeros(8, bool), 2, target._step_direction, np)
    need(drift["activity"][6] == FLAT_ACTIVITY
         and drift["direction"][6] == UNKNOWN
         and drift["activity"][7] == ACTIVE
         and drift["state"][7] == target.UP, "SYNTHETIC_QUIET_ANCHOR")
    gap_axis = np.array([0, 1, 2, 10, 11, 12], dtype=np.int64) * MINUTE_NS
    gap_open = np.full(6, 100.0)
    gap_high = np.array([102.0, 102.0, 102.0, 100.1, 100.1, 100.1])
    gap_low = np.full(6, 100.0)
    gap_close = np.full(6, 100.0)
    gap = candidate_path(gap_axis, gap_open, gap_high, gap_low, gap_close,
        np.ones(6), np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.1]),
        np.array([False, False, False, True, False, False]), 3,
        target._step_direction, np)
    need(np.array_equal(gap["activity"][3:], [UNKNOWN, UNKNOWN, FLAT_ACTIVITY])
         and not gap["complete"][3] and not gap["complete"][4]
         and gap["complete"][5], "SYNTHETIC_UNKNOWN_GAP")
    wide_axis = np.array([0], dtype=np.int64)
    interior = candidate_path(wide_axis, np.array([100.0]), np.array([101.2]),
        np.array([99.8]), np.array([100.5]), np.array([1.0]), np.array([1.4]),
        np.array([False]), 1, target._step_direction, np)
    endpoint = candidate_path(wide_axis, np.array([100.0]), np.array([101.2]),
        np.array([99.8]), np.array([101.2]), np.array([1.0]), np.array([1.4]),
        np.array([False]), 1, target._step_direction, np)
    need(interior["activity"][0] == ACTIVE and interior["state"][0] == UNKNOWN
         and not interior["path_observed"][0]
         and endpoint["state"][0] == target.UP
         and not endpoint["path_observed"][0], "SYNTHETIC_WIDE_BAR")
    code = np.array([1, 2, 2, 2, 1, -1, 2], np.int8)
    path = np.ones(7, bool)
    e = episode_report(axis, code, path, 0, 7 * MINUTE_NS, 2, np)
    need(e["UP"]["confirmed_starts"] == 1
         and e["UP"]["exact_plus_own_tf_utc"]["survived_proven"] == 1
         and e["UP"]["left_truncated"] == 1, "SYNTHETIC_EPISODE")
    bad = False
    try:
        latch_step(UNKNOWN, True, -0.1, 1.0, 0.5)
    except RuntimeError as exc:
        bad = str(exc) == "HYSTERESIS_SEMANTICS_LATCH_DOMAIN"
    need(bad, "SYNTHETIC_ERROR_KEY")
    return {"synthetic_success_key": "HYSTERESIS_SEMANTICS_SYNTHETIC_OK",
            "synthetic_error_key": "HYSTERESIS_SEMANTICS_LATCH_DOMAIN"}


def publish(name, value, old):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    need(len(raw) <= BUDGET["max_json_bytes"], "JSON_BUDGET")
    _publish_bytes(ROOT / name, raw)
    (ROOT / name).chmod(0o444)
    return old.ref(ROOT / name)


def audit(plan, binding, claim, snapshots, old, target, np):
    import pandas as pd
    import pyarrow.parquet as pq

    for item in plan["inputs"].values():
        need(old.ref(item["path"]) == item, "INPUT_HASH")
    prior = old.read(plan["inputs"]["prior_result"])
    semantic = old.read(plan["inputs"]["semantic_result"])
    transition = old.read(plan["inputs"]["transition_result"])
    semantic_terminal = old.read(plan["inputs"]["semantic_terminal"])
    transition_terminal = old.read(plan["inputs"]["transition_terminal"])
    need(prior["series"] == plan["inputs"]["prior_series"]
         and prior["target_sources"]["m1"] == plan["inputs"]["m1"]
         and prior["target_contract_sha256"] == plan["existing_contract_sha256"]
         and prior["test_data_used"] is False, "PRIOR_BINDING")
    for key, name in (("source_reference", "calendar_source_reference"),
                      ("closure_authority", "closure_authority"),
                      ("exact_schedule", "exact_schedule"),
                      ("published_calendar", "published_calendar")):
        need(prior["calendar_binding"][key] == plan["inputs"][name], "CALENDAR_BINDING")
    need(semantic_terminal["status"] == transition_terminal["status"] == "COMPLETE"
         and semantic_terminal["result"] == plan["inputs"]["semantic_result"]
         and transition_terminal["result"] == plan["inputs"]["transition_result"]
         and semantic["report_cutoff_ns"] == transition["report_cutoff_ns"] == plan["report_cutoff_ns"]
         and semantic["report_parents"] == plan["report_parents"]
         and semantic["model_calls"] == transition["model_calls"] == 0
         and semantic["test_data_used"] is transition["test_data_used"] is False
         and semantic["target_factor_selected"] is transition["factor_selected"] is False,
         "SAVED_RECEIPTS")
    manifest = old.read(plan["inputs"]["prior_series"])
    for item in manifest["arrays"].values():
        snapshots[item["path"]] = old.regular(item["path"])
    base = target.load_dynamic_mtf_state_series(plan["inputs"]["prior_series"],
        expected_source_refs=prior["target_sources"],
        expected_cutoff_ns=prior["fold"]["series_cutoff_ns"])
    a = base.arrays
    axis = a["validity_ns"]
    cutoff = plan["report_cutoff_ns"]
    prefix = int(np.searchsorted(axis, cutoff))
    issues = a["issue_ns"][:plan["report_parents"]]
    parents = a["parent_rows"][:plan["report_parents"]]
    need(np.array_equal(parents, np.arange(len(parents)))
         and int(np.searchsorted(a["issue_ns"], cutoff)) == len(issues)
         and axis[prefix - 1] < cutoff <= axis[prefix], "REPORT_CLOCK")
    proof = prior["clock_slice_proof"]
    source = pq.read_table(plan["inputs"]["m1"]["path"],
        columns=["time", "open", "high", "low", "close", "bid_close", "ask_close"],
        filters=[("time", ">=", pd.Timestamp(proof["requested_start_ns"], tz="UTC").to_pydatetime()),
                 ("time", "<", pd.Timestamp(proof["requested_end_exclusive_ns"], tz="UTC").to_pydatetime())],
        use_threads=False).to_pandas()
    need(len(source) == proof["selected_rows"]
         and np.array_equal(pd.DatetimeIndex(source["time"]).as_unit("ns").asi8 + MINUTE_NS, axis),
         "SOURCE_CLOCK")
    values = {k: source[k].to_numpy(np.float64) for k in
              ("open", "high", "low", "close", "bid_close", "ask_close")}
    need(all(np.isfinite(v).all() for v in values.values())
         and np.all(values["ask_close"] > values["bid_close"])
         and np.all(values["close"] > 0), "SOURCE_QUOTES")
    times = target._clock(source, target.EXIT_DECISION_BAR_SECONDS)
    known, closure_binding = target._known_closures(times, plan["inputs"]["m1"],
        old.read(plan["inputs"]["closure_authority"]),
        old.read(plan["inputs"]["exact_schedule"]))
    unknown_gap = np.zeros(prefix, bool)
    for row in np.flatnonzero(np.diff(axis[:prefix]) != MINUTE_NS) + 1:
        unknown_gap[row] = int(row) not in known
    need(closure_binding["authority_sha256"] ==
         old.read(plan["inputs"]["closure_authority"])["artifact_sha256"], "CLOSURE_HASH")
    monthly = {str(start): {} for start in plan["month_edges_ns"][:-1]}
    for col, tf in enumerate(target.TIMEFRAMES):
        theta_release = a["threshold"][:prefix, col]
        theta_enter = 2.0 * theta_release
        local_range = a["local_range"][:prefix, col]
        tape = candidate_path(axis[:prefix], *(values[k][:prefix] for k in
            ("open", "high", "low", "close")), theta_enter, local_range,
            unknown_gap, target.LOCAL_WINDOW_M1_BARS[tf], target._step_direction, np)
        need(np.all(~tape["complete"] | ((a["reason"][:prefix, col] & 7) == 0))
             and np.all(~tape["complete"] | (theta_enter > 0)), "COMPLETE_SUPPORT")
        for start, stop in zip(plan["month_edges_ns"][:-1], plan["month_edges_ns"][1:]):
            ilo, ihi = np.searchsorted(issues, [start, stop])
            slo, shi = np.searchsorted(axis[:prefix], [start, stop])
            lead = plan["lead_minutes"][col]
            wanted = issues[ilo:ihi] + lead * MINUTE_NS
            mature = wanted < cutoff
            pos = np.searchsorted(axis[:prefix], wanted)
            present = mature & (pos < prefix) & (axis[np.minimum(pos, prefix - 1)] == wanted)
            rows = pos[present]
            need(len(rows) > 0 and slo < shi, "MONTH_SUPPORT")
            saved = semantic["monthly"][str(start)][tf]
            # Existing numerical factors are checked only on their original
            # non-hysteretic activity domain. Candidate side can differ after reset.
            original_complete = ((a["reason"][rows, col] & 7) == 0) & (theta_release[rows] > 0)
            flat05 = original_complete & (local_range[rows] < theta_release[rows])
            flat10 = original_complete & (local_range[rows] < theta_enter[rows])
            for factor, flat in (("0.5", flat05), ("1.0", flat10)):
                reference = saved["factors"][factor]
                need(reference["complete_activity"] == int(original_complete.sum())
                     and reference["flat"] == int(flat.sum())
                     and reference["nonflat"] == int((original_complete & ~flat).sum()),
                     "SAVED_ACTIVITY_PARITY")
            base_known = a["observed"][rows, col]
            base_state = a["state"][rows, col]
            need(saved["factors"]["0.5"]["saved_threeclass_counts"] == {
                name: int(np.count_nonzero(base_known & (base_state == code)))
                for code, name in enumerate(target.STATE_NAMES)}, "SAVED_0_5_SIDE_PARITY")
            candidate_known = tape["state"][rows] >= 0
            candidate_activity = tape["activity"][rows]
            spread = values["ask_close"][rows] - values["bid_close"][rows]
            scale = tape["complete"][rows] & (spread > 0)
            threshold_only, binary_flips, atr_updated = 0, 0, 0
            state_flips, state_flips_at_atr_update = 0, 0
            side_flips, side_flips_atr_clock_changed, side_flips_atr_value_changed = 0, 0, 0
            for row in range(max(slo + 1, 1), shi):
                if axis[row] - axis[row - 1] != MINUTE_NS:
                    continue
                old_state, new_state = tape["state"][row - 1], tape["state"][row]
                atr_clock_changed = (a["atr_available_ns"][row, col] !=
                                     a["atr_available_ns"][row - 1, col])
                if old_state >= 0 and new_state >= 0 and old_state != new_state:
                    state_flips += 1
                    state_flips_at_atr_update += int(atr_clock_changed)
                    if {int(old_state), int(new_state)} == {target.DOWN, target.UP}:
                        side_flips += 1
                        side_flips_atr_clock_changed += int(atr_clock_changed)
                        side_flips_atr_value_changed += int(
                            theta_enter[row] != theta_enter[row - 1])
                left, right = tape["activity"][row - 1], tape["activity"][row]
                if left == right or left == UNKNOWN or right == UNKNOWN:
                    continue
                binary_flips += 1
                previous_boundary = theta_enter[row - 1] if right == ACTIVE else theta_release[row - 1]
                range_would_flip = (local_range[row] >= previous_boundary if right == ACTIVE
                                    else local_range[row] < previous_boundary)
                threshold_only += int(not range_would_flip)
                atr_updated += int(a["atr_available_ns"][row, col] !=
                                   a["atr_available_ns"][row - 1, col])
            need(threshold_only <= binary_flips, "TRANSITION_PARTITION")
            per = {
                "issues": int(ihi - ilo), "mature": int(mature.sum()),
                "exact_native_validity": int(present.sum()),
                "query_unavailable": int((mature & ~present).sum()),
                "immature": int((~mature).sum()),
                "activity_known": int((candidate_activity >= 0).sum()),
                "activity_unknown": int((candidate_activity == UNKNOWN).sum()),
                "active_with_unknown_side_up_or_down": int(((candidate_activity == ACTIVE)
                    & ~candidate_known).sum()),
                "direction_evidence_without_activity": int(((candidate_activity == UNKNOWN)
                    & (tape["direction"][rows] >= 0)).sum()),
                "direction_evidence_before_complete_window": int((~tape["complete"][rows]
                    & (tape["direction"][rows] >= 0)).sum()),
                "state_observed": int(candidate_known.sum()),
                "state_counts": {name: int(np.count_nonzero(candidate_known &
                    (tape["state"][rows] == code)))
                    for code, name in enumerate(target.STATE_NAMES)},
                "strict_trailing_window_path_observed": int(tape["path_observed"][rows].sum()),
                "physical_scale_at_exact_validity": {
                    "enter_usd": quantiles(theta_enter[rows][scale], np),
                    "release_usd": quantiles(theta_release[rows][scale], np),
                    "enter_bps": quantiles(10000 * theta_enter[rows][scale] /
                                           values["close"][rows][scale], np),
                    "release_bps": quantiles(10000 * theta_release[rows][scale] /
                                             values["close"][rows][scale], np),
                    "enter_to_quoted_spread": quantiles(theta_enter[rows][scale] / spread[scale], np),
                    "release_to_quoted_spread": quantiles(theta_release[rows][scale] / spread[scale], np)},
                "strict_consecutive_activity_flips": binary_flips,
                "threshold_update_required_activity_flips": threshold_only,
                "activity_flips_with_atr_clock_update": atr_updated,
                "strict_consecutive_threeclass_state_changes": state_flips,
                "threeclass_state_changes_with_atr_clock_update": state_flips_at_atr_update,
                "strict_consecutive_known_up_down_flips": side_flips,
                "known_up_down_flips_with_atr_clock_change": side_flips_atr_clock_changed,
                "known_up_down_flips_with_atr_value_change": side_flips_atr_value_changed,
                "known_up_down_flips_without_atr_value_change":
                    side_flips - side_flips_atr_value_changed,
                "source_rows": int(shi - slo),
                "source_complete_after_reset": int(tape["complete"][slo:shi].sum()),
                "source_atr_unavailable": int((theta_enter[slo:shi] == 0).sum()),
                "source_unknown_gap_successors": int(unknown_gap[slo:shi].sum()),
                "source_known_closure_successors": int(sum(
                    row in known for row in range(max(slo, 1), shi)
                    if axis[row] - axis[row - 1] != MINUTE_NS)),
                "source_activity_unknown": int((tape["activity"][slo:shi] == UNKNOWN).sum()),
                "source_direction_evidence_without_activity": int(((tape["activity"][slo:shi] == UNKNOWN)
                    & (tape["direction"][slo:shi] >= 0)).sum()),
                "source_direction_evidence_before_complete_window": int((~tape["complete"][slo:shi]
                    & (tape["direction"][slo:shi] >= 0)).sum()),
                "source_ambiguous_step": int(tape["ambiguous_step"][slo:shi].sum()),
                "source_verified_releases": int(tape["release"][slo:shi].sum()),
                "source_quiet_unknown_resolutions": int(tape["quiet_resolution"][slo:shi].sum()),
                "episodes_strict_trailing_path": episode_report(axis[:prefix], tape["state"],
                    tape["path_observed"], start, stop, lead, np),
                "old_activity_parity": {factor: {"complete_activity": int(original_complete.sum()),
                    "flat": int(flat.sum()), "nonflat": int((original_complete & ~flat).sum())}
                    for factor, flat in (("0.5", flat05), ("1.0", flat10))}}
            need(per["state_observed"] + per["active_with_unknown_side_up_or_down"]
                 + per["activity_unknown"] == per["exact_native_validity"], "MASK_PARTITION")
            monthly[str(start)][tf] = per
    old.freeze(binding, plan, snapshots)
    for item in plan["inputs"].values():
        need(old.ref(item["path"]) == item, "BOOKEND_INPUT")
    return {
        "schema_version": STAGE + "_result_v1", "plan": binding["plan"],
        "policy": binding["policy"], "claim": claim, "source_head": binding["source_head"],
        "report_parents": len(issues), "report_cutoff_ns": cutoff,
        "lead_minutes": plan["lead_minutes"], "month_edges_ns": plan["month_edges_ns"],
        "monthly": monthly,
        "saved_series_reused": True, "raw_m1_reads": 1, "alternative_target_builds": 0,
        "model_calls": 0, "optimizer_steps": 0, "test_data_used": False,
        "later_aggregates_computed": False, "target_selected": False,
        "limits": ["TRAIN-only target semantics; no fit, generalization or economic edge",
            "A complete W-bar window since unknown source gap or ATR loss is required for activity",
            "Direction proof before complete activity is reported but never promoted to three-class state",
            "OHLC wide-bar endpoint may prove ending side but not intraminute event order",
            "Known UP/DOWN flip counts describe endpoint labels; strict path episodes censor uncertain order",
            "Strict trailing-W path mask censors W observed bars after ambiguity and can over-censor",
            "Month edge, unknown label, partial side and absent M1 minute censor episodes",
            "Confirmed starts require consecutive prior known opposite or FLAT state",
            "Survival bounds are descriptive episode counts, not independent statistical confidence intervals",
            "ATR activity attribution is a one-step counterfactual, not market causality",
            "UP/DOWN ATR update coincidence is descriptive; stateful old-threshold side counterfactual is unmeasured",
            "Overlapping M5 issues are not independent episodes",
            "Quoted spread is historical and not executable cost",
            "No parity claim for post-release 1.0 directional side because this candidate resets its anchors"]}


def main():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
    checks = synthetic_checks(np, target)
    if len(sys.argv) == 2 and sys.argv[1] == "--help":
        print(json.dumps({"stage": STAGE, "scope": "TRAIN-only semantic audit; no data run",
                          **checks}, sort_keys=True))
        return
    need(len(sys.argv) == 1 and Path.cwd() == REPO, "CLI")
    from gx1.scripts import audit_dynamic_mtf_state_target_sensitivity_v1 as old
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    need(ROOT.is_dir() and not any(p.is_symlink() for p in (ROOT, *ROOT.parents)), "RUN_ROOT")
    binding = old.read(old.ref(ROOT / "EXECUTION_BINDING.json"))
    need(set(binding) == {"plan", "policy", "worker", "source_head"}
         and binding["plan"]["path"] == str(PLAN_PATH)
         and binding["policy"]["path"] == str(REPO / "NEXT_RUN_POLICY.json")
         and binding["worker"] == old.ref(__file__), "BINDING")
    plan = old.read(binding["plan"])
    need(plan["schema_version"] == STAGE + "_plan_v1"
         and plan["stage"] == STAGE and plan["run_root"] == str(ROOT)
         and plan["worker"] == binding["worker"] and plan["budget"] == BUDGET
         and plan["report_parents"] == 29849
         and plan["report_cutoff_ns"] == 1727740800000000000
         and plan["month_edges_ns"] == [1714521600000000000, 1717200000000000000,
              1719792000000000000, 1722470400000000000, 1725148800000000000,
              1727740800000000000]
         and plan["lead_minutes"] == [5, 15, 60, 240, 1440]
         and plan["enter_atr"] == 1.0 and plan["release_atr"] == 0.5
         and plan["existing_contract_sha256"] ==
             target.dynamic_mtf_state_target_contract()["contract_sha256"]
         and set(plan["inputs"]) == {"prior_result", "prior_series", "m1", "m1_manifest",
             "calendar_source_reference", "closure_authority", "exact_schedule",
             "published_calendar", "semantic_result", "semantic_terminal",
             "transition_result", "transition_terminal"}
         and all("TEST" not in Path(x["path"]).parts for x in plan["inputs"].values()),
         "PLAN")
    policy = old.read(binding["policy"])
    gate = policy[STAGE]
    need(policy["training_enabled"] is False and gate["plan"] == binding["plan"]
         and gate["execution_authorized"] is True and gate["data_reads_authorized"] is True
         and gate["target_builds_authorized"] == gate["model_calls_authorized"] == 0
         and gate["claimed_invocations"] == gate["completed_invocations"] == 0
         and gate["max_invocations"] == 1 and gate["test_authorized"] is False
         and gate["later_aggregate_authorized"] is False, "ADMISSION")
    need(all(not (ROOT / name).exists() and not (ROOT / name).is_symlink()
             for name in ("CLAIM.json", "RESULT.json", "TERMINAL.json")), "CONSUMED")
    limits = require_capped_cpu_audit_execution()
    limits["lock"] = require_capped_lock_ancestry()
    need(all(limits[k] == BUDGET[k] for k in
             ("memory_max", "swap", "pids", "numerical_threads")), "CAPS")
    snapshots = {x["path"]: old.regular(x["path"]) for x in
                 (*plan["inputs"].values(), *plan["source_owners"].values(),
                  binding["plan"], binding["worker"], binding["policy"])}
    old.freeze(binding, plan, snapshots)
    claim = publish("CLAIM.json", {
        "schema_version": STAGE + "_claim_v1", "plan": binding["plan"],
        "policy": binding["policy"], "worker": binding["worker"],
        "source_head": binding["source_head"], "limits": limits,
        "target_builds": 0, "model_calls": 0,
        "scope": "May-September 2024 TRAIN hysteresis episode semantics only"}, old)
    started, before = time.monotonic(), old.stats()

    def deadline(*_):
        raise TimeoutError("HYSTERESIS_SEMANTICS_1800_SECOND_DEADLINE")

    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.alarm(BUDGET["max_wall_seconds"])
    try:
        result = audit(plan, binding, claim, snapshots, old, target, np)
        result["elapsed_seconds"] = time.monotonic() - started
        result["io_before"], result["io_after"] = before, old.stats()
        output = publish("RESULT.json", result, old)
        publish("TERMINAL.json", {
            "schema_version": STAGE + "_terminal_v1", "status": "COMPLETE", "exit_code": 0,
            "claim": claim, "result": output, "source_head": binding["source_head"],
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": old.stats()["peak_rss_bytes"], "test_data_used": False}, old)
    except BaseException as exc:
        publish("TERMINAL.json", {
            "schema_version": STAGE + "_terminal_v1", "status": "FAILED_CONSUMED", "exit_code": 1,
            "claim": claim, "source_head": binding["source_head"],
            "error": type(exc).__name__ + ":" + str(exc)[:500],
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": old.stats()["peak_rss_bytes"], "test_data_used": False}, old)
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
