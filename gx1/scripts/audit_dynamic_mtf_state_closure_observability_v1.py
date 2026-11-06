"""One-use TRAIN source diagnostic for partial activity and closure observability."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import time

REPO = Path("/home/andre2/src/GX1_CURRENT")
STAGE = "dynamic_mtf_state_closure_observability_20261106_001"
PLAN_PATH = REPO / "configs/research/DYNAMIC_MTF_STATE_CLOSURE_OBSERVABILITY_20261106_001.json"
ROOT = Path("/home/andre2/GX1_RUNS/DYNAMIC_MTF_STATE_CLOSURE_OBSERVABILITY_20261106_001")
BUDGET = {"memory_max": 4294967296, "swap": 536870912, "pids": 64,
          "numerical_threads": 1, "max_wall_seconds": 1800,
          "max_invocations": 1, "max_json_bytes": 16777216}
MINUTE_NS = 60000000000


def need(ok, key):
    if not ok:
        raise RuntimeError("CLOSURE_OBSERVABILITY_" + key)


def early_active_proof(highs, lows, threshold, unknown_gap, window, np):
    """Only post-reset observed OHLC range; never infer FLAT before full W."""
    n = len(highs)
    need(n > 0 and window > 0 and all(len(x) == n for x in
         (lows, threshold, unknown_gap)) and np.isfinite(highs).all()
         and np.isfinite(lows).all() and np.isfinite(threshold).all()
         and np.all(highs >= lows) and np.all(threshold >= 0),
         "EARLY_PROOF_DOMAIN")
    proof = np.zeros(n, bool)
    complete = np.zeros(n, bool)
    count, lo, hi = 0, float("inf"), float("-inf")
    for row in range(n):
        if unknown_gap[row] or threshold[row] <= 0:
            count, lo, hi = 0, float("inf"), float("-inf")
        if threshold[row] <= 0:
            continue
        count += 1
        lo, hi = min(lo, float(lows[row])), max(hi, float(highs[row]))
        complete[row] = count >= window
        proof[row] = count < window and hi - lo >= float(threshold[row])
    return proof, complete


def closure_episode_report(axis, state, path, known_successor, start, stop, horizon, np, hys):
    """Exact wall-UTC horizon; cross only declared closures, never fill missing endpoints."""
    lo, hi = np.searchsorted(axis, [start, stop])
    need(hi > lo and horizon > 0 and all(len(x) == len(axis) for x in
         (state, path, known_successor)), "EPISODE_DOMAIN")
    clock = axis[lo:hi]
    code = np.where(path[lo:hi], state[lo:hi], hys.UNKNOWN)
    adjacent = np.zeros(len(clock), bool)
    adjacent[1:] = (np.diff(clock) == MINUTE_NS) | known_successor[lo + 1:hi]
    out = {}
    for value, name in enumerate(("DOWN", "FLAT", "UP")):
        starts, observed, elapsed, crossed, survived, failed, censored = 0, [], [], 0, 0, 0, 0
        survived_crossed = 0
        segments, left_truncated = 0, 0
        i = 0
        while i < len(code):
            if code[i] != value:
                i += 1
                continue
            j = i
            while j + 1 < len(code) and adjacent[j + 1] and code[j + 1] == value:
                j += 1
            segments += 1
            observed.append(j - i + 1)
            elapsed.append((int(clock[j]) - int(clock[i])) / MINUTE_NS)
            closure_count = int(known_successor[lo + i + 1:lo + j + 1].sum())
            crossed += int(closure_count > 0)
            confirmed_start = i > 0 and adjacent[i] and code[i - 1] >= 0 and code[i - 1] != value
            if not confirmed_start:
                left_truncated += 1
            else:
                starts += 1
                wanted = int(clock[i]) + horizon * MINUTE_NS
                exact = int(np.searchsorted(clock, wanted))
                if (wanted < stop and exact < len(clock)
                        and int(clock[exact]) == wanted and exact <= j):
                    survived += 1
                    survived_crossed += int(known_successor[lo + i + 1:lo + exact + 1].any())
                elif (j + 1 < len(code) and adjacent[j + 1]
                      and code[j + 1] >= 0 and code[j + 1] != value
                      and int(clock[j + 1]) <= wanted):
                    failed += 1
                else:
                    censored += 1
            i = j + 1
        need(starts == survived + failed + censored, "EPISODE_PARTITION")
        out[name] = {
            "segments": segments, "confirmed_starts": starts,
            "left_truncated": left_truncated, "segments_crossing_known_closure": crossed,
            "observed_m1_minutes_lower_bound": hys.quantiles(observed, np),
            "wall_elapsed_first_to_last_validity_minutes": hys.quantiles(elapsed, np),
            "exact_plus_own_tf_utc": {
                "horizon_minutes": horizon, "denominator_confirmed_starts": starts,
                "survived_proven": survived, "survived_with_known_closure": survived_crossed,
                "ended_by_horizon_confirmed": failed, "censored_before_horizon": censored,
                "survival_lower_bound": survived / starts if starts else None,
                "survival_upper_bound": (survived + censored) / starts if starts else None}}
    return out


def strict_clock_max(axis, start, stop, np):
    lo, hi = np.searchsorted(axis, [start, stop])
    need(hi > lo, "STRICT_CLOCK")
    breaks = np.r_[lo, np.flatnonzero(np.diff(axis[lo:hi]) != MINUTE_NS) + lo + 1, hi]
    return int(np.diff(breaks).max())


def synthetic_checks(np, hys):
    theta = np.ones(5)
    proof, complete = early_active_proof(
        np.array([100.2, 101.2, 101.2, 100.2, 100.3]),
        np.array([99.8, 100.0, 100.0, 100.0, 100.0]), theta,
        np.array([False, False, False, True, False]), 3, np)
    need(np.array_equal(proof, [False, True, False, False, False])
         and np.array_equal(complete, [False, False, True, False, False]),
         "SYNTHETIC_EARLY_PROOF")
    changed, _ = early_active_proof(
        np.array([100.2, 101.2]), np.array([99.8, 100.0]),
        np.array([1.0, 2.0]), np.zeros(2, bool), 3, np)
    need(not changed.any(), "SYNTHETIC_CURRENT_THRESHOLD")
    axis = np.array([0, 1, 2, 10, 11, 12], np.int64) * MINUTE_NS
    state = np.array([1, 2, 2, 2, 2, 1], np.int8)
    path = np.ones(6, bool)
    known = np.array([False, False, False, True, False, False])
    a = closure_episode_report(axis, state, path, known, 0, 13 * MINUTE_NS, 10, np, hys)
    z = a["UP"]["exact_plus_own_tf_utc"]
    need(z["denominator_confirmed_starts"] == z["survived_proven"] ==
         z["survived_with_known_closure"] == 1, "SYNTHETIC_CLOSURE_EXACT")
    a = closure_episode_report(axis, state, path, known, 0, 13 * MINUTE_NS, 8, np, hys)
    z = a["UP"]["exact_plus_own_tf_utc"]
    need(z["survived_proven"] == z["ended_by_horizon_confirmed"] == 0
         and z["censored_before_horizon"] == 1, "SYNTHETIC_MISSING_ENDPOINT")
    a = closure_episode_report(axis, state, path, np.zeros(6, bool),
                               0, 13 * MINUTE_NS, 10, np, hys)
    z = a["UP"]["exact_plus_own_tf_utc"]
    need(z["survived_proven"] == z["ended_by_horizon_confirmed"] == 0
         and z["censored_before_horizon"] == 1, "SYNTHETIC_UNKNOWN_GAP")
    a = closure_episode_report(axis, np.array([1, 2, 2, 0, 0, 1], np.int8),
                               path, known, 0, 13 * MINUTE_NS, 10, np, hys)
    z = a["UP"]["exact_plus_own_tf_utc"]
    need(z["ended_by_horizon_confirmed"] == 1, "SYNTHETIC_CONFIRMED_END")
    a = closure_episode_report(axis, state, path, known, 0, 10 * MINUTE_NS, 10, np, hys)
    need(a["UP"]["exact_plus_own_tf_utc"]["censored_before_horizon"] == 1,
         "SYNTHETIC_MONTH_EDGE")
    need(strict_clock_max(axis, 0, 13 * MINUTE_NS, np) == 3,
         "SYNTHETIC_STRICT_MAX")
    caught = False
    try:
        need(False, "EPISODE_PARTITION")
    except RuntimeError as exc:
        caught = str(exc) == "CLOSURE_OBSERVABILITY_EPISODE_PARTITION"
    need(caught, "SYNTHETIC_ERROR_KEY")
    return {"synthetic_success_key": "CLOSURE_OBSERVABILITY_SYNTHETIC_OK",
            "synthetic_error_key": "CLOSURE_OBSERVABILITY_EPISODE_PARTITION"}


def publish(name, value, old):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    need(len(raw) <= BUDGET["max_json_bytes"], "JSON_BUDGET")
    _publish_bytes(ROOT / name, raw)
    (ROOT / name).chmod(0o444)
    return old.ref(ROOT / name)


def audit(plan, binding, claim, snapshots, old, target, hys, np):
    import pandas as pd
    import pyarrow.parquet as pq

    for item in plan["inputs"].values():
        need(old.ref(item["path"]) == item, "INPUT_HASH")
    prior_plan = old.read(plan["inputs"]["hysteresis_plan"])
    prior_result = old.read(plan["inputs"]["hysteresis_result"])
    prior_terminal = old.read(plan["inputs"]["hysteresis_terminal"])
    need(prior_plan["inputs"] == plan["inherited_inputs"]
         and prior_plan["worker"] == plan["source_owners"]["hysteresis_worker"]
         and prior_result["plan"] == plan["inputs"]["hysteresis_plan"]
         and prior_result["claim"] == prior_terminal["claim"]
         and prior_terminal["status"] == "COMPLETE"
         and prior_terminal["result"] == plan["inputs"]["hysteresis_result"]
         and prior_result["target_selected"] is False
         and prior_result["model_calls"] == 0
         and prior_result["test_data_used"] is False, "PRIOR_HYSTERESIS")
    for item in plan["inherited_inputs"].values():
        need(old.ref(item["path"]) == item, "INHERITED_INPUT_HASH")
    prior = old.read(plan["inherited_inputs"]["prior_result"])
    need(prior["series"] == plan["inherited_inputs"]["prior_series"]
         and prior["target_sources"]["m1"] == plan["inherited_inputs"]["m1"]
         and prior["test_data_used"] is False, "PRIOR_SOURCE")
    for key, name in (("source_reference", "calendar_source_reference"),
                      ("closure_authority", "closure_authority"),
                      ("exact_schedule", "exact_schedule"),
                      ("published_calendar", "published_calendar")):
        need(prior["calendar_binding"][key] == plan["inherited_inputs"][name],
             "CALENDAR_BINDING")
    manifest = old.read(plan["inherited_inputs"]["prior_series"])
    for item in manifest["arrays"].values():
        snapshots[item["path"]] = old.regular(item["path"])
    base = target.load_dynamic_mtf_state_series(
        plan["inherited_inputs"]["prior_series"],
        expected_source_refs=prior["target_sources"],
        expected_cutoff_ns=prior["fold"]["series_cutoff_ns"])
    arrays = base.arrays
    axis = arrays["validity_ns"]
    cutoff = plan["report_cutoff_ns"]
    prefix = int(np.searchsorted(axis, cutoff))
    issues = arrays["issue_ns"][:plan["report_parents"]]
    parents = arrays["parent_rows"][:plan["report_parents"]]
    need(np.array_equal(parents, np.arange(len(parents)))
         and int(np.searchsorted(arrays["issue_ns"], cutoff)) == len(issues)
         and axis[prefix - 1] < cutoff <= axis[prefix], "REPORT_CLOCK")
    proof = prior["clock_slice_proof"]
    source = pq.read_table(plan["inherited_inputs"]["m1"]["path"],
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
         and np.all(values["high"] >= np.maximum(values["open"], values["close"]))
         and np.all(values["low"] <= np.minimum(values["open"], values["close"]))
         and np.all(values["ask_close"] > values["bid_close"]), "SOURCE_OHLC")
    times = target._clock(source, target.EXIT_DECISION_BAR_SECONDS)
    known, closure_binding = target._known_closures(
        times, plan["inherited_inputs"]["m1"],
        old.read(plan["inherited_inputs"]["closure_authority"]),
        old.read(plan["inherited_inputs"]["exact_schedule"]))
    known_successor = np.zeros(prefix, bool)
    unknown_gap = np.zeros(prefix, bool)
    for row in np.flatnonzero(np.diff(axis[:prefix]) != MINUTE_NS) + 1:
        known_successor[row] = int(row) in known
        unknown_gap[row] = int(row) not in known
    need(closure_binding["authority_sha256"] ==
         old.read(plan["inherited_inputs"]["closure_authority"])["artifact_sha256"],
         "CLOSURE_HASH")
    monthly = {str(start): {} for start in plan["month_edges_ns"][:-1]}
    for col, tf in enumerate(target.TIMEFRAMES):
        theta = 2.0 * arrays["threshold"][:prefix, col]
        tape = hys.candidate_path(axis[:prefix],
            *(values[k][:prefix] for k in ("open", "high", "low", "close")),
            theta, arrays["local_range"][:prefix, col], unknown_gap,
            target.LOCAL_WINDOW_M1_BARS[tf], target._step_direction, np)
        partial, complete = early_active_proof(
            values["high"][:prefix], values["low"][:prefix], theta, unknown_gap,
            target.LOCAL_WINDOW_M1_BARS[tf], np)
        need(np.array_equal(complete, tape["complete"])
             and not np.any(partial & complete), "PARTIAL_SUPPORT")
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
            saved = prior_result["monthly"][str(start)][tf]
            expected = {
                "issues": ihi - ilo, "mature": int(mature.sum()),
                "exact_native_validity": int(present.sum()),
                "query_unavailable": int((mature & ~present).sum()),
                "immature": int((~mature).sum()),
                "activity_known": int((tape["activity"][rows] >= 0).sum()),
                "activity_unknown": int((tape["activity"][rows] == hys.UNKNOWN).sum()),
                "active_with_unknown_side_up_or_down": int(
                    ((tape["activity"][rows] == hys.ACTIVE) &
                     (tape["state"][rows] == hys.UNKNOWN)).sum()),
                "state_observed": int((tape["state"][rows] >= 0).sum()),
                "strict_trailing_window_path_observed": int(tape["path_observed"][rows].sum()),
            }
            need(all(saved[k] == v for k, v in expected.items())
                 and saved["state_counts"] == {
                     name: int(np.count_nonzero(tape["state"][rows] == code))
                     for code, name in enumerate(target.STATE_NAMES)}
                 and saved["episodes_strict_trailing_path"] ==
                     hys.episode_report(axis[:prefix], tape["state"], tape["path_observed"],
                                        start, stop, lead, np), "PRIOR_PARITY")
            closure = closure_episode_report(axis[:prefix], tape["state"],
                tape["path_observed"], known_successor, start, stop, lead, np, hys)
            gap_rows = [row for row in range(max(slo, 1), shi)
                        if axis[row] - axis[row - 1] != MINUTE_NS]
            known_lengths, unknown_lengths = [], []
            for row in gap_rows:
                missing = int((axis[row] - axis[row - 1]) // MINUTE_NS - 1)
                (known_lengths if known_successor[row] else unknown_lengths).append(missing)
            need(saved["source_rows"] == shi - slo
                 and saved["source_known_closure_successors"] == len(known_lengths)
                 and saved["source_unknown_gap_successors"] == len(unknown_lengths),
                 "PRIOR_SOURCE_GAP_PARITY")
            per = {
                "issues": int(ihi - ilo), "exact_native_validity": int(present.sum()),
                "endpoint_state_counts": saved["state_counts"],
                "endpoint_activity_unknown": saved["activity_unknown"],
                "endpoint_partial_up_or_down": saved["active_with_unknown_side_up_or_down"],
                "source_rows": int(shi - slo),
                "source_known_closure_successors": len(known_lengths),
                "source_unknown_gap_successors": len(unknown_lengths),
                "known_closure_missing_minutes": hys.quantiles(known_lengths, np),
                "unknown_gap_missing_minutes": hys.quantiles(unknown_lengths, np),
                "strict_max_consecutive_m1_validities": strict_clock_max(axis[:prefix], start, stop, np),
                "strict_horizon_structurally_impossible":
                    strict_clock_max(axis[:prefix], start, stop, np) <= lead,
                "early_active_proof_source": int(partial[slo:shi].sum()),
                "early_active_proof_exact": int(partial[rows].sum()),
                "early_active_proof_exact_with_direction_evidence":
                    int((partial[rows] & (tape["direction"][rows] >= 0)).sum()),
                "early_active_proof_exact_without_direction_evidence":
                    int((partial[rows] & (tape["direction"][rows] < 0)).sum()),
                "early_active_proof_label_promotion": False,
                "strict_episode_parity": True,
                "closure_aware_episodes": closure,
            }
            need(per["early_active_proof_exact"] ==
                 per["early_active_proof_exact_with_direction_evidence"] +
                 per["early_active_proof_exact_without_direction_evidence"],
                 "EARLY_PARTITION")
            monthly[str(start)][tf] = per
    old.freeze(binding, plan, snapshots)
    for item in (*plan["inputs"].values(), *plan["inherited_inputs"].values()):
        need(old.ref(item["path"]) == item, "BOOKEND_INPUT")
    return {
        "schema_version": STAGE + "_result_v1", "plan": binding["plan"],
        "policy": binding["policy"], "claim": claim, "source_head": binding["source_head"],
        "report_parents": len(issues), "report_cutoff_ns": cutoff,
        "month_edges_ns": plan["month_edges_ns"], "lead_minutes": plan["lead_minutes"],
        "monthly": monthly,
        "source_max_strict_m1_validities_may_sep": strict_clock_max(
            axis[:prefix], plan["month_edges_ns"][0], cutoff, np),
        "saved_hysteresis_parity": True, "raw_m1_reads": 1,
        "alternative_target_builds": 0, "model_calls": 0, "optimizer_steps": 0,
        "test_data_used": False, "later_aggregates_computed": False,
        "target_selected": False,
        "limits": [
            "TRAIN-only source observability; no model, target lock, generalization or economy",
            "Early post-reset high-low proof is separate activity evidence, never a promoted label",
            "A sub-threshold early range does not prove FLAT; unknown remains unknown",
            "Direction evidence is reported only, not independently re-certified at current threshold",
            "Closure-aware episodes bridge only authority-known market closures and leave missing endpoint minutes unlabeled",
            "Unknown source gaps, ambiguous OHLC, month edges and missing endpoints censor",
            "Known closure can preserve observed endpoint state but does not imply known intraminute order",
            "Exact future UTC leads are retained; no nearest-row or next-open substitution",
            "Survival bounds are descriptive episode counts, not independent confidence intervals",
            "Overlapping M5 forecast issues are not independent episodes",
        ]}


def main():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
    from gx1.scripts import audit_dynamic_mtf_state_hysteresis_semantics_v1 as hys
    checks = synthetic_checks(np, hys)
    if len(sys.argv) == 2 and sys.argv[1] == "--help":
        print(json.dumps({"stage": STAGE, "scope": "TRAIN-only source diagnostic; no data run",
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
         and set(plan["inputs"]) == {"hysteresis_plan", "hysteresis_result",
                                    "hysteresis_terminal"}
         and set(plan["inherited_inputs"]) == {"prior_result", "prior_series", "m1",
             "m1_manifest", "calendar_source_reference", "closure_authority",
             "exact_schedule", "published_calendar", "semantic_result",
             "semantic_terminal", "transition_result", "transition_terminal"}
         and all("TEST" not in Path(x["path"]).parts for x in
                 (*plan["inputs"].values(), *plan["inherited_inputs"].values())),
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
                 (*plan["inputs"].values(), *plan["inherited_inputs"].values(),
                  *plan["source_owners"].values(), binding["plan"],
                  binding["worker"], binding["policy"])}
    old.freeze(binding, plan, snapshots)
    claim = publish("CLAIM.json", {
        "schema_version": STAGE + "_claim_v1", "plan": binding["plan"],
        "policy": binding["policy"], "worker": binding["worker"],
        "source_head": binding["source_head"], "limits": limits,
        "target_builds": 0, "model_calls": 0,
        "scope": "May-September 2024 TRAIN source closure observability only"}, old)
    started, before = time.monotonic(), old.stats()

    def deadline(*_):
        raise TimeoutError("CLOSURE_OBSERVABILITY_1800_SECOND_DEADLINE")

    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.alarm(BUDGET["max_wall_seconds"])
    try:
        result = audit(plan, binding, claim, snapshots, old, target, hys, np)
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
