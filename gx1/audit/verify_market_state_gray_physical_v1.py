"""One-use TRAIN-only physical audit of the proposed market-state gray band.

No target is selected. The two saved candidates select three fixed, fully
observed local-range cohorts; raw M1 OHLC measures net movement and close-only
path diagnostics. Existing closed-TF ATR and quoted-spread bindings provide
scale. Nothing here fits a model, reads TEST, or defines a trading rule.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

from gx1.audit import build_full_train_market_state_endpoint_candidate_v1 as base

REPO = base.REPO
STAGE = "market_state_gray_physical_20261107_001"
PLAN_SCHEMA = "gx1_market_state_gray_physical_plan_v1"
CLAIM_SCHEMA = "gx1_market_state_gray_physical_claim_v1"
SOURCE_ROW_OFFSET = 5193205
COHORTS = ("below_0_5", "gray_0_5_to_1_0", "at_least_1_0")
SCOPE = {"train_start_utc": base.TRAIN_START,
         "train_end_utc_exclusive": base.TRAIN_END,
         "m1_rows": base.M1_ROWS, "m5_issues": 118706,
         "lead_minutes": list(base.LEADS), "gray_cells": 273200,
         "test_data_used": False}
LIMITS = {"memory_max": "4G", "swap_max": "512M", "wall_seconds": 1800,
          "execution_class": "audit", "model_forwards": 0, "optimizer_steps": 0}
FULL_ROOT = Path("/home/andre2/GX1_RUNS/FULL_TRAIN_MARKET_STATE_ENDPOINT_CANDIDATE_20261107_001")
SINGLE_ROOT = Path("/home/andre2/GX1_RUNS/SINGLE_THRESHOLD_MARKET_STATE_CANDIDATE_20261107_001")
INPUTS = {
    "full_plan": {"path": str(REPO / "configs/research/FULL_TRAIN_MARKET_STATE_ENDPOINT_CANDIDATE_20261107_001.json"),
                  "sha256": "c3c789f4e7ed3edba2880bfb9e51decc9efad779324351bbd19ac86d2a607ce2"},
    "full_result": {"path": str(FULL_ROOT / "RESULT.json"),
                    "sha256": "3a508ce45755c8db2ad8c078d56a6afef8b88a6af3e0a0b159ca388def38ebc5"},
    "full_masks": {"path": str(FULL_ROOT / "ROW_MASKS.npz"),
                   "sha256": "ad0e7d56c58b463bdd3d61fe6b2fcf2b206afa56e899dfb36a850e63c5ec3020"},
    "single_plan": {"path": str(REPO / "configs/research/SINGLE_THRESHOLD_MARKET_STATE_CANDIDATE_20261107_001.json"),
                    "sha256": "9de70e39d0618ea579c4ccd95a8f7ba3e174511a8052befb3728d9f810d521a9"},
    "single_result": {"path": str(SINGLE_ROOT / "RESULT.json"),
                      "sha256": "1bad8a8a88d61a67ee2487e387c246da06f4f4ccf319e7edd8ce35d39d13eba1"},
    "single_masks": {"path": str(SINGLE_ROOT / "ROW_MASKS_AND_RATIOS.npz"),
                     "sha256": "35257429efa4ec553293e1b10bec0993c3c494f04e7f5be72d55877112e590e5"},
    "single_terminal": {"path": str(SINGLE_ROOT / "TERMINAL.json"),
                        "sha256": "9ce11435f5be8174dd629c5f6c564f8a945cc432fb037afed14de47222798295"},
}
SOURCE_PATHS = {
    "worker": "gx1/audit/verify_market_state_gray_physical_v1.py",
    "full_train_loader": "gx1/audit/build_full_train_market_state_endpoint_candidate_v1.py",
    "single_worker": "gx1/audit/verify_single_threshold_market_state_candidate_v1.py",
    "target": "gx1/contracts/entry_dynamic_mtf_state_targets_v1.py",
    "capped_owner": "gx1/contracts/gx1_capped_execution_v1.py",
    "publication_owner": "gx1/contracts/immutable_event_authority_v1.py",
}
OLD_KEYS = {"issue_ns", "m5_source_row", "lead_minutes", "allowed_set", "m1_source_row"}
NEW_KEYS = OLD_KEYS | {"complete_range_atr", "complete_range_atr_valid",
                       "atr_to_quoted_spread", "atr_to_quoted_spread_valid"}


def need(ok: object, name: str) -> None:
    if not bool(ok):
        raise RuntimeError("GRAY_PHYSICAL_" + name)


def ref(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def git_state() -> tuple[str, str, str]:
    def command(*args: str) -> str:
        return subprocess.check_output(("git", *args), cwd=REPO, text=True).strip()
    return command("rev-parse", "HEAD"), command("branch", "--show-current"), command("status", "--porcelain")


def check_plan(plan: dict, plan_ref: dict) -> tuple[dict, dict]:
    need(set(plan) == {"schema_version", "stage", "status", "inputs", "source_owners",
                       "scope", "limits", "output_dir", "test_data_used"}
         and plan["schema_version"] == PLAN_SCHEMA and plan["stage"] == STAGE
         and plan["status"] == "PREPARED_NOT_ADMITTED"
         and plan["inputs"] == INPUTS and plan["scope"] == SCOPE
         and plan["limits"] == LIMITS and plan["test_data_used"] is False,
         "PLAN")
    output = Path(plan["output_dir"])
    need(output == Path("/home/andre2/GX1_RUNS") / STAGE.upper()
         and len(plan["source_owners"]) == len(SOURCE_PATHS)
         and set(plan["source_owners"]) == set(SOURCE_PATHS), "OUTPUT_AND_OWNERS")
    for name, relative in SOURCE_PATHS.items():
        owner = plan["source_owners"][name]
        need(owner["path"] == str(REPO / relative), "OWNER_PATH_" + name)
        base.checked(owner)
    base.checked(plan_ref)
    full_plan = base.read_json(INPUTS["full_plan"])
    base.check_plan(full_plan, INPUTS["full_plan"], FULL_ROOT)
    single_plan = base.read_json(INPUTS["single_plan"])
    need(single_plan["schema_version"] == "gx1_single_threshold_market_state_candidate_plan_v1"
         and single_plan["old_full"]["result"] == INPUTS["full_result"]
         and single_plan["old_full"]["row_masks"] == INPUTS["full_masks"]
         and all(single_plan[k] == full_plan[k] for k in ("m1", "m5", "closure"))
         and single_plan["factor"] == 1.0
         and single_plan["lead_minutes"] == list(base.LEADS)
         and single_plan["train_start_utc"] == base.TRAIN_START
         and single_plan["train_end_utc_exclusive"] == base.TRAIN_END
         and single_plan["test_data_used"] is False,
         "SINGLE_PLAN")
    for owner in single_plan["source_owners"].values():
        base.checked(owner)
    full_result = base.read_json(INPUTS["full_result"])
    single_result = base.read_json(INPUTS["single_result"])
    terminal = base.read_json(INPUTS["single_terminal"])
    need(full_result["decision"] == "PASS_FULL_TRAIN_CANDIDATE_SOURCE_ONLY"
         and full_result["plan"] == INPUTS["full_plan"]
         and full_result["row_masks"] == INPUTS["full_masks"]
         and single_result["decision"] == "PASS_SINGLE_THRESHOLD_CANDIDATE_SOURCE_ONLY"
         and single_result["plan"] == INPUTS["single_plan"]
         and single_result["row_masks_and_ratios"] == INPUTS["single_masks"]
         and single_result["source_refs"]["old_full"]["result"] == INPUTS["full_result"]
         and single_result["factor"] == 1.0
         and terminal["status"] == "COMPLETE" and terminal["exit_code"] == 0
         and terminal["result"] == INPUTS["single_result"]
         and terminal["row_masks_and_ratios"] == INPUTS["single_masks"]
         and all(x["m1_rows"] == base.M1_ROWS and x["m5_issues"] == SCOPE["m5_issues"]
                 and x["model_calls"] == x["optimizer_steps"] == 0
                 and x["test_data_used"] is False and x["target_locked"] is False
                 for x in (full_result, single_result))
         and terminal["test_data_used"] is False and terminal["target_locked"] is False,
         "PRIOR_RESULTS")
    return full_plan, single_result


def load_masks(single_result: dict) -> tuple[dict, dict, dict]:
    raw_old, _ = base.checked(INPUTS["full_masks"], max_bytes=64 * 1024 * 1024)
    raw_new, _ = base.checked(INPUTS["single_masks"], max_bytes=64 * 1024 * 1024)
    need(raw_old is not None and raw_new is not None, "MASK_BYTES")
    with np.load(io.BytesIO(raw_old), allow_pickle=False) as z:
        need(set(z.files) == OLD_KEYS, "OLD_MASK_KEYS")
        old = {key: np.array(z[key]) for key in OLD_KEYS}
    with np.load(io.BytesIO(raw_new), allow_pickle=False) as z:
        need(set(z.files) == NEW_KEYS, "NEW_MASK_KEYS")
        new = {key: np.array(z[key]) for key in NEW_KEYS}
    need(old["issue_ns"].shape == (SCOPE["m5_issues"],)
         and old["allowed_set"].shape == new["allowed_set"].shape ==
             (SCOPE["m5_issues"], len(base.LEADS))
         and all(np.array_equal(old[k], new[k]) for k in OLD_KEYS - {"allowed_set"})
         and np.array_equal(old["lead_minutes"], np.asarray(base.LEADS, np.int16))
         and np.all(np.diff(old["issue_ns"]) > 0)
         and all(new[k].shape == old["allowed_set"].shape for k in
                 NEW_KEYS - OLD_KEYS), "MASK_AXES")
    ratio = new["complete_range_atr"]
    valid = new["complete_range_atr_valid"]
    source = new["m1_source_row"]
    need(np.isfinite(ratio).all() and np.all(ratio >= 0)
         and np.isfinite(new["atr_to_quoted_spread"]).all()
         and np.all(new["atr_to_quoted_spread"] >= 0)
         and np.all(source[valid] >= SOURCE_ROW_OFFSET)
         and np.all(np.isin(old["allowed_set"], (1, 2, 4, 5, 7)))
         and np.all(np.isin(new["allowed_set"], (1, 2, 4, 5, 7))), "MASK_VALUES")
    cohorts = {
        "below_0_5": valid & (ratio < 0.5),
        "gray_0_5_to_1_0": valid & (ratio >= 0.5) & (ratio < 1.0),
        "at_least_1_0": valid & (ratio >= 1.0),
    }
    gray_by_tf = [sum(single_result["monthly"][month][tf]["comparison_to_old"]
                      ["old_gray_to_flat"] for month in single_result["monthly"])
                  for tf in ("M5", "M15", "H1", "H4", "D1")]
    need(np.array_equal(cohorts["below_0_5"],
                        valid & (old["allowed_set"] == 2) & (new["allowed_set"] == 2))
         and np.array_equal(cohorts["gray_0_5_to_1_0"],
                            (old["allowed_set"] == 7) & (new["allowed_set"] == 2))
         and np.all(np.isin(old["allowed_set"][cohorts["at_least_1_0"]], (1, 4, 5)))
         and np.array_equal(old["allowed_set"][cohorts["at_least_1_0"]],
                            new["allowed_set"][cohorts["at_least_1_0"]])
         and int(cohorts["gray_0_5_to_1_0"].sum()) == SCOPE["gray_cells"]
         and [int(cohorts["gray_0_5_to_1_0"][:, j].sum()) for j in range(5)]
             == gray_by_tf, "COHORTS")
    return old, new, cohorts


def unknown_gap_mask(validity_ns: np.ndarray, known_successor_rows: set[int]) -> np.ndarray:
    gap_rows = np.flatnonzero(np.diff(validity_ns) != base.MINUTE_NS) + 1
    need(all(row in gap_rows for row in known_successor_rows), "KNOWN_CLOSURE_ROW")
    mask = np.zeros(len(validity_ns), bool)
    mask[gap_rows] = ~np.isin(gap_rows, list(known_successor_rows))
    return mask


def require_endpoint_and_window(issues: np.ndarray, rows: np.ndarray,
                                source_rows: np.ndarray, validity_ns: np.ndarray,
                                lead: int, window: int,
                                unknown_gap: np.ndarray) -> np.ndarray:
    starts = rows - window + 1
    need(len(rows) == len(issues) == len(source_rows)
         and np.all((starts >= 0) & (rows < len(validity_ns)))
         and np.array_equal(source_rows, rows + SOURCE_ROW_OFFSET)
         and np.array_equal(validity_ns[rows], issues + lead * base.MINUTE_NS),
         "ENDPOINT")
    prefix = np.r_[0, np.cumsum(unknown_gap.astype(np.int64))]
    # An unknown gap at the first observed bar is allowed after W new bars;
    # an unknown gap strictly inside the W bars invalidates the window.
    need(np.all(prefix[rows + 1] - prefix[starts + 1] == 0), "UNKNOWN_GAP_WINDOW")
    return starts


def raw_window_range(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, unknown_gap: np.ndarray,
                     window: int) -> np.ndarray:
    need(len(open_) == len(high) == len(low) == len(close) == len(unknown_gap)
         and window > 0, "RAW_SHAPE")
    previous_close = np.r_[open_[0], close[:-1]]
    previous_close[unknown_gap] = open_[unknown_gap]
    local_high = np.maximum(high, previous_close)
    local_low = np.minimum(low, previous_close)
    return (pd.Series(local_high).rolling(window, min_periods=1).max()
            - pd.Series(local_low).rolling(window, min_periods=1).min()).to_numpy(np.float64)


def require_ratio_parity(raw_range: np.ndarray, atr: np.ndarray,
                         stored_ratio: np.ndarray) -> None:
    need(np.all(np.isfinite(raw_range)) and np.all(raw_range >= 0)
         and np.all(np.isfinite(atr)) and np.all(atr > 0), "PHYSICAL_SCALE")
    physical = raw_range / atr
    need(np.array_equal(physical.view(np.uint64), stored_ratio.view(np.uint64)),
         "RANGE_ATR_PARITY")


def close_diagnostics(close: np.ndarray, starts: np.ndarray,
                      end_rows: np.ndarray, start_open: np.ndarray,
                      net: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Close-only start-price crossings and terminal run; zero closes ignored."""
    need(len(starts) == len(end_rows) == len(start_open) == len(net)
         and np.all(end_rows - starts + 1 == window), "CLOSE_SHAPE")
    crossings = np.zeros(len(starts), np.int32)
    terminal_run = np.zeros(len(starts), np.int32)
    offsets = np.arange(window, dtype=np.int64)
    for lo in range(0, len(starts), 512):
        hi = min(lo + 512, len(starts))
        positions = starts[lo:hi, None] + offsets[None, :]
        sign = np.sign(close[positions] - start_open[lo:hi, None]).astype(np.int8)
        nonzero = sign != 0
        last_index = np.maximum.accumulate(
            np.where(nonzero, offsets[None, :], -1), axis=1)
        last_sign = np.take_along_axis(sign, np.maximum(last_index, 0), axis=1)
        last_sign[last_index < 0] = 0
        crossings[lo:hi] = np.count_nonzero(
            (last_sign[:, 1:] != last_sign[:, :-1])
            & (last_sign[:, 1:] != 0) & (last_sign[:, :-1] != 0), axis=1)
        final = np.sign(net[lo:hi]).astype(np.int8)
        reverse = sign[:, ::-1]
        opposite = (reverse != 0) & (reverse != final[:, None])
        runs = np.count_nonzero(
            (reverse == final[:, None])
            & ~np.maximum.accumulate(opposite, axis=1), axis=1)
        runs[final == 0] = 0
        terminal_run[lo:hi] = runs
    return crossings, terminal_run


def quantiles(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"p10": None, "median": None, "p90": None}
    return {"p10": float(np.quantile(values, 0.1)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9))}


def summarize(net_atr: np.ndarray, efficiency: np.ndarray,
              net_spreads: np.ndarray, crossings: np.ndarray,
              terminal_run: np.ndarray, wall_minutes: np.ndarray) -> dict:
    need(all(len(x) == len(net_atr) for x in
             (efficiency, net_spreads, crossings, terminal_run, wall_minutes)),
         "SUMMARY_SHAPE")
    valid_efficiency = np.isfinite(efficiency)
    valid_spread = np.isfinite(net_spreads)
    return {"n": len(net_atr),
            "net_down_zero_up": [int((net_atr < 0).sum()),
                                 int((net_atr == 0).sum()),
                                 int((net_atr > 0).sum())],
            "signed_net_atr": quantiles(net_atr),
            "absolute_net_atr": quantiles(np.abs(net_atr)),
            "directional_efficiency_valid": int(valid_efficiency.sum()),
            "absolute_net_over_local_range": quantiles(efficiency[valid_efficiency]),
            "quoted_spread_valid": int(valid_spread.sum()),
            "absolute_net_quoted_spreads": quantiles(np.abs(net_spreads[valid_spread])),
            "start_price_close_crossings": quantiles(crossings),
            "terminal_same_side_nonzero_closes_since_last_opposite": quantiles(terminal_run),
            "window_wall_minutes": quantiles(wall_minutes)}


def read_prepared(plan_ref: dict) -> tuple[dict, dict, dict, dict]:
    plan = base.read_json(plan_ref)
    full_plan, single_result = check_plan(plan, plan_ref)
    old, new, cohorts = load_masks(single_result)
    return plan, full_plan, new, cohorts


def check_claim_and_policy(plan: dict, plan_ref: dict, state: tuple[str, str, str]):
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry,
    )
    need(Path.cwd() == REPO and state[1:] == ("work/gx1-current", ""), "SOURCE_TREE")
    output = Path(plan["output_dir"])
    need(output.is_dir() and not output.is_symlink()
         and {p.name for p in output.iterdir()} == {"CLAIM.json"}, "OUTPUT")
    claim_path = output / "CLAIM.json"
    claim = json.loads(base.checked(ref(claim_path))[0])
    need(set(claim) == {"schema_version", "stage", "plan", "policy", "source_head",
                        "worker", "source_only_audit", "test_data_used"}
         and claim["schema_version"] == CLAIM_SCHEMA
         and claim["stage"] == STAGE and claim["plan"] == plan_ref
         and claim["worker"] == plan["source_owners"]["worker"]
         and claim["source_head"] == state[0]
         and claim["source_only_audit"] is True
         and claim["test_data_used"] is False, "CLAIM")
    policy_ref = claim["policy"]
    need(policy_ref["path"] == str(REPO / "NEXT_RUN_POLICY.json"), "POLICY_PATH")
    policy = base.read_json(policy_ref)
    gate = policy.get(STAGE)
    need(isinstance(gate, dict) and gate.get("status") == "ADMITTED"
         and gate.get("plan") == plan_ref
         and gate.get("worker") == plan["source_owners"]["worker"]
         and gate.get("output_root") == str(output)
         and gate.get("execution_authorized") is True
         and gate.get("data_reads_authorized") is True
         and gate.get("claimed_invocations") == gate.get("completed_invocations") == 0
         and gate.get("max_invocations") == 1 and gate.get("relaunch_allowed") is False
         and gate.get("memory_max") == "4G" and gate.get("swap_max") == "512M"
         and gate.get("wall_seconds") == 1800
         and gate.get("model_forwards_authorized") is False
         and gate.get("optimizer_steps_authorized") is False
         and gate.get("checkpoint_reads_authorized") is False
         and gate.get("broker_access_allowed") is False
         and gate.get("test_data_used") is False
         and policy.get("training_enabled") is False
         and policy.get("native_training_enabled") is False
         and policy.get("native_launch_authorized") is False, "POLICY")
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap["memory_max"] == cap["memory_high"] == 4294967296
         and cap["swap"] == 536870912 and cap["pids"] == 64
         and cap["numerical_threads"] == 1
         and lock["scope"] == cap["cgroup_path"][len("/sys/fs/cgroup"):], "CAP")
    return {"claim": ref(claim_path), "policy": policy_ref, "cap": cap, "lock": lock}


def run(plan_path: Path, plan_sha256: str) -> dict:
    from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

    plan_ref = {"path": str(plan_path), "sha256": plan_sha256}
    plan, full_plan, saved, cohorts = read_prepared(plan_ref)
    original_state = git_state()
    context = check_claim_and_policy(plan, plan_ref, original_state)
    old_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError("GRAY_PHYSICAL_WALL_LIMIT")))
    signal.alarm(LIMITS["wall_seconds"])
    started = time.monotonic()
    try:
        schedule, authority = base.read_closure(full_plan)
        m1, m5, issues, m5_source_rows, stats = base.load_sources(full_plan, authority)
        need(np.array_equal(issues, saved["issue_ns"])
             and np.array_equal(m5_source_rows, saved["m5_source_row"]), "ISSUE_PARITY")
        cutoff = base.pd_ns(base.TRAIN_END)
        atr = target.build_causal_tf_atr(m5, cutoff_ns=cutoff)
        m1_times = target._clock(m1, target.EXIT_DECISION_BAR_SECONDS)
        validity = m1_times.asi8 + base.MINUTE_NS
        known, _ = target._known_closures(
            m1_times, full_plan["m1"]["source"], authority, schedule)
        unknown = unknown_gap_mask(validity, known)
        need(len(validity) == base.M1_ROWS
             and int(unknown.sum()) == authority["unknown_source_gap_count"], "SOURCE_GAPS")
        values = target._ohlc(m1)
        open_, high, low, close = (values[key] for key in ("open", "high", "low", "close"))
        monthly: dict[str, dict] = {name: {} for name in (
            time.strftime("%Y-%m", time.gmtime(edge // 10**9))
            for edge in base.month_edges()[:-1])}
        totals = {}
        for col, tf in enumerate(target.TIMEFRAMES):
            window = target.LOCAL_WINDOW_M1_BARS[tf]
            source_rows = saved["m1_source_row"][:, col]
            complete = saved["complete_range_atr_valid"][:, col]
            chosen = np.flatnonzero(complete)
            rows = source_rows[chosen] - SOURCE_ROW_OFFSET
            starts = require_endpoint_and_window(
                issues[chosen], rows, source_rows[chosen], validity,
                base.LEADS[col], window, unknown)
            local_range = raw_window_range(open_, high, low, close, unknown, window)
            surface = atr["surfaces"][tf]
            position = np.searchsorted(surface["available_ns"], validity[rows], side="right") - 1
            need(np.all(position >= 0) and np.all(surface["available_ns"][position] <= validity[rows]),
                 "ATR_CLOCK_" + tf)
            scale = surface["atr"][position]
            require_ratio_parity(local_range[rows], scale,
                                 saved["complete_range_atr"][chosen, col])
            net = close[rows] - open_[starts]
            net_atr = net / scale
            eff = np.full(len(chosen), np.nan, np.float64)
            positive_range = local_range[rows] > 0
            eff[positive_range] = np.abs(net[positive_range]) / local_range[rows][positive_range]
            need(np.all((eff[positive_range] >= 0) & (eff[positive_range] <= 1)),
                 "EFFICIENCY_" + tf)
            spread_ratio = saved["atr_to_quoted_spread"][chosen, col]
            spread_valid = saved["atr_to_quoted_spread_valid"][chosen, col]
            net_spreads = np.full(len(chosen), np.nan, np.float64)
            net_spreads[spread_valid] = net_atr[spread_valid] * spread_ratio[spread_valid]
            crossings, run = close_diagnostics(close, starts, rows, open_[starts], net, window)
            wall = (validity[rows] - (validity[starts] - base.MINUTE_NS)) / base.MINUTE_NS
            issue_month = np.searchsorted(base.month_edges()[1:], issues[chosen], side="right")
            need(np.all((issue_month >= 0) & (issue_month < 20)), "MONTH_" + tf)
            totals[tf] = {}
            for name in COHORTS:
                selected = cohorts[name][chosen, col]
                totals[tf][name] = summarize(
                    net_atr[selected], eff[selected], net_spreads[selected],
                    crossings[selected], run[selected], wall[selected])
                for month_index, month_name in enumerate(monthly):
                    within = selected & (issue_month == month_index)
                    monthly[month_name].setdefault(tf, {})[name] = summarize(
                        net_atr[within], eff[within], net_spreads[within],
                        crossings[within], run[within], wall[within])
                need(sum(monthly[m][tf][name]["n"] for m in monthly)
                     == totals[tf][name]["n"], "MONTH_PARTITION_" + tf + "_" + name)
            need(sum(totals[tf][name]["n"] for name in COHORTS) == len(chosen),
                 "COHORT_PARTITION_" + tf)
        need(sum(totals[tf]["gray_0_5_to_1_0"]["n"] for tf in target.TIMEFRAMES)
             == SCOPE["gray_cells"], "GRAY_TOTAL")
        need(git_state() == original_state, "FINAL_SOURCE")
        for path, stat in stats.items():
            need(base.regular(Path(path)) == stat, "FINAL_RAW_STAT")
        for name, relative in SOURCE_PATHS.items():
            need(ref(REPO / relative) == plan["source_owners"][name],
                 "FINAL_OWNER_" + name)
        for value in INPUTS.values():
            base.checked(value, max_bytes=64 * 1024 * 1024)
        output = Path(plan["output_dir"])
        need({p.name for p in output.iterdir()} == {"CLAIM.json"}
             and ref(output / "CLAIM.json") == context["claim"], "FINAL_OUTPUT")
        result = {
            "schema_version": "gx1_market_state_gray_physical_result_v1",
            "status": "COMPLETE", "decision": "DIAGNOSTIC_ONLY_TARGET_OPEN",
            "plan": plan_ref, "claim": context["claim"],
            "source_head": original_state[0],
            "source_refs": {"m1": full_plan["m1"], "m5": full_plan["m5"],
                            "closure": full_plan["closure"],
                            "candidates": INPUTS, "owners": plan["source_owners"]},
            "source_read_scope": "TRAIN M1 OHLC; M5 boundary row group decodes Jan2026 VAL then removes it before ATR and all summaries. No TEST.",
            "m1_rows": len(validity), "m5_issues": len(issues),
            "cohorts": COHORTS, "total": totals, "monthly": monthly,
            "net_rule": "Close at exact endpoint minus open of first of W observed M1 bars; excludes predecessor-close or reopening jump before that first open.",
            "local_range_rule": "OHLC extrema plus prior observed close; unknown-gap successor uses its own open; known closure preserves predecessor close.",
            "close_diagnostic_rule": "Strict sign changes in completed M1 closes relative to first open, ignoring equal closes; terminal count skips equal closes and stops at the last opposite close, so it is neither consecutive-bar duration nor a state episode.",
            "overlap_note": "M5-issued windows overlap across issues and TF; counts are cells, not independent market episodes.",
            "quoted_spread_note": "Net quoted-spread scale reuses the hashbound single-threshold ATR/spread ratio at the same exact M1 endpoint; it is not executable bid/ask economics.",
            "model_forwards": 0, "optimizer_steps": 0,
            "test_data_used": False, "target_locked": False,
            "elapsed_seconds": time.monotonic() - started,
            "cap": {"execution": context["cap"], "lock": context["lock"]},
        }
        raw_result = base.encode(result)
        need(len(raw_result) <= 16 * 1024 * 1024, "RESULT_SIZE")
        _publish_bytes(output / "RESULT.json", raw_result)
        return result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    plan_ref = {"path": args.plan, "sha256": args.plan_sha256}
    if args.preflight:
        _, _, _, cohorts = read_prepared(plan_ref)
        print(json.dumps({"stage": STAGE, "preflight": "PASS_NO_CLAIM_NO_RAW_READ",
                          "gray_cells": int(cohorts["gray_0_5_to_1_0"].sum())},
                         sort_keys=True))
    else:
        run(Path(args.plan), args.plan_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
