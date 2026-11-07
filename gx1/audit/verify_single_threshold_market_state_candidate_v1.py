"""One-use source audit of a PROPOSED single 1.0-ATR market-state threshold.

Reuses the verified full-TRAIN M1/M5/closure readers. It builds one 1.0 tape,
not a fit, and publishes aligned allowed masks plus observed range/spread
ratios. The threshold is a candidate, never selected by a model score here.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from gx1.audit import build_full_train_market_state_endpoint_candidate_v1 as base

REPO = base.REPO
STAGE = "single_threshold_market_state_candidate_20261107_001"
PLAN_SCHEMA = "gx1_single_threshold_market_state_candidate_plan_v1"
CLAIM_SCHEMA = "gx1_single_threshold_market_state_candidate_claim_v1"
BUDGET = {"memory_max": 4294967296, "swap": 536870912, "pids": 64,
          "numerical_threads": 1, "max_wall_seconds": 1800,
          "max_invocations": 1, "max_json_bytes": 16777216,
          "max_npz_bytes": 67108864}
OLD_ROOT = Path("/home/andre2/GX1_RUNS/FULL_TRAIN_MARKET_STATE_ENDPOINT_CANDIDATE_20261107_001")
OLD_REFS = {
    "result": {"path": str(OLD_ROOT / "RESULT.json"),
               "sha256": "3a508ce45755c8db2ad8c078d56a6afef8b88a6af3e0a0b159ca388def38ebc5"},
    "terminal": {"path": str(OLD_ROOT / "TERMINAL.json"),
                 "sha256": "528a96b83f5d5191c32d2ccd89df772284edb9e7815d974554dc5ff7e5839922"},
    "row_masks": {"path": str(OLD_ROOT / "ROW_MASKS.npz"),
                  "sha256": "ad0e7d56c58b463bdd3d61fe6b2fcf2b206afa56e899dfb36a850e63c5ec3020"},
}
OWNERS = {
    "worker": "gx1/audit/verify_single_threshold_market_state_candidate_v1.py",
    "full_train_helper": "gx1/audit/build_full_train_market_state_endpoint_candidate_v1.py",
    "target": "gx1/contracts/entry_dynamic_mtf_state_targets_v1.py",
    "rowwise_helper": "gx1/audit/verify_market_state_endpoint_candidate_v2.py",
    "closure_helper": "gx1/scripts/audit_dynamic_mtf_state_closure_observability_v1.py",
    "publication_owner": "gx1/contracts/immutable_event_authority_v1.py",
    "capped_owner": "gx1/contracts/gx1_capped_execution_v1.py",
}
DOWN, FLAT, UP, ACTIVE, ALL = 1, 2, 4, 5, 7


def need(condition: bool, key: str) -> None:
    if not condition:
        raise RuntimeError("SINGLE_THRESHOLD_CANDIDATE_" + key)


def _check_plan(plan: dict, plan_ref: dict, root: Path) -> None:
    expected_keys = {"schema_version", "policy_gate", "m1", "m5", "closure",
                     "old_full", "train_start_utc", "train_end_utc_exclusive",
                     "m1_row_groups", "m5_row_groups", "m1_expected_rows",
                     "m5_expected_issues", "lead_minutes", "factor", "source_owners",
                     "output_root", "budget", "test_data_used"}
    need(set(plan) == expected_keys and plan["schema_version"] == PLAN_SCHEMA
         and plan["policy_gate"] == STAGE
         and plan["m1"] == base.FIXED_REFS["m1"]
         and plan["m5"] == base.FIXED_REFS["m5"]
         and plan["closure"] == base.FIXED_REFS["closure"]
         and plan["old_full"] == OLD_REFS
         and plan["train_start_utc"] == base.TRAIN_START
         and plan["train_end_utc_exclusive"] == base.TRAIN_END
         and plan["m1_row_groups"] == list(base.M1_GROUPS)
         and plan["m5_row_groups"] == list(base.M5_GROUPS)
         and plan["m1_expected_rows"] == base.M1_ROWS
         and plan["m5_expected_issues"] == 118706
         and plan["lead_minutes"] == list(base.LEADS)
         and plan["factor"] == 1.0 and plan["budget"] == BUDGET
         and plan["output_root"] == str(root)
         and plan["test_data_used"] is False, "PLAN_BINDING")
    need(root.is_absolute() and root.parent == Path("/home/andre2/GX1_RUNS")
         and root.is_dir() and not root.is_symlink(), "ROOT")
    need(isinstance(plan["source_owners"], dict)
         and set(plan["source_owners"]) == set(OWNERS), "OWNER_SHAPE")
    for name, relative in OWNERS.items():
        owner = plan["source_owners"][name]
        need(owner.get("path") == str(REPO / relative), "OWNER_PATH:" + name)
        base.checked(owner)
    base.checked(plan_ref)


def _check_claim(root: Path, plan_ref: dict, worker_ref: dict) -> tuple[dict, dict]:
    path = root / "CLAIM.json"
    before = base.regular(path)
    need(before[3] <= BUDGET["max_json_bytes"], "CLAIM_SIZE")
    raw = path.read_bytes()
    need(before == base.regular(path), "CLAIM_CHANGED")
    claim = json.loads(raw)
    claim_ref = base.ref(path, raw)
    need(isinstance(claim, dict) and set(claim) == {
        "schema_version", "plan", "worker", "policy", "source_head",
        "source_only_candidate", "test_data_used",
    } and claim["schema_version"] == CLAIM_SCHEMA
         and claim["plan"] == plan_ref and claim["worker"] == worker_ref
         and claim["source_only_candidate"] is True and claim["test_data_used"] is False
         and {item.name for item in root.iterdir()} == {"CLAIM.json"}, "CLAIM")
    return claim, claim_ref


def _check_admission(plan: dict, plan_ref: dict, claim: dict) -> dict:
    need(Path.cwd() == REPO and subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
        == "work/gx1-current", "WORKDIR_BRANCH")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    need(claim["source_head"] == head and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True) == "", "CLEAN_HEAD")
    policy_ref = claim["policy"]
    need(isinstance(policy_ref, dict)
         and policy_ref.get("path") == str(REPO / "NEXT_RUN_POLICY.json"), "POLICY_PATH")
    policy = base.read_json(policy_ref)
    gate = policy[STAGE]
    expected = {"decision": "ADMITTED", "plan": plan_ref,
                "execution_authorized": True, "data_reads_authorized": True,
                "target_builds_authorized": 1, "model_calls_authorized": 0,
                "test_authorized": False, "training_authorized": False,
                "claimed_invocations": 0, "completed_invocations": 0,
                "max_invocations": 1, "output_root": plan["output_root"]}
    need(policy["training_enabled"] is False and isinstance(gate, dict)
         and all(gate.get(key) == value for key, value in expected.items()), "POLICY")
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry,
    )
    limits = require_capped_cpu_audit_execution()
    limits["lock"] = require_capped_lock_ancestry()
    need(all(limits[key] == BUDGET[key] for key in
             ("memory_max", "swap", "pids", "numerical_threads")), "CAPS")
    return limits


def _read_old(plan: dict, np) -> tuple[dict, dict]:
    result = base.read_json(plan["old_full"]["result"])
    terminal = base.read_json(plan["old_full"]["terminal"])
    need(result["decision"] == "PASS_FULL_TRAIN_CANDIDATE_SOURCE_ONLY"
         and result["row_masks"] == plan["old_full"]["row_masks"]
         and terminal["status"] == "COMPLETE" and terminal["exit_code"] == 0
         and terminal["result"] == plan["old_full"]["result"]
         and result["source_refs"]["m1"] == plan["m1"]
         and result["source_refs"]["m5"] == plan["m5"]
         and result["source_refs"]["closure"] == plan["closure"]
         and result["m5_issues"] == plan["m5_expected_issues"]
         and len(result["monthly"]) == 20
         and result["test_data_used"] is False and result["target_locked"] is False
         and result["model_calls"] == result["optimizer_steps"] == 0
         and terminal["test_data_used"] is False and terminal["target_locked"] is False,
         "OLD_RESULT")
    raw, _ = base.checked(plan["old_full"]["row_masks"],
                          max_bytes=BUDGET["max_npz_bytes"])
    need(raw is not None, "OLD_NPZ")
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        need(set(z.files) == {"issue_ns", "m5_source_row", "lead_minutes",
                               "allowed_set", "m1_source_row"}, "OLD_KEYS")
        old = {name: np.array(z[name]) for name in z.files}
    need(old["allowed_set"].shape == old["m1_source_row"].shape ==
         (plan["m5_expected_issues"], len(base.LEADS))
         and np.array_equal(old["lead_minutes"], base.LEADS)
         and np.all(np.diff(old["issue_ns"]) > 0), "OLD_SHAPE")
    return result, old


def classify_one_factor(tape: dict, early, rows, col: int, target, np):
    """One activity boundary; side only from the owner's conservative proof."""
    threshold = tape["threshold"][rows, col]
    local_range = tape["local_range"][rows, col]
    blocked = (target.ATR_UNAVAILABLE | target.LOCAL_SPAN_INCOMPLETE
               | target.UNKNOWN_SOURCE_GAP)
    complete = ((tape["reason"][rows, col] & blocked) == 0) & (threshold > 0)
    flat = complete & (local_range < threshold)
    active = complete & (local_range >= threshold)
    state, observed = tape["state"][rows, col], tape["observed"][rows, col]
    down = active & observed & (state == target.DOWN)
    up = active & observed & (state == target.UP)
    partial_complete = active & ~down & ~up
    partial_early = early[rows]
    need(not np.any(flat & active) and not np.any(partial_early & complete)
         and not np.any(active & observed & (state == target.FLAT)), "DISJOINT")
    masks = np.full(len(rows), ALL, np.uint8)
    masks[flat], masks[down], masks[up] = FLAT, DOWN, UP
    masks[partial_complete | partial_early] = ACTIVE
    counts = {"flat": int(flat.sum()), "down": int(down.sum()), "up": int(up.sum()),
              "partial_complete": int(partial_complete.sum()),
              "partial_early": int(partial_early.sum())}
    need(sum(counts.values()) <= len(rows), "COUNT")
    return masks, complete, counts


def ratio_summary(values, np, *, spread_units: bool = False) -> dict:
    """Describe a finite nonnegative ratio; only spread ratios get spread units."""
    data = np.asarray(values, dtype=np.float64)
    need(data.ndim == 1 and np.isfinite(data).all() and np.all(data >= 0), "CALIBRATION_DATA")
    summary = {"count": int(len(data)), "median": None, "p10": None, "p90": None}
    if len(data):
        summary.update(median=float(np.median(data)), p10=float(np.quantile(data, 0.1)),
                       p90=float(np.quantile(data, 0.9)))
    if spread_units:
        summary["share_threshold_le_two_spreads"] = (
            float(np.mean(data <= 2.0)) if len(data) else None)
    return summary


def _read_quote_spreads(plan: dict, expected_rows: int, m1_stat: tuple, np):
    import pyarrow.parquet as pq
    source = Path(plan["m1"]["source"]["path"])
    parquet = pq.ParquetFile(source)
    need(all(name in parquet.schema_arrow.names for name in ("bid_close", "ask_close")),
         "QUOTE_COLUMNS")
    quotes = parquet.read_row_groups(list(range(*base.M1_GROUPS)),
                                     columns=["bid_close", "ask_close"],
                                     use_threads=False).to_pandas()
    need(len(quotes) == expected_rows, "QUOTE_ROWS")
    bid = quotes["bid_close"].to_numpy(np.float64)
    ask = quotes["ask_close"].to_numpy(np.float64)
    valid_quotes = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid)
    spread = np.full(len(quotes), np.nan, np.float64)
    spread[valid_quotes] = ask[valid_quotes] - bid[valid_quotes]
    quality = {"rows": len(quotes),
               "invalid_or_crossed_quote_rows": int((~valid_quotes).sum()),
               "zero_spread_rows": int((valid_quotes & (spread == 0)).sum())}
    need(base.regular(source) == m1_stat, "QUOTE_SOURCE_CHANGED")
    return spread, quality


def _compare_old(old_masks, new_masks, old_month: dict, np) -> dict:
    need(old_masks.shape == new_masks.shape and
         np.all(new_masks[old_masks != ALL] == old_masks[old_masks != ALL]),
         "OLD_NONUNKNOWN_CHANGED")
    changed = (old_masks == ALL) & (new_masks == FLAT)
    unchanged_unknown = (old_masks == ALL) & (new_masks == ALL)
    need(np.all((old_masks != ALL) | changed | unchanged_unknown)
         and int(changed.sum()) == old_month["middle_unknown"]
         and int(unchanged_unknown.sum()) ==
         old_month["missing_exact_endpoint"] + old_month["other_unknown"],
         "OLD_GRAY_PARITY")
    return {"old_gray_to_flat": int(changed.sum()),
            "old_other_unknown_stays_unknown": int(unchanged_unknown.sum()),
            "old_nonunknown_unchanged": int((old_masks != ALL).sum())}


def run(plan_path: Path, plan_sha256: str, root: Path) -> dict:
    import numpy as np
    from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
    from gx1.audit.verify_market_state_endpoint_candidate_v2 import exact_positions
    from gx1.scripts.audit_dynamic_mtf_state_closure_observability_v1 import early_active_proof
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

    plan_ref = {"path": str(plan_path), "sha256": plan_sha256}
    plan = base.read_json(plan_ref)
    _check_plan(plan, plan_ref, root)
    claim, claim_ref = _check_claim(root, plan_ref, plan["source_owners"]["worker"])
    limits = _check_admission(plan, plan_ref, claim)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError("SINGLE_THRESHOLD_CANDIDATE_DEADLINE")))
    signal.alarm(BUDGET["max_wall_seconds"])
    started = time.monotonic()
    try:
        old_result, old = _read_old(plan, np)
        schedule, authority = base.read_closure(plan)
        m1, m5, issues, m5_source_rows, source_stats = base.load_sources(plan, authority)
        need(len(issues) == plan["m5_expected_issues"]
             and np.array_equal(old["issue_ns"], issues)
             and np.array_equal(old["m5_source_row"], m5_source_rows), "ISSUES")
        m1_stat = source_stats[plan["m1"]["source"]["path"]]
        quoted_spread, quote_quality = _read_quote_spreads(plan, len(m1), m1_stat, np)
        cutoff = base.pd_ns(base.TRAIN_END)
        need(target.MINIMUM_MOVE_ATR == 0.5
             and tuple(target.TIMEFRAMES) == ("M5", "M15", "H1", "H4", "D1"), "TARGET")
        atr = target.build_causal_tf_atr(m5, cutoff_ns=cutoff)
        target.MINIMUM_MOVE_ATR = 1.0
        try:
            atr["definition_sha256"] = target.dynamic_mtf_state_target_contract()["contract_sha256"]
            series = target.build_dynamic_mtf_state_series(
                m1, tf_atr=atr, source_ref=plan["m1"]["source"],
                atr_source_ref=plan["m5"]["source"], cutoff_ns=cutoff,
                source_row_offset=5193205, closure_authority=authority,
                exact_schedule=schedule)
        finally:
            target.MINIMUM_MOVE_ATR = 0.5
        tape = series.arrays
        axis = np.asarray(tape["validity_ns"], dtype=np.int64)
        need(len(axis) == base.M1_ROWS and np.all(np.diff(axis) > 0)
             and np.array_equal(tape["source_row"],
                                np.arange(5193205, 5193205 + base.M1_ROWS)), "TAPE_CLOCK")
        m1_times = target._clock(m1, target.EXIT_DECISION_BAR_SECONDS)
        known, _ = target._known_closures(m1_times, plan["m1"]["source"], authority, schedule)
        unknown = np.zeros(len(axis), bool)
        for row in np.flatnonzero(np.diff(axis) != base.MINUTE_NS) + 1:
            unknown[row] = int(row) not in known
        need(int(unknown.sum()) == authority["unknown_source_gap_count"], "UNKNOWN_GAP")
        highs = m1["high"].to_numpy(np.float64)
        lows = m1["low"].to_numpy(np.float64)
        early = {}
        for col, tf in enumerate(target.TIMEFRAMES):
            early[tf], _ = early_active_proof(
                highs, lows, tape["threshold"][:, col], unknown,
                target.LOCAL_WINDOW_M1_BARS[tf], np)
        shape = (len(issues), len(base.LEADS))
        allowed = np.full(shape, ALL, np.uint8)
        source_rows = np.full(shape, -1, np.int64)
        range_atr = np.zeros(shape, np.float64)
        range_atr_valid = np.zeros(shape, bool)
        atr_spread = np.zeros(shape, np.float64)
        atr_spread_valid = np.zeros(shape, bool)
        monthly = {}
        edges = base.month_edges()
        for start, stop in zip(edges[:-1], edges[1:]):
            lo, hi = np.searchsorted(issues, [start, stop])
            need(hi > lo, "MONTH")
            base_month = old_result["monthly"][time.strftime(
                "%Y-%m", time.gmtime(start // 10**9))]
            observed_month = {}
            for col, tf in enumerate(target.TIMEFRAMES):
                pos, present = exact_positions(axis, issues[lo:hi],
                                               base.LEADS[col], cutoff, np)
                rows = pos[present]
                masks, complete, counts = classify_one_factor(
                    tape, early[tf], rows, col, target, np)
                allowed[lo:hi, col][present] = masks
                source_rows[lo:hi, col][present] = tape["source_row"][rows]
                need(np.array_equal(source_rows[lo:hi, col],
                                    old["m1_source_row"][lo:hi, col]),
                     "ENDPOINT_ROW_PARITY")
                theta = tape["threshold"][rows, col]
                local_range = tape["local_range"][rows, col]
                ratios = np.zeros(len(rows), np.float64)
                ratios[complete] = local_range[complete] / theta[complete]
                need(np.isfinite(ratios).all() and np.all(ratios >= 0), "RANGE_RATIO")
                range_atr[lo:hi, col][present] = ratios
                range_atr_valid[lo:hi, col][present] = complete
                spread = quoted_spread[rows]
                spread_valid = complete & np.isfinite(spread) & (spread > 0)
                spread_ratios = np.zeros(len(rows), np.float64)
                spread_ratios[spread_valid] = theta[spread_valid] / spread[spread_valid]
                need(np.isfinite(spread_ratios).all()
                     and np.all(spread_ratios >= 0), "SPREAD_RATIO")
                atr_spread[lo:hi, col][present] = spread_ratios
                atr_spread_valid[lo:hi, col][present] = spread_valid
                compared = _compare_old(old["allowed_set"][lo:hi, col],
                                        allowed[lo:hi, col], base_month[tf], np)
                unknown_all = int((allowed[lo:hi, col] == ALL).sum())
                exact = counts["flat"] + counts["down"] + counts["up"]
                partial = counts["partial_complete"] + counts["partial_early"]
                need(exact + partial + unknown_all == hi - lo
                     and int(complete.sum()) ==
                     base_month[tf]["exact_singleton"]
                     + base_month[tf]["partial_complete"]
                     + base_month[tf]["middle_unknown"], "MONTH_PARTITION")
                observed_month[tf] = {"issues": int(hi - lo),
                    "exact_m1_endpoint": int(present.sum()),
                    **counts, "exact_singleton": exact, "partial_active": partial,
                    "unknown_all": unknown_all,
                    "complete_range_atr": int(complete.sum()),
                    "valid_quoted_spread": int(spread_valid.sum()),
                    "invalid_spread_on_complete": int(complete.sum() - spread_valid.sum()),
                    "range_atr": ratio_summary(ratios[complete], np),
                    "atr_to_quoted_spread": ratio_summary(spread_ratios[spread_valid], np, spread_units=True),
                    "comparison_to_old": compared}
            monthly[time.strftime("%Y-%m", time.gmtime(start // 10**9))] = observed_month
        need(len(monthly) == 20 and all(len(item) == 5 for item in monthly.values()),
             "MONTHS")
        totals = {tf: {key: sum(month[tf][key] for month in monthly.values())
                       for key in ("issues", "exact_singleton", "partial_active", "unknown_all",
                                   "complete_range_atr", "valid_quoted_spread")}
                  for tf in target.TIMEFRAMES}
        need(all(totals[tf]["issues"] == len(issues)
                 and totals[tf]["exact_singleton"]
                 == old_result["totals"][tf]["exact_singleton"]
                    + old_result["totals"][tf]["middle_unknown"]
                 and totals[tf]["partial_active"]
                 == old_result["totals"][tf]["partial_active"]
                 and totals[tf]["unknown_all"]
                 == old_result["totals"][tf]["unknown_all"]
                    - old_result["totals"][tf]["middle_unknown"]
                 for tf in target.TIMEFRAMES), "TOTAL_PARITY")
        need(not np.any((allowed != ALL) & ~range_atr_valid & (allowed != ACTIVE)),
             "RATIO_MASK")
        # Closing state against the exact immutable source and admission before publication.
        need(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                     text=True).strip() == claim["source_head"], "HEAD_CHANGED")
        for group in (plan["m1"], plan["m5"], plan["closure"],
                      plan["old_full"], plan["source_owners"]):
            for name, item in group.items():
                if name == "source" and group in (plan["m1"], plan["m5"]):
                    continue  # Already SHA-checked before decode; stat bookend follows.
                if "path" in item:
                    base.checked(item, max_bytes=BUDGET["max_npz_bytes"])
                else:
                    for nested in item.values():
                        base.checked(nested, max_bytes=BUDGET["max_npz_bytes"])
        need(base.regular(Path(plan["m1"]["source"]["path"])) == m1_stat
             and base.regular(Path(plan["m5"]["source"]["path"])) ==
             source_stats[plan["m5"]["source"]["path"]], "SOURCE_CHANGED")
        _check_plan(plan, plan_ref, root)
        need(_check_claim(root, plan_ref, plan["source_owners"]["worker"])
             == (claim, claim_ref)
             and _check_admission(plan, plan_ref, claim) == limits, "BOOKEND")
        buf = io.BytesIO()
        np.savez_compressed(buf, issue_ns=issues, m5_source_row=m5_source_rows,
                            lead_minutes=np.asarray(base.LEADS, np.int16),
                            allowed_set=allowed, m1_source_row=source_rows,
                            complete_range_atr=range_atr,
                            complete_range_atr_valid=range_atr_valid,
                            atr_to_quoted_spread=atr_spread,
                            atr_to_quoted_spread_valid=atr_spread_valid)
        npz_bytes = buf.getvalue()
        need(len(npz_bytes) <= BUDGET["max_npz_bytes"], "NPZ_BUDGET")
        row_path = root / "ROW_MASKS_AND_RATIOS.npz"
        row_ref = base.ref(row_path, npz_bytes)
        output = {"schema_version": STAGE + "_result_v1",
                  "decision": "PASS_SINGLE_THRESHOLD_CANDIDATE_SOURCE_ONLY",
                  "plan": plan_ref, "claim": claim_ref, "policy": claim["policy"],
                  "source_head": claim["source_head"], "limits": limits,
                  "source_refs": {key: plan[key] for key in ("m1", "m5", "closure", "old_full")},
                  "train_start_utc": base.TRAIN_START,
                  "train_end_utc_exclusive": base.TRAIN_END,
                  "m1_rows": len(axis), "m5_issues": len(issues),
                  "factor": 1.0, "quote_quality": quote_quality,
                  "allowed_bits": {"DOWN": DOWN, "FLAT": FLAT,
                                                "UP": UP, "ACTIVE_SIDE_UNKNOWN": ACTIVE,
                                                "ALL_UNKNOWN": ALL},
                  "monthly": monthly, "totals": totals,
                  "row_masks_and_ratios": row_ref,
                  "elapsed_seconds": time.monotonic() - started,
                  "test_data_used": False, "model_calls": 0, "optimizer_steps": 0,
                  "target_locked": False,
                  "source_read_scope": "Both pre-TEST parquet files hashed as opaque integrity bytes. M1 decodes exact TRAIN OHLC and bid/ask close groups 1820..2024. M5 group270 decodes Jan2026 VAL prices then filters them before ATR, labels or summaries. No TEST.",
                  "limit": "One predeclared 1.0 ATR physical candidate. Quoted-spread calibration, source masks and range ratios are not model fit, threshold selection, target lock, forecast skill or economic edge."}
        result_bytes = base.encode(output)
        need(len(result_bytes) <= BUDGET["max_json_bytes"], "JSON_BUDGET")
        _publish_bytes(row_path, npz_bytes)
        row_path.chmod(0o444)
        result_path = root / "RESULT.json"
        _publish_bytes(result_path, result_bytes)
        result_path.chmod(0o444)
        return output
    finally:
        signal.alarm(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(json.dumps(run(args.plan, args.plan_sha256, args.out_dir), sort_keys=True,
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
