"""One-use read-only fixed persistence reference for the M5 structure candidate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys

sys.dont_write_bytecode = True

import numpy as np

from gx1.audit.measure_ta_m5_structure_candidate_v1 import measure_candidate
from gx1.audit.run_ta_m5_structure_fold2_readonly_v1 import file_sha, read_json
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution

REPO = Path("/home/andre2/src/GX1_CURRENT")
PLAN = REPO / "configs/research/TA_M5_STRUCTURE_PERSISTENCE_20261116_001.json"
STAGE = "ta_m5_structure_persistence_20261116_001"
START = int(np.datetime64("2024-05-01", "ns").astype(np.int64))
SPLIT = int(np.datetime64("2024-09-01", "ns").astype(np.int64))
END = int(np.datetime64("2025-01-01", "ns").astype(np.int64))
CLASSES = (1, 2, 0, 3)  # UP, DOWN, MIXED, TIE; fixed prior tie order


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError("TA_M5_PERSISTENCE_" + reason)


def _timeout(_signum: int, _frame: object) -> None:
    raise TimeoutError("TA_M5_PERSISTENCE_WALL_LIMIT")


def preflight(plan_sha: str, *, admitted: bool) -> tuple[dict, dict, dict]:
    import pyarrow.parquet as pq

    need(file_sha(PLAN) == plan_sha, "PLAN_SHA")
    plan = read_json(PLAN)
    policy = read_json(REPO / "NEXT_RUN_POLICY.json")
    gate = policy[STAGE]
    need(plan["stage"] == STAGE and plan["status"] == "PREPARED_NOT_ADMITTED"
         and plan["execution_authorized"] is False and plan["model_fits_authorized"] is False
         and plan["test_data_used"] is False and plan["target_locked"] is False
         and plan["scope"] == {"prior_issue_start_utc": "2024-05-01T00:00:00Z",
                               "prior_mature_before_utc": "2024-09-01T00:00:00Z",
                               "evaluation_issue_start_utc": "2024-09-01T00:00:00Z",
                               "evaluation_target_end_utc_exclusive": "2025-01-01T00:00:00Z",
                               "fixed_smc_state_to_pair_class": {"0": 1, "1": 0, "2": 0, "3": 2, "4": "FROZEN_PRIOR"},
                               "unknown_pair_excluded_from_accuracy": True,
                               "weekly_block_bootstrap_seed": 20261116,
                               "weekly_block_bootstrap_draws": 2000,
                               "test_data_used": False}
         and plan["limits"] == {"one_use": True, "class": "audit", "memory_max": "4G",
                                "swap_max": "512M", "wall_seconds": 900, "gpu": False,
                                "model_forwards": 0, "optimizer_steps": 0,
                                "no_restart_on_failure": True}, "PLAN_SCOPE")
    need(gate["plan"] == {"path": str(PLAN), "sha256": plan_sha}
         and gate["claimed_invocations"] == 0
         and (gate["status"] == "ADMITTED_ONE_USE" and gate["execution_authorized"] is True
              if admitted else gate["status"] == "PREPARED_NOT_ADMITTED"
              and gate["execution_authorized"] is False), "POLICY_GATE")
    branch = subprocess.check_output(("git", "branch", "--show-current"), cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(("git", "status", "--porcelain"), cwd=REPO, text=True)
    need(branch == "work/gx1-current" and not dirty, "CLEAN_SOURCE")
    for ref in plan["source_owners"].values():
        need(file_sha(Path(ref["path"])) == ref["sha256"], "SOURCE_SHA")
    parent_ref = plan["parent_plan"]
    need(file_sha(Path(parent_ref["path"])) == parent_ref["sha256"], "PARENT_PLAN_SHA")
    parent = read_json(Path(parent_ref["path"]))
    for ref in parent["source_owners"].values():
        need(file_sha(Path(ref["path"])) == ref["sha256"], "PARENT_SOURCE_SHA")
    previous_ref = plan["parent_result"]
    need(file_sha(Path(previous_ref["path"])) == previous_ref["sha256"], "PARENT_RESULT_SHA")
    previous = read_json(Path(previous_ref["path"]))
    need(previous["decision"] == "SOURCE_COVERAGE_ONLY_EDGE_UNQUALIFIED"
         and previous["plan_sha256"] == parent_ref["sha256"], "PARENT_RESULT")
    output = Path(plan["output_root"])
    need(output == Path("/home/andre2/GX1_RUNS/TA_M5_STRUCTURE_PERSISTENCE_20261116_001")
         and not output.exists(), "OUTPUT_ABSENT")
    m5 = parent["input"]
    manifest_ref = m5["manifest"]
    need(file_sha(Path(manifest_ref["path"])) == manifest_ref["sha256"], "MANIFEST_SHA")
    manifest = read_json(Path(manifest_ref["path"]))
    need(manifest["output_parquet"] == m5["path"]
         and manifest["output_parquet_sha256"] == m5["sha256_expected_from_manifest"]
         and m5["row_groups_inclusive"] == [241, 252], "INPUT_BINDING")
    parquet = pq.ParquetFile(m5["path"])
    need(parquet.metadata.num_rows == manifest["rows"]
         and set(("time", "high", "low", "close", "atr", "smc_swing_state")) <= set(parquet.schema.names)
         and parquet.metadata.num_row_groups > 252, "PARQUET_META")
    return plan, parent, previous


def score(time: np.ndarray, current_state: np.ndarray, outcome: dict[str, np.ndarray]) -> dict:
    issues = outcome["issue_row"]
    target = outcome["pair_class"]
    state = current_state[issues].astype(np.int8)
    need(bool(np.all((state >= 0) & (state <= 4))), "STATE_VALUES")
    known = target >= 0
    confirmed = np.full(len(issues), np.iinfo(np.int64).max, dtype=np.int64)
    idx = np.flatnonzero(known)
    confirmed[idx] = np.maximum(time[outcome["next_high_confirm_row"][idx]],
                                time[outcome["next_low_confirm_row"][idx]])
    prior = known & (time[issues] >= START) & (time[issues] < SPLIT) & (confirmed < SPLIT)
    evaluation = known & (time[issues] >= SPLIT) & (time[issues] < END)
    need(bool(prior.sum() > 0 and evaluation.sum() > 0), "COHORTS")
    counts = {str(code): int((target[prior] == code).sum()) for code in CLASSES}
    frozen = max(CLASSES, key=lambda code: counts[str(code)])
    y = target[evaluation]
    now = state[evaluation]
    persistence = np.full(len(y), frozen, dtype=np.int8)
    for smc_state, predicted in ((0, 1), (1, 0), (2, 0), (3, 2)):
        persistence[now == smc_state] = predicted
    prior_correct = (y == frozen)
    persistence_correct = (y == persistence)
    delta = persistence_correct.astype(np.int8) - prior_correct.astype(np.int8)
    week = time[issues[evaluation]] // (7 * 24 * 60 * 60 * 10**9)
    _, block = np.unique(week, return_inverse=True)
    numerator = np.bincount(block, weights=delta)
    denominator = np.bincount(block)
    rng = np.random.default_rng(20261116)
    draws = rng.integers(0, len(numerator), size=(2000, len(numerator)))
    sampled = numerator[draws].sum(axis=1) / denominator[draws].sum(axis=1)
    crosstab = {str(s): {str(c): int(((now == s) & (y == c)).sum()) for c in CLASSES}
                for s in range(5)}
    return {
        "prior_issues_mature_before_split": int(prior.sum()),
        "prior_class_counts": counts, "frozen_prior_class": int(frozen),
        "evaluation_issues": int(((time[issues] >= SPLIT) & (time[issues] < END)).sum()),
        "evaluation_known_classes": int(len(y)),
        "evaluation_class_counts": {str(c): int((y == c).sum()) for c in CLASSES},
        "fixed_rule": {"0": 1, "1": 0, "2": 0, "3": 2, "4": int(frozen)},
        "frozen_prior_accuracy": float(prior_correct.mean()),
        "persistence_accuracy": float(persistence_correct.mean()),
        "persistence_minus_prior_accuracy": float(delta.mean()),
        "paired_utc_week_block_bootstrap_95pct": [float(v) for v in np.quantile(sampled, (0.025, 0.975))],
        "utc_week_blocks": int(len(numerator)),
        "current_smc_state_by_future_class": crosstab,
    }


def run(plan_sha: str) -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    plan, parent, previous = preflight(plan_sha, admitted=True)
    cap = require_capped_cpu_audit_execution()
    need(cap["memory_max"] <= 4 * 1024**3 and cap["swap"] <= 512 * 1024**2, "CAP")
    output = Path(plan["output_root"])
    output.mkdir()
    (output / "CLAIM.json").write_text(json.dumps({"stage": STAGE, "plan_sha256": plan_sha,
        "source_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=REPO, text=True).strip(),
        "claimed_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    previous_alarm = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(900)
    try:
        m5 = parent["input"]
        need(file_sha(Path(m5["path"])) == m5["sha256_expected_from_manifest"], "INPUT_SHA")
        parquet = pq.ParquetFile(m5["path"])
        keys = ("time", "high", "low", "close", "atr", "smc_swing_state")
        pieces = {key: [] for key in keys}
        for row_group in range(241, 253):
            table = parquet.read_row_group(row_group, columns=list(keys))
            for key in keys:
                column = table[key].to_pandas()
                pieces[key].append(np.asarray(pd.to_datetime(column, utc=True).array.asi8
                                              if key == "time" else column.to_numpy()))
        arrays = {key: np.concatenate(value) for key, value in pieces.items()}
        cutoff = int(np.searchsorted(arrays["time"], END))
        need(0 < cutoff < len(arrays["time"]), "CROP")
        arrays = {key: value[:cutoff] for key, value in arrays.items()}
        time = arrays["time"].astype(np.int64)
        issues = np.flatnonzero(time >= START).astype(np.int64)
        authority = parent["closure_authority"]
        need(file_sha(Path(authority["path"])) == authority["sha256"], "CLOSURE_SHA")
        closure = read_json(Path(authority["path"]))
        intervals = np.array([[pd.Timestamp(x["closure_start_utc"]).value,
                               pd.Timestamp(x["closure_end_utc_exclusive"]).value]
                              for x in closure["intervals"] if x["successor_across_gap_allowed"] is True],
                             dtype=np.int64).reshape(-1, 2)
        outcome = measure_candidate(time, arrays["high"], arrays["low"], arrays["close"],
                                    arrays["atr"], issues, fold_target_end_ns=END,
                                    known_closure_ns=intervals)
        prior_counts = previous["aggregate"]["pair_classes"]
        actual_counts = {"UNKNOWN": int((outcome["pair_class"] == -1).sum()),
                         "MIXED": int((outcome["pair_class"] == 0).sum()),
                         "UP": int((outcome["pair_class"] == 1).sum()),
                         "DOWN": int((outcome["pair_class"] == 2).sum()),
                         "TIE": int((outcome["pair_class"] == 3).sum())}
        need(len(issues) == previous["aggregate"]["issue_count"]
             and actual_counts == prior_counts, "TARGET_PARITY")
        measured = score(time, arrays["smc_swing_state"], outcome)
        report = {"decision": "FIXED_CAUSAL_REFERENCE_ONLY_EDGE_UNQUALIFIED",
                  "plan_sha256": plan_sha, "parent_result_sha256": plan["parent_result"]["sha256"],
                  "metrics": measured, "model_forwards": 0, "optimizer_steps": 0,
                  "target_locked": False, "test_data_used": False, "economic_edge_proven": False}
        with (output / "RESULT.json").open("x") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        terminal = {"status": "COMPLETE", "exit_code": 0, "plan_sha256": plan_sha}
    except Exception as exc:
        terminal = {"status": "FAILED", "exit_code": 1, "plan_sha256": plan_sha,
                    "error": type(exc).__name__ + ":" + str(exc)}
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        with (output / "TERMINAL.json").open("x") as handle:
            json.dump(terminal, handle, indent=2)
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    need(args.plan == PLAN, "PLAN_PATH")
    if args.preflight:
        preflight(args.plan_sha256, admitted=False)
        print("PASS_NO_PRICE_ROWS_READ_NO_CLAIM")
    else:
        run(args.plan_sha256)


if __name__ == "__main__":
    main()
