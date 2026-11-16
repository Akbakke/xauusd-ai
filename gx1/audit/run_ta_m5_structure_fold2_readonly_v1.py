"""One-use, read-only Fold 2 TRAIN M5 structure coverage audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True

import numpy as np

from gx1.audit.measure_ta_m5_structure_candidate_v1 import (
    _classify_clock_breaks, measure_candidate,
)

REPO = Path("/home/andre2/src/GX1_CURRENT")
POLICY = REPO / "NEXT_RUN_POLICY.json"
STAGE = "ta_m5_structure_fold2_audit_20261116_001"
PLAN = REPO / "configs/research/TA_M5_STRUCTURE_FOLD2_AUDIT_20261116_001.json"
START = int(np.datetime64("2024-05-01", "ns").astype(np.int64))
END = int(np.datetime64("2025-01-01", "ns").astype(np.int64))
CLASSES = {-1: "UNKNOWN", 0: "MIXED", 1: "UP", 2: "DOWN", 3: "TIE"}


def need(condition: object, reason: str) -> None:
    if not bool(condition):
        raise RuntimeError("TA_M5_STRUCTURE_AUDIT_" + reason)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def preflight(plan_sha: str, admitted: bool) -> tuple[dict, dict, object]:
    import pyarrow.parquet as pq

    need(file_sha(PLAN) == plan_sha, "PLAN_SHA")
    plan = read_json(PLAN)
    gate = read_json(POLICY)[STAGE]
    need(plan["stage"] == STAGE and plan["status"] == "PREPARED_NOT_ADMITTED"
         and plan["execution_authorized"] is False
         and plan["target_locked"] is False
         and plan["model_fits_authorized"] is False
         and plan["test_data_used"] is False
         and plan["limits"] == {"one_use": True, "class": "audit", "memory_max": "4G",
                                "swap_max": "512M", "wall_seconds": 900, "gpu": False,
                                "model_forwards": 0, "optimizer_steps": 0,
                                "no_restart_on_failure": True}
         and plan["scope"]["issue_start_utc"] == "2024-05-01T00:00:00Z"
         and plan["scope"]["fold_target_end_utc_exclusive"] == "2025-01-01T00:00:00Z"
         and plan["input"]["row_groups_inclusive"] == [241, 252], "PLAN_SCOPE")
    need(gate["plan"] == {"path": str(PLAN), "sha256": plan_sha}
         and gate["claimed_invocations"] == 0
         and gate["test_data_used"] is False
         and gate["model_fits_authorized"] is False
         and (gate["status"] == "ADMITTED_ONE_USE" and gate["execution_authorized"] is True
              if admitted else
              gate["status"] == "PLAN_PREPARED_NOT_ADMITTED" and gate["execution_authorized"] is False),
         "POLICY_GATE")
    branch = subprocess.check_output(("git", "branch", "--show-current"), cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(("git", "status", "--porcelain"), cwd=REPO, text=True)
    need(branch == "work/gx1-current" and not dirty, "CLEAN_SOURCE")
    for ref in plan["source_owners"].values():
        need(file_sha(Path(ref["path"])) == ref["sha256"], "SOURCE_SHA")
    manifest_ref = plan["input"]["manifest"]
    need(file_sha(Path(manifest_ref["path"])) == manifest_ref["sha256"], "MANIFEST_SHA")
    manifest = read_json(Path(manifest_ref["path"]))
    need(manifest["decision"] == "PASS"
         and manifest["output_parquet"] == plan["input"]["path"]
         and manifest["output_parquet_sha256"] == plan["input"]["sha256_expected_from_manifest"]
         and manifest["causal_contract"]["decision_uses_closed_m5_bar"] is True,
         "MANIFEST")
    closure_ref = plan["closure_authority"]
    need(file_sha(Path(closure_ref["path"])) == closure_ref["sha256"], "CLOSURE_SHA")
    closure = read_json(Path(closure_ref["path"]))
    need(closure["decision"] == "PASS" and closure["test_data_used"] is False
         and closure["known_market_closure_count"] == 413
         and closure["unknown_source_gap_count"] == 95
         and closure["coverage_start_utc"].startswith("2024-05-01")
         and closure["coverage_end_utc_exclusive"].startswith("2026-01-01"), "CLOSURE")
    output = Path(plan["report"]["private_output_root"])
    need(output == Path("/home/andre2/GX1_RUNS/TA_M5_STRUCTURE_FOLD2_AUDIT_20261116_001")
         and not output.exists(), "OUTPUT_ABSENT")
    parquet = pq.ParquetFile(plan["input"]["path"])
    fields = parquet.schema.names
    need(set(plan["scope"]["source_columns"]) <= set(fields)
         and parquet.metadata.num_row_groups > 252
         and parquet.metadata.num_rows == manifest["rows"], "PARQUET_META")
    time_col = fields.index("time")
    first = parquet.metadata.row_group(241).column(time_col).statistics
    last = parquet.metadata.row_group(252).column(time_col).statistics
    need(str(first.min) == "2024-04-30 05:55:00+00:00"
         and str(last.max) == "2025-01-08 03:50:00+00:00", "ROWGROUP_BOUNDARY")
    return plan, closure, parquet


def aggregate(time: np.ndarray, result: dict[str, np.ndarray]) -> dict:
    import pandas as pd

    issues = result["issue_row"]
    both = result["high_observed"] & result["low_observed"]
    summary = {"issue_count": int(len(issues)), "both_observed": int(both.sum()),
               "atr_valid": int(result["atr_valid"].sum()),
               "pair_classes": {name: int((result["pair_class"] == code).sum())
                                for code, name in CLASSES.items()},
               "same_pivot_bar": int(result["same_pivot_bar"].sum()),
               "both_levels_hit": int(result["both_levels_hit"].sum()),
               "no_frozen_level_hit": int(result["no_frozen_level_hit"].sum())}
    for side in ("high", "low"):
        observed = result[f"{side}_observed"]
        delay = result[f"next_{side}_confirm_row"][observed] - issues[observed]
        summary[side] = {
            key: int(result[key].sum()) for key in
            (f"frozen_{side}_valid", f"next_{side}_pivot_found", f"future_{side}_valid",
             f"{side}_observed", f"{side}_hit", f"{side}_wick_reject", f"{side}_close_break")
        }
        summary[side]["confirm_delay_bars_p50_p90"] = (
            [float(x) for x in np.percentile(delay, (50, 90))] if len(delay) else None)
    months = pd.to_datetime(time[issues], utc=True).strftime("%Y-%m")
    summary["monthly"] = {
        month: {"issues": int(mask.sum()), "both_observed": int((mask & both).sum())}
        for month in sorted(set(months))
        for mask in [np.asarray(months == month)]
    }
    examples = []
    for code in (1, 2, 0, 3):
        for index in np.flatnonzero(result["pair_class"] == code)[:2]:
            i = int(index)
            examples.append({"class": CLASSES[code], "issue_utc": str(pd.Timestamp(time[issues[i]], tz="UTC")),
                             "frozen_high": float(result["frozen_high"][i]),
                             "frozen_low": float(result["frozen_low"][i]),
                             "next_high_pivot_utc": str(pd.Timestamp(time[result["next_high_idx"][i]], tz="UTC")),
                             "next_low_pivot_utc": str(pd.Timestamp(time[result["next_low_idx"][i]], tz="UTC")),
                             "next_high_confirm_utc": str(pd.Timestamp(time[result["next_high_confirm_row"][i]], tz="UTC")),
                             "next_low_confirm_utc": str(pd.Timestamp(time[result["next_low_confirm_row"][i]], tz="UTC")),
                             "next_high_delta_atr": float(result["high_delta_atr"][i]),
                             "next_low_delta_atr": float(result["low_delta_atr"][i])})
    return {"aggregate": summary, "private_examples": examples}


def _wall_timeout(_signum: int, _frame: object) -> None:
    raise TimeoutError("TA_M5_STRUCTURE_AUDIT_WALL_LIMIT")


def execute(plan_sha: str) -> None:
    import pandas as pd
    from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution

    plan, closure, parquet = preflight(plan_sha, admitted=True)
    cap = require_capped_cpu_audit_execution()
    need(cap["memory_max"] <= 4 * 1024**3
         and cap["swap"] <= 512 * 1024**2, "CAP")
    output = Path(plan["report"]["private_output_root"])
    output.mkdir()
    claim = {"stage": STAGE, "plan_sha256": plan_sha,
             "source_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=REPO, text=True).strip(),
             "claimed_utc": datetime.now(timezone.utc).isoformat()}
    (output / "CLAIM.json").write_text(json.dumps(claim, indent=2) + "\n")
    previous_alarm = signal.signal(signal.SIGALRM, _wall_timeout)
    signal.alarm(900)
    try:
        need(file_sha(Path(plan["input"]["path"])) == plan["input"]["sha256_expected_from_manifest"],
             "PARQUET_SHA")
        pieces = {key: [] for key in plan["scope"]["source_columns"]}
        for row_group in range(241, 253):
            table = parquet.read_row_group(row_group, columns=list(pieces))
            for key in pieces:
                column = table[key].to_pandas()
                values = pd.to_datetime(column, utc=True).array.asi8 if key == "time" else column.to_numpy()
                pieces[key].append(np.asarray(values))
        arrays = {key: np.concatenate(value) for key, value in pieces.items()}
        time = arrays["time"].astype(np.int64)
        cutoff = int(np.searchsorted(time, END))
        need(cutoff > 0 and cutoff < len(time), "FOLD_CROP")
        arrays = {key: value[:cutoff] for key, value in arrays.items()}
        time = arrays["time"].astype(np.int64)
        issues = np.flatnonzero(time >= START).astype(np.int64)
        intervals = np.array([
            [pd.Timestamp(item["closure_start_utc"]).value,
             pd.Timestamp(item["closure_end_utc_exclusive"]).value]
            for item in closure["intervals"] if item["successor_across_gap_allowed"] is True
        ], dtype=np.int64).reshape(-1, 2)
        measured = measure_candidate(time, arrays["high"], arrays["low"],
                                     arrays["close"], arrays["atr"], issues,
                                     fold_target_end_ns=END, known_closure_ns=intervals)
        report = aggregate(time, measured)
        known_breaks, unknown_breaks = _classify_clock_breaks(time, intervals)
        report["aggregate"]["known_closure_bridged_m5_breaks"] = int(known_breaks.sum())
        report["aggregate"]["unknown_m5_clock_breaks"] = int(unknown_breaks.sum())
        report.update({"decision": "SOURCE_COVERAGE_ONLY_EDGE_UNQUALIFIED",
                       "plan_sha256": plan_sha, "input_sha256": plan["input"]["sha256_expected_from_manifest"],
                       "model_forwards": 0, "optimizer_steps": 0, "test_data_used": False,
                       "target_locked": False, "economic_edge_proven": False})
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
        execute(args.plan_sha256)


if __name__ == "__main__":
    main()
