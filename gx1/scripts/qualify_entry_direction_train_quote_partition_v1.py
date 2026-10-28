"""Bounded TRAIN-year quote join and saved 1/5/15 midpoint parity audit.

No model or fit lives here. An actual invocation additionally requires the
specific separately admitted one-shot plan and capped worker. The direct
pretest quote parquet and every 2026 source partition remain out of scope.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.contracts.entry_direction_price_projection_v1 import compare_direction_native_midpoint

MINUTE_NS = 60_000_000_000
HORIZONS = (1, 5, 15)
SAVED_MEMBERS = (
    "parent_rows", "bar_start_ns", "dynamic_entry_m1_row", "decision_ns",
    "terminal_time_ns", "elapsed_wall_seconds", "terminal_source_row", "observed",
    "mid_move", "mid_return_bps", "sign",
)
TRAIN_START_NS = int(pd.Timestamp("2024-05-01T00:00:00Z").value)
TRAIN_END_NS = int(pd.Timestamp("2026-01-01T00:00:00Z").value)
PARENT_COUNT = 118249
YEAR_ROOT = "/home/andre2/GX1_DATA/data/native_xau/XAU_M1_NATIVE_2009_20260701_PAIR_WEEKCLOSED_20260927"
TARGET_REF = {
    "path": "/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_BASELINE_DATA_20261024_245/OUTPUT/PRICE_TARGETS.npz",
    "sha256": "9d4a5fb3c473b088aff5a5d5a176c199a64053b679229c14f125697b93af577d",
}
YEAR_REFS = (
    {"year": 2024, "path": YEAR_ROOT + "/year=2024/part-000.parquet",
     "sha256": "2776a1cb17b915d506eaf0d7ed5323ba975e0d0d1c52c0c02d2da20589d1c41a",
     "rows": 354866, "offset": 5076522,
     "first": "2024-01-01T23:03:00+00:00", "last": "2024-12-31T21:59:00+00:00"},
    {"year": 2025, "path": YEAR_ROOT + "/year=2025/part-000.parquet",
     "sha256": "1746d0c3c1bd13fa3d19c628fccea05dfe4715e681fcba9aa1b5a559961308dc",
     "rows": 353499, "offset": 5431388,
     "first": "2025-01-01T23:04:00+00:00", "last": "2025-12-31T21:59:00+00:00"},
)
ACTUAL_STAGE = "direction_train_year_quote_parity_actual_20261028_001"


def _require(ok, reason):
    if not ok:
        raise RuntimeError("TRAIN_QUOTE_PARITY_" + reason)


def empty_quote_buffers(parent_count):
    _require(type(parent_count) is int and parent_count > 0, "PARENT_COUNT")
    def block(shape):
        return {"time_ns": np.zeros(shape, np.int64),
                "source_row": np.full(shape, -1, np.int64),
                "observed": np.zeros(shape, np.bool_),
                "open": np.zeros(shape, np.float64),
                "bid_open": np.zeros(shape, np.float64),
                "ask_open": np.zeros(shape, np.float64)}
    return {"entry": block((parent_count,)), "terminal": block((parent_count, 3))}


def add_year_partition(saved, buffers, *, year, times_ns, prices, row_offset, cutoff_ns):
    """Join only exact entry/terminal timestamps in this one physical year."""
    _require(type(year) is int and year in (2024, 2025)
             and type(row_offset) is int and row_offset >= 0
             and type(cutoff_ns) is int and cutoff_ns <= TRAIN_END_NS, "YEAR_SCOPE")
    _require(isinstance(times_ns, np.ndarray) and times_ns.dtype == np.int64
             and times_ns.ndim == 1 and len(times_ns) > 0
             and np.all(np.diff(times_ns) > 0) and np.all(times_ns % MINUTE_NS == 0), "YEAR_CLOCK")
    first = int(pd.Timestamp(f"{year}-01-01T00:00:00Z").value)
    end = int(pd.Timestamp(f"{year + 1}-01-01T00:00:00Z").value)
    _require(np.all((times_ns >= first) & (times_ns < end))
             and np.all(times_ns < cutoff_ns), "YEAR_BOUNDARY")
    _require(isinstance(prices, dict) and set(prices) == {"open", "bid_open", "ask_open"}, "PRICE_KEYS")
    for value in prices.values():
        _require(isinstance(value, np.ndarray) and value.dtype == np.float64
                 and value.shape == times_ns.shape and np.isfinite(value).all()
                 and np.all(value > 0), "YEAR_PRICES")
    _require(np.all(prices["ask_open"] >= prices["bid_open"]), "CROSSED_QUOTES")
    for key, axis in (("entry", saved["decision_ns"]), ("terminal", saved["terminal_time_ns"])):
        dest = buffers[key]
        _require(isinstance(axis, np.ndarray) and axis.dtype == np.int64
                 and axis.shape == dest["observed"].shape, "TARGET_AXIS")
        flat = axis.reshape(-1)
        wanted = np.flatnonzero((flat >= first) & (flat < end) & (flat < cutoff_ns))
        if not len(wanted):
            continue
        positions = np.searchsorted(times_ns, flat[wanted], side="left")
        present = np.flatnonzero(positions < len(times_ns))
        exact = present[times_ns[positions[present]] == flat[wanted[present]]]
        out, source = wanted[exact], positions[exact]
        _require(not dest["observed"].reshape(-1)[out].any(), "DUPLICATE_QUOTE")
        dest["observed"].reshape(-1)[out] = True
        dest["time_ns"].reshape(-1)[out] = times_ns[source]
        dest["source_row"].reshape(-1)[out] = source + row_offset
        for field in ("open", "bid_open", "ask_open"):
            dest[field].reshape(-1)[out] = prices[field][source]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def _verified_payload(ref):
    path = Path(ref["path"])
    _require(path.is_absolute() and not path.is_symlink() and path.resolve(strict=True) == path,
             "PAYLOAD_PATH")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno())
        digest = hashlib.sha256()
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
        _require(digest.hexdigest() == ref["sha256"], "PAYLOAD_SHA")
        handle.seek(0)
        yield handle
        after = os.fstat(handle.fileno())
        _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                 == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "PAYLOAD_CHANGED")


def _validate_saved(saved):
    _require(set(saved) == set(SAVED_MEMBERS), "SAVED_MEMBERS")
    parents = saved["parent_rows"]
    _require(isinstance(parents, np.ndarray) and parents.dtype == np.int64
             and parents.shape == (PARENT_COUNT,)
             and np.array_equal(parents, np.arange(PARENT_COUNT, dtype=np.int64)), "SAVED_PARENT_IDS")
    decisions = saved["decision_ns"]
    _require(decisions.dtype == np.int64 and decisions.shape == parents.shape
             and np.all((decisions >= TRAIN_START_NS) & (decisions < TRAIN_END_NS))
             and np.all(np.diff(decisions) > 0)
             and np.array_equal(decisions, saved["bar_start_ns"] + 5 * MINUTE_NS), "SAVED_DECISIONS")
    terminal = saved["terminal_time_ns"]
    _require(terminal.dtype == np.int64 and terminal.shape == (PARENT_COUNT, 3)
             and np.array_equal(terminal, decisions[:, None] + np.asarray(HORIZONS, np.int64) * MINUTE_NS)
             and saved["observed"].dtype == np.bool_ and saved["observed"].shape == terminal.shape
             and np.all(terminal[saved["observed"]] < TRAIN_END_NS), "SAVED_TERMINAL_SCOPE")


def _bound_plan(path):
    plan_path = Path(path)
    _require(plan_path.is_absolute() and not plan_path.is_symlink(), "PLAN_PATH")
    plan = json.loads(plan_path.read_text())
    policy_path = Path("NEXT_RUN_POLICY.json")
    policy = json.loads(policy_path.read_text())
    stage = policy.get(ACTUAL_STAGE, {})
    plan_sha = _sha(plan_path)
    _require(stage.get("execution_authorized") is True and stage.get("claimed_invocations") == 0
             and stage.get("max_invocations") == 1 and stage.get("relaunch_allowed") is False
             and stage.get("plan") == {"path": str(plan_path), "sha256": plan_sha}, "POLICY")
    _require(plan.get("schema_version") == "gx1_train_year_quote_parity_plan_v1"
             and plan.get("target") == TARGET_REF and plan.get("years") == list(YEAR_REFS)
             and plan.get("parent_count") == PARENT_COUNT and plan.get("cutoff_ns") == TRAIN_END_NS
             and plan.get("test_control_exit_authorized") is False, "PLAN_SCOPE")
    output = Path(plan["output_dir"])
    _require(output.is_absolute() and output.is_dir() and not output.is_symlink(), "OUTPUT_DIR")
    claim = json.loads((output / "INVOCATION_CLAIM.json").read_text())
    _require(claim.get("plan_sha256") == plan_sha and claim.get("policy_sha256") == _sha(policy_path)
             and claim.get("invocation") == 1, "CLAIM")
    for owner in plan["source_owners"]:
        _require(_sha(owner["path"]) == owner["sha256"], "SOURCE_DRIFT")
    return plan, output


def run(plan_path):
    import pyarrow.parquet as pq
    plan, output = _bound_plan(plan_path)
    with _verified_payload(TARGET_REF) as handle:
        with np.load(handle, allow_pickle=False) as archive:
            _require(archive.files == list(SAVED_MEMBERS), "ARCHIVE_MEMBERS")
            saved = {key: archive[key] for key in SAVED_MEMBERS}
    _validate_saved(saved)
    buffers = empty_quote_buffers(PARENT_COUNT)
    year_reports = []
    for ref in YEAR_REFS:
        with _verified_payload(ref) as handle:
            parquet = pq.ParquetFile(handle)
            clocks = pd.DatetimeIndex(pd.to_datetime(
                parquet.read(columns=["time"]).column("time").to_pandas(), utc=True)).as_unit("ns")
            times = np.asarray(clocks.asi8, dtype=np.int64)
            _require(len(times) == ref["rows"] and len(times) > 0
                     and times[0] == pd.Timestamp(ref["first"]).value
                     and times[-1] == pd.Timestamp(ref["last"]).value
                     and np.all(times < TRAIN_END_NS), "YEAR_METADATA")
            table = parquet.read(columns=["open", "bid_open", "ask_open"])
            prices = {key: np.asarray(table.column(key).combine_chunks().to_numpy(), dtype=np.float64)
                      for key in ("open", "bid_open", "ask_open")}
            add_year_partition(saved, buffers, year=ref["year"], times_ns=times,
                               prices=prices, row_offset=ref["offset"], cutoff_ns=TRAIN_END_NS)
        year_reports.append({"year": ref["year"], "rows": len(times), "file_sha256_verified": True})
    report = compare_direction_native_midpoint(saved, entry_quotes=buffers["entry"],
                                               terminal_quotes=buffers["terminal"], cutoff_ns=TRAIN_END_NS)
    result = {"schema_version": "gx1_train_year_quote_parity_result_v1",
              "status": "COMPLETE_BIT_PARITY", "parent_rows": PARENT_COUNT,
              "years": year_reports, "legacy_bit_parity": report["legacy_bit_parity"],
              "per_horizon": report["per_horizon"],
              "cutoff_ns_exclusive": TRAIN_END_NS, "test_control_accessed": False,
              "combined_pretest_quote_opened": False, "model_forwards": 0,
              "optimizer_steps": 0, "fits": 0,
              "limits": ["Source/endpoint parity only; no first-hurdle target, learning or trading edge measured."]}
    for owner in plan["source_owners"]:
        _require(_sha(owner["path"]) == owner["sha256"], "SOURCE_DRIFT_AFTER")
    _require(_sha("NEXT_RUN_POLICY.json") == json.loads((output / "INVOCATION_CLAIM.json").read_text())["policy_sha256"],
             "POLICY_DRIFT_AFTER")
    with (output / "RESULT.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan), indent=2))


if __name__ == "__main__":
    main()
