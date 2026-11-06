"""One-use full TRAIN rowwise audit of the PROPOSED five-TF endpoint state.

The existing target owner builds two fixed factor tapes. This audit publishes
allowed-class masks and source coverage, never a locked target or model score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time

REPO = Path("/home/andre2/src/GX1_CURRENT")
STAGE = "full_train_market_state_endpoint_candidate_20261107_001"
PLAN_SCHEMA = "gx1_full_train_market_state_endpoint_candidate_plan_v1"
CLAIM_SCHEMA = "gx1_full_train_market_state_endpoint_candidate_claim_v1"
BUDGET = {"memory_max": 4294967296, "swap": 536870912, "pids": 64,
          "numerical_threads": 1, "max_wall_seconds": 1800,
          "max_invocations": 1, "max_json_bytes": 16777216,
          "max_npz_bytes": 67108864}
TRAIN_START = "2024-05-01T00:00:00+00:00"
TRAIN_END = "2026-01-01T00:00:00+00:00"
M1_GROUPS = (1820, 2025)
M5_GROUPS = (0, 271)
M1_ROWS = 591682
LEADS = (5, 15, 60, 240, 1440)
FACTORS = (0.5, 1.0)
MINUTE_NS = 60_000_000_000
M5_NS = 5 * MINUTE_NS
M1_ROOT = Path("/home/andre2/GX1_DATA/data/data/prebuilt/HISTORY2009W_BOOTSTRAP_20260927/artifacts/pretest_direct_m1_quotes_source")
M5_ROOT = Path("/home/andre2/GX1_DATA/data/data/prebuilt/XAU_GC_NATIVE_26M_20261018_003")
CLOSURE_ROOT = Path("/home/andre2/GX1_RUNS/FULL_TRAIN_XAU_CLOSURE_AUTHORITY_20261107_001")
PRIOR_ROOT = Path("/home/andre2/GX1_RUNS/MARKET_STATE_ENDPOINT_CANDIDATE_ROWWISE_20261106_001")
FIXED_REFS = {
    "m1": {
        "source": {"path": str(M1_ROOT / "m1_quotes.parquet"),
                   "sha256": "54b66c5e9b5a0125f089afcc9d3327c79af9c3543eebdb2644281ed9d3c2734c"},
        "manifest": {"path": str(M1_ROOT / "manifest.json"),
                     "sha256": "42e2b20bbb3b1661c704352003461583eab488bba6dbacbd8404a134f060de1d"},
    },
    "m5": {
        "source": {"path": str(M5_ROOT / "M5_ENRICHED.parquet"),
                   "sha256": "5e5cb256fc60a5e9a7ffe1a161aa875660f01806b8e8de2bf601fc2f528e7f99"},
        "manifest": {"path": str(M5_ROOT / "M5_ENRICHED.parquet.manifest.json"),
                     "sha256": "d1c0f94bcfa7c837143b287fc8a4355e3ea564cffcbae2d96c53c64e7bb62de6"},
    },
    "closure": {
        "result": {"path": str(CLOSURE_ROOT / "RESULT.json"),
                   "sha256": "b58875df44811da3dafde11edbb89347ec49c74d830249f2ed0903a53b502d1c"},
        "terminal": {"path": str(CLOSURE_ROOT / "TERMINAL.json"),
                     "sha256": "9c2869f6dd1bf6f33d8a574e211edc8c13bf0d2b6e37f7307044efdef710f91c"},
        "source_reference": {"path": str(CLOSURE_ROOT / "SOURCE_REFERENCE.json"),
                             "sha256": "11e4199a8e0ea17ab1f9f602d15787aba990144fb8d0486bbabc51651354ee7e"},
        "exact_schedule": {"path": str(CLOSURE_ROOT / "exact_schedule.json"),
                           "sha256": "ddcefbc63404162b4f682179cf1277f9262227afc5da819116b7fe01835b1fc7"},
        "closure_authority": {"path": str(CLOSURE_ROOT / "closure_authority.json"),
                              "sha256": "4fcde066583cef98fd1fcd8a748c4544b4f11eb5be8fbf2705159fc3b1bfe978"},
    },
    "prior": {
        "result": {"path": str(PRIOR_ROOT / "RESULT.json"),
                   "sha256": "395e2604029e00b388f0fc97dfa18c58c1ca1f0c8abfe45c399a50af40caccdf"},
        "row_masks": {"path": str(PRIOR_ROOT / "ROW_MASKS.npz"),
                      "sha256": "5de51adef82cdec0bb717564ec5effc02d407f5972570b1ebfd1f8a93ad4a019"},
    },
}
OWNERS = {
    "worker": "gx1/audit/build_full_train_market_state_endpoint_candidate_v1.py",
    "target": "gx1/contracts/entry_dynamic_mtf_state_targets_v1.py",
    "rowwise_helper": "gx1/audit/verify_market_state_endpoint_candidate_v2.py",
    "closure_helper": "gx1/scripts/audit_dynamic_mtf_state_closure_observability_v1.py",
    "closure_owner": "gx1/contracts/unified_exit_market_closure_authority_v1.py",
    "publication_owner": "gx1/contracts/immutable_event_authority_v1.py",
    "capped_owner": "gx1/contracts/gx1_capped_execution_v1.py",
}
DOWN, FLAT, UP, ACTIVE, ALL = 1, 2, 4, 5, 7


def need(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError("FULL_TRAIN_ENDPOINT_" + code)


def regular(path: Path) -> tuple[int, ...]:
    need(path.is_absolute() and ".." not in path.parts
         and not any(part.is_symlink() for part in (path, *path.parents)), "PATH")
    s = path.stat()
    need(stat.S_ISREG(s.st_mode), "FILE")
    return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def checked(ref: dict, *, max_bytes: int | None = 16 * 1024 * 1024):
    need(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "REF_SHAPE")
    path = Path(ref["path"])
    before = regular(path)
    need(max_bytes is None or before[3] <= max_bytes, "REF_SIZE")
    digest = hashlib.sha256()
    chunks = [] if max_bytes is not None else None
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
    need(regular(path) == before and digest.hexdigest() == ref["sha256"], "REF_CHANGED")
    return (b"".join(chunks) if chunks is not None else None), before


def read_json(ref: dict) -> dict:
    raw, _ = checked(ref)
    need(raw is not None, "JSON")
    value = json.loads(raw)
    need(isinstance(value, dict), "JSON_SHAPE")
    return value


def ref(path: Path, raw: bytes) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def encode(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def month_edges() -> list[int]:
    edges = []
    for year in (2024, 2025):
        first = 5 if year == 2024 else 1
        for month in range(first, 13):
            edges.append(int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp()) * 10**9)
    edges.append(int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 10**9)
    need(len(edges) == 21, "MONTH_EDGES")
    return edges


def check_plan(plan: dict, plan_ref: dict, root: Path) -> None:
    need(set(plan) == {"schema_version", "policy_gate", "m1", "m5", "closure",
                       "prior", "train_start_utc", "train_end_utc_exclusive",
                       "m1_row_groups", "m5_row_groups", "m1_expected_rows",
                       "lead_minutes", "factors", "source_owners", "output_root",
                       "budget", "test_data_used"}, "PLAN_SHAPE")
    need(plan["schema_version"] == PLAN_SCHEMA and plan["policy_gate"] == STAGE
         and all(plan[k] == FIXED_REFS[k] for k in FIXED_REFS)
         and plan["train_start_utc"] == TRAIN_START
         and plan["train_end_utc_exclusive"] == TRAIN_END
         and plan["m1_row_groups"] == list(M1_GROUPS)
         and plan["m5_row_groups"] == list(M5_GROUPS)
         and plan["m1_expected_rows"] == M1_ROWS
         and plan["lead_minutes"] == list(LEADS)
         and plan["factors"] == list(FACTORS)
         and plan["budget"] == BUDGET
         and plan["output_root"] == str(root)
         and plan["test_data_used"] is False, "PLAN_BINDING")
    need(root.is_absolute() and root.parent == Path("/home/andre2/GX1_RUNS")
         and root.is_dir() and not root.is_symlink(), "ROOT")
    need(isinstance(plan["source_owners"], dict)
         and set(plan["source_owners"]) == set(OWNERS), "OWNERS")
    for name, relative in OWNERS.items():
        expected = REPO / relative
        need(plan["source_owners"][name].get("path") == str(expected), "OWNER_PATH")
        checked(plan["source_owners"][name])
    checked(plan_ref)


def check_claim(root: Path, plan_ref: dict, worker_ref: dict) -> tuple[dict, dict]:
    path = root / "CLAIM.json"
    before = regular(path)
    need(before[3] <= 16 * 1024 * 1024, "CLAIM_SIZE")
    raw = path.read_bytes()
    need(regular(path) == before, "CLAIM_CHANGED")
    claim = json.loads(raw)
    claim_ref = ref(path, raw)
    need(isinstance(claim, dict) and set(claim) ==
         {"schema_version", "plan", "worker", "policy", "source_head",
          "source_only_candidate", "test_data_used"}
         and claim["schema_version"] == CLAIM_SCHEMA
         and claim["plan"] == plan_ref and claim["worker"] == worker_ref
         and claim["source_only_candidate"] is True
         and claim["test_data_used"] is False
         and {item.name for item in root.iterdir()} == {"CLAIM.json"}, "CLAIM")
    return claim, claim_ref


def check_admission(plan: dict, plan_ref: dict, claim: dict) -> dict:
    need(Path.cwd() == REPO, "CWD")
    need(subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
         == "work/gx1-current", "BRANCH")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    need(claim["source_head"] == head
         and subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True) == "",
         "CLEAN_HEAD")
    policy_ref = claim["policy"]
    need(isinstance(policy_ref, dict)
         and policy_ref.get("path") == str(REPO / "NEXT_RUN_POLICY.json"), "POLICY_PATH")
    policy = read_json(policy_ref)
    gate = policy[STAGE]
    expected = {"decision": "ADMITTED", "plan": plan_ref, "execution_authorized": True,
                "data_reads_authorized": True, "target_builds_authorized": 2,
                "model_calls_authorized": 0, "test_authorized": False,
                "training_authorized": False, "claimed_invocations": 0,
                "completed_invocations": 0, "max_invocations": 1,
                "output_root": plan["output_root"]}
    need(policy["training_enabled"] is False and isinstance(gate, dict)
         and all(gate.get(k) == v for k, v in expected.items()), "POLICY")
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry,
    )
    limits = require_capped_cpu_audit_execution()
    limits["lock"] = require_capped_lock_ancestry()
    need(all(limits[k] == BUDGET[k] for k in
             ("memory_max", "swap", "pids", "numerical_threads")), "CAPS")
    return limits


def read_closure(plan: dict) -> tuple[dict, dict]:
    c = plan["closure"]
    result, terminal = read_json(c["result"]), read_json(c["terminal"])
    source, schedule, authority = (read_json(c[k]) for k in
                                   ("source_reference", "exact_schedule", "closure_authority"))
    need(result["decision"] == "PASS_SOURCE_GAP_AUTHORITY_ONLY"
         and terminal["status"] == "COMPLETE" and terminal["exit_code"] == 0
         and terminal["result"] == c["result"]
         and result["source_reference"] == c["source_reference"]
         and result["exact_schedule"] == c["exact_schedule"]
         and result["closure_authority"] == c["closure_authority"]
         and result["m1"] == plan["m1"] and result["m1_rows"] == M1_ROWS
         and result["observed_gap_count"] == authority["observed_gap_count"] == 508
         and result["known_market_closure_count"] == authority["known_market_closure_count"] == 413
         and result["unknown_source_gap_count"] == authority["unknown_source_gap_count"] == 95
         and source["m1_source"] == plan["m1"]["source"]
         and source["m1_manifest"] == plan["m1"]["manifest"]
         and source["m1_clock_sha256"] == authority["m1_clock_sha256"]
         and source["physical_source_row_start"] == 5193205
         and result["coverage_start_utc"] == TRAIN_START
         and result["coverage_end_utc_exclusive"] == TRAIN_END
         and not any(x["test_data_used"] for x in (result, terminal, source, schedule, authority)),
         "CLOSURE")
    return schedule, authority


def load_sources(plan: dict, authority: dict):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    m1m, m5m = read_json(plan["m1"]["manifest"]), read_json(plan["m5"]["manifest"])
    need(m1m["instrument"] == "XAU_USD" and m1m["timeframe"] == "M1"
         and m1m["timestamp_semantics"] == "bar_start_utc"
         and m1m["output_parquet"] == plan["m1"]["source"]["path"]
         and m1m["output_parquet_sha256"] == plan["m1"]["source"]["sha256"]
         and m1m["row_count"] == 5959045 and m1m["test_accessed"] is False
         and m1m["test_boundary_utc"] == "2026-07-01T00:00:00+00:00"
         and m5m["decision"] == "PASS" and m5m["timeframe"] == "M5"
         and m5m["base_bar_seconds"] == 300
         and m5m["output_parquet"] == plan["m5"]["source"]["path"]
         and m5m["output_parquet_sha256"] == plan["m5"]["source"]["sha256"]
         and m5m["rows"] == 1140982, "MANIFEST")
    _, m1_stat = checked(plan["m1"]["source"], max_bytes=None)
    _, m5_stat = checked(plan["m5"]["source"], max_bytes=None)
    m1p = pq.ParquetFile(plan["m1"]["source"]["path"])
    m5p = pq.ParquetFile(plan["m5"]["source"]["path"])
    m1meta, m5meta = m1p.metadata, m5p.metadata
    start, end = pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)
    need(m1meta.num_rows == m1m["row_count"] and m5meta.num_rows == m5m["rows"]
         and m1meta.num_row_groups > M1_GROUPS[1]
         and m5meta.num_row_groups > M5_GROUPS[1]
         and str(m1p.schema_arrow.field("time").type) == "timestamp[ns, tz=UTC]"
         and str(m5p.schema_arrow.field("time").type) == "timestamp[ns, tz=UTC]"
         and m1meta.row_group(m1meta.num_row_groups - 1).column(0).statistics.max
         < pd.Timestamp("2026-07-01T00:00:00+00:00")
         and m5meta.row_group(m5meta.num_row_groups - 1).column(0).statistics.max
         < pd.Timestamp("2026-07-01T00:00:00+00:00"), "PARQUET_HEADER")
    need(m1meta.row_group(M1_GROUPS[0]-1).column(0).statistics.max < start
         and m1meta.row_group(M1_GROUPS[1]).column(0).statistics.min >= end
         and m5meta.row_group(M5_GROUPS[1]-1).column(0).statistics.min < end
         <= m5meta.row_group(M5_GROUPS[1]-1).column(0).statistics.max
         and m5meta.row_group(M5_GROUPS[1]).column(0).statistics.min >= end,
         "GROUP_BOUNDARY")
    m1_indices = range(*M1_GROUPS)
    need(sum(m1meta.row_group(i).num_rows for i in m1_indices) == M1_ROWS
         and all(start <= m1meta.row_group(i).column(0).statistics.min
                 <= m1meta.row_group(i).column(0).statistics.max < end
                 for i in m1_indices), "M1_GROUPS")
    cols = ["time", "open", "high", "low", "close"]
    m1 = m1p.read_row_groups(list(m1_indices), columns=cols, use_threads=False).to_pandas()
    m5 = m5p.read_row_groups(list(range(*M5_GROUPS)), columns=cols, use_threads=False).to_pandas()
    m1time = pd.DatetimeIndex(m1["time"]).as_unit("ns")
    m5time = pd.DatetimeIndex(m5["time"]).as_unit("ns")
    need(len(m1) == M1_ROWS and m1time.is_monotonic_increasing and m1time.is_unique
         and m1time[0] == start and m1time[-1] == pd.Timestamp("2025-12-31T21:59:00+00:00")
         and hashlib.sha256(np.asarray(m1time.asi8, dtype="<i8").tobytes()).hexdigest()
         == authority["m1_clock_sha256"]
         and m5time.is_monotonic_increasing and m5time.is_unique
         and np.all(m5time.asi8 % M5_NS == 0), "SOURCE_CLOCK")
    # Row group 270 also holds Jan 2026 VAL prices. They were decoded by Arrow
    # but are removed before ATR construction, issue selection, or scoring.
    keep, issues, m5_source_rows = bound_m5_issue_clock(
        np.asarray(m5time.asi8, dtype=np.int64), start.value, end.value, np)
    need(int((~keep).sum()) > 0 and bool(keep.any()), "M5_BOUNDARY_ROWS")
    m5 = m5.loc[keep].reset_index(drop=True)
    m5time = m5time[keep]
    need(m5time[-1] < end and len(m5) > 0, "M5_PREFIX")
    need(len(issues) > 0 and np.all(np.diff(issues) > 0), "ISSUE_CLOCK")
    need(regular(Path(plan["m1"]["source"]["path"])) == m1_stat
         and regular(Path(plan["m5"]["source"]["path"])) == m5_stat, "SOURCE_STAT_CHANGED")
    return m1, m5, issues, m5_source_rows, source_stats(plan, m1_stat, m5_stat)


def source_stats(plan: dict, m1_stat: tuple, m5_stat: tuple) -> dict:
    return {plan["m1"]["source"]["path"]: m1_stat,
            plan["m5"]["source"]["path"]: m5_stat}


def bound_m5_issue_clock(m5_bar_starts, start_ns: int, end_ns: int, np):
    """Discard Jan boundary-group rows and exclude the prior April M5 bar."""
    keep = m5_bar_starts < end_ns
    bounded = m5_bar_starts[keep]
    issue_mask = (bounded >= start_ns) & (bounded + M5_NS < end_ns)
    return keep, bounded[issue_mask] + M5_NS, np.flatnonzero(issue_mask).astype(np.int64)


def summarize_partition(masks, present, counts: dict, np) -> dict:
    """A missing exact future endpoint cannot become an observed FLAT."""
    need(np.all(masks[~present] == ALL), "MISSING_TO_UNKNOWN")
    unknown_all = int(np.count_nonzero(masks == ALL))
    missing = int((~present).sum())
    residual = unknown_all - missing - counts["middle_unknown"]
    exact = counts["flat"] + counts["down"] + counts["up"]
    partial = counts["partial_complete"] + counts["partial_early"]
    need(residual >= 0 and exact + partial + unknown_all == len(masks), "MONTH_PARTITION")
    return {**counts, "exact_singleton": exact, "partial_active": partial,
            "unknown_all": unknown_all, "missing_exact_endpoint": missing,
            "other_unknown": residual}


def safe_prior_diagnostic(plan: dict, issues, allowed, m1_source_rows, axis, authority, np) -> dict:
    """Compare only shared, post-reset/full-D1-window observed endpoints."""
    from gx1.audit.verify_market_state_endpoint_candidate_v2 import exact_positions

    prior = read_json(plan["prior"]["result"])
    need(prior["row_masks"] == plan["prior"]["row_masks"]
         and prior["test_data_used"] is False and prior["target_locked"] is False, "PRIOR")
    old_semantic_plan = read_json(prior["inputs"]["semantic_plan"])
    old_target = read_json(old_semantic_plan["inputs"]["prior_result"])
    old_authority = read_json(old_semantic_plan["inputs"]["closure_authority"])
    need(old_target["target_sources"] == {"m1": plan["m1"]["source"],
                                           "m5_atr": plan["m5"]["source"]},
         "PRIOR_SOURCE_IDENTITY")
    raw, _ = checked(plan["prior"]["row_masks"], max_bytes=64 * 1024 * 1024)
    need(raw is not None, "PRIOR_NPZ")
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        need(set(z.files) == {"issue_ns", "lead_minutes", "allowed_set", "source_row"},
             "PRIOR_NPZ_KEYS")
        old_issues = np.asarray(z["issue_ns"], dtype=np.int64)
        old_allowed = np.asarray(z["allowed_set"], dtype=np.uint8)
        old_rows = np.asarray(z["source_row"], dtype=np.int64)
        old_leads = np.asarray(z["lead_minutes"], dtype=np.int16)
    need(old_allowed.shape == old_rows.shape == (29849, 5)
         and np.array_equal(old_leads, LEADS)
         and np.all(np.diff(old_issues) > 0), "PRIOR_SHAPE")
    first_unknown = next(item for item in authority["intervals"]
                         if not item["successor_across_gap_allowed"])
    old_first_unknown = next(item for item in old_authority["intervals"]
                             if not item["successor_across_gap_allowed"])
    need(old_first_unknown["next_bar_start_utc"] == first_unknown["next_bar_start_utc"],
         "PRIOR_RESET_IDENTITY")
    reset_start = pd_ns(first_unknown["next_bar_start_utc"])
    reset_row = int(np.searchsorted(axis, reset_start + MINUTE_NS))
    need(reset_row < len(axis) and axis[reset_row] == reset_start + MINUTE_NS,
         "PARITY_RESET_CLOCK")
    from gx1.contracts.entry_dynamic_mtf_state_targets_v1 import LOCAL_WINDOW_M1_BARS
    safe_row = reset_row + max(LOCAL_WINDOW_M1_BARS.values()) - 1
    need(safe_row < len(axis), "PARITY_WARMUP")
    safe_validity = int(axis[safe_row])
    # The old tape began on Apr 30, so labels before the shared unknown-gap
    # reset and full D1 window have different causal initialization.
    overlap = (old_issues >= safe_validity) & (old_issues < pd_ns("2024-10-01T00:00:00+00:00"))
    query = old_issues[overlap]
    pos = np.searchsorted(issues, query)
    valid = pos < len(issues)
    valid[valid] &= issues[pos[valid]] == query[valid]
    old_a = old_allowed[overlap][valid]
    old_r = old_rows[overlap][valid]
    new_a = allowed[pos[valid]]
    new_r = m1_source_rows[pos[valid]]
    old_physical_start = old_target["clock_slice_proof"]["row_start"]
    need(old_physical_start == 5191834, "PRIOR_PHYSICAL_OFFSET")
    comparable = (old_r >= 0) & (new_r >= 0) & (old_r + old_physical_start == new_r)
    compared = int(comparable.sum())
    mismatch = int(np.count_nonzero(old_a[comparable] != new_a[comparable]))
    return {
        "role": "diagnostic_only_not_target_acceptance",
        "old_source_start_utc": "2024-04-30T00:05:00+00:00",
        "new_source_start_utc": TRAIN_START,
        "shared_unknown_reset_utc": first_unknown["next_bar_start_utc"],
        "safe_comparison_validity_utc": datetime.fromtimestamp(safe_validity / 1e9, timezone.utc).isoformat(),
        "prior_issue_rows_in_safe_window": int(overlap.sum()),
        "matching_issue_rows": int(valid.sum()),
        "exact_physical_endpoint_cells_compared": compared,
        "allowed_set_mismatches": mismatch,
        "excluded_missing_or_physical_mismatch_cells": int(comparable.size - compared),
        "limit": "Shared source refs and first new unknown reset plus 1440 observed M1 bars define a conservative diagnostic window. Parity does not prove all May-Sep rows because the old tape had Apr30 warmup.",
    }


def pd_ns(value: str) -> int:
    import pandas as pd
    return int(pd.Timestamp(value).value)


def run(plan_path: Path, plan_sha256: str, root: Path) -> dict:
    import numpy as np
    from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
    from gx1.audit.verify_market_state_endpoint_candidate_v2 import (
        classify_rows, exact_positions,
    )
    from gx1.scripts.audit_dynamic_mtf_state_closure_observability_v1 import early_active_proof
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

    plan_ref = {"path": str(plan_path), "sha256": plan_sha256}
    plan = read_json(plan_ref)
    check_plan(plan, plan_ref, root)
    claim, claim_ref = check_claim(root, plan_ref, plan["source_owners"]["worker"])
    limits = check_admission(plan, plan_ref, claim)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("FULL_TRAIN_ENDPOINT_DEADLINE")))
    signal.alarm(BUDGET["max_wall_seconds"])
    started = time.monotonic()
    try:
        schedule, authority = read_closure(plan)
        m1, m5, issues, m5_source_rows, stats = load_sources(plan, authority)
        cutoff = pd_ns(TRAIN_END)
        need(target.MINIMUM_MOVE_ATR == FACTORS[0]
             and tuple(target.TIMEFRAMES) == ("M5", "M15", "H1", "H4", "D1"), "TARGET")
        atr = target.build_causal_tf_atr(m5, cutoff_ns=cutoff)
        a05_series = target.build_dynamic_mtf_state_series(
            m1, tf_atr=atr, source_ref=plan["m1"]["source"],
            atr_source_ref=plan["m5"]["source"], cutoff_ns=cutoff,
            source_row_offset=5193205, closure_authority=authority,
            exact_schedule=schedule)
        target.MINIMUM_MOVE_ATR = FACTORS[1]
        try:
            atr["definition_sha256"] = target.dynamic_mtf_state_target_contract()["contract_sha256"]
            a10_series = target.build_dynamic_mtf_state_series(
                m1, tf_atr=atr, source_ref=plan["m1"]["source"],
                atr_source_ref=plan["m5"]["source"], cutoff_ns=cutoff,
                source_row_offset=5193205, closure_authority=authority,
                exact_schedule=schedule)
        finally:
            target.MINIMUM_MOVE_ATR = FACTORS[0]
        a05, a10 = a05_series.arrays, a10_series.arrays
        axis = np.asarray(a05["validity_ns"], dtype=np.int64)
        need(np.array_equal(axis, a10["validity_ns"])
             and len(axis) == M1_ROWS
             and np.array_equal(a05["source_row"], np.arange(5193205, 5193205 + M1_ROWS)),
             "TAPE_CLOCK")
        m1_times = target._clock(m1, target.EXIT_DECISION_BAR_SECONDS)
        known, _ = target._known_closures(
            m1_times, plan["m1"]["source"], authority, schedule)
        unknown = np.zeros(len(axis), bool)
        for row in np.flatnonzero(np.diff(axis) != MINUTE_NS) + 1:
            unknown[row] = int(row) not in known
        need(int(unknown.sum()) == authority["unknown_source_gap_count"], "UNKNOWN_GAP_PARITY")
        highs, lows = m1["high"].to_numpy(np.float64), m1["low"].to_numpy(np.float64)
        early = {}
        for col, tf in enumerate(target.TIMEFRAMES):
            early[tf], _ = early_active_proof(
                highs, lows, a10["threshold"][:, col], unknown,
                target.LOCAL_WINDOW_M1_BARS[tf], np)
        edges = month_edges()
        allowed = np.full((len(issues), len(LEADS)), ALL, np.uint8)
        source_rows = np.full(allowed.shape, -1, np.int64)
        monthly = {}
        for start, stop in zip(edges[:-1], edges[1:]):
            lo, hi = np.searchsorted(issues, [start, stop])
            need(hi > lo, "MONTH_ISSUES")
            month = {}
            for col, tf in enumerate(target.TIMEFRAMES):
                pos, present = exact_positions(axis, issues[lo:hi], LEADS[col], cutoff, np)
                rows = pos[present]
                mask, counts = classify_rows(a05, a10, early[tf], rows, col, target, np)
                allowed[lo:hi, col][present] = mask
                source_rows[lo:hi, col][present] = a05["source_row"][rows]
                summary = summarize_partition(allowed[lo:hi, col], present, counts, np)
                month[tf] = {"issues": int(hi - lo),
                             "exact_m1_endpoint": int(present.sum()), **summary}
            monthly[datetime.fromtimestamp(start / 1e9, timezone.utc).strftime("%Y-%m")] = month
        need(len(monthly) == 20 and all(len(v) == 5 for v in monthly.values()), "MONTH_COUNT")
        totals = {tf: {key: sum(month[tf][key] for month in monthly.values())
                       for key in ("issues", "exact_singleton", "partial_active",
                                   "unknown_all", "missing_exact_endpoint",
                                   "middle_unknown", "other_unknown")}
                  for tf in target.TIMEFRAMES}
        need(all(totals[tf]["issues"] == len(issues) and
                 totals[tf]["exact_singleton"] + totals[tf]["partial_active"]
                 + totals[tf]["unknown_all"] == len(issues)
                 for tf in target.TIMEFRAMES), "TOTALS")
        prior = safe_prior_diagnostic(plan, issues, allowed, source_rows, axis, authority, np)
        need(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
             == claim["source_head"]
             and subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True) == "",
             "HEAD_CHANGED")
        need(regular(Path(plan["m1"]["source"]["path"])) == stats[plan["m1"]["source"]["path"]]
             and regular(Path(plan["m5"]["source"]["path"])) == stats[plan["m5"]["source"]["path"]],
             "SOURCE_CHANGED")
        check_plan(plan, plan_ref, root)
        need(check_claim(root, plan_ref, plan["source_owners"]["worker"]) == (claim, claim_ref)
             and check_admission(plan, plan_ref, claim) == limits, "BOOKEND")
        buf = io.BytesIO()
        np.savez_compressed(buf, issue_ns=issues, m5_source_row=m5_source_rows,
                            lead_minutes=np.asarray(LEADS, np.int16),
                            allowed_set=allowed, m1_source_row=source_rows)
        npz_bytes = buf.getvalue()
        need(len(npz_bytes) <= BUDGET["max_npz_bytes"], "NPZ_BUDGET")
        row_path = root / "ROW_MASKS.npz"
        row_ref = ref(row_path, npz_bytes)
        output = {
            "schema_version": STAGE + "_result_v1",
            "decision": "PASS_FULL_TRAIN_CANDIDATE_SOURCE_ONLY",
            "plan": plan_ref, "claim": claim_ref, "policy": claim["policy"],
            "source_head": claim["source_head"], "limits": limits,
            "source_refs": {k: plan[k] for k in FIXED_REFS},
            "source_read_scope": "Both pre-TEST parquet files are hashed as opaque integrity bytes, including VAL bytes. M1 decodes only TRAIN row groups 1820..2024. Arrow decodes M5 boundary group 270 including Jan2026 VAL prices; rows at/after Jan1 are discarded before ATR, issue clocks, labels and reports. No TEST data.",
            "train_start_utc": TRAIN_START, "train_end_utc_exclusive": TRAIN_END,
            "m1_rows": len(axis), "m5_issues": len(issues),
            "m1_clock_sha256": authority["m1_clock_sha256"],
            "known_market_closure_count": authority["known_market_closure_count"],
            "unknown_source_gap_count": authority["unknown_source_gap_count"],
            "allowed_bits": {"DOWN": DOWN, "FLAT": FLAT, "UP": UP,
                             "ACTIVE_SIDE_UNKNOWN": ACTIVE, "ALL_UNKNOWN": ALL},
            "monthly": monthly, "totals": totals, "prior_safe_overlap": prior,
            "row_masks": row_ref, "elapsed_seconds": time.monotonic() - started,
            "test_data_used": False, "model_calls": 0, "optimizer_steps": 0,
            "target_locked": False,
            "limit": "Coverage and allowed labels only. Overlapping M5 forecasts are not independent; no calibrated forecast, learned direction, economic edge or target lock.",
        }
        result_bytes = encode(output)
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
    print(json.dumps(run(args.plan, args.plan_sha256, args.out_dir), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
