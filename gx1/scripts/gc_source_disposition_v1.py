"""Run the frozen, local-only plan-003 source-disposition audit."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from gx1.contracts import gc_source_disposition_v1 as contract
from gx1.contracts.gc_order_flow_source_v1 import require_gc_file_declaration
from gx1.scripts.research_ta_campaign_v1 import _gc_bound_local_path
from gx1.contracts.gx1_capped_execution_v1 import (
    _require_capped_cgroup_limits,
    require_capped_lock_ancestry,
)

REPO = Path("/home/andre2/src/GX1_CURRENT")
PLAN_REL = "configs/research/GC_SOURCE_DISPOSITION_20261016_003.json"
PLAN_SHA256 = "7b3c5614307d7849edd7694f6b5297dd1dfb8b66cab400e590bd1d089f5d5082"
APPROVAL_SHA256 = "6bd49d90767016c3baa7237f23e9e13a02dff06489543071d1630013d9769406"
GC_PLAN_REL = "configs/research/GC_SOURCE_AUDIT_20261016_002.json"
GC_PLAN_SHA256 = "3cca1eef4bd965e0194a5f353f76645fbb6a030c509d0f3d24850745f4d0310d"
RUN_ROOT = Path("/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001/GC_SOURCE_DISPOSITION_20261016_003")
EXPECTED_APPROVAL = RUN_ROOT / "OPERATOR_APPROVAL.json"
EXPECTED_BINDING = RUN_ROOT / "EXECUTION_BINDING.json"
EXPECTED_CLAIM = RUN_ROOT / "EXECUTION_CLAIM.json"
RESULTS_DIR = RUN_ROOT / "AUDIT_RESULTS"
GC_ROOT = Path("/home/andre2/GX1_DATA/research/GC_ORDER_FLOW_RESEARCH_001")
WORKER_COUNT = 8
WORKER_OANDA_BAR_INDEX: dict[int, int] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("GC_DISPOSITION_JSON_OBJECT_REQUIRED")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def require_regular_file(path: Path, tag: str) -> None:
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
        raise RuntimeError(f"{tag}_PATH_NOT_CANONICAL")


def verify_digest(path: Path, expected: str, tag: str) -> None:
    require_regular_file(path, tag)
    if sha256_file(path) != expected:
        raise RuntimeError(f"{tag}_SHA256_MISMATCH")


def require_runtime(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if args.plan.resolve(strict=True) != (REPO / PLAN_REL) or args.plan.is_symlink():
        raise RuntimeError("GC_DISPOSITION_PLAN_PATH_INVALID")
    if args.approval.resolve(strict=True) != EXPECTED_APPROVAL or args.approval.is_symlink():
        raise RuntimeError("GC_DISPOSITION_APPROVAL_PATH_INVALID")
    if args.binding.resolve(strict=True) != EXPECTED_BINDING or args.binding.is_symlink():
        raise RuntimeError("GC_DISPOSITION_BINDING_PATH_INVALID")
    verify_digest(args.plan, PLAN_SHA256, "PLAN")
    plan = read_json(args.plan)
    if plan.get("plan_id") != "GC_SOURCE_DISPOSITION_20261016_003":
        raise RuntimeError("GC_DISPOSITION_PLAN_ID_INVALID")
    if plan.get("authorization", {}).get("authorized") is not False:
        raise RuntimeError("GC_DISPOSITION_FROZEN_PLAN_WAS_MUTATED")
    verify_digest(args.approval, APPROVAL_SHA256, "APPROVAL")
    approval = read_json(args.approval)
    binding = read_json(args.binding)
    require_regular_file(EXPECTED_CLAIM, "EXECUTION_CLAIM")
    claim = read_json(EXPECTED_CLAIM)
    binding_sha = sha256_file(args.binding)
    if (approval.get("status") != "APPROVED_SINGLE_USE_PENDING_EXECUTION"
            or approval.get("approved_plan_sha256") != PLAN_SHA256
            or approval.get("single_use") is not True
            or claim.get("status") != "CLAIMED"
            or claim.get("invocation") != 1
            or claim.get("execution_binding_sha256") != binding_sha
            or binding.get("approved_plan_sha256") != PLAN_SHA256
            or binding.get("approval_receipt_sha256") != sha256_file(args.approval)):
        raise RuntimeError("GC_DISPOSITION_SINGLE_USE_BINDING_INVALID")
    policy_path = REPO / "NEXT_RUN_POLICY.json"
    policy = read_json(policy_path)
    if binding.get("policy_sha256") != sha256_file(policy_path):
        raise RuntimeError("GC_DISPOSITION_POLICY_HASH_MISMATCH")
    auth = policy["current_work"]["gc_goal_resume_authorization"]
    policy_plan = policy["gc_order_flow_research_20261006"]["source_disposition_plan_003"]
    recorded_approval = auth.get("new_one_off_audit_approval", {})
    if (auth.get("new_one_off_audit_plan_execution_allowed") is not False
            or auth.get("new_one_off_audit_plan_execution_consumed") is not True
            or recorded_approval.get("approved_plan_sha256") != PLAN_SHA256
            or recorded_approval.get("approval_receipt_sha256") != sha256_file(args.approval)
            or policy_plan.get("status") != "EXECUTION_CLAIMED"
            or policy_plan.get("execution_authorized") is not False
            or policy_plan.get("claimed_invocations") != 1
            or policy_plan.get("completed_invocations") != 0):
        raise RuntimeError("GC_DISPOSITION_POLICY_CLAIM_INVALID")
    if binding.get("source_head") != git_head():
        raise RuntimeError("GC_DISPOSITION_CODE_HEAD_MISMATCH")
    expected_limits = {
        "memory_max_and_high_bytes": 4 * 1024**3,
        "swap_max_bytes": 512 * 1024**2,
        "cpu_affinity": list(range(8)),
        "tasks_max": 64,
        "worker_processes": WORKER_COUNT,
        "numerical_threads_per_worker": 1,
        "runner": "scripts/gx1_capped_run.sh --class audit",
    }
    if binding.get("limits") != expected_limits:
        raise RuntimeError("GC_DISPOSITION_LIMIT_BINDING_INVALID")
    for relative in (
        "gx1/contracts/gc_source_disposition_v1.py",
        "gx1/scripts/gc_source_disposition_v1.py",
        "tests/test_gc_source_disposition_v1.py",
    ):
        path = REPO / relative
        if sha256_file(path) != binding.get("code_sha256", {}).get(relative):
            raise RuntimeError("GC_DISPOSITION_CODE_HASH_MISMATCH")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout
    if status:
        raise RuntimeError("GC_DISPOSITION_WORKTREE_NOT_CLEAN")
    if os.environ.get("GX1_CAPPED_CLASS") != "audit":
        raise RuntimeError("GC_DISPOSITION_AUDIT_CGROUP_REQUIRED")
    if not hasattr(os, "sched_getaffinity") or os.sched_getaffinity(0) != set(range(8)):
        raise RuntimeError("GC_DISPOSITION_CPU_AFFINITY_INVALID")
    cap = _require_capped_cgroup_limits(
        environ=os.environ, read_text=None, max_memory_bytes=4 * 1024**3,
        error_prefix="GC_DISPOSITION", max_pids=64,
    )
    cap["lock"] = require_capped_lock_ancestry()
    if RESULTS_DIR.exists() or (RUN_ROOT / "POST_AUDIT_RECONCILIATION.json").exists():
        raise FileExistsError("GC_DISPOSITION_OUTPUT_ALREADY_EXISTS")
    return plan, approval, binding


def _bound_gc_inputs(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = plan["inputs"]
    gc_plan_path = REPO / GC_PLAN_REL
    verify_digest(gc_plan_path, GC_PLAN_SHA256, "GC_PLAN")
    gc_plan = read_json(gc_plan_path)
    if (source["gc_plan"].get("path") != GC_PLAN_REL
            or source["gc_plan"].get("sha256") != GC_PLAN_SHA256
            or gc_plan.get("research_id") != "GC_ORDER_FLOW_RESEARCH_001"
            or gc_plan.get("source_audit_schema") != "gx1_gc_order_flow_source_audit_v2"
            or len(gc_plan.get("files", [])) != 66):
        raise RuntimeError("GC_DISPOSITION_GC_PLAN_INVALID")
    verify_digest(Path(source["gc_result"]["path"]), source["gc_result"]["sha256"], "GC_RESULT")
    verify_digest(Path(source["gc_reconciliation"]["path"]), source["gc_reconciliation"]["sha256"], "GC_RECONCILIATION")
    source_root = Path(gc_plan["source_root"])
    if source_root != GC_ROOT or source_root.is_symlink() or source_root.resolve(strict=True) != source_root:
        raise RuntimeError("GC_DISPOSITION_GC_ROOT_INVALID")
    groups: dict[str, list[dict[str, Any]]] = {}
    all_items = []
    for raw in gc_plan["files"]:
        item = require_gc_file_declaration(raw)
        path = _gc_bound_local_path(item["path"], source_root)
        receipt = _gc_bound_local_path(item["receipt_path"], source_root)
        if path.stat().st_size != item["bytes"]:
            raise RuntimeError("GC_DISPOSITION_GC_FILE_SIZE_MISMATCH")
        if sha256_file(path) != item["sha256"] or sha256_file(receipt) != item["receipt_sha256"]:
            raise RuntimeError("GC_DISPOSITION_GC_FILE_OR_RECEIPT_HASH_MISMATCH")
        item["path"] = str(path)
        item["receipt_path"] = str(receipt)
        groups.setdefault(item["raw_symbol"], []).append(item)
        all_items.append((path, item["sha256"], receipt, item["receipt_sha256"]))
    if len(groups) != 15:
        raise RuntimeError("GC_DISPOSITION_EXPECTED_15_OUTRIGHTS")
    for items in groups.values():
        ordered = sorted(items, key=lambda item: int(item["start_ns"]))
        previous_end = None
        for item in ordered:
            if previous_end is not None and int(item["start_ns"]) < previous_end:
                raise RuntimeError("GC_DISPOSITION_DECLARED_WINDOWS_OVERLAP")
            previous_end = int(item["end_ns"])
    ancillary_hashes = [
        (gc_plan_path, GC_PLAN_SHA256),
        (Path(source["gc_result"]["path"]), source["gc_result"]["sha256"]),
        (Path(source["gc_reconciliation"]["path"]), source["gc_reconciliation"]["sha256"]),
    ]
    return [
        {"raw_symbol": symbol, "items": sorted(items, key=lambda item: int(item["start_ns"]))}
        for symbol, items in sorted(groups.items())
    ], {
        "gc_plan_path": str(gc_plan_path),
        "gc_plan_sha256": GC_PLAN_SHA256,
        "gc_result_sha256": source["gc_result"]["sha256"],
        "gc_reconciliation_sha256": source["gc_reconciliation"]["sha256"],
        "gc_source_files": all_items,
        "gc_ancillary_files": ancillary_hashes,
    }


def _load_oanda(plan: dict[str, Any]) -> tuple[dict[int, int], dict[str, Any], list[tuple[Path, str]]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = plan["inputs"]["oanda_manifest"]
    manifest_path = Path(source["path"])
    verify_digest(manifest_path, source["sha256"], "OANDA_MANIFEST")
    if len(source["partitions"]) != 3:
        raise RuntimeError("GC_DISPOSITION_OANDA_PARTITION_LIST_INVALID")
    partition_hashes = [(Path(item["path"]), item["sha256"]) for item in source["partitions"]]
    for path, expected in partition_hashes:
        verify_digest(path, expected, "OANDA_PARTITION")
    index: dict[int, int] = {}
    partition_reports = []
    previous = None
    total_rows = support_rows = study_rows = boundary_excluded = 0
    for item in source["partitions"]:
        path = Path(item["path"])
        parquet = pq.ParquetFile(path)
        field = parquet.schema_arrow.field("time")
        if str(field.type) != "timestamp[ns, tz=UTC]":
            raise RuntimeError("GC_DISPOSITION_OANDA_TIME_SCHEMA_INVALID")
        partition_rows = 0
        for batch in parquet.iter_batches(columns=["time"], batch_size=262144, use_threads=False):
            raw_times = batch.column(0).cast(pa.int64()).to_pylist()
            for timestamp_ns in raw_times:
                if previous is not None and timestamp_ns <= previous:
                    raise RuntimeError("GC_DISPOSITION_OANDA_TIME_NOT_STRICTLY_SORTED")
                if timestamp_ns % contract.MINUTE_NS:
                    raise RuntimeError("GC_DISPOSITION_OANDA_TIME_NOT_M1_ALIGNED")
                previous = timestamp_ns
                partition_rows += 1
                total_rows += 1
                if contract.SUPPORT_START_NS <= timestamp_ns < contract.WINDOW_END_NS:
                    index[timestamp_ns] = len(index)
                    support_rows += 1
                    if timestamp_ns >= contract.STUDY_START_NS:
                        study_rows += 1
                    if timestamp_ns + contract.AVAILABILITY_OFFSET_NS >= contract.WINDOW_END_NS:
                        boundary_excluded += 1
        if partition_rows != parquet.metadata.num_rows:
            raise RuntimeError("GC_DISPOSITION_OANDA_PARTITION_ROWCOUNT_MISMATCH")
        partition_reports.append({
            "path": str(path),
            "sha256": item["sha256"],
            "rows_read_from_time_column": partition_rows,
        })
    if not index:
        raise RuntimeError("GC_DISPOSITION_EMPTY_OANDA_SUPPORT_WINDOW")
    return index, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": source["sha256"],
        "partitions": partition_reports,
        "rows_read_from_time_column": total_rows,
        "support_bar_starts": support_rows,
        "study_bar_starts": study_rows,
        "strict_decision_eligible_support_bar_starts": support_rows - boundary_excluded,
        "strict_decision_boundary_excluded_bar_starts": boundary_excluded,
        "read_only_column": "time",
        "bar_start_semantics": "UTC M1 bar start",
        "availability_offset_seconds": 60,
        "strict_decision_condition": "bar_start + 60 seconds < 2026-07-01T00:00:00Z",
    }, [(manifest_path, source["sha256"]), *partition_hashes]


def _audit_worker(items: list[dict[str, Any]]) -> dict[str, Any]:
    return contract.audit_symbol_group(items, WORKER_OANDA_BAR_INDEX)


def _rehash_sources(gc_hashes: dict[str, Any], oanda_hashes: list[tuple[Path, str]]) -> None:
    for path, expected in gc_hashes["gc_ancillary_files"]:
        if sha256_file(path) != expected:
            raise RuntimeError("GC_DISPOSITION_GC_BINDING_CHANGED_DURING_AUDIT")
    for path, expected, receipt, receipt_expected in gc_hashes["gc_source_files"]:
        if sha256_file(path) != expected or sha256_file(receipt) != receipt_expected:
            raise RuntimeError("GC_DISPOSITION_GC_SOURCE_CHANGED_DURING_AUDIT")
    for path, expected in oanda_hashes:
        if sha256_file(path) != expected:
            raise RuntimeError("GC_DISPOSITION_OANDA_SOURCE_CHANGED_DURING_AUDIT")


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan, approval, binding = require_runtime(args)
    cap = _require_capped_cgroup_limits(
        environ=os.environ, read_text=None, max_memory_bytes=4 * 1024**3,
        error_prefix="GC_DISPOSITION", max_pids=64,
    )
    cap["lock"] = require_capped_lock_ancestry()
    groups, gc_hashes = _bound_gc_inputs(plan)
    oanda_index, oanda_report, oanda_hashes = _load_oanda(plan)
    global WORKER_OANDA_BAR_INDEX
    WORKER_OANDA_BAR_INDEX = oanda_index
    context = mp.get_context("fork")
    results = []
    with ProcessPoolExecutor(max_workers=WORKER_COUNT, mp_context=context) as pool:
        futures = [pool.submit(_audit_worker, group["items"]) for group in groups]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["raw_symbol"])
    if len(results) != 15:
        raise RuntimeError("GC_DISPOSITION_CONTRACT_RESULT_COUNT_INVALID")
    _rehash_sources(gc_hashes, oanda_hashes)
    for relative, expected in binding["code_sha256"].items():
        verify_digest(REPO / relative, expected, "CODE_POST_AUDIT")
    verify_digest(args.plan, PLAN_SHA256, "PLAN_POST_AUDIT")
    if sha256_file(args.approval) != binding["approval_receipt_sha256"]:
        raise RuntimeError("GC_DISPOSITION_APPROVAL_CHANGED_DURING_AUDIT")
    claim = read_json(EXPECTED_CLAIM)
    if sha256_file(args.binding) != claim["execution_binding_sha256"]:
        raise RuntimeError("GC_DISPOSITION_BINDING_CHANGED_DURING_AUDIT")
    if sha256_file(REPO / "NEXT_RUN_POLICY.json") != binding["policy_sha256"]:
        raise RuntimeError("GC_DISPOSITION_POLICY_CHANGED_DURING_AUDIT")

    presence_names = (
        "_record_presence", "_trade_presence",
        "_study_record_presence", "_study_trade_presence",
    )
    union_presence = {
        name: contract.merge_presence_bitmaps([result[name] for result in results])
        for name in presence_names
    }
    public_symbols = []
    per_symbol_bar_counts = {}
    for result in results:
        public = {key: value for key, value in result.items() if not key.startswith("_")}
        public["unique_support_oanda_m1_bars_with_gc_record"] = sum(result["_record_presence"])
        public["unique_support_oanda_m1_bars_with_gc_trade"] = sum(result["_trade_presence"])
        public["unique_study_oanda_m1_bars_with_gc_record"] = sum(result["_study_record_presence"])
        public["unique_study_oanda_m1_bars_with_gc_trade"] = sum(result["_study_trade_presence"])
        public_symbols.append(public)
        per_symbol_bar_counts[result["raw_symbol"]] = {
            "support_bars_with_record": public["unique_support_oanda_m1_bars_with_gc_record"],
            "support_bars_with_trade": public["unique_support_oanda_m1_bars_with_gc_trade"],
            "study_bars_with_record": public["unique_study_oanda_m1_bars_with_gc_record"],
            "study_bars_with_trade": public["unique_study_oanda_m1_bars_with_gc_trade"],
        }
    worker_affinity = sorted(os.sched_getaffinity(0))
    result_payload = {
        "schema_version": contract.GC_DISPOSITION_SCHEMA,
        "status": "SOURCE_DISPOSITION_COMPLETE_NOT_ADMITTED",
        "research_id": "GC_ORDER_FLOW_RESEARCH_001",
        "plan": {"path": PLAN_REL, "sha256": PLAN_SHA256},
        "approval": {
            "path": str(args.approval),
            "sha256": sha256_file(args.approval),
            "approved_utc": approval["approved_utc"],
            "single_use": True,
        },
        "execution_binding": {
            "path": str(args.binding),
            "sha256": sha256_file(args.binding),
            "source_head": binding["source_head"],
            "code_sha256": binding["code_sha256"],
        },
        "windows_utc": {
            "support": ["2024-04-01T00:00:00Z", "2026-07-01T00:00:00Z"],
            "study": ["2024-05-01T00:00:00Z", "2026-07-01T00:00:00Z"],
            "decision_must_be_strictly_before": "2026-07-01T00:00:00Z",
        },
        "inputs": {
            "gc_plan_path": gc_hashes["gc_plan_path"],
            "gc_plan_sha256": gc_hashes["gc_plan_sha256"],
            "gc_result_sha256": gc_hashes["gc_result_sha256"],
            "gc_reconciliation_sha256": gc_hashes["gc_reconciliation_sha256"],
            "gc_source_file_count": len(gc_hashes["gc_source_files"]),
            "oanda": oanda_report,
        },
        "coverage_by_symbol": public_symbols,
        "support_bar_presence_counts_by_symbol": per_symbol_bar_counts,
        "contract_union_upper_bound": {
            "contract_symbols": [result["raw_symbol"] for result in results],
            "contract_symbol_count": len(results),
            "unique_support_oanda_m1_bars_with_any_gc_record": sum(union_presence["_record_presence"]),
            "unique_support_oanda_m1_bars_with_any_gc_trade": sum(union_presence["_trade_presence"]),
            "unique_study_oanda_m1_bars_with_any_gc_record": sum(union_presence["_study_record_presence"]),
            "unique_study_oanda_m1_bars_with_any_gc_trade": sum(union_presence["_study_trade_presence"]),
            "records_or_volumes_summed_across_contracts": False,
            "roll_rule_selected": False,
            "semantics": "Unique OANDA M1 timestamps touched by at least one separately reported dated outright; upper bound only, not a continuous GC series.",
        },
        "session_schedule": {
            "regular_template_timezone": "America/Chicago",
            "regular_template": "Sunday 17:00 CT through Friday 16:00 CT, with daily 16:00-17:00 CT maintenance",
            "historical_holiday_exception_status": "SCHEDULE_UNKNOWN",
            "event_silence_semantics": "A zero-row interval is event silence, not evidence of packet loss or feed failure.",
            "official_references": plan["method"]["cme_refs"],
            "per_contract_disposition": "Reported in coverage_by_symbol.contract_session_template_disposition",
        },
        "execution": {
            "capped_class": "audit",
            "memory_max_and_high_bytes": cap["memory_max"],
            "swap_max_bytes": cap["swap"],
            "tasks_max": cap["pids"],
            "cpu_affinity": worker_affinity,
            "worker_processes": WORKER_COUNT,
            "numerical_threads_per_worker": 1,
            "network_calls": 0,
            "new_downloads": 0,
            "new_spending": False,
            "features_built": False,
            "fit_started": False,
            "test_accessed": False,
            "model_admission": False,
            "broker_live_paper_or_trading": False,
        },
        "interpretation": "Structural timestamp/source disposition only. Unknown aggressor side remains separate; F_BAD_TS_RECV rows are counted and excluded from causal matching; F_MAYBE_BAD_BOOK is counted as vendor channel-gap evidence without claiming book recovery or OFI validity.",
        "admitted_to_model_or_economic_test": False,
    }
    run_root = RUN_ROOT
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=False, exist_ok=False)
    result_path = results_dir / "RESULT.json"
    recon_path = run_root / "POST_AUDIT_RECONCILIATION.json"
    write_json(result_path, result_payload)
    result_sha = sha256_file(result_path)
    reconciliation = {
        "schema_version": "gx1_gc_source_disposition_reconciliation_v1",
        "status": "ALL_DECLARED_INPUT_HASHES_REVERIFIED_AFTER_SCAN",
        "plan_sha256": PLAN_SHA256,
        "result_sha256": result_sha,
        "gc_source_files_reverified": len(gc_hashes["gc_source_files"]),
        "gc_source_receipts_reverified": len(gc_hashes["gc_source_files"]),
        "oanda_manifest_and_partitions_reverified": len(oanda_hashes),
        "oanda_rows_read_from_time_column": oanda_report["rows_read_from_time_column"],
        "excluded_fields_not_accessed": ["OANDA prices", "OANDA volume", "GC ts_event", "GC prices", "GC top-of-book fields"],
        "gc_csv_handling": "Only whitelisted columns are indexed or converted; excluded CSV cells are not interpreted.",
        "network_calls": 0,
        "new_downloads": 0,
        "new_spending": False,
        "features_built": False,
        "fit_started": False,
        "test_accessed": False,
        "admitted_to_model_or_economic_test": False,
    }
    write_json(recon_path, reconciliation)
    terminal = {
        "status": "COMPLETE_NOT_ADMITTED",
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "result_sha256": result_sha,
        "reconciliation_sha256": sha256_file(recon_path),
        "plan_sha256": PLAN_SHA256,
        "approval_receipt_sha256": sha256_file(args.approval),
        "execution_binding_sha256": sha256_file(args.binding),
        "invocation": 1,
        "admitted_to_model_or_economic_test": False,
    }
    write_json(results_dir / "TERMINAL.json", terminal)
    return terminal


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        terminal = run(args)
    except Exception as exc:
        print(f"GC_SOURCE_DISPOSITION_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
