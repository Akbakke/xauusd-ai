#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from gx1.execution.logging_transport_v1 import (
    EVENT_TRACE_FILE,
    MANIFEST_FILE,
    RINGBUFFER_FILE,
    STATUS_FILE,
    SUMMARY_FILE,
    build_logging_surface_inventory_rows,
    ensure_policy_rows_have_valid_contract,
)


LAYER_ID = "LOGGING_TRANSPORT_INFO_THROTTLE_V1"
CONTRACT = "logging_transport_contract_v1.json"
INVENTORY = "logging_surface_inventory_v1.csv"
SPAM_PLAN = "hot_path_spam_reduction_plan_v1.csv"
COUNTER_SPEC = "structured_counter_summary_spec_v1.json"
RATE_LIMIT_CONTRACT = "rate_limit_debug_contract_v1.json"
NO_SEMANTIC_CHANGE_GUARD = "no_semantic_change_guard_v1.json"
IMPLEMENTATION_PLAN = "implementation_artifact_plan_v1.json"
NEXT_STEP = "next_step_recommendation_v1.json"
CONSISTENCY_AUDIT = "logging_transport_consistency_audit_v1.json"
SUMMARY = "logging_transport_summary_v1.json"
REPORT = "logging_transport_report_v1.md"
MANIFEST_STATUS = "logging_transport_manifest_status_v1.json"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> Path:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _source_locator_rows(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    return {
        "REPLAY_PROGRESS_HEARTBEAT": {
            "source_path": repo_root / "gx1/execution/oanda_demo_runner.py",
            "must_contain": "[REPLAY PROGRESS]",
        },
        "REPLAY_PROGRESS_REDUNDANT": {
            "source_path": repo_root / "gx1/execution/oanda_demo_runner.py",
            "must_contain": "[REPLAY] Progress:",
        },
        "ENTRY_NO_TRADE_CLOSED_WINDOW_PROOF": {
            "source_path": repo_root / "gx1/execution/entry_manager.py",
            "must_contain": "[ENTRY_NO_TRADE_CLOSED_WINDOW_PROOF]",
        },
        "ENTRY_GAP_GUARD": {
            "source_path": repo_root / "gx1/execution/entry_manager.py",
            "must_contain": "[ENTRY_GAP_GUARD]",
        },
        "MID_EDGE_10_50_PROBE": {
            "source_path": repo_root / "gx1/execution/exit_manager.py",
            "must_contain": "[MID_EDGE_10_50_PROBE]",
        },
        "EXIT_CATA_GUARD_EVENT": {
            "source_path": repo_root / "gx1/execution/exit_manager.py",
            "must_contain": "[EXIT_CATA_GUARD_EVENT]",
        },
        "EXIT_INPUT_PREP_BREAKDOWN": {
            "source_path": repo_root / "gx1/execution/exit_manager.py",
            "must_contain": "[EXIT_INPUT_PREP_BREAKDOWN]",
        },
        "EXIT_DECISION_RESULT": {
            "source_path": repo_root / "gx1/execution/exit_manager.py",
            "must_contain": "[EXIT_DECISION_RESULT]",
        },
        "EXIT_REPLAY_THRESHOLD_REJECT_HOLD_FASTPATH": {
            "source_path": repo_root / "gx1/execution/exit_manager.py",
            "must_contain": "[EXIT_REPLAY_THRESHOLD_REJECT_HOLD_FASTPATH]",
        },
    }


def _spam_reduction_rows() -> List[Dict[str, Any]]:
    return [
        {
            "family": "ENTRY_NO_TRADE_CLOSED_WINDOW_PROOF",
            "current_behavior": "INFO per minute inside closed-market window.",
            "why_it_spams": "Per-bar proof logging during predictable closed windows.",
            "must_keep_as_proof": "That the closed-window rule activated, which pattern applied, and total blocked bars/window span.",
            "aggregate_to_summary": "count per pattern, first_seen, last_seen, suppressed_count",
            "move_to_trace": "emitted first-signal only, plus ringbuffer on abnormal mode",
            "info_policy": "FIRST_ONLY",
        },
        {
            "family": "ENTRY_GAP_GUARD",
            "current_behavior": "Logs gap detection and every cooldown bar.",
            "why_it_spams": "Cooldown bars repeat until guard expires.",
            "must_keep_as_proof": "First gap hit, cooldown size, total blocked bars per gap.",
            "aggregate_to_summary": "gap_detected count, cooldown_block count, first/last seen per key",
            "move_to_trace": "first emitted event only unless debug",
            "info_policy": "FIRST_ONLY for GAP_DETECTED, SUMMARY_ONLY for GAP_COOLDOWN_BLOCK",
        },
        {
            "family": "MID_EDGE_10_50_PROBE",
            "current_behavior": "Logs many state transitions per trade in protected-profit path.",
            "why_it_spams": "Transition reasons change often across open trades in hot path.",
            "must_keep_as_proof": "ARMED transitions, abnormal first skip reason, count of common skip reasons.",
            "aggregate_to_summary": "reason counts, unique trades, first/last seen, sample events",
            "move_to_trace": "ARMED and abnormal emitted events only; full detail only in debug mode",
            "info_policy": "SUMMARY_ONLY normally, FIRST_ONLY for ARMED or unexpected skip",
        },
        {
            "family": "EXIT_CATA_GUARD_EVENT",
            "current_behavior": "Same guard-close emits several near-identical INFO lines.",
            "why_it_spams": "Trigger, signal, arbiter handoff and runtime-safety message all log separately.",
            "must_keep_as_proof": "One structured event with guard payload and thresholds.",
            "aggregate_to_summary": "count, affected trades, first/last seen",
            "move_to_trace": "single emitted event plus ringbuffer",
            "info_policy": "KEEP_INFO as one merged event",
        },
        {
            "family": "EXIT_INPUT_PREP_BREAKDOWN",
            "current_behavior": "Large profiling payload logged at first and every 2000 windows.",
            "why_it_spams": "Breakdown dict is large and noisy in heavy weeks.",
            "must_keep_as_proof": "Profiling totals and worst snapshots.",
            "aggregate_to_summary": "worst_events, sample_events, latest total_sec",
            "move_to_trace": "none by default; emitted checkpoints only",
            "info_policy": "FIRST_EVERY_N_FINAL",
        },
    ]


def _counter_spec() -> Dict[str, Any]:
    return {
        "required_counters_v1": [
            "counts per reason-code",
            "counts per logger family",
            "unique trade count affected",
            "first seen / last seen",
            "suppressed count",
            "emitted count",
            "optional top-N keys",
            "optional worst profiling snapshots",
        ],
        "runtime_artifacts_v1": {
            "event_trace": f"observability_transport_v1/{EVENT_TRACE_FILE}",
            "summary": f"observability_transport_v1/{SUMMARY_FILE}",
            "manifest": f"observability_transport_v1/{MANIFEST_FILE}",
            "status": f"observability_transport_v1/{STATUS_FILE}",
            "ringbuffer": f"observability_transport_v1/{RINGBUFFER_FILE}",
        },
        "replacement_principle_v1": "Summary becomes richer as INFO becomes quieter; INFO is never proof-canonical.",
        "ringbuffer_behavior_v1": "Recent emitted or abnormal events are checkpointed for postmortem without keeping the whole hot-path stdout stream.",
    }


def _rate_limit_contract() -> Dict[str, Any]:
    return {
        "supported_info_policies_v1": [
            "FIRST_ONLY",
            "FIRST_AND_FINAL",
            "FIRST_EVERY_N_FINAL",
            "SUMMARY_ONLY",
            "DEBUG_ONLY",
        ],
        "normal_runtime_v1": "Heartbeat-dominated INFO; proof detail moved to summary/trace artifacts.",
        "debug_override_v1": "GX1_LOG_TRANSPORT_DEBUG=1 or canary mode enables richer event capture.",
        "heavy_week_mode_v1": "GX1_LOG_TRANSPORT_HEAVY_WEEK=1 suppresses KEEP_INFO families down to first occurrence when needed.",
        "fail_capture_v1": "Failure marks force summary + ringbuffer flush; SIGKILL remains outside Python's control.",
        "ringbuffer_v1": {
            "env_key": "GX1_LOG_TRANSPORT_RINGBUFFER_MAX",
            "default": 200,
        },
        "flush_checkpoint_v1": {
            "env_key": "GX1_LOG_TRANSPORT_FLUSH_SEC",
            "default": 30.0,
            "checkpoint_on_v1": [
                "main replay heartbeat",
                "run_replay finally",
                "failure mark",
            ],
        },
    }


def _no_semantic_change_guard(repo_root: Path) -> Dict[str, Any]:
    touched = [
        "gx1/execution/logging_transport_v1.py",
        "gx1/execution/oanda_demo_runner.py",
        "gx1/execution/entry_manager.py",
        "gx1/execution/exit_manager.py",
        "gx1/scripts/materialize_logging_transport_cleanup_v1.py",
    ]
    return {
        "touched_files_v1": touched,
        "transport_only_statement_v1": [
            "no policy threshold changes",
            "no guard threshold changes",
            "no close-authority changes",
            "no feature-schema changes",
            "no AS_OF/HINDSIGHT mixing",
            "no truth/canonical result mutation by design",
        ],
        "preserved_truth_surfaces_v1": [
            "EXIT_EVAL_TRACE.csv",
            "pred_trace_*.jsonl",
            "trade_log.csv",
            "trade_journal parquet",
            "trade_outcomes parquet",
            "REPLAY_SUMMARY.json",
        ],
        "logging_only_functions_v1": {
            "gx1/execution/entry_manager.py": [
                "EntryManager.evaluate_entry",
                "EntryManager._record_runner_observability_event",
            ],
            "gx1/execution/exit_manager.py": [
                "ExitManager._log_mid_edge_10_50_probe",
                "ExitManager.evaluate_and_close_trades",
                "ExitManager._build_exit_window_array",
                "ExitManager._record_runner_observability_event",
            ],
            "gx1/execution/oanda_demo_runner.py": [
                "_init_replay_observability_transport",
                "_record_replay_observability_event",
                "_flush_replay_observability_transport",
                "_mark_replay_observability_failure",
                "run_replay",
                "_run_replay_impl",
            ],
        },
        "notes_v1": "Changes only affect log routing, counters, checkpoints and redundant INFO suppression.",
        "repo_root_v1": str(repo_root),
    }


def _implementation_plan(repo_root: Path) -> Dict[str, Any]:
    return {
        "repo_root_v1": str(repo_root),
        "runtime_outputs_v1": {
            "artifact_dir_suffix": "observability_transport_v1",
            "summary_file": SUMMARY_FILE,
            "manifest_file": MANIFEST_FILE,
            "status_file": STATUS_FILE,
            "ringbuffer_file": RINGBUFFER_FILE,
            "event_trace_file": EVENT_TRACE_FILE,
        },
        "info_design_requirements_v1": [
            "INFO is not proof-canonical",
            "terminal shows heartbeat and first abnormal signal, not entire proof package",
            "structured keys and reason-codes stay stable",
            "summary gets richer as INFO gets slimmer",
            "debug mode is explicit and opt-in",
        ],
        "runtime_hot_path_changes_v1": [
            "closed-window proof now emits first signal only and counts the rest",
            "gap guard emits first gap event and summarizes cooldown bars",
            "mid-edge probe emits ARMED and abnormal first skip; common skip reasons go to summary",
            "catastrophic guard emits one structured INFO event instead of several near-duplicates",
            "input prep profiling checkpoints are summarized instead of repeated full INFO payloads",
            "duplicate replay progress line is downgraded to DEBUG",
        ],
    }


def _next_step() -> Dict[str, Any]:
    return {
        "decision_v1": "SAFE_TO_APPLY_WITHOUT_REPLAY_LOGIC_RISK",
        "why_v1": "The implementation is transport-only, keeps proof in artifacts/summary, and avoids policy/guard changes.",
        "follow_on_v1": [
            "Use heavy debug mode only for pathological weeks.",
            "Let the existing replay queue finish before evaluating runtime log-volume deltas.",
        ],
    }


def _consistency_audit(repo_root: Path, inventory_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors = ensure_policy_rows_have_valid_contract(inventory_rows)
    source_locator = _source_locator_rows(repo_root)
    source_checks: List[Dict[str, Any]] = []
    for family, locator in source_locator.items():
        path = locator["source_path"]
        exists = path.exists()
        contains = False
        if exists:
            contains = locator["must_contain"] in path.read_text(encoding="utf-8")
        source_checks.append(
            {
                "family": family,
                "source_path": str(path),
                "source_exists": exists,
                "source_contains_locator": contains,
                "locator": locator["must_contain"],
            }
        )
        if not exists or not contains:
            errors.append(f"source locator missing family={family}")
    return {
        "checked_at_utc": _now_utc(),
        "policy_row_count": len(inventory_rows),
        "policy_contract_errors": errors,
        "source_checks": source_checks,
        "status_v1": "PASS" if not errors else "FAIL",
    }


def _status_split(consistency: Dict[str, Any]) -> Dict[str, str]:
    source_checks = consistency.get("source_checks", [])
    all_locators_ok = bool(source_checks) and all(bool(row.get("source_contains_locator")) for row in source_checks)
    return {
        "BEVIST": (
            "Hot-path families are classified in a locked policy table; runtime code routes targeted spam families through "
            "the observability transport; source locators and policy contracts passed."
            if all_locators_ok and consistency.get("status_v1") == "PASS"
            else "Policy/source contract not fully proven."
        ),
        "INDIKERT": (
            "Lower stdout volume is expected because duplicate and repeated INFO emissions are now throttled or moved to summary. "
            "Full runtime volume improvement should be confirmed on the already-running replay queue."
        ),
        "IKKE_ETABLERT": (
            "No new full replay was started for this job, and SIGKILL-style process death cannot be fully captured by Python ringbuffer flushing."
        ),
    }


def _markdown_report(
    *,
    extension_dir: Path,
    inventory_rows: List[Dict[str, Any]],
    spam_rows: List[Dict[str, Any]],
    next_step_payload: Dict[str, Any],
    status_split: Dict[str, str],
) -> str:
    lines = [
        f"# {LAYER_ID}",
        "",
        "## Contract",
        "- INFO is not proof-canonical.",
        "- Important truth stays in trace / artifacts / summary.",
        "- Summary becomes richer as INFO becomes slimmer.",
        "- Debug mode is explicit, narrow and opt-in.",
        "",
        "## Classified Families",
    ]
    for row in inventory_rows:
        lines.append(
            f"- `{row['family']}`: {row['policy_action']} / {row['default_info_policy']} / {row['observability_purpose']} / {row['frequency_type']}"
        )
    lines.extend(["", "## Spam Reduction",])
    for row in spam_rows:
        lines.append(
            f"- `{row['family']}` -> `{row['info_policy']}`; proof kept via {row['aggregate_to_summary']} and {row['move_to_trace']}."
        )
    lines.extend(
        [
            "",
            "## Runtime Artifacts",
            f"- `{extension_dir.name}` documents the transport plan; runtime replay writes `observability_transport_v1/{SUMMARY_FILE}` and related artifacts inside each run/chunk directory.",
            "",
            "## Next Step",
            f"- `{next_step_payload['decision_v1']}`: {next_step_payload['why_v1']}",
            "",
            "## Hard Status",
            f"- `BEVIST`: {status_split['BEVIST']}",
            f"- `INDIKERT`: {status_split['INDIKERT']}",
            f"- `IKKE_ETABLERT`: {status_split['IKKE_ETABLERT']}",
            "",
        ]
    )
    return "\n".join(lines)


def materialize(*, repo_root: Path, reports_root: Path) -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    extension_dir = reports_root / f"{LAYER_ID}_{timestamp}"
    extension_dir.mkdir(parents=True, exist_ok=True)

    inventory_rows = build_logging_surface_inventory_rows()
    spam_rows = _spam_reduction_rows()
    contract_payload = {
        "layer_id": LAYER_ID,
        "created_at_utc": _now_utc(),
        "repo_root_v1": str(repo_root),
        "reports_root_v1": str(reports_root),
        "core_principle_v1": "Keep facts, change medium. INFO gets calmer; summary/trace gets richer.",
        "non_goals_v1": [
            "no replay semantics change",
            "no trade decision change",
            "no threshold or guard change",
            "no truth or audit contract change",
            "no AS_OF/HINDSIGHT mixing",
        ],
    }
    counter_spec = _counter_spec()
    rate_limit_contract = _rate_limit_contract()
    no_semantic_guard = _no_semantic_change_guard(repo_root)
    implementation_plan = _implementation_plan(repo_root)
    next_step_payload = _next_step()
    consistency = _consistency_audit(repo_root, inventory_rows)
    status_split = _status_split(consistency)

    _write_json(extension_dir / CONTRACT, contract_payload)
    _write_csv(extension_dir / INVENTORY, inventory_rows)
    _write_csv(extension_dir / SPAM_PLAN, spam_rows)
    _write_json(extension_dir / COUNTER_SPEC, counter_spec)
    _write_json(extension_dir / RATE_LIMIT_CONTRACT, rate_limit_contract)
    _write_json(extension_dir / NO_SEMANTIC_CHANGE_GUARD, no_semantic_guard)
    _write_json(extension_dir / IMPLEMENTATION_PLAN, implementation_plan)
    _write_json(extension_dir / NEXT_STEP, next_step_payload)
    _write_json(extension_dir / CONSISTENCY_AUDIT, consistency)

    summary_payload = {
        "layer_id": LAYER_ID,
        "created_at_utc": _now_utc(),
        "extension_dir": str(extension_dir),
        "inventory_row_count_v1": len(inventory_rows),
        "spam_plan_row_count_v1": len(spam_rows),
        "next_step_v1": next_step_payload,
        "consistency_status_v1": consistency["status_v1"],
        "status_split_v1": status_split,
        "updated_files_v1": no_semantic_guard["touched_files_v1"],
    }
    _write_json(extension_dir / SUMMARY, summary_payload)

    manifest_status = {
        "layer_id": LAYER_ID,
        "created_at_utc": _now_utc(),
        "extension_dir": str(extension_dir),
        "artifacts_v1": {
            "contract": str(extension_dir / CONTRACT),
            "inventory": str(extension_dir / INVENTORY),
            "spam_plan": str(extension_dir / SPAM_PLAN),
            "counter_spec": str(extension_dir / COUNTER_SPEC),
            "rate_limit_contract": str(extension_dir / RATE_LIMIT_CONTRACT),
            "no_semantic_change_guard": str(extension_dir / NO_SEMANTIC_CHANGE_GUARD),
            "implementation_plan": str(extension_dir / IMPLEMENTATION_PLAN),
            "next_step": str(extension_dir / NEXT_STEP),
            "consistency_audit": str(extension_dir / CONSISTENCY_AUDIT),
            "summary": str(extension_dir / SUMMARY),
            "report": str(extension_dir / REPORT),
        },
        "status_v1": consistency["status_v1"],
    }
    _write_json(extension_dir / MANIFEST_STATUS, manifest_status)

    report_text = _markdown_report(
        extension_dir=extension_dir,
        inventory_rows=inventory_rows,
        spam_rows=spam_rows,
        next_step_payload=next_step_payload,
        status_split=status_split,
    )
    (extension_dir / REPORT).write_text(report_text, encoding="utf-8")
    return {
        "extension_dir": str(extension_dir),
        "summary": summary_payload,
        "consistency": consistency,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/home/andre2/src/GX1_ENGINE"))
    parser.add_argument("--reports-root", type=Path, default=Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity"))
    args = parser.parse_args()
    result = materialize(repo_root=args.repo_root, reports_root=args.reports_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
