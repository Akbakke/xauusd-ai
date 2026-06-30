#!/usr/bin/env python3
"""Review active Exit Transformer train-execution readiness without opening training.

This report binds the pretrain manifest, fail-closed train wrapper, RAM
guardrails and weak-slice disclosure into one train-execution review artifact.
It deliberately keeps Exit training closed until a separate explicit human
vedtak package enables training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT, REPO


DEFAULT_TRAINING_PLAN_JSON = (
    REPORTS_ROOT
    / "entry_exit_transformer_training_plan_readiness_20260630_v1/ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READINESS_latest.json"
)
DEFAULT_WRAPPER_READINESS_JSON = (
    REPORTS_ROOT
    / "entry_exit_transformer_trainer_wrapper_readiness_20260630_v1/ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READINESS_latest.json"
)
DEFAULT_PRETRAIN_MANIFEST_JSON = (
    REPORTS_ROOT
    / "entry_exit_transformer_pretrain_manifest_20260630_v1/ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_latest.json"
)
DEFAULT_SLICE_ROBUSTNESS_JSON = (
    REPORTS_ROOT
    / "entry_exit_model_dataset_slice_robustness_20260630_v1/ENTRY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_exit_transformer_train_execution_review_20260630_v1"

READY_TRAINING_PLAN_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW"
READY_WRAPPER_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READY_FOR_IMPLEMENTATION_REVIEW"
READY_PRETRAIN_DECISION = "ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_READY_FOR_TRAIN_EXECUTION_REVIEW"
READY_SLICE_DECISION = "ENTRY_EXIT_MODEL_DATASET_SLICE_ROBUSTNESS_READY_WITH_WEAK_SLICE_DISCLOSURE"
READY_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_READY_FOR_EXPLICIT_VEDTAK_PACKAGE"
BLOCKED_DECISION = "BLOCKED_BY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW"
VEDTAK_PREFIX = "ENTRY_EXIT_TRANSFORMER_TRAIN_"


def _check(name: str, condition: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(condition), "details": details or {}}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _git_status_short() -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return [line for line in (proc.stdout or "").splitlines() if line.strip()]


def _weak_slice_policy(slice_report: dict[str, Any]) -> dict[str, Any]:
    slice_review = slice_report.get("slice_review") if isinstance(slice_report.get("slice_review"), dict) else {}
    weak_slices = slice_review.get("weak_slices") if isinstance(slice_review.get("weak_slices"), list) else []
    unsupported = slice_review.get("unsupported_slices") if isinstance(slice_review.get("unsupported_slices"), list) else []
    return {
        "weak_slice_count": int(slice_review.get("weak_slice_count") or 0),
        "unsupported_slice_count": int(slice_review.get("unsupported_slice_count") or 0),
        "weak_slice_examples": weak_slices[:12],
        "unsupported_slices": unsupported,
        "train_execution_requirements": {
            "must_report_session_regime_side_metrics": True,
            "must_report_direction_and_tail_metrics": True,
            "must_not_promote_from_broad_average": True,
            "must_compare_weak_slices_separately": True,
            "must_block_shadow_live_on_unsupported_slice": True,
            "must_keep_weak_slice_disclosure_in_post_train_audit": True,
        },
    }


def _review_contract(training_plan: dict[str, Any], slice_report: dict[str, Any]) -> dict[str, Any]:
    plan = training_plan.get("training_plan") if isinstance(training_plan.get("training_plan"), dict) else {}
    resources = plan.get("resource_guardrails") if isinstance(plan.get("resource_guardrails"), dict) else {}
    command = plan.get("future_training_command_contract") if isinstance(plan.get("future_training_command_contract"), dict) else {}
    weak_policy = _weak_slice_policy(slice_report)
    num_workers = resources.get("num_workers")
    ready = bool(
        command.get("requires_explicit_vedtak") is True
        and str(command.get("vedtak_prefix_required") or "") == VEDTAK_PREFIX
        and command.get("requires_clean_git") is True
        and command.get("requires_ram_guard") is True
        and num_workers is not None
        and int(num_workers) == 0
        and float(resources.get("max_process_rss_gib") or 0.0) <= 8.0
        and float(resources.get("abort_if_mem_available_below_gib") or 0.0) >= 8.0
        and weak_policy["unsupported_slice_count"] == 0
        and all(weak_policy["train_execution_requirements"].values())
    )
    return {
        "ready": ready,
        "future_train_command_contract": command,
        "resource_guardrails": resources,
        "weak_slice_policy": weak_policy,
        "training_remains_closed_until_explicit_enablement": True,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Exit Transformer Train Execution Review",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Weak slice count: `{report['review_contract']['weak_slice_policy']['weak_slice_count']}`",
        f"- Unsupported slice count: `{report['review_contract']['weak_slice_policy']['unsupported_slice_count']}`",
        f"- Exit training allowed: `{report['exit_training_allowed']}`",
        f"- Exit training allowed with explicit vedtak: `{report['exit_training_allowed_with_explicit_vedtak']}`",
        f"- Next required gate: `{report['next_required_gate']}`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- `{failure['check']}`")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    training_plan_json = Path(args.training_plan_json).expanduser().resolve()
    wrapper_readiness_json = Path(args.wrapper_readiness_json).expanduser().resolve()
    pretrain_manifest_json = Path(args.pretrain_manifest_json).expanduser().resolve()
    slice_robustness_json = Path(args.slice_robustness_json).expanduser().resolve()
    training_plan = _read_json_or_empty(training_plan_json)
    wrapper_readiness = _read_json_or_empty(wrapper_readiness_json)
    pretrain_manifest = _read_json_or_empty(pretrain_manifest_json)
    slice_robustness = _read_json_or_empty(slice_robustness_json)
    review_contract = _review_contract(training_plan, slice_robustness)
    git_status = _git_status_short()
    checks = [
        _check("active Exit Transformer training plan readiness is ready", training_plan.get("decision") == READY_TRAINING_PLAN_DECISION, {"path": str(training_plan_json), "decision": training_plan.get("decision")}),
        _check("active Exit Transformer trainer wrapper readiness is ready", wrapper_readiness.get("decision") == READY_WRAPPER_DECISION, {"path": str(wrapper_readiness_json), "decision": wrapper_readiness.get("decision")}),
        _check("active Exit Transformer pretrain manifest is ready", pretrain_manifest.get("decision") == READY_PRETRAIN_DECISION, {"path": str(pretrain_manifest_json), "decision": pretrain_manifest.get("decision")}),
        _check("active Exit model dataset slice robustness is ready", slice_robustness.get("decision") == READY_SLICE_DECISION, {"path": str(slice_robustness_json), "decision": slice_robustness.get("decision")}),
        _check("train execution review accounts for weak slices and RAM guardrails", bool(review_contract.get("ready")), review_contract),
        _check(
            "train execution review never trains, replays, distills, promotes, shadows, or starts live",
            True,
            {
                "trainer_started": False,
                "replay_started": False,
                "iql_distillation_started": False,
                "exit_training_allowed": False,
                "exit_training_allowed_with_explicit_vedtak": False,
                "exit_iql_allowed": False,
                "promotion_shadow_live_allowed": False,
            },
        ),
    ]
    failures = [{"check": check["name"], "details": check.get("details") or {}} for check in checks if not check["ok"]]
    ready = not failures
    decision = READY_DECISION if ready else BLOCKED_DECISION
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_{timestamp}.json"
    md_path = out_dir / f"ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_{timestamp}.md"
    report = {
        "schema_version": "entry_exit_transformer_train_execution_review_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "training_plan_json": str(training_plan_json),
        "training_plan_json_sha256": _sha256_file(training_plan_json) if training_plan_json.exists() else "",
        "wrapper_readiness_json": str(wrapper_readiness_json),
        "wrapper_readiness_json_sha256": _sha256_file(wrapper_readiness_json) if wrapper_readiness_json.exists() else "",
        "pretrain_manifest_json": str(pretrain_manifest_json),
        "pretrain_manifest_json_sha256": _sha256_file(pretrain_manifest_json) if pretrain_manifest_json.exists() else "",
        "slice_robustness_json": str(slice_robustness_json),
        "slice_robustness_json_sha256": _sha256_file(slice_robustness_json) if slice_robustness_json.exists() else "",
        "review_contract": review_contract,
        "current_git_status_short": git_status,
        "checks": checks,
        "failures": failures,
        "exit_training_allowed": False,
        "exit_training_allowed_with_explicit_vedtak": False,
        "exit_iql_allowed": False,
        "trainer_started": False,
        "replay_started": False,
        "iql_distillation_started": False,
        "promotion_shadow_live_allowed": False,
        "future_vedtak_prefix_required": VEDTAK_PREFIX,
        "next_required_gate": (
            "explicit Exit Transformer train-execution enablement vedtak package; training remains closed"
            if ready
            else "repair Exit Transformer train-execution review before any training enablement discussion"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps({"decision": decision, "failures": failures, "json_path": str(json_path)}, indent=2, sort_keys=True))
    if args.fail_on_not_ready and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-plan-json", default=str(DEFAULT_TRAINING_PLAN_JSON))
    ap.add_argument("--wrapper-readiness-json", default=str(DEFAULT_WRAPPER_READINESS_JSON))
    ap.add_argument("--pretrain-manifest-json", default=str(DEFAULT_PRETRAIN_MANIFEST_JSON))
    ap.add_argument("--slice-robustness-json", default=str(DEFAULT_SLICE_ROBUSTNESS_JSON))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
