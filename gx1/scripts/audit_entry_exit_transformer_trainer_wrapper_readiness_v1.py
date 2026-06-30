#!/usr/bin/env python3
"""Audit active Exit Transformer trainer wrapper readiness.

This gate proves the future train command is wired but fail-closed. It checks
the active training-plan/readiness report and exercises only rejection paths on
the wrapper. It never starts a trainer, replay, IQL, shadow, live or promotion
path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT, REPO


DEFAULT_TRAINING_PLAN_JSON = (
    REPORTS_ROOT
    / "entry_exit_transformer_training_plan_readiness_20260630_v1/ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READINESS_latest.json"
)
DEFAULT_WRAPPER_PATH = REPO / "scripts/run_entry_exit_transformer_train.sh"
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_exit_transformer_trainer_wrapper_readiness_20260630_v1"

READY_TRAINING_PLAN_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW"
READY_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READY_FOR_IMPLEMENTATION_REVIEW"
BLOCKED_DECISION = "BLOCKED_BY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READINESS"
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


def _run_wrapper_case(wrapper: Path, args: list[str], required_text: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["ENTRY_EXIT_TRANSFORMER_TRAIN_MEM_CAP"] = "8G"
    env["ENTRY_EXIT_TRANSFORMER_TRAIN_SWAP_CAP"] = "1G"
    proc = subprocess.run(
        [str(wrapper), *args],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    return {
        "argv": [str(wrapper), *args],
        "returncode": int(proc.returncode),
        "required_text": required_text,
        "required_text_found": required_text in combined,
        "stdout_tail": (proc.stdout or "")[-800:],
        "stderr_tail": (proc.stderr or "")[-800:],
    }


def _source_review(wrapper: Path) -> dict[str, Any]:
    text = wrapper.read_text(encoding="utf-8") if wrapper.exists() else ""
    required_tokens = {
        "vedtak_prefix": VEDTAK_PREFIX,
        "training_plan_ready_decision": READY_TRAINING_PLAN_DECISION,
        "trainer_disabled_flag": "TRAINER_IMPLEMENTATION_ENABLED=0",
        "trainer_disabled_fatal": "active Exit Transformer trainer implementation is not enabled",
        "train_execution_review_json": "TRAIN_EXECUTION_REVIEW_JSON",
        "train_execution_review_ready_decision": "ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_READY_FOR_EXPLICIT_VEDTAK_PACKAGE",
        "post_train_audit_contract_json": "POST_TRAIN_AUDIT_CONTRACT_JSON",
        "post_train_audit_contract_ready_decision": "ENTRY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT_READY",
        "feature_alignment_json": "FEATURE_ALIGNMENT_JSON",
        "feature_alignment_ready_decision": "ENTRY_EXIT_FEATURE_ALIGNMENT_READY_FOR_EXIT_TRANSFORMER_TRAINING_REVIEW",
        "capped_run": "scripts/gx1_capped_run.sh",
        "num_workers_guard": "NUM_WORKERS=0",
        "num_workers_train_arg": "--num-workers",
        "no_side_effect_statement": "does not train, replay, distill",
    }
    missing = {name: token for name, token in required_tokens.items() if token not in text}
    return {
        "ready": bool(wrapper.exists() and os.access(wrapper, os.X_OK) and not missing),
        "wrapper_path": str(wrapper),
        "wrapper_exists": wrapper.exists(),
        "wrapper_executable": bool(wrapper.exists() and os.access(wrapper, os.X_OK)),
        "wrapper_sha256": _sha256_file(wrapper) if wrapper.exists() else "",
        "required_tokens": required_tokens,
        "missing_tokens": missing,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Exit Transformer Trainer Wrapper Readiness",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Exit training allowed: `{report['exit_training_allowed']}`",
        f"- Exit training allowed with explicit vedtak: `{report['exit_training_allowed_with_explicit_vedtak']}`",
        f"- Trainer started: `{report['trainer_started']}`",
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
    wrapper = Path(args.wrapper_path).expanduser().resolve()
    training_plan = _read_json_or_empty(training_plan_json)
    source_review = _source_review(wrapper)
    no_vedtak_case = _run_wrapper_case(wrapper, [], "--vedtak is required") if wrapper.exists() else {}
    bad_vedtak_case = (
        _run_wrapper_case(wrapper, ["--vedtak", "BAD_EXIT_TRAIN", "--dry-run"], f"must start with {VEDTAK_PREFIX}")
        if wrapper.exists()
        else {}
    )
    checks = [
        _check("active Exit Transformer training plan readiness exists", training_plan_json.exists(), {"path": str(training_plan_json)}),
        _check(
            "active Exit Transformer training plan readiness is ready",
            str(training_plan.get("decision")) == READY_TRAINING_PLAN_DECISION,
            {"decision": training_plan.get("decision"), "required": READY_TRAINING_PLAN_DECISION},
        ),
        _check("active Exit Transformer train wrapper is executable and fail-closed in source", bool(source_review.get("ready")), source_review),
        _check(
            "active Exit Transformer train wrapper rejects missing vedtak before side effects",
            bool(no_vedtak_case)
            and no_vedtak_case.get("returncode") == 2
            and no_vedtak_case.get("required_text_found") is True,
            no_vedtak_case,
        ),
        _check(
            "active Exit Transformer train wrapper rejects wrong vedtak prefix before side effects",
            bool(bad_vedtak_case)
            and bad_vedtak_case.get("returncode") == 2
            and bad_vedtak_case.get("required_text_found") is True,
            bad_vedtak_case,
        ),
        _check(
            "trainer wrapper readiness never trains, replays, distills, promotes, shadows, or starts live",
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
    json_path = out_dir / f"ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READINESS_{timestamp}.json"
    md_path = out_dir / f"ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READINESS_{timestamp}.md"
    report = {
        "schema_version": "entry_exit_transformer_trainer_wrapper_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "training_plan_json": str(training_plan_json),
        "training_plan_json_sha256": _sha256_file(training_plan_json) if training_plan_json.exists() else "",
        "wrapper_path": str(wrapper),
        "wrapper_sha256": _sha256_file(wrapper) if wrapper.exists() else "",
        "source_review": source_review,
        "wrapper_rejection_cases": {
            "missing_vedtak": no_vedtak_case,
            "bad_vedtak_prefix": bad_vedtak_case,
        },
        "checks": checks,
        "failures": failures,
        "exit_training_allowed": False,
        "exit_training_allowed_with_explicit_vedtak": False,
        "exit_iql_allowed": False,
        "trainer_started": False,
        "replay_started": False,
        "iql_distillation_started": False,
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "implement active Exit Transformer trainer core plus pretrain-manifest audit; training still closed"
            if ready
            else "repair active Exit Transformer trainer wrapper readiness before trainer implementation review"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READINESS_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READINESS_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": decision,
                    "failures": failures,
                    "json_path": str(json_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
    if args.fail_on_not_ready and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-plan-json", default=str(DEFAULT_TRAINING_PLAN_JSON))
    ap.add_argument("--wrapper-path", default=str(DEFAULT_WRAPPER_PATH))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
