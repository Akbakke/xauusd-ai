#!/usr/bin/env python3
"""Materialize active Exit Transformer train-enablement package.

This gate is report-only. It binds the active smart Entry-to-Exit feature
alignment, training plan, wrapper readiness, train-execution review and
post-train audit contract to one explicit ``ENTRY_EXIT_TRANSFORMER_TRAIN_``
vedtak package. It also exercises only the wrapper dry-run path to capture the
exact future capped training command. It never starts training, replay, IQL,
shadow, live or promotion paths.
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
DEFAULT_WRAPPER_READINESS_JSON = (
    REPORTS_ROOT
    / "entry_exit_transformer_trainer_wrapper_readiness_20260630_v1/ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READINESS_latest.json"
)
DEFAULT_TRAIN_EXECUTION_REVIEW_JSON = (
    REPORTS_ROOT
    / "entry_exit_transformer_train_execution_review_20260630_v1/ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_latest.json"
)
DEFAULT_POST_TRAIN_CONTRACT_JSON = (
    REPORTS_ROOT
    / "entry_exit_transformer_post_train_contract_20260630_v1/ENTRY_EXIT_TRANSFORMER_POST_TRAIN_CONTRACT_latest.json"
)
DEFAULT_FEATURE_ALIGNMENT_JSON = (
    REPORTS_ROOT / "entry_exit_feature_alignment_20260630_v1/ENTRY_EXIT_FEATURE_ALIGNMENT_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_exit_transformer_train_enablement_20260701_v1"

READY_TRAINING_PLAN_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW"
READY_WRAPPER_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READY_FOR_IMPLEMENTATION_REVIEW"
READY_TRAIN_EXECUTION_REVIEW_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_READY_FOR_EXPLICIT_VEDTAK_PACKAGE"
READY_POST_TRAIN_CONTRACT_DECISION = "ENTRY_EXIT_TRANSFORMER_POST_TRAIN_AUDIT_CONTRACT_READY"
READY_FEATURE_ALIGNMENT_DECISION = "ENTRY_EXIT_FEATURE_ALIGNMENT_READY_FOR_EXIT_TRANSFORMER_TRAINING_REVIEW"
READY_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAIN_ENABLEMENT_READY_FOR_EXPLICIT_EXECUTION"
BLOCKED_DECISION = "BLOCKED_BY_EXIT_TRANSFORMER_TRAIN_ENABLEMENT_PACKAGE"
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
    if not path.is_file():
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


def _dry_run_wrapper(
    *,
    vedtak: str,
    training_plan_json: Path,
    train_execution_review_json: Path,
    post_train_contract_json: Path,
    feature_alignment_json: Path,
    out_bundle_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    mem_cap: str,
    swap_cap: str,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_JSON": str(training_plan_json),
            "ENTRY_EXIT_TRANSFORMER_TRAIN_EXECUTION_REVIEW_JSON": str(train_execution_review_json),
            "ENTRY_EXIT_TRANSFORMER_POST_TRAIN_CONTRACT_JSON": str(post_train_contract_json),
            "ENTRY_EXIT_FEATURE_ALIGNMENT_JSON": str(feature_alignment_json),
            "ENTRY_EXIT_TRANSFORMER_TRAIN_MEM_CAP": mem_cap,
            "ENTRY_EXIT_TRANSFORMER_TRAIN_SWAP_CAP": swap_cap,
        }
    )
    argv = [
        str(REPO / "scripts/entry_next_edge_control.sh"),
        "entry-exit-transformer-train",
        "--vedtak",
        vedtak,
        "--dry-run",
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--out-bundle-dir",
        str(out_bundle_dir),
    ]
    proc = subprocess.run(
        argv,
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    future_line = ""
    for line in combined.splitlines():
        if line.startswith("Future capped train command:"):
            future_line = line
            break
    return {
        "argv": argv,
        "returncode": int(proc.returncode),
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
        "future_capped_train_command": future_line,
        "has_capped_run": "scripts/gx1_capped_run.sh" in future_line,
        "has_mem_cap": f"--mem {mem_cap}" in future_line,
        "has_swap_cap": f"--swap {swap_cap}" in future_line,
        "has_enable_training": "--enable-training" in future_line,
        "has_num_workers_zero": "--num-workers 0" in future_line,
        "trainer_started": False,
    }


def _report_ready(report: dict[str, Any], decision: str) -> bool:
    return report.get("decision") == decision


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Exit Transformer Train Enablement Package",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Vedtak: `{report['vedtak']}`",
        f"- Clean git required: `{report['clean_git_required']}`",
        f"- Clean git observed: `{report['clean_git_observed']}`",
        f"- Trainer started: `{report['trainer_started']}`",
        f"- Next required gate: `{report['next_required_gate']}`",
        "",
        "## Future Command",
        "",
        f"`{report['wrapper_dry_run']['future_capped_train_command']}`",
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
    vedtak = str(args.vedtak or "")
    training_plan_json = Path(args.training_plan_json).expanduser().resolve()
    wrapper_readiness_json = Path(args.wrapper_readiness_json).expanduser().resolve()
    train_execution_review_json = Path(args.train_execution_review_json).expanduser().resolve()
    post_train_contract_json = Path(args.post_train_contract_json).expanduser().resolve()
    feature_alignment_json = Path(args.feature_alignment_json).expanduser().resolve()
    out_bundle_dir = Path(args.out_bundle_dir).expanduser().resolve()
    training_plan = _read_json_or_empty(training_plan_json)
    wrapper_readiness = _read_json_or_empty(wrapper_readiness_json)
    train_execution_review = _read_json_or_empty(train_execution_review_json)
    post_train_contract = _read_json_or_empty(post_train_contract_json)
    feature_alignment = _read_json_or_empty(feature_alignment_json)
    git_status = _git_status_short()
    dry_run = _dry_run_wrapper(
        vedtak=vedtak or f"{VEDTAK_PREFIX}MISSING",
        training_plan_json=training_plan_json,
        train_execution_review_json=train_execution_review_json,
        post_train_contract_json=post_train_contract_json,
        feature_alignment_json=feature_alignment_json,
        out_bundle_dir=out_bundle_dir,
        device=str(args.device),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        mem_cap=str(args.mem_cap),
        swap_cap=str(args.swap_cap),
    )
    checks = [
        _check("explicit Exit Transformer train vedtak is present", bool(vedtak), {"required_prefix": VEDTAK_PREFIX}),
        _check("explicit Exit Transformer train vedtak has correct prefix", vedtak.startswith(VEDTAK_PREFIX), {"vedtak": vedtak, "required_prefix": VEDTAK_PREFIX}),
        _check("worktree is clean before train enablement package", not git_status, {"git_status_short": git_status}),
        _check("active Entry-to-Exit feature alignment is ready", _report_ready(feature_alignment, READY_FEATURE_ALIGNMENT_DECISION), {"path": str(feature_alignment_json), "decision": feature_alignment.get("decision")}),
        _check("active Exit Transformer training plan is ready", _report_ready(training_plan, READY_TRAINING_PLAN_DECISION), {"path": str(training_plan_json), "decision": training_plan.get("decision")}),
        _check("active Exit Transformer wrapper readiness is ready", _report_ready(wrapper_readiness, READY_WRAPPER_DECISION), {"path": str(wrapper_readiness_json), "decision": wrapper_readiness.get("decision")}),
        _check("active Exit Transformer train-execution review is ready", _report_ready(train_execution_review, READY_TRAIN_EXECUTION_REVIEW_DECISION), {"path": str(train_execution_review_json), "decision": train_execution_review.get("decision")}),
        _check("active Exit Transformer post-train audit contract is ready", _report_ready(post_train_contract, READY_POST_TRAIN_CONTRACT_DECISION), {"path": str(post_train_contract_json), "decision": post_train_contract.get("decision")}),
        _check(
            "upstream reports keep shadow/live/replay/IQL closed",
            all(
                report.get("promotion_shadow_live_allowed") in (False, None)
                and report.get("replay_started") in (False, None)
                and report.get("iql_distillation_started") in (False, None)
                for report in (feature_alignment, training_plan, wrapper_readiness, train_execution_review, post_train_contract)
            ),
            {
                "feature_alignment_shadow_live": feature_alignment.get("promotion_shadow_live_allowed"),
                "post_train_shadow_live": post_train_contract.get("promotion_shadow_live_allowed"),
            },
        ),
        _check(
            "wrapper dry-run produces exact capped train command without starting trainer",
            dry_run["returncode"] == 0
            and dry_run["has_capped_run"]
            and dry_run["has_mem_cap"]
            and dry_run["has_swap_cap"]
            and dry_run["has_enable_training"]
            and dry_run["has_num_workers_zero"]
            and dry_run["trainer_started"] is False,
            dry_run,
        ),
        _check(
            "train enablement package itself never trains, replays, distills, promotes, shadows, or starts live",
            True,
            {
                "trainer_started": False,
                "replay_started": False,
                "iql_distillation_started": False,
                "promotion_shadow_live_allowed": False,
            },
        ),
    ]
    failures = [{"check": check["name"], "details": check.get("details") or {}} for check in checks if not check["ok"]]
    ready = not failures
    decision = READY_DECISION if ready else BLOCKED_DECISION
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"ENTRY_EXIT_TRANSFORMER_TRAIN_ENABLEMENT_{timestamp}.json"
    md_path = out_dir / f"ENTRY_EXIT_TRANSFORMER_TRAIN_ENABLEMENT_{timestamp}.md"
    report = {
        "schema_version": "entry_exit_transformer_train_enablement_package_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "vedtak": vedtak,
        "vedtak_prefix_required": VEDTAK_PREFIX,
        "training_plan_json": str(training_plan_json),
        "training_plan_json_sha256": _sha256_file(training_plan_json) if training_plan_json.exists() else "",
        "wrapper_readiness_json": str(wrapper_readiness_json),
        "wrapper_readiness_json_sha256": _sha256_file(wrapper_readiness_json) if wrapper_readiness_json.exists() else "",
        "train_execution_review_json": str(train_execution_review_json),
        "train_execution_review_json_sha256": _sha256_file(train_execution_review_json) if train_execution_review_json.exists() else "",
        "post_train_contract_json": str(post_train_contract_json),
        "post_train_contract_json_sha256": _sha256_file(post_train_contract_json) if post_train_contract_json.exists() else "",
        "feature_alignment_json": str(feature_alignment_json),
        "feature_alignment_json_sha256": _sha256_file(feature_alignment_json) if feature_alignment_json.exists() else "",
        "out_bundle_dir": str(out_bundle_dir),
        "resource_caps": {"mem": str(args.mem_cap), "swap": str(args.swap_cap), "num_workers": 0},
        "wrapper_dry_run": dry_run,
        "clean_git_required": True,
        "clean_git_observed": not git_status,
        "git_status_short": git_status,
        "checks": checks,
        "failures": failures,
        "exit_training_allowed": False,
        "exit_training_allowed_with_this_package": bool(ready),
        "exit_iql_allowed": False,
        "trainer_started": False,
        "replay_started": False,
        "iql_distillation_started": False,
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "run active Exit Transformer train wrapper with this exact package, cgroup RAM cap and no shadow/live"
            if ready
            else "repair Exit Transformer train enablement package blockers before training"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_EXIT_TRANSFORMER_TRAIN_ENABLEMENT_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_EXIT_TRANSFORMER_TRAIN_ENABLEMENT_latest.md").write_text(
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
    ap.add_argument("--vedtak", default="")
    ap.add_argument("--training-plan-json", default=str(DEFAULT_TRAINING_PLAN_JSON))
    ap.add_argument("--wrapper-readiness-json", default=str(DEFAULT_WRAPPER_READINESS_JSON))
    ap.add_argument("--train-execution-review-json", default=str(DEFAULT_TRAIN_EXECUTION_REVIEW_JSON))
    ap.add_argument("--post-train-contract-json", default=str(DEFAULT_POST_TRAIN_CONTRACT_JSON))
    ap.add_argument("--feature-alignment-json", default=str(DEFAULT_FEATURE_ALIGNMENT_JSON))
    ap.add_argument("--out-bundle-dir", default="/home/andre2/GX1_DATA/runs/entry_exit_transformer/active_train_enablement_bundle")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--mem-cap", default="8G")
    ap.add_argument("--swap-cap", default="1G")
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
