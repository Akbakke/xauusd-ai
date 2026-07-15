#!/usr/bin/env python3
"""Materialize smart seq520 XAU smoke-train enablement package.

This gate is report-only. It binds the latest smart smoke-readiness and
trainability-readiness reports to one explicit ``SMART_SEQ520_XAU_SMOKE_``
vedtak package, then exercises only the smart smoke wrapper dry-run path to
capture the exact future capped smoke-training command. It never starts
training, replay, IQL, shadow, live or promotion paths.
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


DEFAULT_SMOKE_READINESS_JSON = (
    REPORTS_ROOT
    / "entry_smart_seq520_smoke_readiness_20260630_v1/ENTRY_SMART_SEQ520_SMOKE_READINESS_latest.json"
)
DEFAULT_TRAINABILITY_READINESS_JSON = (
    REPORTS_ROOT
    / "entry_smart_seq520_trainability_readiness_20260630_v1/ENTRY_SMART_SEQ520_TRAINABILITY_READINESS_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_smart_seq520_smoke_train_enablement_20260715_v1"

READY_SMOKE_DECISION = "READY_FOR_SMART_SEQ520_SMOKE_MANIFEST_REVIEW"
READY_TRAINABILITY_DECISION = "READY_FOR_SMART_SEQ520_TRAINABILITY_REVIEW"
READY_DECISION = "ENTRY_SMART_SEQ520_SMOKE_TRAIN_ENABLEMENT_READY_FOR_EXPLICIT_EXECUTION"
BLOCKED_DECISION = "BLOCKED_BY_ENTRY_SMART_SEQ520_SMOKE_TRAIN_ENABLEMENT_PACKAGE"
VEDTAK_PREFIX = "SMART_SEQ520_XAU_SMOKE_"
REQUIRED_DIRECTION_ENV = {
    "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT": "8.00",
    "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT": "3.00",
    "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS": "8",
    "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE": "3",
    "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS": "6",
    "ENTRY_CKPT_DIRECTION_SLICE_GUARD": "1",
    "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION": "mean_max",
    "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT": "4.00",
    "ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS": "15.0",
    "ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN": "0.10",
    "ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT": "8.00",
    "ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS": "8",
    "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION": "0.50",
    "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR": "0.10",
    "ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN": "0.10",
}


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


def _line_with_prefix(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line
    return ""


def _future_train_contract(smoke_readiness: dict[str, Any], trainability: dict[str, Any]) -> dict[str, Any]:
    trainability_contract = trainability.get("future_train_contract")
    if isinstance(trainability_contract, dict) and trainability_contract:
        return trainability_contract
    contracts = smoke_readiness.get("future_command_contracts")
    if isinstance(contracts, dict) and isinstance(contracts.get("smart_smoke_train"), dict):
        return contracts["smart_smoke_train"]
    return {}


def _direction_env_template(contract: dict[str, Any]) -> dict[str, Any]:
    env = contract.get("direction_balance_env_template")
    return env if isinstance(env, dict) else {}


def _dry_run_wrapper(
    *,
    vedtak: str,
    device: str,
    epochs: int,
    batch_size: int,
    mem_cap: str,
    swap_cap: str,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "ENTRY_FOUNDATION_SMOKE_RUN_MEM": mem_cap,
            "ENTRY_FOUNDATION_SMOKE_RUN_SWAP": swap_cap,
        }
    )
    argv = [
        str(REPO / "scripts/entry_next_edge_control.sh"),
        "smart-smoke-train",
        "--vedtak",
        vedtak,
        "--require-edge-audit",
        "--dry-run",
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
    ]
    proc = subprocess.run(
        argv,
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    capped_line = _line_with_prefix(combined, "Capped smoke train command:")
    smoke_line = _line_with_prefix(combined, "Smoke train command:")
    audit_line = _line_with_prefix(combined, "Post-smoke audit command:")
    return {
        "argv": argv,
        "returncode": int(proc.returncode),
        "stdout_tail": (proc.stdout or "")[-3000:],
        "stderr_tail": (proc.stderr or "")[-3000:],
        "capped_smoke_train_command": capped_line,
        "smoke_train_command": smoke_line,
        "post_smoke_audit_command": audit_line,
        "has_capped_run": "scripts/gx1_capped_run.sh" in capped_line,
        "has_mem_cap": f"--mem {mem_cap}" in capped_line,
        "has_swap_cap": f"--swap {swap_cap}" in capped_line,
        "has_num_workers_zero": "--num-workers 0" in capped_line,
        "has_global_prior_match": "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT=8.00" in capped_line
        and "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE=0.02" in capped_line
        and "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE=0.10" in capped_line,
        "has_prior_match": "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT=3.00" in capped_line
        and "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE=0.02" in capped_line,
        "has_hard_red_stop": "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE=3" in capped_line
        and "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS=6" in capped_line,
        "has_utility_margin": "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT=4.00" in capped_line
        and "ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS=15.0" in capped_line
        and "ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN=0.10" in capped_line,
        "has_flat_starvation": "ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT=8.00" in capped_line
        and "ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE=0.10" in capped_line
        and "ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS=8" in capped_line
        and "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION=0.50" in capped_line
        and "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR=0.10" in capped_line
        and "ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN=0.10" in capped_line,
        "has_xau_repair_heads": "--enable-xau-direction-repair-heads" in capped_line,
        "has_strict_edge_audit": "--require-edge" in audit_line and "--edge-test-scope strict" in audit_line,
        "trainer_started": False,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Smart Seq520 XAU Smoke Train Enablement",
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
        f"`{report['wrapper_dry_run']['capped_smoke_train_command']}`",
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
    smoke_readiness_json = Path(args.smoke_readiness_json).expanduser().resolve()
    trainability_readiness_json = Path(args.trainability_readiness_json).expanduser().resolve()
    smoke_readiness = _read_json_or_empty(smoke_readiness_json)
    trainability = _read_json_or_empty(trainability_readiness_json)
    future_train = _future_train_contract(smoke_readiness, trainability)
    direction_env = _direction_env_template(future_train)
    git_status = _git_status_short()
    dry_run = _dry_run_wrapper(
        vedtak=vedtak or f"{VEDTAK_PREFIX}MISSING",
        device=str(args.device),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        mem_cap=str(args.mem_cap),
        swap_cap=str(args.swap_cap),
    )
    readiness_closed = (
        smoke_readiness.get("training_allowed") is False
        and smoke_readiness.get("execution_allowed_now") is False
        and trainability.get("training_allowed") is False
        and trainability.get("execution_allowed_now") is False
        and trainability.get("candidate_training_allowed") is False
        and trainability.get("replay_allowed") is False
        and trainability.get("iql_allowed") is False
        and trainability.get("shadow_live_promotion_allowed") is False
    )
    checks = [
        _check("explicit smart XAU smoke train vedtak is present", bool(vedtak), {"required_prefix": VEDTAK_PREFIX}),
        _check(
            "explicit smart XAU smoke train vedtak has correct prefix",
            vedtak.startswith(VEDTAK_PREFIX),
            {"vedtak": vedtak, "required_prefix": VEDTAK_PREFIX},
        ),
        _check("worktree is clean before smart smoke train enablement package", not git_status, {"git_status_short": git_status}),
        _check(
            "smart smoke readiness is ready",
            smoke_readiness.get("decision") == READY_SMOKE_DECISION,
            {"path": str(smoke_readiness_json), "decision": smoke_readiness.get("decision")},
        ),
        _check(
            "smart trainability readiness is ready",
            trainability.get("decision") == READY_TRAINABILITY_DECISION,
            {"path": str(trainability_readiness_json), "decision": trainability.get("decision")},
        ),
        _check(
            "upstream readiness remains report-only and keeps candidate/replay/IQL/shadow/live closed",
            readiness_closed,
            {
                "smoke_training_allowed": smoke_readiness.get("training_allowed"),
                "smoke_execution_allowed_now": smoke_readiness.get("execution_allowed_now"),
                "trainability_training_allowed": trainability.get("training_allowed"),
                "trainability_execution_allowed_now": trainability.get("execution_allowed_now"),
                "candidate_training_allowed": trainability.get("candidate_training_allowed"),
                "replay_allowed": trainability.get("replay_allowed"),
                "iql_allowed": trainability.get("iql_allowed"),
                "shadow_live_promotion_allowed": trainability.get("shadow_live_promotion_allowed"),
            },
        ),
        _check(
            "future smart smoke train contract keeps replay/IQL/shadow/live closed",
            future_train.get("starts_trainer") is True
            and future_train.get("starts_replay") is False
            and future_train.get("starts_iql_distillation") is False
            and future_train.get("touches_shadow_or_live") is False
            and future_train.get("requires_ram_cap") is True
            and future_train.get("requires_edge_audit") is True,
            {
                "starts_trainer": future_train.get("starts_trainer"),
                "starts_replay": future_train.get("starts_replay"),
                "starts_iql_distillation": future_train.get("starts_iql_distillation"),
                "touches_shadow_or_live": future_train.get("touches_shadow_or_live"),
                "requires_ram_cap": future_train.get("requires_ram_cap"),
                "requires_edge_audit": future_train.get("requires_edge_audit"),
            },
        ),
        _check(
            "future smart smoke train contract declares required prior-match and hard-red env",
            all(direction_env.get(key) == value for key, value in REQUIRED_DIRECTION_ENV.items()),
            {"required_env": REQUIRED_DIRECTION_ENV, "observed_env": direction_env},
        ),
        _check(
            "wrapper dry-run produces exact capped smart smoke command without starting trainer",
            dry_run["returncode"] == 0
            and dry_run["has_capped_run"]
            and dry_run["has_mem_cap"]
            and dry_run["has_swap_cap"]
            and dry_run["has_num_workers_zero"]
            and dry_run["has_global_prior_match"]
            and dry_run["has_prior_match"]
            and dry_run["has_hard_red_stop"]
            and dry_run["has_utility_margin"]
            and dry_run["has_flat_starvation"]
            and dry_run["has_xau_repair_heads"]
            and dry_run["has_strict_edge_audit"]
            and dry_run["trainer_started"] is False,
            dry_run,
        ),
        _check(
            "smart smoke train enablement package itself never trains, replays, distills, promotes, shadows, or starts live",
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
    json_path = out_dir / f"ENTRY_SMART_SEQ520_SMOKE_TRAIN_ENABLEMENT_{timestamp}.json"
    md_path = out_dir / f"ENTRY_SMART_SEQ520_SMOKE_TRAIN_ENABLEMENT_{timestamp}.md"
    report = {
        "schema_version": "entry_smart_seq520_smoke_train_enablement_package_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "vedtak": vedtak,
        "vedtak_prefix_required": VEDTAK_PREFIX,
        "smoke_readiness_json": str(smoke_readiness_json),
        "smoke_readiness_json_sha256": _sha256_file(smoke_readiness_json) if smoke_readiness_json.exists() else "",
        "trainability_readiness_json": str(trainability_readiness_json),
        "trainability_readiness_json_sha256": (
            _sha256_file(trainability_readiness_json) if trainability_readiness_json.exists() else ""
        ),
        "resource_caps": {"mem": str(args.mem_cap), "swap": str(args.swap_cap), "num_workers": 0},
        "wrapper_dry_run": dry_run,
        "future_train_contract": future_train,
        "clean_git_required": True,
        "clean_git_observed": not git_status,
        "git_status_short": git_status,
        "checks": checks,
        "failures": failures,
        "training_allowed": False,
        "smart_smoke_training_allowed_with_this_package": bool(ready),
        "candidate_training_allowed": False,
        "replay_allowed": False,
        "iql_allowed": False,
        "trainer_started": False,
        "replay_started": False,
        "iql_distillation_started": False,
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "run bounded smart smoke train wrapper with this exact package, cgroup RAM cap and hard-red monitoring; candidate/replay/IQL remain closed"
            if ready
            else "repair smart smoke train enablement package blockers before any trainer start"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_SMART_SEQ520_SMOKE_TRAIN_ENABLEMENT_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_SMART_SEQ520_SMOKE_TRAIN_ENABLEMENT_latest.md").write_text(
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
    ap.add_argument("--smoke-readiness-json", default=str(DEFAULT_SMOKE_READINESS_JSON))
    ap.add_argument("--trainability-readiness-json", default=str(DEFAULT_TRAINABILITY_READINESS_JSON))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mem-cap", default="22G")
    ap.add_argument("--swap-cap", default="2G")
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
