#!/usr/bin/env python3
"""Materialize active Exit Transformer pretrain manifest.

This gate imports the active trainer core, runs a tiny CPU-only finite forward
preflight, hashes the active contracts and keeps real Exit training closed.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.models.exit_sequence_transformer.train_v1 import EXPECTED_HEADS, run_preflight
from gx1.scripts.verify_entry_foundation_state_v1 import REPORTS_ROOT


DEFAULT_TRAINING_PLAN_JSON = (
    REPORTS_ROOT
    / "entry_exit_transformer_training_plan_readiness_20260630_v1/ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READINESS_latest.json"
)
DEFAULT_WRAPPER_READINESS_JSON = (
    REPORTS_ROOT
    / "entry_exit_transformer_trainer_wrapper_readiness_20260630_v1/ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READINESS_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_exit_transformer_pretrain_manifest_20260630_v1"

READY_TRAINING_PLAN_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAINING_PLAN_READY_FOR_VEDTAK_REVIEW"
READY_WRAPPER_DECISION = "ENTRY_EXIT_TRANSFORMER_TRAINER_WRAPPER_READY_FOR_IMPLEMENTATION_REVIEW"
READY_DECISION = "ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_READY_FOR_TRAIN_EXECUTION_REVIEW"
BLOCKED_DECISION = "BLOCKED_BY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST"


def _check(name: str, condition: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(condition), "details": details or {}}


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Exit Transformer Pretrain Manifest",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Preflight decision: `{report['preflight_manifest'].get('decision')}`",
        f"- Output heads: `{report['preflight_manifest'].get('output_heads')}`",
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
    wrapper_readiness_json = Path(args.wrapper_readiness_json).expanduser().resolve()
    training_plan_report = _read_json_or_empty(training_plan_json)
    wrapper_readiness = _read_json_or_empty(wrapper_readiness_json)
    manifest_path = out_dir / "entry_exit_transformer_pretrain_manifest.json"
    preflight_manifest: dict[str, Any] = {}
    preflight_error = ""
    try:
        preflight_manifest = run_preflight(
            argparse.Namespace(
                training_plan_json=str(training_plan_json),
                preflight_only=True,
                out_manifest_json=str(manifest_path),
                split=str(args.split),
                max_episodes=int(args.max_episodes),
                device="cpu",
            )
        )
    except Exception as exc:  # pragma: no cover - details are reported as gate failure.
        preflight_error = str(exc)
    resources = (
        (training_plan_report.get("training_plan") or {}).get("resource_guardrails")
        if isinstance(training_plan_report.get("training_plan"), dict)
        else {}
    )
    checks = [
        _check("active Exit Transformer training plan readiness exists", training_plan_json.exists(), {"path": str(training_plan_json)}),
        _check(
            "active Exit Transformer training plan readiness is ready",
            str(training_plan_report.get("decision")) == READY_TRAINING_PLAN_DECISION,
            {"decision": training_plan_report.get("decision"), "required": READY_TRAINING_PLAN_DECISION},
        ),
        _check("active Exit Transformer trainer wrapper readiness exists", wrapper_readiness_json.exists(), {"path": str(wrapper_readiness_json)}),
        _check(
            "active Exit Transformer trainer wrapper readiness is ready",
            str(wrapper_readiness.get("decision")) == READY_WRAPPER_DECISION,
            {"decision": wrapper_readiness.get("decision"), "required": READY_WRAPPER_DECISION},
        ),
        _check(
            "active Exit Transformer trainer core finite forward preflight passes",
            preflight_manifest.get("decision") == "PASS"
            and preflight_manifest.get("output_heads") == list(EXPECTED_HEADS)
            and all((preflight_manifest.get("finite_by_head") or {}).values()),
            {"preflight_error": preflight_error, "preflight_manifest": preflight_manifest},
        ),
        _check(
            "pretrain manifest preserves RAM guardrails",
            int(resources.get("num_workers") if resources.get("num_workers") is not None else -1) == 0
            and float(resources.get("max_process_rss_gib") or 0.0) <= 8.0
            and float(resources.get("abort_if_mem_available_below_gib") or 0.0) >= 8.0,
            {"resource_guardrails": resources},
        ),
        _check(
            "pretrain manifest never trains, replays, distills, promotes, shadows, or starts live",
            True,
            {
                "trainer_started": False,
                "optimizer_steps": 0,
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
    json_path = out_dir / f"ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_{timestamp}.json"
    md_path = out_dir / f"ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_{timestamp}.md"
    report = {
        "schema_version": "entry_exit_transformer_pretrain_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "training_plan_json": str(training_plan_json),
        "wrapper_readiness_json": str(wrapper_readiness_json),
        "preflight_manifest_json": str(manifest_path),
        "preflight_manifest": preflight_manifest,
        "preflight_error": preflight_error,
        "checks": checks,
        "failures": failures,
        "exit_training_allowed": False,
        "exit_training_allowed_with_explicit_vedtak": False,
        "exit_iql_allowed": False,
        "trainer_started": False,
        "optimizer_steps": 0,
        "replay_started": False,
        "iql_distillation_started": False,
        "promotion_shadow_live_allowed": False,
        "next_required_gate": (
            "train-execution enablement remains blocked until explicit review opens active Exit Transformer training"
            if ready
            else "repair active Exit Transformer pretrain manifest before train-execution review"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    (out_dir / "ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ENTRY_EXIT_TRANSFORMER_PRETRAIN_MANIFEST_latest.md").write_text(
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
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-episodes", type=int, default=4)
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
