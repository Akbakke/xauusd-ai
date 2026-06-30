"""Behavioral guardrails for the active Entry foundation-freeze.

This check exercises blocking and orientation paths only. It does not start
training, replay, shadow, promotion, live runners, or order placement.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/home/andre2/src/GX1_ENGINE")
PY = REPO / ".venv/bin/python"
DEFAULT_OUT_DIR = Path("/home/andre2/GX1_DATA/reports/entry_foundation_guardrails_20260628_v1")


@dataclass(frozen=True)
class CommandCase:
    name: str
    cmd: list[str]
    expected_returncode: int
    required_text: str
    env: dict[str, str] | None = None
    forbidden_texts: tuple[str, ...] = ()


def _run_case(case: CommandCase) -> dict[str, Any]:
    env = os.environ.copy()
    if case.env:
        env.update(case.env)
    proc = subprocess.run(
        case.cmd,
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != case.expected_returncode:
        raise RuntimeError(
            f"{case.name}: expected returncode {case.expected_returncode}, got {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    if case.required_text not in combined:
        raise RuntimeError(
            f"{case.name}: required text not found: {case.required_text!r}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    found_forbidden = [text for text in case.forbidden_texts if text in combined]
    if found_forbidden:
        raise RuntimeError(
            f"{case.name}: forbidden text found: {found_forbidden!r}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return {
        "name": case.name,
        "status": "PASS",
        "expected_returncode": case.expected_returncode,
        "observed_returncode": proc.returncode,
        "required_text": case.required_text,
        "forbidden_texts": list(case.forbidden_texts),
        "stdout_tail": (proc.stdout or "")[-1200:],
        "stderr_tail": (proc.stderr or "")[-1200:],
    }


def _source_contract_checks() -> list[dict[str, Any]]:
    control = (REPO / "scripts/entry_next_edge_control.sh").read_text(encoding="utf-8")
    handover = (REPO / "scripts/gx1_handover.sh").read_text(encoding="utf-8")

    handover_gate = (
        'if [[ "${GX1_ALLOW_LEGACY_HANDOVER:-}" != "20260627_ALLOW_LEGACY_HANDOVER" ]]; then'
    )
    active_block_start = handover.find(handover_gate)
    active_block_exit = handover.find("  exit 0", active_block_start)
    legacy_body_start = handover.find("OPEN-MORE WAVE ARMED")
    rows = [
        {
            "name": "control_verify_dispatches_to_foundation_state",
            "ok": "verify_entry_foundation_state_v1" in control,
        },
        {
            "name": "control_blocks_shadow_live_train_legacy_paths",
            "ok": all(token in control for token in ("preview-shadow", "start-shadow", "train|retrain|promote|pin|live")),
        },
        {
            "name": "control_exposes_foundation_adoption_candidate_report",
            "ok": "foundation-adoption-candidate" in control
            and "verify_entry_foundation_adoption_candidate_v1" in control,
        },
        {
            "name": "control_exposes_foundation_activation_plan_report",
            "ok": "foundation-activation-plan" in control
            and "plan_entry_foundation_activation_v1" in control,
        },
        {
            "name": "control_exposes_vedtak_gated_foundation_activation_apply",
            "ok": "foundation-activation-apply" in control
            and "apply_entry_foundation_activation_v1" in control,
        },
        {
            "name": "control_exposes_vedtak_gated_foundation_activation_post_apply",
            "ok": "foundation-activation-post-apply" in control
            and "run_entry_foundation_activation_post_apply_v1" in control,
        },
        {
            "name": "handover_legacy_requires_explicit_env_token",
            "ok": active_block_start >= 0,
        },
        {
            "name": "handover_default_exits_before_legacy_body",
            "ok": active_block_start >= 0
            and active_block_exit > active_block_start
            and legacy_body_start > active_block_exit,
        },
        {
            "name": "handover_default_announces_active_foundation",
            "ok": "active Entry foundation seq146" in handover
            and "docs/ENTRY_FOUNDATION_AUDIT_20260628.md" in handover
            and "scripts/entry_next_edge_control.sh" in handover,
        },
    ]
    failed = [row["name"] for row in rows if not row["ok"]]
    if failed:
        raise RuntimeError(f"source contract checks failed: {failed}")
    return rows


def _readiness_policy_checks() -> list[dict[str, Any]]:
    env = os.environ.copy()
    env["GX1_READINESS_REPORT_POLICY_SNAPSHOT"] = "20260629_GUARDRAIL_POLICY_ONLY"
    proc = subprocess.run(
        ["scripts/entry_next_edge_control.sh", "readiness-report", "--json"],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "readiness policy snapshot failed\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"readiness policy snapshot did not emit JSON: {exc}\n{proc.stdout[:2000]}") from exc

    commands = payload.get("commands") if isinstance(payload.get("commands"), dict) else {}
    side_effects = payload.get("side_effects_started") if isinstance(payload.get("side_effects_started"), dict) else {}
    allowed_now = payload.get("allowed_now") if isinstance(payload.get("allowed_now"), list) else []
    status_summary = payload.get("status_summary") if isinstance(payload.get("status_summary"), dict) else {}

    safe_now = (
        "handover",
        "readiness_report",
        "readiness_report_json",
        "verify",
        "selftest",
        "foundation_guardrails",
        "foundation_activation_plan",
        "foundation_activation_apply_dry_run",
        "foundation_activation_post_apply_dry_run",
        "worktree_hygiene",
        "train_readiness_report",
        "candidate_readiness_report",
        "candidate_readiness_seq215_report",
        "replay_readiness_report",
        "replay_readiness_seq215_report",
        "feature_ai_inventory",
        "chart_geometry_audit",
        "candlestick_audit",
        "challenger_extension_manifest",
        "challenger_smart_extension_manifest",
        "smart_rebuild_preflight",
        "smart_post_rebuild_readiness",
        "smart_smoke_readiness",
        "smart_ablation_replay_plan",
        "stage_foundation_cleanup_dry_run",
        "iql_slice_audit",
        "entry_exit_materialize",
        "entry_exit_handoff",
        "entry_exit_reconstruction_audit",
        "entry_exit_state_reward_contract",
        "entry_exit_split_leakage_audit",
        "entry_exit_model_dataset_readiness",
        "entry_exit_feature_alignment",
        "entry_exit_transformer_architecture_readiness",
        "entry_exit_transformer_training_plan_readiness",
        "entry_exit_transformer_trainer_wrapper_readiness",
        "entry_exit_transformer_pretrain_manifest",
        "entry_exit_model_dataset_slice_robustness",
        "entry_exit_transformer_train_execution_review",
        "entry_exit_transformer_post_train_contract",
    )
    blocked_downstream = (
        "smoke_train",
        "smoke_train_seq215",
        "foundation_activation_apply",
        "foundation_activation_post_apply",
        "smart_smoke_manifest",
        "candidate_train",
        "candidate_train_seq215",
        "selective_edge",
        "replay_evidence",
        "iql_distill",
        "iql_student_trade_log",
        "iql_replay_evidence",
        "iql_compare",
        "entry_exit_transformer_train",
        "preview_shadow",
        "start_shadow",
        "live",
    )
    expected_commands = (
        "handover",
        "readiness_report",
        "readiness_report_json",
        "worktree_hygiene",
        "stage_foundation_cleanup_dry_run",
        "stage_foundation_cleanup_apply",
        "smoke_manifest",
        "smoke_manifest_seq215",
        "smoke_train",
        "smoke_train_seq215",
        "verify",
        "selftest",
        "foundation_guardrails",
        "foundation_activation_plan",
        "foundation_activation_apply_dry_run",
        "foundation_activation_apply",
        "foundation_activation_post_apply_dry_run",
        "foundation_activation_post_apply",
        "train_readiness_report",
        "candidate_readiness_report",
        "candidate_readiness_seq215_report",
        "replay_readiness_report",
        "replay_readiness_seq215_report",
        "feature_ai_inventory",
        "chart_geometry_audit",
        "candlestick_audit",
        "challenger_extension_manifest",
        "challenger_smart_extension_manifest",
        "smart_rebuild_preflight",
        "smart_post_rebuild_readiness",
        "smart_smoke_manifest",
        "smart_smoke_readiness",
        "smart_ablation_replay_plan",
        "candidate_train",
        "candidate_train_seq215",
        "selective_edge",
        "replay_evidence",
        "iql_distill",
        "iql_student_trade_log",
        "iql_replay_evidence",
        "iql_compare",
        "iql_slice_audit",
        "entry_exit_materialize",
        "entry_exit_handoff",
        "entry_exit_reconstruction_audit",
        "entry_exit_state_reward_contract",
        "entry_exit_split_leakage_audit",
        "entry_exit_model_dataset_readiness",
        "entry_exit_feature_alignment",
        "entry_exit_transformer_architecture_readiness",
        "entry_exit_transformer_training_plan_readiness",
        "entry_exit_transformer_trainer_wrapper_readiness",
        "entry_exit_transformer_pretrain_manifest",
        "entry_exit_model_dataset_slice_robustness",
        "entry_exit_transformer_train_execution_review",
        "entry_exit_transformer_post_train_contract",
        "entry_exit_transformer_train",
        "preview_shadow",
        "start_shadow",
        "live",
    )
    required_command_keys = (
        "argv",
        "allowed",
        "mode",
        "requires_vedtak",
        "requires_clean_git",
        "mutates_git_index",
        "starts_trainer",
        "starts_replay",
        "starts_iql_distillation",
        "touches_shadow_or_live",
        "description",
        "execution_allowed_now",
        "allowed_after_explicit_vedtak",
        "not_executable_now_reason",
    )
    bool_command_keys = (
        "allowed",
        "requires_vedtak",
        "requires_clean_git",
        "mutates_git_index",
        "starts_trainer",
        "starts_replay",
        "starts_iql_distillation",
        "touches_shadow_or_live",
        "execution_allowed_now",
        "allowed_after_explicit_vedtak",
    )
    command_schema_failures: list[dict[str, Any]] = []
    for name, command in sorted(commands.items()):
        if not isinstance(command, dict):
            command_schema_failures.append({"command": name, "failure": "command payload is not an object"})
            continue
        missing_keys = [key for key in required_command_keys if key not in command]
        non_bool_keys = [
            key
            for key in bool_command_keys
            if key in command and not isinstance(command.get(key), bool)
        ]
        if missing_keys:
            command_schema_failures.append({"command": name, "failure": "missing keys", "keys": missing_keys})
        if non_bool_keys:
            command_schema_failures.append({"command": name, "failure": "non-bool keys", "keys": non_bool_keys})
        if not isinstance(command.get("argv"), list) or not command.get("argv"):
            command_schema_failures.append({"command": name, "failure": "argv must be a non-empty list"})
        if not str(command.get("mode") or "").strip():
            command_schema_failures.append({"command": name, "failure": "mode must be non-empty"})
        if not str(command.get("description") or "").strip():
            command_schema_failures.append({"command": name, "failure": "description must be non-empty"})
        reason = command.get("not_executable_now_reason")
        if reason is not None and not str(reason).strip():
            command_schema_failures.append(
                {"command": name, "failure": "not_executable_now_reason must be null or non-empty"}
            )

    rows: list[dict[str, Any]] = [
        {
            "name": "readiness_policy_snapshot_json_parseable",
            "ok": payload.get("schema_version") == "entry_next_edge_readiness_report_v1",
            "details": {"schema_version": payload.get("schema_version")},
        },
        {
            "name": "readiness_policy_command_set_exact",
            "ok": set(commands) == set(expected_commands),
            "details": {
                "expected_commands": list(expected_commands),
                "observed_commands": sorted(commands),
                "missing_commands": sorted(set(expected_commands) - set(commands)),
                "unexpected_commands": sorted(set(commands) - set(expected_commands)),
            },
        },
        {
            "name": "readiness_policy_command_schema_complete",
            "ok": not command_schema_failures,
            "details": {
                "required_command_keys": list(required_command_keys),
                "bool_command_keys": list(bool_command_keys),
                "schema_failures": command_schema_failures,
            },
        },
        {
            "name": "readiness_policy_snapshot_report_only",
            "ok": payload.get("report_only") is True
            and payload.get("refresh_skipped") is True
            and all(value is False for value in side_effects.values()),
            "details": {"refresh_skipped": payload.get("refresh_skipped"), "side_effects_started": side_effects},
        },
        {
            "name": "readiness_policy_allowed_now_has_no_vedtak_placeholders",
            "ok": not any("--vedtak <id>" in str(item) for item in allowed_now),
            "details": {"allowed_now": allowed_now},
        },
        {
            "name": "readiness_policy_adoption_candidate_does_not_activate_without_vedtak",
            "ok": status_summary.get("activation_allowed_without_vedtak") is False,
            "details": {
                "foundation_adoption_candidate_ready": status_summary.get("foundation_adoption_candidate_ready"),
                "foundation_adoption_candidate_report": status_summary.get("foundation_adoption_candidate_report"),
                "activation_allowed_without_vedtak": status_summary.get("activation_allowed_without_vedtak"),
            },
        },
    ]

    for name in safe_now:
        command = commands.get(name) if isinstance(commands.get(name), dict) else {}
        rows.append(
            {
                "name": f"readiness_policy_safe_now_{name}",
                "ok": command.get("execution_allowed_now") is True
                and command.get("requires_vedtak") is False
                and command.get("mutates_git_index") is False
                and command.get("starts_trainer") is False
                and command.get("starts_replay") is False
                and command.get("starts_iql_distillation") is False
                and command.get("touches_shadow_or_live") is False,
                "details": command,
            }
        )

    for name in blocked_downstream:
        command = commands.get(name) if isinstance(commands.get(name), dict) else {}
        rows.append(
            {
                "name": f"readiness_policy_blocks_{name}",
                "ok": command.get("execution_allowed_now") is False
                and bool(command.get("not_executable_now_reason")),
                "details": command,
            }
        )

    rows.extend(
        [
            {
                "name": "readiness_policy_candidate_train_declares_trainer",
                "ok": (commands.get("candidate_train") or {}).get("starts_trainer") is True
                and (commands.get("candidate_train") or {}).get("requires_vedtak") is True
                and (commands.get("candidate_train") or {}).get("requires_clean_git") is True,
                "details": commands.get("candidate_train"),
            },
            {
                "name": "readiness_policy_candidate_train_seq215_declares_trainer",
                "ok": (commands.get("candidate_train_seq215") or {}).get("starts_trainer") is True
                and (commands.get("candidate_train_seq215") or {}).get("requires_vedtak") is True
                and (commands.get("candidate_train_seq215") or {}).get("requires_clean_git") is True
                and (commands.get("candidate_train_seq215") or {}).get("execution_allowed_now") is False,
                "details": commands.get("candidate_train_seq215"),
            },
            {
                "name": "readiness_policy_smoke_train_seq215_declares_ram_edge_seq215_contract",
                "ok": (commands.get("smoke_train_seq215") or {}).get("requires_seq215_vedtak") is True
                and (commands.get("smoke_train_seq215") or {}).get("requires_edge_audit") is True
                and (commands.get("smoke_train_seq215") or {}).get("requires_ram_cap") is True
                and (commands.get("smoke_train_seq215") or {}).get("ram_cap_runner")
                == "scripts/gx1_capped_run.sh"
                and (commands.get("smoke_train_seq215") or {}).get("num_workers") == 0
                and (commands.get("smoke_train_seq215") or {}).get("specialist_contract_mode")
                == "challenger_seq215"
                and (commands.get("smoke_train_seq215") or {}).get("expected_signal_dim") == 215
                and (commands.get("smoke_train_seq215") or {}).get("required_training_specialist_count") == 8
                and (commands.get("smoke_train_seq215") or {}).get("requires_exact_specialist_contract_proof")
                is True,
                "details": commands.get("smoke_train_seq215"),
            },
            {
                "name": "readiness_policy_candidate_train_seq215_declares_ram_edge_seq215_contract",
                "ok": (commands.get("candidate_train_seq215") or {}).get("requires_seq215_vedtak") is True
                and (commands.get("candidate_train_seq215") or {}).get("requires_candidate_readiness") is True
                and (commands.get("candidate_train_seq215") or {}).get("requires_smoke_bundle_edge_audit") is True
                and (commands.get("candidate_train_seq215") or {}).get("requires_ram_cap") is True
                and (commands.get("candidate_train_seq215") or {}).get("ram_cap_runner")
                == "scripts/gx1_capped_run.sh"
                and (commands.get("candidate_train_seq215") or {}).get("num_workers") == 0
                and (commands.get("candidate_train_seq215") or {}).get("specialist_contract_mode")
                == "challenger_seq215"
                and (commands.get("candidate_train_seq215") or {}).get("expected_signal_dim") == 215
                and (commands.get("candidate_train_seq215") or {}).get("required_training_specialist_count") == 8
                and (commands.get("candidate_train_seq215") or {}).get("requires_exact_specialist_contract_proof")
                is True,
                "details": commands.get("candidate_train_seq215"),
            },
            {
                "name": "readiness_policy_iql_distill_declares_iql_side_effect",
                "ok": (commands.get("iql_distill") or {}).get("starts_iql_distillation") is True
                and (commands.get("iql_distill") or {}).get("requires_vedtak") is True,
                "details": commands.get("iql_distill"),
            },
            {
                "name": "readiness_policy_entry_exit_transformer_train_declares_trainer",
                "ok": (commands.get("entry_exit_transformer_train") or {}).get("starts_trainer") is True
                and (commands.get("entry_exit_transformer_train") or {}).get("requires_vedtak") is True
                and (commands.get("entry_exit_transformer_train") or {}).get("requires_clean_git") is True
                and (commands.get("entry_exit_transformer_train") or {}).get("execution_allowed_now") is False,
                "details": commands.get("entry_exit_transformer_train"),
            },
            {
                "name": "readiness_policy_shadow_live_declares_live_touch",
                "ok": all((commands.get(name) or {}).get("touches_shadow_or_live") is True for name in ("preview_shadow", "start_shadow", "live")),
                "details": {
                    name: commands.get(name)
                    for name in ("preview_shadow", "start_shadow", "live")
                },
            },
        ]
    )

    failed = [row["name"] for row in rows if not row["ok"]]
    if failed:
        raise RuntimeError(f"readiness policy checks failed: {failed}")
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = [
        CommandCase(
            name="control_readiness_report_active_foundation",
            cmd=["scripts/entry_next_edge_control.sh", "readiness-report", "--json"],
            expected_returncode=0,
            required_text="entry_next_edge_readiness_report_v1",
            env={"GX1_READINESS_REPORT_POLICY_SNAPSHOT": "20260629_GUARDRAIL_POLICY_ONLY"},
        ),
        CommandCase(
            name="control_preview_shadow_blocked",
            cmd=["scripts/entry_next_edge_control.sh", "preview-shadow"],
            expected_returncode=2,
            required_text="blocked by active Entry foundation-freeze",
        ),
        CommandCase(
            name="control_start_shadow_blocked",
            cmd=["scripts/entry_next_edge_control.sh", "start-shadow"],
            expected_returncode=2,
            required_text="blocked by active Entry foundation-freeze",
        ),
        CommandCase(
            name="control_verify_shadow_blocked",
            cmd=["scripts/entry_next_edge_control.sh", "verify-shadow"],
            expected_returncode=2,
            required_text="blocked by active Entry foundation-freeze",
        ),
        CommandCase(
            name="direct_no_xgb_shadow_launcher_blocked",
            cmd=["bash", "scripts/run_entry_tabular_no_xgb_shadow_only.sh", "--dry-run"],
            expected_returncode=2,
            required_text="blocked by active Entry foundation-freeze",
        ),
        CommandCase(
            name="legacy_plan_verifier_closed",
            cmd=[str(PY), "-m", "gx1.scripts.verify_entry_next_edge_plan_state_v1"],
            expected_returncode=2,
            required_text="LEGACY_PLAN_CLOSED",
        ),
        CommandCase(
            name="handover_points_at_foundation",
            cmd=["bash", "scripts/gx1_handover.sh"],
            expected_returncode=0,
            required_text="active Entry foundation seq146",
            env={
                "GX1_HANDOVER_SKIP_GUARDRAILS": "1",
                "GX1_HANDOVER_SKIP_TRAIN_READINESS": "1",
            },
            forbidden_texts=("OPEN-MORE WAVE ARMED",),
        ),
        CommandCase(
            name="generic_train_blocked",
            cmd=["scripts/entry_next_edge_control.sh", "train"],
            expected_returncode=2,
            required_text="blocked by active Entry foundation-freeze",
        ),
    ]
    source_checks = _source_contract_checks()
    readiness_policy_checks = _readiness_policy_checks()
    results = [_run_case(case) for case in cases]
    report: dict[str, Any] = {
        "schema_version": "entry_foundation_guardrails_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS",
        "cases_passed": len(results),
        "cases": results,
        "source_checks": source_checks,
        "readiness_policy_checks": readiness_policy_checks,
        "promotion_shadow_live_allowed": False,
    }

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ENTRY_FOUNDATION_GUARDRAILS_latest.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["out"] = str(out_path)

    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--env-file", default="", help=argparse.SUPPRESS)
    ap.add_argument("--run-context", default="", help=argparse.SUPPRESS)
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
