#!/usr/bin/env python3
"""Run the audited post-activation refresh commands after alias activation.

Default mode is dry-run. Real execution requires that the activation apply
report proves the canonical alias switch was already performed and that a
separate post-apply vedtak is supplied. The commands refresh audits/smoke data
and readiness only; they must not start training, replay, IQL, shadow or live.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.scripts.verify_entry_foundation_state_v1 import REPO, REPORTS_ROOT


DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_foundation_activation_post_apply_20260629_v1"
REQUIRED_VEDTAK_PREFIX = "ENTRY_FOUNDATION_POST_APPLY_"
EXPECTED_ORDER = (
    "refresh_canonical_feature_audit",
    "refresh_canonical_target_audit",
    "refresh_canonical_specialist_audit",
    "refresh_canonical_smoke_dataset",
    "verify_active_foundation_state",
    "verify_train_readiness",
)
EXPECTED_MODULES = {
    "refresh_canonical_feature_audit": "gx1.scripts.audit_entry_foundation_features_v1",
    "refresh_canonical_target_audit": "gx1.scripts.audit_entry_foundation_targets_v1",
    "refresh_canonical_specialist_audit": "gx1.scripts.audit_entry_specialist_feature_groups_v1",
    "refresh_canonical_smoke_dataset": "gx1.scripts.materialize_entry_foundation_smoke_dataset_v1",
}
FORBIDDEN_TOKENS = (
    "smoke-train",
    "candidate-train",
    "replay-evidence",
    "iql-distill",
    "iql-replay",
    "iql-compare",
    "preview-shadow",
    "start-shadow",
    "live",
    "promote",
    "pin",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details or {}}


def _command_argv(command: dict[str, Any]) -> list[str]:
    argv = command.get("argv") if isinstance(command.get("argv"), list) else []
    return [str(part) for part in argv]


def _safe_command_check(command: dict[str, Any]) -> dict[str, Any]:
    name = str(command.get("name") or "")
    argv = _command_argv(command)
    joined = " ".join(argv)
    forbidden = [token for token in FORBIDDEN_TOKENS if token in joined]
    ok = bool(argv) and not forbidden
    if name in EXPECTED_MODULES:
        ok = (
            ok
            and len(argv) >= 4
            and argv[0] == ".venv/bin/python"
            and argv[1] == "-m"
            and argv[2] == EXPECTED_MODULES[name]
            and "--quiet" in argv
        )
    elif name == "verify_active_foundation_state":
        ok = argv == [
            "scripts/entry_next_edge_control.sh",
            "verify",
            "--quiet",
        ]
    elif name == "verify_train_readiness":
        ok = argv == [
            "scripts/entry_next_edge_control.sh",
            "train-readiness",
            "--quiet",
            "--no-fail-on-not-ready",
        ]
    else:
        ok = False
    return _check(
        f"post-apply command is safe: {name}",
        ok,
        {"name": name, "argv": argv, "forbidden_tokens": forbidden},
    )


def _validate_commands(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = tuple(str(command.get("name") or "") for command in commands)
    checks = [
        _check(
            "post-apply command order is exact",
            names == EXPECTED_ORDER,
            {"expected_order": list(EXPECTED_ORDER), "observed_order": list(names)},
        ),
    ]
    checks.extend(_safe_command_check(command) for command in commands)
    return checks


def _run_command(argv: list[str], *, timeout_seconds: int) -> dict[str, Any]:
    proc = subprocess.run(
        argv,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    return {
        "argv": argv,
        "returncode": int(proc.returncode),
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    activation_apply_json = Path(args.activation_apply_json).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    apply = bool(args.apply)
    dry_run = bool(args.dry_run)
    vedtak = str(args.vedtak or "")
    activation = _read_json(activation_apply_json)
    commands = activation.get("post_apply_commands") if isinstance(activation.get("post_apply_commands"), list) else []
    commands = [command for command in commands if isinstance(command, dict)]
    checks = [
        _check("activation apply report exists", activation_apply_json.exists(), {"path": str(activation_apply_json)}),
        _check(
            "activation apply report schema is expected",
            activation.get("schema_version") == "entry_foundation_activation_apply_v1",
            {"schema_version": activation.get("schema_version")},
        ),
        _check(
            "activation apply did not allow training",
            activation.get("training_allowed") is False,
            {"training_allowed": activation.get("training_allowed")},
        ),
        _check(
            "dry-run and apply flags are mutually exclusive",
            not (dry_run and apply),
            {"dry_run": dry_run, "apply": apply},
        ),
        _check(
            "apply requires explicit post-apply vedtak",
            (not apply) or (bool(vedtak) and vedtak.startswith(REQUIRED_VEDTAK_PREFIX)),
            {"apply": apply, "vedtak": vedtak, "required_prefix": REQUIRED_VEDTAK_PREFIX},
        ),
    ]
    checks.extend(_validate_commands(commands))
    mutation_performed = bool(activation.get("mutation_performed"))
    if apply:
        checks.append(
            _check(
                "activation alias switch was already applied",
                mutation_performed and activation.get("decision") == "APPLIED_ALIAS_SWITCH",
                {"decision": activation.get("decision"), "mutation_performed": mutation_performed},
            )
        )
    failures = [check for check in checks if not check["ok"]]
    command_results: list[dict[str, Any]] = []
    post_apply_mutations_performed = False
    if apply and not failures:
        for command in commands:
            argv = _command_argv(command)
            result = _run_command(argv, timeout_seconds=int(args.timeout_seconds))
            result["name"] = str(command.get("name") or "")
            command_results.append(result)
            if int(result["returncode"]) != 0:
                failures.append(
                    _check(
                        f"post-apply command exited cleanly: {result['name']}",
                        False,
                        result,
                    )
                )
                break
        post_apply_mutations_performed = not failures

    if failures:
        decision = "NOT_READY"
    elif not mutation_performed:
        decision = "WAITING_FOR_ACTIVATION_APPLY"
    elif apply:
        decision = "POST_APPLY_REFRESH_COMPLETED"
    else:
        decision = "READY_FOR_POST_APPLY_REFRESH"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_foundation_activation_post_apply_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "activation_apply_json": str(activation_apply_json),
        "apply_requested": apply,
        "dry_run_requested": dry_run,
        "vedtak": vedtak or None,
        "activation_alias_mutation_performed": mutation_performed,
        "post_apply_mutations_performed": post_apply_mutations_performed,
        "training_allowed": False,
        "commands": commands,
        "command_results": command_results,
        "checks": checks,
        "failures": failures,
        "next_required_action": (
            "run activation apply with explicit vedtak first"
            if decision == "WAITING_FOR_ACTIVATION_APPLY"
            else (
                "rerun train-readiness and require READY_FOR_VEDTAK_SMOKE_TRAIN before smoke training"
                if decision == "POST_APPLY_REFRESH_COMPLETED"
                else (
                    f"rerun with --apply --vedtak {REQUIRED_VEDTAK_PREFIX}<id> after activation apply"
                    if decision == "READY_FOR_POST_APPLY_REFRESH"
                    else "fix post-apply refresh failures"
                )
            )
        ),
    }
    json_path = out_dir / f"ENTRY_FOUNDATION_ACTIVATION_POST_APPLY_{timestamp}.json"
    latest_json = out_dir / "ENTRY_FOUNDATION_ACTIVATION_POST_APPLY_latest.json"
    report["json_path"] = str(json_path)
    report["latest_json_path"] = str(latest_json)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": report["failures"],
                    "json_path": report["json_path"],
                    "next_required_action": report["next_required_action"],
                    "post_apply_mutations_performed": report["post_apply_mutations_performed"],
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activation-apply-json", required=True)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--dry-run", action="store_true", help="Accepted for clarity; dry-run is default unless --apply is set.")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--vedtak", default="")
    ap.add_argument("--timeout-seconds", type=int, default=1800)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    report = run(build_parser().parse_args())
    if report["decision"] == "NOT_READY":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
