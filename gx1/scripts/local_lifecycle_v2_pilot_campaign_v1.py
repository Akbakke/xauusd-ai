"""CPU-only launch eligibility for the lifecycle-v2 local pilot campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from gx1.contracts.local_lifecycle_v2_pilot_campaign_v1 import (
    PilotCampaignError,
    file_sha256,
    next_action,
    RECEIPT_SCHEMA,
    canonical_bytes,
    canonical_sha256,
    require_clean_source,
    require_plan,
    require_receipt,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.resolve() != path:
        raise argparse.ArgumentTypeError("absolute normalized path required")
    return path


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PilotCampaignError("JSON object required")
    return value


def _receipts(root: Path) -> list[dict]:
    receipt_dir = root / "receipts"
    if not receipt_dir.exists():
        return []
    if receipt_dir.is_symlink() or not receipt_dir.is_dir():
        raise PilotCampaignError("receipt directory invalid")
    paths = sorted(receipt_dir.glob("invocation-*.json"))
    expected = [receipt_dir / f"invocation-{number:04d}.json" for number in range(1, len(paths) + 1)]
    if paths != expected:
        raise PilotCampaignError("receipt filenames are not contiguous")
    return [_read(path) for path in paths]


def inspect_campaign(
    *, plan_path: Path, plan_sha256: str, current_windows_boot_utc: str
) -> dict:
    if file_sha256(plan_path) != plan_sha256:
        raise PilotCampaignError("plan file digest mismatch")
    plan = require_plan(_read(plan_path), verify_files=True)
    runtime = Path(plan["runtime_root"])
    active_marker = runtime / "ACTIVE_INVOCATION.json"
    if active_marker.exists():
        if active_marker.is_symlink() or not active_marker.is_file():
            raise PilotCampaignError("active invocation marker invalid")
        decision = {
            "decision": "BLOCKED_RECOVERY_RECEIPT_REQUIRED",
            "stage": None,
            "invocation_number": len(_receipts(runtime)) + 1,
        }
    else:
        decision = next_action(
            plan, _receipts(runtime),
            current_boot_utc=current_windows_boot_utc,
        )
    if decision["decision"].startswith("LAUNCH") or decision["decision"].startswith("RESUME"):
        require_clean_source(plan)
        stage = decision["stage"]
        decision["invocation_manifest"] = plan["invocations"][stage]
        decision["full_launch_gate"] = plan["launch_gate"]
        decision["physical_power_limit_w_required"] = 160
        decision["maximum_power_draw_w"] = 160
        decision["automatic_power_limit_change"] = False
        decision["local_telemetry_sample_seconds"] = 1
        decision["human_status_cadence_seconds"] = 900
    return {
        "ok": True,
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "action": decision,
        "test_authorized": False,
        "cuda_started": False,
    }



def record_invocation(
    *, plan_path: Path, plan_sha256: str, stage: str, invocation_number: int,
    boot_utc: str, outcome: str, pointer_before_sha256: str,
    telemetry_jsonl: Path,
) -> dict:
    if file_sha256(plan_path) != plan_sha256:
        raise PilotCampaignError("plan file digest mismatch")
    plan = require_plan(_read(plan_path), verify_files=True)
    manifest_binding = plan["invocations"].get(stage)
    if manifest_binding is None:
        raise PilotCampaignError("stage invalid")
    manifest = _read(Path(manifest_binding["path"]))
    pointer = Path(manifest["pointer_path"])
    if not pointer.is_file() or telemetry_jsonl.is_symlink() or not telemetry_jsonl.is_file():
        raise PilotCampaignError("runtime evidence unavailable")
    rows = []
    with telemetry_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if (not isinstance(value, dict)
                    or value.get("schema_version") != "gx1_local_lifecycle_v2_pilot_telemetry_v1"
                    or value.get("stage") != stage
                    or value.get("invocation_number") != invocation_number
                    or value.get("gpu_uuid") != plan["gpu_uuid"]
                    or value.get("safety_decision") not in {"PASS", "KILL_AND_BLOCK"}):
                raise PilotCampaignError("telemetry sample invalid")
            rows.append(value)
    if not rows:
        raise PilotCampaignError("telemetry is empty")
    max_limit = max(float(row["power_limit_w"]) for row in rows)
    max_draw = max(float(row["power_draw_w"]) for row in rows)
    if any(row["safety_decision"] == "KILL_AND_BLOCK" for row in rows):
        outcome = "FAILED"
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "invocation_number": invocation_number,
        "stage": stage,
        "boot_utc": boot_utc,
        "outcome": outcome,
        "pointer_before_sha256": pointer_before_sha256,
        "pointer_after_sha256": file_sha256(pointer),
        "telemetry_jsonl": str(telemetry_jsonl),
        "telemetry_sha256": file_sha256(telemetry_jsonl),
        "telemetry_sample_count": len(rows),
        "maximum_observed_power_limit_w": max_limit,
        "maximum_observed_power_draw_w": max_draw,
        "test_data_used": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    checked = require_receipt(receipt, plan_sha256=plan["plan_sha256"])
    path = Path(plan["runtime_root"]) / "receipts" / f"invocation-{invocation_number:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PilotCampaignError("receipt already exists")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_bytes(checked))
    temporary.replace(path)
    return {"ok": True, "receipt": checked, "path": str(path), "sha256": file_sha256(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--plan-json", required=True, type=_absolute)
    inspect.add_argument("--plan-sha256", required=True)
    inspect.add_argument("--current-windows-boot-utc", required=True)
    record = commands.add_parser("record")
    record.add_argument("--plan-json", required=True, type=_absolute)
    record.add_argument("--plan-sha256", required=True)
    record.add_argument("--stage", required=True, choices=("smoke", "epoch1"))
    record.add_argument("--invocation-number", required=True, type=int)
    record.add_argument("--boot-utc", required=True)
    record.add_argument("--outcome", required=True, choices=("COMPLETE", "RESUMABLE", "FAILED"))
    record.add_argument("--pointer-before-sha256", required=True)
    record.add_argument("--telemetry-jsonl", required=True, type=_absolute)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_campaign(
                plan_path=args.plan_json,
                plan_sha256=args.plan_sha256,
                current_windows_boot_utc=args.current_windows_boot_utc,
            )
        else:
            result = record_invocation(
                plan_path=args.plan_json, plan_sha256=args.plan_sha256,
                stage=args.stage, invocation_number=args.invocation_number,
                boot_utc=args.boot_utc, outcome=args.outcome,
                pointer_before_sha256=args.pointer_before_sha256,
                telemetry_jsonl=args.telemetry_jsonl,
            )
    except (PilotCampaignError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
