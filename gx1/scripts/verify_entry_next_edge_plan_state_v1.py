"""Legacy tombstone for the closed 2026-06-27 Entry next-edge plan.

The active Entry workstream is the foundation seq146 path. This module is kept
so old command references fail closed with a useful message instead of running
stale no-XGB shadow checks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/home/andre2/src/GX1_ENGINE")
PY = REPO / ".venv/bin/python"
FOUNDATION_AUDIT = REPO / "docs/ENTRY_FOUNDATION_AUDIT_20260628.md"
SPECIALIST_BLUEPRINT = REPO / "docs/ENTRY_SEQUENTIAL_AI_SPECIALIST_BLUEPRINT_20260628.md"


def run(args: argparse.Namespace) -> dict[str, Any]:
    proc = subprocess.run(
        [str(PY), "-m", "gx1.scripts.verify_entry_foundation_state_v1", "--quiet"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    report: dict[str, Any] = {
        "schema_version": "entry_next_edge_plan_state_legacy_tombstone_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "LEGACY_PLAN_CLOSED",
        "status": "BLOCKED_BY_ENTRY_FOUNDATION_FREEZE",
        "foundation_verify_returncode": int(proc.returncode),
        "foundation_verify_passed": proc.returncode == 0,
        "foundation_audit": str(FOUNDATION_AUDIT),
        "sequential_specialist_blueprint": str(SPECIALIST_BLUEPRINT),
        "canonical_control": "scripts/entry_next_edge_control.sh",
        "allowed_checks": [
            "scripts/entry_next_edge_control.sh verify",
            "scripts/entry_next_edge_control.sh selftest",
            "scripts/entry_next_edge_control.sh foundation-guardrails",
            "scripts/entry_next_edge_control.sh train-readiness",
        ],
        "next_allowed_command_after_explicit_vedtak": (
            "scripts/entry_next_edge_control.sh smoke-train --vedtak <id> --require-edge-audit"
        ),
        "blocked": [
            "preview-shadow",
            "start-shadow",
            "verify-shadow",
            "direct v12_paper_runner shadow",
            "legacy no-XGB shadow launchers",
        ],
    }
    if proc.returncode != 0:
        report["foundation_verify_stderr"] = proc.stderr[-2000:]
        report["foundation_verify_stdout"] = proc.stdout[-2000:]

    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["out"] = str(out)

    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_closed:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--fail-closed", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--candidate-manifest", default="", help=argparse.SUPPRESS)
    ap.add_argument("--env-file", default="", help=argparse.SUPPRESS)
    ap.add_argument("--out-dir", default="", help=argparse.SUPPRESS)
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
