"""
GX1 gate guard — explicit green-gate enforcement.

Covers two guardrails that cost real money / GPU time if they slip:
  - NEVER auto-retrain: a retrain requires an explicit, dated vedtak.
  - No R6 / freeze / promo / live / package build without a green gate.

A "gate" is just a small JSON file under gates/ that you create deliberately.
No file -> the operation aborts. This is fail-closed by construction.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from gx1_guards import REPO_ROOT

# Repo-root-relative so gates resolve regardless of the caller's CWD.
GATES_DIR = REPO_ROOT / "gates"

GATED_STAGES = {"r6", "freeze", "promo", "live", "package"}


class GateError(Exception):
    """Raised when a gated operation is attempted without a valid green gate."""


def _gate_path(stage: str) -> Path:
    return GATES_DIR / f"{stage}.gate.json"


def require_gate(stage: str) -> dict:
    """
    Call at the entrypoint of any gated stage. Aborts unless a green gate
    exists, is dated today, and is explicitly marked green.
    """
    stage = stage.lower()
    if stage not in GATED_STAGES:
        raise GateError(f"Unknown gated stage {stage!r}. Known: {sorted(GATED_STAGES)}")

    gp = _gate_path(stage)
    if not gp.exists():
        raise GateError(
            f"No green gate for stage '{stage}' ({gp}). "
            f"Create it deliberately before running. Refusing to proceed."
        )

    gate = json.loads(gp.read_text())
    if gate.get("status") != "GREEN":
        raise GateError(f"Gate '{stage}' status is {gate.get('status')!r}, not GREEN.")
    if gate.get("date") != date.today().isoformat():
        raise GateError(
            f"Gate '{stage}' is dated {gate.get('date')!r}, not today "
            f"({date.today().isoformat()}). Stale gates are not valid — re-confirm."
        )
    return gate


def require_retrain_vedtak(vedtak_id: str | None) -> str:
    """
    Call at the top of every retrain entrypoint. NEVER auto-retrain.
    Pass the --vedtak argument through here; missing/empty aborts the run.
    """
    if not vedtak_id or not vedtak_id.strip():
        raise GateError(
            "Retrain blocked: no --vedtak provided. "
            "Retraining requires an explicit user decision. NEVER auto-retrain."
        )
    return vedtak_id.strip()


if __name__ == "__main__":
    # CLI:  python -m gx1_guards.gates require-gate r6
    #       python -m gx1_guards.gates require-vedtak <id>
    if len(sys.argv) < 3:
        print("usage: gates.py [require-gate <stage> | require-vedtak <id>]")
        sys.exit(2)
    cmd, arg = sys.argv[1], sys.argv[2]
    try:
        if cmd == "require-gate":
            require_gate(arg)
            print(f"OK: gate '{arg}' is GREEN and current.")
        elif cmd == "require-vedtak":
            require_retrain_vedtak(arg)
            print(f"OK: vedtak '{arg}' accepted.")
        else:
            print(f"unknown command {cmd!r}")
            sys.exit(2)
    except GateError as e:
        print(f"BLOCKED: {e}")
        sys.exit(2)
