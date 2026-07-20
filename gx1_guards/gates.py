"""
GX1 gate guard — explicit green-gate enforcement.

Covers retained non-Entry write/launch guardrails:
  - legacy Exit/data-mutation entrypoints still using an explicit decision ID;
    model-native Entry uses evidence-gated ``entry_run_id`` lineage instead.
  - No R6 / freeze / promo / live / package build without a green gate.

A "gate" is just a small JSON file under gates/ that you create deliberately.
No file -> the operation aborts. This is fail-closed by construction.
"""

from __future__ import annotations

import json
import re
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
    Validate the retained legacy Exit/data-write decision identifier.

    Model-native Entry must use ``require_entry_run_id`` instead; this helper
    is not an Entry authorization contract.
    """
    value = str(vedtak_id or "").strip()
    if not value:
        raise GateError(
            "Training/rebuild blocked: no --vedtak provided. "
            "Writing model artifacts requires an explicit user decision."
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}", value):
        raise GateError(
            "Training/rebuild blocked: --vedtak must be an explicit 8-128 character decision ID "
            "using only letters, digits, dot, underscore, colon, or hyphen."
        )
    if value.upper() in {"TODO", "TBD", "PLACEHOLDER", "EXPLICIT_VEDTAK_ID"}:
        raise GateError("Training/rebuild blocked: placeholder --vedtak values are forbidden.")
    return value


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
