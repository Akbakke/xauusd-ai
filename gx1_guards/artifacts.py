"""
GX1 artifact loading — fail-closed.

Enforces guardrails that are too important to leave to AGENTS.md text:
  - No implicit latest/glob selection for decisioning.
  - Only artifacts marked ACTIVE in the selection contract are decision-valid.
  - No EURUSD artifacts in an XAUUSD run (project isolation).
  - No old/invalidated artifacts for decisioning.

A rule in a markdown file is a hope. A function that raises is a guarantee.
"""

from __future__ import annotations

import json
from pathlib import Path

from gx1_guards import REPO_ROOT

# The selection contract is the ONE source of truth for what is decision-valid.
# Repo-root-relative so it resolves regardless of the caller's CWD.
SELECTION_CONTRACT = REPO_ROOT / "PROJECT_STATE_artifacts.json"

THIS_PROJECT = "XAUUSD"
FORBIDDEN_TOKENS = ("EURUSD", "eurusd", "eur_usd")


class ArtifactGuardError(Exception):
    """Raised when a forbidden / unsafe artifact selection is attempted."""


def _load_contract() -> dict:
    if not SELECTION_CONTRACT.exists():
        raise ArtifactGuardError(
            f"No selection contract at {SELECTION_CONTRACT}. "
            "Decisioning requires an explicit ACTIVE contract — refusing to guess."
        )
    return json.loads(SELECTION_CONTRACT.read_text())


def _check_isolation(path_str: str) -> None:
    if any(tok in path_str for tok in FORBIDDEN_TOKENS):
        raise ArtifactGuardError(
            f"Project isolation violation: '{path_str}' references EURUSD "
            f"inside a {THIS_PROJECT} run. These are separate projects — never mix."
        )


def load_decision_entry(role: str) -> dict:
    """
    Return the full ACTIVE entry dict for a role — path resolved + active_variant,
    active_fold(s), active_aggregator and any other fields the contract carries.

    Same guard semantics as ``load_decision_artifact`` (status ACTIVE, no
    in_sample_only, project isolation, path exists). Use this when live-wiring
    needs the per-role config (variant/fold/aggregator), not just the path.
    """
    contract = _load_contract()
    if contract.get("project") != THIS_PROJECT:
        raise ArtifactGuardError(
            f"Contract project is {contract.get('project')!r}, "
            f"expected {THIS_PROJECT!r}. Refusing cross-project load."
        )
    entry = contract.get("active", {}).get(role)
    if entry is None:
        raise ArtifactGuardError(
            f"No ACTIVE artifact declared for role '{role}'. "
            "Add an explicit entry to the selection contract first (vedtak required)."
        )
    if entry.get("status") != "ACTIVE":
        raise ArtifactGuardError(
            f"Artifact for '{role}' has status {entry.get('status')!r}, not ACTIVE."
        )
    if entry.get("in_sample_only"):
        raise ArtifactGuardError(
            f"Artifact for '{role}' is flagged in_sample_only — never decision-valid."
        )
    path_str = entry["path"]
    _check_isolation(path_str)
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise ArtifactGuardError(
            f"ACTIVE artifact for '{role}' not found on disk: {path}."
        )
    out = dict(entry)
    out["path"] = path  # absolute resolved Path overrides string
    return out


def load_decision_artifact(role: str) -> Path:
    """
    Return the ACTIVE artifact path for a given decision role
    (e.g. 'v10_entry', 'v3_exit', 'xgb', 'entry_iql', 'exit_iql').

    Thin wrapper over load_decision_entry — kept for callers that only need
    the path. There is deliberately NO glob, NO 'latest', NO mtime sorting.
    """
    return load_decision_entry(role)["path"]


def forbid_synthetic(data_source: str, *, allow_synthetic: bool = False) -> None:
    """
    Call at the top of any decisioning path that ingests data.
    Refuses dummy/synthetic/degraded fallbacks unless explicitly allowed
    (allow_synthetic should ONLY ever be True in unit tests, never in decisioning).
    """
    bad = ("dummy", "synthetic", "fallback", "degraded", "placeholder", "mock")
    if not allow_synthetic and any(b in data_source.lower() for b in bad):
        raise ArtifactGuardError(
            f"Refusing synthetic/degraded input for decisioning: {data_source!r}. "
            "Decisioning must use real artifacts only."
        )
